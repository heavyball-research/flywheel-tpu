# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kernel choice for softmax attention: block sizes and tuned configs.

get_kernel builds the dense and packed forward kernels from a tuned config or
the analytic default_block_sizes, through make_flash_attn_mha, which checks one
forward build and picks its dense, runtime-offset or varlen kernel.
resolve_paged_block_sizes picks the paged forward kernel's tiles;
resolve_bwd_config picks the backward's blocks and head fold and
resolve_kvcache_config the KV-cache decode kernel's config. This
is the only module that reads tuned_configs.json, the family's kernel configs
measured per TPU generation and exact problem shape; the kernels never read
it. The table maps a device name (get_device_name: 'TPU v6', 'TPU v7', ...) to
the configs measured on that chip, each under the exact key of the kernel
build it was measured for.

Keys are exact, not bucketed: q_block must divide the real seq_len and
head_fold the real num_heads, so a rounded-up entry could hand back a config
the shape cannot build. A missing key returns None and the build takes its
analytic default config; on a TPU that issues a UserWarning, once per key. A
kvcache| key holds the contiguous KV-cache decode kernel's (block_kv,
block_kv_compute, num_kv_stages[, items_per_step]); an mla| key the MLA decode
kernel's (target block_kv, max_items_per_step); every other key holds a dense
fwd / bwd CONFIG_FIELDS tuple. mla| keys alone are bucketed (batch and
max_seqlen_k to powers of two), as their target block is only a target: the
kernel fits it to the call's pages and VMEM.
"""

import functools
import json
import math
import pathlib
import re
import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from flywheel_tpu.core.tiling import round_up
from flywheel_tpu.core.tpu import BF16_TILE_ROWS, NUM_LANES
from flywheel_tpu.softmax_attention.config import (
    FWD_BLOCKS,
    FWD_KV_COMPUTE_BLOCKS,
    FWD_Q_COMPUTE_BLOCKS,
    MIN_NUM_STAGES,
    MIN_Q_TILES_PER_BLOCK,
    PAIRED_HEAD_DIM,
    BlockSizes,
    KVCacheConfig,
    QKVLayout,
    TokenMajorInfo,
    default_block,
    next_pow2,
    pick_tile,
    vmem_limit_bytes,
)
from flywheel_tpu.softmax_attention.kernels.decode import (
    kvcache_config_vmem_bytes,
    validate_kvcache_config,
)
from flywheel_tpu.softmax_attention.kernels.fwd_dense import (
    flash_attn_forward,
    flash_attn_forward_runtime_offset,
)
from flywheel_tpu.softmax_attention.kernels.fwd_paged import estimate_vmem_bytes
from flywheel_tpu.softmax_attention.kernels.fwd_varlen import (
    flash_attn_forward_varlen,
)
from flywheel_tpu.softmax_attention.kv_layout import MIN_BLOCK_KV
from flywheel_tpu.softmax_attention.schedule import (
    VarlenSchedule,
    dense_fwd_params,
)
from flywheel_tpu.softmax_attention.softmax import overflow_guard_threshold
from flywheel_tpu.softmax_attention.traits import FwdTraits
from flywheel_tpu.softmax_attention.variant import Variant

TUNED_CONFIGS_PATH = pathlib.Path(__file__).with_name("tuned_configs.json")

# Note (david): field order of a dense fwd / bwd tuned_configs.json value,
# shared by every tuple get_tuned_config returns.
CONFIG_FIELDS = (
    "q_block",
    "kv_block",
    "q_cblock",
    "kv_cblock",
    "stages",
    "head_fold",
    "qkv_layout",
    "transposed_pv",
)
# A dense fwd / bwd tuned_configs.json entry, in CONFIG_FIELDS order.
TunedConfig = tuple[int | str | bool, ...]
# A tuning table's entry as loaded, before its reader checks it.
TableConfig = list[int | str | bool]

# Note (david): varlen runs a runtime multi-step schedule whose best fold
# differs from a dense single-block kernel at the same physical block span.
# Keys are (head_dim, block_span, causal): the runtime cu_seqlens stay out of
# the compile key and a span costs the same per step at any packed total, so
# this table cannot live in tuned_configs.json's exact-shape key space.
# Measured on TPU v6 lite with 16384 packed tokens, 16 heads, bf16 and uniform
# block-aligned sequences.
TUNED_VARLEN_FWD_HEAD_FOLD: dict[str, dict[tuple[int, int, bool], int]] = {
    "TPU v6": {
        (128, 512, True): 8,
        (128, 512, False): 8,
        (128, 1024, True): 2,
        (128, 1024, False): 4,
    },
}

# Note (david): a packed (or right-only) window caps both physical forward
# blocks. A q block row loads the window plus bq + bkv keys of partial blocks,
# all through the masked fragment pipeline, so large blocks waste about
# (bq + bkv) / W, and small ones pay a pipeline step per few fragments. On v6e
# varlen 16K prefill (2 x 8K, 16:8 heads of 256, 32:8 of 128, W 1024 and
# 4096), 1024 beat 512 by 1.25-1.64x and 2048 by up to 1.44x, except at W 4096,
# where 2048 was 2-7% faster.
WINDOW_BLOCK = 1024
# Note (david): a dense window with a left bound runs the dense loop's static
# band copies, which skip and mask per compute tile, so its blocks stay at the
# full-attention size. On v6e dense 16K causal prefill (16:8 heads of 256,
# 32:8 of 128, W 128-4096), 2048 blocks with 256-key kv tiles beat 512-key ones
# by 7-26%, while 128-key tiles fell off a code-size cliff (5.5-12.7 ms at
# W 512). A d256 window wider than a block pair runs the 64-fragment mask-free
# copy, which ran 9.63 ms at 256-key tiles (W 4096) against 1.98 at 512.
BAND_KV_COMPUTE_BLOCKS = (256, 128)
# Note (david): a K or V tile pushed to the MXU serves the fragment's
# gqa_fold * block_q_compute rows, so the q compute tile is the largest that
# keeps a fragment within this many rows. v6e, chunks over 16K, heads of 256:
# 32:4 with 1024-token chunks ran 9.14 ms at 1024
# rows against 10.03 at 2048 (and 10.78 unfolded at 512); 32:32 ran 10.60 at
# 512 rows against 11.03 at 256, and with 2048-token chunks 10.38 at 1024
# rows against 10.54 at 512.
MAX_FRAGMENT_ROWS = 1024
# Note (david): a 2048 x 2048 q x kv block ran 3.4x slower than 2048 x 1024
# on v6e (Qwen3-4B's 32:8 heads of 128, a 2040-token chunk over a 16K prefix:
# 7.72 against 2.19 ms) though it fits VMEM, and 1024 x 2048 ran at par, so
# the block area is capped at 2048 x 1024. It counts every folded q row: at
# 32:4 a 512 x 1024 block of 8 folded heads ran 32.3 against 9.14 ms at
# 512 x 512.
MAX_PAGED_BLOCK_AREA = 2048 * 1024
# Note (david): a sequence shorter than block_q still fills a whole q block,
# and a server pads cu_seqlens_q to its request slots, so block_q cannot
# follow the longest sequence. v6e, 32:8 d128, 16K-128K token batches:
# 512 x 1024 beat 1024 x 512 on every batch, by 1.35-1.6x on mostly short
# sequences (3.56 against 5.69 ms).
PAGED_MAX_BLOCK_Q = 512
# Note (david): where the block area caps the pair, keeping block_kv and
# shrinking block_q won: 256 x 1024 beat 512 x 512 by up to 1.5x at 32:4, and
# 128 x 1024 beat 256 x 512 on six of seven batches at 32:2 (v6e).
PAGED_KV_PER_Q_BLOCK = 4
PAGED_MAX_BLOCK_KV = 1024
# Note (david): a sliding window caps both blocks, as WINDOW_BLOCK does on the
# dense entries. On v6e chunked prefill (d256, W 1024 and 4096), 512 beat 256
# by 1.4-2.7x and uncapped blocks by up to 1.55x. 1024 saved 26% at 32:32 W
# 4096 but cost 13% at W 1024, so the cap stays at 512.
PAGED_WINDOW_BLOCK = 512
DEFAULT_NUM_KV_STAGES = 2
KVCACHE_KV_COMPUTE_BLOCKS = (1024, 512, 384, 256, 128)
STAGED_KV_ELEMENT_BUDGET = 8 * 1024 * 1024
# Note (david): a row runs only the kv compute tiles that hold its keys, but a
# bigger block still issues a copy for every page in it. At Qwen3-4B's 32:8
# heads of 128 with a 32K-token table row on v6e, 256 rows of 64-1024 tokens
# took 1.15 ms with 4096-token blocks against 1.08 ms with 1024-token ones
# (1.82 ms when every tile of the block ran), while a 16K context took 15.53
# against 15.33 ms.
KVCACHE_MAX_BLOCK_KV = 1024
# Note (david): a merged-cache decode's default packed config sizes like
# batched RPA's decode: the device's items a step, never more than the batch's
# rows, and fewer only when the smallest block does not fit; the block grows by
# the MXU's 256 columns (v6e, v7x) while the kernel's VMEM stays within 80% of
# the limit, up to the device's max block. Unlike the one-item loop, a packed
# step runs every compute tile of its blocks, padding items included, so a row
# pays for the empty keys of its last block, and a step wider than the batch
# pays for whole blocks of nothing. The limits are heuristics, measured at one
# shape, 4:1 heads of 256 with an fp8 cache: on v6e 768-token blocks had the
# smallest worst case against the best block at each row length, and RPA's
# larger blocks lost most on 256-token rows; on v7x 16 items of 512 tokens beat
# 8 of 768 in all but one case. Other shapes take the same limits untuned.
# Values are (items a step, max block_kv).
DEFAULT_PACKED_STEP_LIMITS = (8, 768)
PACKED_STEP_LIMITS = {"TPU v7": (16, 512)}
PACKED_BLOCK_KV_GRANULE = 256
PACKED_VMEM_FRACTION = 0.8
# Note (david): the dq accumulator and stage buffers cap the useful backward
# block at 1024, one rung below the forward's ladder.
BWD_BLOCKS = (1024, 512, 256, 128)
# Note (david): a flat min(bkv, 256) breaks a bkv like 384, so the kv compute
# tile is the largest of these dividing bkv; bkv % NUM_SUBLANES == 0 makes the
# search always hit.
BWD_KV_COMPUTE_BLOCKS = (256, 128, 64, 32, 16, 8)
# Note (david): the head-fold gain saturates by about 8 heads. The single-block
# kernel halves its target past FULL_FOLD_MAX_BLOCK so the fold-scaled buffers
# still fit VMEM.
MAX_SINGLE_BLOCK_HEAD_FOLD = 8
FULL_FOLD_MAX_BLOCK = 512

default_bwd_block = functools.partial(default_block, candidates=BWD_BLOCKS)


def get_device_variant_name(device_kind: str | None = None) -> str:
    """'TPU v6 lite' / 'TPU7x' -> 'TPU v6e' / 'TPU v7' (generation plus e/p).

    device_kind defaults to the chip JAX runs on.
    """
    if device_kind is None:
        device_kind = jax.devices()[0].device_kind
    kind_match = re.fullmatch(r"TPU\s*v?(\d+)(?:(e|p)|\s+(lite))?", device_kind)
    if device_kind == "TPU7x":
        return "TPU v7"
    elif kind_match is None:
        return device_kind
    elif kind_match.group(2):
        return f"TPU v{kind_match.group(1)}{kind_match.group(2)}"
    elif kind_match.group(3):
        return f"TPU v{kind_match.group(1)}e"
    else:
        return f"TPU v{kind_match.group(1)}"


def get_device_name(device_kind: str | None = None) -> str:
    """'TPU v6 lite' / 'TPU v5e' / ... -> 'TPU v6' (generation key, as JAX
    does)."""
    return re.sub(
        r"^(TPU v\d+)[ep]$", r"\1", get_device_variant_name(device_kind)
    )


@functools.cache
def load_table(table: pathlib.Path) -> dict[str, dict[str, TableConfig]]:
    return json.loads(table.read_text(encoding="utf-8"))


@functools.cache
def _warn_untuned_once(
    table: pathlib.Path, device_name: str, key: str, reason: str
) -> None:
    detail = f" ({reason})" if reason else ""
    warnings.warn(
        f"flywheel_tpu: no usable tuned config for {key} on {device_name} in"
        f" {table.name}{detail}; falling back to the default config."
    )


def warn_untuned(table: pathlib.Path, key: str, reason: str = "") -> None:
    """Warn, once per table, device, key and reason, that a kernel build on a
    TPU falls back to its default config. Off TPU there are no tables to
    miss."""
    device_name = get_device_name()
    if device_name.startswith("TPU"):
        _warn_untuned_once(table, device_name, key, reason)


def lookup(table: pathlib.Path, key: str) -> TableConfig | None:
    """The current device's entry for key in table, warning when there is
    none."""
    config = load_table(table).get(get_device_name(), {}).get(key)
    if config is None:
        warn_untuned(table, key)
    return config


def get_varlen_head_fold(
    key: tuple[int, int, bool],
    num_heads: int,
    head_dim: int,
    head_dim_v: int,
) -> int:
    """Measured varlen forward head_fold for key, halved until it divides
    num_heads; 1 when untuned.

    Every entry was measured at head_dim_v == head_dim; the folded scratch
    scales with head_dim_v, so a wider value head is a different shape and gets
    no fold.
    """
    if head_dim_v == head_dim:
        device_table = TUNED_VARLEN_FWD_HEAD_FOLD.get(get_device_name(), {})
    else:
        device_table = {}
    head_fold = device_table.get(key, 1)
    while head_fold > 1 and num_heads % head_fold:
        head_fold //= 2
    return head_fold


def tuned_config_key(
    direction: str,
    *,
    token_major: bool,
    num_heads: int,
    batch: int | None,
    seq_len: int,
    head_dim: int,
    causal: bool,
    max_seqlen: int | None,
    return_lse: bool,
    num_kv_heads: int | None = None,
    heads_outer_batch: int | None = None,
) -> str:
    """The tuned_configs.json key for one kernel build.

    Every field is what the kernel sees: num_heads is the flat batch * heads
    count head-major and the per-row count token-major, batch is set only for
    3-D token-major refs, seq_len is the padded q axis (the whole packed buffer
    on varlen), and max_seqlen is the varlen schedule's bucket (None on dense).
    num_kv_heads (when it differs from num_heads) and heads_outer_batch extend
    the key only when present, so a GQA or heads-outer build never hits an entry
    measured on another head map.
    """
    base_key = (
        f"{direction}|{'tm' if token_major else 'hm'}|bfloat16"
        f"|h{num_heads}|b{'-' if batch is None else batch}"
        f"|s{seq_len}|d{head_dim}"
        f"|c{int(bool(causal))}|m{'-' if max_seqlen is None else max_seqlen}"
        f"|l{int(bool(return_lse))}"
    )
    if num_kv_heads is not None and num_kv_heads != num_heads:
        kv_heads_suffix = f"|k{num_kv_heads}"
    else:
        kv_heads_suffix = ""
    if heads_outer_batch is not None:
        heads_outer_suffix = f"|o{heads_outer_batch}"
    else:
        heads_outer_suffix = ""
    return base_key + kv_heads_suffix + heads_outer_suffix


def get_tuned_config(
    direction: str, *, head_dim_v: int, **key_fields: int | bool | None
) -> TunedConfig | None:
    """Return the measured config tuple (CONFIG_FIELDS order), or None.

    head_dim_v is checked, not keyed: every entry was measured at
    head_dim_v == head_dim. An entry that is not one value per CONFIG_FIELDS
    field raises.
    """
    if head_dim_v != key_fields["head_dim"]:
        return None
    key = tuned_config_key(direction, **key_fields)
    config = lookup(TUNED_CONFIGS_PATH, key)
    if config is None:
        return None
    elif len(config) != len(CONFIG_FIELDS):
        raise ValueError(
            f"{TUNED_CONFIGS_PATH.name} entry {key} holds {len(config)}"
            f" values; a dense fwd / bwd entry holds {CONFIG_FIELDS}."
        )
    else:
        return tuple(config)


def kvcache_config_key(
    *,
    dtype: DTypeLike,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    capacity: int,
    batch: int,
    append: bool,
    sliding_window: int | None,
    return_lse: bool,
) -> str:
    """The tuned_configs.json key for one contiguous KV-cache decode build."""
    return (
        f"kvcache|{jnp.dtype(dtype).name}|h{num_query_heads}|k{num_kv_heads}"
        f"|b{batch}|s{capacity}|d{head_dim}|a{int(bool(append))}"
        f"|w{'-' if sliding_window is None else sliding_window}"
        f"|l{int(bool(return_lse))}"
    )


def get_tuned_mla_config(
    *,
    cache_dtype: DTypeLike,
    num_query_rows: int,
    batch: int,
    max_seqlen_k: int,
) -> tuple[int, int] | None:
    """The tuned (target block_kv, max_items_per_step) of an MLA decode build,
    keyed on its cache dtype, its s_q * H query rows, and its batch and
    max_seqlen_k rounded up to powers of two; None when untuned."""
    key = (
        f"mla|{jnp.dtype(cache_dtype).name}|r{num_query_rows}"
        f"|b{next_pow2(batch)}|kv{max(512, next_pow2(max_seqlen_k))}"
    )
    config = lookup(TUNED_CONFIGS_PATH, key)
    if config is None:
        return None
    else:
        block_kv, max_items_per_step = config
        return block_kv, max_items_per_step


def get_tuned_kvcache_config(
    **key_fields: DTypeLike | int | bool | None,
) -> KVCacheConfig | None:
    config = lookup(TUNED_CONFIGS_PATH, kvcache_config_key(**key_fields))
    return None if config is None else tuple(config)


def resolve_tuned_kvcache_config(
    *,
    page_size: int | None,
    has_packed_default: bool,
    **key_fields: DTypeLike | int | bool | None,
) -> KVCacheConfig | None:
    """The tuned config of one KV-cache decode build, or None for the
    kernel's default config.

    key_fields are kvcache_config_key's; page_size is the paged cache's, None
    for the contiguous pair. has_packed_default skips a one-item (3-field)
    entry. A paged build rejects an entry whose block_kv is not whole pages,
    with a warning.
    """
    tuned_config = get_tuned_kvcache_config(**key_fields)
    # TODO (david): the tuned table holds one-item configs, tuned before the
    # packed kernel existed, and needs re-tuning with packed (4-field)
    # entries. Until then a merged-cache decode with a packed default skips a
    # 3-field entry and takes that default; a 4-field config still applies,
    # items_per_step 1 for the one-item kernel.
    if tuned_config is None or (has_packed_default and len(tuned_config) == 3):
        return None
    # Note (david): a paged block is whole pages, so a tuned block_kv that does
    # not tile the page cannot be used as-is.
    if page_size is not None and tuned_config[0] % page_size:
        warn_untuned(
            TUNED_CONFIGS_PATH,
            kvcache_config_key(**key_fields),
            f"the entry's block_kv {tuned_config[0]} is not a multiple of the"
            f" page size {page_size}, and a paged block must be whole pages",
        )
        return None
    return tuned_config


def default_packed_config(
    *,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    capacity: int,
    kv_itemsize: int,
    batch: int,
    page_size: int | None,
    device_name: str,
) -> KVCacheConfig | None:
    """The merged-cache decode's default (block_kv, block_kv, num_kv_stages,
    items_per_step), or None when the batch has one row or not even two
    items a step fit the VMEM budget.

    A packed step runs every compute tile of its blocks, so its compute tile
    is the whole block.
    """
    max_items_per_step, max_block_kv = PACKED_STEP_LIMITS.get(
        device_name, DEFAULT_PACKED_STEP_LIMITS
    )
    if page_size is None:
        granule = PACKED_BLOCK_KV_GRANULE
    else:
        granule = math.lcm(PACKED_BLOCK_KV_GRANULE, page_size)
    vmem_budget = PACKED_VMEM_FRACTION * vmem_limit_bytes()

    def _fits(block_kv, items_per_step):
        return (
            kvcache_config_vmem_bytes(
                block_kv,
                block_kv,
                DEFAULT_NUM_KV_STAGES,
                items_per_step,
                num_query_heads=num_query_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                kv_itemsize=kv_itemsize,
                batch=batch,
            )
            <= vmem_budget
        )

    # Note (david): every active row holds at least one item, so with every
    # row active a step no wider than the batch is at most half padding. At
    # one row the one-item loop runs only the compute tiles that hold the
    # row's keys.
    items_per_step = min(max_items_per_step, batch)
    while items_per_step > 1 and not _fits(granule, items_per_step):
        items_per_step -= 1
    if items_per_step == 1:
        return None
    # Note (david): a block is whole pages, so a page past the max block
    # still makes a block of one granule.
    block_kv = granule
    while block_kv + granule <= min(capacity, max_block_kv) and _fits(
        block_kv + granule, items_per_step
    ):
        block_kv += granule
    # Note (david): a paged table is padded to whole blocks, but a cache
    # without pages is split into them, so its block divides the capacity.
    if page_size is None:
        block_kv = next(
            block
            for block in range(block_kv, 0, -MIN_BLOCK_KV)
            if capacity % block == 0
        )
    return block_kv, block_kv, DEFAULT_NUM_KV_STAGES, items_per_step


def default_kvcache_config(
    *,
    capacity: int,
    page_size: int | None,
    head_dim: int,
    num_kv_heads: int,
) -> KVCacheConfig:
    """The KV-cache decode kernel's analytic (block_kv, block_kv_compute,
    num_kv_stages) for a per-row capacity.

    page_size is the paged cache's, None for the contiguous pair, whose
    capacity block_kv divides; the kernel pads a paged table to whole blocks.
    """
    if capacity % MIN_BLOCK_KV:
        raise ValueError(
            f"capacity must be a multiple of {MIN_BLOCK_KV}; got {capacity}."
        )
    granule = MIN_BLOCK_KV if page_size is None else page_size
    target_block_kv = min(
        capacity,
        STAGED_KV_ELEMENT_BUDGET // head_dim // next_pow2(2 * num_kv_heads),
        KVCACHE_MAX_BLOCK_KV,
    )
    aligned_target = max(target_block_kv // granule * granule, granule)
    if page_size is None:
        # Note (david): capacity is a multiple of granule, so the divisor
        # search always ends.
        block_kv = next(
            block
            for block in range(aligned_target, 0, -granule)
            if capacity % block == 0
        )
    else:
        # Note (david): the table is padded to whole blocks below, so a page
        # count with few factors (37 pages of 256 tokens) keeps the target
        # block instead of one page per pipeline step.
        block_kv = aligned_target
    block_kv_compute = next(
        candidate
        for candidate in KVCACHE_KV_COMPUTE_BLOCKS
        if block_kv % candidate == 0
    )
    return (block_kv, block_kv_compute, DEFAULT_NUM_KV_STAGES)


def resolve_kvcache_config(
    *,
    page_size: int | None,
    dtype: DTypeLike,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    capacity: int,
    batch: int,
    append: bool,
    variant: Variant,
    merged_cache: bool,
    query_tokens: int,
) -> KVCacheConfig:
    """The (block_kv, block_kv_compute, num_kv_stages, items_per_step) of one
    KV-cache decode build: its tuned entry, else default_packed_config for a
    merged-cache decode, else default_kvcache_config, validated against the
    build.

    The arguments are resolve_tuned_kvcache_config's, with the key's
    sliding_window (the left window bound) and return_lse read from variant;
    num_query_heads counts a folded window's query_tokens as query heads, as
    the kernel does. merged_cache is the kernel's one merged cache operand.
    """
    # Note (david): only a merged-cache decode, one query token a row, takes
    # the packed default. A window folds its tokens into the query rows, and
    # the packed step's score tiles, which the VMEM model leaves out, grow
    # with them: 46 MiB past the model at 64 tokens of a 3-head GQA.
    if merged_cache and query_tokens == 1:
        packed_default = default_packed_config(
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            capacity=capacity,
            kv_itemsize=jnp.dtype(dtype).itemsize,
            batch=batch,
            page_size=page_size,
            device_name=get_device_name(),
        )
    else:
        packed_default = None
    tuned_config = resolve_tuned_kvcache_config(
        page_size=page_size,
        has_packed_default=packed_default is not None,
        dtype=dtype,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        capacity=capacity,
        batch=batch,
        append=append,
        sliding_window=variant.window[0],
        return_lse=variant.return_lse,
    )
    if tuned_config is not None:
        config = tuned_config
    elif packed_default is not None:
        config = packed_default
    else:
        config = default_kvcache_config(
            capacity=capacity,
            page_size=page_size,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
        )
    return validate_kvcache_config(
        config,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        capacity=capacity,
        kv_itemsize=jnp.dtype(dtype).itemsize,
        batch=batch,
        page_size=page_size,
    )


def resolve_bwd_config(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    variant: Variant,
    token_major: bool,
    head_dim: int | None,
) -> tuple[BlockSizes, int | None]:
    """The (block_sizes, head_fold) of one flash_attn_bwd build.

    q, k, v, variant, token_major and head_dim are flash_attn_bwd's; the
    variant's causal flag and causal_offset key the choice. The table is keyed
    on square, zero-offset MHA shapes, so any other backward skips the lookup;
    it and a miss take the analytic blocks and fold. head_fold is None under
    GQA/MQA, whose fold is each kv head's q group.
    """
    causal, causal_offset = variant.causal, variant.causal_offset
    if token_major:
        num_heads = q.shape[-1] // head_dim
        num_kv_heads = k.shape[-1] // head_dim
        batch = q.shape[0]
        head_dim_qk = head_dim
        head_dim_v = v.shape[-1] // num_kv_heads
    else:
        num_heads, _, head_dim_qk = q.shape
        num_kv_heads = k.shape[0]
        batch = None
        head_dim_v = v.shape[-1]
    q_seq_len, kv_seq_len = q.shape[-2], k.shape[-2]
    is_gqa = num_heads != num_kv_heads
    if q_seq_len != kv_seq_len or not variant.matches_tuned_table or is_gqa:
        tuned_config = None
    else:
        tuned_config = get_tuned_config(
            "bwd",
            token_major=token_major,
            num_heads=num_heads,
            batch=batch,
            seq_len=q_seq_len,
            head_dim=head_dim_qk,
            head_dim_v=head_dim_v,
            causal=causal,
            max_seqlen=None,
            return_lse=False,
        )
    if tuned_config is None:
        block_q = default_bwd_block(q_seq_len)
        block_kv = default_bwd_block(kv_seq_len)
        block_sizes = BlockSizes(
            block_q=block_q,
            block_kv=block_kv,
            block_kv_compute=pick_tile(BWD_KV_COMPUTE_BLOCKS, block_kv),
            block_q_compute=block_q,
            num_stages=MIN_NUM_STAGES,
        )
        head_fold = None
    else:
        # Note (david): the block knobs, head_fold and num_stages are co-tuned
        # (head_fold > 1 needs block == seq_len, block_kv_compute must divide
        # block), so the table applies only when all of them are unset. It is
        # keyed on square, zero-offset MHA shapes.
        tuned = dict(zip(CONFIG_FIELDS, tuned_config))
        assert (
            tuned["qkv_layout"] == "head_dim_minor"
            and not tuned["transposed_pv"]
        ), tuned_config
        block_sizes = BlockSizes(
            block_q=tuned["q_block"],
            block_kv=tuned["kv_block"],
            block_kv_compute=tuned["kv_cblock"],
            block_q_compute=tuned["q_cblock"],
            num_stages=tuned["stages"],
        )
        head_fold = tuned["head_fold"]
    if is_gqa:
        return block_sizes, None
    elif head_fold is None and (
        q_seq_len != kv_seq_len
        or causal_offset != 0
        or q_seq_len // block_sizes.block_q != 1
        or head_dim_v != head_dim_qk
    ):
        # Note (david): the single-block kernel is square, zero-offset and
        # one-block only, and its fold was never modelled for MLA's wider
        # do/v/dv.
        head_fold = 1
    elif head_fold is None:
        # Note (david): the fold-scaled buffers grow with the block and
        # head_dim, so the target halves for large blocks and shrinks per extra
        # lane tile.
        if block_sizes.block_q <= FULL_FOLD_MAX_BLOCK:
            block_fold_target = MAX_SINGLE_BLOCK_HEAD_FOLD
        else:
            block_fold_target = MAX_SINGLE_BLOCK_HEAD_FOLD // 2
        fold_target = max(
            1, block_fold_target // max(1, head_dim_qk // NUM_LANES)
        )
        flat_num_heads = num_heads if batch is None else batch * num_heads
        head_fold = 1
        while (
            head_fold * 2 <= min(fold_target, flat_num_heads)
            and flat_num_heads % (head_fold * 2) == 0
        ):
            head_fold *= 2
    # Note (david): only a caller-chosen fold raises when illegal; the tuned
    # table's and this function's own picks get clamped instead.
    # Note (david): a fold group is head_fold consecutive flat heads, which
    # stays inside one token-major batch row only if head_fold also divides the
    # per-row head count. Automatic picks are keyed on the flat count, so they
    # halve to the largest legal group instead of raising.
    while token_major and head_fold > 1 and num_heads % head_fold:
        head_fold //= 2
    # Note (david): on d64 the fold is the DMA-alignment mechanism (a group must
    # span whole lane tiles), so it must be even, even where folding does not
    # pay; 2 always divides the per-row head count, which was checked to be
    # even.
    if token_major and head_dim_qk % NUM_LANES and head_fold % 2:
        head_fold = 2
    return block_sizes, head_fold


def default_block_sizes(
    q_seqlen: int,
    kv_seqlen: int,
    max_seqlen: int | None = None,
    max_seqlen_q: int | None = None,
    *,
    window_size: tuple[int, int] = (-1, -1),
    head_dim: int = NUM_LANES,
) -> BlockSizes:
    # Note (david): the kernel masks in absolute coordinates, so block_q and
    # block_kv come from their own axes. Chunked-prefill q chunks are far
    # shorter than their kv contexts, and a q block sized to the kv cap
    # straddles several sequences and leaves the block-diagonal schedule little
    # to skip, so max_seqlen_q caps the q axis on its own.
    seq_q_cap = max_seqlen if max_seqlen_q is None else max_seqlen_q
    is_dense_band = max_seqlen is None and window_size[0] != -1
    # Note (david): only the dense grid needs blocks that divide both axes. The
    # per-seq schedule clamps every window to the padded end, so a packed total
    # of 128 times an odd count keeps the max_seqlen-capped block, not 128 rows.
    must_divide = max_seqlen is None
    if is_dense_band:
        # Note (david): the band copies need bq % bkv == 0.
        block_q = default_block(q_seqlen, FWD_BLOCKS)
        block_kv = default_block(kv_seqlen, FWD_BLOCKS, block_q)
    elif window_size != (-1, -1):
        block_q = default_block(
            q_seqlen,
            FWD_BLOCKS,
            WINDOW_BLOCK if seq_q_cap is None else min(seq_q_cap, WINDOW_BLOCK),
            must_divide=must_divide,
        )
        block_kv = default_block(
            kv_seqlen,
            FWD_BLOCKS,
            WINDOW_BLOCK
            if max_seqlen is None
            else min(max_seqlen, WINDOW_BLOCK),
            must_divide=must_divide,
        )
    else:
        block_q = default_block(
            q_seqlen, FWD_BLOCKS, seq_q_cap, must_divide=must_divide
        )
        block_kv = default_block(
            kv_seqlen, FWD_BLOCKS, max_seqlen, must_divide=must_divide
        )
    q_compute_tile = pick_tile(FWD_Q_COMPUTE_BLOCKS, block_q)
    if block_q // q_compute_tile < MIN_Q_TILES_PER_BLOCK:
        block_q_compute = q_compute_tile // MIN_Q_TILES_PER_BLOCK
    else:
        block_q_compute = q_compute_tile
    if is_dense_band and (
        head_dim <= NUM_LANES or window_size[0] < block_q + block_kv - 1
    ):
        kv_compute_blocks = BAND_KV_COMPUTE_BLOCKS
    else:
        kv_compute_blocks = FWD_KV_COMPUTE_BLOCKS
    return BlockSizes(
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=pick_tile(kv_compute_blocks, block_kv),
        block_q_compute=block_q_compute,
    )


def make_flash_attn_mha(
    num_heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    *,
    variant: Variant,
    block_sizes: BlockSizes,
    head_dim: int,
    head_dim_v: int | None = None,
    num_kv_heads: int | None = None,
    interpret: bool = False,
    head_fold: int = 1,
    transposed_pv: bool = False,
    varlen_max_seqlen_kv: int | None = None,
    token_major: TokenMajorInfo | None = None,
    heads_outer_batch: int | None = None,
) -> Callable[..., jax.Array | tuple[jax.Array, jax.Array]]:
    """Build the MHA forward kernel for one static attention configuration.

    The returned callable takes (q, k, v), or (q, k, v, cu_seqlens_q,
    cu_seqlens_k) when varlen_max_seqlen_kv is set, or (q, k, v, causal_offset)
    under variant.runtime_causal_offset, plus optional rotary and
    rotary_interleaved kwargs, and returns o, or (o, lse) under
    variant.return_lse. Except under runtime_causal_offset, an optional sinks
    argument after those, (nheads,) float sink logits, one per sequence head
    (flat q head f reads f % nheads, or f // heads_outer_batch heads-outer),
    joins each head's softmax denominator. Batch-outer only checks that the
    sinks' length divides the flat q heads, so the caller must pass exactly
    nheads logits.

    num_heads / num_kv_heads: per-row counts under token_major, flat otherwise;
      num_kv_heads=None is MHA and MQA is num_kv_heads=1.
    head_dim / head_dim_v: the q/k and v head widths; head_dim_v=None is
      head_dim. Under token_major they must be its widths.
    variant: the scale, softcap, mask and outputs. Under varlen its mask is
      bottom-right per sequence and its causal_offset must be 0;
      runtime_causal_offset is dense causal only, without a window.
    block_sizes: block_q must hold at least two block_q_compute tiles.
    head_fold: consecutive heads folded into one physical block; 1 disables it.
    transposed_pv: compute PV as o^T [head_dim, q_block], which requires
      block_sizes.qkv_layout == SEQ_MINOR; the output is swapped back.
    varlen_max_seqlen_kv: switches to the cu_seqlens-driven block schedule;
      only its presence is read.
    token_major: (batch, seq, heads * head_dim) or (total, heads * head_dim)
      q/k/v addressing; None is head-major (heads, seq, head_dim).
    heads_outer_batch: B for a heads-outer dense fold of B sequences, where q
      fold index n * B + b reads kv fold index (n // q_heads_per_kv_head) * B +
      b; None keeps the batch-outer fold.
    """
    if variant.runtime_causal_offset and (
        not variant.causal
        or variant.window_size != (-1, -1)
        or varlen_max_seqlen_kv is not None
    ):
        raise ValueError(
            "runtime_causal_offset requires dense causal attention without an"
            " explicit window."
        )
    if type(transposed_pv) is not bool:
        raise TypeError("transposed_pv must be a bool.")
    if variant.return_lse and transposed_pv:
        raise NotImplementedError(
            "return_lse is only wired for transposed_pv=False; got"
            f" {transposed_pv=}."
        )
    if (
        token_major is not None
        and block_sizes.qkv_layout != QKVLayout.HEAD_DIM_MINOR
    ):
        raise ValueError(
            "token-major requires qkv_layout == HEAD_DIM_MINOR: a SEQ_MINOR"
            " slab cannot be sliced out of a (.., seq, heads*head_dim) ref."
        )
    # Note (david): the transposed PV matmul contracts v's block_kv axis in
    # place, which only a SEQ_MINOR v has. With the token-major check above this
    # also rejects token-major x transposed_pv, since they demand opposite
    # layouts.
    if transposed_pv and block_sizes.qkv_layout != QKVLayout.SEQ_MINOR:
        raise ValueError(
            "transposed_pv=True requires block_sizes.qkv_layout =="
            " QKVLayout.SEQ_MINOR (v must already be the [head_dim, block_kv]"
            " slab the transposed PV matmul contracts directly); got"
            f" {block_sizes.qkv_layout!r}."
        )
    if (
        block_sizes.block_q // block_sizes.block_q_compute
        < MIN_Q_TILES_PER_BLOCK
    ):
        raise ValueError(
            "the static-anchor softmax requires >= 2 q tiles per block"
            " (block_q_compute < block_q): the distance-1 pipeline must not"
            " revisit a q slice while its rescale alpha is still pending."
        )
    is_varlen = varlen_max_seqlen_kv is not None
    # Note (david): per-seq blocks address HBM at arbitrary token offsets, which
    # only a second-minor (sublane) token axis supports.
    if is_varlen and (
        transposed_pv or block_sizes.qkv_layout != QKVLayout.HEAD_DIM_MINOR
    ):
        raise ValueError(
            "the per-seq varlen schedule requires a HEAD_DIM_MINOR qkv_layout"
            " and transposed_pv=False (token axis must stay second-minor)."
        )
    head_dim_v = head_dim if head_dim_v is None else head_dim_v
    if token_major is None:
        is_lane_paired = False
    else:
        is_lane_paired = (
            token_major.head_dim_qk % NUM_LANES != 0
            or token_major.head_dim_v % NUM_LANES != 0
        )
        if is_lane_paired and (
            token_major.head_dim_qk,
            token_major.head_dim_v,
        ) != (PAIRED_HEAD_DIM, PAIRED_HEAD_DIM):
            raise ValueError(
                "token-major requires head_dim % 128 == 0, or head_dim =="
                f" head_dim_v == 64 (paired); got {token_major!r}."
            )
        # Note (david): the schedule reads num_heads / num_kv_heads while the
        # DMA windows read token_major, so a disagreement would silently address
        # the wrong lanes.
        expected_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if (token_major.num_q_heads, token_major.num_kv_heads) != (
            num_heads,
            expected_kv_heads,
        ):
            raise ValueError(
                f"token_major carries per-row heads ({token_major.num_q_heads},"
                f" {token_major.num_kv_heads}) but the builder was given"
                f" ({num_heads}, {expected_kv_heads})."
            )
        if (token_major.head_dim_qk, token_major.head_dim_v) != (
            head_dim,
            head_dim_v,
        ):
            raise ValueError(
                f"token_major carries head widths ({token_major.head_dim_qk},"
                f" {token_major.head_dim_v}) but the builder was given"
                f" ({head_dim}, {head_dim_v})."
            )
        if token_major.batch is None and not is_varlen:
            raise ValueError(
                "token-major with batch=None is 2-D packed storage, which only"
                " the runtime (varlen) schedule consumes; dense x 2-D has no"
                " caller. Pass batch for dense, or varlen_max_seqlen_kv."
            )
        if head_fold > 1 and num_heads % head_fold:
            raise ValueError(
                "token-major head_fold must divide the PER-ROW num_heads (a"
                " fold group cannot straddle a batch boundary); got"
                f" {head_fold=}, per-row {num_heads=}."
            )
    # Note (david): token-major head counts are per fused row while the schedule
    # runs on flat heads, so widen them by the batch rows (2-D storage is one
    # row of heads).
    num_batch_rows = 1 if token_major is None else (token_major.batch or 1)
    num_q_heads = num_heads * num_batch_rows
    guard_threshold = overflow_guard_threshold(seqlen_kv)

    num_kv_heads = (
        num_q_heads if num_kv_heads is None else num_kv_heads * num_batch_rows
    )
    if num_q_heads % num_kv_heads:
        raise ValueError(f"{num_kv_heads=} must divide {num_q_heads=}.")
    q_heads_per_kv_head = num_q_heads // num_kv_heads

    if type(head_fold) is not int:
        raise TypeError(f"head_fold must be an int, got {type(head_fold)}.")
    if head_fold <= 0:
        raise ValueError(f"head_fold must be positive, got {head_fold}.")
    if is_lane_paired and head_fold % 2:
        raise ValueError(
            "token-major d64 requires an even head_fold (a DMA group must span"
            f" whole 128-lane tiles); got {head_fold=}."
        )
    if head_fold > 1:
        if q_heads_per_kv_head != 1:
            raise ValueError(
                "head_fold > 1 requires one query head per kv head (no GQA);"
                f" got {q_heads_per_kv_head=}."
            )
        if num_q_heads % head_fold:
            raise ValueError(f"{head_fold=} must divide {num_q_heads=}.")
        # Note (david): a d64 lane-pair fold is the DMA-alignment mechanism, not
        # an optimization, so it skips the single-block restriction that is only
        # a performance heuristic for other dense folds.
        if is_varlen:
            has_foldable_blocks = block_sizes.block_q == block_sizes.block_kv
        else:
            has_foldable_blocks = is_lane_paired or (
                block_sizes.block_q == seqlen_q
                and block_sizes.block_kv == seqlen_q
            )
        if not has_foldable_blocks:
            raise ValueError(
                "head_fold > 1 requires the whole sequence to be one physical"
                " block for dense attention or square physical blocks for"
                f" varlen attention: block_q={block_sizes.block_q},"
                f" block_kv={block_sizes.block_kv}, seq_len={seqlen_q},"
                f" varlen={is_varlen}."
            )
    if heads_outer_batch is not None:
        if type(heads_outer_batch) is not int or heads_outer_batch <= 0:
            raise ValueError(
                "heads_outer_batch must be a positive int; got"
                f" {heads_outer_batch!r}."
            )
        if token_major is not None or is_varlen:
            raise ValueError(
                "heads_outer_batch is a head-major dense fold; got"
                f" token_major={token_major!r},"
                f" varlen_max_seqlen_kv={varlen_max_seqlen_kv!r}."
            )
        if head_fold != 1:
            raise ValueError(
                "heads_outer_batch requires head_fold == 1 (a group of"
                " consecutive fold indices would span sequences); got"
                f" {head_fold=}."
            )
        if num_q_heads % heads_outer_batch or num_kv_heads % heads_outer_batch:
            raise ValueError(
                f"heads_outer_batch={heads_outer_batch} must divide the folded"
                f" {num_q_heads=} and {num_kv_heads=}."
            )
    # Note (david): packed causal and window edges sit on each sequence's own
    # bottom-right diagonal, which no single static offset expresses once q and
    # kv pack independently, so the per-seq schedule carries them and the kernel
    # is built without a static mask.
    if is_varlen and variant.causal_offset != 0:
        raise ValueError(
            "a packed forward derives its bottom-right offset per sequence from"
            f" cu_seqlens; a kernel-wide causal_offset={variant.causal_offset}"
            " would be applied on top of it. Pass causal_offset=0."
        )
    left, right = variant.window
    traits = FwdTraits(
        block_sizes=block_sizes,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        head_dim_v=head_dim_v,
        head_fold=head_fold,
        transposed_pv=transposed_pv,
        token_major=token_major,
        is_per_seq=is_varlen,
        guard_threshold=guard_threshold,
    )

    if is_varlen:
        return functools.partial(
            flash_attn_forward_varlen,
            varlen_sched=VarlenSchedule(
                padded_total_q=seqlen_q, padded_total_k=seqlen_kv
            ),
            traits=traits,
            variant=variant,
            interpret=interpret,
        )
    else:
        forward = (
            flash_attn_forward_runtime_offset
            if variant.runtime_causal_offset
            else flash_attn_forward
        )
        return functools.partial(
            forward,
            dense_params=dense_fwd_params(
                num_q_heads=num_q_heads,
                head_fold=head_fold,
                q_heads_per_kv_head=q_heads_per_kv_head,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                bq=block_sizes.block_q,
                bkv=block_sizes.block_kv,
                left=left,
                right=right,
                offset=variant.causal_offset,
                heads_outer_batch=heads_outer_batch,
            ),
            traits=traits,
            variant=variant,
            interpret=interpret,
        )


@functools.cache
def get_kernel(
    num_heads: int,
    num_kv_heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    variant: Variant,
    interpret: bool,
    head_dim: int,
    head_dim_v: int,
    varlen_max_seqlen_kv: int | None = None,
    varlen_max_seqlen_q: int | None = None,
    token_major: bool = False,
    kernel_batch: int | None = None,
    blocks_override: tuple[int, int, int, int] | None = None,
    heads_outer_batch: int | None = None,
) -> functools.partial:
    # Note (david): seqlen_q and seqlen_kv are padded lengths while
    # causal_offset comes from the original ones. Different original pairs can
    # pad to the same multiples of 128, so a rectangular mask is detected by
    # causal_offset too.
    is_rectangular = seqlen_q != seqlen_kv or variant.causal_offset != 0
    is_varlen = varlen_max_seqlen_kv is not None
    has_window = variant.window_size != (-1, -1)
    # Note (david): a tuned config fixes the blocks and the head fold together,
    # so a hit replaces the analytic block sizes wholesale. Windows, softcap and
    # rectangular masks were never measured and never consult it.
    if (
        blocks_override is None
        and variant.matches_tuned_table
        and not is_rectangular
    ):
        tuned_config = get_tuned_config(
            "fwd",
            token_major=token_major,
            num_heads=num_heads,
            batch=kernel_batch,
            seq_len=seqlen_q,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            causal=variant.causal,
            max_seqlen=varlen_max_seqlen_kv,
            return_lse=variant.return_lse,
            num_kv_heads=num_kv_heads,
            heads_outer_batch=heads_outer_batch,
        )
    else:
        tuned_config = None
    if tuned_config is not None:
        tuned = dict(zip(CONFIG_FIELDS, tuned_config))
        head_fold = tuned["head_fold"]
        # Note (david): the per-sequence varlen schedule DMAs at arbitrary token
        # offsets, which needs the token axis second-minor, so it overrides a
        # tuned SEQ_MINOR layout and transposed_pv.
        if is_varlen:
            qkv_layout = QKVLayout.HEAD_DIM_MINOR
            transposed_pv = False
        else:
            qkv_layout = QKVLayout[tuned["qkv_layout"].upper()]
            transposed_pv = tuned["transposed_pv"]
        block_sizes = BlockSizes(
            block_q=tuned["q_block"],
            block_kv=tuned["kv_block"],
            block_kv_compute=tuned["kv_cblock"],
            block_q_compute=tuned["q_cblock"],
            num_stages=tuned["stages"],
            qkv_layout=qkv_layout,
        )
    else:
        transposed_pv = False
        if blocks_override is None:
            block_sizes = default_block_sizes(
                seqlen_q,
                seqlen_kv,
                varlen_max_seqlen_kv,
                varlen_max_seqlen_q,
                window_size=variant.window_size,
                head_dim=head_dim,
            )
        else:
            # Note (david): the caller already padded both axes to multiples of
            # these blocks, and the kernel build rejects infeasible tiles.
            block_q, block_kv, block_q_compute, block_kv_compute = (
                blocks_override
            )
            block_sizes = BlockSizes(
                block_q=block_q,
                block_kv=block_kv,
                block_kv_compute=block_kv_compute,
                block_q_compute=block_q_compute,
            )
        # Note (david): only varlen has a fold table of its own, so a dense
        # build that missed the tuned configs runs unfolded. The kernel folds
        # varlen only on square physical blocks, which chunked prefill breaks
        # while keeping the padded axes equal. The table was measured head-major
        # but serves token-major too: the fold amortizes per-head step costs the
        # layout does not change, and halving it until it divides the per-row
        # num_heads keeps a token-major group inside one batch row.
        if (
            not is_varlen
            or num_heads != num_kv_heads
            or has_window
            or is_rectangular
            or block_sizes.block_q != block_sizes.block_kv
        ):
            head_fold = 1
        else:
            head_fold = get_varlen_head_fold(
                (head_dim, block_sizes.block_q, variant.causal),
                num_heads,
                head_dim,
                head_dim_v,
            )
    # Note (david): token-major d64 pairs heads, so the fold is the DMA
    # alignment unit and an odd fold is bumped to the minimal pair.
    if token_major and head_dim % NUM_LANES and head_fold % 2:
        kernel_head_fold = 2
    else:
        kernel_head_fold = head_fold
    if token_major:
        token_major_info = TokenMajorInfo(
            batch=kernel_batch,
            num_q_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_v=head_dim_v,
        )
    else:
        token_major_info = None
    return make_flash_attn_mha(
        num_heads,
        seqlen_q,
        seqlen_kv,
        variant=variant,
        block_sizes=block_sizes,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_kv_heads=num_kv_heads,
        interpret=interpret,
        head_fold=kernel_head_fold,
        transposed_pv=transposed_pv,
        varlen_max_seqlen_kv=varlen_max_seqlen_kv,
        token_major=token_major_info,
        heads_outer_batch=heads_outer_batch,
    )


@functools.cache
def resolve_paged_tiles(
    *,
    max_seqlen_q_bucket: int,
    max_seqlen_k_bucket: int,
    page_size: int,
    head_dim: int,
    return_lse: bool,
    rotary_dtype: jnp.dtype | None,
    gqa_fold: int,
    has_window: bool,
    kv_dtype: jnp.dtype,
    block_sizes: BlockSizes | None = None,
) -> BlockSizes:
    """block_sizes of one paged build, whose q sizes count tokens per head.

    Among the builds whose TPU staging fits vmem_limit_bytes() and whose
    (gqa_fold * q) x kv block area is at most MAX_PAGED_BLOCK_AREA, picks the
    largest kv block at most a kv target, then the largest q block at most a q
    target; with none at most a target, the smallest above it. The q target is
    the longest sequence's bucket, at most PAGED_MAX_BLOCK_Q; the kv target is
    PAGED_KV_PER_Q_BLOCK q blocks, at most PAGED_MAX_BLOCK_KV. The q compute
    tile keeps a fragment within MAX_FRAGMENT_ROWS rows. has_window caps both
    blocks at PAGED_WINDOW_BLOCK tokens, and block_kv at no less than one page.
    A pinned block_sizes is only checked against the budget.
    """
    target_q = min(max(max_seqlen_q_bucket, FWD_BLOCKS[-1]), PAGED_MAX_BLOCK_Q)
    target_kv = min(PAGED_KV_PER_Q_BLOCK * target_q, PAGED_MAX_BLOCK_KV)
    if block_sizes is not None:
        candidate_blocks = [block_sizes]
    else:
        block_cap = PAGED_WINDOW_BLOCK if has_window else FWD_BLOCKS[0]
        # Note (david): block_kv is whole pages, so a 384- or 640-token page
        # takes five or three pages (1920 tokens), where no power-of-two block
        # ends on a page edge; smaller kv blocks halve the page count.
        kv_block_cap = max(min(max_seqlen_k_bucket, block_cap), page_size)
        max_kv_pages = kv_block_cap // page_size
        kv_blocks = [
            page_size * (max_kv_pages >> shift)
            for shift in range(max_kv_pages.bit_length())
        ]
        # Note (david): a fragment reloads its rows' softmax state per kv
        # compute tile, so no larger q block is bought with a kv block below one
        # full kv compute tile. At 32:4 heads of 256 (1024-token chunks over
        # 16K, v6e, gqa fold 8) a 1024 x 256 block took 12.44 ms against 9.14 at
        # 512 x 512.
        min_kv_block = min(FWD_KV_COMPUTE_BLOCKS[0], kv_blocks[0])
        kv_blocks = [
            block_kv for block_kv in kv_blocks if block_kv >= min_kv_block
        ]
        q_block_cap = min(max(max_seqlen_q_bucket, FWD_BLOCKS[-1]), block_cap)
        # Note (david): FWD_BLOCKS are powers of two, so the smaller of two
        # powers of two divides the q block; a folded tile stays whole bf16
        # tiles.
        head_tile_rows = MAX_FRAGMENT_ROWS // gqa_fold
        q_compute_cap = max(
            BF16_TILE_ROWS, 1 << (head_tile_rows.bit_length() - 1)
        )
        candidate_blocks = [
            BlockSizes(
                block_q=block_q,
                block_kv=block_kv,
                block_kv_compute=pick_tile(FWD_KV_COMPUTE_BLOCKS, block_kv),
                block_q_compute=min(
                    q_compute_cap, block_q // MIN_Q_TILES_PER_BLOCK
                ),
            )
            for block_q in FWD_BLOCKS
            if block_q <= q_block_cap
            for block_kv in kv_blocks
            if gqa_fold * block_q * block_kv <= MAX_PAGED_BLOCK_AREA
        ]
    estimates = {
        blocks: estimate_vmem_bytes(
            blocks,
            head_dim=head_dim,
            return_lse=return_lse,
            rotary_dtype=rotary_dtype,
            gqa_fold=gqa_fold,
            kv_dtype=kv_dtype,
        )
        for blocks in candidate_blocks
    }
    fitting = [
        blocks
        for blocks, estimate in estimates.items()
        if estimate <= vmem_limit_bytes()
    ]
    if not fitting:
        blocks, estimate = min(
            estimates.items(), key=lambda blocks_and_bytes: blocks_and_bytes[1]
        )
        raise ValueError(
            "paged attention does not fit the scoped VMEM budget of"
            f" {vmem_limit_bytes()} bytes: its smallest build (block_q="
            f"{blocks.block_q}, block_kv={blocks.block_kv}) needs ~{estimate}"
            f" bytes at {head_dim=}."
        )

    def _nearest_below(size, target):
        return (size <= target, size if size <= target else -size)

    return max(
        fitting,
        key=lambda blocks: (
            _nearest_below(blocks.block_kv, target_kv),
            _nearest_below(blocks.block_q, target_q),
        ),
    )


def resolve_paged_block_sizes(
    q: jax.Array,
    kv_cache: jax.Array,
    block_table: jax.Array,
    *,
    num_kv_heads: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    variant: Variant,
    rotary: tuple[jax.Array, None] | None,
    block_sizes: BlockSizes | None,
) -> BlockSizes:
    """block_sizes of one flash_attn_varlen_paged build, on operands
    check_paged_inputs accepted (num_kv_heads is what it returned).

    max_seqlen_q and max_seqlen_k only pick the tiles, so an underestimate is
    slow, not wrong. variant is the build's: whether it returns lse and sets a
    left window bound size the tiles. A pinned block_sizes is only checked
    against the VMEM budget.
    """
    num_q_heads, total_q, head_dim = q.shape
    page_size = kv_cache.shape[1]
    capacity = jnp.shape(block_table)[1] * page_size
    if rotary is None:
        rotary_dtype = None
    else:
        rotary_dtype = jnp.dtype(rotary[0].dtype)
    # Note (david): the bounds key the tile cache, so power-of-two buckets keep
    # a wobbling longest sequence on one build. The caps hold because no q
    # block passes the 128-padded buffer and no sequence outgrows its table row.
    return resolve_paged_tiles(
        max_seqlen_q_bucket=min(
            next_pow2(max_seqlen_q), round_up(total_q, NUM_LANES)
        ),
        max_seqlen_k_bucket=min(next_pow2(max_seqlen_k), capacity),
        page_size=page_size,
        head_dim=head_dim,
        return_lse=variant.return_lse,
        rotary_dtype=rotary_dtype,
        gqa_fold=num_q_heads // num_kv_heads,
        has_window=variant.window[0] is not None,
        block_sizes=block_sizes,
        kv_dtype=kv_cache.dtype,
    )
