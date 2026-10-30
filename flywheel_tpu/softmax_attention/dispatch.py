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
the analytic default_block_sizes; resolve_tuned_bwd_config and
resolve_tuned_kvcache_config find the configs the backward and the KV-cache
decode kernels take. This is the only module that reads tuned_configs.json,
the family's kernel configs measured per TPU generation and exact problem
shape; the kernels never read it.

Keys are exact, not bucketed: q_block must divide the real seq_len and
head_fold the real num_heads, so a rounded-up entry could hand back a config
the shape cannot build. A missing key returns None and the caller uses its
analytic defaults; on a TPU that issues a UserWarning, once per key. A
kvcache| key holds the contiguous KV-cache decode kernel's (block_kv,
block_kv_compute, num_kv_stages); every other key holds a dense fwd / bwd
CONFIG_FIELDS tuple.
"""

import functools
import pathlib

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from ..core import tuning
from ..core.tiling import pick_tile
from ..core.tpu import NUM_LANES, get_device_name
from .config import (
    FWD_BLOCKS,
    FWD_KV_COMPUTE_BLOCKS,
    FWD_Q_COMPUTE_BLOCKS,
    MIN_Q_TILES_PER_BLOCK,
    BlockSizes,
    KVCacheConfig,
    QKVLayout,
    TokenMajorInfo,
    TunedConfig,
    default_block,
)
from .kernels.fwd_dense import make_flash_attn_mha

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
# Note (david): a dense window with a left bound runs fwd_pipeline's static
# band copies, which skip and mask per compute tile, so its blocks stay at the
# full-attention size. On v6e dense 16K causal prefill (16:8 heads of 256,
# 32:8 of 128, W 128-4096), 2048 blocks with 256-key kv tiles beat 512-key ones
# by 7-26%, while 128-key tiles fell off a code-size cliff (5.5-12.7 ms at
# W 512). A d256 window wider than a block pair runs the 64-fragment mask-free
# copy, which ran 9.63 ms at 256-key tiles (W 4096) against 1.98 at 512.
BAND_KV_COMPUTE_BLOCKS = (256, 128)


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
    head_dim_v == head_dim.
    """
    if head_dim_v != key_fields["head_dim"]:
        return None
    else:
        config = tuning.lookup(
            TUNED_CONFIGS_PATH, tuned_config_key(direction, **key_fields)
        )
        return None if config is None else tuple(config)


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


def get_tuned_kvcache_config(
    **key_fields: DTypeLike | int | bool | None,
) -> KVCacheConfig | None:
    config = tuning.lookup(TUNED_CONFIGS_PATH, kvcache_config_key(**key_fields))
    return None if config is None else tuple(config)


def resolve_tuned_kvcache_config(
    *, page_size: int | None, **key_fields: DTypeLike | int | bool | None
) -> KVCacheConfig | None:
    """The tuned config of one KV-cache decode build, or None for the
    kernel's default config.

    key_fields are kvcache_config_key's; page_size is the paged cache's, None
    for the contiguous pair. A paged build rejects an entry whose block_kv is
    not whole pages, with a warning.
    """
    tuned_config = get_tuned_kvcache_config(**key_fields)
    # Note (david): a paged block is whole pages, so a tuned block_kv that does
    # not tile the page cannot be used as-is.
    if (
        tuned_config is not None
        and page_size is not None
        and tuned_config[0] % page_size
    ):
        tuning.warn_untuned(
            TUNED_CONFIGS_PATH,
            kvcache_config_key(**key_fields),
            f"the entry's block_kv {tuned_config[0]} is not a multiple of the"
            f" page size {page_size}, and a paged block must be whole pages",
        )
        return None
    else:
        return tuned_config


def resolve_tuned_bwd_config(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    causal: bool,
    causal_offset: int,
    token_major: bool,
    head_dim: int | None,
) -> TunedConfig | None:
    """The tuned config of a flash_attn_bwd build with every block knob
    unset, or None.

    q, k, v, causal, causal_offset, token_major and head_dim are
    flash_attn_bwd's. The table is keyed on square, zero-offset MHA shapes, so
    any other backward gets None without a lookup.
    """
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
    q_seq_len = q.shape[-2]
    if (
        q_seq_len != k.shape[-2]
        or causal_offset != 0
        or num_heads != num_kv_heads
    ):
        return None
    else:
        return get_tuned_config(
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


@functools.cache
def get_kernel(
    num_heads: int,
    num_kv_heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    causal: bool,
    window_size: tuple[int, int],
    interpret: bool,
    softcap: float,
    head_dim: int,
    head_dim_v: int,
    causal_offset: int,
    varlen_max_seqlen_kv: int | None = None,
    varlen_max_seqlen_q: int | None = None,
    token_major: bool = False,
    kernel_batch: int | None = None,
    q_scale: float = 1.0,
    blocks_override: tuple[int, int, int, int] | None = None,
    return_lse: bool = False,
    heads_outer_batch: int | None = None,
) -> functools.partial:
    # Note (david): seqlen_q and seqlen_kv are padded lengths while
    # causal_offset comes from the original ones. Different original pairs can
    # pad to the same multiples of 128, so a rectangular mask is detected by
    # causal_offset too.
    is_rectangular = seqlen_q != seqlen_kv or causal_offset != 0
    is_varlen = varlen_max_seqlen_kv is not None
    has_window = window_size != (-1, -1)
    # Note (david): a tuned config fixes the blocks and the head fold together,
    # so a hit replaces the analytic block sizes wholesale. Windows, softcap and
    # rectangular masks were never measured and never consult it.
    if (
        blocks_override is None
        and not has_window
        and softcap == 0.0
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
            causal=causal,
            max_seqlen=varlen_max_seqlen_kv,
            return_lse=return_lse,
            num_kv_heads=num_kv_heads,
            heads_outer_batch=heads_outer_batch,
        )
    else:
        tuned_config = None
    if tuned_config is not None:
        (
            block_q,
            block_kv,
            block_q_compute,
            block_kv_compute,
            num_stages,
            head_fold,
            qkv_layout_name,
            tuned_transposed_pv,
        ) = tuned_config
        # Note (david): the per-sequence varlen schedule DMAs at arbitrary token
        # offsets, which needs the token axis second-minor, so it overrides a
        # tuned SEQ_MINOR layout and transposed_pv.
        if is_varlen:
            qkv_layout = QKVLayout.HEAD_DIM_MINOR
            transposed_pv = False
        else:
            qkv_layout = QKVLayout[qkv_layout_name.upper()]
            transposed_pv = tuned_transposed_pv
        block_sizes = BlockSizes(
            block_q=block_q,
            block_kv=block_kv,
            block_kv_compute=block_kv_compute,
            block_q_compute=block_q_compute,
            num_stages=num_stages,
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
                window_size=window_size,
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
                (head_dim, block_sizes.block_q, causal),
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
    if has_window:
        window_left = None if window_size[0] == -1 else window_size[0]
        # Note (david): window semantics: causal forces the right bound to 0,
        # and -1 means unbounded.
        if causal:
            window_right = 0
        elif window_size[1] == -1:
            window_right = None
        else:
            window_right = window_size[1]
        window = (window_left, window_right)
    else:
        window = None
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
        causal=causal,
        causal_offset=causal_offset,
        window=window,
        block_sizes=block_sizes,
        num_kv_heads=num_kv_heads,
        interpret=interpret,
        softcap=softcap,
        return_lse=return_lse,
        head_fold=kernel_head_fold,
        transposed_pv=transposed_pv,
        varlen_max_seqlen_kv=varlen_max_seqlen_kv,
        token_major=token_major_info,
        q_scale=q_scale,
        heads_outer_batch=heads_outer_batch,
    )
