# Copyright 2026 heavyball-research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Merged KV-cache word layout: packing K and V rows into u32 words and
reading them back."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from flywheel_tpu.core.tiling import round_up
from flywheel_tpu.core.tpu import NUM_LANES

MIN_BLOCK_KV = 128
BF16_BITS = 16
FP8_BITS = 8
KV_WORD_BYTES = 4
MERGED_KV_DTYPES = (
    jnp.dtype(jnp.bfloat16),
    jnp.dtype(jnp.float8_e4m3fn),
    jnp.dtype(jnp.float8_e5m2),
)
# Note (david): one interleave step packs about 8 u32 vregs (32 KiB) per
# operand.
INTERLEAVE_STEP_ELEMENTS = 8192

# Note (david): a merged-cache token's rows are each KV head's K row then its
# V row, [k0, v0, k1, v1, ...], for every dtype, so one u32 word per lane holds
# a bf16 head's (K, V) or an fp8 head pair's (K, V, K, V), and one load yields
# both. An odd fp8 head count pads one zero head. An fp8 head's (K, V) bytes
# share one 16-bit half (batched RPA's order), so its unpack is one u32 -> u16
# pack and a u16-level shift for V. The head pair's (K, K, V, V) order this
# replaced kept V in the other half, a u32 shift of every staged word, which
# added VALU work to every fp8 decode step at one KV head.


def check_merged_kv_dtype(dtype: jnp.dtype) -> None:
    if jnp.dtype(dtype) not in MERGED_KV_DTYPES:
        raise NotImplementedError(
            "the merged KV cache holds bfloat16, float8_e4m3fn or float8_e5m2;"
            f" got {jnp.dtype(dtype)}."
        )


def kv_word_rows(dtype: jnp.dtype) -> int:
    """Merged-cache head rows one u32 word per lane holds: 2 for bf16, 4 for
    fp8."""
    return KV_WORD_BYTES // jnp.dtype(dtype).itemsize


def kv_heads_per_word(dtype: jnp.dtype) -> int:
    return kv_word_rows(dtype) // 2


def merged_cache_heads(num_kv_heads: int, dtype: jnp.dtype) -> int:
    """Head rows of one merged-cache token: 2 * num_kv_heads, padded to whole
    u32 words."""
    return 2 * round_up(num_kv_heads, kv_heads_per_word(dtype))


def resolve_merged_kv_heads(
    cache_heads: int, dtype: jnp.dtype, nheads_k: int | None
) -> int:
    """KV heads of a merged cache whose tokens hold cache_heads head rows:
    nheads_k, else half the rows of a bf16 cache. Checks the dtype and that
    cache_heads is that head count's merged_cache_heads."""
    check_merged_kv_dtype(dtype)
    if nheads_k is not None:
        num_kv_heads = nheads_k
    elif kv_heads_per_word(dtype) > 1:
        raise ValueError(
            "an fp8 cache pads an odd nheads_k, so its head axis does not give"
            " the KV head count; pass nheads_k."
        )
    else:
        num_kv_heads = cache_heads // 2
    if merged_cache_heads(num_kv_heads, dtype) != cache_heads:
        raise ValueError(
            f"a merged {jnp.dtype(dtype)} cache of {num_kv_heads} KV heads"
            f" holds {merged_cache_heads(num_kv_heads, dtype)} head rows per"
            f" token; got {cache_heads}."
        )
    return num_kv_heads


def interleave_kv_rows(k_rows: jax.Array, v_rows: jax.Array) -> jax.Array:
    """(..., heads, head_dim) K and V rows -> (..., 2 * heads, head_dim) merged
    cache rows, each head's K row then its V row."""
    # Note (david): the rows move as integers, so every bit pattern (NaN
    # payloads, subnormals) survives; a head's (K, V) pair is one integer of
    # twice the dtype's width, which the bitcast back splits into its 2 rows.
    bits = jnp.dtype(k_rows.dtype).itemsize * 8
    element = jnp.dtype(f"uint{bits}")
    pair = jnp.dtype(f"uint{2 * bits}")
    k_bits = pltpu.bitcast(k_rows, element).astype(jnp.uint32)
    v_bits = pltpu.bitcast(v_rows, element).astype(jnp.uint32)
    return pltpu.bitcast(
        (k_bits | (v_bits << jnp.uint32(bits))).astype(pair), k_rows.dtype
    )


def kv_word_halves(
    words: jax.Array, kv_head: jax.Array | int, dtype: jnp.dtype
) -> tuple[jax.Array, jax.Array]:
    """(keys, values) of kv_head from its (rows, head_dim) u32 merged words;
    rows is even for an fp8 cache."""
    if jnp.dtype(dtype) == jnp.bfloat16:
        keys, values = bf16_half(words, 0), bf16_half(words, 1)
    else:
        # Note (david): the head's (K, V) bytes are the word's low (even head)
        # or high (odd head) 16-bit half; pairs of those halves, viewed as u32,
        # shift V into the low byte of each.
        if isinstance(kv_head, int) and kv_head % 2 == 0:
            head_bits = words
        else:
            head_bits = words >> (
                jnp.asarray(kv_head % kv_heads_per_word(dtype), jnp.uint32)
                * jnp.uint32(BF16_BITS)
            )
        halves = head_bits.astype(jnp.uint16)
        keys = pltpu.bitcast(halves.astype(jnp.uint8), dtype)
        value_bits = pltpu.bitcast(
            pltpu.bitcast(halves, jnp.uint32) >> jnp.uint32(FP8_BITS),
            jnp.uint16,
        )
        values = pltpu.bitcast(value_bits.astype(jnp.uint8), dtype)
    return keys, values


def load_kv_fragment(
    kv_stage: jax.Array,
    slot: jax.Array,
    head_group: int,
    kv_compute_index: int,
    *,
    block_kv_compute: int,
    kv_heads_per_fragment: int,
    is_cache_head_major: bool,
    is_bitcast_load: bool,
) -> jax.Array:
    """(kv_heads_per_fragment * block_kv_compute, head_dim) bf16 rows of one
    head group, read from a slot of kv_stage: keys from the K staging buffer,
    values from the V staging buffer."""
    head_dim = kv_stage.shape[-1]
    compute_start = pl.multiple_of(
        kv_compute_index * block_kv_compute, MIN_BLOCK_KV
    )
    if is_cache_head_major:
        return kv_stage[
            slot,
            pl.ds(head_group, 1),
            pl.ds(compute_start, block_kv_compute),
            :,
        ].reshape(block_kv_compute, head_dim)
    elif not is_bitcast_load:
        return kv_stage[
            slot,
            pl.ds(compute_start, block_kv_compute),
            pl.ds(head_group, 1),
            :,
        ].reshape(block_kv_compute, head_dim)
    else:
        word_stride = kv_stage.shape[2]
        words_ref = staged_words(kv_stage, slot)
        pair_index = (
            head_group if kv_heads_per_fragment == 2 else head_group // 2
        )
        word_start = compute_start * word_stride + pair_index
        words = load_staged_words(
            words_ref, word_start, block_kv_compute, word_stride
        )
        if kv_heads_per_fragment == 2:
            return pltpu.bitcast(words, jnp.bfloat16)
        else:
            return bf16_half(words, head_group % 2)


def bank_safe_word_stride(head_pairs: int) -> int:
    """u32 word rows a pair-packed staging slot gives each token: its
    head_pairs, plus one unused pad row when head_pairs is even."""
    # Note (david): a strided word load reads its 8 sublanes from rows stride
    # apart, and an even stride lands several of them in one VMEM bank, which
    # splits the load into one per bank collision joined with vor. An odd
    # stride keeps the 8 rows in distinct banks (RPA v3's has_bank_conflicts
    # pad). v7x merged decode, block-loop bundles: 32 KV heads of 256 (one
    # load per sublane unpadded) 22753 against 7149 per 512 tokens, 4 KV heads
    # (a 2-way split) 2450 against 1758 per 1024 tokens.
    return head_pairs + 1 if head_pairs % 2 == 0 else head_pairs


def staged_words(kv_stage: jax.Array, slot: jax.Array) -> jax.Array:
    """A (block_kv * word_stride, head_dim) u32 view of one pair-packed staging
    slot of (stages, block_kv, word_stride, 2, head_dim) bf16; word row
    token * word_stride + pair holds that pair's two heads."""
    _, block_kv, word_stride, _, head_dim = kv_stage.shape
    return (
        kv_stage.bitcast(jnp.uint32)
        .at[slot]
        .reshape(block_kv * word_stride, head_dim)
    )


def load_staged_words(
    words_ref: jax.Array,
    word_start: jax.Array | int,
    num_rows: int,
    stride: int,
) -> jax.Array:
    """(num_rows, head_dim) u32 words at rows word_start + i * stride."""
    num_word_rows, head_dim = words_ref.shape
    folds = head_dim // NUM_LANES
    if head_dim <= NUM_LANES:
        return words_ref[pl.ds(word_start, num_rows, stride)]
    else:
        # Note (david): a wide head loads each 128-lane half with its own
        # strided load and joins the halves along lanes. Loading (rows, halves,
        # 128) and reshaping instead interleaves sublanes; on v7x decode at 32
        # heads of 256 that was 40K rotate/combine ops per 4096-token block,
        # 8.55 against 5.22 ms a step at 4 KV heads (flywheel-tpu PR #7).
        lanes_ref = words_ref.reshape(num_word_rows * folds, NUM_LANES)
        return jnp.concatenate(
            [
                lanes_ref[
                    pl.ds(word_start * folds + fold, num_rows, stride * folds)
                ]
                for fold in range(folds)
            ],
            axis=1,
        )


def bf16_half(words: jax.Array, lane: jax.Array | int) -> jax.Array:
    """The bf16 head in a u32 word's low (lane 0) or high (lane 1) half."""
    lane_bits = words >> jnp.uint32(lane * BF16_BITS)
    return pltpu.bitcast(lane_bits.astype(jnp.uint16), jnp.bfloat16)
