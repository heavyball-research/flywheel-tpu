"""DMA stream helpers shared by the forward and backward kernels."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .block_sizes import NUM_LANES, QKVLayout, TokenMajorInfo, round_up

MIN_BLOCK_KV = 128
BF16_BITS = 16
FP8_BITS = 8
KV_WORD_BYTES = 4
MERGED_KV_DTYPES = (
    jnp.dtype(jnp.bfloat16),
    jnp.dtype(jnp.float8_e4m3fn),
    jnp.dtype(jnp.float8_e5m2),
)

# Note (david): a merged-cache token keeps each KV head's K and V in one u32
# word per lane, so one load yields both. A bf16 word is one head's (K, V),
# rows [k0, v0, k1, v1, ...]; an fp8 word is a head pair's (K, K, V, V), rows
# [k0, k1, v0, v1, ...], the bf16 interleave at 16-bit granularity, so the
# fp8 pack needs no cross-sublane shuffle. An odd fp8 head count pads one zero
# head.


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
    cache rows; an fp8 operand needs an even head count."""
    # Note (david): the halves move as integers, so every bit pattern (NaN
    # payloads, subnormals) survives; an fp8 pair packs into the 16-bit half.
    k_bits = pltpu.bitcast(k_rows, jnp.uint16).astype(jnp.uint32)
    v_bits = pltpu.bitcast(v_rows, jnp.uint16).astype(jnp.uint32)
    return pltpu.bitcast(k_bits | (v_bits << BF16_BITS), k_rows.dtype)


def kv_word_halves(
    words: jax.Array, kv_head: jax.Array | int, dtype: jnp.dtype
) -> tuple[jax.Array, jax.Array]:
    """(keys, values) of kv_head from its (rows, head_dim) u32 merged words."""
    if jnp.dtype(dtype) == jnp.bfloat16:
        keys, values = bf16_half(words, 0), bf16_half(words, 1)
    else:
        head_in_word = jnp.asarray(
            kv_head % kv_heads_per_word(dtype), jnp.uint32
        )
        key_shift = head_in_word * jnp.uint32(FP8_BITS)
        key_bits = words >> key_shift
        keys = pltpu.bitcast(key_bits.astype(jnp.uint8), dtype)
        values = pltpu.bitcast((key_bits >> BF16_BITS).astype(jnp.uint8), dtype)
    return keys, values


def advance(
    key: jax.Array,
    key_next: jax.Array,
    start_cur: Callable[[jax.Array], None],
    start_next: Callable[[jax.Array], None],
    wait: Callable[[jax.Array], None],
    loaded: jax.Array,
    inflight: jax.Array,
    slot: jax.Array,
    num_stages: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One ring-buffer step of a DMA stream; returns (loaded, inflight, slot).

    Invariant at the top of a step: slot holds (or is receiving) the block for
    key loaded unless loaded == -1, and at most this step's own block is in
    flight, prefetched by the previous step as key inflight. The step waits for
    key's block, starting it first on a prefetch miss, then prefetches key_next
    into the following slot when it differs from key.
    """
    is_new = key != loaded
    is_cold = jnp.logical_and(is_new, key != inflight)
    next_slot = lax.rem(slot + 1, num_stages)

    @pl.when(is_cold)
    def _():
        start_cur(next_slot)

    @pl.when(is_new)
    def _():
        wait(next_slot)

    new_slot = jnp.where(is_new, next_slot, slot)
    should_prefetch = key_next != key

    @pl.when(should_prefetch)
    def _():
        start_next(lax.rem(new_slot + 1, num_stages))

    new_inflight = jnp.where(should_prefetch, key_next, -1)
    return key, new_inflight, new_slot


def drain(
    num_copies: jax.Array,
    num_stages: int,
    wait: Callable[[int | jax.Array], None],
) -> None:
    """Wait out the last num_stages of num_copies started copies, oldest first.

    The copies still in flight occupy slots (num_copies - num_stages) ..
    (num_copies - 1); each wait is guarded by its copy having started at all.
    """
    for pending in range(num_stages, 0, -1):

        @pl.when(num_copies >= pending)
        def _wait_pending(pending=pending):
            wait((num_copies - pending) % num_stages)


def hbm_head_slice(
    ref: jax.Array,
    head: jax.Array | int,
    token_start: jax.Array | int,
    num_rows: int,
    layout: QKVLayout,
    fold: int,
) -> jax.Array:
    """Window of fold consecutive heads of a head-major ref as one HBM region.

    ref is (num_heads, seq, head_dim), or (num_heads, head_dim, seq) for
    SEQ_MINOR; head is the fold group's base head and the window spans num_rows
    tokens from token_start, so it needs no block alignment.
    """
    # Note (david): slicing the 3-D ref directly keeps the same HBM region; a
    # jnp.reshape outside the kernel made XLA materialize a copy whenever
    # head_dim needed lane padding.
    row_window = pl.ds(token_start, num_rows)
    head_window = head if fold == 1 else pl.ds(head, fold)
    if layout == QKVLayout.HEAD_DIM_MINOR:
        return ref.at[head_window, row_window, :]
    else:
        return ref.at[head_window, :, row_window]


def hbm_tm_slice(
    ref: jax.Array,
    head: jax.Array | int,
    token_start: jax.Array | int,
    num_rows: int,
    heads_per_row: int,
    width: int,
    is_batched: bool,
    fold: int,
) -> jax.Array:
    """Window of fold consecutive heads of a token-major ref, as lane columns.

    head is flat (batch * heads_per_row + local head) and, at fold > 1, the
    group's base head; heads_per_row and width are this stream's per-row head
    count and head_dim. is_batched addresses 3-D (batch, seq, heads * width)
    refs, otherwise 2-D (total, heads * width) refs. Consecutive heads occupy
    consecutive lanes, so a fold group is one contiguous column window; builders
    keep fold dividing heads_per_row, so a group never crosses a batch row.
    """
    batch_idx = head // heads_per_row
    local_head = head % heads_per_row
    row_window = pl.ds(token_start, num_rows)
    # Note (david): Mosaic needs a 128-aligned lane offset for an HBM DMA window.
    # A sub-tile width (d64) relies on builders enforcing
    # fold * width % 128 == 0; local_head is a multiple of fold, so the offset is
    # aligned, but the compiler cannot derive that from a traced head.
    if width % NUM_LANES:
        assert (fold * width) % NUM_LANES == 0, (fold, width)
        lane_start = pl.multiple_of(local_head * width, NUM_LANES)
    else:
        lane_start = local_head * width
    lane_window = pl.ds(lane_start, fold * width)
    if is_batched:
        return ref.at[batch_idx, row_window, lane_window]
    else:
        return ref.at[row_window, lane_window]


def hbm_window(
    ref: jax.Array,
    head: jax.Array | int,
    token_start: jax.Array | int,
    num_rows: int,
    *,
    width: int,
    heads_per_row: int,
    layout: QKVLayout,
    token_major: TokenMajorInfo | None,
    fold: int,
) -> jax.Array:
    """One stream's HBM window for whichever layout the build runs.

    Head-major (token_major is None) takes a leading-axis head window and layout
    picks the token axis; token-major takes a lane-axis column window of the
    stream's per-head width and ignores layout.
    """
    if token_major is None:
        return hbm_head_slice(ref, head, token_start, num_rows, layout, fold)
    else:
        return hbm_tm_slice(
            ref,
            head,
            token_start,
            num_rows,
            heads_per_row,
            width,
            token_major.batch is not None,
            fold,
        )


def fold_row(
    ref: jax.Array,
    fold_idx: jax.Array | int,
    block: int | None,
    head_fold: int,
    layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR,
    lane_width: int | None = None,
) -> jax.Array:
    """Select fold group fold_idx from a per-slot VMEM scratch ref.

    head_fold == 1 slices the plain (block, head_dim) ref, or (head_dim, block)
    for SEQ_MINOR. head_fold > 1 indexes the leading fold axis, or, when
    lane_width is set (token-major), the lane column window at
    fold_idx * lane_width.
    """
    # Note (david): fold_idx may be traced. Leading-axis indexing and
    # 128-aligned lane offsets are valid dynamically, but a sub-tile lane_width
    # (d64) gives only 64-aligned offsets, which Mosaic accepts for a static
    # fold_idx only, so those builds pass a Python int.
    if head_fold == 1 and layout == QKVLayout.HEAD_DIM_MINOR:
        return ref.at[pl.ds(fold_idx * block, block), :]
    elif head_fold == 1:
        return ref.at[:, pl.ds(fold_idx * block, block)]
    elif lane_width is not None:
        return ref.at[:, pl.ds(fold_idx * lane_width, lane_width)]
    else:
        return ref.at[fold_idx]


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
    """(kv_heads_per_fragment * block_kv_compute, head_dim) bf16 rows of one head
    group, read from a slot of kv_stage: keys from the K staging buffer, values
    from the V staging buffer."""
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
        # Note (david): a wide head loads each 128-lane half with its own strided
        # load and joins the halves along lanes. Loading (rows, halves, 128) and
        # reshaping instead interleaves sublanes; on v7x decode at 32 heads of 256
        # that was 40K rotate/combine ops per 4096-token block, 8.55 against 5.22
        # ms a step at 4 KV heads (flywheel-tpu PR #7).
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
