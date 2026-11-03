"""Single-token KV-cache decode: one persistent Pallas call appends every row's
new token and attends its whole cache."""

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..tuned_block_sizes import (
    KVCacheConfig,
    get_device_name,
    get_tuned_kvcache_config,
    kvcache_config_key,
    warn_untuned,
)
from .block_sizes import (
    BF16_BYTES,
    F32_BYTES,
    LOG2E,
    NN_DIM_NUMBERS,
    NT_DIM_NUMBERS,
    NUM_LANES,
    NUM_SUBLANES,
    STAGES,
    next_pow2,
    round_up,
    vmem_limit_bytes,
)
from .copy_utils import (
    FP8_BITS,
    MIN_BLOCK_KV,
    bank_safe_word_stride,
    kv_heads_per_word,
    kv_word_halves,
    kv_word_rows,
    load_kv_fragment,
    load_staged_words,
    merged_cache_heads,
    resolve_merged_kv_heads,
    staged_words,
)
from .fwd_pipeline import BF16_TILE_ROWS, fold_sink, overflow_guard_threshold

SUPPORTED_HEAD_DIMS = (64, 128, 256)
DEFAULT_NUM_KV_STAGES = 2
NUM_IO_STAGES = 2
SEM_K = 0
SEM_V = 1
SEM_K_WRITE = 2
SEM_V_WRITE = 3
SEM_Q = 4
SEM_OUT = 5
NUM_SEMS = 6
# Note (david): a row's running state alternates between two parities, so a
# row's last PV and output finish can run inside the next row's first block;
# the trash accumulator takes the pending PV issued before any block left one.
NUM_ROW_PARITIES = 2
TRASH_PARITY = NUM_ROW_PARITIES
PENDING_SLOT = 0
PENDING_KV_COMPUTE = 1
PENDING_PARITY = 2
PENDING_FINISH_ROW = 3
NUM_PENDING_FIELDS = 4
ITEM_ROW_FIRST = 1
ITEM_ROW_LAST = 2
ITEM_EMPTY_ROW = 4
BLOCK_KV_COMPUTE_CANDIDATES = (1024, 512, 384, 256, 128)
STAGED_KV_ELEMENT_BUDGET = 8 * 1024 * 1024
# Note (david): a row runs only the kv compute tiles that hold its keys, but a
# bigger block still issues a copy for every page in it. At Qwen3-4B's 32:8
# heads of 128 with a 32K-token table row on v6e, 256 rows of 64-1024 tokens
# took 1.15 ms with 4096-token blocks against 1.08 ms with 1024-token ones
# (1.82 ms when every tile of the block ran), while a 16K context took 15.53
# against 15.33 ms.
MAX_BLOCK_KV = 1024
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


def kv_staging_slots(num_kv_stages: int, items_per_step: int) -> int:
    """Staging slots of the KV ring; a packed slot holds a whole step."""
    # Note (david): the one-item ring holds one block more than the prefetch
    # needs. A block's last PV runs in the next item, after that item's first
    # softmax, and reads its V where the block was staged, so the prefetch the
    # next item starts must land in another slot. A packed step finishes every
    # PV it starts, so its ring is the prefetch alone.
    return num_kv_stages + 1 if items_per_step == 1 else num_kv_stages


def cache_write_slots(num_kv_stages: int, items_per_step: int) -> int:
    """Append write-back tiles; row r writes through tile r % slots. A packed
    call writes its new tokens straight from the new_kv block and takes
    none."""
    return num_kv_stages if items_per_step == 1 else 0


def lane_broadcast(value: jax.Array, width: int) -> jax.Array:
    """(rows, width) from (rows, w) with w == 1 or w dividing width.

    A (rows, 128) value whose lanes all hold the row's number repeats along
    the lane axis as whole vregs, which costs no data movement, where a
    (rows, 1) slice loaded from VMEM needs a cross-lane broadcast per use.
    """
    rows, current = value.shape
    if current == width:
        return value
    elif current == 1:
        return jnp.broadcast_to(value, (rows, width))
    elif width < current:
        return value[:, :width]
    else:
        assert width % current == 0, (current, width)
        return jnp.concatenate([value] * (width // current), axis=1)


def static_anchor_update(
    scores: jax.Array,
    row_max_prev: jax.Array,
    row_sum_prev: jax.Array,
    *,
    is_first: bool | jax.Array,
    guard_threshold: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Online-softmax step in log2 units over (rows, w) running state, w == 1
    or NUM_LANES with every lane holding its row's value.

    Returns (probabilities, row_max, row_sum, rescale), the state at the
    width it came in.
    """
    # Note (david): row_max moves only once a fragment's max exceeds it by more
    # than guard_threshold, which keeps every exp2 finite without tracking the
    # exact running max.
    fragment_max = scores.max(axis=-1, keepdims=True)
    should_raise_max = jnp.logical_and(
        jnp.logical_not(is_first),
        fragment_max - row_max_prev > guard_threshold,
    )
    row_max = jnp.where(
        is_first,
        fragment_max,
        jnp.where(
            should_raise_max,
            jnp.maximum(row_max_prev, fragment_max),
            row_max_prev,
        ),
    )
    probabilities = jnp.exp2(scores - lane_broadcast(row_max, scores.shape[-1]))
    fragment_sum = probabilities.sum(axis=-1, keepdims=True)
    rescale = jnp.where(is_first, 1.0, jnp.exp2(row_max_prev - row_max))
    row_sum = jnp.where(
        is_first,
        fragment_sum,
        rescale * row_sum_prev + fragment_sum,
    )
    return probabilities, row_max, row_sum, rescale


def load_kv_pair_fragment(
    kv_stage: jax.Array,
    slot: jax.Array,
    kv_head: int,
    kv_compute_index: int,
    *,
    block_kv_compute: int,
    is_bitcast_load: bool,
    kv_heads_per_fragment: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """(keys, values) of kv_head, each (block_kv_compute, head_dim) in the
    cache dtype, read from a merged-cache staging slot.

    With kv_heads_per_fragment 2 (an fp8 cache), kv_head starts a u32 word's
    head pair and each of keys and values is (2 * block_kv_compute, head_dim):
    row 2 * t holds token t of kv_head, row 2 * t + 1 token t of kv_head + 1.

    A staged token row holds every KV head's K and V rows: (stages, block_kv,
    word_stride, kv_word_rows, head_dim) with is_bitcast_load, its first
    cache_heads / kv_word_rows word rows live, else (stages, block_kv,
    cache_heads, head_dim).
    """
    head_dim = kv_stage.shape[-1]
    dtype = kv_stage.dtype
    heads_per_word = kv_heads_per_word(dtype)
    compute_start = pl.multiple_of(
        kv_compute_index * block_kv_compute, MIN_BLOCK_KV
    )
    if kv_heads_per_fragment == 2:
        if not is_bitcast_load:
            # Note (david): the pair's word window is its rows [k, v, k', v'].
            window = kv_stage[
                slot,
                pl.ds(compute_start, block_kv_compute),
                pl.ds(2 * kv_head, 2 * heads_per_word),
                :,
            ].reshape(block_kv_compute, heads_per_word, 2, head_dim)
            return tuple(
                window[:, :, kv_part, :].reshape(2 * block_kv_compute, head_dim)
                for kv_part in range(2)
            )
        # Note: a word's 16-bit halves are the pair's two heads, each its K
        # byte then its V byte, so the u16 view packed to bytes is the
        # head-interleaved K tile, and the words shifted by a byte give the V
        # tile the same way.
        word_stride = kv_stage.shape[2]
        words = load_staged_words(
            staged_words(kv_stage, slot),
            compute_start * word_stride + kv_head // heads_per_word,
            block_kv_compute,
            word_stride,
        )
        return tuple(
            pltpu.bitcast(
                pltpu.bitcast(part_bits, jnp.uint16).astype(jnp.uint8), dtype
            )
            for part_bits in (words, words >> jnp.uint32(FP8_BITS))
        )
    if not is_bitcast_load:
        keys, values = (
            kv_stage[
                slot,
                pl.ds(compute_start, block_kv_compute),
                pl.ds(2 * kv_head + kv_part, 1),
                :,
            ].reshape(block_kv_compute, head_dim)
            for kv_part in range(2)
        )
        return keys, values
    else:
        # Note (david): the K and V of KV head h share u32 word row token *
        # word_stride + h // heads_per_word, so one strided word load yields
        # both. At one KV head this costs 1.4-1.5x the head-major decode of an
        # unmerged K/V cache (6 query heads, head_dim 256, v6e), yet splitting
        # the pair into dense K/V buffers per block (1.45-2.25x), folding the
        # heads into the row width (6.5-12x) and scoring the pair interleaved
        # (1.46-2.02x) all measured worse. Scoring it interleaved at one query
        # head per KV head (32 KV heads of 256) doubles the MXU pushes and lost
        # too: 9750 against 7149 v7x bundles per 512-token block, 9.38 against
        # 6.19 ms a compute-only v6e step at batch 32.
        word_stride = kv_stage.shape[2]
        words = load_staged_words(
            staged_words(kv_stage, slot),
            compute_start * word_stride + kv_head // heads_per_word,
            block_kv_compute,
            word_stride,
        )
        return kv_word_halves(words, kv_head, dtype)


def fragment_geometry(
    *,
    num_query_heads: int,
    num_kv_heads: int,
    cache_dtype: jnp.dtype,
    is_merged_cache: bool,
    is_bitcast_load: bool,
) -> tuple[int, int]:
    """(kv_heads_per_fragment, kv_heads_per_group): the KV heads one score tile
    reads and the head groups one 8-row query tile folds."""
    query_heads_per_kv_head = num_query_heads // num_kv_heads
    # Note (david): a bitcast head-pair tile interleaves its two heads per
    # token, so both heads' query groups ride one MXU pass with the off-parity
    # columns masked. That is free while the pair's query rows fit one 8-sublane
    # f32 tile; past it the masked half doubles the softmax work, so larger
    # groups unpack one head per fragment. A merged row's u32 word pairs a
    # head's K with its V instead, so a merged build always unpacks one head per
    # fragment.
    # Note: an fp8 merged word holds a head pair, each head's K byte then its
    # V byte in one 16-bit half, so the pair rides one fragment like the bf16
    # head-pair bitcast.
    is_pair_word_fragment = (
        is_merged_cache
        and kv_heads_per_word(cache_dtype) == 2
        and NUM_SUBLANES // 2 < 2 * query_heads_per_kv_head <= NUM_SUBLANES
        and num_kv_heads % 2 == 0
    )
    if (
        is_bitcast_load
        and 2 * query_heads_per_kv_head <= NUM_SUBLANES
        and not is_merged_cache
    ) or is_pair_word_fragment:
        kv_heads_per_fragment = 2
    else:
        kv_heads_per_fragment = 1
    # Note (david): a merged build whose KV head serves fewer than 8 query
    # heads scores several KV heads through one 8-row query tile: each head's
    # keys are pushed once, as before, but the LHS carries the whole group's
    # query rows, and the head keeps its own rows of the result. The group then
    # runs one softmax over a full f32 tile instead of one per head over a
    # mostly empty one, and its running max, sum and accumulator are whole
    # tiles, which drops the per-head cross-lane broadcasts, sublane rotates
    # and single-row state stores.
    if (
        is_merged_cache
        and not is_pair_word_fragment
        and NUM_SUBLANES % query_heads_per_kv_head == 0
        and num_kv_heads % (NUM_SUBLANES // query_heads_per_kv_head) == 0
    ):
        kv_heads_per_group = NUM_SUBLANES // query_heads_per_kv_head
    else:
        kv_heads_per_group = 1
    return kv_heads_per_fragment, kv_heads_per_group


class DecodeScheduleLayout(NamedTuple):
    """Static layout of the decode schedule, one flat int32 SMEM array.

    An item is one KV block of one row, items in row order; a row with no block
    in its window gets one empty item that only finishes its output. A step is
    items_per_step consecutive items, and max_items is whole steps. Field f of
    item i sits at f * max_items + i, then each step's staged token count, then
    each write row's target, then the item count.

    A packed paged schedule (pages_per_item > 0) also carries each item's
    copies: the block_table index of its block's first page (table_base) and
    page p's staged token count (tokens_p).
    A packed schedule with an append (write_rows > 0) carries each row's write
    target: the cache row (the physical page when paged) at
    row_write_offset + row, and the token's offset in it write_rows later.
    """

    max_items: int
    items_per_step: int = 1
    pages_per_item: int = 0
    write_rows: int = 0

    ITEM_FIELDS = ("row", "block", "load", "flags")

    @property
    def item_fields(self) -> tuple[str, ...]:
        if not self.pages_per_item:
            return self.ITEM_FIELDS
        return (
            self.ITEM_FIELDS
            + ("table_base",)
            + tuple(f"tokens_{page}" for page in range(self.pages_per_item))
        )

    def item_offset(self, name: str) -> int:
        return self.item_fields.index(name) * self.max_items

    @property
    def max_steps(self) -> int:
        return self.max_items // self.items_per_step

    @property
    def step_load_offset(self) -> int:
        return len(self.item_fields) * self.max_items

    @property
    def row_write_offset(self) -> int:
        return self.step_load_offset + self.max_steps

    @property
    def num_items_offset(self) -> int:
        return self.row_write_offset + 2 * self.write_rows

    @property
    def size(self) -> int:
        return self.num_items_offset + 1


def build_decode_schedule(
    cache_seqlens: jax.Array,
    num_active: jax.Array,
    *,
    batch: int,
    layout: DecodeScheduleLayout,
    block_kv: int,
    has_new: bool,
    left: int | None,
    cache_tile_rows: int,
    cache_batch_idx: jax.Array | None = None,
    block_table: jax.Array | None = None,
    page_size: int | None = None,
    pages_per_seq: int | None = None,
) -> jax.Array:
    """The decode schedule of one call, a pure function of the call's lengths,
    so XLA builds it once for every layer that shares them. A packed layout
    also reads cache_batch_idx and, when paged, the (padded, flat)
    block_table, to resolve its items' pages and its rows' write targets.

    Each item carries its row, block index, staged token count (whole cache
    tiles, as the block's wait sizes them) and its ITEM_* flags. The kernel
    copies only the pages that hold those tokens.

    The items that pad the last step stage nothing and continue the last row
    with a block past its keys, so they leave its state, and the output the
    packed kernel rewrites from it, exactly as they were.
    """
    if layout.max_items % layout.items_per_step:
        raise ValueError(
            f"max_items={layout.max_items} must be whole steps of"
            f" {layout.items_per_step} items."
        )
    rows = jnp.arange(batch, dtype=jnp.int32)
    is_active = rows < num_active
    cache_len = jnp.where(is_active, cache_seqlens.astype(jnp.int32), 0)
    total_len = cache_len + int(has_new)
    if left is None:
        first_key = jnp.zeros_like(total_len)
    else:
        first_key = jnp.maximum(0, total_len - 1 - left)
    first_block = first_key // block_kv
    num_blocks = jnp.maximum(
        (total_len + block_kv - 1) // block_kv - first_block, 0
    )
    row_items = jnp.where(is_active, jnp.maximum(num_blocks, 1), 0)
    row_item_end = jnp.cumsum(row_items)
    row_item_start = row_item_end - row_items
    copy_limit = (
        (total_len + cache_tile_rows - 1) // cache_tile_rows * cache_tile_rows
    )

    # Note (david): items find their row through a (max_items, batch)
    # membership mask and its row reductions, all vector work; gathering and
    # scattering over a compacted page list instead is scalar work that grows
    # with the page count.
    items = jnp.arange(layout.max_items, dtype=jnp.int32)
    is_in_row = (items[:, None] >= row_item_start[None, :]) & (
        items[:, None] < row_item_end[None, :]
    )

    def _at_row(values):
        return jnp.sum(jnp.where(is_in_row, values[None, :], 0), axis=1)

    is_item = items < row_item_end[-1]
    block_in_row = items - _at_row(row_item_start)
    item_block = _at_row(first_block) + block_in_row
    is_empty = _at_row((num_blocks == 0).astype(jnp.int32)) > 0
    item_load = jnp.where(
        is_item & jnp.logical_not(is_empty),
        jnp.clip(_at_row(copy_limit) - item_block * block_kv, 0, block_kv),
        0,
    )
    item_flags = jnp.where(
        is_item,
        jnp.where(block_in_row == 0, ITEM_ROW_FIRST, 0)
        | jnp.where(block_in_row == _at_row(row_items) - 1, ITEM_ROW_LAST, 0)
        | jnp.where(is_empty, ITEM_EMPTY_ROW, 0),
        0,
    )
    # Note (david): a padding item's block starts at or past its row's
    # total_len, so every key it scores is masked, and it is never the append
    # block cache_len // block_kv.
    last_row = jnp.maximum(jnp.minimum(num_active, batch) - 1, 0)
    padding_block = (total_len[last_row] + block_kv - 1) // block_kv
    step_load = item_load.reshape(layout.max_steps, layout.items_per_step).sum(
        axis=1
    )
    item_row = jnp.where(is_item, _at_row(rows), last_row)
    item_block = jnp.where(is_item, item_block, padding_block)
    # Note (david): rows past num_active never copy or append, and the clips
    # keep their arbitrary cache_batch_idx inside the table.
    cache_row = jnp.where(is_active, cache_batch_idx.astype(jnp.int32), 0)
    if layout.pages_per_item:
        # Note (david): a copy then issues with one block_table read, not a
        # ~18-op cache_batch_idx / block_table / clip chain. Gathering every
        # item's pages here instead is a gather per page of every item. A
        # padding item's base is clipped into the table, and its copies are
        # empty.
        table_base = jnp.clip(
            _at_row(cache_row) * pages_per_seq
            + item_block * layout.pages_per_item,
            0,
            block_table.shape[0] - layout.pages_per_item,
        )
        pages = jnp.arange(layout.pages_per_item, dtype=jnp.int32)
        item_tokens = jnp.clip(
            item_load[:, None] - pages[None, :] * page_size, 0, page_size
        )
        page_fields = [table_base] + [
            item_tokens[:, page] for page in range(layout.pages_per_item)
        ]
    else:
        page_fields = []
    # Note (david): a packed call writes each row's new token straight to its
    # cache slot before staging any block, so the target is the token itself,
    # not a cache tile.
    if not layout.write_rows:
        row_write_fields = []
    elif page_size is None:
        row_write_fields = [cache_row, cache_len]
    else:
        row_write_fields = [
            jnp.take(
                block_table,
                cache_row * pages_per_seq + cache_len // page_size,
                mode="clip",
            ),
            cache_len % page_size,
        ]
    return jnp.concatenate([
        item_row,
        item_block,
        item_load,
        item_flags,
        *page_fields,
        step_load,
        *row_write_fields,
        row_item_end[-1:],
    ]).astype(jnp.int32)


def kvcache_kernel(
    cache_seqlens_ref: jax.Array,
    cache_batch_idx_ref: jax.Array,
    num_active_ref: jax.Array,
    v_scale_ref: jax.Array,
    schedule_ref: jax.Array,
    *refs: jax.Array,
    layout: DecodeScheduleLayout,
    batch: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_kv: int,
    block_kv_compute: int,
    num_kv_stages: int,
    items_per_step: int,
    is_cache_head_major: bool,
    is_bitcast_load: bool,
    return_lse: bool,
    has_new: bool,
    left: int | None,
    query_tokens: int,
    causal: bool,
    log2_softcap: float,
    guard_threshold: float,
    page_size: int | None,
    pages_per_seq: int | None,
    is_merged_cache: bool,
    has_sinks: bool,
) -> None:
    remaining_refs = list(refs)
    block_table_ref = remaining_refs.pop(0) if page_size is not None else None
    num_active = num_active_ref[0]
    if is_merged_cache:
        # Note (david): a merged build's new token arrives as one
        # (staged_kv_heads, head_dim) row already in the cache's head order, so
        # the append writes it as a ready token row.
        q_ref, new_kv_ref, k_cache_ref, *remaining_refs = remaining_refs
        k_ref = v_ref = v_cache_ref = None
    else:
        (
            q_ref,
            k_ref,
            v_ref,
            k_cache_ref,
            v_cache_ref,
            *remaining_refs,
        ) = remaining_refs
        new_kv_ref = None
    sinks_ref = remaining_refs.pop(0) if has_sinks else None
    if is_merged_cache:
        out_ref, k_out_ref, *remaining_refs = remaining_refs
        v_out_ref = None
    else:
        out_ref, k_out_ref, v_out_ref, *remaining_refs = remaining_refs
    lse_ref = remaining_refs.pop(0) if return_lse else None
    (
        k_buffers,
        v_buffers,
        k_write_buffers,
        v_write_buffers,
        q_buffers,
        out_buffers,
        row_max_ref,
        row_sum_ref,
        accumulator_ref,
        pending_probabilities_ref,
        pending_rescale_ref,
        pending_ref,
        staged_values_ref,
        sems,
    ) = remaining_refs

    is_packed = items_per_step > 1
    query_heads_per_kv_head = num_query_heads // num_kv_heads
    cache_dtype = k_cache_ref.dtype
    # Note (david): a merged build stages whole token rows, every KV head's
    # (K, V) pair (and an fp8 cache's zero pad head).
    if is_merged_cache:
        staged_kv_heads = merged_cache_heads(num_kv_heads, cache_dtype)
    else:
        staged_kv_heads = num_kv_heads
    word_rows = kv_word_rows(cache_dtype)
    num_kv_compute_fragments = block_kv // block_kv_compute
    kv_heads_per_fragment, kv_heads_per_group = fragment_geometry(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        cache_dtype=cache_dtype,
        is_merged_cache=is_merged_cache,
        is_bitcast_load=is_bitcast_load,
    )
    num_head_groups = (
        num_kv_heads // kv_heads_per_fragment // kv_heads_per_group
    )
    rows_per_fragment = (
        query_heads_per_kv_head * kv_heads_per_fragment * kv_heads_per_group
    )
    fragment_keys = kv_heads_per_fragment * block_kv_compute
    # Note (david): row r of a group's tile belongs to the group's KV head
    # r // query_heads_per_kv_head.
    if kv_heads_per_group == 1:
        score_row_head = pv_row_head = None
    else:
        score_row_head = (
            lax.broadcasted_iota(
                jnp.int32, (rows_per_fragment, fragment_keys), 0
            )
            // query_heads_per_kv_head
        )
        pv_row_head = (
            lax.broadcasted_iota(jnp.int32, (rows_per_fragment, head_dim), 0)
            // query_heads_per_kv_head
        )
    # Note (david): a folded window lays a KV head's rows out token by token, so
    # row r is window token t = r // group, which causal stops
    # query_tokens - 1 - t keys short of total_len.
    if causal and query_tokens > 1:
        row_key_offset = lax.broadcasted_iota(
            jnp.int32, (rows_per_fragment, fragment_keys), 0
        ) % query_heads_per_kv_head // (
            query_heads_per_kv_head // query_tokens
        ) - (
            query_tokens - 1
        )
    else:
        row_key_offset = None
    if kv_heads_per_fragment == 1:
        parity_mask = None
    else:
        column_index = jnp.arange(fragment_keys, dtype=jnp.int32)
        row_parity = (
            jnp.arange(rows_per_fragment, dtype=jnp.int32)
            // query_heads_per_kv_head
        )
        parity_mask = (
            column_index[None, :] % kv_heads_per_fragment == row_parity[:, None]
        )
    if is_merged_cache:
        kv_pair_fragment = functools.partial(
            load_kv_pair_fragment,
            block_kv_compute=block_kv_compute,
            is_bitcast_load=is_bitcast_load,
            kv_heads_per_fragment=kv_heads_per_fragment,
        )
    else:
        kv_fragment = functools.partial(
            load_kv_fragment,
            block_kv_compute=block_kv_compute,
            kv_heads_per_fragment=kv_heads_per_fragment,
            is_cache_head_major=is_cache_head_major,
            is_bitcast_load=is_bitcast_load,
        )

    # Note (david): every cache DMA on the token axis moves whole
    # cache_tile_rows tiles. A head-major stage puts tokens on its bf16 sublane
    # axis, which v7x tiles 16 rows deep (BF16_TILE_ROWS; v6e 8), and Mosaic
    # sizes a DMA and its wait between the 8-row HBM tiles and that stage in
    # whole 16-row tiles, so an 8-row remainder was dropped from the copy and
    # the wait returned before the DMA landed (v7x read stale tokens; the
    # leftover semaphore count then released later waits early). A token-major
    # stage puts tokens on a leading axis, where each token is whole tiles and
    # the 8-row granule is exact.
    cache_tile_rows = BF16_TILE_ROWS if is_cache_head_major else NUM_SUBLANES
    num_slots = kv_staging_slots(num_kv_stages, items_per_step)
    num_write_slots = cache_write_slots(num_kv_stages, items_per_step)
    num_items = schedule_ref[layout.num_items_offset]
    num_steps = pl.cdiv(num_items, items_per_step)

    def _item_field(name, item):
        return schedule_ref[layout.item_offset(name) + item]

    def _cache_tile(ref, hbm_row, token_start, num_tokens):
        if is_cache_head_major:
            return ref.at[
                hbm_row,
                slice(None),
                pl.ds(token_start, num_tokens),
                slice(None),
            ]
        else:
            return ref.at[
                hbm_row,
                pl.ds(token_start, num_tokens),
                slice(None),
                slice(None),
            ]

    def _buffer_tile(buffers, slot, token_start, num_tokens):
        if is_cache_head_major:
            return buffers.at[slot, :, pl.ds(token_start, num_tokens), :]
        elif is_bitcast_load:
            # Note (david): the staged rows past staged_kv_heads are the bank
            # pad, which no DMA fills.
            slot_tokens, word_stride = buffers.shape[1:3]
            return (
                buffers.at[slot]
                .reshape(slot_tokens, word_rows * word_stride, head_dim)
                .at[
                    pl.ds(token_start, num_tokens), pl.ds(0, staged_kv_heads), :
                ]
            )
        else:
            return buffers.at[slot, pl.ds(token_start, num_tokens), :, :]

    # Note (david): a merged row carries each head's V next to its K, so a
    # merged build stages, appends and writes back whole token rows through the
    # K buffers only, one contiguous DMA per page or appended tile.
    k_load = (k_cache_ref, k_buffers, SEM_K)
    k_write = (k_write_buffers, k_out_ref, SEM_K_WRITE)
    if is_merged_cache:
        load_parts = (k_load,)
        write_parts = (k_write,)
        value_buffers = k_buffers
    else:
        load_parts = (k_load, (v_cache_ref, v_buffers, SEM_V))
        write_parts = (k_write, (v_write_buffers, v_out_ref, SEM_V_WRITE))
        value_buffers = v_buffers

    def _item_load(item):
        return pl.multiple_of(_item_field("load", item), cache_tile_rows)

    def _start_item_kv(
        item, slot, token_base=0, branch_free=False, is_live=True
    ):
        """Stage item's tokens into slot at token_base, signalling the slot's
        semaphore. branch_free issues one copy per page of the block, empty
        pages at size 0, with no branch; is_live False makes every copy
        empty."""

        def _issue(hbm_row, hbm_start, num_tokens, buffer_start):
            for cache_ref, buffers, sem_index in load_parts:
                pltpu.make_async_copy(
                    _cache_tile(cache_ref, hbm_row, hbm_start, num_tokens),
                    _buffer_tile(
                        buffers, slot, token_base + buffer_start, num_tokens
                    ),
                    sems.at[sem_index, slot, 0],
                ).start()

        if branch_free:
            # Note (david): a packed step issues its next step's copies inside
            # its straight-line body, where their scalar address work rides
            # the vector bundles, as batched RPA's copy_in does. Guarded by
            # load_size and looped over the block's live pages, they ran as
            # their own basic blocks, bundles that held no vector op.
            if page_size is None:
                _issue(
                    cache_batch_idx_ref[_item_field("row", item)],
                    _item_field("block", item) * block_kv,
                    pl.multiple_of(
                        jnp.where(is_live, _item_field("load", item), 0),
                        cache_tile_rows,
                    ),
                    0,
                )
            else:
                table_base = _item_field("table_base", item)
                for page in range(layout.pages_per_item):
                    _issue(
                        block_table_ref[table_base + page],
                        0,
                        pl.multiple_of(
                            jnp.where(
                                is_live, _item_field(f"tokens_{page}", item), 0
                            ),
                            cache_tile_rows,
                        ),
                        page * page_size,
                    )
            return

        load_size = _item_load(item)

        @pl.when(load_size > 0)
        def _start():
            if page_size is None:
                _issue(
                    cache_batch_idx_ref[_item_field("row", item)],
                    _item_field("block", item) * block_kv,
                    load_size,
                    0,
                )
            else:
                # Note (david): a block copies only the pages that hold its
                # tokens, each a full page but the clipped last; a large block
                # at short context is mostly empty pages, and issuing them all
                # costs a copy each.
                row = _item_field("row", item)
                table_base = cache_batch_idx_ref[
                    row
                ] * pages_per_seq + _item_field("block", item) * (
                    block_kv // page_size
                )

                def _issue_page(page, carry):
                    buffer_start = pl.multiple_of(page * page_size, page_size)
                    _issue(
                        block_table_ref[table_base + page],
                        0,
                        pl.multiple_of(
                            jnp.minimum(load_size - buffer_start, page_size),
                            cache_tile_rows,
                        ),
                        buffer_start,
                    )
                    return carry

                lax.fori_loop(0, pl.cdiv(load_size, page_size), _issue_page, 0)

    def _wait_kv(load_size, slot):
        # Note (david): every segment of a block signals the same K (or V)
        # semaphore and a DMA semaphore counts bytes, so one wait sized to the
        # whole staged window covers them all (the RPA pattern); src == dst
        # because only the size matters.
        @pl.when(load_size > 0)
        def _wait():
            for _, buffers, sem_index in load_parts:
                tile = _buffer_tile(buffers, slot, 0, load_size)
                pltpu.make_async_copy(
                    tile, tile, sems.at[sem_index, slot, 0]
                ).wait()

    def _cache_write_copies(row, slot):
        """Copies of write tile slot to row's appended cache tile."""
        hbm_row, tile_start = _cache_write_target(row)
        return _write_tile_copies(hbm_row, tile_start, slot)

    def _cache_write_target(row):
        cache_row = cache_batch_idx_ref[row]
        cache_len = cache_seqlens_ref[row]
        if page_size is None:
            hbm_row = cache_row
            tile_start = pl.multiple_of(
                cache_len // cache_tile_rows * cache_tile_rows, cache_tile_rows
            )
        else:
            # Note (david): page_size is a multiple of 128, so a write tile at a
            # tile-aligned offset never straddles a page and the paged append
            # stays a single write.
            hbm_row = block_table_ref[
                cache_row * pages_per_seq + cache_len // page_size
            ]
            tile_start = pl.multiple_of(
                cache_len % page_size // cache_tile_rows * cache_tile_rows,
                cache_tile_rows,
            )
        return hbm_row, tile_start

    def _write_tile_copies(hbm_row, tile_start, slot):
        if is_cache_head_major:
            # Note (david): one copy per head, since the head axis of a cache
            # row is not contiguous.
            return [
                pltpu.make_async_copy(
                    buffers.at[slot, pl.ds(head_offset, 1), :, :],
                    out_ref_hbm.at[
                        hbm_row,
                        pl.ds(head_offset, 1),
                        pl.ds(tile_start, cache_tile_rows),
                        :,
                    ],
                    sems.at[sem_index, slot, head_offset],
                )
                for head_offset in range(num_kv_heads)
                for buffers, out_ref_hbm, sem_index in write_parts
            ]
        else:
            return [
                pltpu.make_async_copy(
                    buffers.at[slot].reshape(
                        cache_tile_rows, staged_kv_heads, head_dim
                    )
                    if is_bitcast_load
                    else buffers.at[slot],
                    out_ref_hbm.at[
                        hbm_row, pl.ds(tile_start, cache_tile_rows), :, :
                    ],
                    sems.at[sem_index, slot, 0],
                )
                for buffers, out_ref_hbm, sem_index in write_parts
            ]

    def _q_copy(batch_index, slot):
        return pltpu.make_async_copy(
            q_ref.at[batch_index], q_buffers.at[slot], sems.at[SEM_Q, slot, 0]
        )

    def _output_copy(batch_index, slot):
        return pltpu.make_async_copy(
            out_buffers.at[slot],
            out_ref.at[batch_index],
            sems.at[SEM_OUT, slot, 0],
        )

    # Note (david): staged V rows a block never loads keep whatever VMEM held;
    # start them finite so a zero probability times a stale row stays zero.
    if is_merged_cache and is_bitcast_load:
        # Note: through its (tokens * word_stride, head_dim) word view a slot
        # zeroes in whole 8-row stores; the 5-D view's one-word token rows took
        # one store per row. A head_dim wider than the lanes zeroes through
        # the 128-lane rows its K loads read; the head_dim-wide view stored
        # each token row with 2 of 8 sublanes.
        for zero_slot in range(num_slots):
            slot_words = staged_words(value_buffers, zero_slot)
            num_word_rows, word_lanes = slot_words.shape
            if word_lanes > NUM_LANES:
                zeroed_rows = slot_words.reshape(
                    num_word_rows * word_lanes // NUM_LANES, NUM_LANES
                )
            else:
                zeroed_rows = slot_words
            zeroed_rows[...] = jnp.zeros(zeroed_rows.shape, jnp.uint32)
    else:
        value_words = (
            value_buffers.bitcast(jnp.uint32)
            if is_bitcast_load
            else value_buffers
        )
        value_words[...] = jnp.zeros(value_words.shape, value_words.dtype)

    # Note (deepolar): every sequence folds the same sinks, so they load once.
    sinks = sinks_ref[...] if has_sinks else None

    def _new_token_tile(slot, token_base, local_tile_start, new_row):
        """(tile_index, is_new_row, expand) of the staged cache tile at
        local_tile_start of the block staged in slot at token_base, whose row
        new_row takes the new token."""
        token_row = jnp.arange(cache_tile_rows, dtype=jnp.int32)
        tile_tokens = pl.ds(
            pl.multiple_of(token_base + local_tile_start, cache_tile_rows),
            cache_tile_rows,
        )
        if is_cache_head_major:
            tile_index = (slot, slice(None), tile_tokens, slice(None))
            token_rows = token_row[None, :, None]

            def expand(token):
                return token[:, None, :]

        elif is_bitcast_load:
            tile_index = (
                slot,
                tile_tokens,
                pl.ds(0, staged_kv_heads // word_rows),
                slice(None),
                slice(None),
            )
            token_rows = token_row[:, None, None, None]

            def expand(token):
                return token.reshape(
                    staged_kv_heads // word_rows, word_rows, head_dim
                )[None]

        else:
            tile_index = (slot, tile_tokens, slice(None), slice(None))
            token_rows = token_row[:, None, None]

            def expand(token):
                return token[None]

        return tile_index, token_rows == new_row, expand

    def _new_tokens(row):
        if is_merged_cache:
            return (new_kv_ref[row, :, :],)
        return (k_ref[row, :, :], v_ref[row, :, :])

    def _start_cache_write(row, updated_tiles, write_slot, waits_previous):
        """Copy row's updated tiles into write tile write_slot and start their
        write-back, once the write-back that last used the slot is done."""

        # Note (david): a wait reads only its size and semaphore, so the
        # clamped row stands in for the row that last wrote the slot, and where
        # a predicated wait does not run.
        @pl.when(waits_previous)
        def _wait_previous_cache_write():
            for copy in _cache_write_copies(
                jnp.maximum(row - num_write_slots, 0), write_slot
            ):
                copy.wait()

        for (write_buffers, _, _), updated_tile in zip(
            write_parts, updated_tiles
        ):
            write_buffers[write_slot] = updated_tile
        for copy in _cache_write_copies(row, write_slot):
            copy.start()

    def _append_new_token(slot, token_base, row, block_index):
        """Write row's new token into its block, staged in slot at token_base,
        and start that tile's write-back."""
        cache_len = cache_seqlens_ref[row]
        # Note (david): the append reads, updates and writes back the whole
        # cache tile holding the new token, cache_tile_rows tokens at a
        # tile-aligned offset, which the block load has staged in full.
        local_index = cache_len - block_index * block_kv
        local_tile_start = local_index // cache_tile_rows * cache_tile_rows
        tile_index, is_new_row, expand = _new_token_tile(
            slot, token_base, local_tile_start, local_index - local_tile_start
        )
        updated_tiles = []
        for (_, buffers, _), new_token in zip(load_parts, _new_tokens(row)):
            updated_tile = jnp.where(
                is_new_row, expand(new_token), buffers[tile_index]
            )
            buffers[tile_index] = updated_tile
            updated_tiles.append(updated_tile)
        _start_cache_write(
            row,
            updated_tiles,
            lax.rem(row, num_write_slots),
            row >= num_write_slots,
        )

    def _key_masks(key_start, total_len, first_key):
        """(is_valid, fragment_has_valid_key) of the fragment at key_start."""
        # Note (david): build the mask at full fragment shape; a (1, N) mask
        # needs a sublane-broadcast relayout inside every fragment's select,
        # which dominated MQA decode.
        key_position = (
            key_start
            + lax.broadcasted_iota(
                jnp.int32, (rows_per_fragment, fragment_keys), 1
            )
            // kv_heads_per_fragment
        )
        if row_key_offset is None:
            key_end = total_len
        else:
            key_end = total_len + row_key_offset
        if left is None:
            is_in_window = key_position < key_end
        else:
            is_in_window = (key_position < key_end) & (
                key_position >= first_key
            )
        if parity_mask is None:
            is_valid = is_in_window
        else:
            is_valid = is_in_window & parity_mask
        fragment_has_valid_key = jnp.logical_and(
            key_start < total_len,
            key_start + block_kv_compute > first_key,
        )
        return is_valid, fragment_has_valid_key

    def _score_and_softmax(
        queries,
        slot,
        head_group,
        kv_compute_index,
        is_valid,
        fragment_has_valid_key,
        is_first,
        row_max_prev,
        row_sum_prev,
    ):
        scores, values = _scores(
            queries, slot, head_group, kv_compute_index, is_valid
        )
        probabilities, rescale, row_max, row_sum = _softmax(
            scores, fragment_has_valid_key, is_first, row_max_prev, row_sum_prev
        )
        return probabilities, rescale, values, row_max, row_sum

    def _scores(queries, slot, head_group, kv_compute_index, is_valid):
        """(masked scores, merged values or None) of one fragment."""
        # Note (david): the query heads of one head group are consecutive
        # rows of q, so one (rows_per_fragment, head_dim) tile rides a
        # single K load and MXU pass.
        group_start = head_group * rows_per_fragment
        query = queries[group_start : group_start + rows_per_fragment, :]
        if is_merged_cache:
            # Note (david): a head's K and V share their u32 words, so the
            # score's word load also yields the values, held until this
            # fragment's pv. Loading the words again for pv made MHA decode
            # slower than the blocked layout's, since its 32-row word stride
            # takes the costly strided load; one load is at parity.
            scores = None
            values = []
            for group_member in range(kv_heads_per_group):
                key, value = kv_pair_fragment(
                    k_buffers,
                    slot,
                    (head_group * kv_heads_per_group + group_member)
                    * kv_heads_per_fragment,
                    kv_compute_index,
                )
                values.append(value)
                # Note (david): K and V stay in the cache dtype; the MXU
                # widens an fp8 latched operand to bf16 itself.
                member_scores = lax.dot_general(
                    query,
                    key,
                    NT_DIM_NUMBERS,
                    preferred_element_type=jnp.float32,
                )
                if scores is None:
                    scores = member_scores
                else:
                    scores = jnp.where(
                        score_row_head == group_member,
                        member_scores,
                        scores,
                    )
        else:
            key = kv_fragment(k_buffers, slot, head_group, kv_compute_index)
            values = None
            scores = jnp.dot(query, key.T, preferred_element_type=jnp.float32)
        # Note (david): Q carries softmax_scale / softcap, so log2(e) rides
        # in the tanh output scale. The cap runs before the mask, since tanh
        # would lift a masked -inf to -softcap.
        if log2_softcap:
            capped_scores = jnp.tanh(scores) * log2_softcap
        else:
            capped_scores = scores
        return jnp.where(is_valid, capped_scores, -jnp.inf), values

    def _softmax(
        scores, fragment_has_valid_key, is_first, row_max_prev, row_sum_prev
    ):
        """(probabilities, rescale, row_max, row_sum) of one fragment's online
        softmax step from masked scores."""
        if left is None:
            probabilities, row_max, row_sum, rescale = static_anchor_update(
                scores,
                row_max_prev,
                row_sum_prev,
                is_first=is_first,
                guard_threshold=guard_threshold,
            )
        else:
            # Note (david): a sliding window can leave a fragment with no
            # visible key, whose all -inf scores would turn the update into
            # NaN; score it as zeros and keep the old state. The state stays
            # empty until the first visible key, hence is_first =
            # row_sum_prev == 0.
            safe_scores = jnp.where(
                fragment_has_valid_key, scores, jnp.zeros_like(scores)
            )
            (
                candidate_probabilities,
                candidate_row_max,
                candidate_row_sum,
                candidate_rescale,
            ) = static_anchor_update(
                safe_scores,
                row_max_prev,
                row_sum_prev,
                is_first=row_sum_prev == 0,
                guard_threshold=guard_threshold,
            )
            probabilities = jnp.where(
                fragment_has_valid_key,
                candidate_probabilities,
                jnp.zeros_like(candidate_probabilities),
            )
            row_max = jnp.where(
                fragment_has_valid_key, candidate_row_max, row_max_prev
            )
            row_sum = jnp.where(
                fragment_has_valid_key, candidate_row_sum, row_sum_prev
            )
            rescale = jnp.where(
                fragment_has_valid_key,
                candidate_rescale,
                jnp.ones_like(candidate_rescale),
            )
        return probabilities, rescale, row_max, row_sum

    def _weighted_values(
        slot, head_group, kv_compute_index, probabilities, merged_values
    ):
        """probabilities @ V of one fragment, f32."""
        probabilities = probabilities.astype(jnp.bfloat16)
        if is_merged_cache:
            weighted_values = None
            for group_member, value in enumerate(merged_values):
                member_values = lax.dot_general(
                    probabilities,
                    value,
                    NN_DIM_NUMBERS,
                    preferred_element_type=jnp.float32,
                )
                if weighted_values is None:
                    weighted_values = member_values
                else:
                    weighted_values = jnp.where(
                        pv_row_head == group_member,
                        member_values,
                        weighted_values,
                    )
        else:
            value = kv_fragment(
                value_buffers, slot, head_group, kv_compute_index
            )
            weighted_values = jnp.dot(
                probabilities, value, preferred_element_type=jnp.float32
            )
        return weighted_values

    def _pv(
        slot,
        head_group,
        kv_compute_index,
        probabilities,
        rescale,
        merged_values,
        parity,
    ):
        group_slice = pl.ds(head_group * rows_per_fragment, rows_per_fragment)
        weighted_values = _weighted_values(
            slot, head_group, kv_compute_index, probabilities, merged_values
        )
        # Note (david): the accumulator is stored in q's dtype while rescale
        # and weighted_values are f32; the explicit upcast keeps the kernel
        # tracing under jax_numpy_dtype_promotion=strict.
        accumulator_ref[parity, group_slice, :] = (
            accumulator_ref[parity, group_slice, :].astype(jnp.float32)
            * lane_broadcast(rescale, head_dim)
            + weighted_values
        ).astype(q_ref.dtype)

    def _row_output(row_max, row_sum, accumulator, row_sinks):
        """(out in q's dtype, lse or None) of finished row state; row_sinks
        are the sinks of the state's rows."""
        # Note (david): a quantized cache's V descale is one per-tensor factor,
        # so it rides in the softmax normalization. It arrives in SMEM, so a
        # new descale value compiles no new kernel.
        if row_sinks is None:
            inverse_row_sum = jnp.where(
                row_sum > 0,
                pl.reciprocal(row_sum, approx=True) * v_scale_ref[0],
                0.0,
            )
        else:
            # Note (deepolar): fold_sink is the prefill and extend kernels'
            # fold; a row with nothing to attend returns out = 0 and lse = sink.
            row_max, row_sum, out_scale = fold_sink(row_max, row_sum, row_sinks)
            inverse_row_sum = out_scale * v_scale_ref[0]
        out = (
            accumulator * lane_broadcast(inverse_row_sum, accumulator.shape[-1])
        ).astype(q_ref.dtype)
        if not return_lse:
            return out, None
        lse_log2 = row_max + jnp.log2(row_sum)
        lse = jnp.where(row_sum > 0, lse_log2 * math.log(2.0), -jnp.inf).astype(
            jnp.float32
        )
        return out, lse

    def _finish_row(row, parity):
        output_slot = lax.rem(row, NUM_IO_STAGES)

        @pl.when(row >= NUM_IO_STAGES)
        def _wait_previous_output():
            _output_copy(row - NUM_IO_STAGES, output_slot).wait()

        out, lse = _row_output(
            row_max_ref[parity],
            row_sum_ref[parity],
            accumulator_ref[parity].astype(jnp.float32),
            sinks,
        )
        out_buffers[output_slot] = out
        if return_lse:
            lse_ref[pl.ds(row, 1), :, :] = lse[None, :, :]
        _output_copy(row, output_slot).start()

    # Note (david): the mask-free and live-tile copies leave their last
    # fragment's PV pending, and the next copy issues it after its own first
    # score and softmax, as the varlen kernels do across blocks and rows: a
    # 4:1-head row's score, softmax and PV are one dependent chain, and the
    # pending PV gives the MXU independent work while the next chain waits on
    # its matmul. The PV lands in its row's parity, and the row's output
    # finish follows it when it was the row's last.
    last_head_group = num_head_groups - 1

    def _take_pending():
        slot = pending_ref[PENDING_SLOT]
        kv_compute_index = pending_ref[PENDING_KV_COMPUTE]
        parity = pending_ref[PENDING_PARITY]
        finish_row = pending_ref[PENDING_FINISH_ROW]
        if is_merged_cache:
            values = [
                kv_pair_fragment(
                    k_buffers,
                    slot,
                    (last_head_group * kv_heads_per_group + group_member)
                    * kv_heads_per_fragment,
                    kv_compute_index,
                )[1]
                for group_member in range(kv_heads_per_group)
            ]
        else:
            values = None
        _pv(
            slot,
            last_head_group,
            kv_compute_index,
            pending_probabilities_ref[...],
            pending_rescale_ref[...],
            values,
            parity,
        )

        @pl.when(finish_row >= 0)
        def _finish_pending_row():
            _finish_row(finish_row, parity)

    def _carry_out(
        slot, kv_compute_index, probabilities, rescale, parity, finish_row
    ):
        pending_probabilities_ref[...] = probabilities.astype(
            pending_probabilities_ref.dtype
        )
        pending_rescale_ref[...] = rescale
        pending_ref[PENDING_SLOT] = slot
        pending_ref[PENDING_KV_COMPUTE] = kv_compute_index
        pending_ref[PENDING_PARITY] = parity
        pending_ref[PENDING_FINISH_ROW] = finish_row

    def _attend_item(item, slot, row, block_index, flags, parity):
        is_row_first = (flags & ITEM_ROW_FIRST) != 0
        finish_row = jnp.where((flags & ITEM_ROW_LAST) != 0, row, -1)
        cache_len = cache_seqlens_ref[row]
        total_len = cache_len + int(has_new)
        if left is None:
            first_key = jnp.int32(0)
        else:
            first_key = jnp.maximum(0, total_len - 1 - left)
        queries = q_buffers[lax.rem(row, NUM_IO_STAGES)]
        _wait_kv(_item_load(item), slot)

        if has_new:

            @pl.when(block_index == cache_len // block_kv)
            def _append():
                _append_new_token(slot, 0, row, block_index)

        block_start = block_index * block_kv
        # Note (david): the running state is read as whole (rows, 128) tiles
        # with every lane holding its row's value, so every broadcast
        # against the scores or the accumulator is a vreg repeat, not a
        # cross-lane op.
        group_slices = [
            pl.ds(head_group * rows_per_fragment, rows_per_fragment)
            for head_group in range(num_head_groups)
        ]

        def _fragment_masks(kv_compute_index):
            return _key_masks(
                block_start + kv_compute_index * block_kv_compute,
                total_len,
                first_key,
            )

        def _attend_every_tile():
            fragment_masks = [
                _fragment_masks(kv_compute_index)
                for kv_compute_index in range(num_kv_compute_fragments)
            ]
            # Note (david): software pipeline; each fragment's scores are
            # issued one step ahead of its own pv, so the MXU has the next
            # matmul queued while pv runs.
            pending_fragment = None
            for fragment_index in range(
                num_head_groups * num_kv_compute_fragments
            ):
                head_group, kv_compute_index = divmod(
                    fragment_index, num_kv_compute_fragments
                )
                group_slice = group_slices[head_group]
                if kv_compute_index == 0:
                    row_max = row_max_ref[parity, group_slice, :]
                    row_sum = row_sum_ref[parity, group_slice, :]
                probabilities, rescale, values, row_max, row_sum = (
                    _score_and_softmax(
                        queries,
                        slot,
                        head_group,
                        kv_compute_index,
                        *fragment_masks[kv_compute_index],
                        is_row_first if kv_compute_index == 0 else False,
                        row_max,
                        row_sum,
                    )
                )
                if fragment_index == 0:
                    _take_pending()
                if kv_compute_index == num_kv_compute_fragments - 1:
                    row_max_ref[parity, group_slice, :] = row_max
                    row_sum_ref[parity, group_slice, :] = row_sum
                if pending_fragment is not None:
                    _pv(slot, *pending_fragment, parity)
                pending_fragment = (
                    head_group,
                    kv_compute_index,
                    probabilities,
                    rescale,
                    values,
                )
            _, last_kv_compute_index, probabilities, rescale, _ = (
                pending_fragment
            )
            _carry_out(
                slot,
                last_kv_compute_index,
                probabilities,
                rescale,
                parity,
                finish_row,
            )

        def _attend_live_tiles():
            first_fragment = (
                jnp.maximum(first_key - block_start, 0) // block_kv_compute
            )
            end_fragment = jnp.minimum(
                pl.cdiv(total_len - block_start, block_kv_compute),
                num_kv_compute_fragments,
            )

            def _attend_tile(kv_compute_index, row_states):
                is_valid, fragment_has_valid_key = _fragment_masks(
                    kv_compute_index
                )
                # Note (david): without a window the row's first tile is
                # tile 0 of its first block; a window's state starts at its
                # first visible key instead (row_sum_prev == 0 in
                # _score_and_softmax).
                is_first = jnp.logical_and(is_row_first, kv_compute_index == 0)
                # Note (david): each head group's scores are issued one step
                # ahead of the previous group's pv, and the tile's last pv is
                # left pending for the next tile, block or row.
                next_row_states = []
                pending_fragment = None
                for head_group, (row_max_prev, row_sum_prev) in enumerate(
                    row_states
                ):
                    probabilities, rescale, values, row_max, row_sum = (
                        _score_and_softmax(
                            queries,
                            slot,
                            head_group,
                            kv_compute_index,
                            is_valid,
                            fragment_has_valid_key,
                            is_first,
                            row_max_prev,
                            row_sum_prev,
                        )
                    )
                    if head_group == 0:
                        _take_pending()
                    if pending_fragment is not None:
                        _pv(slot, *pending_fragment, parity)
                    pending_fragment = (
                        head_group,
                        kv_compute_index,
                        probabilities,
                        rescale,
                        values,
                    )
                    next_row_states.append((row_max, row_sum))
                _, _, probabilities, rescale, _ = pending_fragment
                _carry_out(
                    slot,
                    kv_compute_index,
                    probabilities,
                    rescale,
                    parity,
                    jnp.where(
                        kv_compute_index == end_fragment - 1, finish_row, -1
                    ),
                )
                return tuple(next_row_states)

            row_states = lax.fori_loop(
                first_fragment,
                end_fragment,
                _attend_tile,
                tuple(
                    (
                        row_max_ref[parity, group_slice, :],
                        row_sum_ref[parity, group_slice, :],
                    )
                    for group_slice in group_slices
                ),
            )
            for group_slice, (row_max, row_sum) in zip(
                group_slices, row_states
            ):
                row_max_ref[parity, group_slice, :] = row_max
                row_sum_ref[parity, group_slice, :] = row_sum

        if num_kv_compute_fragments == 1:
            _attend_every_tile()
        else:
            # Note (david): a block whose every compute tile holds keys runs
            # one pipeline over all its fragments, as before; only a partial
            # block (a row's last, or a window's first) loops over the tiles
            # holding its keys, as RPA v3 runs cdiv(kv_len, bkv_csz) tiles.
            # Running every tile of a partial block made a short row cost the
            # whole block, as with the 8192-token blocks of #14's v7x
            # entries. The loop drains the pipeline at every tile, so full
            # blocks keep the single pipeline; looping over them was slower.
            is_block_full = jnp.logical_and(
                block_start + block_kv <= total_len,
                first_key <= block_start,
            )
            pl.when(is_block_full)(_attend_every_tile)
            pl.when(jnp.logical_not(is_block_full))(_attend_live_tiles)

    def _process_item(item, carry):
        slot = lax.rem(item, num_slots)
        row = _item_field("row", item)
        block_index = _item_field("block", item)
        flags = _item_field("flags", item)
        parity = lax.rem(row, NUM_ROW_PARITIES)

        @pl.when(item + 1 < num_items)
        def _prefetch_next_item():
            _start_item_kv(item + 1, lax.rem(item + 1, num_slots))

        @pl.when((flags & ITEM_ROW_FIRST) != 0)
        def _start_row():
            row_max_ref[parity] = jnp.full(
                row_max_ref.shape[1:], -jnp.inf, jnp.float32
            )
            row_sum_ref[parity] = jnp.zeros(row_sum_ref.shape[1:], jnp.float32)
            accumulator_ref[parity] = jnp.zeros(
                accumulator_ref.shape[1:], accumulator_ref.dtype
            )
            _q_copy(row, lax.rem(row, NUM_IO_STAGES)).wait()

            @pl.when(row + 1 < num_active)
            def _prefetch_next_q():
                _q_copy(row + 1, lax.rem(row + 1, NUM_IO_STAGES)).start()

        is_empty_row = (flags & ITEM_EMPTY_ROW) != 0

        # Note (david): a row with no key in its window still writes its
        # output (zeros, or its sinks' fold), after the previous row's.
        @pl.when(is_empty_row)
        def _finish_empty_row():
            _take_pending()
            pending_ref[PENDING_PARITY] = TRASH_PARITY
            pending_ref[PENDING_FINISH_ROW] = -1
            _finish_row(row, parity)

        @pl.when(jnp.logical_not(is_empty_row))
        def _attend():
            _attend_item(item, slot, row, block_index, flags, parity)

        return carry

    group_slices = [
        pl.ds(head_group * rows_per_fragment, rows_per_fragment)
        for head_group in range(num_head_groups)
    ]

    def _start_step_kv(step, slot, is_live=True):
        """Stage step's items into slot; a step that is not live (past the
        last) stages nothing."""
        for position in range(items_per_step):
            _start_item_kv(
                jnp.minimum(
                    step * items_per_step + position, layout.max_items - 1
                ),
                slot,
                position * block_kv,
                branch_free=True,
                is_live=is_live,
            )

    def _packed_item(position, row, block_index, flags):
        """Scalars of the step's item at position."""
        total_len = cache_seqlens_ref[row] + int(has_new)
        if left is None:
            first_key = jnp.int32(0)
        else:
            first_key = jnp.maximum(0, total_len - 1 - left)
        return dict(
            position=position,
            row=row,
            block_index=block_index,
            block_start=block_index * block_kv,
            is_row_first=(flags & ITEM_ROW_FIRST) != 0,
            total_len=total_len,
            first_key=first_key,
            # Note (david): only an empty row (no cached key, no new token)
            # sees no key; its state is whatever its masked scores made of it,
            # so its output takes the empty state, out = 0 and lse = -inf (or
            # the sink).
            has_keys=total_len > 0,
        )

    def _staged_value_index(base, position, head_group, fragment, member):
        return (
            base
            + (
                (position * num_kv_compute_fragments + fragment)
                * num_head_groups
                + head_group
            )
            * kv_heads_per_group
            + member
        )

    def _fragments():
        return [
            (head_group, fragment)
            for head_group in range(num_head_groups)
            for fragment in range(num_kv_compute_fragments)
        ]

    def _attend_packed_step(slot, items, states, value_base):
        """Run the step's items through the prefill kernels' fragment pipeline
        (fwd_pipeline) in two halves: a half scores its items and continues
        the online softmax through them, then issues the PV the previous half
        left pending; the last half's PV runs at the step's end. states are
        per head group (row_max, row_sum, accumulator).

        Returns the new states and, per item and head group, the (out, lse)
        its row would finish with after that item.
        """
        # Note (david): value_base is 0, but a runtime 0: at static indices
        # the V tiles' VMEM stores and loads were promoted back to values,
        # and the scheduler again reloaded the staged words.
        # Note (david): a PV matmul waits for its P, ~214 cycles after the QK
        # matmul plus the softmax, and the in-order MXU pushes at most one V
        # tile ahead of it, so only the K pushes ahead of it in program order
        # fill that wait. Two halves put the second half's K ahead of the
        # first half's PV and every K push ahead of every V push. Groups of 1
        # or 2 items left the MXU waiting at each PV while the K unpack ran
        # ahead into spills.
        half_size = -(-len(items) // 2)
        running = [(row_max, row_sum) for row_max, row_sum, _ in states]
        accumulators = [accumulator for _, _, accumulator in states]
        item_outputs = [None] * len(items)

        def _half_pv(half_items, softmax, row_states):
            for item in half_items:
                outputs = []
                for head_group in range(num_head_groups):
                    accumulator = jnp.where(
                        item["is_row_first"], 0.0, accumulators[head_group]
                    )
                    for fragment in range(num_kv_compute_fragments):
                        key = (item["position"], head_group, fragment)
                        probabilities, rescale = softmax[key]
                        kv_compute_index = (
                            item["position"] * num_kv_compute_fragments
                            + fragment
                        )
                        values = [
                            staged_values_ref[
                                _staged_value_index(value_base, *key, member)
                            ]
                            for member in range(kv_heads_per_group)
                        ]
                        accumulator = accumulator * lane_broadcast(
                            rescale, head_dim
                        ) + _weighted_values(
                            slot,
                            head_group,
                            kv_compute_index,
                            probabilities,
                            values,
                        )
                    accumulators[head_group] = accumulator
                    row_max, row_sum = row_states[item["position"], head_group]
                    group_start = head_group * rows_per_fragment
                    outputs.append(
                        _row_output(
                            jnp.where(item["has_keys"], row_max, -jnp.inf),
                            jnp.where(item["has_keys"], row_sum, 0.0),
                            jnp.where(item["has_keys"], accumulator, 0.0),
                            None
                            if sinks is None
                            else sinks[
                                group_start : group_start + rows_per_fragment
                            ],
                        )
                    )
                item_outputs[item["position"]] = outputs

        pending_pv = None
        for half_start in range(0, len(items), half_size):
            half_items = items[half_start : half_start + half_size]
            scores = {}
            for item in half_items:
                queries = q_ref[item["row"]]
                for head_group, fragment in _fragments():
                    kv_compute_index = (
                        item["position"] * num_kv_compute_fragments + fragment
                    )
                    is_valid, has_valid_key = _key_masks(
                        item["block_start"] + fragment * block_kv_compute,
                        item["total_len"],
                        item["first_key"],
                    )
                    fragment_scores, values = _scores(
                        queries, slot, head_group, kv_compute_index, is_valid
                    )
                    # Note (david): the V words the QK loaded are unpacked
                    # there and kept in VMEM for the PV. Left as values, the
                    # scheduler loaded every staged word twice, for the QK and
                    # for the PV.
                    for member, value in enumerate(values):
                        staged_values_ref[
                            _staged_value_index(
                                value_base,
                                item["position"],
                                head_group,
                                fragment,
                                member,
                            )
                        ] = value
                    scores[item["position"], head_group, fragment] = (
                        fragment_scores,
                        has_valid_key,
                    )
            softmax = {}
            row_states = {}
            for item in half_items:
                for head_group in range(num_head_groups):
                    row_max, row_sum = running[head_group]
                    if left is not None:
                        row_sum = jnp.where(item["is_row_first"], 0.0, row_sum)
                    for fragment in range(num_kv_compute_fragments):
                        fragment_scores, has_valid_key = scores[
                            item["position"], head_group, fragment
                        ]
                        probabilities, rescale, row_max, row_sum = _softmax(
                            fragment_scores,
                            has_valid_key,
                            item["is_row_first"] if fragment == 0 else False,
                            row_max,
                            row_sum,
                        )
                        softmax[item["position"], head_group, fragment] = (
                            probabilities,
                            rescale,
                        )
                    row_state = (row_max, row_sum)
                    running[head_group] = row_state
                    row_states[item["position"], head_group] = row_state
            if pending_pv is not None:
                _half_pv(*pending_pv)
            pending_pv = (half_items, softmax, row_states)
        _half_pv(*pending_pv)
        new_states = [
            (*running[head_group], accumulators[head_group])
            for head_group in range(num_head_groups)
        ]
        return new_states, item_outputs

    # Note (david): a step issues the copies num_kv_stages steps ahead after
    # its compute, into the slot it just read, so they overlap a whole step of
    # compute even at two stages. Issued before the compute, num_kv_stages - 1
    # steps ahead into the slot the step before freed, they still sank to the
    # step's end, where the next step waited on them at once, and two stages
    # needed a third to hide them.
    prefetch_distance = num_kv_stages

    def _process_step(step, carry):
        """items_per_step consecutive items in one straight-line body after
        the step's wait, then the copies prefetch_distance steps ahead, into
        the slot the items read: the scheduler hides their issue behind the
        last PV's result latency."""
        slot = lax.rem(step, num_kv_stages)

        # Note (david): the wait comes first, so the items' scalar header below
        # rides the vector work after it; behind the header every step held
        # scalar-only bundles. The K loads follow
        # the wait, which no vector op crosses (the region, if-converted, is no
        # block boundary).
        @pl.when(step >= 0)
        def _stage_fence():
            _wait_kv(
                pl.multiple_of(
                    schedule_ref[layout.step_load_offset + step],
                    cache_tile_rows,
                ),
                slot,
            )

        items = []
        for position in range(items_per_step):
            item = step * items_per_step + position
            items.append(
                _packed_item(
                    position,
                    _item_field("row", item),
                    _item_field("block", item),
                    _item_field("flags", item),
                )
            )

        states = [
            (
                row_max_ref[0, group_slice, :],
                row_sum_ref[0, group_slice, :],
                accumulator_ref[0, group_slice, :],
            )
            for group_slice in group_slices
        ]
        states, item_outputs = _attend_packed_step(
            slot, items, states, jnp.minimum(step, 0)
        )
        # Note (david): every output store follows all of the step's compute.
        # The rows are dynamic, so the scheduler cannot tell an out store from
        # a later item's q and K loads. In item order, a row's last item
        # writes last.
        for item, outputs in zip(items, item_outputs):
            for group_slice, (out, lse) in zip(group_slices, outputs):
                out_ref[item["row"], group_slice, :] = out
                if return_lse:
                    lse_ref[pl.ds(item["row"], 1), group_slice, :] = lse[None]
        for group_slice, (row_max, row_sum, accumulator) in zip(
            group_slices, states
        ):
            row_max_ref[0, group_slice, :] = row_max
            row_sum_ref[0, group_slice, :] = row_sum
            accumulator_ref[0, group_slice, :] = accumulator
        # Note (david): the copies overwrite the slot the items read, and the
        # staged loads keep their hazards on it, so each copy follows the
        # slot's last load. The last steps' prefetches issue empty copies, with
        # no branch.
        _start_step_kv(
            step + prefetch_distance,
            slot,
            is_live=step + prefetch_distance < num_steps,
        )
        return carry

    if is_packed:
        if has_new:
            # Note (david): every staged block then already holds its row's
            # new token, so a step stitches no append into its staging.
            # Stitched per step, the items' tile read-modify-writes and
            # write-backs were a basic block of their own in every step, and
            # most of the step's spills. A row past num_active writes 0
            # tokens, with no branch.
            num_writes = jnp.minimum(num_active, batch)
            for row in range(batch):
                num_tokens = jnp.where(row < num_writes, 1, 0)
                pltpu.make_async_copy(
                    new_kv_ref.at[pl.ds(row, num_tokens)],
                    k_out_ref.at[
                        schedule_ref[layout.row_write_offset + row],
                        pl.ds(
                            schedule_ref[layout.row_write_offset + batch + row],
                            num_tokens,
                        ),
                    ],
                    sems.at[SEM_K_WRITE, 0, 0],
                ).start()

            # Note (david): a DMA semaphore counts bytes, so one wait sized to
            # every row's token covers all the writes (src == dst: only the
            # size matters).
            @pl.when(num_writes > 0)
            def _wait_new_tokens():
                tokens = new_kv_ref.at[pl.ds(0, num_writes)]
                pltpu.make_async_copy(
                    tokens, tokens, sems.at[SEM_K_WRITE, 0, 0]
                ).wait()

        for first_step in range(prefetch_distance):
            _start_step_kv(
                first_step, first_step, is_live=first_step < num_steps
            )
        lax.fori_loop(0, num_steps, _process_step, 0)
    else:
        # Note (david): the first item takes a pending PV aimed at the trash
        # accumulator, so no copy branches on whether one is pending.
        pending_ref[PENDING_SLOT] = 0
        pending_ref[PENDING_KV_COMPUTE] = 0
        pending_ref[PENDING_PARITY] = TRASH_PARITY
        pending_ref[PENDING_FINISH_ROW] = -1
        pending_probabilities_ref[...] = jnp.zeros(
            pending_probabilities_ref.shape, pending_probabilities_ref.dtype
        )
        pending_rescale_ref[...] = jnp.zeros(
            pending_rescale_ref.shape, jnp.float32
        )

        @pl.when(num_active > 0)
        def _prefetch_first_q():
            _q_copy(0, 0).start()

        @pl.when(num_items > 0)
        def _prefetch_first_item():
            _start_item_kv(0, 0)

        lax.fori_loop(0, num_items, _process_item, 0)
        # Note (david): the last block's PV and its row's finish run here.
        _take_pending()

    # Note (david): the kernel must not exit with DMAs in flight, and the
    # one-item loop leaves its last NUM_IO_STAGES output copies and last
    # num_write_slots cache writes running (a packed call's output stays in
    # VMEM and its new tokens land before its loop). The candidate rows are the
    # static tail of a full batch, each guarded against running past the
    # active prefix.
    def _drain(tail_length, wait_row):
        for tail_offset in range(min(batch, tail_length)):
            batch_index = num_active - 1 - tail_offset

            # Note (david): pl.when traces the body here, inside the iteration
            # that set batch_index.
            @pl.when(batch_index >= 0)
            def _wait_tail_row():
                wait_row(batch_index)  # pylint: disable=cell-var-from-loop

    def _wait_output(batch_index):
        _output_copy(batch_index, lax.rem(batch_index, NUM_IO_STAGES)).wait()

    if not is_packed:
        _drain(NUM_IO_STAGES, _wait_output)

    if has_new and not is_packed:

        def _wait_cache_write(batch_index):
            write_slot = lax.rem(batch_index, num_write_slots)
            for copy in _cache_write_copies(batch_index, write_slot):
                copy.wait()

        _drain(num_write_slots, _wait_cache_write)


def resolve_block_kv_compute(block_kv: int) -> int:
    return next(
        candidate
        for candidate in BLOCK_KV_COMPUTE_CANDIDATES
        if block_kv % candidate == 0
    )


def kvcache_config_vmem_bytes(
    block_kv: int,
    block_kv_compute: int,
    num_kv_stages: int,
    items_per_step: int,
    *,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    kv_itemsize: int,
    batch: int,
) -> int:
    """VMEM bytes of the kernel's scratch at a config."""
    # Note (david): this must mirror the scratch shapes below; the extra
    # BF16_TILE_ROWS rows per stage bound the cache write tiles, and the two
    # extra heads per token bound a staged block's bank pad word row. The
    # one-item staging ring holds one block more than num_kv_stages, the
    # pending PV's, whose probabilities are at most one 8-row query tile of a
    # head pair's keys. A packed slot holds a step's blocks, and the packed
    # kernel keeps the whole q and out in VMEM.
    if items_per_step == 1:
        staged_tokens = (num_kv_stages + 1) * (block_kv + BF16_TILE_ROWS)
        q_o_rows = 2
        accumulator_itemsize = BF16_BYTES
    else:
        staged_tokens = (
            kv_staging_slots(num_kv_stages, items_per_step)
            * items_per_step
            * block_kv
            + (cache_write_slots(num_kv_stages, items_per_step) + 1)
            * BF16_TILE_ROWS
        )
        q_o_rows = batch
        accumulator_itemsize = F32_BYTES
        # Note (david): the step's unpacked V, every KV head of every item.
        staged_tokens += items_per_step * block_kv // 2
    kv_staging_bytes = (
        2 * staged_tokens * (num_kv_heads + 2) * head_dim * kv_itemsize
    )
    q_o_buffer_bytes = 2 * q_o_rows * num_query_heads * head_dim * BF16_BYTES
    row_state_bytes = (
        NUM_ROW_PARITIES * 2 * num_query_heads * NUM_LANES * F32_BYTES
    )
    accumulator_bytes = (
        (NUM_ROW_PARITIES + 1)
        * num_query_heads
        * head_dim
        * accumulator_itemsize
    )
    pending_bytes = (
        NUM_SUBLANES * 2 * block_kv_compute * BF16_BYTES
        + NUM_SUBLANES * NUM_LANES * F32_BYTES
    )
    return (
        kv_staging_bytes
        + q_o_buffer_bytes
        + row_state_bytes
        + accumulator_bytes
        + pending_bytes
    )


def validate_kvcache_config(
    config: KVCacheConfig,
    *,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    capacity: int,
    kv_itemsize: int,
    batch: int,
    page_size: int | None,
) -> tuple[int, int, int, int]:
    """Validate a (block_kv, block_kv_compute, num_kv_stages[,
    items_per_step]) config; returns it with items_per_step."""
    if len(config) == 3:
        block_kv, block_kv_compute, num_kv_stages = config
        items_per_step = 1
    elif len(config) == 4:
        block_kv, block_kv_compute, num_kv_stages, items_per_step = config
    else:
        raise ValueError(
            "a KV-cache config is (block_kv, block_kv_compute, num_kv_stages"
            f"[, items_per_step]); got {config}."
        )
    if items_per_step < 1:
        raise ValueError(f"items_per_step must be >= 1; got {items_per_step}.")
    # Note (david): a paged table is padded to whole blocks, so a paged block
    # need only be whole pages; a cache without pages is split into blocks, so
    # its block divides the capacity.
    if page_size is None:
        if block_kv % MIN_BLOCK_KV or capacity % block_kv:
            raise ValueError(
                f"block_kv={block_kv} must be a {MIN_BLOCK_KV}-aligned divisor"
                f" of capacity={capacity}."
            )
    elif block_kv % MIN_BLOCK_KV or block_kv % page_size:
        raise ValueError(
            f"block_kv={block_kv} must be a {MIN_BLOCK_KV}-aligned multiple"
            f" of page_size={page_size}."
        )
    if block_kv_compute % MIN_BLOCK_KV or block_kv % block_kv_compute:
        raise ValueError(
            f"block_kv_compute={block_kv_compute} must be a"
            f" {MIN_BLOCK_KV}-aligned divisor of block_kv={block_kv}."
        )
    if num_kv_stages not in STAGES:
        raise ValueError(
            f"num_kv_stages must be one of {STAGES}; got {num_kv_stages}."
        )
    required_vmem = kvcache_config_vmem_bytes(
        block_kv,
        block_kv_compute,
        num_kv_stages,
        items_per_step,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_itemsize=kv_itemsize,
        batch=batch,
    )
    if required_vmem > vmem_limit_bytes():
        raise ValueError(
            f"KV-cache config requires {required_vmem} VMEM bytes, exceeding"
            f" limit {vmem_limit_bytes()}."
        )
    return block_kv, block_kv_compute, num_kv_stages, items_per_step


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
) -> tuple[int, int, int, int] | None:
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


def flash_attn_kvcache_pallas(
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array | None,
    k: jax.Array,
    v: jax.Array,
    cache_seqlens: jax.Array,
    cache_batch_idx: jax.Array,
    *,
    cache_head_major: bool,
    has_new: bool,
    left: int | None,
    return_lse: bool,
    interpret: bool,
    block_table: jax.Array | None = None,
    num_active: int | jax.Array | None = None,
    merged_cache: bool = False,
    softcap: float = 0.0,
    causal: bool = False,
    nheads_k: int | None = None,
    v_scale: float | jax.Array = 1.0,
    sinks: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
    """One-token cache append and decode attention for every row in one call.

    q: (batch, num_query_heads, head_dim), already scaled into log2 units, or
    by softmax_scale / softcap when softcap > 0, which applies Gemma-2's
    tanh(s / softcap) * softcap to the scaled scores. A (batch, window,
    num_query_heads, head_dim) q is a window already in the cache, folded into
    the query heads so each row's cache is read once; out and lse keep q's
    layout, and causal bounds window token t at cache_seqlens - window + 1 + t.
    k_cache/v_cache: (rows, capacity, num_kv_heads, head_dim), or head-major
    (rows, num_kv_heads, capacity, head_dim) with cache_head_major. With
    block_table, the flat (batch * pages_per_seq,) int32 page table, the caches
    are (num_pages, page_size, ...) pools and the per-row capacity is
    pages_per_seq * page_size.

    merged_cache (v_cache=None, token-major) makes k_cache one
    (rows, capacity, merged_cache_heads(nheads_k), head_dim) array whose token
    rows hold the KV heads' K and V rows in copy_utils' merged order (bf16
    [k0, v0, k1, v1, ...], RPA v3's, for bf16 and fp8), and returns
    (out, kv_cache[, lse]) instead of (out, k_cache, v_cache[, lse]). nheads_k
    defaults to half the cache's head axis; an fp8 cache, which pads an odd
    head count, needs it. An fp8 merged cache feeds K and V to the MXU as
    stored, k and v must be in its dtype, and v_scale, a scalar that may be
    traced, multiplies the output (q carries the K scale). lse is (batch,
    num_query_heads, 128) float32, (batch, window, num_query_heads, 128) for a
    window, with each value broadcast over lanes.

    num_active: rows at or past it are padding (RPA's distribution[-1]
    contract); the kernel never reads their cache_seqlens, cache_batch_idx or
    block_table entries, appends nothing for them and leaves their out and lse
    rows unspecified. None means every row is active.

    sinks: None or (num_query_heads,) float sink logits, on the scale of the
    softcapped scores; exp(sinks[h]) joins head h's softmax denominator.
    """
    assert (
        v_cache is None
    ) == merged_cache, (
        "merged_cache takes one cache operand and v_cache=None, a pair both."
    )
    assert not merged_cache or (
        not cache_head_major and k_cache.shape[2] % 2 == 0
    ), "merged_cache takes one token-major cache with an even head axis."
    cache_operands = (k_cache,) if merged_cache else (k_cache, v_cache)
    cache_shape = k_cache.shape
    is_window = q.ndim == 4
    if is_window:
        batch, query_tokens, window_heads, head_dim = q.shape
        sink_heads = window_heads
    else:
        batch, num_query_heads, head_dim = q.shape
        query_tokens = 1
        sink_heads = num_query_heads
    if sinks is not None and sinks.shape != (sink_heads,):
        raise ValueError(
            f"sinks must be ({sink_heads},), one per query head; got"
            f" {sinks.shape}."
        )
    if merged_cache:
        # Note (david): a merged build stages, appends and writes back whole
        # token rows, so no DMA cuts the head axis, and a staged u32 word holds
        # whole heads' K and V: the head count needs no tile alignment or
        # parity.
        row_tokens = cache_shape[1]
        num_kv_heads = resolve_merged_kv_heads(
            cache_shape[2], k_cache.dtype, nheads_k
        )
    elif k_cache.dtype != jnp.bfloat16:
        raise NotImplementedError(
            f"the K/V cache pair is bfloat16 only; got {k_cache.dtype}."
        )
    elif cache_head_major:
        num_kv_heads, row_tokens = cache_shape[1], cache_shape[2]
    else:
        row_tokens, num_kv_heads = cache_shape[1], cache_shape[2]
        if num_kv_heads != 1 and num_kv_heads % 2:
            # Note (david): the packed load bitcasts two bf16 heads into one u32
            # lane. The plain strided read an odd axis would need leaves a
            # singleton minor-but-one buffer dim that Mosaic pads to a full
            # 8-sublane tile, measured 5-10x slower at batch >= 32 and behind
            # the RPA v3 baseline.
            raise ValueError(
                "token-major cache needs an even head axis (the packed load"
                " carries two bf16 heads per u32 lane); got"
                f" num_kv_heads={num_kv_heads}."
            )
    # Note (david): with a singleton head axis a token-major
    # (rows, capacity, 1, head_dim) cache is the same bytes as head-major
    # (rows, 1, capacity, head_dim), whose tile is a dense (block_kv, head_dim)
    # window with no head pair to pack and no 8-sublane pad.
    is_token_major_mqa = (
        not cache_head_major and not merged_cache and num_kv_heads == 1
    )
    is_cache_head_major = cache_head_major or is_token_major_mqa
    if is_token_major_mqa:
        kernel_caches = tuple(
            cache.reshape(cache.shape[0], 1, row_tokens, head_dim)
            for cache in cache_operands
        )
    else:
        kernel_caches = cache_operands
    if block_table is None:
        page_size = None
        pages_per_seq = None
        capacity = row_tokens
    else:
        page_size = row_tokens
        pages_per_seq = block_table.shape[0] // batch
        capacity = pages_per_seq * page_size
    if head_dim not in SUPPORTED_HEAD_DIMS:
        raise ValueError(
            f"head_dim must be one of {SUPPORTED_HEAD_DIMS}; got {head_dim}."
        )
    if is_window:
        assert window_heads % num_kv_heads == 0
        assert not has_new and left is None, "a window is already in the cache"
        # Note (david): putting window token t of query head h * group + j at
        # folded head (h * query_tokens + t) * group + j lets KV head h's rows
        # share one read of its keys.
        group = window_heads // num_kv_heads
        num_query_heads = num_kv_heads * query_tokens * group
        q = (
            q.reshape(batch, query_tokens, num_kv_heads, group, head_dim)
            .transpose(0, 2, 1, 3, 4)
            .reshape(batch, num_query_heads, head_dim)
        )
        if sinks is not None:
            # Note (deepolar): folded head (h * query_tokens + t) * group + j is
            # query head h * group + j, so every window token takes that head's
            # sink.
            sinks = jnp.broadcast_to(
                sinks.reshape(num_kv_heads, 1, group),
                (num_kv_heads, query_tokens, group),
            ).reshape(num_query_heads)
    if capacity % MIN_BLOCK_KV:
        raise ValueError(
            f"capacity must be a multiple of {MIN_BLOCK_KV}; got {capacity}."
        )
    # Note (david): softcap stays out of the tuned key because decode is bound
    # by the cache read; the tanh did not move the v6e step time. q is always
    # bf16, so the key's dtype is the cache's.
    tuned_key_fields = dict(
        dtype=k_cache.dtype,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        capacity=capacity,
        batch=batch,
        append=has_new,
        sliding_window=left,
        return_lse=return_lse,
    )
    tuned_config = get_tuned_kvcache_config(**tuned_key_fields)
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
            kv_itemsize=k_cache.dtype.itemsize,
            batch=batch,
            page_size=page_size,
            device_name=get_device_name(),
        )
    else:
        packed_default = None
    # TODO (david): the tuned table holds one-item configs, tuned before the
    # packed kernel existed, and needs re-tuning with packed (4-field)
    # entries. Until then a merged-cache decode with a packed default skips a
    # 3-field entry and takes that default; a 4-field config still applies,
    # items_per_step 1 for the one-item kernel.
    if (
        packed_default is not None
        and tuned_config is not None
        and len(tuned_config) == 3
    ):
        tuned_config = None
    # Note (david): a paged block is whole pages, so a tuned block_kv that does
    # not tile the page cannot be used as-is.
    if (
        tuned_config is not None
        and page_size is not None
        and tuned_config[0] % page_size
    ):
        warn_untuned(
            kvcache_config_key(**tuned_key_fields),
            f"the entry's block_kv {tuned_config[0]} is not a multiple of the"
            f" page size {page_size}, and a paged block must be whole pages",
        )
        tuned_config = None
    if tuned_config is not None:
        config = tuned_config
    elif packed_default is not None:
        config = packed_default
    else:
        granule = MIN_BLOCK_KV if page_size is None else page_size
        target_block_kv = min(
            capacity,
            STAGED_KV_ELEMENT_BUDGET // head_dim // next_pow2(2 * num_kv_heads),
            MAX_BLOCK_KV,
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
        config = (
            block_kv,
            resolve_block_kv_compute(block_kv),
            DEFAULT_NUM_KV_STAGES,
        )
    block_kv, block_kv_compute, num_kv_stages, items_per_step = (
        validate_kvcache_config(
            config,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            capacity=capacity,
            kv_itemsize=k_cache.dtype.itemsize,
            batch=batch,
            page_size=page_size,
        )
    )
    if items_per_step > 1 and not merged_cache:
        raise NotImplementedError(
            "items_per_step > 1 packs merged-cache decode only; got a K/V"
            " cache pair."
        )
    is_packed = items_per_step > 1
    # Note (david): a row's keys stay within the caller's capacity, which the
    # padded table does not raise, so the guard is sized on it.
    guard_threshold = overflow_guard_threshold(capacity)
    if page_size is not None:
        # Note (david): the padded entries lie past every row's length and each
        # page's load is clipped to it, so page 0 stands in and is never read.
        padded_pages_per_seq = round_up(pages_per_seq, block_kv // page_size)
        block_table = jnp.pad(
            block_table.reshape(batch, pages_per_seq),
            ((0, 0), (0, padded_pages_per_seq - pages_per_seq)),
        ).reshape(-1)
        pages_per_seq = padded_pages_per_seq
    kv_spec = pl.BlockSpec(
        (batch, num_kv_heads, head_dim), lambda *_: (0, 0, 0)
    )
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    aliased_out_shapes = (
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        *(
            jax.ShapeDtypeStruct(cache.shape, cache.dtype)
            for cache in kernel_caches
        ),
    )
    if return_lse:
        lse_shape = (batch, num_query_heads, NUM_LANES)
        out_shapes = (
            *aliased_out_shapes,
            jax.ShapeDtypeStruct(lse_shape, jnp.float32),
        )
        lse_specs = (pl.BlockSpec(lse_shape, lambda *_: (0, 0, 0)),)
    else:
        out_shapes = aliased_out_shapes
        lse_specs = ()
    # Note (david): a packed step reads its rows' q and writes their out at
    # dynamic rows of whole-batch VMEM blocks, so no per-row DMA and its
    # branch splits the step; the one-item loop streams them through HBM.
    if is_packed:
        q_out_spec = pl.BlockSpec(q.shape, lambda *_: (0, 0, 0))
    else:
        q_out_spec = hbm_spec
    out_specs = (
        (q_out_spec,) + (hbm_spec,) * (len(aliased_out_shapes) - 1) + lse_specs
    )
    # Note (david): the pair-packed load bitcasts refs, which only the TPU build
    # supports; interpret mode reads the staged heads as bf16.
    is_bitcast_load = not interpret and not is_cache_head_major
    staged_kv_heads = cache_shape[2] if merged_cache else num_kv_heads
    word_rows = kv_word_rows(k_cache.dtype)
    num_slots = kv_staging_slots(num_kv_stages, items_per_step)
    num_write_slots = cache_write_slots(num_kv_stages, items_per_step)

    def _staging_shape(num_stages, num_tokens, word_stride):
        if is_cache_head_major:
            return (num_stages, num_kv_heads, num_tokens, head_dim)
        elif is_bitcast_load:
            return (num_stages, num_tokens, word_stride, word_rows, head_dim)
        else:
            return (num_stages, num_tokens, staged_kv_heads, head_dim)

    # Note (david): only the staged blocks are read with strided word loads,
    # so only they take the bank pad; the write tiles stay dense for their
    # one-DMA write-back. A write tile is one cache tile of tokens: 16 on the
    # head-major token axis (the bf16 sublane tile v7x uses), 8 token-major
    # (see cache_tile_rows in the kernel).
    token_words = staged_kv_heads // word_rows
    cache_buffer_shape = _staging_shape(
        num_slots,
        items_per_step * block_kv,
        bank_safe_word_stride(token_words),
    )
    write_tile_rows = BF16_TILE_ROWS if is_cache_head_major else NUM_SUBLANES
    # Note (david): a packed call writes no tile (see cache_write_slots); one
    # unused tile keeps its write scratch from being empty.
    num_write_tiles = max(num_write_slots, 1)
    write_buffer_shape = _staging_shape(
        num_write_tiles, write_tile_rows, token_words
    )
    # Note (david): a merged build never touches the V staging, so it shrinks to
    # the write-tile size instead of a whole KV block.
    value_buffer_shape = (
        write_buffer_shape if merged_cache else cache_buffer_shape
    )
    io_buffer_shape = (NUM_IO_STAGES, num_query_heads, head_dim)
    state_shape = (NUM_ROW_PARITIES, num_query_heads, NUM_LANES)
    kv_heads_per_fragment, kv_heads_per_group = fragment_geometry(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        cache_dtype=k_cache.dtype,
        is_merged_cache=merged_cache,
        is_bitcast_load=is_bitcast_load,
    )
    rows_per_fragment = (
        num_query_heads
        // num_kv_heads
        * kv_heads_per_fragment
        * kv_heads_per_group
    )
    num_head_groups = (
        num_kv_heads // kv_heads_per_fragment // kv_heads_per_group
    )
    staged_value_tiles = (
        items_per_step
        * (block_kv // block_kv_compute)
        * num_head_groups
        * kv_heads_per_group
        if is_packed
        else 1
    )
    # Note (david): the one-item kernel never touches it, so it takes one
    # 32-row tile, whole for every cache dtype.
    staged_values_shape = (
        staged_value_tiles,
        kv_heads_per_fragment * block_kv_compute if is_packed else 32,
        head_dim,
    )
    scratch_shapes = (
        pltpu.VMEM(cache_buffer_shape, k_cache.dtype),
        pltpu.VMEM(value_buffer_shape, k_cache.dtype),
        pltpu.VMEM(write_buffer_shape, k_cache.dtype),
        pltpu.VMEM(write_buffer_shape, k_cache.dtype),
        pltpu.VMEM(io_buffer_shape, q.dtype),
        pltpu.VMEM(io_buffer_shape, q.dtype),
        pltpu.VMEM(state_shape, jnp.float32),
        pltpu.VMEM(state_shape, jnp.float32),
        # Note (david): a packed step accumulates in f32 and keeps only the
        # state it hands the next step here, so that state stays f32.
        pltpu.VMEM(
            (NUM_ROW_PARITIES + 1, num_query_heads, head_dim),
            jnp.float32 if is_packed else q.dtype,
        ),
        pltpu.VMEM(
            (rows_per_fragment, kv_heads_per_fragment * block_kv_compute),
            jnp.bfloat16,
        ),
        pltpu.VMEM((rows_per_fragment, NUM_LANES), jnp.float32),
        pltpu.SMEM((NUM_PENDING_FIELDS,), jnp.int32),
        # Note (david): a packed step's unpacked V tiles, one per item,
        # fragment, head group and group member (see _attend_packed_step).
        pltpu.VMEM(staged_values_shape, k_cache.dtype),
        pltpu.SemaphoreType.DMA((
            NUM_SEMS,
            max(num_slots, num_write_tiles, NUM_IO_STAGES),
            num_kv_heads,
        )),
    )
    if num_active is None:
        num_active_array = jnp.full((1,), batch, jnp.int32)
    else:
        num_active_array = jnp.asarray(num_active, jnp.int32).reshape(1)
    v_scale_array = jnp.asarray(v_scale, jnp.float32).reshape(1)
    # Note (david): a row's keys stay within capacity, so it spans at most
    # cdiv(capacity, block_kv) blocks; a row with none still takes one (empty)
    # item.
    layout = DecodeScheduleLayout(
        max_items=round_up(batch * -(-capacity // block_kv), items_per_step),
        items_per_step=items_per_step,
        pages_per_item=(
            block_kv // page_size if is_packed and page_size is not None else 0
        ),
        write_rows=batch if is_packed and has_new else 0,
    )
    schedule = build_decode_schedule(
        cache_seqlens,
        num_active_array[0],
        batch=batch,
        layout=layout,
        block_kv=block_kv,
        has_new=has_new,
        left=left,
        cache_tile_rows=write_tile_rows,
        cache_batch_idx=cache_batch_idx,
        block_table=block_table,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
    )
    scalar_prefetches = (
        cache_seqlens,
        cache_batch_idx,
        num_active_array,
        v_scale_array,
        schedule,
    ) + (() if page_size is None else (block_table,))
    num_prefetch = len(scalar_prefetches)
    if merged_cache:
        # Note (david): the new K and V are interleaved into the cache's head
        # order here, a (batch, staged_kv_heads, head_dim) op far below one
        # cache read, so the kernel appends a ready token row and writes each
        # appended tile back with one contiguous DMA.
        new_kv_shape = (batch, staged_kv_heads, head_dim)
        head_pad = ((0, 0), (0, staged_kv_heads // 2 - num_kv_heads), (0, 0))
        new_kv = jnp.stack(
            [jnp.pad(new_rows, head_pad) for new_rows in (k, v)], axis=2
        ).reshape(new_kv_shape)
        dense_inputs = (q, new_kv)
        new_token_specs = (pl.BlockSpec(new_kv_shape, lambda *_: (0, 0, 0)),)
    else:
        dense_inputs = (q, k, v)
        new_token_specs = (kv_spec, kv_spec)
    # Note (david): each cache aliases its own output, so the append lands in
    # place, and the one-item kernel's q aliases out (same shape and dtype).
    # The packed kernel's out must not alias q: their VMEM blocks would share
    # a buffer, and a row's earlier items rewrite the out the later ones read
    # as q.
    input_output_aliases = ({} if is_packed else {num_prefetch: 0}) | {
        num_prefetch + len(dense_inputs) + index: 1 + index
        for index in range(len(kernel_caches))
    }
    kernel = functools.partial(
        kvcache_kernel,
        layout=layout,
        batch=batch,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_kv=block_kv,
        block_kv_compute=block_kv_compute,
        num_kv_stages=num_kv_stages,
        items_per_step=items_per_step,
        is_cache_head_major=is_cache_head_major,
        is_bitcast_load=is_bitcast_load,
        return_lse=return_lse,
        has_new=has_new,
        left=left,
        query_tokens=query_tokens,
        causal=causal,
        log2_softcap=softcap * LOG2E,
        guard_threshold=guard_threshold,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        is_merged_cache=merged_cache,
        has_sinks=sinks is not None,
    )
    # Note (deepolar): the sinks follow the caches, so the alias indices above
    # stay put; they arrive in log2 units, broadcast over the row-state lanes.
    if sinks is None:
        sink_inputs, sink_specs = (), ()
    else:
        sink_shape = (num_query_heads, NUM_LANES)
        sink_inputs = (
            jnp.broadcast_to(
                (sinks.astype(jnp.float32) * LOG2E)[:, None], sink_shape
            ),
        )
        sink_specs = (pl.BlockSpec(sink_shape, lambda *_: (0, 0)),)
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=num_prefetch,
            grid=(1,),
            in_specs=(q_out_spec, *new_token_specs)
            + (hbm_spec,) * len(kernel_caches)
            + sink_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shapes,
        input_output_aliases=input_output_aliases,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_bytes(),
            disable_bounds_checks=True,
            disable_semaphore_checks=True,
        ),
        name="flash_attn_with_kvcache",
        interpret=pltpu.InterpretParams() if interpret else False,
    )
    kernel_outputs = call(
        *scalar_prefetches, *dense_inputs, *kernel_caches, *sink_inputs
    )
    num_caches = len(cache_operands)
    if is_token_major_mqa:
        caches_out = tuple(
            out_cache.reshape(operand.shape)
            for out_cache, operand in zip(
                kernel_outputs[1 : 1 + num_caches], cache_operands
            )
        )
    else:
        caches_out = tuple(kernel_outputs[1 : 1 + num_caches])
    if is_window:
        out = (
            kernel_outputs[0]
            .reshape(batch, num_kv_heads, query_tokens, group, head_dim)
            .transpose(0, 2, 1, 3, 4)
            .reshape(batch, query_tokens, window_heads, head_dim)
        )
        lse_out = tuple(
            lse.reshape(batch, num_kv_heads, query_tokens, group, NUM_LANES)
            .transpose(0, 2, 1, 3, 4)
            .reshape(batch, query_tokens, window_heads, NUM_LANES)
            for lse in kernel_outputs[1 + num_caches :]
        )
    else:
        out = kernel_outputs[0]
        lse_out = tuple(kernel_outputs[1 + num_caches :])
    return (out, *caches_out, *lse_out)
