"""Masked-dense DSA MLA: dense MQA over the split fp8 cache, with each token's
scores masked to its top-k selection; and the SparseCore call that builds the
selection bitmaps."""

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu, tpu_sc as plsc

from flywheel_tpu.core.tpu import (
    BF16_BYTES,
    BF16_TILE_ROWS,
    F32_BYTES,
    NUM_LANES,
    NUM_SUBLANES,
)
from flywheel_tpu.softmax_attention.config import (
    LOG2E,
    NT_DIM_NUMBERS,
    next_pow2,
    vmem_limit_bytes,
)
from flywheel_tpu.softmax_attention.kernels.decode import lane_broadcast
from flywheel_tpu.softmax_attention.kernels.mla_decode import (
    LATENT_DIM,
    ROPE_DIM,
)
from flywheel_tpu.softmax_attention.kernels.mla_prefill import (
    Q_BLOCK_MASK,
    REQUEST_SHIFT,
    mla_prefill_schedule,
)
from flywheel_tpu.sparse_attention.dsa.kernels.gather import (
    NOPE_WORDS,
    QUAD_TOKENS,
    ROPE_TOKENS_PER_ROW,
    ROPE_WORDS,
    WORD_BITS,
    sparse_core_mesh,
)
from flywheel_tpu.sparse_attention.dsa.kernels.sparse_mla import (
    byte_planes,
)
from flywheel_tpu.topk.kernels.sparse_core_topk import WORD_BYTES

FP8_DTYPE = jnp.dtype(jnp.float8_e4m3fn)
FP8_BYTES = FP8_DTYPE.itemsize
# A staged key row: the latent dims, the rope dims, then zero lanes up to a
# whole lane tile, which the queries' zero lanes meet.
KEY_ROW_DIM = LATENT_DIM + NUM_LANES
ROPE_LANES = QUAD_TOKENS * ROPE_WORDS
ROPE_WORD_SHIFT = ROPE_WORDS.bit_length() - 1
# Note (david): the copies of the next two items run while one computes, and
# a q block's copies land while the block before it computes.
NUM_KV_STAGES = 3
NUM_QUERY_SLOTS = 2
# Note (david): the mask rides in the key rows' pad lanes. A token's rows hold
# FLAG_QUERY in its lane and a key holds MASKED_KEY there where the token does
# not see it, so the QK matmul itself puts a masked score far below any real
# one. A q block holds at most MAX_BLOCK_Q tokens, one per mask lane.
FLAG_QUERY = 2.0**100
MASKED_KEY = -448.0
MAX_BLOCK_Q = KEY_ROW_DIM - LATENT_DIM - ROPE_LANES
# Note (david): a row's running max starts at NO_KEY_MAX, far above any masked
# score at any positive score scale and below any real one, so a row that has
# seen no key yet keeps every probability exactly 0.
NO_KEY_MAX = -1e20
# Note (david): big q blocks amortize a kv block's copy and staging over many
# score rows. Row groups of GROUP_ROWS keep the MXU fed while the vector units
# run the next group's softmax; a q block of few rows takes a longer kv block
# to keep an item near TARGET_ITEM_SCORES scores.
TARGET_ROWS = 4096
GROUP_ROWS = 1024
MIN_BLOCK_KV = 512
MAX_BLOCK_KV = 2048
TARGET_ITEM_SCORES = TARGET_ROWS * MIN_BLOCK_KV
VMEM_BUDGET_FRACTION = 0.75
# A bitmap word row covers SLAB_POSITIONS positions: bit b of lane l flags
# position b * NUM_LANES + l of the slab.
SLAB_POSITIONS = WORD_BITS * NUM_LANES
SLAB_SHIFT = SLAB_POSITIONS.bit_length() - 1
LANE_SHIFT = NUM_LANES.bit_length() - 1
SEM_NOPE = 0
SEM_ROPE = 1
SEM_QUERY = 2
SEM_OUTPUT = 3
NUM_SEMS = 4
SEM_INDEX = 0
SEM_WORDS = 1
NUM_BITMAP_SEMS = 2
NUM_BITMAP_BUFFERS = 2
BITMAP_UNROLL = 4


class MaskedDenseConfig(NamedTuple):
    """Query tokens of a q block, cache tokens of a kv block, and the row
    groups an item splits the q block's rows into."""

    block_q: int
    block_kv: int
    row_groups: int


class KernelParams(NamedTuple):
    """Static parameters of one masked-dense call; bitmap_rows is 0 for a
    call without bitmaps."""

    config: MaskedDenseConfig
    num_heads: int
    page_size: int
    pages_per_seq: int
    max_q_blocks: int
    top_k: int
    num_slabs: int
    bitmap_rows: int
    sm_scale_log2: float
    return_lse: bool


class KernelRefs(NamedTuple):
    """The masked-dense kernel's operands, outputs and scratch."""

    schedule: jax.Array
    cu_q_lens: jax.Array
    kv_lens: jax.Array
    block_table: jax.Array
    kv_scale: jax.Array
    q_nope: jax.Array
    q_pe_t: jax.Array
    nope_cache: jax.Array
    rope_cache: jax.Array
    bitmap: jax.Array | None
    out: jax.Array
    lse: jax.Array | None
    nope_ring: jax.Array
    rope_ring: jax.Array
    keys: jax.Array
    rope_words: jax.Array
    query_buffer: jax.Array
    q_pe_stage: jax.Array
    out_stage: jax.Array
    row_max: jax.Array
    row_sum: jax.Array
    accumulator: jax.Array
    sems: jax.Array
    bitmap_stage: jax.Array | None
    lse_stage: jax.Array | None


class QueryBlock(NamedTuple):
    """Scalars of one q block of the schedule."""

    ordinal: jax.Array
    request: jax.Array
    q_start: jax.Array
    q_count: jax.Array
    kv_len: jax.Array
    first_position: jax.Array
    num_kv_blocks: jax.Array


def bitmap_rows(num_slabs: int) -> int:
    """Word rows of a token's bitmap: its slabs in whole sublane tiles."""
    return pl.cdiv(num_slabs, NUM_SUBLANES) * NUM_SUBLANES


def selection_bitmap_kernel(
    indices_hbm: jax.Array,
    lengths_hbm: jax.Array,
    bitmap_hbm: jax.Array,
    index_vmem: jax.Array,
    word_vmem: jax.Array,
    length_vmem: jax.Array,
    sems: jax.Array,
    *,
    num_tokens: int,
    num_slabs: int,
    tokens_per_worker: int,
    num_subcores: int,
    num_lanes: int,
) -> None:
    """Each worker writes the bitmaps of its run of tokens: a used index i
    sets bit (i // NUM_LANES) % WORD_BITS of word (i // SLAB_POSITIONS,
    i % NUM_LANES). A token's used indices are distinct, so adding their bits
    ORs them."""
    rows_per_token, top_k = word_vmem.shape[1], index_vmem.shape[1]
    worker = lax.axis_index("core") * num_subcores + lax.axis_index("subcore")
    first_token = worker * tokens_per_worker
    num_local = jnp.clip(num_tokens - first_token, 0, tokens_per_worker)

    def _index_copy(local, slot):
        return pltpu.make_async_copy(
            indices_hbm.at[pl.ds((first_token + local) * top_k, top_k)],
            index_vmem.at[slot],
            sems.at[SEM_INDEX, slot],
        )

    def _word_copy(local, slot):
        token_rows = pl.ds(
            (first_token + local) * rows_per_token, rows_per_token
        )
        return pltpu.make_async_copy(
            word_vmem.at[slot],
            bitmap_hbm.at[token_rows],
            sems.at[SEM_WORDS, slot],
        )

    def _set_bits(local, slot):
        zeros = jnp.zeros((num_lanes,), jnp.int32)
        for row in range(num_slabs):
            for lane in range(0, NUM_LANES, num_lanes):
                word_vmem[slot, row, pl.ds(lane, num_lanes)] = zeros
        # Note (david): the length is lane 0 of a vector load from the token's
        # word on, which holds even when the worker has a single token.
        length = length_vmem[pl.ds(local, num_lanes)][0]
        lane_slot = lax.broadcasted_iota(jnp.int32, (num_lanes,), 0)

        @plsc.parallel_loop(0, top_k // num_lanes, unroll=BITMAP_UNROLL)
        def _vector(vector):
            index = index_vmem[slot, pl.ds(vector * num_lanes, num_lanes)]
            # Note (david): -1 compares as a large unsigned value, so one
            # compare drops it and every index past the token; a slot past
            # the row length is unused whatever it holds.
            is_kept = (
                lax.bitcast_convert_type(index, jnp.uint32)
                < lax.bitcast_convert_type(length, jnp.uint32)
            ) & (vector * num_lanes + lane_slot < length)
            bit = lax.shift_right_logical(index, LANE_SHIFT) & (WORD_BITS - 1)
            plsc.addupdate_scatter(
                word_vmem.at[slot],
                [
                    lax.shift_right_logical(index, SLAB_SHIFT),
                    index & (NUM_LANES - 1),
                ],
                lax.shift_left(jnp.ones_like(index), bit),
                mask=is_kept,
            )

    @pl.when(num_local > 0)
    def _work():
        _index_copy(0, 0).start()
        pltpu.sync_copy(
            lengths_hbm.at[worker],
            length_vmem.at[pl.ds(0, lengths_hbm.shape[1])],
        )

        @pl.loop(0, num_local)
        def _token(local):
            slot = lax.rem(local, NUM_BITMAP_BUFFERS)

            @pl.when(local + 1 < num_local)
            def _prefetch():
                _index_copy(local + 1, 1 - slot).start()

            @pl.when(local >= NUM_BITMAP_BUFFERS)
            def _wait_former_words():
                _word_copy(local - NUM_BITMAP_BUFFERS, slot).wait()

            _index_copy(local, slot).wait()
            _set_bits(local, slot)
            _word_copy(local, slot).start()

        # Note (david): a kernel must not exit with DMAs in flight.
        for distance in range(NUM_BITMAP_BUFFERS, 0, -1):
            local = num_local - distance

            @pl.when(local >= 0)
            def _wait_last_words(local=local):
                _word_copy(local, lax.rem(local, NUM_BITMAP_BUFFERS)).wait()


def dsa_selection_bitmap_pallas(
    topk_indices: jax.Array, row_lengths: jax.Array, *, num_slabs: int
) -> jax.Array:
    """Each token's selection bitmap on the SparseCores.

    Args:
      topk_indices: (T, top_k) int32 positions, -1 for an unused slot; a
        token's entries below its row length in its first row-length slots
        distinct.
      row_lengths: (T,) int32, a token's position + 1, 0 for padding: only
        its first row-length slots are read, and only their entries below
        it set bits.
      num_slabs: slabs of SLAB_POSITIONS positions a bitmap covers.

    Returns:
      (T * bitmap_rows(num_slabs), NUM_LANES) int32: token t's rows from
      t * bitmap_rows(num_slabs), bit b of row s lane l flagging position
      s * SLAB_POSITIONS + b * NUM_LANES + l. Rows past num_slabs are
      unspecified.
    """
    num_tokens, top_k = topk_indices.shape
    sparse_core = pltpu.get_tpu_info().sparse_core
    num_workers = sparse_core.num_cores * sparse_core.num_subcores
    rows_per_token = bitmap_rows(num_slabs)
    tokens_per_worker = pl.cdiv(num_tokens, num_workers)
    # Note (david): a worker copies its lengths as whole lane vectors; the clip
    # keeps every bit within the bitmap.
    worker_lanes = (
        pl.cdiv(tokens_per_worker, sparse_core.num_lanes)
        * sparse_core.num_lanes
    )
    lengths = jnp.pad(
        jnp.clip(row_lengths, 0, num_slabs * SLAB_POSITIONS),
        (0, num_workers * tokens_per_worker - num_tokens),
    ).reshape(num_workers, tokens_per_worker)
    lengths = jnp.pad(lengths, ((0, 0), (0, worker_lanes - tokens_per_worker)))
    body = functools.partial(
        selection_bitmap_kernel,
        num_tokens=num_tokens,
        num_slabs=num_slabs,
        tokens_per_worker=tokens_per_worker,
        num_subcores=sparse_core.num_subcores,
        num_lanes=sparse_core.num_lanes,
    )
    # Note (david): the indices pass flat, as the gather reads them, so the
    # top-k's output needs no relayout.
    return pl.kernel(
        body,
        out_type=jax.ShapeDtypeStruct(
            (num_tokens * rows_per_token, NUM_LANES), jnp.int32
        ),
        mesh=sparse_core_mesh(),
        scratch_types=(
            pltpu.VMEM((NUM_BITMAP_BUFFERS, top_k), jnp.int32),
            pltpu.VMEM(
                (NUM_BITMAP_BUFFERS, rows_per_token, NUM_LANES), jnp.int32
            ),
            # A lane vector past the last token's word, for its load.
            pltpu.VMEM((worker_lanes + sparse_core.num_lanes,), jnp.int32),
            pltpu.SemaphoreType.DMA((NUM_BITMAP_SEMS, NUM_BITMAP_BUFFERS)),
        ),
        # Note (david): Pallas reuses a ref access's lowering across the kernels
        # of one jit, and the top-k in this jit takes no layout passes, so
        # neither does this.
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=False,
            disable_bounds_checks=True,
        ),
        name="dsa_selection_bitmap",
    )(topk_indices.reshape(-1), lengths)


class MaskedDenseKernel:
    """One masked-dense call: the schedule's items, (q block, kv block) pairs
    in q block order. Each item stages its kv block's keys from the split cache
    with its selection mask, then updates the q block's rows by online softmax,
    a row group at a time.

    Score rows are a q block's tokens times its heads, token-major (row
    t * H + h); scores are in log2 units.
    """

    def __init__(self, params: KernelParams, refs: KernelRefs):
        self.params = params
        self.refs = refs
        self.block_q, self.block_kv, self.row_groups = params.config
        self.num_heads = params.num_heads
        self.num_rows = self.block_q * self.num_heads
        self.group_tokens = self.block_q // self.row_groups
        self.group_rows = self.group_tokens * self.num_heads
        self.num_q_blocks = refs.schedule[2 * params.max_q_blocks]
        self.num_items = refs.schedule[2 * params.max_q_blocks + 1]
        self.kv_scale = refs.kv_scale[0]
        # Note (david): K and V share the cache's kv_scale, so it folds into
        # the score scale and the output normalization.
        self.score_scale = self.kv_scale * params.sm_scale_log2

    def query_block(self, ordinal: jax.Array) -> QueryBlock:
        max_q_blocks = self.params.max_q_blocks
        # Note (david): the prefetch cursor runs past the last q block on the
        # last items, so its read stays in the table.
        entry_index = jnp.minimum(ordinal, max_q_blocks - 1)
        entry = self.refs.schedule[entry_index]
        request = lax.shift_right_logical(entry, REQUEST_SHIFT)
        q_end = self.refs.cu_q_lens[request + 1]
        q_start = (
            self.refs.cu_q_lens[request] + (entry & Q_BLOCK_MASK) * self.block_q
        )
        kv_len = self.refs.kv_lens[request]
        return QueryBlock(
            ordinal,
            request,
            q_start,
            jnp.minimum(self.block_q, q_end - q_start),
            kv_len,
            kv_len - q_end + q_start,
            self.refs.schedule[max_q_blocks + entry_index],
        )

    @staticmethod
    def next_item(
        block: QueryBlock, kv_block: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        is_block_end = kv_block + 1 >= block.num_kv_blocks
        return (
            jnp.where(is_block_end, block.ordinal + 1, block.ordinal),
            jnp.where(is_block_end, 0, kv_block + 1),
        )

    def cache_windows(
        self, kv_block: jax.Array
    ) -> list[tuple[int, jax.Array, jax.Array, int]]:
        """(ring row, page index, page row, tokens) of each cache window the
        kv block reads: its whole pages, or its part of one page."""
        page_size = self.params.page_size
        if self.block_kv >= page_size:
            pages_per_block = self.block_kv // page_size
            return [
                (
                    page * page_size,
                    kv_block * pages_per_block + page,
                    0,
                    page_size,
                )
                for page in range(pages_per_block)
            ]
        else:
            blocks_per_page = page_size // self.block_kv
            page_row = pl.multiple_of(
                lax.rem(kv_block, blocks_per_page) * self.block_kv,
                self.block_kv,
            )
            return [(0, kv_block // blocks_per_page, page_row, self.block_kv)]

    def kv_copies(
        self, block: QueryBlock, kv_block: jax.Array, slot: jax.Array
    ) -> list[tuple[jax.Array, tuple]]:
        """(is_copied, (nope copy, rope copy)) of each cache window; a window
        that holds no position below kv_len copies nothing."""
        pages_per_seq = self.params.pages_per_seq
        first_key = kv_block * self.block_kv
        copies = []
        for ring_row, page_index, page_row, num_tokens in self.cache_windows(
            kv_block
        ):
            is_copied = block.kv_len > first_key + ring_row
            table_index = block.request * pages_per_seq + jnp.minimum(
                page_index, pages_per_seq - 1
            )
            page = jnp.where(is_copied, self.refs.block_table[table_index], 0)
            rope_row = page_row // ROPE_TOKENS_PER_ROW
            num_rope_rows = num_tokens // ROPE_TOKENS_PER_ROW
            nope_copy = pltpu.make_async_copy(
                self.refs.nope_cache.at[page, pl.ds(page_row, num_tokens)],
                self.refs.nope_ring.at[slot, pl.ds(ring_row, num_tokens)],
                self.refs.sems.at[SEM_NOPE, slot],
            )
            rope_copy = pltpu.make_async_copy(
                self.refs.rope_cache.at[page, pl.ds(rope_row, num_rope_rows)],
                self.refs.rope_ring.at[
                    slot,
                    pl.ds(ring_row // ROPE_TOKENS_PER_ROW, num_rope_rows),
                ],
                self.refs.sems.at[SEM_ROPE, slot],
            )
            copies.append((is_copied, (nope_copy, rope_copy)))
        return copies

    def start_kv_copies(
        self, block: QueryBlock, kv_block: jax.Array, slot: jax.Array
    ) -> None:
        for is_copied, window_copies in self.kv_copies(block, kv_block, slot):
            pl.when(is_copied)(functools.partial(_start_all, window_copies))

    def wait_kv_copies(
        self, block: QueryBlock, kv_block: jax.Array, slot: jax.Array
    ) -> None:
        for is_copied, window_copies in self.kv_copies(block, kv_block, slot):
            pl.when(is_copied)(functools.partial(_wait_all, window_copies))

    def query_copies(self, block: QueryBlock, slot: jax.Array) -> tuple:
        refs = self.refs
        sem = refs.sems.at[SEM_QUERY, slot]
        tokens = pl.ds(block.q_start, block.q_count)
        staged = pl.ds(0, block.q_count)
        copies = (
            pltpu.make_async_copy(
                refs.q_nope.at[tokens],
                refs.query_buffer.at[slot, staged, :, pl.ds(0, LATENT_DIM)],
                sem,
            ),
            pltpu.make_async_copy(
                refs.q_pe_t.at[tokens], refs.q_pe_stage.at[slot, staged], sem
            ),
        )
        if refs.bitmap is None:
            return copies
        else:
            rows = self.params.bitmap_rows
            return copies + (
                pltpu.make_async_copy(
                    refs.bitmap.at[
                        pl.ds(block.q_start * rows, block.q_count * rows)
                    ],
                    refs.bitmap_stage.at[slot, pl.ds(0, block.q_count * rows)],
                    sem,
                ),
            )

    def output_copies(self, block: QueryBlock, slot: jax.Array) -> tuple:
        refs = self.refs
        sem = refs.sems.at[SEM_OUTPUT, slot]
        out_copy = pltpu.make_async_copy(
            refs.out_stage.at[slot, pl.ds(0, block.q_count)],
            refs.out.at[pl.ds(block.q_start, block.q_count)],
            sem,
        )
        if refs.lse is None:
            return (out_copy,)
        else:
            return (
                out_copy,
                pltpu.make_async_copy(
                    refs.lse_stage.at[slot], refs.lse.at[block.ordinal], sem
                ),
            )

    def rope_keys(self, kv_slot: jax.Array, valid: jax.Array) -> None:
        """The kv block's rope keys as 4-token words in rope_words: the rope in
        lanes [0, ROPE_LANES) and zeros after; tokens at or past valid zero.
        """
        rows = self.refs.rope_ring[kv_slot]
        lane = lax.broadcasted_iota(jnp.int32, rows.shape, 1)
        lane_group = lax.shift_right_logical(lane, ROPE_WORD_SHIFT)
        token = (
            ROPE_TOKENS_PER_ROW * lax.broadcasted_iota(jnp.int32, rows.shape, 0)
            + lane_group
        )
        rows = jnp.where(token < valid, rows, 0)
        bands = byte_planes([
            rows if t == 0 else pltpu.roll(rows, NUM_LANES - ROPE_WORDS * t, 1)
            for t in range(QUAD_TOKENS)
        ])
        lane_band = lane_group & (QUAD_TOKENS - 1)
        merged = bands[0]
        for band in range(1, QUAD_TOKENS):
            merged = jnp.where(
                lane_band == band,
                pltpu.roll(bands[band], ROPE_WORDS * band, 1),
                merged,
            )
        is_rope_lane = lane < ROPE_LANES
        num_rows = rows.shape[0]
        self.refs.rope_words[pl.ds(0, num_rows, stride=2)] = jnp.where(
            is_rope_lane, merged, 0
        )
        self.refs.rope_words[pl.ds(1, num_rows, stride=2)] = jnp.where(
            is_rope_lane, pltpu.roll(merged, ROPE_LANES, 1), 0
        )

    def stage_keys(
        self,
        block: QueryBlock,
        kv_block: jax.Array,
        kv_slot: jax.Array,
        query_slot: jax.Array,
    ) -> None:
        """The kv block's (block_kv, KEY_ROW_DIM) fp8 key rows into keys: the
        latent, the rope, then each q block token's mask lane. Tokens at or past
        kv_len are zero, as their ring rows are stale or unwritten.
        """
        valid = block.kv_len - kv_block * self.block_kv
        quarter = self.block_kv // QUAD_TOKENS
        # Note (david): the nope words are banded: byte b of word w is dim
        # 128 b + w.
        row = lax.broadcasted_iota(jnp.int32, (quarter, NOPE_WORDS), 0)
        words = [
            jnp.where(
                QUAD_TOKENS * row + token < valid,
                self.refs.nope_ring[
                    kv_slot, pl.ds(token, quarter, stride=QUAD_TOKENS)
                ],
                0,
            )
            for token in range(QUAD_TOKENS)
        ]
        for band, plane in enumerate(byte_planes(words)):
            self.refs.keys[:, pl.ds(band * NUM_LANES, NUM_LANES)] = (
                pltpu.bitcast(plane, FP8_DTYPE)
            )
        self.rope_keys(kv_slot, valid)
        # Note (david): every kv block takes its mask lanes, even one its q
        # block sees whole: a branch around them would end the item's basic
        # block before its matmuls, which costs more than the lanes.
        mask_words = pltpu.bitcast(
            self.mask_lanes(block, kv_block, query_slot).astype(FP8_DTYPE),
            jnp.int32,
        )
        self.refs.keys[:, pl.ds(LATENT_DIM, NUM_LANES)] = pltpu.bitcast(
            self.refs.rope_words[...] | mask_words, FP8_DTYPE
        )

    def selected_keys(
        self, first_key: jax.Array, query_slot: jax.Array
    ) -> jax.Array:
        """(block_kv, NUM_LANES) bool: whether each token's bitmap selects
        each key of the kv block, token t in lane ROPE_LANES + t."""
        slab = lax.shift_right_logical(first_key, SLAB_SHIFT)
        words = jnp.zeros((self.block_q, NUM_LANES), jnp.int32)
        for slab_row in range(self.params.num_slabs):
            words = jnp.where(
                slab == slab_row,
                self.refs.bitmap_stage[
                    query_slot,
                    pl.ds(
                        slab_row, self.block_q, stride=self.params.bitmap_rows
                    ),
                ],
                words,
            )
        # Note (david): key k's flag is bit (k // NUM_LANES) % WORD_BITS of its
        # token's word in lane k % NUM_LANES, so the transposed words hold every
        # key tile's flags, one bit each.
        words = jnp.concatenate(
            [words, jnp.zeros((NUM_LANES - self.block_q, NUM_LANES), jnp.int32)]
        )
        key_words = pltpu.roll(words.T, ROPE_LANES, 1)
        first_bit = lax.shift_right_logical(
            first_key & (SLAB_POSITIONS - 1), LANE_SHIFT
        )
        tiles = [
            lax.shift_right_logical(key_words, first_bit + tile) & 1
            for tile in range(self.block_kv // NUM_LANES)
        ]
        return jnp.concatenate(tiles, axis=0) != 0

    def mask_lanes(
        self, block: QueryBlock, kv_block: jax.Array, query_slot: jax.Array
    ) -> jax.Array:
        """(block_kv, NUM_LANES) f32 key lanes that mask the scores: lane
        ROPE_LANES + t of a key is MASKED_KEY where q block token t does not
        see it, else 0.

        A request of at most top_k positions sees each token's causal prefix.
        A longer one sees the part of each token's bitmap within that prefix.
        """
        first_key = kv_block * self.block_kv
        shape = (self.block_kv, NUM_LANES)
        key = lax.broadcasted_iota(jnp.int32, shape, 0)
        lane = lax.broadcasted_iota(jnp.int32, shape, 1)
        is_mask_lane = lane >= ROPE_LANES
        mask = jnp.where(
            key + (first_key - block.first_position) <= lane - ROPE_LANES,
            0.0,
            MASKED_KEY,
        )
        if self.refs.bitmap_stage is not None:
            is_bitmap_request = block.kv_len > self.params.top_k
            mask = jnp.where(
                jnp.logical_and(
                    is_bitmap_request,
                    jnp.logical_not(self.selected_keys(first_key, query_slot)),
                ),
                MASKED_KEY,
                mask,
            )
        return jnp.where(is_mask_lane, mask, 0.0)

    def softmax(
        self, query_slot: jax.Array, group: int, is_first: jax.Array
    ) -> tuple[pl.Slice, jax.Array, jax.Array]:
        """Row group group's online-softmax update by the staged kv block:
        returns its rows, fp8 probabilities and the rescale of its earlier
        sums."""
        refs = self.refs
        rows = pl.ds(group * self.group_rows, self.group_rows)
        tokens = pl.ds(group * self.group_tokens, self.group_tokens)
        queries = refs.query_buffer[query_slot, tokens].reshape(
            self.group_rows, KEY_ROW_DIM
        )
        scores = lax.dot_general(
            queries,
            refs.keys[...],
            NT_DIM_NUMBERS,
            preferred_element_type=jnp.float32,
        )
        scores = scores * self.score_scale
        previous_max = jnp.where(is_first, NO_KEY_MAX, refs.row_max[rows])
        row_max = jnp.maximum(
            previous_max,
            lane_broadcast(scores.max(axis=1, keepdims=True), NUM_LANES),
        )
        rescale = jnp.exp2(previous_max - row_max)
        probabilities = jnp.exp2(
            scores - lane_broadcast(row_max, self.block_kv)
        )
        refs.row_max[rows] = row_max
        previous_sum = jnp.where(is_first, 0.0, refs.row_sum[rows])
        # Note (david): row_sum keeps a partial sum per lane, which the q
        # block's finish adds up once.
        refs.row_sum[rows] = rescale * previous_sum + _lane_sums(probabilities)
        # Note (david): probabilities round to fp8 for P @ V, as the sparse
        # MLA's do; each is at most 1, so none overflows.
        return rows, probabilities.astype(FP8_DTYPE), rescale

    def accumulate(
        self,
        rows: pl.Slice,
        probabilities: jax.Array,
        rescale: jax.Array,
        is_first: jax.Array,
    ) -> None:
        refs = self.refs
        pv = jnp.dot(
            probabilities,
            refs.keys[:, pl.ds(0, LATENT_DIM)],
            preferred_element_type=jnp.float32,
        )
        previous = jnp.where(is_first, 0.0, refs.accumulator[rows])
        refs.accumulator[rows] = (
            lane_broadcast(rescale, LATENT_DIM) * previous + pv
        )

    def attend(self, kv_block: jax.Array, query_slot: jax.Array) -> None:
        """The q block's update by the staged kv block, each row group's
        P @ V after the next group's softmax."""
        is_first = kv_block == 0
        pending = self.softmax(query_slot, 0, is_first)
        for group in range(1, self.row_groups):
            current = self.softmax(query_slot, group, is_first)
            self.accumulate(*pending, is_first)
            pending = current
        self.accumulate(*pending, is_first)

    def stage_outputs(self, slot: jax.Array) -> None:
        """The q block's normalized rows into out_stage at slot, and their
        lse into lse_stage; a row that saw no key gets 0 and +inf."""
        refs = self.refs
        row_max = refs.row_max[...]
        row_sum = lane_broadcast(
            refs.row_sum[...].sum(axis=1, keepdims=True), NUM_LANES
        )
        has_keys = row_max > 0.5 * NO_KEY_MAX
        inverse_row_sum = jnp.where(has_keys, self.kv_scale / row_sum, 0.0)
        refs.out_stage[slot] = (
            (
                refs.accumulator[...]
                * lane_broadcast(inverse_row_sum, LATENT_DIM)
            )
            .reshape(self.block_q, self.num_heads, LATENT_DIM)
            .astype(refs.out_stage.dtype)
        )
        if refs.lse_stage is not None:
            lse = jnp.where(
                has_keys, (row_max + jnp.log2(row_sum)) * math.log(2.0), jnp.inf
            )
            # Note (david): a token's lse leaves as one lane-dense row, lane h
            # its head h.
            shape = (self.num_rows, NUM_LANES)
            row = lax.broadcasted_iota(jnp.int32, shape, 0)
            head = row - row // self.num_heads * self.num_heads
            is_head_lane = lax.broadcasted_iota(jnp.int32, shape, 1) == head
            refs.lse_stage[slot] = (
                jnp.where(is_head_lane, lse, 0.0)
                .reshape(self.block_q, self.num_heads, NUM_LANES)
                .sum(axis=1)
            )

    def wait_output(self, ordinal: jax.Array) -> None:
        for copy in self.output_copies(
            self.query_block(ordinal), lax.rem(ordinal, NUM_QUERY_SLOTS)
        ):
            copy.wait()

    def finish_query_block(self, block: QueryBlock, slot: jax.Array) -> None:
        previous_ordinal = block.ordinal - NUM_QUERY_SLOTS
        pl.when(previous_ordinal >= 0)(
            functools.partial(self.wait_output, previous_ordinal)
        )
        self.stage_outputs(slot)
        for copy in self.output_copies(block, slot):
            copy.start()

    def start_query_block(self, block: QueryBlock, slot: jax.Array) -> None:
        """Wait for the q block's copies, put its rope queries in their
        lanes, and start the next q block's copies."""
        refs = self.refs
        for copy in self.query_copies(block, slot):
            copy.wait()
        # Note (david): the rope queries arrive heads minor and narrower than a
        # lane tile, so no copy window can address their lanes; a transpose and
        # vector stores place them.
        q_pe = jnp.swapaxes(refs.q_pe_stage[slot].astype(jnp.float32), 1, 2)
        refs.query_buffer[slot, :, :, pl.ds(LATENT_DIM, ROPE_DIM)] = (
            q_pe.astype(refs.query_buffer.dtype)
        )
        next_ordinal = block.ordinal + 1

        @pl.when(next_ordinal < self.num_q_blocks)
        def _start_next_query_copies():
            for copy in self.query_copies(
                self.query_block(next_ordinal),
                lax.rem(next_ordinal, NUM_QUERY_SLOTS),
            ):
                copy.start()

    def run_item(self, item: jax.Array, carry: tuple) -> tuple:
        block_ordinal, kv_block, prefetch_ordinal, prefetch_kv_block = carry
        prefetch_item = item + NUM_KV_STAGES - 1
        prefetch_block = self.query_block(prefetch_ordinal)

        # Note (david): the copies go to the slot the previous item has just
        # read, so they start after its last load.
        @pl.when(prefetch_item < self.num_items)
        def _prefetch():
            self.start_kv_copies(
                prefetch_block,
                prefetch_kv_block,
                lax.rem(prefetch_item, NUM_KV_STAGES),
            )

        prefetch_ordinal, prefetch_kv_block = self.next_item(
            prefetch_block, prefetch_kv_block
        )
        block = self.query_block(block_ordinal)
        query_slot = lax.rem(block_ordinal, NUM_QUERY_SLOTS)
        kv_slot = lax.rem(item, NUM_KV_STAGES)
        pl.when(kv_block == 0)(
            functools.partial(self.start_query_block, block, query_slot)
        )
        self.wait_kv_copies(block, kv_block, kv_slot)
        self.stage_keys(block, kv_block, kv_slot, query_slot)
        self.attend(kv_block, query_slot)
        pl.when(kv_block == block.num_kv_blocks - 1)(
            functools.partial(self.finish_query_block, block, query_slot)
        )
        return (
            *self.next_item(block, kv_block),
            prefetch_ordinal,
            prefetch_kv_block,
        )

    def zero_padding(self) -> None:
        """Zero the output rows past cu_q_lens[B], which no q block holds."""
        refs = self.refs
        num_tokens = refs.out.shape[0]
        first_padding = refs.cu_q_lens[refs.kv_lens.shape[0]]
        num_chunks = (
            num_tokens - first_padding + self.block_q - 1
        ) // self.block_q

        @pl.when(num_chunks > 0)
        def _zero():
            refs.out_stage[0] = jnp.zeros(
                refs.out_stage.shape[1:], refs.out_stage.dtype
            )

            def _chunk(chunk, carry):
                start = first_padding + chunk * self.block_q
                count = jnp.minimum(self.block_q, num_tokens - start)
                copy = pltpu.make_async_copy(
                    refs.out_stage.at[0, pl.ds(0, count)],
                    refs.out.at[pl.ds(start, count)],
                    refs.sems.at[SEM_OUTPUT, 0],
                )
                copy.start()
                copy.wait()
                return carry

            lax.fori_loop(0, num_chunks, _chunk, 0)

    def run(self) -> None:
        refs = self.refs
        # Note (david): token t's rows hold FLAG_QUERY in mask lane t and
        # zeros in the others, the same in every q block; the scratch's
        # garbage there could be NaN, which a zero key lane turns into NaN.
        shape = refs.query_buffer.shape[1:-1] + (MAX_BLOCK_Q,)
        flags = jnp.where(
            lax.broadcasted_iota(jnp.int32, shape, 0)
            == lax.broadcasted_iota(jnp.int32, shape, 2),
            FLAG_QUERY,
            0.0,
        ).astype(refs.query_buffer.dtype)
        for slot in range(NUM_QUERY_SLOTS):
            refs.query_buffer[
                slot, :, :, pl.ds(LATENT_DIM + ROPE_LANES, MAX_BLOCK_Q)
            ] = flags

        @pl.when(self.num_q_blocks > 0)
        def _start_first_query_copies():
            for copy in self.query_copies(self.query_block(jnp.int32(0)), 0):
                copy.start()

        prefetch_ordinal = jnp.int32(0)
        prefetch_kv_block = jnp.int32(0)
        for item in range(NUM_KV_STAGES - 1):
            prefetch_block = self.query_block(prefetch_ordinal)
            pl.when(item < self.num_items)(
                functools.partial(
                    self.start_kv_copies,
                    prefetch_block,
                    prefetch_kv_block,
                    item,
                )
            )
            prefetch_ordinal, prefetch_kv_block = self.next_item(
                prefetch_block, prefetch_kv_block
            )
        lax.fori_loop(
            0,
            self.num_items,
            self.run_item,
            (jnp.int32(0), jnp.int32(0), prefetch_ordinal, prefetch_kv_block),
        )
        # Note (david): the kernel must not exit with copies in flight; each
        # earlier output was waited for by the q block that reused its slot.
        for distance in range(NUM_QUERY_SLOTS, 0, -1):
            ordinal = self.num_q_blocks - distance
            pl.when(ordinal >= 0)(functools.partial(self.wait_output, ordinal))
        self.zero_padding()


def _start_all(copies: tuple) -> None:
    for copy in copies:
        copy.start()


def _wait_all(copies: tuple) -> None:
    for copy in copies:
        copy.wait()


def _lane_sums(values: jax.Array) -> jax.Array:
    """values (rows, n * NUM_LANES) summed pairwise over its lane tiles into
    (rows, NUM_LANES): each lane keeps its own partial sum."""
    lane_tiles = [
        values[:, i : i + NUM_LANES]
        for i in range(0, values.shape[1], NUM_LANES)
    ]
    while len(lane_tiles) > 1:
        pairs = [a + b for a, b in zip(lane_tiles[0::2], lane_tiles[1::2])]
        lane_tiles = pairs + lane_tiles[2 * len(pairs) :]
    return lane_tiles[0]


def split_refs(refs: tuple, has_bitmap: bool, return_lse: bool) -> KernelRefs:
    """The pallas_call's refs, in dsa_masked_dense_pallas's order, as
    KernelRefs; the optional ones None when absent."""
    refs = list(refs)
    taken = []
    for name in KernelRefs._fields:
        is_absent = (name in ("bitmap", "bitmap_stage") and not has_bitmap) or (
            name in ("lse", "lse_stage") and not return_lse
        )
        taken.append(None if is_absent else refs.pop(0))
    assert not refs, f"{len(refs)} refs left over."
    return KernelRefs(*taken)


def dsa_masked_dense_kernel(*refs: jax.Array, params: KernelParams) -> None:
    kernel_refs = split_refs(refs, params.bitmap_rows > 0, params.return_lse)
    MaskedDenseKernel(params, kernel_refs).run()


def masked_dense_vmem_bytes(
    config: MaskedDenseConfig, *, num_heads: int, rows_per_token: int
) -> int:
    """VMEM bytes of the masked-dense scratch at config, plus a row group's
    score temporaries; rows_per_token is a token's bitmap rows, or 0."""
    block_q, block_kv, row_groups = config
    num_rows = block_q * num_heads
    group_rows = num_rows // row_groups
    staged_rows = block_q * pl.cdiv(num_heads, BF16_TILE_ROWS) * BF16_TILE_ROWS
    q_rows = pl.cdiv(block_q, NUM_SUBLANES) * NUM_SUBLANES
    ring_bytes = (
        NUM_KV_STAGES
        * (block_kv + block_kv // ROPE_TOKENS_PER_ROW)
        * NOPE_WORDS
        * WORD_BYTES
    )
    key_bytes = (
        block_kv * KEY_ROW_DIM * FP8_BYTES
        + block_kv // QUAD_TOKENS * NUM_LANES * WORD_BYTES
    )
    stage_bytes = (
        NUM_QUERY_SLOTS
        * staged_rows
        * (KEY_ROW_DIM + NUM_LANES + LATENT_DIM)
        * BF16_BYTES
    )
    bitmap_bytes = (
        NUM_QUERY_SLOTS * block_q * rows_per_token * NUM_LANES * WORD_BYTES
    )
    state_bytes = num_rows * (2 * NUM_LANES + LATENT_DIM) * F32_BYTES
    lse_bytes = NUM_QUERY_SLOTS * q_rows * NUM_LANES * F32_BYTES
    # Note (david): f32 scores, their exponentials and fp8 probabilities of
    # two row groups are live at once.
    score_bytes = 2 * group_rows * block_kv * (2 * F32_BYTES + FP8_BYTES)
    return (
        ring_bytes
        + key_bytes
        + stage_bytes
        + bitmap_bytes
        + state_bytes
        + lse_bytes
        + score_bytes
    )


def default_masked_dense_config(
    *,
    num_heads: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    rows_per_token: int,
) -> MaskedDenseConfig:
    """The default blocks of a call: a q block of about TARGET_ROWS score rows,
    halved past the VMEM budget, and a kv block of about TARGET_ITEM_SCORES
    scores."""
    block_q = min(
        max(1, TARGET_ROWS // num_heads), MAX_BLOCK_Q, next_pow2(max_seqlen_q)
    )
    vmem_budget = VMEM_BUDGET_FRACTION * vmem_limit_bytes()
    while True:
        block_kv = min(
            max(MIN_BLOCK_KV, TARGET_ITEM_SCORES // (block_q * num_heads)),
            MAX_BLOCK_KV,
            max(NUM_LANES, next_pow2(max_seqlen_k)),
        )
        config = MaskedDenseConfig(
            block_q, block_kv, max(1, block_q * num_heads // GROUP_ROWS)
        )
        needed = masked_dense_vmem_bytes(
            config, num_heads=num_heads, rows_per_token=rows_per_token
        )
        if needed <= vmem_budget:
            return config
        elif block_q > 1:
            block_q //= 2
        else:
            raise ValueError(
                f"The masked-dense MLA needs {needed} bytes of VMEM at a"
                f" one-token q block, over {vmem_budget}."
            )


class MaskedDensePlan(NamedTuple):
    """A call's blocks, and its schedule: q blocks, each with its kv blocks
    up to its last token, a pure function of cu_q_lens and kv_lens."""

    config: MaskedDenseConfig
    max_q_blocks: int
    schedule: jax.Array
    request_first_q_block: jax.Array


def masked_dense_plan(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    *,
    num_tokens: int,
    max_seqlen_q: int,
    config: MaskedDenseConfig,
) -> MaskedDensePlan:
    """The plan of a call of num_tokens tokens at config."""
    batch = kv_lens.shape[0]
    block_q = config.block_q
    if (
        batch >= 1 << (WORD_BITS - 1 - REQUEST_SHIFT)
        or pl.cdiv(max_seqlen_q, block_q) > Q_BLOCK_MASK
    ):
        raise ValueError(
            f"batch {batch} or {pl.cdiv(max_seqlen_q, block_q)} q blocks a"
            " request overflow a schedule entry."
        )
    # Note (david): a request of n tokens takes at most n // block_q + 1 q
    # blocks.
    max_q_blocks = max(
        1,
        min(
            num_tokens // block_q + min(batch, num_tokens),
            batch * pl.cdiv(max_seqlen_q, block_q),
        ),
    )
    schedule, request_first_q_block = mla_prefill_schedule(
        cu_q_lens,
        kv_lens,
        block_q=block_q,
        block_kv=config.block_kv,
        max_q_blocks=max_q_blocks,
        causal=True,
    )
    return MaskedDensePlan(
        config, max_q_blocks, schedule, request_first_q_block
    )


def scratch_shapes(params: KernelParams, q_dtype: jnp.dtype) -> list:
    """The kernel's scratch, in KernelRefs order."""
    block_q, block_kv, _ = params.config
    num_heads = params.num_heads
    num_rows = block_q * num_heads
    shapes = [
        pltpu.VMEM((NUM_KV_STAGES, block_kv, NOPE_WORDS), jnp.int32),
        pltpu.VMEM(
            (NUM_KV_STAGES, block_kv // ROPE_TOKENS_PER_ROW, NOPE_WORDS),
            jnp.int32,
        ),
        pltpu.VMEM((block_kv, KEY_ROW_DIM), FP8_DTYPE),
        pltpu.VMEM((block_kv // QUAD_TOKENS, NUM_LANES), jnp.int32),
        pltpu.VMEM((NUM_QUERY_SLOTS, block_q, num_heads, KEY_ROW_DIM), q_dtype),
        pltpu.VMEM((NUM_QUERY_SLOTS, block_q, ROPE_DIM, num_heads), q_dtype),
        pltpu.VMEM((NUM_QUERY_SLOTS, block_q, num_heads, LATENT_DIM), q_dtype),
        pltpu.VMEM((num_rows, NUM_LANES), jnp.float32),
        pltpu.VMEM((num_rows, NUM_LANES), jnp.float32),
        pltpu.VMEM((num_rows, LATENT_DIM), jnp.float32),
        pltpu.SemaphoreType.DMA((NUM_SEMS, NUM_KV_STAGES)),
    ]
    if params.bitmap_rows:
        shapes.append(
            pltpu.VMEM(
                (NUM_QUERY_SLOTS, block_q * params.bitmap_rows, NUM_LANES),
                jnp.int32,
            )
        )
    if params.return_lse:
        shapes.append(
            pltpu.VMEM((NUM_QUERY_SLOTS, block_q, NUM_LANES), jnp.float32)
        )
    return shapes


def dsa_masked_dense_pallas(
    q_nope: jax.Array,
    q_pe_t: jax.Array,
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    bitmap: jax.Array | None,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
    kv_scale: jax.Array,
    *,
    plan: MaskedDensePlan,
    sm_scale: float,
    top_k: int,
    num_slabs: int,
    return_lse: bool,
) -> tuple[jax.Array, ...]:
    """The masked-dense pallas_call on checked operands.

    Args:
      q_nope: (T, H, LATENT_DIM) bf16.
      q_pe_t: (T, ROPE_DIM, H) bf16, the rope queries heads minor.
      nope_cache: (num_pages, page_size, NOPE_WORDS) int32 split cache.
      rope_cache: (num_pages, page_size / 8, NOPE_WORDS) int32.
      bitmap: dsa_selection_bitmap_pallas's (T * bitmap_rows(num_slabs),
        NUM_LANES) int32, or None when no request holds more than top_k
        positions.
      cu_q_lens, kv_lens, block_table: as dsa_masked_dense takes them.
      kv_scale: (1,) f32.
      plan: masked_dense_plan of the call.
      sm_scale: positive scale of q_nope . latent + q_pe . rope.
      top_k: the selection's width.
      num_slabs: the bitmap's slabs a token.
      return_lse: also return the lse blocks.

    Returns:
      (out, [lse blocks]): out (T, H, LATENT_DIM) bf16, and q block j's lse
      of token i at [j, i, head].
    """
    num_heads = q_nope.shape[1]
    page_size, pages_per_seq = nope_cache.shape[1], block_table.shape[1]
    config, max_q_blocks = plan.config, plan.max_q_blocks
    params = KernelParams(
        config=config,
        num_heads=num_heads,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        max_q_blocks=max_q_blocks,
        top_k=top_k,
        num_slabs=num_slabs,
        bitmap_rows=0 if bitmap is None else bitmap_rows(num_slabs),
        sm_scale_log2=sm_scale * LOG2E,
        return_lse=return_lse,
    )
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    inputs = [kv_scale, q_nope, q_pe_t, nope_cache, rope_cache]
    in_specs = [pl.BlockSpec(memory_space=pltpu.SMEM)] + [hbm_spec] * 4
    if bitmap is not None:
        inputs.append(bitmap)
        in_specs.append(hbm_spec)
    out_shapes = [jax.ShapeDtypeStruct(q_nope.shape, q_nope.dtype)]
    if return_lse:
        out_shapes.append(
            jax.ShapeDtypeStruct(
                (max_q_blocks, config.block_q, NUM_LANES), jnp.float32
            )
        )
    scalar_prefetches = (
        plan.schedule,
        cu_q_lens,
        kv_lens,
        block_table.reshape(-1),
    )
    outputs = pl.pallas_call(
        functools.partial(dsa_masked_dense_kernel, params=params),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            grid=(1,),
            in_specs=in_specs,
            out_specs=[hbm_spec] * len(out_shapes),
            scratch_shapes=scratch_shapes(params, q_nope.dtype),
        ),
        out_shape=out_shapes,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_bytes(),
        ),
        name="dsa_masked_dense",
    )(*scalar_prefetches, *inputs)
    return tuple(outputs)
