"""Exact top-k positions of score rows on SparseCore.

A row of length L <= k selects every valid position. A row of length up to
direct_max selects from all its scores. A longer row uses a sampled threshold
tau; it is streamed again with tau = -inf if fewer than k scores reach tau.
Selection is a radix select on int32 keys of the f32 bits, finished by a
vector sort. When a core has fewer rows than subcores, a group of subcores
shares a row (see run_parts).
"""

import dataclasses
import enum
import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

WORD_BITS = 32
# XOR with this after an arithmetic shift by WORD_BITS - 1 maps f32 bits to
# an int32 key that orders like the floats; the map is its own inverse.
KEY_FLIP_MASK = 0x7FFFFFFF
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
# Two bf16 quiet NaNs: an f32 NaN in every layout, which no score is >= to.
NAN_BITS = 0x7FC07FC0
# A bf16 value is the high half of the f32 with the same value.
HALF_BITS = 16
HIGH_HALF_MASK = -(1 << HALF_BITS)
# Key of -inf: the smallest key of a non-NaN score.
NEG_INF_KEY = (0xFF800000 - 2**32) ^ KEY_FLIP_MASK
# The first selection level spreads the candidates' key range over
# SELECT_BUCKETS buckets; tie refinement uses up to that many.
SELECT_BUCKETS_LOG2 = 10
SELECT_BUCKETS = 1 << SELECT_BUCKETS_LOG2
# All-keys rows bucket the full key range by its top bits.
DIRECT_SHIFT = WORD_BITS - SELECT_BUCKETS_LOG2
# The sample threshold is located on the top SAMPLE_LEVELS * SAMPLE_BITS key
# bits.
SAMPLE_BITS = 8
SAMPLE_LEVELS = 2
SAMPLE_BUCKETS = 1 << SAMPLE_BITS
# TensorCore tiles are (TILE_ROWS, TILE_COLUMNS) words; a copy of a row of
# a TensorCore-tiled array starts at and spans a multiple of TILE_COLUMNS.
TILE_ROWS = 8
TILE_COLUMNS = 128
# A sample is a power-of-two number of pieces of a layout's sample piece
# words, 2**MIN_SAMPLE_WORDS_LOG2 to 2**MAX_SAMPLE_WORDS_LOG2 words in all.
FLAT_SAMPLE_PIECE_WORDS = 32
MIN_SAMPLE_WORDS_LOG2 = 9
MAX_SAMPLE_WORDS_LOG2 = 11
MAX_SAMPLE_WORDS = 1 << MAX_SAMPLE_WORDS_LOG2
# log2 of a row's sample words per 2 k of its scores, before the clip to the
# range above.
SAMPLE_DENSITY_LOG2 = 7
# Margin of the sample threshold, in standard deviations of the sampled count.
SAMPLE_SIGMAS = 4
# Ring of chunk buffers that stream the score rows from HBM.
NUM_CHUNK_BUFFERS = 4
WALK_UNROLL = 8
SMALL_UNROLL = 4
# The partition walk keeps two masks per vector; a deeper unroll spills them.
PARTITION_UNROLL = 4
# VMEM words of a chunk buffer and of a candidate list, as fractions of a
# subcore's VMEM words.
CHUNK_VMEM_SHIFT = 4
CAPACITY_VMEM_NUMERATOR = 5
CAPACITY_VMEM_SHIFT = 5
# Rows up to this many scores are selected from all their keys.
DIRECT_MAX = 8192
# A part leaves at most PART_CAPACITY_K * k candidates in shared VMEM.
PART_CAPACITY_K = 2
# Candidates move through shared VMEM in blocks of this many words. A part
# publishes PART_LISTS lists (keys, positions); the leader merges the parts
# through MERGE_SLOTS ring slots.
SHARED_BLOCK_WORDS = 256
PART_LISTS = 2
MERGE_SLOTS = 2
# Lanes of a part's header vector: its candidate count, smallest key and
# largest key.
HEADER_COUNT, HEADER_BASE, HEADER_TOP = range(3)
# A subcore's VMEM holds the lengths of at most this many of its rows at a
# time, so that its VMEM use does not grow with the rows of a call.
MAX_LENGTHS_BLOCK = 512
WORD_BYTES = 4


class ScoreLayout(enum.Enum):
    """How the kernel reads the score rows in place.

    FLAT: (T, S / TILE_COLUMNS, TILE_COLUMNS) f32, S a multiple of
      TILE_ROWS * TILE_COLUMNS, so each row is a contiguous run of S words;
      the operands use SparseCore tiling, so the (T, k) output is row-major.
    ROWS: (T, N) f32 in TensorCore tiling, N a multiple of TILE_COLUMNS; a
      chunk is a strided copy of one row.
    ROW_PAIRS: (T, N) bf16 in TensorCore tiling, T even and N a multiple of
      TILE_COLUMNS, read as (T / 2, N) words: the low half of word [p, n]
      holds row 2 p, the high half row 2 p + 1.
    """

    FLAT = "flat"
    ROWS = "rows"
    ROW_PAIRS = "row_pairs"


class OutputTransform(enum.Enum):
    """The map from a row's selected positions p to its output; -1 stays.

    OFFSET: p + base, base the row's int32 value.
    PAGE_TABLE: page * page_size + p % page_size for the row's page table
      entry p // page_size. base is the row's table row when
      TopkGeometry.holds_table_row, else the row's first entry in a flat table.
    """

    NONE = "none"
    OFFSET = "offset"
    PAGE_TABLE = "page_table"


@dataclasses.dataclass(frozen=True)
class TopkGeometry:
    """Static sizes of one SparseCore top-k compile. Row lengths are clamped
    to [0, num_columns]; without row lengths every row has num_columns."""

    num_rows: int
    row_stride: int
    k: int
    lanes: int
    num_cores: int
    num_subcores: int
    chunk_words: int
    capacity: int
    direct_max: int
    layout: ScoreLayout
    num_columns: int
    has_row_lengths: bool
    transform: OutputTransform
    page_shift: int
    pages_per_row: int

    @property
    def num_workers(self) -> int:
        return self.num_cores * self.num_subcores

    @property
    def is_split(self) -> bool:
        """Whether some SparseCore has fewer rows than subcores."""
        return self.num_rows < self.num_workers

    @property
    def uses_tensor_core_tiling(self) -> bool:
        return self.layout is not ScoreLayout.FLAT

    @property
    def sample_piece_words(self) -> int:
        if self.uses_tensor_core_tiling:
            return TILE_COLUMNS
        return FLAT_SAMPLE_PIECE_WORDS

    @property
    def has_row_bases(self) -> bool:
        return self.transform is not OutputTransform.NONE

    @property
    def num_inputs(self) -> int:
        """Kernel inputs: [row lengths], [row bases], [page table], scores."""
        return (
            1
            + self.has_row_lengths
            + self.has_row_bases
            + (self.transform is OutputTransform.PAGE_TABLE)
        )

    @property
    def table_words(self) -> int:
        """Words of a page table row in whole tiles."""
        return -(-self.pages_per_row // TILE_COLUMNS) * TILE_COLUMNS

    @property
    def holds_table_row(self) -> bool:
        """Whether a row's page table row is loaded whole into the
        candidate keys, free once the row is selected."""
        return (
            self.transform is OutputTransform.PAGE_TABLE
            and self.table_words <= self.capacity
        )

    @property
    def k_words(self) -> int:
        """Words of k positions in whole vectors."""
        return -(-self.k // self.lanes) * self.lanes

    @property
    def part_capacity(self) -> int:
        """Candidates a part leaves in shared VMEM: whole blocks."""
        words = PART_CAPACITY_K * self.k
        return -(-words // SHARED_BLOCK_WORDS) * SHARED_BLOCK_WORDS

    @property
    def ring_words(self) -> int:
        """Chunk buffers; in split mode also the merge slots."""
        chunk_words = NUM_CHUNK_BUFFERS * self.chunk_words
        if not self.is_split:
            return chunk_words
        merge_words = MERGE_SLOTS * PART_LISTS * self.part_capacity
        return max(chunk_words, merge_words)

    @property
    def headers_words(self) -> int:
        """Shared VMEM words of the subcores' header vectors."""
        return self.num_subcores * self.lanes

    @property
    def shared_words(self) -> int:
        """Shared VMEM words: headers, then each subcore's keys and
        positions."""
        lists_words = PART_LISTS * self.part_capacity
        return self.headers_words + self.num_subcores * lists_words

    @property
    def lengths_block(self) -> int:
        """Rows of a worker whose lengths its VMEM holds at a time: all of
        them in whole vectors, up to MAX_LENGTHS_BLOCK."""
        worker_rows = -(-self.num_rows // self.num_workers)
        whole_vectors = -(-worker_rows // self.lanes) * self.lanes
        return min(MAX_LENGTHS_BLOCK, whole_vectors)

    @property
    def lengths_words(self) -> int:
        """VMEM words of a block of row lengths: its rows and the one after
        them, read as one vector."""
        return self.lengths_block + self.lanes

    @property
    def bases_offset(self) -> int:
        """Word of row_values where the block of row bases starts."""
        return self.lengths_words if self.has_row_lengths else 0

    @property
    def row_values_words(self) -> int:
        """VMEM words of a block of row lengths, then one of row bases; one
        block's words when there are neither."""
        blocks = max(1, self.has_row_lengths + self.has_row_bases)
        return blocks * self.lengths_words

    @property
    def num_lengths_blocks(self) -> int:
        """Blocks of lengths_block rows that cover a worker's rows."""
        return -(-self.num_rows // (self.num_workers * self.lengths_block))

    @property
    def worker_rows(self) -> int:
        """Words of a worker's run of lengths in the worker-major lengths:
        whole blocks of its rows, then a vector of zeros that the last
        block's read runs into."""
        return self.num_lengths_blocks * self.lengths_block + self.lanes


def sparse_core_vmem_words() -> int:
    """32-bit words of a SparseCore vector subcore's VMEM on this chip."""
    return plsc.get_sparse_core_info().vmem_capacity_bytes // WORD_BYTES


def topk_geometry(
    num_rows: int,
    row_stride: int,
    k: int,
    vmem_words: int,
    *,
    layout: ScoreLayout,
    num_columns: int,
    has_row_lengths: bool,
    transform: OutputTransform = OutputTransform.NONE,
    page_shift: int = 0,
    pages_per_row: int = 0,
) -> TopkGeometry:
    """Sizes for this chip's SparseCore: chunk and candidate list words scale
    with vmem_words, the top-k's share of a subcore's VMEM. row_stride is a
    multiple of TILE_COLUMNS."""
    info = plsc.get_sparse_core_info()
    chunk_words = min(vmem_words >> CHUNK_VMEM_SHIFT, row_stride)
    capacity = (vmem_words * CAPACITY_VMEM_NUMERATOR) >> CAPACITY_VMEM_SHIFT
    return TopkGeometry(
        num_rows=num_rows,
        row_stride=row_stride,
        k=k,
        lanes=info.num_lanes,
        num_cores=info.num_cores,
        num_subcores=info.num_subcores,
        chunk_words=chunk_words,
        capacity=capacity,
        direct_max=min(DIRECT_MAX, capacity),
        layout=layout,
        num_columns=num_columns,
        has_row_lengths=has_row_lengths,
        transform=transform,
        page_shift=page_shift,
        pages_per_row=pages_per_row,
    )


def check_k(geometry: TopkGeometry) -> None:
    """Fails unless k fits the candidate list: twice over, and next to a
    chunk."""
    max_k = min(
        geometry.capacity // PART_CAPACITY_K // geometry.lanes * geometry.lanes,
        geometry.capacity - geometry.chunk_words,
    )
    if not 1 <= geometry.k <= max_k:
        raise ValueError(
            f"k must be in [1, {max_k}] on this chip; got {geometry.k}."
        )


# Phases of a row's work.
STREAM, SELECT, MERGE, RETRY, PUBLISH, BARRIER, DONE = range(7)


class _Row(NamedTuple):
    """Runtime facts of the row a subcore works on."""

    local_row: jax.Array
    row: jax.Array
    length: jax.Array
    is_direct: jax.Array
    is_sampled: jax.Array
    is_leader: jax.Array
    decode_shift: jax.Array


class _Job(NamedTuple):
    """Loop state of a row's phases."""

    phase: jax.Array
    stream: tuple
    chunk: jax.Array
    end: jax.Array
    count: jax.Array
    tau_key: jax.Array
    base: jax.Array
    top: jax.Array
    resume: jax.Array
    member: jax.Array
    issued: jax.Array
    is_part: jax.Array
    is_cut: jax.Array


def monotone_key(bits: jax.Array) -> jax.Array:
    """int32 key of f32 bits that orders like the floats (NaN excluded)."""
    sign = lax.shift_right_arithmetic(bits, WORD_BITS - 1)
    return bits ^ (sign & KEY_FLIP_MASK)


def bit_length(value: jax.Array) -> jax.Array:
    """Bits needed for an int32 scalar read as unsigned."""
    length = jnp.int32(0)
    step = WORD_BITS // 2
    while step:
        is_longer = lax.shift_right_logical(value, length + step) != 0
        length = jnp.where(is_longer, length + step, length)
        step //= 2
    return length + (value != 0).astype(jnp.int32)


def decode_shift(layout: ScoreLayout, row: jax.Array) -> jax.Array:
    """Left shift that moves row's half of a ROW_PAIRS word to the top."""
    if layout is ScoreLayout.ROW_PAIRS:
        return HALF_BITS * (1 - (row & 1))
    return jnp.int32(0)


def start_or_wait(copy, is_start: bool) -> None:
    if is_start:
        copy.start()
    else:
        copy.wait()


def integer_sqrt(value: jax.Array, bits: int) -> jax.Array:
    """floor(sqrt(value)) of an int32 scalar below 4**bits."""
    root = jnp.int32(0)
    for bit in reversed(range(bits)):
        trial = root | (1 << bit)
        root = jnp.where(trial * trial <= value, trial, root)
    return root


class _TopkKernel:
    """The kernel body of one subcore; methods share its refs and sizes."""

    def __init__(
        self,
        geometry: TopkGeometry,
        scratch: tuple,
        *,
        lengths_hbm,
        bases_hbm,
        table_hbm,
        scores_bits,
        out_hbm,
    ):
        self.geometry = geometry
        self.lanes = geometry.lanes
        self.lanes_log2 = geometry.lanes.bit_length() - 1
        (
            self.row_values,
            self.ring,
            self.cand_key,
            self.cand_idx,
            self.ties,
            self.hist,
            self.totals,
            self.out_stage,
            self.sample,
            self.headers,
            self.ring_sems,
            self.sample_sem,
            self.out_sem,
            self.merge_sems,
            self.table_sem,
            self.shared,
        ) = scratch
        self.lengths_hbm = lengths_hbm
        self.bases_hbm = bases_hbm
        self.table_hbm = table_hbm
        self.scores_bits = scores_bits
        self.out_hbm = out_hbm
        self.lane_iota = lax.iota(jnp.int32, self.lanes)
        self.core = lax.axis_index("core")
        self.subcore = lax.axis_index("subcore")
        # Rows alternate between cores so that both get work at small T.
        self.core_rows = (
            geometry.num_rows - self.core + geometry.num_cores - 1
        ) // geometry.num_cores
        self.num_local_rows = (
            self.core_rows - self.subcore + geometry.num_subcores - 1
        ) // geometry.num_subcores
        self.part = jnp.int32(0)
        self.num_parts = jnp.int32(1)
        self.is_parts = False
        # Local row i is global row column + num_workers * i, with its length at
        # row_values[i - first_row] in the loaded block.
        self.column = self.core + geometry.num_cores * self.subcore
        self.first_row = jnp.int32(0)
        self.rows_end = self.num_local_rows

    def row_of(self, i):
        """Global row of local row i."""
        if self.is_parts:
            return self.column
        return self.column + self.geometry.num_workers * i

    def load_lengths(self, first_row):
        """Loads the lengths and bases of local rows [first_row, first_row +
        lengths_block] from the column's runs of the worker-major arrays."""
        geometry = self.geometry
        start = pl.multiple_of(
            self.column * geometry.worker_rows + first_row, self.lanes
        )
        if geometry.has_row_lengths:
            pltpu.sync_copy(
                self.lengths_hbm.at[pl.ds(start, geometry.lengths_words)],
                self.row_values.at[pl.ds(0, geometry.lengths_words)],
            )
        if geometry.has_row_bases:
            pltpu.sync_copy(
                self.bases_hbm.at[pl.ds(start, geometry.lengths_words)],
                self.row_values.at[
                    pl.ds(geometry.bases_offset, geometry.lengths_words)
                ],
            )
        self.first_row = first_row

    # ---- small helpers -------------------------------------------------

    def scalar_at(self, ref, index):
        return ref[pl.ds(index, self.lanes)][0]

    def decode(self, bits, shift):
        """f32 bits of the row's scores in score words."""
        if self.geometry.layout is ScoreLayout.ROW_PAIRS:
            return lax.shift_left(bits, shift) & HIGH_HALF_MASK
        return bits

    def score_words(self, row, offset, size):
        """HBM words [offset, offset + size) of a row's scores."""
        geometry = self.geometry
        if geometry.layout is ScoreLayout.FLAT:
            return self.scores_bits.at[
                pl.ds(row * geometry.row_stride + offset, size)
            ]
        if geometry.layout is ScoreLayout.ROW_PAIRS:
            row = lax.shift_right_logical(row, 1)
        return self.scores_bits.at[row, pl.ds(offset, size)]

    def vectors(self, count):
        return lax.shift_right_logical(
            count + (self.lanes - 1), self.lanes_log2
        )

    def for_vectors(self, count, body, carry, unroll=WALK_UNROLL):
        """Runs body(v, mask, carry) over the vectors of [0, count); mask is
        None for full vectors and the valid lanes of the last partial one."""
        full = lax.shift_right_logical(count, self.lanes_log2)
        carry = plsc.parallel_loop(0, full, unroll=unroll, carry=carry)(
            lambda v, c: body(v, None, c)
        )
        remainder = count - full * self.lanes
        return lax.cond(
            remainder > 0,
            lambda c: body(full, self.lane_iota < remainder, c),
            lambda c: c,
            carry,
        )

    def tie_positions(self, v, mask):
        """Vector v of the tie list; lanes past its end read position 0."""
        positions = self.ties[pl.ds(v * self.lanes, self.lanes)]
        return positions if mask is None else jnp.where(mask, positions, 0)

    def zero_hist(self, num_buckets):
        @plsc.parallel_loop(
            0,
            lax.shift_right_logical(num_buckets, self.lanes_log2),
            unroll=WALK_UNROLL,
        )
        def _(v):
            self.hist[pl.ds(v * self.lanes, self.lanes)] = jnp.zeros(
                (self.lanes,), jnp.int32
            )

    def count_buckets(self, buckets, mask):
        plsc.addupdate_scatter(
            self.hist, (buckets,), jnp.ones((self.lanes,), jnp.int32), mask=mask
        )

    def locate(self, num_buckets, rank):
        """Bucket b that holds the rank-th largest counted key, for a power of
        two num_buckets >= lanes**2; returns (b, keys above b, keys in b)."""
        lanes = self.lanes
        num_vectors = lax.shift_right_logical(num_buckets, self.lanes_log2)
        num_groups = lax.shift_right_logical(num_vectors, self.lanes_log2)

        @plsc.parallel_loop(0, num_vectors, unroll=SMALL_UNROLL)
        def _(v):
            counts = self.hist[pl.ds(v * lanes, lanes)]
            self.hist[pl.ds(v * lanes, lanes)] = plsc.cumsum(counts)

        def group_prefix(u, running):
            last_lanes = (u * lanes + self.lane_iota) * lanes + (lanes - 1)
            sums = plsc.cumsum(plsc.load_gather(self.hist, (last_lanes,)))
            sums = sums + running
            self.totals[pl.ds(u * lanes, lanes)] = sums
            return sums[lanes - 1]

        total = plsc.parallel_loop(0, num_groups, carry=jnp.int32(0))(
            group_prefix
        )
        limit = total - rank

        def count_at_most(u, count):
            at_most = self.totals[pl.ds(u * lanes, lanes)] <= limit
            return count + plsc.all_reduce_population_count(at_most)[0]

        vector = plsc.parallel_loop(0, num_groups, carry=jnp.int32(0))(
            count_at_most
        )
        before = jnp.where(
            vector > 0,
            self.scalar_at(self.totals, jnp.maximum(vector - 1, 0)),
            0,
        )
        sums = self.hist[pl.ds(vector * lanes, lanes)] + before
        lane = plsc.all_reduce_population_count(sums <= limit)[0]
        bucket = vector * lanes + lane
        through = self.scalar_at(self.hist, bucket) + before
        below = jnp.where(
            lane > 0,
            self.scalar_at(self.hist, jnp.maximum(bucket - 1, 0)) + before,
            before,
        )
        return bucket, total - through, through - below

    # ---- selection -----------------------------------------------------

    def select(self, count, base, shift, is_direct, emit_positions):
        """Writes to out_stage[:k] the positions of the k largest of count
        candidate keys: list positions if emit_positions, row positions
        otherwise. An all-keys row's first level was counted while streaming."""
        geometry = self.geometry
        lanes = self.lanes
        pl.when(~is_direct)(lambda: self.count_level1(count, base, shift))
        bucket, above, in_bucket = self.locate(SELECT_BUCKETS, geometry.k)
        quota = geometry.k - above
        take_whole = quota == in_bucket
        cut = bucket + 1 - take_whole.astype(jnp.int32)
        tie_bucket = jnp.where(take_whole, -1, bucket)
        maps_through = ~(is_direct | emit_positions)

        def output_of(positions, v):
            if v is None:
                mapped = plsc.load_gather(self.cand_idx, (positions,))
            else:
                mapped = self.cand_idx[pl.ds(v * lanes, lanes)]
            return jnp.where(maps_through, mapped, positions)

        def buckets(key):
            return lax.shift_right_logical(key - base, shift)

        emitted, num_ties = self.partition_candidates(
            count, buckets, cut, tie_bucket, output_of
        )
        quota = jnp.where(take_whole, 0, quota)
        return self.refine_ties(emitted, num_ties, quota, output_of)

    def count_level1(self, count, base, shift):
        """Histogram of the candidate keys' first-level buckets."""
        lanes = self.lanes
        self.zero_hist(SELECT_BUCKETS)

        def count_level(v, mask, carry):
            key = self.cand_key[pl.ds(v * lanes, lanes)]
            self.count_buckets(lax.shift_right_logical(key - base, shift), mask)
            return carry

        self.for_vectors(count, count_level, jnp.int32(0))

    def partition_candidates(self, count, buckets, cut, tie_bucket, output_of):
        """Emits the candidates whose bucket is at least cut to out_stage and
        lists those in tie_bucket; returns (emitted, ties)."""
        lanes = self.lanes

        def partition(v, mask, carry):
            emitted, num_ties, positions = carry
            b = buckets(self.cand_key[pl.ds(v * lanes, lanes)])
            selected = b >= cut
            tied = b == tie_bucket
            if mask is not None:
                selected = selected & mask
                tied = tied & mask
            plsc.store_compressed(
                self.out_stage.at[pl.ds(emitted, lanes)],
                output_of(positions, v),
                mask=selected,
            )
            plsc.store_compressed(
                self.ties.at[pl.ds(num_ties, lanes)], positions, mask=tied
            )
            return (
                emitted + plsc.all_reduce_population_count(selected)[0],
                num_ties + plsc.all_reduce_population_count(tied)[0],
                positions + lanes,
            )

        emitted, num_ties, _ = self.for_vectors(
            count,
            partition,
            (jnp.int32(0), jnp.int32(0), self.lane_iota),
            PARTITION_UNROLL,
        )
        return emitted, num_ties

    def tie_key_range(self, num_ties):
        """(min, max) key of the tie list's candidates."""
        lanes = self.lanes

        def bounds(v, mask, carry):
            low, high = carry
            key = plsc.load_gather(
                self.cand_key, (self.tie_positions(v, mask),)
            )
            if mask is not None:
                low = jnp.minimum(low, jnp.where(mask, key, INT32_MAX))
                high = jnp.maximum(high, jnp.where(mask, key, INT32_MIN))
                return low, high
            return jnp.minimum(low, key), jnp.maximum(high, key)

        low, high = self.for_vectors(
            num_ties,
            bounds,
            (
                jnp.full((lanes,), INT32_MAX, jnp.int32),
                jnp.full((lanes,), INT32_MIN, jnp.int32),
            ),
            SMALL_UNROLL,
        )
        return jnp.min(low), jnp.max(high)

    def refine_level(self, emitted, num_ties, quota, low, high, output_of):
        """Buckets the tie list over its key range [low, high] and emits the
        ties above the bucket of the quota-th largest; keeps that bucket."""
        lanes = self.lanes
        min_bits = 2 * self.lanes_log2
        bits = jnp.clip(bit_length(num_ties) + 1, min_bits, SELECT_BUCKETS_LOG2)
        shift = jnp.maximum(bit_length(high - low) - bits, 0)
        num_buckets = lax.shift_left(jnp.int32(1), bits)
        self.zero_hist(num_buckets)

        def tie_bucket(v, mask):
            positions = self.tie_positions(v, mask)
            key = plsc.load_gather(self.cand_key, (positions,))
            return positions, lax.shift_right_logical(key - low, shift)

        def count_level(v, mask, carry):
            self.count_buckets(tie_bucket(v, mask)[1], mask)
            return carry

        self.for_vectors(num_ties, count_level, jnp.int32(0), SMALL_UNROLL)
        bucket, above, _ = self.locate(num_buckets, quota)

        def partition(v, mask, carry):
            emitted, kept = carry
            positions, b = tie_bucket(v, mask)
            selected = b > bucket
            tied = b == bucket
            if mask is not None:
                selected = selected & mask
                tied = tied & mask
            plsc.store_compressed(
                self.out_stage.at[pl.ds(emitted, lanes)],
                output_of(positions, None),
                mask=selected,
            )
            plsc.store_compressed(
                self.ties.at[pl.ds(kept, lanes)], positions, mask=tied
            )
            return (
                emitted + plsc.all_reduce_population_count(selected)[0],
                kept + plsc.all_reduce_population_count(tied)[0],
            )

        emitted, kept = self.for_vectors(
            num_ties, partition, (emitted, jnp.int32(0)), SMALL_UNROLL
        )
        return emitted, kept, quota - above, shift == 0

    def refine_ties(self, emitted, num_ties, quota, output_of):
        """Emits the quota largest of the tie list's candidates: radix levels
        while more than one vector of them is unresolved, then a sort."""
        lanes = self.lanes

        def unresolved(state):
            _, num_ties, quota, is_equal = state
            return (
                (quota > 0)
                & (quota < num_ties)
                & ~is_equal
                & (num_ties > lanes)
            )

        def level(state):
            emitted, num_ties, quota, _ = state
            low, high = self.tie_key_range(num_ties)
            return lax.cond(
                low == high,
                lambda: (emitted, num_ties, quota, jnp.bool_(True)),
                lambda: self.refine_level(
                    emitted, num_ties, quota, low, high, output_of
                ),
            )

        emitted, num_ties, quota, _ = lax.while_loop(
            unresolved, level, (emitted, num_ties, quota, jnp.bool_(False))
        )

        is_sorted = (quota > 0) & (quota < num_ties) & (num_ties <= lanes)
        pl.when(is_sorted)(
            lambda: self.sort_ties(emitted, num_ties, quota, output_of)
        )
        emitted = jnp.where(is_sorted, emitted + quota, emitted)
        quota = jnp.where(is_sorted, 0, quota)
        return self.emit_ties(emitted, quota, output_of)

    def sort_ties(self, emitted, num_ties, quota, output_of):
        """Emits the quota largest of at most one vector of ties."""
        lanes = self.lanes
        valid = self.lane_iota < num_ties
        positions = self.tie_positions(0, valid)
        keys = plsc.load_gather(self.cand_key, (positions,))
        _, positions, _ = plsc.sort_key_val(
            keys, positions, mask=valid, descending=True
        )
        plsc.store_compressed(
            self.out_stage.at[pl.ds(emitted, lanes)],
            output_of(positions, None),
            mask=self.lane_iota < quota,
        )

    def emit_ties(self, emitted, quota, output_of):
        """Emits the first quota ties; returns the emitted count."""

        def emit_first(v, mask, emitted):
            positions = self.tie_positions(v, mask)
            taken = self.lane_iota >= 0 if mask is None else mask
            plsc.store_compressed(
                self.out_stage.at[pl.ds(emitted, self.lanes)],
                output_of(positions, None),
                mask=taken,
            )
            return emitted + plsc.all_reduce_population_count(taken)[0]

        return self.for_vectors(quota, emit_first, emitted, SMALL_UNROLL)

    # ---- row metadata --------------------------------------------------

    def row_meta(self, i):
        """(row, length) of local row i. The clamp to [0, num_columns] is a
        memory-safety bound: no length makes the kernel read past its row."""
        geometry = self.geometry
        row = self.row_of(i)
        if not geometry.has_row_lengths:
            return row, jnp.int32(geometry.num_columns)
        length = self.scalar_at(self.row_values, i - self.first_row)
        return row, jnp.clip(length, 0, geometry.num_columns)

    def chunk_range(self, length, is_whole):
        """Chunks [first, end) of a row this subcore streams: all of them, or
        its part of a sampled row unless is_whole."""
        geometry = self.geometry
        chunks = self.num_chunks(length)
        per_part = (chunks + self.num_parts - 1) // self.num_parts
        is_partial = (length > geometry.direct_max) & jnp.logical_not(is_whole)
        first = jnp.where(
            is_partial, jnp.minimum(self.part * per_part, chunks), 0
        )
        end = jnp.where(
            is_partial, jnp.minimum(first + per_part, chunks), chunks
        )
        return first, end

    def sample_shape(self, length):
        """(log2 sample pieces, rank of the threshold in the sample): the
        expected sampled count above the k-th largest score, rounded up,
        plus SAMPLE_SIGMAS standard deviations."""
        geometry = self.geometry
        piece_log2 = geometry.sample_piece_words.bit_length() - 1
        pieces_log2 = (
            jnp.clip(
                bit_length(length // (2 * geometry.k)) + SAMPLE_DENSITY_LOG2,
                MIN_SAMPLE_WORDS_LOG2,
                MAX_SAMPLE_WORDS_LOG2,
            )
            - piece_log2
        )
        samples = lax.shift_left(
            jnp.int32(geometry.sample_piece_words), pieces_log2
        )
        expected = (geometry.k * samples + length - 1) // length
        root_bits = (
            -(-geometry.k * MAX_SAMPLE_WORDS // geometry.direct_max)
        ).bit_length() // 2 + 1
        rank = expected + SAMPLE_SIGMAS * (
            integer_sqrt(expected, root_bits) + 1
        )
        return pieces_log2, jnp.minimum(rank, samples)

    def num_chunks(self, length):
        geometry = self.geometry
        chunks = (length + geometry.chunk_words - 1) // geometry.chunk_words
        return jnp.where(length <= geometry.k, 0, chunks)

    def chunk_start(self, chunk):
        geometry = self.geometry
        start = jnp.minimum(
            chunk * geometry.chunk_words,
            geometry.row_stride - geometry.chunk_words,
        )
        return pl.multiple_of(start, TILE_COLUMNS)

    def next_row_where(self, local_row, predicate):
        """First local row from local_row on whose length satisfies
        predicate, or rows_end."""

        def skip(i):
            return (i < self.rows_end) & ~predicate(self.row_meta(i)[1])

        return lax.while_loop(skip, lambda i: i + 1, local_row)

    # ---- chunk stream --------------------------------------------------

    def chunk_copy(self, row, chunk, slot):
        geometry = self.geometry
        return pltpu.make_async_copy(
            self.score_words(
                row, self.chunk_start(chunk), geometry.chunk_words
            ),
            self.ring.at[
                pl.ds(slot * geometry.chunk_words, geometry.chunk_words)
            ],
            self.ring_sems.at[slot],
        )

    def stream_at(self, issued, consumed, local_row, is_whole=False):
        """Stream state at the first row from local_row on with chunks to
        stream: (issued, consumed, local row, row, first chunk, end chunk)."""

        def has_chunks(length):
            first, end = self.chunk_range(length, is_whole)
            return first < end

        local_row = self.next_row_where(local_row, has_chunks)
        row, length = self.row_meta(local_row)
        first, end = self.chunk_range(length, is_whole)
        return (issued, consumed, local_row, row, first, end)

    def fill(self, stream):
        """Issues chunk copies until NUM_CHUNK_BUFFERS - 1 are in flight."""

        return lax.while_loop(self.can_issue, self.issue, stream)

    def can_issue(self, stream):
        issued, consumed, local_row = stream[:3]
        return (issued < consumed + NUM_CHUNK_BUFFERS - 1) & (
            local_row < self.rows_end
        )

    def issue(self, stream):
        """Starts the stream's next chunk copy."""
        issued, consumed, local_row, row, chunk, end = stream
        self.chunk_copy(row, chunk, issued % NUM_CHUNK_BUFFERS).start()
        return lax.cond(
            chunk + 1 < end,
            lambda: (issued + 1, consumed, local_row, row, chunk + 1, end),
            lambda: self.stream_at(issued + 1, consumed, local_row + 1),
        )

    def restart_stream(self, stream, local_row):
        """Drops the copies in flight and restarts the stream at local_row."""
        issued, consumed = stream[:2]

        def drain(q, carry):
            self.chunk_copy(0, 0, q % NUM_CHUNK_BUFFERS).wait()
            return carry

        lax.fori_loop(consumed, issued, drain, 0)
        return self.fill(
            self.stream_at(issued, issued, local_row, is_whole=True)
        )

    # ---- per-chunk work ------------------------------------------------

    def copy_keys(self, slot, start, first, end, shift):
        """Stores the keys of the chunk's ring words [0, end) at their row
        positions start + word, and counts the first select level of words
        [first, end)."""
        lanes = self.lanes
        base = slot * self.geometry.chunk_words

        def copy(v, mask):
            bits = self.ring[pl.ds(base + v * lanes, lanes)]
            key = monotone_key(self.decode(bits, shift))
            self.cand_key[pl.ds(start + v * lanes, lanes)] = key
            self.count_buckets(
                lax.shift_right_logical(key ^ INT32_MIN, DIRECT_SHIFT), mask
            )

        full = end // lanes
        plsc.parallel_loop(first // lanes, full, unroll=WALK_UNROLL)(
            lambda v: copy(v, None)
        )

        @pl.when(full * lanes < end)
        def _():
            copy(full, full * lanes + self.lane_iota < end)

    def filter_chunk(self, slot, start, first, end, tau, count, top, shift):
        """Appends the row positions of the chunk's scores >= tau among its
        ring words [first, end) and their keys to the candidate list; returns
        (count, running max key). Words outside become NaN."""
        lanes = self.lanes
        base = slot * self.geometry.chunk_words
        tail = base + (self.vectors(end) - 1) * lanes
        self.ring[pl.ds(tail, lanes)] = jnp.where(
            tail - base + self.lane_iota < end,
            self.ring[pl.ds(tail, lanes)],
            NAN_BITS,
        )

        @plsc.parallel_loop(0, first // lanes)
        def _(v):
            self.ring[pl.ds(base + v * lanes, lanes)] = jnp.full(
                (lanes,), NAN_BITS, jnp.int32
            )

        def keep_loop(decode):
            def keep(v, carry):
                kept, positions = carry
                bits = decode(self.ring[pl.ds(base + v * lanes, lanes)])
                passed = lax.bitcast_convert_type(bits, jnp.float32) >= tau
                plsc.store_compressed(
                    self.cand_idx.at[pl.ds(kept, lanes)], positions, mask=passed
                )
                kept = kept + plsc.all_reduce_population_count(passed)[0]
                return kept, positions + lanes

            # Note (david): each vector's count reaches the scalar unit late,
            # so one vector per lane stays in flight.
            return plsc.parallel_loop(
                0,
                self.vectors(end),
                unroll=self.lanes,
                carry=(count, start + self.lane_iota),
            )(keep)[0]

        if self.geometry.layout is ScoreLayout.ROW_PAIRS:
            # Note (david): one loop per half, each decoding with one op.
            new_count = lax.cond(
                shift == 0,
                lambda: keep_loop(lambda bits: bits & HIGH_HALF_MASK),
                lambda: keep_loop(lambda bits: lax.shift_left(bits, HALF_BITS)),
            )
        else:
            new_count = keep_loop(lambda bits: bits)
        return new_count, self.fetch_keys(
            base, start, (count, new_count), top, shift
        )

    def fetch_keys(self, base, start, span, top, shift):
        """Stores the keys of candidates span = [count, new_count), all from
        the chunk at ring word base and row position start; returns the
        running max key."""
        lanes = self.lanes
        count, new_count = span

        def fetch(u, top):
            positions = u * lanes + self.lane_iota
            valid = (positions >= count) & (positions < new_count)
            local = self.cand_idx[pl.ds(u * lanes, lanes)] - start + base
            bits = plsc.load_gather(self.ring, (jnp.where(valid, local, base),))
            key = monotone_key(self.decode(bits, shift))
            plsc.store_compressed(
                self.cand_key.at[pl.ds(jnp.maximum(u * lanes, count), lanes)],
                key,
                mask=valid,
            )
            return jnp.maximum(top, jnp.where(valid, key, INT32_MIN))

        return plsc.parallel_loop(
            count // lanes,
            self.vectors(new_count),
            unroll=SMALL_UNROLL,
            carry=top,
        )(fetch)

    # ---- sample threshold ----------------------------------------------

    def sample_copy(self, row, length, pieces_log2, piece):
        piece_words = self.geometry.sample_piece_words
        spacing = lax.shift_right_logical(length - piece_words, pieces_log2)
        offset = piece * spacing + lax.shift_right_logical(spacing, 1)
        offset = pl.multiple_of(offset & -piece_words, piece_words)
        return pltpu.make_async_copy(
            self.score_words(row, offset, piece_words),
            self.sample.at[pl.ds(piece * piece_words, piece_words)],
            self.sample_sem.at[0],
        )

    def sample_copies(self, local_row, is_start):
        """Starts or waits the sample copies of local row local_row."""
        row, length = self.row_meta(local_row)
        pieces_log2, _ = self.sample_shape(length)

        def each(piece, carry):
            copy = self.sample_copy(row, length, pieces_log2, piece)
            start_or_wait(copy, is_start)
            return carry

        lax.fori_loop(0, lax.shift_left(jnp.int32(1), pieces_log2), each, 0)

    def prefetch_sample(self, from_row):
        """Starts the sample copies of the next sampled row from from_row."""
        geometry = self.geometry
        local_row = self.next_row_where(
            from_row, lambda length: length > geometry.direct_max
        )

        @pl.when(local_row < self.rows_end)
        def _():
            self.sample_copies(local_row, is_start=True)

    def sample_threshold(self, local_row):
        """Key of a value that about rank of the row's sampled scores reach,
        located on the top SAMPLE_LEVELS * SAMPLE_BITS key bits."""
        lanes = self.lanes
        row, length = self.row_meta(local_row)
        pieces_log2, rank = self.sample_shape(length)
        shift_in = decode_shift(self.geometry.layout, row)
        self.sample_copies(local_row, is_start=False)
        count = self.geometry.sample_piece_words * lax.shift_left(
            jnp.int32(1), pieces_log2
        )

        def level(level, carry):
            prefix, quota = carry
            shift = WORD_BITS - (level + 1) * SAMPLE_BITS
            self.zero_hist(SAMPLE_BUCKETS)

            def count_level(v, mask, unused):
                bits = self.sample[pl.ds(v * lanes, lanes)]
                key = monotone_key(self.decode(bits, shift_in))
                digits = lax.shift_right_logical(key ^ INT32_MIN, shift)
                in_prefix = (
                    lax.shift_right_logical(digits, SAMPLE_BITS) == prefix
                )
                mask = in_prefix if mask is None else mask & in_prefix
                self.count_buckets(digits & (SAMPLE_BUCKETS - 1), mask)
                return unused

            self.for_vectors(count, count_level, jnp.int32(0))
            bucket, above, _ = self.locate(SAMPLE_BUCKETS, quota)
            return lax.shift_left(prefix, SAMPLE_BITS) | bucket, quota - above

        prefix, _ = lax.fori_loop(0, SAMPLE_LEVELS, level, (jnp.int32(0), rank))
        threshold_shift = WORD_BITS - SAMPLE_LEVELS * SAMPLE_BITS
        # The lower edge of the bucket that holds -inf is a NaN key.
        tau_key = lax.shift_left(prefix, threshold_shift) ^ INT32_MIN
        return jnp.maximum(tau_key, NEG_INF_KEY)

    # ---- rows ----------------------------------------------------------

    def range_shift(self, base, top):
        """First-level shift that spreads keys [base, max(top)] over the
        select buckets."""
        span = jnp.max(top) - base
        return jnp.maximum(bit_length(span) - SELECT_BUCKETS_LOG2, 0)

    def compact(self):
        """Keeps the k candidates at the list positions in out_stage[:k];
        returns the smallest kept key."""
        geometry = self.geometry
        lanes = self.lanes
        num_vectors = geometry.k_words // lanes

        def gather(v, low):
            is_kept = v * lanes + self.lane_iota < geometry.k
            positions = self.out_stage[pl.ds(v * lanes, lanes)]
            positions = jnp.where(is_kept, positions, 0)
            key = plsc.load_gather(self.cand_key, (positions,))
            self.ties[pl.ds(v * lanes, lanes)] = key
            self.ties[pl.ds(geometry.k_words + v * lanes, lanes)] = (
                plsc.load_gather(self.cand_idx, (positions,))
            )
            return jnp.minimum(low, jnp.where(is_kept, key, INT32_MAX))

        low = plsc.parallel_loop(
            0,
            num_vectors,
            unroll=SMALL_UNROLL,
            carry=jnp.full((lanes,), INT32_MAX, jnp.int32),
        )(gather)

        @plsc.parallel_loop(0, num_vectors, unroll=SMALL_UNROLL)
        def _(v):
            keys = self.ties[pl.ds(v * lanes, lanes)]
            self.cand_key[pl.ds(v * lanes, lanes)] = keys
            positions = self.ties[pl.ds(geometry.k_words + v * lanes, lanes)]
            self.cand_idx[pl.ds(v * lanes, lanes)] = positions

        return jnp.min(low)

    def process_chunk(self, slot, chunk, row, job):
        """Consumes one ring chunk of the row into the candidate list;
        returns (count, running max key)."""
        geometry = self.geometry
        start = self.chunk_start(chunk)
        first = chunk * geometry.chunk_words - start
        end = jnp.minimum(row.length - start, geometry.chunk_words)

        def direct():
            self.copy_keys(slot, start, first, end, row.decode_shift)
            return job.count, job.top

        def filtered():
            tau = lax.bitcast_convert_type(
                jnp.full((self.lanes,), monotone_key(job.tau_key), jnp.int32),
                jnp.float32,
            )
            return self.filter_chunk(
                slot,
                start,
                first,
                end,
                tau,
                job.count,
                job.top,
                row.decode_shift,
            )

        return lax.cond(row.is_direct, direct, filtered)

    def trivial_row(self, row):
        """Writes the output of a row of at most k scores and starts its
        copy."""
        lanes = self.lanes

        @plsc.parallel_loop(
            0, self.geometry.k_words // lanes, unroll=SMALL_UNROLL
        )
        def _(v):
            positions = v * lanes + self.lane_iota
            self.out_stage[pl.ds(v * lanes, lanes)] = jnp.where(
                positions < row.length, positions, -1
            )

        self.write_row(row)

    # ---- output transform ----------------------------------------------

    def write_row(self, row):
        """Maps out_stage[:k] by the output transform and starts its copy to
        the row's output."""
        transform = self.geometry.transform
        if transform is not OutputTransform.NONE:
            base = self.scalar_at(
                self.row_values,
                self.geometry.bases_offset + row.local_row - self.first_row,
            )
            if transform is OutputTransform.OFFSET:
                self.map_positions(lambda v, p, valid: p + base, self.out_stage)
            else:
                self.page_table_lookup(base)
        self.out_copy(row.row).start()

    def map_positions(self, mapped, target):
        """target[:k] = mapped(v, p, valid) for p the positions in vector v
        of out_stage where valid, -1 where p is -1 and in the lanes past
        k."""
        lanes = self.lanes

        @plsc.parallel_loop(
            0, self.geometry.k_words // lanes, unroll=SMALL_UNROLL
        )
        def _(v):
            positions = self.out_stage[pl.ds(v * lanes, lanes)]
            valid = (positions >= 0) & (
                v * lanes + self.lane_iota < self.geometry.k
            )
            target[pl.ds(v * lanes, lanes)] = jnp.where(
                valid, mapped(v, positions, valid), -1
            )

    def page_table_lookup(self, base):
        """Translates out_stage[:k] through the page table, from table row base
        if it is loaded whole, else from entry base on. Candidate buffers are
        free here."""
        geometry = self.geometry
        lanes = self.lanes
        page_mask = (1 << geometry.page_shift) - 1

        def entries(p):
            return lax.shift_right_logical(p, geometry.page_shift)

        if geometry.holds_table_row:
            pltpu.sync_copy(
                self.table_hbm.at[base, pl.ds(0, geometry.table_words)],
                self.cand_key.at[pl.ds(0, geometry.table_words)],
            )

            def page_of(v, p, valid):
                del v
                index = jnp.where(valid, entries(p), 0)
                return plsc.load_gather(self.cand_key, (index,))

        else:
            self.map_positions(lambda v, p, valid: base + entries(p), self.ties)
            indices = plsc.Indices(
                self.ties.at[pl.ds(0, geometry.k_words)], ignored_value=-1
            )
            pltpu.async_copy(
                self.table_hbm.at[indices],
                self.cand_idx.at[pl.ds(0, geometry.k_words)],
                self.table_sem.at[0],
            ).wait()

            def page_of(v, p, valid):
                del p, valid
                return self.cand_idx[pl.ds(v * lanes, lanes)]

        def physical(v, p, valid):
            page = page_of(v, p, valid)
            return lax.shift_left(page, geometry.page_shift) | (p & page_mask)

        self.map_positions(physical, self.out_stage)

    def out_copy(self, row):
        geometry = self.geometry
        return pltpu.make_async_copy(
            self.out_stage.at[pl.ds(0, geometry.k)],
            self.out_hbm.at[row],
            self.out_sem.at[0],
        )

    def new_job(self, row, phase, stream, tau_key, is_part):
        pl.when(row.is_direct)(lambda: self.zero_hist(SELECT_BUCKETS))
        first, end = self.chunk_range(row.length, ~is_part)
        base = jnp.where(
            row.is_direct, INT32_MIN, tau_key - (tau_key == 0).astype(jnp.int32)
        )
        return _Job(
            phase=phase,
            stream=stream,
            chunk=first,
            end=end,
            count=jnp.int32(0),
            tau_key=tau_key,
            base=base,
            top=jnp.full((self.lanes,), INT32_MIN, jnp.int32),
            resume=jnp.int32(STREAM),
            member=jnp.int32(1),
            issued=jnp.int32(0),
            is_part=is_part,
            is_cut=jnp.bool_(False),
        )

    def run_job(self, row, job):
        """Runs a row's phases until DONE; one instance of each phase's code
        serves every row."""
        geometry = self.geometry
        if geometry.is_split:
            branches = (
                self.stream_phase,
                self.select_phase,
                self.merge_phase,
                self.retry_phase,
                self.publish_phase,
                self.barrier_phase,
            )
        else:

            def unused(row, job):
                del row
                return job

            branches = (
                self.stream_phase,
                self.select_phase,
                unused,
                self.retry_phase,
                unused,
                unused,
            )
        branches = [functools.partial(branch, row) for branch in branches]
        return lax.while_loop(
            lambda job: job.phase != DONE,
            lambda job: lax.switch(job.phase, branches, job),
            job,
        )

    def stream_phase(self, row, job):
        """Consumes chunks until the row's range ends or the candidate list
        fills up, then picks the next phase."""
        geometry = self.geometry
        limit = geometry.capacity - geometry.chunk_words

        def more(job):
            return (job.chunk < job.end) & (job.count <= limit)

        def step(job):
            issued, consumed = job.stream[:2]
            slot = consumed % NUM_CHUNK_BUFFERS
            self.chunk_copy(0, 0, slot).wait()
            count, top = self.process_chunk(slot, job.chunk, row, job)
            stream = (issued, consumed + 1, *job.stream[2:])
            stream = lax.cond(
                self.can_issue(stream), self.issue, lambda s: s, stream
            )
            return job._replace(
                stream=stream, chunk=job.chunk + 1, count=count, top=top
            )

        job = lax.while_loop(more, step, job)
        count = jnp.where(row.is_direct, row.length, job.count)
        publishes = job.is_part & ~row.is_leader
        is_full = ~row.is_direct & (count > limit)
        is_cut = is_full | (publishes & (count > geometry.part_capacity))
        phase = jnp.where(
            row.is_sampled & (count < geometry.k) & ~job.is_part, RETRY, SELECT
        )
        phase = jnp.where(job.is_part, BARRIER, phase)
        phase = jnp.where(publishes, PUBLISH, phase)
        phase = jnp.where(is_cut, SELECT, phase)
        return job._replace(
            phase=phase.astype(jnp.int32),
            count=count,
            is_cut=is_cut,
            resume=jnp.int32(STREAM),
        )

    def select_phase(self, row, job):
        """Selects the k largest candidates: to cut the list down to them,
        or as the row's output."""
        geometry = self.geometry
        shift = jnp.where(
            row.is_direct, DIRECT_SHIFT, self.range_shift(job.base, job.top)
        )
        self.select(job.count, job.base, shift, row.is_direct, job.is_cut)

        def cut(job):
            kth = self.compact()
            return job._replace(
                phase=job.resume,
                count=jnp.int32(geometry.k),
                base=kth,
                tau_key=kth + 1,
            )

        def output(job):
            self.write_row(row)
            is_barrier_due = geometry.is_split & ~row.is_sampled
            phase = jnp.where(is_barrier_due, BARRIER, DONE)
            return job._replace(phase=phase.astype(jnp.int32))

        return lax.cond(job.is_cut, cut, output, job)

    def retry_phase(self, row, job):
        """Streams the whole row again with tau = -inf."""
        stream = self.restart_stream(job.stream, row.local_row)
        return self.new_job(
            row,
            jnp.int32(STREAM),
            stream,
            jnp.int32(NEG_INF_KEY),
            jnp.bool_(False),
        )

    def list_at(self, subcore, list_index, offset, size):
        """Words [offset, offset + size) of a subcore's published keys
        (list 0) or positions (list 1)."""
        geometry = self.geometry
        list_start = (
            PART_LISTS * subcore + list_index
        ) * geometry.part_capacity
        start = geometry.headers_words + list_start
        return self.shared.at[pl.ds(start + offset, size)]

    def publish_phase(self, row, job):
        """Leaves this part's candidates in shared VMEM for the leader."""
        del row
        lanes = self.lanes
        header = jnp.where(self.lane_iota == HEADER_COUNT, job.count, 0)
        header = jnp.where(self.lane_iota == HEADER_BASE, job.base, header)
        header = jnp.where(
            self.lane_iota == HEADER_TOP, jnp.max(job.top), header
        )
        self.headers[pl.ds(0, lanes)] = header
        pltpu.sync_copy(
            self.headers.at[pl.ds(0, lanes)],
            self.shared.at[pl.ds(self.subcore * lanes, lanes)],
        )

        def block(b, carry):
            offset = b * SHARED_BLOCK_WORDS
            for list_index, source in enumerate((self.cand_key, self.cand_idx)):
                pltpu.sync_copy(
                    source.at[pl.ds(offset, SHARED_BLOCK_WORDS)],
                    self.list_at(
                        self.subcore, list_index, offset, SHARED_BLOCK_WORDS
                    ),
                )
            return carry

        lax.fori_loop(0, self.shared_blocks(job.count), block, 0)
        return job._replace(phase=jnp.int32(BARRIER))

    def shared_blocks(self, count):
        return (count + SHARED_BLOCK_WORDS - 1) // SHARED_BLOCK_WORDS

    def barrier_phase(self, row, job):
        geometry = self.geometry
        plsc.subcore_barrier()
        is_merger = row.is_sampled & row.is_leader

        @pl.when(is_merger)
        def _():
            pltpu.sync_copy(
                self.shared.at[pl.ds(0, geometry.headers_words)], self.headers
            )

        phase = jnp.where(is_merger, MERGE, DONE)
        return job._replace(phase=phase.astype(jnp.int32))

    def member_header(self, member, field):
        return self.scalar_at(
            self.headers, (self.subcore + member) * self.lanes + field
        )

    def member_copies(self, member):
        """Copies of a group member's published lists into ring slot
        member % MERGE_SLOTS: keys, then positions."""
        geometry = self.geometry
        slot = member % MERGE_SLOTS
        source = self.subcore + member

        def copies(b):
            offset = b * SHARED_BLOCK_WORDS
            return [
                pltpu.make_async_copy(
                    self.list_at(
                        source, list_index, offset, SHARED_BLOCK_WORDS
                    ),
                    self.ring.at[
                        pl.ds(
                            (PART_LISTS * slot + list_index)
                            * geometry.part_capacity
                            + offset,
                            SHARED_BLOCK_WORDS,
                        )
                    ],
                    self.merge_sems.at[slot],
                )
                for list_index in range(PART_LISTS)
            ]

        return copies, self.shared_blocks(
            self.member_header(member, HEADER_COUNT)
        )

    def member_dma(self, member, is_start):
        copies, blocks = self.member_copies(member)

        def each(b, carry):
            for copy in copies(b):
                start_or_wait(copy, is_start)
            return carry

        lax.fori_loop(0, blocks, each, 0)

    def merge_phase(self, row, job):
        """Appends the next group member's candidates, cutting the list first if
        they do not fit. The copy of the member after it runs meanwhile."""
        del row
        geometry = self.geometry

        def finished(job):
            phase = jnp.where(job.count < geometry.k, RETRY, SELECT)
            return job._replace(
                phase=phase.astype(jnp.int32), is_cut=jnp.bool_(False)
            )

        def next_member(job):
            member = job.member
            part_count = self.member_header(member, HEADER_COUNT)

            def cut(job):
                return job._replace(
                    phase=jnp.int32(SELECT),
                    is_cut=jnp.bool_(True),
                    resume=jnp.int32(MERGE),
                )

            def take(job):
                last = jnp.minimum(member + 1, self.num_parts - 1)

                def issue(issued):
                    self.member_dma(issued + 1, is_start=True)
                    return issued + 1

                issued = lax.while_loop(lambda i: i < last, issue, job.issued)
                self.member_dma(member, is_start=False)
                self.append_part(member, job.count, part_count)
                return job._replace(
                    count=job.count + part_count,
                    base=jnp.minimum(
                        job.base, self.member_header(member, HEADER_BASE)
                    ),
                    top=jnp.maximum(
                        job.top, self.member_header(member, HEADER_TOP)
                    ),
                    member=member + 1,
                    issued=issued,
                )

            return lax.cond(
                job.count + part_count > geometry.capacity, cut, take, job
            )

        return lax.cond(
            job.member >= self.num_parts, finished, next_member, job
        )

    def append_part(self, member, count, part_count):
        """Appends a member's candidates from its ring slot to the list."""
        geometry = self.geometry
        lanes = self.lanes
        keys = PART_LISTS * (member % MERGE_SLOTS) * geometry.part_capacity

        @plsc.parallel_loop(0, self.vectors(part_count), unroll=SMALL_UNROLL)
        def _(v):
            valid = v * lanes + self.lane_iota < part_count
            plsc.store_compressed(
                self.cand_key.at[pl.ds(count + v * lanes, lanes)],
                self.ring[pl.ds(keys + v * lanes, lanes)],
                mask=valid,
            )
            plsc.store_compressed(
                self.cand_idx.at[pl.ds(count + v * lanes, lanes)],
                self.ring[
                    pl.ds(keys + geometry.part_capacity + v * lanes, lanes)
                ],
                mask=valid,
            )

    def row_info(self, local_row, is_member, is_leader):
        geometry = self.geometry
        row, length = self.row_meta(local_row)
        return _Row(
            local_row=local_row,
            row=row,
            length=length,
            is_direct=length <= geometry.direct_max,
            is_sampled=is_member & (length > geometry.direct_max),
            is_leader=is_leader,
            decode_shift=decode_shift(geometry.layout, row),
        )

    def row_step(self, local_row, stream):
        """Row mode: selects local row local_row whole."""
        geometry = self.geometry
        row = self.row_info(local_row, jnp.bool_(True), jnp.bool_(True))

        @pl.when(local_row > 0)
        def _():
            self.out_copy(row.row).wait()

        def short(stream):
            self.trivial_row(row)
            return stream

        def selected(stream):
            def sampled():
                tau_key = self.sample_threshold(local_row)
                self.prefetch_sample(local_row + 1)
                return tau_key

            tau_key = lax.cond(
                row.is_direct, lambda: jnp.int32(NEG_INF_KEY), sampled
            )
            job = self.new_job(
                row, jnp.int32(STREAM), stream, tau_key, jnp.bool_(False)
            )
            return self.run_job(row, job).stream

        return lax.cond(row.length <= geometry.k, short, selected, stream)

    def run(self):
        if self.geometry.is_split:
            self.run_parts()
        else:
            self.run_rows()

    def run_rows(self):
        """Every subcore selects whole rows, block by block of their lengths,
        streaming the next row's first chunks and sample while it selects."""
        block_rows = self.geometry.lengths_block

        def block(b, carry):
            first_row = b * block_rows
            self.load_lengths(first_row)
            self.rows_end = jnp.minimum(
                self.num_local_rows, first_row + block_rows
            )
            self.prefetch_sample(first_row)
            stream = self.fill(
                self.stream_at(jnp.int32(0), jnp.int32(0), first_row)
            )
            lax.fori_loop(first_row, self.rows_end, self.row_step, stream)
            return carry

        # Note (david): rows that fit one block skip the block loop, which
        # slows the row loop inside it.
        if self.geometry.num_lengths_blocks == 1:
            block(jnp.int32(0), 0)
        else:
            num_blocks = (self.num_local_rows + block_rows - 1) // block_rows
            lax.fori_loop(0, num_blocks, block, 0)

        @pl.when(self.num_local_rows > 0)
        def _():
            self.out_copy(0).wait()

    # ---- driven row by row ---------------------------------------------

    def empty_stream(self):
        """Stream state before the first select_row."""
        return (jnp.int32(0),) * 6

    def select_row(self, local_row, stream):
        """Selects local row local_row into out_stage[:k] and starts its output
        copy, for a caller that drives the rows in order. Returns the stream
        state."""
        block_rows = self.geometry.lengths_block
        first_row = local_row - lax.rem(local_row, block_rows)
        self.first_row = first_row
        self.rows_end = jnp.minimum(self.num_local_rows, first_row + block_rows)

        def start_block(stream):
            self.load_lengths(first_row)
            self.prefetch_sample(first_row)
            issued = stream[0]
            return self.fill(self.stream_at(issued, issued, first_row))

        is_block_start = local_row == first_row
        stream = lax.cond(is_block_start, start_block, lambda s: s, stream)
        return self.row_step(local_row, stream)

    def finish_rows(self):
        """Waits the last output copy of the rows select_row selected."""

        @pl.when(self.num_local_rows > 0)
        def _():
            self.out_copy(0).wait()

    def share_selection(self):
        """Split mode, after run_parts: each group member's out_stage[:k] gets
        the row's output through the leader's list in shared VMEM, which no
        part publishes into."""
        k_words = self.geometry.k_words
        leader = self.subcore - self.part
        is_member = self.column < self.geometry.num_rows
        leader_list = self.list_at(leader, 0, 0, k_words)
        stage = self.out_stage.at[pl.ds(0, k_words)]

        @pl.when(is_member & (self.part == 0))
        def _():
            pltpu.sync_copy(stage, leader_list)

        plsc.subcore_barrier()

        @pl.when(is_member & (self.part != 0))
        def _():
            pltpu.sync_copy(leader_list, stage)

    def run_parts(self):
        """Each of a core's rows gets a group of subcores. A sampled row is cut
        into one chunk range per subcore, and the leader selects from the
        parts' candidates in shared VMEM. Other rows go whole to the leader."""
        geometry = self.geometry
        group_size = geometry.num_subcores // jnp.maximum(self.core_rows, 1)
        self.is_parts = True
        group = self.subcore // group_size
        self.part = self.subcore - group * group_size
        self.num_parts = group_size
        self.column = self.core + geometry.num_cores * group
        self.load_lengths(jnp.int32(0))
        is_member = group < self.core_rows
        is_leader = is_member & (self.part == 0)
        row = self.row_info(jnp.int32(0), is_member, is_leader)
        self.num_local_rows = (row.is_sampled | is_leader).astype(jnp.int32)
        self.rows_end = self.num_local_rows
        self.prefetch_sample(jnp.int32(0))
        stream = self.fill(
            self.stream_at(jnp.int32(0), jnp.int32(0), jnp.int32(0))
        )
        is_short = row.length <= geometry.k

        @pl.when(is_leader & is_short)
        def _():
            self.trivial_row(row)

        tau_key = lax.cond(
            row.is_sampled,
            lambda: self.sample_threshold(jnp.int32(0)),
            lambda: jnp.int32(NEG_INF_KEY),
        )
        is_streamed = row.is_sampled | (is_leader & ~is_short)
        phase = jnp.where(is_streamed, STREAM, BARRIER)
        job = self.new_job(row, phase, stream, tau_key, row.is_sampled)
        self.run_job(row, job)

        @pl.when(is_leader)
        def _():
            self.out_copy(row.row).wait()


SCRATCH_NAMES = (
    "row_values",
    "ring",
    "cand_key",
    "cand_idx",
    "ties",
    "hist",
    "totals",
    "out_stage",
    "sample",
    "headers",
    "ring_sems",
    "sample_sem",
    "out_sem",
    "merge_sems",
    "table_sem",
    "shared",
)


def scratch_types(geometry: TopkGeometry):
    lanes = geometry.lanes
    return (
        pltpu.VMEM((geometry.row_values_words,), jnp.int32),
        pltpu.VMEM((geometry.ring_words,), jnp.int32),
        pltpu.VMEM((geometry.capacity + lanes,), jnp.int32),
        pltpu.VMEM((geometry.capacity + lanes,), jnp.int32),
        pltpu.VMEM((geometry.capacity + lanes,), jnp.int32),
        pltpu.VMEM((SELECT_BUCKETS + lanes,), jnp.int32),
        pltpu.VMEM((SELECT_BUCKETS // lanes + lanes,), jnp.int32),
        pltpu.VMEM((geometry.k_words + lanes,), jnp.int32),
        pltpu.VMEM((MAX_SAMPLE_WORDS,), jnp.int32),
        pltpu.VMEM((geometry.headers_words,), jnp.int32),
        pltpu.SemaphoreType.DMA((NUM_CHUNK_BUFFERS,)),
        pltpu.SemaphoreType.DMA((1,)),
        pltpu.SemaphoreType.DMA((1,)),
        pltpu.SemaphoreType.DMA((MERGE_SLOTS,)),
        pltpu.SemaphoreType.DMA((1,)),
        pltpu.VMEM_SHARED(
            (geometry.shared_words if geometry.is_split else lanes,), jnp.int32
        ),
    )


def scratch_vmem_words(scratch: tuple, num_subcores: int) -> int:
    """Words of a subcore's VMEM a kernel's scratch takes: its own VMEM and
    its share of the shared VMEM, which lies in the same memory."""
    words = 0
    for buffer in scratch:
        if buffer.memory_space == pltpu.VMEM:
            words += math.prod(buffer.shape)
        elif buffer.memory_space == pltpu.VMEM_SHARED:
            words += -(-math.prod(buffer.shape) // num_subcores)
    return words


def check_vmem(scratch: tuple, num_subcores: int, name: str) -> None:
    """Fails unless the scratch fits a subcore's VMEM."""
    words = scratch_vmem_words(scratch, num_subcores)
    if words > sparse_core_vmem_words():
        raise ValueError(
            f"{name}'s scratch takes {words} words of SparseCore VMEM; a"
            f" subcore has {sparse_core_vmem_words()}."
        )


def worker_major_rows(values: jax.Array, geometry: TopkGeometry) -> jax.Array:
    """The (num_workers * worker_rows,) int32 per-row values the kernel
    reads: worker column c's rows c, c + num_workers, ... from word
    c * worker_rows on, zero padded."""
    num_words = geometry.num_workers * geometry.worker_rows
    padded = jnp.pad(values, (0, num_words - geometry.num_rows))
    return padded.reshape(geometry.worker_rows, geometry.num_workers).T.reshape(
        -1
    )


def topk_worker(*refs, geometry: TopkGeometry) -> _TopkKernel:
    """One subcore's kernel body on refs: the geometry's inputs, the (T, k)
    output, then the scratch in SCRATCH_NAMES order. run() selects every row;
    select_row drives them one at a time."""
    inputs = list(refs[: geometry.num_inputs])
    lengths_hbm = inputs.pop(0) if geometry.has_row_lengths else None
    bases_hbm = inputs.pop(0) if geometry.has_row_bases else None
    is_paged = geometry.transform is OutputTransform.PAGE_TABLE
    table_hbm = inputs.pop(0) if is_paged else None
    (scores_hbm,) = inputs
    scores_bits = scores_hbm.bitcast(jnp.int32)
    if geometry.layout is ScoreLayout.FLAT:
        scores_bits = scores_bits.reshape(
            geometry.num_rows * geometry.row_stride
        )
    return _TopkKernel(
        geometry,
        refs[geometry.num_inputs + 1 :],
        lengths_hbm=lengths_hbm,
        bases_hbm=bases_hbm,
        table_hbm=table_hbm,
        scores_bits=scores_bits,
        out_hbm=refs[geometry.num_inputs],
    )


def topk_kernel(*refs, geometry: TopkGeometry):
    """topk_worker's body selecting every row."""
    topk_worker(*refs, geometry=geometry).run()


def topk_pallas(inputs: tuple, geometry: TopkGeometry, name: str) -> jax.Array:
    """The kernel on inputs as topk_kernel takes them; returns (num_rows, k)
    int32 positions, row-major for ScoreLayout.FLAT, TensorCore tiled
    otherwise."""
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=geometry.num_cores,
        num_subcores=geometry.num_subcores,
    )
    scratch = scratch_types(geometry)
    check_vmem(scratch, geometry.num_subcores, name)
    kernel = pl.kernel(
        functools.partial(topk_kernel, geometry=geometry),
        out_type=jax.ShapeDtypeStruct(
            (geometry.num_rows, geometry.k), jnp.int32
        ),
        mesh=mesh,
        scratch_types=scratch,
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=geometry.uses_tensor_core_tiling,
            needs_layout_passes=False,
        ),
        name=name,
    )
    # pylint infers pl.kernel's decorator-factory return value here.
    return kernel(*inputs)  # pylint: disable=too-many-function-args
