"""SparseCore gather of each token's top-k latent rows from the split DSA
cache into dense buffers the TensorCore attention reads, from given
indices, or in one call with the top-k that selects them."""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu, tpu_sc as plsc

from flywheel_tpu.topk.kernels.sparse_core_topk import (
    SCRATCH_NAMES as TOPK_SCRATCH_NAMES,
    TopkGeometry,
    scratch_types as topk_scratch_types,
    scratch_vmem_words,
    sparse_core_vmem_words,
    topk_worker,
    worker_major_rows,
)

# A token's 512 fp8 nope dims, 4 a 32-bit word: byte b of word w is dim
# 128 b + w.
NOPE_WORDS = 128
# Its 64 fp8 rope dims, banded the same way: byte b of word w is dim 16 b + w.
ROPE_WORDS = 16
ROPE_TOKENS_PER_ROW = NOPE_WORDS // ROPE_WORDS
# Note (david): the gathered rope packs 4 tokens a word (byte t of word d is
# token 4 q + t's dim d), the tile the TensorCore bitcasts to fp8 (N, 128).
QUAD_TOKENS = 4
# Words of a quad's rope, half of a gathered rope row.
ROPE_HALF_WORDS = QUAD_TOKENS * ROPE_WORDS
# Top-k slots one gather moves. A call of one token cuts them in half, so
# that every worker takes a chunk.
CHUNK_SLOTS = 128
MIN_CHUNK_SLOTS = 64
# Note (david): a block's rope rows hold its first half's quads in their low
# words and its second half's in their high ones: fp8 row i holds slot i's
# rope, then slot ROPE_HALF_SLOTS + i's.
ROPE_HALF_SLOTS = CHUNK_SLOTS
ROPE_BLOCK_SLOTS = 2 * ROPE_HALF_SLOTS
ROPE_ROW_SLOTS = 2 * QUAD_TOKENS
# Note (david): a ring step starts the chunk GATHER_LEAD chunks ahead of the
# one it finishes, which takes GATHER_LEAD + 1 buffers; one more buffer lets
# a refill wait on a write-back whose read has already finished.
GATHER_LEAD = 2
MIN_BUFFERS = GATHER_LEAD + 1
MAX_BUFFERS = GATHER_LEAD + 2
# Note (david): the row of an unused slot, which its gather skips.
IGNORED_ROW = -1
# The logical index of an unused slot.
UNUSED_INDEX = -1
# Note (david): the gathered flag rows hold a word a slot, FLAG_ROW_SLOTS
# slots a row, UNUSED_FLAG for an unused slot and 0 for a used one.
FLAG_ROW_SLOTS = 128
UNUSED_FLAG = 1
BYTE_BITS = 8
HALF_BITS = 16
WORD_BITS = 32
# Lanes of a token's metadata row: its request's block-table offset, the last
# position it may attend (-1 for padding), and the slots the gather fills and
# the attention reads, from slot 0 on.
META_WORDS = 16
META_TABLE_OFFSET = 0
META_POSITION = 1
META_SLOTS = 2
# Rows of the buffers' state: whether the chunk last started in a buffer is
# active (its gather and write-back issued), and whether the buffer holds
# finite words throughout.
STATE_ACTIVE = 0
STATE_FILLED = 1
NUM_STATE_ROWS = 2
# Lanes of the call's chunk row: the index row of its first token, and that
# token's row of the metadata.
CHUNK_WORDS = 16
CHUNK_FIRST_INDEX_ROW = 0
CHUNK_FIRST_TOKEN = 1
SEM_NOPE = 0
SEM_ROPE = 1
SEM_OUT = 2
NUM_SEM_ROWS = 3
# Words of the outputs per slot: its nope row and its shares of the rope and
# flag rows.
GATHERED_WORDS_PER_SLOT = (
    NOPE_WORDS + NOPE_WORDS // ROPE_ROW_SLOTS + NOPE_WORDS // FLAG_ROW_SLOTS
)
# VMEM words of a chunk buffer per slot: its index, row and flag words, its
# rope row, its nope row and its share of the rope stage.
BUFFER_WORDS_PER_SLOT = (
    3 + ROPE_WORDS + NOPE_WORDS + ROPE_HALF_WORDS // QUAD_TOKENS
)
# Note (david): the top-k takes this shift of a subcore's VMEM in the fused
# call; the gather takes the rest.
FUSED_TOPK_SHIFT = 1
# Note (david): the fused call prefetches a worker's next token's metadata at
# the token's first chunk and its table row part of the way through, so no
# token start waits on HBM.
NUM_PREFETCH_BUFFERS = 2
TABLE_PREFETCH_SHARE = 2
SEM_PREFETCH_META = 0
SEM_PREFETCH_TABLE = 1
NUM_PREFETCH_SEMS = 2
LOW_HALF = 0x0000FFFF
LOW_BYTES = 0x00FF00FF


def _delta_swap(x, y, shift, mask):
    swapped = (lax.shift_right_logical(x, shift) ^ y) & mask
    return x ^ lax.shift_left(swapped, shift), y ^ swapped


def transpose_quad(rows):
    """Four tokens' banded rope words -> four bands of 4-token words: out[b]
    word w byte t is token t's byte b of word w."""
    d0, d1, d2, d3 = rows
    d0, d2 = _delta_swap(d0, d2, HALF_BITS, LOW_HALF)
    d1, d3 = _delta_swap(d1, d3, HALF_BITS, LOW_HALF)
    d0, d1 = _delta_swap(d0, d1, BYTE_BITS, LOW_BYTES)
    d2, d3 = _delta_swap(d2, d3, BYTE_BITS, LOW_BYTES)
    return d0, d1, d2, d3


def gather_slots(top_k: int) -> int:
    """Slots a token of a top_k selection takes in the gather and the
    attention: top_k up to whole rope blocks, the slots past it unused."""
    return pl.cdiv(top_k, ROPE_BLOCK_SLOTS) * ROPE_BLOCK_SLOTS


def gather_layout(
    num_tokens: int, top_k: int, pages_per_seq: int, vmem_words: int
) -> tuple[int, int]:
    """(chunk_slots, num_buffers) of a gather whose scratch fits vmem_words
    of a subcore's VMEM next to its table row: the largest chunk of which
    MIN_BUFFERS fit, CHUNK_SLOTS only if every worker gets a chunk, and as
    many buffers as fit up to MAX_BUFFERS."""
    sparse_core = pltpu.get_tpu_info().sparse_core
    num_workers = sparse_core.num_cores * sparse_core.num_subcores
    if num_tokens * top_k // CHUNK_SLOTS < num_workers:
        slot_choices = (MIN_CHUNK_SLOTS,)
    else:
        slot_choices = (CHUNK_SLOTS, MIN_CHUNK_SLOTS)
    free_words = vmem_words - (CHUNK_WORDS + META_WORDS + pages_per_seq)
    for chunk_slots in slot_choices:
        num_buffers = free_words // (chunk_slots * BUFFER_WORDS_PER_SLOT)
        if num_buffers >= MIN_BUFFERS:
            return chunk_slots, min(MAX_BUFFERS, num_buffers)
    raise ValueError(
        f"A {pages_per_seq}-entry block-table row leaves room for"
        f" {num_buffers} gather buffers in {vmem_words} words of SparseCore"
        f" VMEM; the gather needs {MIN_BUFFERS}."
    )


class GatherPlan(NamedTuple):
    """The gather's kernel body on its operands, outputs and scratch."""

    body: functools.partial
    out_type: tuple[jax.ShapeDtypeStruct, ...]
    scratch_types: tuple


def gather_plan(
    num_tokens: int,
    top_k: int,
    page_size: int,
    pages_per_seq: int,
    vmem_words: int,
) -> GatherPlan:
    """The gather of num_tokens tokens' top_k slots on this chip's
    SparseCores, its scratch within vmem_words of each subcore's VMEM."""
    sparse_core = pltpu.get_tpu_info().sparse_core
    num_workers = sparse_core.num_cores * sparse_core.num_subcores
    chunk_slots, num_buffers = gather_layout(
        num_tokens, top_k, pages_per_seq, vmem_words
    )
    chunks_per_token = top_k // chunk_slots
    num_chunks = num_tokens * chunks_per_token
    body = functools.partial(
        dsa_gather_kernel,
        num_chunks=num_chunks,
        chunks_per_worker=pl.cdiv(num_chunks, num_workers),
        chunks_per_token=chunks_per_token,
        page_shift=page_size.bit_length() - 1,
        num_subcores=sparse_core.num_subcores,
        num_lanes=sparse_core.num_lanes,
    )
    scratch_types = (
        pltpu.VMEM((num_buffers, chunk_slots), jnp.int32),
        pltpu.SemaphoreType.DMA((num_buffers,)),
        pltpu.VMEM((META_WORDS,), jnp.int32),
        pltpu.VMEM((pages_per_seq,), jnp.int32),
        *ring_scratch_types(num_buffers, chunk_slots),
    )
    return GatherPlan(body, gathered_types(num_tokens, top_k), scratch_types)


def gathered_types(
    num_tokens: int, top_k: int
) -> tuple[jax.ShapeDtypeStruct, ...]:
    """The gather's outputs: num_tokens tokens' top_k nope rows, their rope
    rows and their flag rows."""
    num_slots = num_tokens * top_k
    return tuple(
        jax.ShapeDtypeStruct((num_slots // row_slots, NOPE_WORDS), jnp.int32)
        for row_slots in (1, ROPE_ROW_SLOTS, FLAG_ROW_SLOTS)
    )


def ring_scratch_types(num_buffers: int, chunk_slots: int) -> tuple:
    """A _GatherRing's scratch, in RingRefs order."""
    return (
        pltpu.VMEM((CHUNK_WORDS,), jnp.int32),
        pltpu.VMEM((num_buffers, chunk_slots), jnp.int32),
        pltpu.VMEM((num_buffers, chunk_slots), jnp.int32),
        pltpu.SMEM((NUM_STATE_ROWS, num_buffers), jnp.int32),
        pltpu.VMEM((num_buffers, chunk_slots, NOPE_WORDS), jnp.int32),
        pltpu.VMEM((num_buffers, chunk_slots, ROPE_WORDS), jnp.int32),
        pltpu.VMEM(
            (num_buffers, chunk_slots // QUAD_TOKENS, ROPE_HALF_WORDS),
            jnp.int32,
        ),
        pltpu.SemaphoreType.DMA((NUM_SEM_ROWS, num_buffers)),
    )


def sparse_core_mesh() -> plsc.VectorSubcoreMesh:
    sparse_core = pltpu.get_tpu_info().sparse_core
    return plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=sparse_core.num_cores,
        num_subcores=sparse_core.num_subcores,
    )


def gather_compiler_params() -> pltpu.CompilerParams:
    # Note (david): untiled operands let the gather take 16-word rope rows. The
    # top-k it shares a call with takes no layout passes, so neither does it.
    return pltpu.CompilerParams(
        use_tc_tiling_on_sc=False,
        needs_layout_passes=False,
        disable_bounds_checks=True,
    )


def dsa_gather_pallas(
    chunk_row: jax.Array,
    indices: jax.Array,
    token_meta: jax.Array,
    block_table: jax.Array,
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    *,
    num_tokens: int,
    top_k: int,
    page_size: int,
    pages_per_seq: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The gather's SparseCore call on checked operands, for num_tokens
    consecutive tokens of a call.

    Args:
      chunk_row: (CHUNK_WORDS,) int32, lane CHUNK_FIRST_INDEX_ROW the index
        row of the first token and lane CHUNK_FIRST_TOKEN its metadata row;
        traced, so every chunk of one size shares one kernel.
      indices: (R * top_k,) int32 flat rows of top_k logical positions, a
        real token's row inside them; read in place.
      token_meta: (T, META_WORDS) int32 rows of token_metadata.
      block_table: (B * pages_per_seq,) int32 flat block table.
      nope_cache: (num_rows, NOPE_WORDS) int32 token rows.
      rope_cache: (num_rows, ROPE_WORDS) int32 token rows.
      num_tokens: the tokens gathered.
      top_k: slots a token, a multiple of ROPE_BLOCK_SLOTS.
      page_size: tokens a page, a power of two.
      pages_per_seq: block-table entries a request, a multiple of 8.

    Returns:
      (nope_rows, rope_rows, flag_rows): (num_tokens * top_k, NOPE_WORDS)
      int32 gathered rows; (num_tokens * top_k / ROPE_ROW_SLOTS,
      NOPE_WORDS) int32, the slots' quads as 4-token words, two quads a row
      (see ROPE_ROW_SLOTS); and (num_tokens * top_k / FLAG_ROW_SLOTS,
      NOPE_WORDS) int32 unused-slot flags. Only the rows of each token's
      first META_SLOTS slots are written, an unused slot's finite.
    """
    plan = gather_plan(
        num_tokens, top_k, page_size, pages_per_seq, sparse_core_vmem_words()
    )
    kernel = pl.kernel(
        plan.body,
        out_type=plan.out_type,
        mesh=sparse_core_mesh(),
        scratch_types=plan.scratch_types,
        compiler_params=gather_compiler_params(),
        name="dsa_gather",
    )
    return kernel(
        chunk_row, indices, token_meta, block_table, nope_cache, rope_cache
    )


def fused_topk_vmem_words() -> int:
    """The top-k's share of a subcore's VMEM in the top-k and gather
    call."""
    return sparse_core_vmem_words() >> FUSED_TOPK_SHIFT


class RingRefs(NamedTuple):
    """A _GatherRing's operands, outputs and scratch (ring_scratch_types)."""

    token_meta_hbm: jax.Array
    block_table_hbm: jax.Array
    nope_cache_hbm: jax.Array
    rope_cache_hbm: jax.Array
    nope_out_hbm: jax.Array
    rope_out_hbm: jax.Array
    flag_out_hbm: jax.Array
    chunk_vmem: jax.Array
    row_vmem: jax.Array
    flag_vmem: jax.Array
    state_smem: jax.Array
    nope_buffer: jax.Array
    rope_buffer: jax.Array
    rope_stage: jax.Array
    sems: jax.Array


class _GatherRing:
    """A subcore's ring of chunk buffers. A chunk is chunk_slots top-k slots
    of one token: its rows are gathered into a buffer while earlier chunks
    finish, then written back at the chunk's place, slot c * chunk_slots of
    the call's token-major slots.

    A chunk is active when it starts below its token's slot count. A used
    slot (its index in [0, the token's position]) reads the cache row of its
    index through the token's block-table row. An unused slot reads nothing
    and is flagged; its output rows hold finite words the attention weighs
    by 0.
    """

    def __init__(self, refs: RingRefs, *, page_shift: int, num_lanes: int):
        self.refs = refs
        self.num_buffers, self.chunk_slots = refs.row_vmem.shape
        self.quads_per_chunk = self.chunk_slots // QUAD_TOKENS
        self.page_shift = page_shift
        self.num_lanes = num_lanes

    def load_chunk_row(self, chunk_hbm: jax.Array) -> jax.Array:
        """The call's chunk row, as a vector."""
        pltpu.sync_copy(chunk_hbm, self.refs.chunk_vmem)
        return self.refs.chunk_vmem[pl.ds(0, self.num_lanes)]

    def meta_copy(self, token, meta, sem):
        """The copy of the token's metadata row into meta."""
        return pltpu.make_async_copy(
            self.refs.token_meta_hbm.at[token], meta, sem
        )

    def table_copy(self, meta, table, sem):
        """The copy of the table row of meta's request into table."""
        offset = self.meta_row(meta)[META_TABLE_OFFSET]
        return pltpu.make_async_copy(
            self.refs.block_table_hbm.at[pl.ds(offset, table.shape[0])],
            table,
            sem,
        )

    def load_token(self, token, meta, table) -> None:
        """Loads the token's metadata row into meta and its table row into
        table."""
        pltpu.sync_copy(self.refs.token_meta_hbm.at[token], meta)
        offset = self.meta_row(meta)[META_TABLE_OFFSET]
        pltpu.sync_copy(
            self.refs.block_table_hbm.at[pl.ds(offset, table.shape[0])], table
        )

    def meta_row(self, meta) -> jax.Array:
        return meta[pl.ds(0, self.num_lanes)]

    def gather_copies(self, slot):
        refs = self.refs
        rows = plsc.Indices(refs.row_vmem.at[slot], ignored_value=IGNORED_ROW)
        return (
            pltpu.make_async_copy(
                refs.nope_cache_hbm.at[rows],
                refs.nope_buffer.at[slot],
                refs.sems.at[SEM_NOPE, slot],
            ),
            pltpu.make_async_copy(
                refs.rope_cache_hbm.at[rows],
                refs.rope_buffer.at[slot],
                refs.sems.at[SEM_ROPE, slot],
            ),
        )

    def out_copies(self, chunk, slot):
        """The chunk's nope, rope and flag write-backs from buffer slot. The
        chunk fills half of a rope block's rope rows. A push stream ignores a
        destination semaphore, and one without a source semaphore signals the
        subcore's shared push flag, so the slot's source semaphore tracks the
        read of its own buffer.
        """
        refs = self.refs
        first_slot = chunk * self.chunk_slots
        block, within = (
            first_slot // ROPE_BLOCK_SLOTS,
            first_slot % ROPE_BLOCK_SLOTS,
        )
        rope_row = (
            block * (ROPE_BLOCK_SLOTS // ROPE_ROW_SLOTS)
            + within % ROPE_HALF_SLOTS // QUAD_TOKENS
        )
        rope_word = within // ROPE_HALF_SLOTS * ROPE_HALF_WORDS
        sem = refs.sems.at[SEM_OUT, slot]
        return (
            pltpu.make_async_copy(
                refs.nope_buffer.at[slot],
                refs.nope_out_hbm.at[pl.ds(first_slot, self.chunk_slots)],
                src_sem=sem,
            ),
            pltpu.make_async_copy(
                refs.rope_stage.at[slot],
                refs.rope_out_hbm.at[
                    pl.ds(rope_row, self.quads_per_chunk),
                    pl.ds(rope_word, ROPE_HALF_WORDS),
                ],
                src_sem=sem,
            ),
            pltpu.make_async_copy(
                refs.flag_vmem.at[slot],
                refs.flag_out_hbm.at[
                    first_slot // FLAG_ROW_SLOTS,
                    pl.ds(first_slot % FLAG_ROW_SLOTS, self.chunk_slots),
                ],
                src_sem=sem,
            ),
        )

    def compute_rows(self, slot, first_slot, index_at, meta, table) -> None:
        """The chunk's physical rows and unused-slot flags; index_at(lane) is
        the vector of the chunk's logical indices from lane on."""
        refs = self.refs
        lanes = self.num_lanes
        page_mask = (1 << self.page_shift) - 1
        bound = jnp.full((lanes,), self.meta_row(meta)[META_POSITION])
        for lane in range(0, self.chunk_slots, lanes):
            index = index_at(lane)
            slot_number = (
                first_slot + lane + lax.broadcasted_iota(jnp.int32, (lanes,), 0)
            )
            is_used = (index >= 0) & (index <= bound) & (slot_number <= bound)
            position = jnp.where(is_used, index, 0)
            page = plsc.load_gather(
                table, [lax.shift_right_logical(position, self.page_shift)]
            )
            row = lax.shift_left(page, self.page_shift) | (position & page_mask)
            refs.row_vmem[slot, pl.ds(lane, lanes)] = jnp.where(
                is_used, row, IGNORED_ROW
            )
            refs.flag_vmem[slot, pl.ds(lane, lanes)] = jnp.where(
                is_used, 0, UNUSED_FLAG
            )

    def start_chunk(self, slot, first_slot, index_at, meta, table) -> None:
        """Starts the gather of the chunk from first_slot into buffer slot, if
        the chunk is active. A write wait would fence every stream of the
        subcore, so drain awaits the writes at exit."""
        is_active = first_slot < self.meta_row(meta)[META_SLOTS]
        state = self.refs.state_smem

        @pl.when(state[STATE_ACTIVE, slot] != 0)
        def _wait_former_out():
            for copy in self.out_copies(0, slot):
                copy.wait_read()

        state[STATE_ACTIVE, slot] = is_active.astype(jnp.int32)

        @pl.when(is_active)
        def _gather():
            self.compute_rows(slot, first_slot, index_at, meta, table)
            for copy in self.gather_copies(slot):
                copy.start()

    def zero_unused_rows(self, slot) -> None:
        """Zeros in the slot's buffer rows of unused slots, which their
        gather skips, so that rows the slot never held are finite."""
        refs = self.refs
        lanes = self.num_lanes
        zeros = jnp.zeros((lanes,), jnp.int32)

        @plsc.parallel_loop(0, self.chunk_slots)
        def _row(row):
            is_unused = (
                plsc.load_gather(
                    refs.flag_vmem.at[slot],
                    [jnp.full((lanes,), row, jnp.int32)],
                )
                != 0
            )
            for buffer, num_words in (
                (refs.nope_buffer, NOPE_WORDS),
                (refs.rope_buffer, ROPE_WORDS),
            ):
                for word in range(0, num_words, lanes):
                    plsc.store_compressed(
                        buffer.at[slot, row, pl.ds(word, lanes)],
                        zeros,
                        mask=is_unused,
                    )

    def transpose_rope(self, slot) -> None:
        """The chunk's rope as 4-token words in rope_stage, a quad a row."""
        refs = self.refs
        lanes = self.num_lanes

        @plsc.parallel_loop(0, self.quads_per_chunk)
        def _quad(quad):
            for lane in range(0, ROPE_WORDS, lanes):
                bands = transpose_quad([
                    refs.rope_buffer[
                        slot, quad * QUAD_TOKENS + token, pl.ds(lane, lanes)
                    ]
                    for token in range(QUAD_TOKENS)
                ])
                for band, words in enumerate(bands):
                    refs.rope_stage[
                        slot, quad, pl.ds(band * ROPE_WORDS + lane, lanes)
                    ] = words

    def finish(self, chunk, slot) -> None:
        """Writes back the chunk in buffer slot once its gathers land."""
        state = self.refs.state_smem
        lanes = self.num_lanes

        @pl.when(state[STATE_ACTIVE, slot] != 0)
        def _write_back():
            for copy in self.gather_copies(slot):
                copy.wait()

            # Note (david): a buffer that held no chunk yet may hold any
            # words where the chunk's gathers skip; after its first chunk it
            # holds rows of the cache and zeros only.
            @pl.when(state[STATE_FILLED, slot] == 0)
            def _first_fill():
                unused = jnp.zeros((lanes,), jnp.int32)
                for lane in range(0, self.chunk_slots, lanes):
                    unused |= self.refs.flag_vmem[slot, pl.ds(lane, lanes)]
                pl.when(jnp.max(unused) != 0)(
                    functools.partial(self.zero_unused_rows, slot)
                )
                state[STATE_FILLED, slot] = 1

            self.transpose_rope(slot)
            for copy in self.out_copies(chunk, slot):
                copy.start()

    def reset(self) -> None:
        """Empty buffers."""
        for slot in range(self.num_buffers):
            self.refs.state_smem[STATE_ACTIVE, slot] = 0
            self.refs.state_smem[STATE_FILLED, slot] = 0

    def drain(self) -> None:
        """Waits the write-backs in flight: a kernel must not exit with DMAs
        in flight."""
        for slot in range(self.num_buffers):

            @pl.when(self.refs.state_smem[STATE_ACTIVE, slot] != 0)
            def _wait_last_out(slot=slot):
                for copy in self.out_copies(0, slot):
                    copy.wait()


def dsa_gather_kernel(
    chunk_hbm: jax.Array,
    indices_hbm: jax.Array,
    token_meta_hbm: jax.Array,
    block_table_hbm: jax.Array,
    nope_cache_hbm: jax.Array,
    rope_cache_hbm: jax.Array,
    nope_out_hbm: jax.Array,
    rope_out_hbm: jax.Array,
    flag_out_hbm: jax.Array,
    index_vmem: jax.Array,
    index_sems: jax.Array,
    meta_vmem: jax.Array,
    table_vmem: jax.Array,
    *ring_scratch: jax.Array,
    num_chunks: int,
    chunks_per_worker: int,
    chunks_per_token: int,
    page_shift: int,
    num_subcores: int,
    num_lanes: int,
) -> None:
    """Each worker gathers its run of chunks in a _GatherRing, chunk c being
    chunk_slots top-k slots of the call's token c // chunks_per_token.

    The call's tokens are the metadata rows from the chunk row's first token
    on, and their index rows from its first index row on. ring_scratch:
    ring_scratch_types.
    """
    ring = _GatherRing(
        RingRefs(
            token_meta_hbm,
            block_table_hbm,
            nope_cache_hbm,
            rope_cache_hbm,
            nope_out_hbm,
            rope_out_hbm,
            flag_out_hbm,
            *ring_scratch,
        ),
        page_shift=page_shift,
        num_lanes=num_lanes,
    )
    num_buffers, chunk_slots = ring.num_buffers, ring.chunk_slots
    worker = lax.axis_index("core") * num_subcores + lax.axis_index("subcore")
    first_chunk = worker * chunks_per_worker
    num_local = jnp.clip(num_chunks - first_chunk, 0, chunks_per_worker)
    chunk_row = ring.load_chunk_row(chunk_hbm)
    first_token = chunk_row[CHUNK_FIRST_TOKEN]
    first_index_slot = chunk_row[CHUNK_FIRST_INDEX_ROW] * (
        chunks_per_token * chunk_slots
    )
    # Note (david): a padding token's rows ignore its indices, which may lie
    # past the index array; its copy reads the array's last chunk instead.
    last_index_slot = indices_hbm.shape[0] - chunk_slots

    def index_copy(local):
        """The copy of the indices of the worker's chunk local, which starts
        while the chunk before it starts."""
        index_slot = jnp.minimum(
            first_index_slot + (first_chunk + local) * chunk_slots,
            last_index_slot,
        )
        slot = lax.rem(local, num_buffers)
        return pltpu.make_async_copy(
            indices_hbm.at[pl.ds(index_slot, chunk_slots)],
            index_vmem.at[slot],
            index_sems.at[slot],
        )

    def start(local):
        chunk = first_chunk + local
        slot = lax.rem(local, num_buffers)
        first_slot = lax.rem(chunk, chunks_per_token) * chunk_slots
        pl.when(local + 1 < num_local)(index_copy(local + 1).start)
        # Note (david): a worker's chunks are consecutive, so a token's rows
        # are loaded once, at its first chunk or the worker's first.
        pl.when((local == 0) | (first_slot == 0))(
            functools.partial(
                ring.load_token,
                first_token + chunk // chunks_per_token,
                meta_vmem,
                table_vmem,
            )
        )
        index_copy(local).wait()
        ring.start_chunk(
            slot,
            first_slot,
            lambda lane: index_vmem[slot, pl.ds(lane, num_lanes)],
            meta_vmem,
            table_vmem,
        )

    @pl.when(num_local > 0)
    def _work():
        ring.reset()
        index_copy(0).start()

        @pl.loop(0, num_local + GATHER_LEAD)
        def _step(step):
            pl.when(step < num_local)(functools.partial(start, step))

            @pl.when(step >= GATHER_LEAD)
            def _finish():
                done = step - GATHER_LEAD
                ring.finish(first_chunk + done, lax.rem(done, num_buffers))

        ring.drain()


class WorkerRows(NamedTuple):
    """A worker's part of the top-k and gather call: its local rows, each
    row's slot chunks [first_chunk, first_chunk + chunks_per_row), and
    whether it selects each row as the ring reaches it."""

    num_rows: jax.Array
    first_chunk: int | jax.Array
    chunks_per_row: int | jax.Array
    selects_in_ring: bool


def worker_rows(topk, chunks_per_token: int) -> WorkerRows:
    """Row mode: the worker's top-k rows whole, each selected as the ring
    reaches it. Split mode: the top-k selects first, then every member of a
    row's group holds the row's selection and gathers its share of the
    row's chunks."""
    if not topk.geometry.is_split:
        return WorkerRows(topk.num_local_rows, 0, chunks_per_token, True)
    topk.run()
    topk.share_selection()
    per_part = pl.cdiv(chunks_per_token, topk.num_parts)
    first_chunk = jnp.minimum(topk.part * per_part, chunks_per_token)
    end = jnp.minimum(first_chunk + per_part, chunks_per_token)
    is_member = topk.column < topk.geometry.num_rows
    return WorkerRows(
        is_member.astype(jnp.int32), first_chunk, end - first_chunk, False
    )


class TokenBuffers(NamedTuple):
    """The top-k and gather call's metadata rows and table rows, of a token
    and the next one, and their prefetch semaphores."""

    meta: jax.Array
    table: jax.Array
    sems: jax.Array


class _SelectGatherRing:
    """A worker's ring over its rows' slot chunks, row-major, whose start of
    a row's first chunk selects the row in row mode: while the ring's last
    chunks of the row before still gather. Each row's metadata and table
    rows are fetched during the row before."""

    def __init__(
        self,
        ring: _GatherRing,
        topk,
        rows: WorkerRows,
        tokens: TokenBuffers,
        first_token: jax.Array,
        *,
        chunks_per_token: int,
    ):
        self.ring = ring
        self.topk = topk
        self.rows = rows
        self.tokens = tokens
        self.first_token = first_token
        self.chunks_per_token = chunks_per_token
        self.num_chunks = rows.num_rows * rows.chunks_per_row

    def place(self, step):
        """(local row, row, slot chunk, chunk within the row) of the
        worker's chunk step."""
        rows = self.rows
        local_row = step // rows.chunks_per_row
        within = step - local_row * rows.chunks_per_row
        return (
            local_row,
            self.topk.row_of(local_row),
            rows.first_chunk + within,
            within,
        )

    def token_rows(self, local_row):
        """(metadata row, table row) buffers of local row local_row."""
        buffer = lax.rem(local_row, NUM_PREFETCH_BUFFERS)
        return self.tokens.meta.at[buffer], self.tokens.table.at[buffer]

    def meta_copy(self, local_row):
        token = self.first_token + self.topk.row_of(local_row)
        meta, _ = self.token_rows(local_row)
        sem = self.tokens.sems.at[SEM_PREFETCH_META]
        return self.ring.meta_copy(token, meta, sem)

    def table_copy(self, local_row):
        meta, table = self.token_rows(local_row)
        sem = self.tokens.sems.at[SEM_PREFETCH_TABLE]
        return self.ring.table_copy(meta, table, sem)

    def start(self, step, stream):
        """Starts chunk step's gather; returns the top-k's stream state."""
        ring, topk = self.ring, self.topk
        local_row, _, chunk, within = self.place(step)
        has_next = local_row + 1 < self.rows.num_rows

        def begin_row(stream):
            if self.rows.selects_in_ring:
                stream = topk.select_row(local_row, stream)
            pl.when(local_row > 0)(lambda: self.table_copy(local_row).wait())
            pl.when(has_next)(lambda: self.meta_copy(local_row + 1).start())
            return stream

        stream = lax.cond(within == 0, begin_row, lambda s: s, stream)

        @pl.when(has_next & (within == self.prefetch_chunk()))
        def _prefetch_table():
            self.meta_copy(local_row + 1).wait()
            self.table_copy(local_row + 1).start()

        first_slot = chunk * ring.chunk_slots
        ring.start_chunk(
            lax.rem(step, ring.num_buffers),
            first_slot,
            self.index_reader(first_slot),
            *self.token_rows(local_row),
        )
        return stream

    def index_reader(self, first_slot):
        """index_at of the chunk from first_slot: the row's selection in the
        top-k's stage, or UNUSED_INDEX in the slots past it. A chunk lies
        wholly inside the selection or past it; one past it reads the
        selection's last chunk, within the stage, and drops it."""
        ring, topk = self.ring, self.topk
        lanes = ring.num_lanes
        num_selected = topk.geometry.k
        if num_selected == self.chunks_per_token * ring.chunk_slots:
            return lambda lane: topk.out_stage[pl.ds(first_slot + lane, lanes)]
        is_selected = jnp.full((lanes,), first_slot) < num_selected
        stage_slot = jnp.minimum(first_slot, num_selected - ring.chunk_slots)
        return lambda lane: jnp.where(
            is_selected,
            topk.out_stage[pl.ds(stage_slot + lane, lanes)],
            UNUSED_INDEX,
        )

    def prefetch_chunk(self):
        """The chunk within a row that starts the next row's table copy."""
        return self.rows.chunks_per_row // TABLE_PREFETCH_SHARE

    def step(self, step, stream):
        """Starts chunk step and finishes chunk step - GATHER_LEAD: one loop
        body, so the top-k's code exists once."""
        stream = lax.cond(
            step < self.num_chunks,
            functools.partial(self.start, step),
            lambda s: s,
            stream,
        )
        done = step - GATHER_LEAD

        @pl.when(done >= 0)
        def _finish():
            _, row, chunk, _ = self.place(done)
            slot = lax.rem(done, self.ring.num_buffers)
            self.ring.finish(row * self.chunks_per_token + chunk, slot)

        return stream

    def run(self) -> None:
        @pl.when(self.num_chunks > 0)
        def _work():
            self.ring.reset()
            # Note (david): a table copy reads its metadata row's table
            # offset, so it is made once that row has landed.
            for make_copy in (self.meta_copy, self.table_copy):
                copy = make_copy(0)
                copy.start()
                copy.wait()
            num_steps = self.num_chunks + GATHER_LEAD
            lax.fori_loop(0, num_steps, self.step, self.topk.empty_stream())
            self.ring.drain()

        if self.rows.selects_in_ring:
            self.topk.finish_rows()


def dsa_topk_gather_kernel(
    chunk_hbm: jax.Array,
    token_meta_hbm: jax.Array,
    block_table_hbm: jax.Array,
    nope_cache_hbm: jax.Array,
    rope_cache_hbm: jax.Array,
    row_lengths_hbm: jax.Array,
    scores_hbm: jax.Array,
    nope_out_hbm: jax.Array,
    rope_out_hbm: jax.Array,
    flag_out_hbm: jax.Array,
    indices_hbm: jax.Array,
    *scratch: jax.Array,
    topk_geometry: TopkGeometry,
    chunks_per_token: int,
    page_shift: int,
    num_lanes: int,
) -> None:
    """The top-k of the call's score rows, and the gather of each token's
    selected rows straight from the top-k's output in the worker's VMEM, in
    a _SelectGatherRing. The top-k writes each row's positions to
    indices_hbm as well. scratch: select_gather_scratch_types, then the
    top-k's."""
    num_tokens_scratch = len(TokenBuffers._fields)
    num_ring_scratch = len(scratch) - len(TOPK_SCRATCH_NAMES)
    ring = _GatherRing(
        RingRefs(
            token_meta_hbm,
            block_table_hbm,
            nope_cache_hbm,
            rope_cache_hbm,
            nope_out_hbm,
            rope_out_hbm,
            flag_out_hbm,
            *scratch[num_tokens_scratch:num_ring_scratch],
        ),
        page_shift=page_shift,
        num_lanes=num_lanes,
    )
    topk = topk_worker(
        row_lengths_hbm,
        scores_hbm,
        indices_hbm,
        *scratch[num_ring_scratch:],
        geometry=topk_geometry,
    )
    first_token = ring.load_chunk_row(chunk_hbm)[CHUNK_FIRST_TOKEN]
    _SelectGatherRing(
        ring,
        topk,
        worker_rows(topk, chunks_per_token),
        TokenBuffers(*scratch[:num_tokens_scratch]),
        first_token,
        chunks_per_token=chunks_per_token,
    ).run()


def select_gather_scratch_types(
    num_buffers: int, chunk_slots: int, pages_per_seq: int
) -> tuple:
    """dsa_topk_gather_kernel's scratch before the top-k's: TokenBuffers,
    then a _GatherRing's in RingRefs order."""
    return (
        pltpu.VMEM((NUM_PREFETCH_BUFFERS, META_WORDS), jnp.int32),
        pltpu.VMEM((NUM_PREFETCH_BUFFERS, pages_per_seq), jnp.int32),
        pltpu.SemaphoreType.DMA((NUM_PREFETCH_SEMS,)),
        *ring_scratch_types(num_buffers, chunk_slots),
    )


def dsa_topk_gather_pallas(
    chunk_row: jax.Array,
    token_meta: jax.Array,
    block_table: jax.Array,
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    row_lengths: jax.Array,
    scores: jax.Array,
    *,
    num_tokens: int,
    top_k: int,
    page_size: int,
    pages_per_seq: int,
    geometry: TopkGeometry,
) -> tuple[jax.Array, ...]:
    """top_k's top-k of num_tokens consecutive tokens' scores and the
    gather of each token's selected rows, as dsa_gather_pallas gathers
    them, in one SparseCore call: the worker that selects a token gathers
    its rows from the selection in its VMEM. geometry is the top-k's of
    the scores and row_lengths within fused_topk_vmem_words; the gather
    takes the rest of a subcore's VMEM.

    Args:
      chunk_row: as dsa_gather_pallas takes it; only its first token is
        read.
      token_meta, block_table, nope_cache, rope_cache, page_size,
        pages_per_seq: as dsa_gather_pallas takes them.
      row_lengths: (num_tokens,) int32 valid scores of each row.
      scores: the tokens' flat scores, as top_k takes them.
      num_tokens: the tokens selected and gathered, geometry.num_rows.
      top_k: slots a token, gather_slots(geometry.k); the slots past the
        geometry.k selected are unused.

    Returns:
      (nope_rows, rope_rows, flag_rows, indices): dsa_gather_pallas's
      outputs, and the top-k's (num_tokens, geometry.k) int32 logical
      positions.
    """
    if (geometry.num_rows, gather_slots(geometry.k)) != (num_tokens, top_k):
        raise ValueError(
            f"The top-k geometry selects {geometry.k} of {geometry.num_rows}"
            f" rows; the gather takes {top_k} slots of {num_tokens}."
        )
    sparse_core = pltpu.get_tpu_info().sparse_core
    topk_scratch = topk_scratch_types(geometry)
    # Note (david): gather_layout counts one token's rows and an index word
    # a slot; the prefetch buffers hold the other tokens' rows.
    prefetch_words = (NUM_PREFETCH_BUFFERS - 1) * (META_WORDS + pages_per_seq)
    chunk_slots, num_buffers = gather_layout(
        num_tokens,
        top_k,
        pages_per_seq,
        sparse_core_vmem_words()
        - scratch_vmem_words(topk_scratch, geometry.num_subcores)
        - prefetch_words,
    )
    body = functools.partial(
        dsa_topk_gather_kernel,
        topk_geometry=geometry,
        chunks_per_token=top_k // chunk_slots,
        page_shift=page_size.bit_length() - 1,
        num_lanes=sparse_core.num_lanes,
    )
    kernel = pl.kernel(
        body,
        out_type=(
            *gathered_types(num_tokens, top_k),
            jax.ShapeDtypeStruct((num_tokens, geometry.k), jnp.int32),
        ),
        mesh=sparse_core_mesh(),
        scratch_types=(
            *select_gather_scratch_types(
                num_buffers, chunk_slots, pages_per_seq
            ),
            *topk_scratch,
        ),
        compiler_params=gather_compiler_params(),
        name="dsa_topk_gather",
    )
    return kernel(
        chunk_row,
        token_meta,
        block_table,
        nope_cache,
        rope_cache,
        worker_major_rows(row_lengths, geometry),
        scores,
    )
