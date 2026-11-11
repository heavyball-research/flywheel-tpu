"""DSA lightning indexer: scores of a ragged batch on a paged index-K cache.

Requests of at most DECODE_MAX_TOKENS new tokens run the decode phase, which
packs their (request, kv block) pairs into items of DECODE_SLOTS token slots.
Longer requests run the prefill phase, which scores a q block of BLOCK_TOKENS
tokens against one kv block per item. Both phases follow a static item schedule
built from the lengths, so one compile serves every batch mix.
"""

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from flywheel_tpu.core.tpu import NUM_LANES, NUM_SUBLANES
from flywheel_tpu.softmax_attention.config import vmem_limit_bytes

INDEX_HEAD_DIM = 128
# A token's index heads fill whole (32, 128) fp8 tiles of query rows.
INDEX_HEADS_MULTIPLE = 32
FP8_DTYPE = jnp.dtype(jnp.float8_e4m3fn)
# The cache packs WORD_BYTES key bytes of one token in a 32-bit word: group g
# holds key dims [4g, 4g + 4), and group SCALE_GROUP holds the UE8M0 scale in
# byte 0 and zeros, so a token takes CACHE_GROUPS words.
WORD_BYTES = 4
KEY_GROUPS = INDEX_HEAD_DIM // WORD_BYTES
SCALE_GROUP = KEY_GROUPS
CACHE_GROUPS = KEY_GROUPS + 1
FP8_MAX = 448.0
# A token's max abs is floored here before its scale is derived.
AMAX_FLOOR = 1e-4
F32_MANTISSA_BITS = 23
F32_MANTISSA_MASK = (1 << F32_MANTISSA_BITS) - 1
F32_EXPONENT_MASK = 0xFF
# The biased exponent of 1 / 2^e is RECIPROCAL_EXPONENT_SUM minus that of 2^e.
RECIPROCAL_EXPONENT_SUM = 254


def ue8m0_exponents(amax: jax.Array) -> jax.Array:
    """Biased exponents (uint32) of the UE8M0 scales
    2^ceil(log2(max(amax, AMAX_FLOOR) / FP8_MAX)) for f32 max abs amax."""
    ratio = jnp.maximum(amax, AMAX_FLOOR) * jnp.float32(1.0 / FP8_MAX)
    bits = lax.bitcast_convert_type(ratio, jnp.uint32)
    exponent = (bits >> F32_MANTISSA_BITS) & F32_EXPONENT_MASK
    return exponent + ((bits & F32_MANTISSA_MASK) != 0).astype(jnp.uint32)


def quantize_columns(columns: jax.Array) -> jax.Array:
    """UE8M0 quantization of NUM_LANES tokens' keys, one per column of the
    (INDEX_HEAD_DIM, NUM_LANES) f32 columns, as cache words: rows [0,
    KEY_GROUPS) the fp8 key words, row SCALE_GROUP the scale exponents, then
    zero rows to whole sublane tiles."""
    exponents = ue8m0_exponents(
        jnp.max(jnp.abs(columns), axis=0, keepdims=True)
    )
    reciprocal = lax.bitcast_convert_type(
        (RECIPROCAL_EXPONENT_SUM - exponents) << F32_MANTISSA_BITS,
        jnp.float32,
    )
    quantized = jnp.clip(columns * reciprocal, -FP8_MAX, FP8_MAX)
    key_words = pltpu.bitcast(quantized.astype(FP8_DTYPE), jnp.uint32)
    padding = jnp.zeros((NUM_SUBLANES - 1, NUM_LANES), jnp.uint32)
    return jnp.concatenate([key_words, exponents, padding], axis=0)


# Note (david): a decode item's slots are its MXU dots; DECODE_SLOTS of them
# keep an item long enough to hide the MXU result latency at its end.
DECODE_SLOTS = 32
# Requests of at most this many new tokens take the decode phase, where their
# tokens share a kv block's keys within an item.
DECODE_MAX_TOKENS = 8
BLOCK_TOKENS = 128
# Rows of a packed bf16 vreg: the tokens of one head-major slab.
SLAB_TOKENS = 16
NUM_SLABS = BLOCK_TOKENS // SLAB_TOKENS
# Query rows (q block tokens x index heads) times key chunks of a prefill kv
# block, so an item's work is the same for every head count.
ITEM_CHUNK_ROWS = 32768
# Heads of one packed bf16 vreg in the decode phase's token-major sum.
HEAD_GROUP = 16
# Lane tiles of keys whose head sums accumulate in the vacc registers at once.
VACC_TILES = 4
CHUNK_KEYS = VACC_TILES * NUM_LANES
DECODE_BLOCK_KV = 1024
PREFILL_STAGES = 2
# Note (david): a decode item's copies are issued DECODE_LOOKAHEAD items ahead,
# so they have whole items to land.
DECODE_LOOKAHEAD = 3
DECODE_STAGES = DECODE_LOOKAHEAD + 1
# Zero pairs past the decode pair table's max_pairs: the items staged past
# the last read up to DECODE_STAGES items of entries beyond it, which then
# need no clamp.
DECODE_PAIR_PADDING = DECODE_STAGES * DECODE_SLOTS
REQUEST_SHIFT = 16
INDEX_MASK = (1 << REQUEST_SHIFT) - 1
SCALE_BYTE_MASK = 0xFF
# VMEM banks a strided load's sublanes spread over (32-bit rows modulo).
VMEM_BANKS = 32
(
    COUNT_DECODE_ITEMS,
    COUNT_LOG2_PAIR_TOKENS,
    COUNT_DECODE_PAIRS,
    COUNT_PREFILL_ITEMS,
    COUNT_PREFILL_BLOCKS,
) = range(5)
SEM_KEYS, SEM_QUERIES, SEM_SCORES, NUM_SEMS = range(4)
# The prefill table's fields of a q block, each max_q_blocks entries long.
(
    BLOCK_REQUEST,
    BLOCK_Q_START,
    BLOCK_Q_COUNT,
    BLOCK_FRONTIER,
    BLOCK_KV_BLOCKS,
    NUM_BLOCK_FIELDS,
) = range(6)


class KvGeometry(NamedTuple):
    """A phase's kv block of block_kv keys, copied in pieces of piece_keys
    lanes (a page, or a window of one) and scored in chunks of chunk_keys."""

    block_kv: int
    piece_keys: int
    pieces: int
    chunk_keys: int
    page_size: int
    pages_per_seq: int


def prefill_block_kv(num_heads: int) -> int:
    """Keys of a prefill kv block for num_heads index heads: whole chunks,
    ITEM_CHUNK_ROWS q block rows times chunks of them, or one chunk."""
    return CHUNK_KEYS * max(1, ITEM_CHUNK_ROWS // (BLOCK_TOKENS * num_heads))


# The longest prefill kv block, at the fewest heads.
MAX_PREFILL_BLOCK_KV = prefill_block_kv(INDEX_HEADS_MULTIPLE)


def kv_geometry(
    block_kv: int, page_size: int, pages_per_seq: int
) -> KvGeometry:
    piece_keys = min(page_size, block_kv)
    return KvGeometry(
        block_kv=block_kv,
        piece_keys=piece_keys,
        pieces=block_kv // piece_keys,
        chunk_keys=min(CHUNK_KEYS, piece_keys),
        page_size=page_size,
        pages_per_seq=pages_per_seq,
    )


class Scalars(NamedTuple):
    """The kernel's scalar-prefetched tables, in SMEM."""

    cu_q_lens: jax.Array
    kv_lens: jax.Array
    block_table: jax.Array
    decode_pairs: jax.Array
    decode_runs: jax.Array
    prefill_table: jax.Array
    counts: jax.Array


class Hbm(NamedTuple):
    """The kernel's HBM operands; weights are lane-broadcast bf16 columns,
    (T, num_heads, NUM_LANES)."""

    queries: jax.Array
    weights: jax.Array
    cache: jax.Array
    scores: jax.Array


def start_piece_copy(hbm, piece_ref, sem, geometry, page, lane, size):
    """Start the copy of size (0 or 1) pieces of the cache, lanes [lane,
    lane + piece_keys) of page, into piece_ref."""
    pltpu.make_async_copy(
        hbm.cache.at[pl.ds(page, size), :, :, pl.ds(lane, geometry.piece_keys)],
        piece_ref.at[pl.ds(0, size), :, :, pl.ds(0, geometry.piece_keys)],
        sem,
    ).start()


def page_and_lane(geometry, block, page_of):
    """The page and first lane of a kv block's pieces: page_of(index) maps a
    request page index to its cache page; block is non-negative, so the
    arithmetic runs unsigned."""
    blocks_per_page = geometry.page_size // geometry.block_kv
    if blocks_per_page:
        unsigned_block = block.astype(jnp.uint32)
        window = (unsigned_block % blocks_per_page).astype(jnp.int32)
        page = page_of((unsigned_block // blocks_per_page).astype(jnp.int32))
        return [(page, pl.multiple_of(window * geometry.block_kv, NUM_LANES))]
    return [
        (page_of(block * geometry.pieces + piece), 0)
        for piece in range(geometry.pieces)
    ]


class PieceCopy(NamedTuple):
    """One piece's copy: size (0 or 1) pieces of lanes [lane, lane +
    piece_keys) of cache page `page`."""

    page: jax.Array
    lane: jax.Array
    size: jax.Array


def plan_key_copies(scalars, geometry, request, block, limit, is_live):
    """The PieceCopy of each of kv block `block`'s pieces of `request`; a
    piece at or past `limit` keys moves nothing."""
    table_row = request * geometry.pages_per_seq
    is_whole = geometry.pieces == 1

    def table_index(index):
        if is_whole:
            return table_row + index
        return table_row + jnp.minimum(index, geometry.pages_per_seq - 1)

    plan = []
    for piece, (index, lane) in enumerate(
        page_and_lane(geometry, block, table_index)
    ):
        is_copied = is_live
        if not is_whole:
            position = block * geometry.block_kv + piece * geometry.piece_keys
            is_copied = jnp.logical_and(is_live, position < limit)
        size = is_copied.astype(jnp.int32)
        page = jnp.where(size > 0, scalars.block_table[index], 0)
        plan.append(PieceCopy(page, jnp.int32(lane), size))
    return tuple(plan)


def start_planned_copies(hbm, keys_stage, sem, geometry, plan):
    """Start a plan's piece copies into pieces [0, len(plan)) of
    keys_stage; returns the live piece count. A plan carried through a loop
    loses its lanes' alignment, which the copy restates."""
    for piece, copy in enumerate(plan):
        start_piece_copy(
            hbm,
            keys_stage.at[pl.ds(piece, 1)],
            sem,
            geometry,
            copy.page,
            pl.multiple_of(copy.lane, NUM_LANES),
            copy.size,
        )
    return sum(copy.size for copy in plan)


def word_rows(keys_ref: jax.Array) -> jax.Array:
    """The staged pieces (..., CACHE_GROUPS, WORD_BYTES, lanes) uint8 as
    32-bit word rows (..., CACHE_GROUPS * lanes / NUM_LANES, NUM_LANES): row
    g * row_tiles + c holds group g of lane tile c."""
    *leading, groups, _, lanes = keys_ref.shape
    return keys_ref.bitcast(jnp.uint32).reshape(
        *leading, groups * (lanes // NUM_LANES), NUM_LANES
    )


def chunk_keys(rows_ref, piece, first_tile, num_tiles):
    """(INDEX_HEAD_DIM, num_tiles * NUM_LANES) fp8 keys, the transposed K a
    dot takes, and their (num_tiles, NUM_LANES) f32 scales (row t for lane
    tile t), from one staged piece's word rows."""
    row_tiles = rows_ref.shape[1] // CACHE_GROUPS
    tiles = [
        pltpu.bitcast(
            rows_ref[
                piece, pl.ds(first_tile + tile, KEY_GROUPS, stride=row_tiles)
            ],
            FP8_DTYPE,
        )
        for tile in range(num_tiles)
    ]
    scale_words = rows_ref[
        piece, pl.ds(SCALE_GROUP * row_tiles + first_tile, num_tiles)
    ]
    scales = lax.bitcast_convert_type(
        (scale_words & SCALE_BYTE_MASK) << F32_MANTISSA_BITS, jnp.float32
    )
    return jnp.concatenate(tiles, axis=1), scales


def staged_lanes(piece_keys: int) -> int:
    """Lanes of a staged piece: its keys, plus a lane tile when a key tile's
    strided load would put two of its sublanes in one VMEM bank."""
    row_tiles = piece_keys // NUM_LANES
    if math.gcd(row_tiles, VMEM_BANKS) > VMEM_BANKS // NUM_SUBLANES:
        return piece_keys + NUM_LANES
    return piece_keys


def bf16_product(relu, weight):
    """relu * weight of two bf16 operands in f32, which lowers to one
    multiply-accumulate per packed vreg."""
    return relu.astype(jnp.float32) * weight.astype(jnp.float32)


def butterfly_rows(parts):
    """Row k of the result is the sum over sublanes of lane tile k of parts."""
    half = NUM_SUBLANES // 2
    while half:
        width = parts.shape[1] // 2
        low, high = parts[:, :width], parts[:, width:]
        row = lax.broadcasted_iota(jnp.int32, low.shape, 0)
        parts = jnp.where(
            (row & half) == 0,
            low + pltpu.roll(low, NUM_SUBLANES - half, 0),
            high + pltpu.roll(high, half, 0),
        )
        half //= 2
    return parts


def token_scores(query, weights, rows_ref, first_piece, geometry):
    """One token's scores against a staged kv block as (block_kv / NUM_LANES,
    NUM_LANES) f32 key tiles."""
    parts = []
    scale_rows = []
    tiles_per_chunk = geometry.chunk_keys // NUM_LANES
    for chunk in range(geometry.block_kv // geometry.chunk_keys):
        start = chunk * geometry.chunk_keys
        keys, scales = chunk_keys(
            rows_ref,
            first_piece + start // geometry.piece_keys,
            (start % geometry.piece_keys) // NUM_LANES,
            tiles_per_chunk,
        )
        scale_rows.append(scales)
        logits = jnp.dot(query, keys, preferred_element_type=jnp.float32)
        relu = jnp.maximum(logits.astype(jnp.bfloat16), 0)
        groups = (query.shape[0] // HEAD_GROUP, HEAD_GROUP, geometry.chunk_keys)
        acc = jnp.sum(
            bf16_product(relu.reshape(groups), weights.reshape(groups)), axis=0
        )
        parts.append(acc[:NUM_SUBLANES] + acc[NUM_SUBLANES:])
    parts = jnp.concatenate(parts, axis=1)
    sums = [
        butterfly_rows(parts[:, first : first + NUM_SUBLANES * NUM_LANES])
        for first in range(0, parts.shape[1], NUM_SUBLANES * NUM_LANES)
    ]
    return jnp.concatenate(sums, axis=0) * jnp.concatenate(scale_rows, axis=0)


def slab_scores(logits, head_weights, slab, num_keys):
    """[SLAB_TOKENS, num_keys] f32 sum over heads of bf16 relu(logits) times
    bf16 weights. Row h * SLAB_TOKENS + t of logits is token t's head h."""
    num_heads = head_weights.shape[1]
    relu = jnp.maximum(logits.astype(jnp.bfloat16), 0).reshape(
        num_heads, SLAB_TOKENS, num_keys
    )
    weights = jnp.concatenate(
        [head_weights[slab]] * (num_keys // NUM_LANES), axis=-1
    )
    return jnp.sum(bf16_product(relu, weights), axis=0)


class DecodeStage(NamedTuple):
    """One stage of DecodeScratch: the buffers an item's copies fill and its
    compute reads."""

    index: jax.Array
    keys: jax.Array
    key_rows: jax.Array
    queries: jax.Array
    weights: jax.Array
    scores: jax.Array
    run_firsts: jax.Array
    run_requests: jax.Array
    counts: jax.Array
    sems: jax.Array

    def sem(self, kind: int) -> jax.Array:
        return self.sems.at[kind, self.index]


class DecodeScratch(NamedTuple):
    """The decode phase's scratch: each array holds DECODE_STAGES buffers along
    its leading axis; item i uses buffer i % DECODE_STAGES."""

    keys: jax.Array
    queries: jax.Array
    weights: jax.Array
    scores: jax.Array
    run_firsts: jax.Array
    run_requests: jax.Array
    counts: jax.Array
    sems: jax.Array

    def stage(self, index: jax.Array) -> DecodeStage:
        return DecodeStage(
            index=index,
            keys=self.keys.at[index],
            key_rows=word_rows(self.keys).at[index],
            queries=self.queries.at[index],
            weights=self.weights.at[index],
            scores=self.scores.at[index],
            run_firsts=self.run_firsts.at[index],
            run_requests=self.run_requests.at[index],
            counts=self.counts.at[index],
            sems=self.sems,
        )


# An entry is one (request, kv block) pair of an item; consecutive entries of
# one request form a run, whose tokens are staged once.
(
    STAGED_FIRST_PAIR,
    STAGED_PIECES,
    STAGED_ENTRIES,
    STAGED_RUNS,
    STAGED_ROWS,
    STAGED_SCORES,
    NUM_STAGED,
) = range(7)


def decode_pair(scalars, index):
    """(request, kv block) of decode pair index."""
    packed = scalars.decode_pairs[index]
    return lax.shift_right_logical(packed, REQUEST_SHIFT), packed & INDEX_MASK


class ItemEntries(NamedTuple):
    """A decode item's entries: entry e is decode pair first + e, live below
    live_entries."""

    first: jax.Array
    live_entries: jax.Array


def item_entries(scalars, item) -> ItemEntries:
    pairs_per_item = DECODE_SLOTS >> scalars.counts[COUNT_LOG2_PAIR_TOKENS]
    first = item * pairs_per_item
    return ItemEntries(
        first=first,
        live_entries=jnp.clip(
            scalars.counts[COUNT_DECODE_PAIRS] - first, 0, pairs_per_item
        ),
    )


def stage_decode_entry(scalars, hbm, stage, geometry, entries, entry, run):
    """Record entry's run, the item's last run before it given, and start its
    key copies, which move nothing for a dead entry; returns (the run, live
    pieces)."""
    request, block = decode_pair(scalars, entries.first + entry)
    is_live = entry < entries.live_entries
    entry_run = scalars.decode_runs[entries.first + entry]
    opens_run = jnp.logical_and(is_live, entry_run != run)
    # Note (david): an entry that opens no run records into the spare last
    # element, so the stores need no branch.
    column = jnp.where(opens_run, entry_run, DECODE_SLOTS)
    stage.run_firsts[column] = entry
    stage.run_requests[column] = request
    plan = plan_key_copies(
        scalars, geometry, request, block, scalars.kv_lens[request], is_live
    )
    live_pieces = start_planned_copies(
        hbm,
        stage.keys.at[pl.ds(entry * geometry.pieces, geometry.pieces)],
        stage.sem(SEM_KEYS),
        geometry,
        plan,
    )
    return jnp.where(is_live, entry_run, run), live_pieces


def record_decode_entries(stage, entries, run, live_pieces):
    stage.counts[STAGED_FIRST_PAIR] = entries.first
    stage.counts[STAGED_ENTRIES] = entries.live_entries
    stage.counts[STAGED_RUNS] = run + 1
    stage.counts[STAGED_PIECES] = live_pieces


def stage_decode_entries(scalars, hbm, stage, geometry, item):
    """Record the item's runs and start its entries' key copies."""
    entries = item_entries(scalars, item)
    run = jnp.int32(-1)
    live_pieces = 0
    for entry in range(DECODE_SLOTS):
        run, pieces = stage_decode_entry(
            scalars, hbm, stage, geometry, entries, entry, run
        )
        live_pieces += pieces
    record_decode_entries(stage, entries, run, live_pieces)


def run_bounds(stage, run):
    """(first entry, entry count, request) of one of the item's runs."""
    counts = stage.counts
    first = stage.run_firsts[run]
    is_last = run + 1 >= counts[STAGED_RUNS]
    end = jnp.where(
        is_last,
        counts[STAGED_ENTRIES],
        stage.run_firsts[jnp.minimum(run + 1, DECODE_SLOTS)],
    )
    return first, end - first, stage.run_requests[run]


def start_decode_queries(scalars, hbm, stage):
    """Start each run's query and weight copies, its request's tokens to
    rows [run * pair tokens, + q_len)."""
    log2_tokens = scalars.counts[COUNT_LOG2_PAIR_TOKENS]

    def start_run(run, rows):
        _, _, request = run_bounds(stage, run)
        q_start = scalars.cu_q_lens[request]
        q_len = scalars.cu_q_lens[request + 1] - q_start
        for source, stage_ref in (
            (hbm.queries, stage.queries),
            (hbm.weights, stage.weights),
        ):
            pltpu.make_async_copy(
                source.at[pl.ds(q_start, q_len)],
                stage_ref.at[pl.ds(run << log2_tokens, q_len)],
                stage.sem(SEM_QUERIES),
            ).start()
        return rows + q_len

    stage.counts[STAGED_ROWS] = lax.fori_loop(
        0, stage.counts[STAGED_RUNS], start_run, 0
    )


def wait_staged(stage_ref, count, sem):
    """Wait for count rows of stage_ref's copies, which signal sem."""
    staged = stage_ref.at[pl.ds(0, count)]
    pltpu.make_async_copy(staged, staged, sem).wait()


def compute_decode_slot(scalars, stage, geometry, item, slot):
    """Score slot (entry e, token offset o) against its staged kv block into
    score row o * pairs_per_item + e, so an offset's entries are adjacent."""
    log2_tokens = scalars.counts[COUNT_LOG2_PAIR_TOKENS]
    pairs_per_item = DECODE_SLOTS >> log2_tokens
    tiles = geometry.block_kv // NUM_LANES
    entry = slot >> log2_tokens
    offset = slot & ((1 << log2_tokens) - 1)
    run = scalars.decode_runs[item * pairs_per_item + entry]
    row = (run << log2_tokens) + offset
    weights = jnp.concatenate(
        [stage.weights[row]] * (geometry.chunk_keys // NUM_LANES), axis=1
    )
    score_index = offset * pairs_per_item + entry
    stage.scores[pl.ds(pl.multiple_of(score_index * tiles, tiles), tiles)] = (
        token_scores(
            stage.queries[row],
            weights,
            stage.key_rows,
            entry * geometry.pieces,
            geometry,
        )
    )


def start_decode_scores(scalars, hbm, stage, geometry):
    """Start, per run and token, one copy of the run's adjacent score rows
    to the token's row, and record the rows copied for the wait."""
    log2_tokens = scalars.counts[COUNT_LOG2_PAIR_TOKENS]
    pairs_per_item = DECODE_SLOTS >> log2_tokens
    tiles = geometry.block_kv // NUM_LANES

    def start_run(run, rows):
        first, length, request = run_bounds(stage, run)
        q_start = scalars.cu_q_lens[request]
        _, block = decode_pair(scalars, stage.counts[STAGED_FIRST_PAIR] + first)
        num_rows = pl.multiple_of(length * tiles, tiles)

        def start_token(offset, carry):
            source = (offset * pairs_per_item + first) * tiles
            pltpu.make_async_copy(
                stage.scores.at[pl.ds(pl.multiple_of(source, tiles), num_rows)],
                hbm.scores.at[
                    q_start + offset,
                    pl.ds(pl.multiple_of(block * tiles, tiles), num_rows),
                ],
                stage.sem(SEM_SCORES),
            ).start()
            return carry

        q_len = scalars.cu_q_lens[request + 1] - q_start
        lax.fori_loop(0, q_len, start_token, 0)
        return rows + q_len * num_rows

    stage.counts[STAGED_SCORES] = lax.fori_loop(
        0, stage.counts[STAGED_RUNS], start_run, 0
    )


def wait_decode_scores(stage):
    wait_staged(
        stage.scores, stage.counts[STAGED_SCORES], stage.sem(SEM_SCORES)
    )


def run_decode_item(scalars, hbm, scratch, geometry, item):
    """Wait for the item's copies, compute its slots, stage item +
    DECODE_LOOKAHEAD, and start the item's score copies. Entry e of the ahead
    item stages after slot e, since an earlier slot may read its buffer."""
    stage = scratch.stage(lax.rem(item, DECODE_STAGES))
    ahead_item = item + DECODE_LOOKAHEAD
    ahead = scratch.stage(lax.rem(ahead_item, DECODE_STAGES))
    wait_staged(
        stage.keys.at[:, :, :, pl.ds(0, geometry.piece_keys)],
        stage.counts[STAGED_PIECES],
        stage.sem(SEM_KEYS),
    )
    for stage_ref in (stage.queries, stage.weights):
        wait_staged(
            stage_ref, stage.counts[STAGED_ROWS], stage.sem(SEM_QUERIES)
        )

    @pl.when(item >= DECODE_STAGES)
    def _free_scores():
        wait_decode_scores(stage)

    entries = item_entries(scalars, ahead_item)
    run = jnp.int32(-1)
    live_pieces = 0
    for slot in range(DECODE_SLOTS):
        compute_decode_slot(scalars, stage, geometry, item, slot)
        run, pieces = stage_decode_entry(
            scalars, hbm, ahead, geometry, entries, slot, run
        )
        live_pieces += pieces
    record_decode_entries(ahead, entries, run, live_pieces)
    start_decode_queries(scalars, hbm, ahead)
    start_decode_scores(scalars, hbm, stage, geometry)


def run_decode_phase(scalars, hbm, scratch, geometry):
    """The decode items, one loop step each on the stage of its index."""
    num_items = scalars.counts[COUNT_DECODE_ITEMS]

    def stage_item(item, carry):
        stage = scratch.stage(item)
        stage_decode_entries(scalars, hbm, stage, geometry, item)
        start_decode_queries(scalars, hbm, stage)
        return carry

    def run_item(item, carry):
        run_decode_item(scalars, hbm, scratch, geometry, item)
        return carry

    def drain(item, carry):
        # Note (david): the last item of each stage still has score copies
        # in flight.
        @pl.when(item < num_items)
        def _drain():
            wait_decode_scores(scratch.stage(item))

        return carry

    lax.fori_loop(0, DECODE_LOOKAHEAD, stage_item, 0)
    lax.fori_loop(0, num_items, run_item, 0)
    lax.fori_loop(0, DECODE_STAGES, drain, 0)


# A stage's SMEM record: the rows its last score copy moves.
ITEM_SCORES, NUM_ITEM_FIELDS = range(2)


class PrefillStage(NamedTuple):
    """One stage of the prefill phase's keys, scores and item record."""

    index: jax.Array
    keys: jax.Array
    key_rows: jax.Array
    scores: jax.Array
    score_rows: jax.Array
    items: jax.Array
    sems: jax.Array

    def sem(self, kind: int) -> jax.Array:
        return self.sems.at[kind, self.index]


class PrefillScratch(NamedTuple):
    """The prefill phase's scratch: keys, scores and item records
    PREFILL_STAGES buffers along their leading axis, item i in buffer i %
    PREFILL_STAGES; queries and weights a q block parity."""

    keys: jax.Array
    queries: jax.Array
    weights: jax.Array
    head_queries: jax.Array
    head_weights: jax.Array
    scores: jax.Array
    items: jax.Array
    sems: jax.Array

    def stage(self, index: jax.Array) -> PrefillStage:
        stages, tokens, pitch, _ = self.scores.shape
        score_rows = self.scores.reshape(stages, tokens * pitch, NUM_LANES)
        return PrefillStage(
            index=index,
            keys=self.keys.at[index],
            key_rows=word_rows(self.keys).at[index],
            scores=self.scores.at[index],
            score_rows=score_rows.at[index],
            items=self.items.at[index],
            sems=self.sems,
        )


class QueryBlock(NamedTuple):
    """A prefill q block: tokens [q_start, q_start + q_count) of request,
    whose last token sees frontier keys, scored in num_blocks kv blocks."""

    request: jax.Array
    q_start: jax.Array
    q_count: jax.Array
    frontier: jax.Array
    num_blocks: jax.Array


def query_block(scalars, ordinal) -> QueryBlock:
    """Prefill q block `ordinal` from the prefill table, its fields read
    independently of each other."""
    max_q_blocks = scalars.prefill_table.shape[0] // NUM_BLOCK_FIELDS
    index = jnp.minimum(ordinal, max_q_blocks - 1)
    fields = [
        scalars.prefill_table[field * max_q_blocks + index]
        for field in range(NUM_BLOCK_FIELDS)
    ]
    return QueryBlock(
        request=fields[BLOCK_REQUEST],
        q_start=fields[BLOCK_Q_START],
        q_count=fields[BLOCK_Q_COUNT],
        frontier=fields[BLOCK_FRONTIER],
        num_blocks=fields[BLOCK_KV_BLOCKS],
    )


class PrefillItem(NamedTuple):
    """A prefill item: kv block `block` of a q block whose tokens [q_start,
    q_start + q_count) its scores cover, and its key copies."""

    block: jax.Array
    q_start: jax.Array
    q_count: jax.Array
    copies: tuple[PieceCopy, ...]


def plan_prefill_item(scalars, geometry, item, state):
    """The PrefillItem of item at (q block, kv block) state, all scalar
    work, and the next item's state."""
    ordinal, block = state
    is_live = item < scalars.counts[COUNT_PREFILL_ITEMS]
    query = query_block(scalars, ordinal)
    copies = plan_key_copies(
        scalars, geometry, query.request, block, query.frontier, is_live
    )
    wraps = block + 1 >= query.num_blocks
    next_state = (
        ordinal + wraps.astype(jnp.int32),
        jnp.where(wraps, 0, block + 1),
    )
    return PrefillItem(block, query.q_start, query.q_count, copies), next_state


def build_head_major(scratch, parity):
    """The q block's queries and weight columns as head-major slabs: query
    row h * SLAB_TOKENS + t and weight column [h, t] hold token t's head h."""
    num_heads = scratch.queries.shape[2]

    def build_slab(slab, carry):
        first = pl.multiple_of(slab * SLAB_TOKENS, SLAB_TOKENS)
        tokens = pl.ds(first, SLAB_TOKENS)
        scratch.head_queries[slab] = jnp.swapaxes(
            scratch.queries[parity, tokens], 0, 1
        ).reshape(num_heads * SLAB_TOKENS, INDEX_HEAD_DIM)
        scratch.head_weights[slab] = jnp.swapaxes(
            scratch.weights[parity, tokens], 0, 1
        )
        return carry

    lax.fori_loop(0, NUM_SLABS, build_slab, 0)


def start_query_copies(scalars, hbm, scratch, ordinal):
    """Start q block ordinal's query and weight copies into the buffers of
    its parity; past the last q block they move nothing."""
    query = query_block(scalars, ordinal)
    is_live = ordinal < scalars.counts[COUNT_PREFILL_BLOCKS]
    count = jnp.where(is_live, query.q_count, 0)
    parity = lax.rem(ordinal, PREFILL_STAGES)
    for source, stage_ref in (
        (hbm.queries, scratch.queries),
        (hbm.weights, scratch.weights),
    ):
        pltpu.make_async_copy(
            source.at[pl.ds(jnp.where(is_live, query.q_start, 0), count)],
            stage_ref.at[parity, pl.ds(0, count)],
            scratch.sems.at[SEM_QUERIES, parity],
        ).start()


def open_query_block(scalars, hbm, scratch, ordinal):
    """Wait for q block ordinal's query and weight copies, start the next q
    block's into the other parity, and build ordinal's head-major slabs."""
    parity = lax.rem(ordinal, PREFILL_STAGES)
    count = query_block(scalars, ordinal).q_count
    for stage_ref in (scratch.queries, scratch.weights):
        wait_staged(
            stage_ref.at[parity], count, scratch.sems.at[SEM_QUERIES, parity]
        )
    start_query_copies(scalars, hbm, scratch, ordinal + 1)
    build_head_major(scratch, parity)


def compute_prefill_item(scratch, stage, geometry):
    """Score the open q block's tokens against the staged kv block into the
    score stage, a token's key tiles score_pitch rows apart."""
    pitch = stage.scores.shape[1]
    tiles_per_chunk = geometry.chunk_keys // NUM_LANES
    for chunk in range(geometry.block_kv // geometry.chunk_keys):
        start = chunk * geometry.chunk_keys
        keys, scales = chunk_keys(
            stage.key_rows,
            start // geometry.piece_keys,
            (start % geometry.piece_keys) // NUM_LANES,
            tiles_per_chunk,
        )
        scale_row = jnp.concatenate(
            [scales[tile : tile + 1] for tile in range(tiles_per_chunk)], axis=1
        )
        for slab in range(NUM_SLABS):
            logits = jnp.dot(
                scratch.head_queries[slab],
                keys,
                preferred_element_type=jnp.float32,
            )
            acc = (
                slab_scores(
                    logits, scratch.head_weights, slab, geometry.chunk_keys
                )
                * scale_row
            )
            for tile in range(tiles_per_chunk):
                first_row = (
                    slab * SLAB_TOKENS * pitch + chunk * tiles_per_chunk + tile
                )
                stage.score_rows[
                    pl.ds(first_row, SLAB_TOKENS, stride=pitch)
                ] = acc[:, tile * NUM_LANES : (tile + 1) * NUM_LANES]


def prefill_scores_window(stage, geometry, count):
    tiles = geometry.block_kv // NUM_LANES
    return stage.scores.at[pl.ds(0, count), pl.ds(0, tiles)]


def start_prefill_scores(hbm, stage, geometry, record):
    """Start the copy of the item's scores to its tokens' rows."""
    tiles = geometry.block_kv // NUM_LANES
    pltpu.make_async_copy(
        prefill_scores_window(stage, geometry, record.q_count),
        hbm.scores.at[
            pl.ds(record.q_start, record.q_count),
            pl.ds(pl.multiple_of(record.block * tiles, tiles), tiles),
        ],
        stage.sem(SEM_SCORES),
    ).start()
    stage.items[ITEM_SCORES] = record.q_count


def wait_prefill_scores(stage, geometry):
    window = prefill_scores_window(stage, geometry, stage.items[ITEM_SCORES])
    pltpu.make_async_copy(window, window, stage.sem(SEM_SCORES)).wait()


def start_prefill_keys(hbm, scratch, geometry, item, record):
    """Start the item's key copies into its stage; returns the live piece
    count."""
    stage = scratch.stage(lax.rem(item, PREFILL_STAGES))
    return start_planned_copies(
        hbm, stage.keys, stage.sem(SEM_KEYS), geometry, record.copies
    )


def run_prefill_item(scalars, hbm, scratch, geometry, item, carry):
    """Start the next item's key copies, wait for this item's keys and the
    score copy that last read its stage, score, and start this item's score
    copy. Item + 2 is planned among the compute."""
    record, pieces, next_record, state = carry
    next_pieces = start_prefill_keys(
        hbm, scratch, geometry, item + 1, next_record
    )
    stage = scratch.stage(lax.rem(item, PREFILL_STAGES))
    wait_staged(
        stage.keys.at[:, :, :, pl.ds(0, geometry.piece_keys)],
        pieces,
        stage.sem(SEM_KEYS),
    )
    wait_prefill_scores(stage, geometry)
    compute_prefill_item(scratch, stage, geometry)
    later_record, later_state = plan_prefill_item(
        scalars, geometry, item + 2, state
    )
    start_prefill_scores(hbm, stage, geometry, record)
    return next_record, next_pieces, later_record, later_state


def run_prefill_phase(scalars, hbm, scratch, geometry):
    """The prefill q blocks, each opened once and then scored against its kv
    blocks, one item per loop step."""

    def run_query_block(ordinal, carry):
        first_item, items = carry
        open_query_block(scalars, hbm, scratch, ordinal)
        end_item = first_item + query_block(scalars, ordinal).num_blocks
        items = lax.fori_loop(
            first_item,
            end_item,
            lambda item, items: run_prefill_item(
                scalars, hbm, scratch, geometry, item, items
            ),
            items,
        )
        return end_item, items

    zero = jnp.int32(0)
    for stage_index in range(PREFILL_STAGES):
        # Note (david): a stage's first score wait then waits for no rows.
        scratch.stage(stage_index).items[ITEM_SCORES] = 0
    start_query_copies(scalars, hbm, scratch, zero)
    first_record, state = plan_prefill_item(
        scalars, geometry, zero, (zero, zero)
    )
    second_record, state = plan_prefill_item(scalars, geometry, 1, state)
    first_pieces = start_prefill_keys(hbm, scratch, geometry, 0, first_record)
    lax.fori_loop(
        0,
        scalars.counts[COUNT_PREFILL_BLOCKS],
        run_query_block,
        (zero, (first_record, first_pieces, second_record, state)),
    )
    for stage_index in range(PREFILL_STAGES):
        wait_prefill_scores(scratch.stage(stage_index), geometry)


def score_pitch(block_kv: int) -> int:
    """Rows between two tokens' key tiles in the prefill score stage: the
    block's tiles, padded off a multiple of 16 rows so that a strided store's
    sublanes fall in distinct VMEM banks."""
    tiles = block_kv // NUM_LANES
    return tiles + NUM_SUBLANES if tiles % (2 * NUM_SUBLANES) == 0 else tiles


def staged_keys_shape(geometry: KvGeometry, num_pieces: int, num_stages: int):
    return pltpu.VMEM(
        (
            num_stages,
            num_pieces,
            CACHE_GROUPS,
            WORD_BYTES,
            staged_lanes(geometry.piece_keys),
        ),
        jnp.uint8,
    )


def decode_scratch_shapes(geometry: KvGeometry, num_heads: int):
    tiles = geometry.block_kv // NUM_LANES
    return (
        staged_keys_shape(
            geometry, DECODE_SLOTS * geometry.pieces, DECODE_STAGES
        ),
        pltpu.VMEM(
            (DECODE_STAGES, DECODE_SLOTS, num_heads, INDEX_HEAD_DIM), FP8_DTYPE
        ),
        pltpu.VMEM(
            (DECODE_STAGES, DECODE_SLOTS, num_heads, NUM_LANES), jnp.bfloat16
        ),
        pltpu.VMEM(
            (DECODE_STAGES, DECODE_SLOTS * tiles, NUM_LANES), jnp.float32
        ),
        pltpu.SMEM((DECODE_STAGES, DECODE_SLOTS + 1), jnp.int32),
        pltpu.SMEM((DECODE_STAGES, DECODE_SLOTS + 1), jnp.int32),
        pltpu.SMEM((DECODE_STAGES, NUM_STAGED), jnp.int32),
        pltpu.SemaphoreType.DMA((NUM_SEMS, DECODE_STAGES)),
    )


def prefill_scratch_shapes(geometry: KvGeometry, num_heads: int):
    return (
        staged_keys_shape(geometry, geometry.pieces, PREFILL_STAGES),
        pltpu.VMEM(
            (PREFILL_STAGES, BLOCK_TOKENS, num_heads, INDEX_HEAD_DIM),
            FP8_DTYPE,
        ),
        pltpu.VMEM(
            (PREFILL_STAGES, BLOCK_TOKENS, num_heads, NUM_LANES),
            jnp.bfloat16,
        ),
        pltpu.VMEM(
            (NUM_SLABS, num_heads * SLAB_TOKENS, INDEX_HEAD_DIM), FP8_DTYPE
        ),
        pltpu.VMEM(
            (NUM_SLABS, num_heads, SLAB_TOKENS, NUM_LANES), jnp.bfloat16
        ),
        pltpu.VMEM(
            (
                PREFILL_STAGES,
                BLOCK_TOKENS,
                score_pitch(geometry.block_kv),
                NUM_LANES,
            ),
            jnp.float32,
        ),
        pltpu.SMEM((PREFILL_STAGES, NUM_ITEM_FIELDS), jnp.int32),
        pltpu.SemaphoreType.DMA((NUM_SEMS, PREFILL_STAGES)),
    )


def indexer_kernel(
    cu_q_lens_ref,
    kv_lens_ref,
    block_table_ref,
    decode_pairs_ref,
    decode_runs_ref,
    prefill_table_ref,
    counts_ref,
    queries_hbm,
    weights_hbm,
    cache_hbm,
    scores_hbm,
    *,
    decode_geometry: KvGeometry,
    prefill_geometry: KvGeometry,
) -> None:
    """Run the decode phase, then the prefill phase, each with its own
    scratch."""
    scalars = Scalars(
        cu_q_lens_ref,
        kv_lens_ref,
        block_table_ref,
        decode_pairs_ref,
        decode_runs_ref,
        prefill_table_ref,
        counts_ref,
    )
    hbm = Hbm(queries_hbm, weights_hbm, cache_hbm, scores_hbm)
    num_heads = queries_hbm.shape[1]
    pl.run_scoped(
        lambda *refs: run_decode_phase(
            scalars, hbm, DecodeScratch(*refs), decode_geometry
        ),
        *decode_scratch_shapes(decode_geometry, num_heads),
    )
    pl.run_scoped(
        lambda *refs: run_prefill_phase(
            scalars, hbm, PrefillScratch(*refs), prefill_geometry
        ),
        *prefill_scratch_shapes(prefill_geometry, num_heads),
    )


def ceil_log2(values: jax.Array, max_log2: int) -> jax.Array:
    return sum(
        (values > (1 << power)).astype(jnp.int32) for power in range(max_log2)
    )


def request_of(ends: jax.Array, count: int) -> jax.Array:
    """For each of count positions, the request whose range of the cumulative
    ends holds it. A dense compare and sum; a gather lowers to selects."""
    position = jnp.arange(count, dtype=jnp.int32)
    return jnp.sum(ends[None, :] <= position[:, None], axis=1).astype(jnp.int32)


def take_request(values: jax.Array, request: jax.Array) -> jax.Array:
    """values[request] as a dense one-hot sum, for the reason request_of
    gives."""
    batch = values.shape[0]
    hits = request[:, None] == jnp.arange(batch, dtype=jnp.int32)[None, :]
    return jnp.sum(jnp.where(hits, values[None, :], 0), axis=1)


def decode_pair_table(
    q_lens: jax.Array, kv_lens: jax.Array, max_pairs: int
) -> tuple[jax.Array, jax.Array]:
    """The decode requests' (request, kv block) pairs in request order as
    request << REQUEST_SHIFT | block, max_pairs entries and
    DECODE_PAIR_PADDING zeros, and their count."""
    batch = kv_lens.shape[0]
    is_decode = jnp.logical_and(q_lens >= 1, q_lens <= DECODE_MAX_TOKENS)
    request_pairs = jnp.where(is_decode, pl.cdiv(kv_lens, DECODE_BLOCK_KV), 0)
    pair_end = jnp.cumsum(request_pairs).astype(jnp.int32)
    request = jnp.minimum(request_of(pair_end, max_pairs), batch - 1)
    pair = jnp.arange(max_pairs, dtype=jnp.int32)
    block = pair - take_request(pair_end - request_pairs, request)
    table = jnp.where(
        pair < pair_end[-1], (request << REQUEST_SHIFT) | block, 0
    )
    table = jnp.pad(table.astype(jnp.int32), (0, DECODE_PAIR_PADDING))
    return table, pair_end[-1]


def decode_run_table(
    decode_pairs: jax.Array, num_pairs: jax.Array, log2_pair_tokens: jax.Array
) -> jax.Array:
    """Each decode pair's run within its item, zero past num_pairs: item i
    holds pairs [i * pairs_per_item, (i + 1) * pairs_per_item), and a run is
    consecutive pairs of one request."""
    num_entries = decode_pairs.shape[0]
    pair = jnp.arange(num_entries, dtype=jnp.int32)
    entry = pair & ((DECODE_SLOTS >> log2_pair_tokens) - 1)
    request = lax.shift_right_logical(decode_pairs, REQUEST_SHIFT)
    previous = jnp.pad(request, (1, 0), constant_values=-1)[:-1]
    opens = jnp.logical_or(entry == 0, request != previous).astype(jnp.int32)
    # Note (david): a run counts the opens from its item's first pair on, at
    # most DECODE_SLOTS - 1 pairs back: a fixed window XLA sums in one pass.
    padded = jnp.pad(opens, (DECODE_SLOTS - 1, 0))
    runs = -1
    for lag in range(DECODE_SLOTS):
        start = DECODE_SLOTS - 1 - lag
        runs = runs + jnp.where(
            lag <= entry, padded[start : start + num_entries], 0
        )
    return jnp.where(pair < num_pairs, runs, 0)


def prefill_table(
    cu_q_lens: jax.Array, kv_lens: jax.Array, max_q_blocks: int, block_kv: int
) -> jax.Array:
    """The prefill q blocks' fields, field-major: entry f * max_q_blocks + j
    is field f (BLOCK_REQUEST, ...) of q block j, zero past the last; kv
    blocks hold block_kv keys."""
    batch = kv_lens.shape[0]
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    request_blocks = jnp.where(
        q_lens > DECODE_MAX_TOKENS, pl.cdiv(q_lens, BLOCK_TOKENS), 0
    )
    block_end = jnp.cumsum(request_blocks).astype(jnp.int32)
    request = jnp.minimum(request_of(block_end, max_q_blocks), batch - 1)
    ordinal = jnp.arange(max_q_blocks, dtype=jnp.int32)
    index = ordinal - take_request(block_end - request_blocks, request)
    q_start = take_request(cu_q_lens[:-1], request) + index * BLOCK_TOKENS
    q_end = take_request(cu_q_lens[1:], request)
    q_count = jnp.minimum(BLOCK_TOKENS, q_end - q_start)
    frontier = take_request(kv_lens, request) - q_end + q_start + q_count
    # In the order of the BLOCK_* fields.
    fields = (
        request,
        q_start,
        q_count,
        frontier,
        pl.cdiv(frontier, block_kv),
    )
    is_live = ordinal < block_end[-1]
    return jnp.concatenate(
        [jnp.where(is_live, field, 0) for field in fields]
    ).astype(jnp.int32)


def indexer_schedule(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    *,
    max_pairs: int,
    max_q_blocks: int,
    block_kv: int,
) -> tuple[jax.Array, ...]:
    """The call's work as tables, a pure function of the lengths, so XLA can
    build them once per step: the decode pair and run tables, the prefill
    table over kv blocks of block_kv keys, and the counts."""
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    decode_pairs, num_pairs = decode_pair_table(q_lens, kv_lens, max_pairs)
    is_decode = jnp.logical_and(q_lens >= 1, q_lens <= DECODE_MAX_TOKENS)
    log2_pair_tokens = ceil_log2(
        jnp.max(jnp.where(is_decode, q_lens, 1)),
        DECODE_MAX_TOKENS.bit_length() - 1,
    )
    prefill = prefill_table(cu_q_lens, kv_lens, max_q_blocks, block_kv)
    kv_blocks = prefill[
        BLOCK_KV_BLOCKS * max_q_blocks : (BLOCK_KV_BLOCKS + 1) * max_q_blocks
    ]
    # In the order of the COUNT_* fields; a live q block has a kv block.
    counts = jnp.stack([
        pl.cdiv(num_pairs << log2_pair_tokens, DECODE_SLOTS),
        log2_pair_tokens,
        num_pairs,
        jnp.sum(kv_blocks),
        jnp.sum(kv_blocks > 0),
    ]).astype(jnp.int32)
    decode_runs = decode_run_table(decode_pairs, num_pairs, log2_pair_tokens)
    return decode_pairs, decode_runs, prefill, counts


def indexer_operands(
    queries: jax.Array,
    weights: jax.Array,
    index_cache: jax.Array,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
    num_columns: int,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    """The kernel's scalar prefetches and HBM inputs."""
    num_tokens, num_heads, _ = queries.shape
    batch = block_table.shape[0]
    # Note (david): a request takes the prefill phase only past
    # DECODE_MAX_TOKENS tokens, so these bound its q blocks; a decode request
    # has a token, so at most min(batch, num_tokens) take the decode phase.
    max_q_blocks = max(
        1,
        num_tokens // BLOCK_TOKENS
        + min(batch, num_tokens // (DECODE_MAX_TOKENS + 1)),
    )
    max_decode_requests = min(batch, num_tokens)
    scalar_prefetches = (
        cu_q_lens,
        kv_lens,
        block_table.reshape(-1),
        *indexer_schedule(
            cu_q_lens,
            kv_lens,
            max_pairs=max_decode_requests * (num_columns // DECODE_BLOCK_KV),
            max_q_blocks=max_q_blocks,
            block_kv=prefill_block_kv(num_heads),
        ),
    )
    # Note (david): the kernel reads a head's weight across lanes, so the
    # weights arrive as lane-broadcast bf16 columns.
    weight_columns = jnp.broadcast_to(
        weights.astype(jnp.bfloat16)[:, :, None],
        (*weights.shape, NUM_LANES),
    )
    # Note (david): XLA may place a small operand in VMEM, where the kernel's
    # copies of it would lower to vector load and store loops; the inputs
    # stay in HBM.
    inputs = tuple(
        pltpu.with_memory_space_constraint(x, pltpu.HBM)
        for x in (queries, weight_columns, index_cache)
    )
    return scalar_prefetches, inputs


def indexer_scores_pallas(
    queries: jax.Array,
    weights: jax.Array,
    index_cache: jax.Array,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
    *,
    num_columns: int,
) -> jax.Array:
    """dsa_indexer_scores's pallas_call on checked operands: scores (T,
    num_columns / NUM_LANES, NUM_LANES) f32, row-major per token."""
    page_size = index_cache.shape[-1]
    pages_per_seq = block_table.shape[1]
    scalar_prefetches, inputs = indexer_operands(
        queries,
        weights,
        index_cache,
        cu_q_lens,
        kv_lens,
        block_table,
        num_columns,
    )
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    call = pl.pallas_call(
        functools.partial(
            indexer_kernel,
            decode_geometry=kv_geometry(
                DECODE_BLOCK_KV, page_size, pages_per_seq
            ),
            prefill_geometry=kv_geometry(
                prefill_block_kv(queries.shape[1]), page_size, pages_per_seq
            ),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            grid=(1,),
            in_specs=[hbm_spec] * len(inputs),
            out_specs=hbm_spec,
        ),
        out_shape=jax.ShapeDtypeStruct(
            (queries.shape[0], num_columns // NUM_LANES, NUM_LANES),
            jnp.float32,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_bytes(),
            disable_bounds_checks=True,
        ),
        name="dsa_indexer_scores",
    )
    return call(*scalar_prefetches, *inputs)
