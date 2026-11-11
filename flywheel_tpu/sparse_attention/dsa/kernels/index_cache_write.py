"""Ragged write of new index keys into the paged DSA index-K cache: each
token's bf16 key is quantized to float8_e4m3fn under its UE8M0 scale and
lands in its cache lane."""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from flywheel_tpu.core.tpu import BF16_TILE_ROWS, NUM_LANES, NUM_SUBLANES
from flywheel_tpu.softmax_attention.config import vmem_limit_bytes
from flywheel_tpu.sparse_attention.dsa.kernels.indexer import (
    CACHE_GROUPS,
    INDEX_HEAD_DIM,
    WORD_BYTES,
    quantize_columns,
    request_of,
)

# Note (david): tiles are written TILE_GROUP at a time without a branch, and
# partial-tile reads start READ_AHEAD_GROUPS groups ahead to hide their latency.
TILE_GROUP = 8
READ_AHEAD_GROUPS = 2
TILE_STAGES = 2 * READ_AHEAD_GROUPS
# Rows of a lane tile's words: the key groups and the scale group, padded to
# whole sublane tiles.
TILE_WORD_ROWS = pl.cdiv(CACHE_GROUPS, NUM_SUBLANES) * NUM_SUBLANES
COUNT_TILES, COUNT_WINDOWS = range(2)
# Note (david): a DMA wait returns once its semaphore has counted its bytes,
# so each stage has its own read and write semaphores; a shared one could let
# a later group's bytes release an earlier group's wait.
SEM_READ, SEM_WRITE, NUM_SEMS = range(3)


class Tile(NamedTuple):
    """One lane tile of a request's chunk: cache lanes [position, position +
    NUM_LANES), with the chunk at [chunk_start, chunk_end)."""

    page: jax.Array
    lane: jax.Array
    position: jax.Array
    chunk_start: jax.Array
    chunk_end: jax.Array
    window: jax.Array
    shift: jax.Array
    is_live: jax.Array
    is_partial: jax.Array


def index_cache_write_kernel(
    tile_requests_ref: jax.Array,
    tile_indices_ref: jax.Array,
    counts_ref: jax.Array,
    cu_q_lens_ref: jax.Array,
    kv_lens_ref: jax.Array,
    block_table_ref: jax.Array,
    keys_hbm: jax.Array,
    _: jax.Array,
    cache_hbm: jax.Array,
    keys_vmem: jax.Array,
    words_vmem: jax.Array,
    old_tiles: jax.Array,
    new_tiles: jax.Array,
    group_counts: jax.Array,
    sems: jax.Array,
    *,
    page_size: int,
    pages_per_seq: int,
) -> None:
    """Quantize every window of NUM_LANES source rows into lane-major words,
    then write the chunks' tiles TILE_GROUP at a time, each tile's lanes
    outside its chunk kept from the cache."""
    num_tiles = counts_ref[COUNT_TILES]
    num_windows = counts_ref[COUNT_WINDOWS]
    load = pltpu.make_async_copy(
        keys_hbm,
        keys_vmem.at[pl.ds(0, keys_hbm.shape[0])],
        sems.at[SEM_READ, 0],
    )
    load.start()
    load.wait()

    def _quantize_window(window, carry):
        rows = pl.ds(pl.multiple_of(window * NUM_LANES, NUM_LANES), NUM_LANES)
        words_vmem[window] = quantize_columns(
            keys_vmem[rows, :].astype(jnp.float32).T
        )
        return carry

    lax.fori_loop(0, num_windows, _quantize_window, 0)

    def _tile(index) -> Tile:
        is_live = index < num_tiles
        index = jnp.minimum(index, num_tiles - 1)
        request = tile_requests_ref[index]
        position = tile_indices_ref[index] * NUM_LANES
        chunk_end = kv_lens_ref[request]
        row = cu_q_lens_ref[request]
        chunk_start = chunk_end - (cu_q_lens_ref[request + 1] - row)
        base = row + position - chunk_start
        # Note (david): base is at least -NUM_LANES + 1, so the shifted floor
        # division stays on non-negative numbers.
        window = (base + NUM_LANES) // NUM_LANES - 1
        unsigned_position = position.astype(jnp.uint32)
        page_index = (unsigned_position // page_size).astype(jnp.int32)
        page = block_table_ref[request * pages_per_seq + page_index]
        return Tile(
            page=jnp.where(is_live, page, 0),
            lane=pl.multiple_of(
                (unsigned_position % page_size).astype(jnp.int32), NUM_LANES
            ),
            position=position,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            window=window,
            shift=base - window * NUM_LANES,
            is_live=is_live,
            is_partial=jnp.logical_and(
                is_live,
                jnp.logical_or(
                    position < chunk_start, position + NUM_LANES > chunk_end
                ),
            ),
        )

    def _start_copy(tile, parity, slot_ref, size, *, is_read):
        cache_window = cache_hbm.at[
            pl.ds(tile.page, size), :, :, pl.ds(tile.lane, NUM_LANES)
        ]
        tiles_window = slot_ref.at[pl.ds(0, size)]
        if is_read:
            source, target, kind = cache_window, tiles_window, SEM_READ
        else:
            source, target, kind = tiles_window, cache_window, SEM_WRITE
        pltpu.make_async_copy(source, target, sems.at[kind, parity]).start()

    def _start_reads(group):
        """Start the reads of the group's partial tiles; record their count."""
        parity = lax.rem(group, TILE_STAGES)
        reads = 0
        for member in range(TILE_GROUP):
            tile = _tile(group * TILE_GROUP + member)
            size = tile.is_partial.astype(jnp.int32)
            _start_copy(
                tile,
                parity,
                old_tiles.at[parity, pl.ds(member, 1)],
                size,
                is_read=True,
            )
            reads += size
        group_counts[parity] = reads

    def _wait(tiles_ref, count, sem):
        window = tiles_ref.at[pl.ds(0, count)]
        pltpu.make_async_copy(window, window, sem).wait()

    def _write_group(group, carry):
        parity = lax.rem(group, TILE_STAGES)
        # Note (david): a stage's semaphores carry only its group's copies, so
        # one wait sized to the group's count covers them.
        reads = group_counts[parity]
        _start_reads(group + READ_AHEAD_GROUPS)
        _wait(old_tiles.at[parity], reads, sems.at[SEM_READ, parity])

        @pl.when(group >= TILE_STAGES)
        def _free_slots():
            _wait(new_tiles.at[parity], TILE_GROUP, sems.at[SEM_WRITE, parity])

        for member in range(TILE_GROUP):
            tile = _tile(group * TILE_GROUP + member)
            new_tiles[parity, member] = blend_tile(
                tile, words_vmem, old_tiles[parity, member]
            )
            _start_copy(
                tile,
                parity,
                new_tiles.at[parity, pl.ds(member, 1)],
                tile.is_live.astype(jnp.int32),
                is_read=False,
            )
        return carry

    num_groups = pl.cdiv(num_tiles, TILE_GROUP)
    for group in range(READ_AHEAD_GROUPS):
        _start_reads(group)
    lax.fori_loop(0, num_groups, _write_group, 0)
    # Note (david): the kernel must not exit with writes in flight.
    last_writes = num_tiles - (num_groups - 1) * TILE_GROUP
    for back in range(TILE_STAGES, 0, -1):
        group = num_groups - back

        @pl.when(group >= 0)
        def _drain(group=group, back=back):
            count = last_writes if back == 1 else TILE_GROUP
            parity = lax.rem(group, TILE_STAGES)
            _wait(new_tiles.at[parity], count, sems.at[SEM_WRITE, parity])


def blend_tile(tile: Tile, words_vmem: jax.Array, old_tile: jax.Array):
    """The tile's new bytes: chunk lanes from the source windows, rolled from
    source lane order to cache lane order, the others from old_tile."""
    num_windows = words_vmem.shape[0]
    rotation = lax.rem(NUM_LANES - tile.shift, NUM_LANES)
    first = words_vmem[jnp.clip(tile.window, 0, num_windows - 1)]
    second = words_vmem[jnp.clip(tile.window + 1, 0, num_windows - 1)]
    lane = lax.broadcasted_iota(jnp.int32, first.shape, 1)
    words = jnp.where(
        lane < NUM_LANES - tile.shift,
        pltpu.roll(first, rotation, 1),
        pltpu.roll(second, rotation, 1),
    )[:CACHE_GROUPS]
    old_words = pltpu.bitcast(
        old_tile.reshape(CACHE_GROUPS * WORD_BYTES, NUM_LANES), jnp.uint32
    )
    position = tile.position + lane[:CACHE_GROUPS]
    in_chunk = jnp.logical_and(
        position >= tile.chunk_start, position < tile.chunk_end
    )
    blended = jnp.where(in_chunk, words, old_words)
    return pltpu.bitcast(blended, jnp.uint8).reshape(
        CACHE_GROUPS, WORD_BYTES, NUM_LANES
    )


def index_cache_write_tiles(
    cu_q_lens: jax.Array, kv_lens: jax.Array, num_tokens: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The chunks' lane tiles in request order as (request, tile index) tables
    of a static length, and (tile count, source window count)."""
    batch = kv_lens.shape[0]
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    first_tile = (kv_lens - q_lens) // NUM_LANES
    tile_counts = jnp.where(
        q_lens > 0, pl.cdiv(kv_lens, NUM_LANES) - first_tile, 0
    )
    tile_ends = jnp.cumsum(tile_counts).astype(jnp.int32)
    # Note (david): a chunk of n tokens spans at most cdiv(n, NUM_LANES) + 1
    # tiles.
    max_tiles = pl.cdiv(num_tokens, NUM_LANES) + 2 * batch
    tile_ids = jnp.arange(max_tiles, dtype=jnp.int32)
    requests = jnp.minimum(request_of(tile_ends, max_tiles), batch - 1)
    indices = (
        first_tile[requests]
        + tile_ids
        - (tile_ends[requests] - tile_counts[requests])
    )
    counts = jnp.stack([
        tile_ends[-1],
        pl.cdiv(cu_q_lens[-1], NUM_LANES).astype(jnp.int32),
    ])
    return requests, indices.astype(jnp.int32), counts


def index_cache_write_pallas(
    keys: jax.Array,
    cache: jax.Array,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
) -> jax.Array:
    """dsa_index_cache_write's pallas_call on checked operands; returns the
    cache."""
    num_tokens = keys.shape[0]
    page_size = cache.shape[-1]
    pages_per_seq = block_table.shape[1]
    padded_tokens = pl.cdiv(num_tokens, NUM_LANES) * NUM_LANES
    # Note (david): a DMA writes whole packed bf16 row pairs, so the keys
    # reach the kernel in whole bf16 tiles; the window loop reads the scratch
    # in whole lane tiles of rows.
    tile_tokens = pl.cdiv(num_tokens, BF16_TILE_ROWS) * BF16_TILE_ROWS
    tile_requests, tile_indices, counts = index_cache_write_tiles(
        cu_q_lens, kv_lens, num_tokens
    )
    scalar_prefetches = (
        tile_requests,
        tile_indices,
        counts,
        cu_q_lens,
        kv_lens,
        block_table.reshape(-1),
    )
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    tile_shape = (TILE_STAGES, TILE_GROUP, CACHE_GROUPS, WORD_BYTES, NUM_LANES)
    call = pl.pallas_call(
        functools.partial(
            index_cache_write_kernel,
            page_size=page_size,
            pages_per_seq=pages_per_seq,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            grid=(1,),
            in_specs=[hbm_spec, hbm_spec],
            out_specs=hbm_spec,
            scratch_shapes=(
                pltpu.VMEM((padded_tokens, INDEX_HEAD_DIM), keys.dtype),
                pltpu.VMEM(
                    (padded_tokens // NUM_LANES, TILE_WORD_ROWS, NUM_LANES),
                    jnp.uint32,
                ),
                pltpu.VMEM(tile_shape, jnp.uint8),
                pltpu.VMEM(tile_shape, jnp.uint8),
                pltpu.SMEM((TILE_STAGES,), jnp.int32),
                pltpu.SemaphoreType.DMA((NUM_SEMS, TILE_STAGES)),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        input_output_aliases={len(scalar_prefetches) + 1: 0},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_bytes(),
            disable_bounds_checks=True,
        ),
        name="dsa_index_cache_write",
    )
    if tile_tokens != num_tokens:
        keys = jnp.pad(keys, ((0, tile_tokens - num_tokens), (0, 0)))
    # Note (david): XLA may place a small operand in VMEM, where the kernel's
    # copies of it would lower to vector load and store loops.
    return call(
        *scalar_prefetches,
        pltpu.with_memory_space_constraint(keys, pltpu.HBM),
        cache,
    )
