"""Ragged write of packed new latent tokens into the paged MLA cache."""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...core.tpu import NUM_SUBLANES
from ..config import vmem_limit_bytes
from .mla_decode import cache_packing, cache_tile_rows

# Note (david): a group's copies run while the group before it computes, and
# its write-backs drain while the next one computes, so three slots keep the
# reads, the shifts and the writes of three groups apart.
NUM_SLOTS = 3
# Note (david): a shift step unrolls its tiles, so a longer step grows the code
# and a shorter one loops more often.
SHIFT_TILES = 4
SEM_READ = 0
SEM_WRITE = 1
WORD_BITS = 32


class Piece(NamedTuple):
    """The cache tiles of one request's chunk that fall in one group: cache
    positions [start, start + tokens), built at rows [row, row + tokens) of
    the group's cache_rows from its source rows [first_source_row,
    end_source_row). Position start + j takes source row source_tile + shift
    + j."""

    request: jax.Array
    chunk_start: jax.Array
    chunk_end: jax.Array
    start: jax.Array
    tokens: jax.Array
    row: jax.Array
    source_tile: jax.Array
    shift: jax.Array
    first_source_row: jax.Array
    end_source_row: jax.Array
    has_head_edge: jax.Array
    has_tail_edge: jax.Array


def mla_cache_write_kernel(
    cu_q_lens_ref: jax.Array,
    kv_lens_ref: jax.Array,
    block_table_ref: jax.Array,
    stream_starts_ref: jax.Array,
    group_first_request_ref: jax.Array,
    new_kv_c_hbm: jax.Array,
    new_k_pe_hbm: jax.Array,
    _: jax.Array,
    cache_hbm: jax.Array,
    latent_stage: jax.Array,
    rope_stage: jax.Array,
    cache_rows: jax.Array,
    stage_first_row_ref: jax.Array,
    staged_rows_ref: jax.Array,
    edge_rows_ref: jax.Array,
    dma_sems: jax.Array,
    *,
    batch: int,
    page_size: int,
    pages_per_seq: int,
    group_tokens: int,
) -> None:
    """Write every active request's chunk rows (latent, rope, zeros) into its
    cache positions [kv_len - q_len, kv_len), one group of group_tokens cache
    positions at a time.

    The requests' cache tiles, laid end to end in request order, form a
    stream that stream_starts_ref indexes and the groups cut; group g starts
    at request group_first_request_ref[g].
    """
    _, _, row_dim = cache_rows.shape
    latent_dim = latent_stage.shape[-1]
    rope_dim = rope_stage.shape[-1]
    cache_dtype = cache_rows.dtype
    packing = cache_packing(cache_dtype)
    tile_rows = cache_tile_rows(cache_dtype)
    element_bits = WORD_BITS // packing
    total_tokens = stream_starts_ref[batch]
    num_groups = pl.cdiv(total_tokens, group_tokens)
    # Note (david): the word layout is a row-major reshape of the token rows,
    # so every copy addresses token rows. The Pallas interpreter takes no ref
    # reshape and is handed the token rows.
    if len(cache_hbm.shape) == 3:
        cache_token_rows = cache_hbm
    else:
        cache_token_rows = cache_hbm.reshape(
            cache_hbm.shape[0], page_size, row_dim
        )

    def _tile_aligned(num_tokens):
        # Note (david): a copy window on the token axis must be whole staging
        # tiles.
        return pl.multiple_of(num_tokens, tile_rows)

    def _fold_pieces(group, body, init):
        """Fold body(piece, carry) over the requests that may hold tiles of
        the group; a request that holds none gives a piece of tokens <= 0."""

        def _request_piece(request, carry):
            source_start = cu_q_lens_ref[request]
            chunk_end = kv_lens_ref[request]
            chunk_start = chunk_end - (
                cu_q_lens_ref[request + 1] - source_start
            )
            tile_start = chunk_start // tile_rows * tile_rows
            tile_end = pl.cdiv(chunk_end, tile_rows) * tile_rows
            group_start = group * group_tokens
            stream_start = stream_starts_ref[request]
            stream_lo = jnp.maximum(stream_start, group_start)
            stream_hi = jnp.minimum(
                stream_starts_ref[request + 1], group_start + group_tokens
            )
            start = tile_start + stream_lo - stream_start
            tokens = stream_hi - stream_lo
            source_row = source_start + start - chunk_start
            # Note (david): the first tile's source row precedes the chunk's
            # by less than a tile, so it is above -tile_rows and the
            # remainder is of a positive number.
            shift = lax.rem(source_row + tile_rows, tile_rows)
            has_head_edge = jnp.logical_and(
                start == tile_start, chunk_start != tile_start
            )
            # Note (david): a chunk inside one tile patches it once, at its
            # head.
            has_tail_edge = jnp.logical_and(
                jnp.logical_and(
                    start + tokens == tile_end, chunk_end != tile_end
                ),
                jnp.logical_not(
                    jnp.logical_and(has_head_edge, tokens == tile_rows)
                ),
            )
            piece = Piece(
                request=request,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                start=start,
                tokens=tokens,
                row=stream_lo - group_start,
                source_tile=source_row - shift,
                shift=shift,
                first_source_row=source_start
                + jnp.maximum(start, chunk_start)
                - chunk_start,
                end_source_row=source_start
                + jnp.minimum(start + tokens, chunk_end)
                - chunk_start,
                has_head_edge=has_head_edge,
                has_tail_edge=has_tail_edge,
            )
            return body(piece, carry)

        # Note (david): the request that holds the next group's first tile
        # may hold some of this group's too.
        return lax.fori_loop(
            group_first_request_ref[group],
            jnp.minimum(group_first_request_ref[group + 1] + 1, batch),
            _request_piece,
            init,
        )

    def _start_group_reads(group, slot):
        """Start the copies of the group's partial edge tiles from the cache
        into cache_rows and of its source rows into its stages, and record
        them for the wait and the shifts."""

        def _start_edge_reads(piece, carry):
            first_source_row, end_source_row, edge_rows = carry
            is_live = piece.tokens > 0
            for has_edge, edge_offset in (
                (piece.has_head_edge, 0),
                (piece.has_tail_edge, piece.tokens - tile_rows),
            ):
                reads_edge = jnp.logical_and(is_live, has_edge)

                @pl.when(reads_edge)
                def _start_edge_read(edge_offset=edge_offset):
                    position = piece.start + edge_offset
                    pltpu.make_async_copy(
                        cache_token_rows.at[
                            block_table_ref[
                                piece.request * pages_per_seq
                                + position // page_size
                            ],
                            pl.ds(
                                _tile_aligned(lax.rem(position, page_size)),
                                tile_rows,
                            ),
                        ],
                        cache_rows.at[
                            slot,
                            pl.ds(
                                _tile_aligned(piece.row + edge_offset),
                                tile_rows,
                            ),
                        ],
                        dma_sems.at[SEM_READ, slot],
                    ).start()

                edge_rows = edge_rows + jnp.where(reads_edge, tile_rows, 0)
            return (
                jnp.where(
                    is_live,
                    jnp.minimum(first_source_row, piece.first_source_row),
                    first_source_row,
                ),
                jnp.where(
                    is_live,
                    jnp.maximum(end_source_row, piece.end_source_row),
                    end_source_row,
                ),
                edge_rows,
            )

        # Note (david): consecutive requests' rows are adjacent in the packed
        # source, so the group's source rows are one run and one copy each.
        first_source_row, end_source_row, edge_rows = _fold_pieces(
            group,
            _start_edge_reads,
            (jnp.int32(jnp.iinfo(jnp.int32).max), jnp.int32(0), jnp.int32(0)),
        )
        stage_first_row = _tile_aligned(
            first_source_row // tile_rows * tile_rows
        )
        num_rows = _tile_aligned(
            pl.cdiv(end_source_row, tile_rows) * tile_rows - stage_first_row
        )
        for source_hbm, stage in (
            (new_kv_c_hbm, latent_stage),
            (new_k_pe_hbm, rope_stage),
        ):
            pltpu.make_async_copy(
                source_hbm.at[pl.ds(stage_first_row, num_rows)],
                stage.at[slot, pl.ds(tile_rows, num_rows)],
                dma_sems.at[SEM_READ, slot],
            ).start()
        stage_first_row_ref[slot] = stage_first_row
        staged_rows_ref[slot] = num_rows
        edge_rows_ref[slot] = edge_rows

    def _wait_group_writes(group, slot):
        written = cache_rows.at[
            slot,
            pl.ds(
                0,
                _tile_aligned(
                    jnp.minimum(
                        group_tokens, total_tokens - group * group_tokens
                    )
                ),
            ),
        ]
        pltpu.make_async_copy(
            written, written, dma_sems.at[SEM_WRITE, slot]
        ).wait()

    def _build_tiles(piece, slot, stage_first_row, first_tile, num_tiles):
        """Build num_tiles cache tiles of the piece in cache_rows, from its
        tile first_tile on, out of its stage rows from shift on; the word
        shift rotates within each tile's vregs and selects between neighbors,
        as a rotation across vregs would pay for in selects."""
        rows = pl.ds(
            _tile_aligned(piece.row + first_tile * tile_rows),
            num_tiles * tile_rows,
        )
        stage_rows = pl.ds(
            _tile_aligned(
                piece.source_tile
                - stage_first_row
                + (first_tile + 1) * tile_rows
            ),
            (num_tiles + 1) * tile_rows,
        )
        word_shift = piece.shift // packing
        bit_shift = (lax.rem(piece.shift, packing) * element_bits).astype(
            jnp.uint32
        )
        # Note (david): the rope is narrower than a lane tile, so the zeros go
        # in as the whole tile and the rope over them.
        cache_rows[slot, rows, pl.ds(latent_dim, row_dim - latent_dim)] = (
            jnp.zeros(
                (num_tiles * tile_rows, row_dim - latent_dim), cache_dtype
            )
        )
        for stage, lanes in (
            (latent_stage, pl.ds(0, latent_dim)),
            (rope_stage, pl.ds(latent_dim, rope_dim)),
        ):
            words = pltpu.bitcast(stage[slot, stage_rows, :], jnp.uint32)
            sublane = lax.broadcasted_iota(
                jnp.int32, (NUM_SUBLANES, words.shape[1]), 0
            )
            shifted_words = []
            for word_offset in (word_shift, word_shift + 1):
                rotation = lax.rem(NUM_SUBLANES - word_offset, NUM_SUBLANES)
                rotated = [
                    pltpu.roll(
                        words[tile * NUM_SUBLANES : (tile + 1) * NUM_SUBLANES],
                        rotation,
                        0,
                    )
                    for tile in range(num_tiles + 1)
                ]
                from_next_tile = sublane >= NUM_SUBLANES - word_offset
                shifted_words.append(
                    jnp.concatenate([
                        jnp.where(
                            from_next_tile, rotated[tile + 1], rotated[tile]
                        )
                        for tile in range(num_tiles)
                    ])
                )
            low_words, high_words = shifted_words[0], shifted_words[1]
            # Note (david): shifted in two steps, so a zero bit_shift drops
            # high_words instead of shifting by the full word width.
            cache_rows[slot, rows, lanes] = pltpu.bitcast(
                (low_words >> bit_shift)
                | ((high_words << 1) << (WORD_BITS - 1 - bit_shift)),
                cache_dtype,
            )

    def _start_page_write(piece, slot, page, carry):
        page_start = page * page_size
        write_start = jnp.maximum(piece.start, page_start)
        num_tokens = _tile_aligned(
            jnp.minimum(piece.start + piece.tokens, page_start + page_size)
            - write_start
        )
        pltpu.make_async_copy(
            cache_rows.at[
                slot,
                pl.ds(
                    _tile_aligned(piece.row + write_start - piece.start),
                    num_tokens,
                ),
            ],
            cache_token_rows.at[
                block_table_ref[piece.request * pages_per_seq + page],
                pl.ds(_tile_aligned(write_start - page_start), num_tokens),
            ],
            dma_sems.at[SEM_WRITE, slot],
        ).start()
        return carry

    def _write_piece(slot, stage_first_row, piece, carry):
        """Build the piece's tiles, its partial edge tiles first, as they
        hold the cache's tiles that the bulk would overwrite, then start its
        page writes."""
        num_tiles = piece.tokens // tile_rows
        build_tiles = functools.partial(
            _build_tiles, piece, slot, stage_first_row
        )

        @pl.when(piece.tokens > 0)
        def _write_live_piece():
            for has_edge, edge_tile in (
                (piece.has_head_edge, 0),
                (piece.has_tail_edge, num_tiles - 1),
            ):

                @pl.when(has_edge)
                def _build_edge_tile(edge_tile=edge_tile):
                    rows = pl.ds(
                        _tile_aligned(piece.row + edge_tile * tile_rows),
                        tile_rows,
                    )
                    cached_words = pltpu.bitcast(
                        cache_rows[slot, rows, :], jnp.uint32
                    )
                    build_tiles(edge_tile, 1)
                    word_start = (
                        piece.start
                        + edge_tile * tile_rows
                        + packing
                        * lax.broadcasted_iota(jnp.int32, cached_words.shape, 0)
                    )
                    chunk_bits = jnp.zeros(cached_words.shape, jnp.uint32)
                    for token_in_word in range(packing):
                        position = word_start + token_in_word
                        chunk_bits |= jnp.where(
                            jnp.logical_and(
                                position >= piece.chunk_start,
                                position < piece.chunk_end,
                            ),
                            jnp.uint32(
                                ((1 << element_bits) - 1)
                                << (token_in_word * element_bits)
                            ),
                            jnp.uint32(0),
                        )
                    new_words = pltpu.bitcast(
                        cache_rows[slot, rows, :], jnp.uint32
                    )
                    cache_rows[slot, rows, :] = pltpu.bitcast(
                        (new_words & chunk_bits) | (cached_words & ~chunk_bits),
                        cache_dtype,
                    )

            first_bulk_tile = piece.has_head_edge.astype(jnp.int32)
            num_bulk_tiles = (
                num_tiles
                - first_bulk_tile
                - piece.has_tail_edge.astype(jnp.int32)
            )
            num_steps = num_bulk_tiles // SHIFT_TILES
            first_rest_tile = first_bulk_tile + num_steps * SHIFT_TILES

            def _build_step(step, step_carry):
                build_tiles(first_bulk_tile + step * SHIFT_TILES, SHIFT_TILES)
                return step_carry

            def _build_rest_tile(tile, tile_carry):
                build_tiles(first_rest_tile + tile, 1)
                return tile_carry

            lax.fori_loop(0, num_steps, _build_step, 0)
            lax.fori_loop(
                0, num_bulk_tiles - num_steps * SHIFT_TILES, _build_rest_tile, 0
            )
            lax.fori_loop(
                piece.start // page_size,
                (piece.start + piece.tokens - 1) // page_size + 1,
                functools.partial(_start_page_write, piece, slot),
                0,
            )

        return carry

    def _run_group(group, carry):
        slot = lax.rem(group, NUM_SLOTS)
        next_group = group + 1
        next_slot = lax.rem(next_group, NUM_SLOTS)

        @pl.when(next_group < num_groups)
        def _prefetch_next_group():
            @pl.when(next_group >= NUM_SLOTS)
            def _free_next_slot():
                _wait_group_writes(next_group - NUM_SLOTS, next_slot)

            _start_group_reads(next_group, next_slot)

        # Note (david): every read of a group signals its slot's semaphore,
        # which counts bytes, so one wait per kind sized to the group's rows
        # covers them all.
        read_sem = dma_sems.at[SEM_READ, slot]
        for stage in (latent_stage, rope_stage):
            staged = stage.at[
                slot, pl.ds(0, _tile_aligned(staged_rows_ref[slot]))
            ]
            pltpu.make_async_copy(staged, staged, read_sem).wait()
        edge_rows = edge_rows_ref[slot]

        @pl.when(edge_rows > 0)
        def _wait_edge_reads():
            edges = cache_rows.at[slot, pl.ds(0, _tile_aligned(edge_rows))]
            pltpu.make_async_copy(edges, edges, read_sem).wait()

        _fold_pieces(
            group,
            functools.partial(_write_piece, slot, stage_first_row_ref[slot]),
            0,
        )
        return carry

    @pl.when(num_groups > 0)
    def _prefetch_first_group():
        _start_group_reads(0, 0)

    lax.fori_loop(0, num_groups, _run_group, 0)
    # Note (david): the kernel must not exit with DMAs in flight, and the last
    # NUM_SLOTS groups' writes still run.
    for groups_from_end in range(NUM_SLOTS, 0, -1):
        group = num_groups - groups_from_end

        @pl.when(group >= 0)
        def _drain_group_writes(group=group):
            _wait_group_writes(group, lax.rem(group, NUM_SLOTS))


def mla_cache_write_pallas(
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    kv_cache: jax.Array,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
    *,
    group_tokens: int,
    interpret: bool,
) -> jax.Array:
    """mla_cache_write's pallas_call on operands it has checked, a group
    group_tokens cache positions long, whole cache tiles; returns the
    cache."""
    num_tokens, latent_dim = new_kv_c.shape
    rope_dim = new_k_pe.shape[-1]
    _, page_words, packing, row_dim = kv_cache.shape
    page_size = page_words * packing
    batch, pages_per_seq = block_table.shape
    cache_dtype = kv_cache.dtype
    tile_rows = cache_tile_rows(cache_dtype)
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    stream_tokens = jnp.where(
        q_lens > 0,
        (kv_lens + tile_rows - 1) // tile_rows * tile_rows
        - (kv_lens - q_lens) // tile_rows * tile_rows,
        0,
    )
    stream_ends = jnp.cumsum(stream_tokens).astype(jnp.int32)
    # Note (david): a chunk's tiles exceed its tokens by less than two tiles.
    max_groups = pl.cdiv(num_tokens + batch * 2 * (tile_rows - 1), group_tokens)
    # Note (david): a group's first request is the count of requests that end
    # by its start, one small fusion where jnp.searchsorted runs a loop.
    group_starts = jnp.arange(max_groups + 1, dtype=jnp.int32) * group_tokens
    scalar_prefetches = (
        cu_q_lens,
        kv_lens,
        block_table.reshape(-1),
        jnp.concatenate([jnp.zeros((1,), jnp.int32), stream_ends]),
        jnp.sum(
            stream_ends[None, :] <= group_starts[:, None],
            axis=1,
            dtype=jnp.int32,
        ),
    )
    # Note (david): a source copy moves whole tiles, the last one too.
    padded_tokens = pl.cdiv(num_tokens, tile_rows) * tile_rows
    token_pad = ((0, padded_tokens - num_tokens), (0, 0))
    # Note (david): the Pallas interpreter takes no ref reshape, so an
    # interpreted call hands the kernel the token rows.
    if interpret:
        cache_operand = kv_cache.reshape(kv_cache.shape[0], page_size, row_dim)
    else:
        cache_operand = kv_cache
    # Note (david): a group's rows rounded out to tiles, plus a tile before for
    # a chunk that starts inside one and a tile after for the last shift.
    stage_rows = group_tokens + 4 * tile_rows
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    call = pl.pallas_call(
        functools.partial(
            mla_cache_write_kernel,
            batch=batch,
            page_size=page_size,
            pages_per_seq=pages_per_seq,
            group_tokens=group_tokens,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            grid=(1,),
            in_specs=[hbm_spec, hbm_spec, hbm_spec],
            out_specs=hbm_spec,
            scratch_shapes=(
                pltpu.VMEM((NUM_SLOTS, stage_rows, latent_dim), cache_dtype),
                pltpu.VMEM((NUM_SLOTS, stage_rows, rope_dim), cache_dtype),
                pltpu.VMEM((NUM_SLOTS, group_tokens, row_dim), cache_dtype),
                pltpu.SMEM((NUM_SLOTS,), jnp.int32),
                pltpu.SMEM((NUM_SLOTS,), jnp.int32),
                pltpu.SMEM((NUM_SLOTS,), jnp.int32),
                pltpu.SemaphoreType.DMA((2, NUM_SLOTS)),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct(cache_operand.shape, cache_dtype),
        # Note (david): the kernel writes only the chunks' tiles, so the output
        # aliases the cache operand to keep every other row.
        input_output_aliases={len(scalar_prefetches) + 2: 0},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_bytes(),
        ),
        name="flash_mla_cache_write",
        interpret=pltpu.InterpretParams() if interpret else False,
    )
    return call(
        *scalar_prefetches,
        jnp.pad(new_kv_c, token_pad),
        jnp.pad(new_k_pe, token_pad),
        cache_operand,
    ).reshape(kv_cache.shape)
