"""The MoE permute on SparseCore: a stable counting sort of the (token, k)
pairs by expert.

The keys are expert ids in [0, E), so a counting sort replaces the XLA
comparison sort, and the destination position of each pair is the inverse
permutation the combine needs, with no second scatter.

One SparseCore, its vector subcores splitting the pairs into contiguous
chunks:

1. Histogram: each subcore counts its chunk per expert (scan_count gives the
   in-vector duplicate counts, addupdate_scatter adds them at each id's last
   occurrence).
2. Offsets: the subcores publish their histograms to shared VMEM; after a
   barrier each one computes, per expert, the exclusive prefix over experts
   of the totals plus the counts of the subcores before it. Subcore 0 writes
   group_sizes.
3. Scatter: each subcore walks its chunk again; a pair's position is its
   expert's running offset plus its in-vector duplicate rank. The positions
   go out linearly as inv_perm, and the token index and routing weight of
   each pair are scattered to their position by indirect DMA into shared
   VMEM. After a barrier the subcores copy the sorted arrays to HBM in
   contiguous pieces.

Every pair keeps its input order within its expert (subcore order, then
order inside the chunk), so the result equals a stable sort by expert.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from flywheel_tpu.core.tiling import round_up
from flywheel_tpu.moe.config import ROW_ALIGNMENT

# Note (david): the most pairs a subcore stages in VMEM per DMA block.
MAX_BLOCK = 8192
# Note (david): a subcore's chunk starts on the 1D HBM tile.
CHUNK_ALIGNMENT = 128
GROUP_SIZE_BYTES = 4
# Note (david): the indirect-DMA position of a pair that is not written (the
# padding beyond the padded row count).
IGNORED_POSITION = -1
# Note (david): one DMA semaphore per copy in flight at once: the weights
# load, the two scatters (then the two write-backs) and group_sizes.
NUM_DMA_SEMS = 4
WEIGHTS_SEM, TOKEN_SEM, SORTED_WEIGHTS_SEM, GROUP_SIZES_SEM = range(
    NUM_DMA_SEMS
)
# Note (david): the pairs of one 1D VMEM tile start their scatter as soon as
# their positions are known, under the next piece's position compute.
SCATTER_PIECE = 128
# Note (david): rough counts that let XLA's scheduler size this call: per
# pair, the ids and weights loads, the inv_perm store and the two scatters
# (5 words), and per sorted row its token and weight write-back (2 words).
FLOPS_PER_PAIR = 16
BYTES_PER_PAIR = 20
BYTES_PER_SORTED_ROW = 8


@functools.cache
def sparse_core_supported() -> bool:
    """Whether the default backend runs the MoE SparseCore kernels: a TPU
    whose SparseCore vector subcores do indirect DMA (v7x on; v6e's do not
    lower it)."""
    if jax.default_backend() != "tpu":
        return False
    else:
        info = pltpu.get_tpu_info()
        return info.sparse_core is not None and info.generation >= 7


def sc_permute_kernel(
    # In (HBM)
    ids_hbm,  # int32[num_alloc]
    weights_hbm,  # f32[num_alloc]
    # Out (HBM)
    token_hbm,  # int32[num_rows_padded]
    sorted_weights_hbm,  # f32[num_rows_padded]
    group_sizes_hbm,  # int32[group_sizes_size]
    inv_perm_hbm,  # int32[num_alloc]
    # Scratch
    ids_v,  # VMEM int32[block]
    weights_v,  # VMEM f32[block]
    pos_v,  # VMEM int32[block]
    token_v,  # VMEM int32[block]
    offsets_v,  # VMEM int32[experts_padded]
    group_sizes_v,  # VMEM int32[experts_padded]
    rows_v,  # VMEM int32[num_subcores, experts_padded]
    shared_hist,  # VMEM_SHARED int32[num_subcores, experts_padded]
    token_sh,  # VMEM_SHARED int32[num_rows_padded]
    weights_sh,  # VMEM_SHARED f32[num_rows_padded]
    sems,  # DMA sem[NUM_DMA_SEMS]
    *,
    num_rows: int,
    num_experts: int,
    top_k: int,
    chunk: int,
    block: int,
    num_subcores: int,
    lanes: int,
    piece: int,
    num_full_pieces: int,
    tail_piece: int,
):
    subcore = lax.axis_index("subcore")
    chunk_start = pl.multiple_of(subcore * chunk, CHUNK_ALIGNMENT)
    experts_padded = offsets_v.shape[0]
    num_blocks = chunk // block
    lane_ids = jnp.arange(lanes, dtype=jnp.int32)
    # Note (david): a one-block chunk keeps its ids in VMEM from the
    # histogram to the scatter, and its weights load under the histogram.
    is_single_block = num_blocks == 1

    def _hbm_block(hbm, b):
        """Block b of this subcore's chunk of a pair array in HBM."""
        start = pl.multiple_of(chunk_start + b * block, CHUNK_ALIGNMENT)
        return hbm.at[pl.ds(start, block)]

    def _weights_copy(b):
        return pltpu.make_async_copy(
            _hbm_block(weights_hbm, b), weights_v, sems.at[WEIGHTS_SEM]
        )

    def _group_sizes_copy():
        size = group_sizes_hbm.shape[0]
        return pltpu.make_async_copy(
            group_sizes_v.at[pl.ds(0, size)],
            group_sizes_hbm,
            sems.at[GROUP_SIZES_SEM],
        )

    if is_single_block:
        _weights_copy(0).start()

    @plsc.parallel_loop(0, experts_padded, lanes)
    def _(i):
        offsets_v[pl.ds(i, lanes)] = jnp.zeros(lanes, jnp.int32)

    # Note (david): the padding ids count like any other, so the fused MLP's
    # padded rows land in the last expert and the rest in the bin past it.
    @pl.loop(0, num_blocks)
    def _(b):
        pltpu.sync_copy(_hbm_block(ids_hbm, b), ids_v)

        @plsc.parallel_loop(0, block, lanes)
        def _(j):
            ids = ids_v[pl.ds(j, lanes)]
            counts, is_last = plsc.scan_count(ids)
            plsc.addupdate_scatter(offsets_v, [ids], counts, mask=is_last)

    pltpu.sync_copy(offsets_v, shared_hist.at[subcore])
    plsc.subcore_barrier()
    pltpu.sync_copy(shared_hist, rows_v)

    def _offsets_chunk(c, carry):
        """The exclusive prefix over experts of the totals, plus the counts
        of the preceding subcores."""
        lane_slice = pl.ds(pl.multiple_of(c * lanes, lanes), lanes)
        total = jnp.zeros(lanes, jnp.int32)
        before = jnp.zeros(lanes, jnp.int32)
        for r in range(num_subcores):
            row = rows_v[r, lane_slice]
            total = total + row
            before = before + jnp.where(subcore > r, row, 0)
        inclusive = plsc.cumsum(total)
        offsets_v[lane_slice] = carry + inclusive - total + before
        group_sizes_v[lane_slice] = total
        return carry + inclusive[lanes - 1]

    lax.fori_loop(0, experts_padded // lanes, _offsets_chunk, jnp.int32(0))

    @pl.when(subcore == 0)
    def _():
        _group_sizes_copy().start()

    # Note (david): token = pair // top_k, stepped per vector without a
    # vector divide: (q0 * top_k + r0 + lane) // top_k
    # = q0 + sum_t [r0 + lane >= t * top_k].
    num_thresholds = (top_k - 1 + lanes - 1) // top_k
    q_step, r_step = divmod(lanes, top_k)

    def _vector_body(block_start, j, q, r):
        """Positions (the running expert offset plus the in-vector rank),
        inv_perm entries and tokens of one vector of pairs."""
        lane_slice = pl.ds(j, lanes)
        ids = ids_v[lane_slice]
        counts, is_last = plsc.scan_count(ids)
        base = plsc.load_gather(offsets_v, [ids])
        plsc.addupdate_scatter(offsets_v, [ids], counts, mask=is_last)
        pos = jnp.where(ids < num_experts, base + counts - 1, IGNORED_POSITION)
        pos_v[lane_slice] = pos
        # Note (david): inv_perm goes in place of the ids; past the padded
        # rows it points at row 0, which the combine's last tile can gather
        # and drop.
        ids_v[lane_slice] = jnp.maximum(pos, 0)
        token = jnp.full((lanes,), q, jnp.int32)
        for t in range(1, num_thresholds + 1):
            token = token + (r + lane_ids >= t * top_k).astype(jnp.int32)
        pair = block_start + j + lane_ids
        token_v[lane_slice] = jnp.where(pair < num_rows, token, 0)
        q_next = q + q_step
        r_next = r + r_step
        is_wrapped = r_next >= top_k
        return (
            jnp.where(is_wrapped, q_next + 1, q_next),
            jnp.where(is_wrapped, r_next - top_k, r_next),
        )

    num_pieces = block // SCATTER_PIECE

    def _scatter_copies(p):
        """The token and weight scatters of piece p of the block."""
        rows = pl.ds(
            pl.multiple_of(p * SCATTER_PIECE, SCATTER_PIECE), SCATTER_PIECE
        )
        positions = plsc.Indices(pos_v.at[rows], ignored_value=IGNORED_POSITION)
        # Note (david): the scatter goes to shared VMEM, not HBM: an indirect
        # DMA of 4-byte elements straight to HBM moves a whole 32-byte DMA
        # granule per element, and the two of them took most of the kernel.
        return [
            pltpu.make_async_copy(
                token_v.at[rows], token_sh.at[positions], sems.at[TOKEN_SEM]
            ),
            pltpu.make_async_copy(
                weights_v.at[rows],
                weights_sh.at[positions],
                sems.at[SORTED_WEIGHTS_SEM],
            ),
        ]

    @pl.loop(0, num_blocks)
    def _(b):
        block_start = chunk_start + b * block
        if not is_single_block:
            pltpu.sync_copy(_hbm_block(ids_hbm, b), ids_v)
            _weights_copy(b).start()
        _weights_copy(b).wait()

        def _piece_body(p, carry):
            def _piece_vector(u, carry):
                j = pl.multiple_of(p * SCATTER_PIECE + u * lanes, lanes)
                return _vector_body(block_start, j, *carry)

            carry = lax.fori_loop(
                0, SCATTER_PIECE // lanes, _piece_vector, carry
            )
            for copy in _scatter_copies(p):
                copy.start()
            return carry

        lax.fori_loop(
            0,
            num_pieces,
            _piece_body,
            (block_start // top_k, block_start % top_k),
        )
        pltpu.sync_copy(ids_v, _hbm_block(inv_perm_hbm, b))

        @pl.loop(0, num_pieces)
        def _(p):
            for copy in _scatter_copies(p):
                copy.wait()

    @pl.when(subcore == 0)
    def _():
        _group_sizes_copy().wait()

    plsc.subcore_barrier()

    def _write_back(start, size):
        """One subcore's contiguous piece of the sorted arrays."""
        copies = [
            pltpu.make_async_copy(
                token_sh.at[pl.ds(start, size)],
                token_hbm.at[pl.ds(start, size)],
                sems.at[TOKEN_SEM],
            ),
            pltpu.make_async_copy(
                weights_sh.at[pl.ds(start, size)],
                sorted_weights_hbm.at[pl.ds(start, size)],
                sems.at[SORTED_WEIGHTS_SEM],
            ),
        ]
        for copy in copies:
            copy.start()
        for copy in copies:
            copy.wait()

    @pl.when(subcore < num_full_pieces)
    def _():
        _write_back(pl.multiple_of(subcore * piece, ROW_ALIGNMENT), piece)

    if tail_piece:

        @pl.when(subcore == num_full_pieces)
        def _():
            _write_back(num_full_pieces * piece, tail_piece)


@functools.partial(jax.jit, static_argnames=("num_experts", "top_k"))
def sc_permute(
    topk_ids: jax.Array,
    topk_weights: jax.Array,
    *,
    num_experts: int,
    top_k: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The expert-sorted rows of the flattened (token, k) pairs.

    Args:
        topk_ids: int32[num_tokens * top_k], the expert of each pair.
        topk_weights: f32[num_tokens * top_k], the routing weight of each
            pair.
        num_experts: E.
        top_k: Pairs per token.

    Returns:
        token_indices: int32[num_rows_padded], the token of each sorted row,
            num_rows_padded = round_up(num_tokens * top_k, 128). The padded
            rows read token 0.
        sorted_weights: f32[num_rows_padded], the routing weights in sorted
            order; 0 on the padded rows.
        group_sizes: int32[E], rows per expert; the padded rows count toward
            the last expert.
        inv_perm: int32[>= num_tokens * top_k], a multiple of 128 long; pair
            i lands on sorted row inv_perm[i]; the entries past the pairs
            are rows too (the padded rows, then row 0).
    """
    if topk_ids.ndim != 1 or topk_ids.dtype != jnp.int32:
        raise ValueError(
            f"topk_ids must be int32[n], got {topk_ids.dtype}"
            f"{list(topk_ids.shape)}"
        )
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            f"topk_weights must be {list(topk_ids.shape)}, got"
            f" {list(topk_weights.shape)}"
        )
    sparse_core = pltpu.get_tpu_info().sparse_core
    if sparse_core is None:
        raise RuntimeError("sc_permute needs a TPU with SparseCores")
    lanes, num_subcores = sparse_core.num_lanes, sparse_core.num_subcores
    num_rows = topk_ids.shape[0]
    num_rows_padded = round_up(num_rows, ROW_ALIGNMENT)
    chunk = round_up(pl.cdiv(num_rows, num_subcores), CHUNK_ALIGNMENT)
    num_blocks = pl.cdiv(chunk, MAX_BLOCK)
    block = round_up(pl.cdiv(chunk, num_blocks), CHUNK_ALIGNMENT)
    chunk = block * num_blocks
    num_alloc = chunk * num_subcores
    # Note (david): the bin past the last expert holds the padding beyond
    # num_rows_padded.
    experts_padded = round_up(num_experts + 1, lanes)
    # Note (david): group_sizes goes out in whole DMA granules, so no slice
    # is needed when E fills them.
    group_sizes_size = round_up(
        num_experts, sparse_core.dma_granule_size_bytes // GROUP_SIZE_BYTES
    )
    num_row_tiles = num_rows_padded // ROW_ALIGNMENT
    tiles_per_piece = pl.cdiv(num_row_tiles, num_subcores)
    num_full_pieces = num_row_tiles // tiles_per_piece
    tail_piece = (num_row_tiles % tiles_per_piece) * ROW_ALIGNMENT

    # Note (david): rows [num_rows, num_rows_padded) are the fused MLP's
    # padding: expert E - 1, weight 0. The rest of the allocation is not
    # written.
    pad_ids = jnp.where(
        jnp.arange(num_rows, num_alloc, dtype=jnp.int32) < num_rows_padded,
        num_experts - 1,
        num_experts,
    )
    ids = jnp.concatenate([topk_ids, pad_ids])
    weights = jnp.pad(
        topk_weights.astype(jnp.float32), (0, num_alloc - num_rows)
    )

    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core", subcore_axis_name="subcore", num_cores=1
    )
    kernel = pl.kernel(
        functools.partial(
            sc_permute_kernel,
            num_rows=num_rows,
            num_experts=num_experts,
            top_k=top_k,
            chunk=chunk,
            block=block,
            num_subcores=num_subcores,
            lanes=lanes,
            piece=tiles_per_piece * ROW_ALIGNMENT,
            num_full_pieces=num_full_pieces,
            tail_piece=tail_piece,
        ),
        out_type=(
            jax.ShapeDtypeStruct((num_rows_padded,), jnp.int32),
            jax.ShapeDtypeStruct((num_rows_padded,), jnp.float32),
            jax.ShapeDtypeStruct((group_sizes_size,), jnp.int32),
            jax.ShapeDtypeStruct((num_alloc,), jnp.int32),
        ),
        mesh=mesh,
        scratch_types=[
            pltpu.VMEM((block,), jnp.int32),
            pltpu.VMEM((block,), jnp.float32),
            pltpu.VMEM((block,), jnp.int32),
            pltpu.VMEM((block,), jnp.int32),
            pltpu.VMEM((experts_padded,), jnp.int32),
            pltpu.VMEM((experts_padded,), jnp.int32),
            pltpu.VMEM((num_subcores, experts_padded), jnp.int32),
            pltpu.VMEM_SHARED((num_subcores, experts_padded), jnp.int32),
            pltpu.VMEM_SHARED((num_rows_padded,), jnp.int32),
            pltpu.VMEM_SHARED((num_rows_padded,), jnp.float32),
            pltpu.SemaphoreType.DMA((NUM_DMA_SEMS,)),
        ],
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False, needs_layout_passes=False
        ),
        name=f"moe_sc_permute-n_{num_rows}-e_{num_experts}-k_{top_k}",
        cost_estimate=pl.CostEstimate(
            flops=num_alloc * FLOPS_PER_PAIR,
            bytes_accessed=num_alloc * BYTES_PER_PAIR
            + num_rows_padded * BYTES_PER_SORTED_ROW,
            transcendentals=0,
        ),
    )
    token_indices, sorted_weights, group_sizes, inv_perm = kernel(ids, weights)
    if group_sizes_size == num_experts:
        return token_indices, sorted_weights, group_sizes, inv_perm
    else:
        return (
            token_indices,
            sorted_weights,
            group_sizes[:num_experts],
            inv_perm,
        )
