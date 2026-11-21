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

"""The MoE combine on SparseCore: the same inverse-permutation gather and f32
top_k sum as combine.py, off the TensorCore, written in the token-major
[T, hidden] layout.

    out[t] = sum_k src[inv_perm[t * top_k + k]]

src is the fused MLP's packing-dim layout [rows, 2, hidden // 2], whose
(2, 128) bf16 tile packs elements (0, c) and (1, c) of a row into one 32-bit
word: a row is hidden // 2 words, the low halves columns [0, hidden / 2) and
the high halves columns [hidden / 2, hidden). out is bf16 [T, hidden] in its
natural tiled layout, whose word packs one column of tokens 2j and 2j + 1.
Both are read and written as int32 words.

A work unit is 16 output tokens (whole HBM tiles) times a 128-word column
slab of src. Per unit, indirect DMAs gather the slab of the tokens' top_k
rows; each bf16 half is summed in f32 in k order from contribution 0 (as
combine.py does), the sums of a token pair are rounded to bf16 and packed
into output words, and two linear DMAs write the low-half and high-half
columns of the tile. The vector subcores of both SparseCores take units
round-robin; gathers run two units ahead of the sum, their index windows
three.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from flywheel_tpu.core.tiling import round_up
from flywheel_tpu.core.tpu import BF16_TILE_ROWS, NUM_LANES, NUM_SUBLANES
from flywheel_tpu.moe.config import BF16_PACKING

HIGH_HALF = -65536  # 0xFFFF0000 as int32
HALF_WORD_BITS = 16
# Note (david): a work unit spans one 128-lane column of the HBM tile.
SLAB = NUM_LANES
# Note (david): Mosaic-SC lowers an indirect DMA only when its index vector
# fits one 128-entry VMEM tile ("Indirect transfer offsets cannot be
# reinterpreted to an untiled contiguous memref" at 160 indices).
MAX_GATHER_INDICES = 128
# Note (david): unit i + NUM_GATHER_SLOTS - 1 gathers while unit i is summed,
# and its index windows load one unit earlier still.
NUM_GATHER_SLOTS = 3
NUM_INDEX_SLOTS = NUM_GATHER_SLOTS + 1
NUM_OUT_SLOTS = 2
WORD_BYTES = 4


def sc_combine_kernel(
    # In (HBM)
    src_hbm,  # bf16[size_m, 2, cols]
    inv_perm_hbm,  # int32[>= num_tiles * 16 * top_k]
    # Out (HBM)
    out_hbm,  # bf16[num_tiles * 16, 2 * cols]
    # Scratch
    idx_v,  # VMEM int32[NUM_INDEX_SLOTS, num_gathers, MAX_GATHER_INDICES]
    rows_v,  # VMEM int32[NUM_GATHER_SLOTS, 16 * top_k, 1, SLAB]
    out_v,  # VMEM int32[NUM_OUT_SLOTS, 2, 8, SLAB]
    index_sem,  # DMA sem[NUM_INDEX_SLOTS]
    gather_sem,  # DMA sem[NUM_GATHER_SLOTS]
    out_sem,  # DMA sem[NUM_OUT_SLOTS]
    *,
    num_tiles: int,
    top_k: int,
    tokens_per_gather: int,
    num_subcores: int,
    num_workers: int,
    lanes: int,
):
    worker = lax.axis_index("core") * num_subcores + lax.axis_index("subcore")
    src_words = src_hbm.bitcast(jnp.int32)  # [size_m, 1, cols]
    out_words = out_hbm.bitcast(jnp.int32)  # [num_tiles * 8, 2 * cols]
    cols = src_words.shape[-1]
    num_slabs = cols // SLAB
    # Note (david): this worker takes units worker, worker + num_workers, ...
    # of the num_tiles * num_slabs units.
    num_worker_units = (
        num_tiles * num_slabs - worker + num_workers - 1
    ) // num_workers
    num_gathers = BF16_TILE_ROWS // tokens_per_gather
    gather_pairs = tokens_per_gather * top_k

    def _unit(i):
        """(tile, first src word) of this worker's unit i."""
        unit_index = worker + i * num_workers
        col = pl.multiple_of(lax.rem(unit_index, num_slabs) * SLAB, SLAB)
        return unit_index // num_slabs, col

    def _index_copies(i):
        tile, _ = _unit(i)
        slot = lax.rem(i, NUM_INDEX_SLOTS)
        copies = []
        for g in range(num_gathers):
            pair = (tile * BF16_TILE_ROWS + g * tokens_per_gather) * top_k
            copies.append(
                pltpu.make_async_copy(
                    inv_perm_hbm.at[
                        pl.ds(pl.multiple_of(pair, NUM_SUBLANES), gather_pairs)
                    ],
                    idx_v.at[slot, g, pl.ds(0, gather_pairs)],
                    index_sem.at[slot],
                )
            )
        return copies

    def _gather_copies(i):
        _, col = _unit(i)
        index_slot = lax.rem(i, NUM_INDEX_SLOTS)
        slot = lax.rem(i, NUM_GATHER_SLOTS)
        return [
            pltpu.make_async_copy(
                src_words.at[
                    plsc.Indices(
                        idx_v.at[index_slot, g, pl.ds(0, gather_pairs)]
                    ),
                    :,
                    pl.ds(col, SLAB),
                ],
                rows_v.at[slot, pl.ds(g * gather_pairs, gather_pairs)],
                gather_sem.at[slot],
            )
            for g in range(num_gathers)
        ]

    def _out_copies(i):
        """The low-half and high-half column writes of unit i."""
        tile, col = _unit(i)
        slot = lax.rem(i, NUM_OUT_SLOTS)
        word_rows = pl.ds(
            pl.multiple_of(tile * NUM_SUBLANES, NUM_SUBLANES),
            NUM_SUBLANES,
        )
        return [
            pltpu.make_async_copy(
                out_v.at[slot, half],
                out_words.at[word_rows, pl.ds(half * cols + col, SLAB)],
                out_sem.at[slot],
            )
            for half in (0, 1)
        ]

    def _start_gathers(i):
        for copy in _index_copies(i):
            copy.wait()
        for copy in _gather_copies(i):
            copy.start()

    def _sum_unit(i):
        slot = lax.rem(i, NUM_GATHER_SLOTS)
        out_slot = lax.rem(i, NUM_OUT_SLOTS)

        @plsc.parallel_loop(0, SLAB, lanes)
        def _(c):
            for pair in range(NUM_SUBLANES):
                token_sums = []
                for token in (2 * pair, 2 * pair + 1):
                    word = rows_v[slot, token * top_k, 0, pl.ds(c, lanes)]
                    low_sum = (word << HALF_WORD_BITS).view(jnp.float32)
                    high_sum = (word & HIGH_HALF).view(jnp.float32)
                    for k in range(1, top_k):
                        word = rows_v[
                            slot, token * top_k + k, 0, pl.ds(c, lanes)
                        ]
                        low_sum = low_sum + (word << HALF_WORD_BITS).view(
                            jnp.float32
                        )
                        high_sum = high_sum + (word & HIGH_HALF).view(
                            jnp.float32
                        )
                    token_sums.append((low_sum, high_sum))
                # Note (david): the pack rounds each f32 sum to bf16 (nearest
                # even), the even token in the low half of the output word.
                for half in (0, 1):
                    packed = plsc.pack(
                        token_sums[0][half],
                        token_sums[1][half],
                        format=plsc.PackFormat.INTERLEAVED,
                    )
                    out_v[out_slot, half, pair, pl.ds(c, lanes)] = plsc.bitcast(
                        packed, jnp.int32
                    )

    for i in range(NUM_GATHER_SLOTS):

        @pl.when(num_worker_units > i)
        def _(i=i):
            for copy in _index_copies(i):
                copy.start()

    for i in range(NUM_GATHER_SLOTS - 1):

        @pl.when(num_worker_units > i)
        def _(i=i):
            _start_gathers(i)

    @pl.loop(0, num_worker_units)
    def _(i):
        @pl.when(i + NUM_GATHER_SLOTS - 1 < num_worker_units)
        def _():
            _start_gathers(i + NUM_GATHER_SLOTS - 1)

        @pl.when(i + NUM_GATHER_SLOTS < num_worker_units)
        def _():
            for copy in _index_copies(i + NUM_GATHER_SLOTS):
                copy.start()

        for copy in _gather_copies(i):
            copy.wait()

        @pl.when(i >= NUM_OUT_SLOTS)
        def _():
            for copy in _out_copies(i - NUM_OUT_SLOTS):
                copy.wait()

        _sum_unit(i)
        for copy in _out_copies(i):
            copy.start()

    for lag in range(NUM_OUT_SLOTS, 0, -1):

        @pl.when(num_worker_units >= lag)
        def _(lag=lag):
            for copy in _out_copies(num_worker_units - lag):
                copy.wait()


def fills_gather_pipelines(num_tokens: int, hidden: int) -> bool:
    """Whether the work units of num_tokens give every vector subcore a full
    gather pipeline (NUM_GATHER_SLOTS units), the batches on which the DMA
    latencies hide and sc_combine outruns the TensorCore combine."""
    sparse_core = pltpu.get_tpu_info().sparse_core
    num_units = pl.cdiv(num_tokens, BF16_TILE_ROWS) * (
        hidden // BF16_PACKING // SLAB
    )
    return num_units >= (
        NUM_GATHER_SLOTS * sparse_core.num_cores * sparse_core.num_subcores
    )


def sc_combine_index_length(num_tokens: int, top_k: int) -> int:
    """The inv_perm entries sc_combine reads: whole 16-token tiles."""
    return round_up(num_tokens, BF16_TILE_ROWS) * top_k


@functools.partial(jax.jit, static_argnames=("top_k", "num_tokens"))
def sc_combine(
    src: jax.Array, inv_perm: jax.Array, top_k: int, *, num_tokens: int
) -> jax.Array:
    """combine.combine on SparseCore, bit for bit, in the [T, hidden] layout.

    Args:
        src: bf16 expert-sorted rows, already scaled by the routing weights,
            in the packing-dim layout [size_m, 2, hidden // 2].
        inv_perm: int32[>= sc_combine_index_length(num_tokens, top_k)]:
            contribution k of token t is src[inv_perm[t * top_k + k]];
            entries past num_tokens * top_k are rows of src too (gathered for
            the last tile's extra tokens and dropped). Note (david): an
            ignored index (-1) in a column-sliced indirect gather halts the
            SparseCore.
        top_k: Contributions per token.
        num_tokens: The output tokens.

    Returns:
        bf16 [round_up(num_tokens, 16), hidden]; rows past num_tokens are
        not meaningful.
    """
    if (
        src.dtype != jnp.bfloat16
        or src.ndim != 3
        or src.shape[1] != BF16_PACKING
    ):
        raise ValueError(
            "src must be bf16 in the packing-dim layout [size_m, 2,"
            f" hidden // 2], got {src.dtype}{list(src.shape)}"
        )
    if inv_perm.ndim != 1 or inv_perm.dtype != jnp.int32:
        raise ValueError(
            f"inv_perm must be int32[n], got {inv_perm.dtype}"
            f"{list(inv_perm.shape)}"
        )
    sparse_core = pltpu.get_tpu_info().sparse_core
    if sparse_core is None:
        raise RuntimeError("sc_combine needs a TPU with SparseCores")
    cols = src.shape[2]
    if cols % SLAB:
        raise ValueError(f"hidden // 2 = {cols} must be a multiple of {SLAB}")
    index_length = sc_combine_index_length(num_tokens, top_k)
    if inv_perm.shape[0] < index_length:
        raise ValueError(
            f"inv_perm has {inv_perm.shape[0]} entries, the 16-token tiles"
            f" read {index_length}"
        )
    # Note (david): one indirect DMA gathers the largest power-of-two share
    # of the 16-token tile whose pairs fit one index tile.
    tokens_per_gather = BF16_TILE_ROWS
    while tokens_per_gather * top_k > MAX_GATHER_INDICES:
        tokens_per_gather //= 2
    if tokens_per_gather == 0:
        raise ValueError(f"top_k={top_k} exceeds {MAX_GATHER_INDICES} indices")
    buffer_shapes = (
        (
            NUM_INDEX_SLOTS,
            BF16_TILE_ROWS // tokens_per_gather,
            MAX_GATHER_INDICES,
        ),
        (NUM_GATHER_SLOTS, BF16_TILE_ROWS * top_k, 1, SLAB),
        (NUM_OUT_SLOTS, BF16_PACKING, NUM_SUBLANES, SLAB),
    )
    vmem_bytes = WORD_BYTES * sum(math.prod(shape) for shape in buffer_shapes)
    if vmem_bytes > sparse_core.vmem_capacity_bytes:
        raise ValueError(
            f"the buffers of top_k={top_k} take {vmem_bytes} B, over a"
            f" subcore's {sparse_core.vmem_capacity_bytes} B of VMEM"
        )
    padded_tokens = round_up(num_tokens, BF16_TILE_ROWS)
    hidden = BF16_PACKING * cols
    kernel = pl.kernel(
        functools.partial(
            sc_combine_kernel,
            num_tiles=padded_tokens // BF16_TILE_ROWS,
            top_k=top_k,
            tokens_per_gather=tokens_per_gather,
            num_subcores=sparse_core.num_subcores,
            num_workers=sparse_core.num_cores * sparse_core.num_subcores,
            lanes=sparse_core.num_lanes,
        ),
        out_type=jax.ShapeDtypeStruct((padded_tokens, hidden), src.dtype),
        mesh=plsc.VectorSubcoreMesh(
            core_axis_name="core",
            subcore_axis_name="subcore",
            num_cores=sparse_core.num_cores,
        ),
        scratch_types=[pltpu.VMEM(shape, jnp.int32) for shape in buffer_shapes]
        + [
            pltpu.SemaphoreType.DMA((NUM_INDEX_SLOTS,)),
            pltpu.SemaphoreType.DMA((NUM_GATHER_SLOTS,)),
            pltpu.SemaphoreType.DMA((NUM_OUT_SLOTS,)),
        ],
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True, needs_layout_passes=False
        ),
        name=f"moe_sc_combine-t_{num_tokens}-k_{top_k}-n_{hidden}",
        cost_estimate=pl.CostEstimate(
            flops=num_tokens * top_k * hidden,
            bytes_accessed=num_tokens * (top_k + 1) * WORD_BYTES * cols
            + WORD_BYTES * num_tokens * top_k,
            transcendentals=0,
        ),
    )
    return kernel(src, inv_perm)
