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

"""The MoE combine: an inverse-permutation row gather and a top_k sum.

The fused MLP has already multiplied the routing weights into its
expert-sorted output rows, so the combine is a pure gather and sum:

    out[t] = sum_k src[inv_perm[t * top_k + k]]

src is in the packing-dim layout [size_m, packing, hidden // packing], every
row one contiguous HBM block, so each of the top_k contributions is one row
DMA. The kernel sums them on the VPU in f32 over the top_k-major buffer dim
and streams the token-ordered result out with variable-size write-back DMAs.
This replaces an XLA take by the inverse permutation, which materializes the
unsorted [T * top_k, hidden] copy in HBM, and a weighted einsum.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from flywheel_tpu.core.tiling import round_up
from flywheel_tpu.core.tpu import F32_BYTES, NUM_LANES
from flywheel_tpu.moe.config import (
    BF16_PACKING,
    sublane_rows,
    vmem_limit_bytes,
)

# Note (david): the stream depths follow the fused MLP's (depth = ahead + 2):
# the index window of a tile lands one tile before its gather, which issues
# GATHER_AHEAD tiles before its sum.
GATHER_DEPTH = 4
GATHER_AHEAD = 2
OUT_DEPTH = 2
# Note (david): the per-row issue loop is unrolled to amortize the scalar
# core's descriptor cost, the dominant gather overhead. The static top_k inner
# loop already emits top_k descriptors per iteration, so the fori unroll tops
# them up to this burst.
TARGET_ISSUE_BURST = 16
TILE_TOKENS = 128
# Note (david): inv_perm has a 1D (128) HBM tiling, so a window DMA starts on
# a multiple of 128.
IDX_WINDOW_ALIGNMENT = 128


def index_window_capacity(tile_t: int, top_k: int) -> int:
    """Elements of a 128-aligned inv_perm window that covers a token tile
    starting anywhere inside a 128-element block."""
    return round_up(tile_t * top_k, IDX_WINDOW_ALIGNMENT) + IDX_WINDOW_ALIGNMENT


def combine_kernel(
    # In (HBM)
    inv_perm_ref,  # int32[num_tokens * top_k], padded to a multiple of 128
    src_ref,  # [size_m, packing, hidden // packing]
    # Out (HBM)
    out_ref,  # [num_tokens, packing, hidden // packing]
    # Scratch
    buf_ref,  # VMEM [GATHER_DEPTH, tile_t, top_k, packing, hidden // packing]
    gather_sem,  # DMA sem [GATHER_DEPTH]
    out_buf_ref,  # VMEM [OUT_DEPTH, tile_t, packing, hidden // packing]
    out_sem,  # DMA sem [OUT_DEPTH]
    idx_buf_ref,  # SMEM [GATHER_DEPTH, index_window_capacity] int32
    idx_sem,  # DMA sem [GATHER_DEPTH]
    *,
    num_tokens: int,
    top_k: int,
    tile_t: int,
):
    """A three-phase pipeline over token tiles: gather, sum, write back.

    inv_perm streams per tile from HBM: prefetching the whole int32 array
    into the 1 MiB SMEM caps num_tokens * top_k at 262144 (T=32768 at
    top_k 8).
    """
    num_steps = pl.cdiv(num_tokens, tile_t)
    flat_size = num_tokens * top_k
    last_position = flat_size - 1
    flat_size_padded = round_up(flat_size, IDX_WINDOW_ALIGNMENT)
    idx_capacity = idx_buf_ref.shape[1]

    def _idx_window(step):
        """(window_start, window_length, row_offset): the 128-aligned slice
        of inv_perm that covers token tile step, clamped to the padded
        length, and the offset of the tile's first entry inside it."""
        base = step * tile_t * top_k
        window_start = base - base % IDX_WINDOW_ALIGNMENT
        window_length = jnp.minimum(
            idx_capacity, flat_size_padded - window_start
        )
        return (
            pl.multiple_of(window_start, IDX_WINDOW_ALIGNMENT),
            pl.multiple_of(window_length, IDX_WINDOW_ALIGNMENT),
            base - window_start,
        )

    def _issue_idx(step, slot):
        """Streams tile step's inv_perm window into SMEM, a tile ahead of
        its gather, so the DMA latency hides under the gather."""
        window_start, window_length, _ = _idx_window(step)
        pltpu.make_async_copy(
            inv_perm_ref.at[pl.ds(window_start, window_length)],
            idx_buf_ref.at[slot, pl.ds(0, window_length)],
            idx_sem.at[slot],
        ).start()

    def _wait_idx(step, slot):
        _, window_length, _ = _idx_window(step)
        buf = idx_buf_ref.at[slot, pl.ds(0, window_length)]
        pltpu.make_async_copy(buf, buf, idx_sem.at[slot]).wait()

    def _issue_gather(step, slot):
        """One row DMA per (token, k) of token tile step into slot.

        Always exactly tile_t * top_k descriptors, so the wait can use the
        full buffer's byte count; positions past num_tokens (last tile)
        clamp to the final inv_perm entry, and their rows are never written
        out.
        """
        base = step * tile_t * top_k
        _, _, row_offset = _idx_window(step)
        tail = last_position - base

        def _issue(should_clamp: bool):
            def _copy_rows_of_token(t, carry):
                for k in range(top_k):
                    unclamped_position = t * top_k + k
                    if should_clamp:
                        position = jnp.minimum(unclamped_position, tail)
                    else:
                        position = unclamped_position
                    row = idx_buf_ref[slot, row_offset + position]
                    pltpu.make_async_copy(
                        src_ref.at[row],
                        buf_ref.at[slot, t, k],
                        gather_sem.at[slot],
                    ).start()
                return carry

            lax.fori_loop(
                0,
                tile_t,
                _copy_rows_of_token,
                0,
                unroll=max(1, TARGET_ISSUE_BURST // top_k),
            )

        # Note (david): only a partial last tile clamps, as a clamp in every
        # descriptor adds scalar work that slows the whole combine.
        if num_tokens % tile_t == 0:
            _issue(should_clamp=False)
        else:

            @pl.when(step == num_steps - 1)
            def _():
                _issue(should_clamp=True)

            @pl.when(step != num_steps - 1)
            def _():
                _issue(should_clamp=False)

    def _out_copy(step, out_slot):
        """Tile step's write-back; issue and wait build the same one."""
        t_start = step * tile_t
        count = jnp.minimum(tile_t, num_tokens - t_start)
        return pltpu.make_async_copy(
            out_buf_ref.at[out_slot, pl.ds(0, count)],
            out_ref.at[pl.ds(t_start, count)],
            out_sem.at[out_slot],
        )

    for s in range(min(GATHER_AHEAD + 1, num_steps)):
        _issue_idx(s, s % GATHER_DEPTH)
    for s in range(min(GATHER_AHEAD, num_steps)):
        _wait_idx(s, s % GATHER_DEPTH)
        _issue_gather(s, s)

    def _tile_step(c, carry):
        slot = lax.rem(c, GATHER_DEPTH)
        out_slot = lax.rem(c, OUT_DEPTH)

        # Note (david): early issue, before any wait: the target slot's last
        # reader retired >= 2 steps ago, so the descriptor burst overlaps the
        # in-flight DMAs.
        @pl.when(c + GATHER_AHEAD + 1 < num_steps)
        def _():
            step = c + GATHER_AHEAD + 1
            _issue_idx(step, lax.rem(step, GATHER_DEPTH))

        @pl.when(c + GATHER_AHEAD < num_steps)
        def _():
            step = c + GATHER_AHEAD
            _wait_idx(step, lax.rem(step, GATHER_DEPTH))
            _issue_gather(step, lax.rem(step, GATHER_DEPTH))

        gathered = buf_ref.at[slot]
        pltpu.make_async_copy(gathered, gathered, gather_sem.at[slot]).wait()

        @pl.when(c >= OUT_DEPTH)
        def _():
            _out_copy(c - OUT_DEPTH, out_slot).wait()

        # Note (david): statically unrolled f32 adds of slabs along the major
        # buffer dim, so the sum needs no relayout.
        acc = buf_ref[slot, :, 0].astype(jnp.float32)
        for k in range(1, top_k):
            acc += buf_ref[slot, :, k].astype(jnp.float32)
        out_buf_ref[out_slot] = acc.astype(out_buf_ref.dtype)
        _out_copy(c, out_slot).start()
        return carry

    lax.fori_loop(0, num_steps, _tile_step, 0)

    for d in range(min(OUT_DEPTH, num_steps), 0, -1):
        c = num_steps - d
        _out_copy(c, c % OUT_DEPTH).wait()


@functools.partial(
    jax.jit, static_argnames=("top_k", "num_tokens", "interpret")
)
def combine(
    src: jax.Array,
    inv_perm: jax.Array,
    top_k: int,
    *,
    num_tokens: int,
    interpret: bool = False,
) -> jax.Array:
    """The token-ordered sum of expert-sorted rows.

    Equal to src[inv_perm].reshape(T, top_k, ...) summed over top_k in f32,
    in k order from contribution 0, with the gather inside the kernel.

    Args:
        src: bf16 expert-sorted rows, already scaled by the routing weights,
            in the packing-dim layout [size_m, 2, hidden // 2].
        inv_perm: int32[>= num_tokens * top_k]; contribution k of token t is
            src[inv_perm[t * top_k + k]]. Entries past num_tokens * top_k
            are never read as rows.
        top_k: Contributions per token.
        num_tokens: The output tokens.
        interpret: Run the Pallas TPU interpreter, e.g. on CPU.

    Returns:
        [num_tokens, 2, hidden // 2] in src.dtype.
    """
    if src.dtype != jnp.bfloat16 or src.ndim != 3:
        raise ValueError(
            "src must be bf16 in the packing-dim layout [size_m, 2,"
            f" hidden // 2], got {src.dtype}{list(src.shape)}"
        )
    packing, cols = src.shape[1], src.shape[2]
    if packing != BF16_PACKING:
        raise ValueError(
            f"src's dim 1 must be {BF16_PACKING}, got {list(src.shape)}"
        )
    if cols % NUM_LANES != 0:
        raise ValueError(
            f"src's minor dim must be a multiple of {NUM_LANES}, got"
            f" {list(src.shape)}"
        )
    if inv_perm.ndim != 1 or inv_perm.dtype != jnp.int32:
        raise ValueError(
            "inv_perm must be int32[num_tokens * top_k], got"
            f" {inv_perm.dtype}{list(inv_perm.shape)}"
        )
    if top_k < 1 or num_tokens < 1 or inv_perm.shape[0] < num_tokens * top_k:
        raise ValueError(
            f"inv_perm's length {inv_perm.shape[0]} must cover"
            f" num_tokens={num_tokens} x top_k={top_k}"
        )
    # Note (david): a token tile whose row count is not a multiple of the
    # sublane tiling (num_tokens 4 or 20 on v6e) crashes the Mosaic compiler,
    # so a short batch rounds its one tile up; the extra rows gather the last
    # pair again and are never written out.
    sublane = sublane_rows()
    tile_t = min(TILE_TOKENS, round_up(num_tokens, sublane))
    row_bytes = packing * cols * src.dtype.itemsize

    def _vmem_bytes(candidate_tile_t):
        """The gather and out buffers and the f32 sum of one tile."""
        return candidate_tile_t * (
            (GATHER_DEPTH * top_k + OUT_DEPTH) * row_bytes
            + packing * cols * F32_BYTES
        )

    # Note (david): wide rows halve the tile until its buffers fit three
    # quarters of the VMEM limit; the compiler's internal scratch takes the
    # rest.
    budget = vmem_limit_bytes() * 3 // 4
    while _vmem_bytes(tile_t) > budget and tile_t > sublane:
        tile_t = max(sublane, round_up(tile_t // 2, sublane))
    if _vmem_bytes(tile_t) > budget:
        raise ValueError(
            f"the combine buffers ({GATHER_DEPTH} gather and {OUT_DEPTH} out"
            f" tiles at tile_t={tile_t}, top_k={top_k}) take"
            f" {_vmem_bytes(tile_t)} B, over 3/4 of the VMEM limit"
            f" {vmem_limit_bytes()} B"
        )

    # Note (david): the window DMAs read whole 128-element blocks, so
    # inv_perm is padded to a multiple of 128; the padded entries are never
    # summed into a token.
    flat_size = num_tokens * top_k
    window_size = round_up(flat_size, IDX_WINDOW_ALIGNMENT)
    if inv_perm.shape[0] < window_size:
        padded_inv_perm = jnp.pad(
            inv_perm, (0, window_size - inv_perm.shape[0])
        )
    else:
        padded_inv_perm = inv_perm
    return pl.pallas_call(
        functools.partial(
            combine_kernel, num_tokens=num_tokens, top_k=top_k, tile_t=tile_t
        ),
        out_shape=jax.ShapeDtypeStruct((num_tokens, packing, cols), src.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM(
                    (GATHER_DEPTH, tile_t, top_k, packing, cols), src.dtype
                ),
                pltpu.SemaphoreType.DMA((GATHER_DEPTH,)),
                pltpu.VMEM((OUT_DEPTH, tile_t, packing, cols), src.dtype),
                pltpu.SemaphoreType.DMA((OUT_DEPTH,)),
                pltpu.SMEM(
                    (GATHER_DEPTH, index_window_capacity(tile_t, top_k)),
                    jnp.int32,
                ),
                pltpu.SemaphoreType.DMA((GATHER_DEPTH,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes(), disable_bounds_checks=True
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=f"fused_moe_combine-t_{num_tokens}-k_{top_k}-n_{packing * cols}",
        cost_estimate=pl.CostEstimate(
            flops=flat_size * packing * cols,
            bytes_accessed=(flat_size + num_tokens) * row_bytes
            + inv_perm.dtype.itemsize * flat_size,
            transcendentals=0,
        ),
    )(padded_inv_perm, src)
