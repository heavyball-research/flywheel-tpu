"""MoE constants and rules the kernels are planned with: the VMEM limit, the
HBM sublane tiling, the activation layout, the fused MLP's tile_m and tail
buckets, and its stream depths."""

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from flywheel_tpu.core.tiling import round_up
from flywheel_tpu.core.tpu import NUM_SUBLANES

# Note (david): the kernels were tuned against 90% of a core's VMEM. The
# Pallas interpreter has no chip to ask, so it takes v6e's capacity.
VMEM_FRACTION = 0.9
V6E_VMEM_CAPACITY_BYTES = 128 * 1024 * 1024

# Note (david): bf16 values per 32-bit word. The packing-dim layout [rows,
# BF16_PACKING, hidden // BF16_PACKING] forces the minimal (2, 128) bf16 HBM
# tile, so every row is one contiguous block that a single DMA moves.
BF16_PACKING = 2

# Note (david): the fused MLP reads token_indices in windows of the 1D HBM
# tile (128 elements), so its row count is padded to a multiple of it.
ROW_ALIGNMENT = 128

# Note (david): every gm tile pushes its expert's whole weights through the
# MXU once, so the whole-expert path sizes a tile to hold a typical expert in
# one tile: the mean rows per expert, plus ROW_SPREAD_SIGMAS standard
# deviations of their binomial spread under random routing, plus a sublane of
# window alignment, rounded up to TILE_M_GRANULE. Heavier experts take more
# tiles and the tail buckets size their last one. MAX_TILE_M caps the lhs and
# out buffers next to the weights and the unrolled bucket bodies in the
# instruction memory. Either weight path steps down by TILE_M_GRANULE to
# MIN_TILE_M while its buffers exceed VMEM.
TILE_M_GRANULE = 64
MAX_TILE_M = 576
ROW_SPREAD_SIGMAS = 2

# Note (david): a tile of fewer rows than MXU_PUSH_FLOOR_ROWS takes about as
# long on the MXU as one of that many: pushing its expert's weights, not
# streaming its rows, bounds it. So no tile_m is smaller, and the tail buckets
# are BASE_TAIL_BUCKETS, plus the mean rows per expert rounded up to
# TILE_M_GRANULE (it holds the lighter half of the experts that fit one
# tile), plus the mean rows of the last tile of an expert that spans several
# tiles, rounded up to the coarser REMAINDER_BUCKET_GRANULE that holds their
# spread. Only when experts average fewer rows than the floor, so that nearly
# every tile is that small, do SMALL_TAIL_BUCKETS pay: the epilogue and the
# stores still scale with the bucket.
MXU_PUSH_FLOOR_ROWS = 128
MIN_TILE_M = MXU_PUSH_FLOOR_ROWS
BASE_TAIL_BUCKETS = (128, 256, 512)
REMAINDER_BUCKET_GRANULE = 128
SMALL_TAIL_BUCKETS = (16, 32, 64)

# Note (david): a TensorCore's instruction memory, in bundles, on v6e and
# v7x. Code past it runs from overlays that are copied in from HBM whenever
# control reaches them, so a loop body larger than it reloads its code on
# every iteration. The Pallas interpreter plans with v6e's generation and
# MXU width, as it has no chip to ask.
INSTRUCTION_MEMORY_BUNDLES = 65536
V6E_GENERATION = 6
V7X_GENERATION = 7
V6E_MXU_COLUMNS = 256

# Note (david): the input streams prefetch one step (lhs) or one group (w1,
# w2) ahead. With no lead every stream exposes its per-step or per-group DMA
# latency, which one step of lead fully covers on this MXU-bound pipeline;
# deeper leads gained nothing. The weight streams double buffer (depth =
# ahead + 1), which holds two whole experts of w1 and of w2. Out double
# buffers the write-back, as a single buffer exposes the write-back retire.
LHS_DEPTH = 3
LHS_AHEAD = 1
W1_DEPTH = 2
W1_AHEAD = 1
W2_DEPTH = 2
W2_AHEAD = 1
OUT_DEPTH = 2
# Note (david): the least depth - ahead of each stream that keeps its
# write-after-read order (see PipelineDepths).
LHS_DEPTH_MARGIN = 2
WEIGHT_DEPTH_MARGIN = 1


@functools.cache
def vmem_limit_bytes() -> int:
    """The scoped VMEM limit the MoE kernels are planned and compiled against.

    Asked on first use, not at import: get_tpu_info() starts the TPU backend,
    which jax.distributed.initialize() has to precede.
    """
    if jax.default_backend() != "tpu":
        capacity = V6E_VMEM_CAPACITY_BYTES
    else:
        capacity = pltpu.get_tpu_info().vmem_capacity_bytes
    return int(capacity * VMEM_FRACTION)


@functools.cache
def sublane_rows() -> int:
    """Rows of a bf16 HBM tile: the alignment of a traced row window.

    It is TpuInfo.get_sublane_tiling for bf16, 8 before v7x and 16 on v7x.
    The kernels' gm tiles start and end on multiples of it.
    """
    if jax.default_backend() != "tpu":
        return NUM_SUBLANES
    else:
        return pltpu.get_tpu_info().get_sublane_tiling(jnp.bfloat16)


@functools.cache
def generation_and_mxu_columns() -> tuple[int, int]:
    """(TPU generation, MXU column count): what the fused MLP's code size is
    estimated from."""
    if jax.default_backend() != "tpu":
        return V6E_GENERATION, V6E_MXU_COLUMNS
    else:
        info = pltpu.get_tpu_info()
        return info.generation, info.mxu_column_size


def default_tile_m(num_rows: int, num_experts: int) -> int:
    """The whole-expert path's tile_m for num_rows expert-sorted rows over
    num_experts experts (see TILE_M_GRANULE)."""
    rows_per_expert = num_rows / num_experts
    typical_rows = (
        rows_per_expert
        + ROW_SPREAD_SIGMAS * math.sqrt(rows_per_expert)
        + sublane_rows()
    )
    tile_m = round_up(math.ceil(typical_rows), TILE_M_GRANULE)
    return min(max(tile_m, MIN_TILE_M), MAX_TILE_M)


def default_tail_bucket_m_values(
    num_rows: int, num_experts: int, tile_m: int
) -> tuple[int, ...]:
    """The tail buckets below tile_m for num_rows expert-sorted rows over
    num_experts experts (see MXU_PUSH_FLOOR_ROWS), in the order the fused MLP
    keeps them while their bodies fit the instruction memory: first the
    buckets that the mean rows per expert predict for most last tiles, then
    the others from the smallest, as a smaller body costs less code."""
    rows_per_expert = math.ceil(num_rows / num_experts)
    predicted = {
        round_up(rows_per_expert, TILE_M_GRANULE),
        round_up(rows_per_expert % tile_m, REMAINDER_BUCKET_GRANULE),
    }
    others = set(BASE_TAIL_BUCKETS)
    if rows_per_expert < MXU_PUSH_FLOOR_ROWS:
        others.update(SMALL_TAIL_BUCKETS)

    def _below_tile_m(buckets):
        return sorted(bucket_m for bucket_m in buckets if 0 < bucket_m < tile_m)

    return tuple(_below_tile_m(predicted) + _below_tile_m(others - predicted))


@dataclasses.dataclass(frozen=True)
class PipelineDepths:
    """Per-stream buffer depth and issue lead of the fused MLP pipeline.

    The input streams (lhs, w1, w2) issue chunk (weight streams: distinct-
    group ordinal) c + ahead during step c (at its top, or inside its compute
    after this tile's own loads of the stream), and have no buffer-free
    semaphores: the only write-after-read guard is program order. Chunk
    c + ahead reuses the slot of chunk c + ahead - depth, whose reader is
    the compute of step c + ahead - depth; the issue must come
    after that step in program order, c + ahead - depth <= c - 1, i.e.
    depth >= ahead + 1 (depth 2, ahead 1 is double buffering). The TensorCore
    issues the vector loads and the DMA starts from one in-order instruction
    stream, so a DMA started after a slot's last load cannot overwrite data
    that load has yet to read. For the weight streams the same bound holds in
    ordinals: the write for ordinal o + depth, into ordinal o's slot, is
    issued on the first tile of ordinal o + depth - ahead, after every tile
    of ordinal o. ahead = 0 degenerates to issue-then-wait inside the same
    step: no prefetch, fully exposed DMA latency, but legal.

    The token_indices and row_scale windows ride the lhs slots one step
    earlier: chunk c + ahead + 1's window is issued at step c, and step c's
    epilogue still reads chunk c's row_scale, so the lhs stream needs
    depth >= ahead + 2 (LHS_DEPTH_MARGIN). The out write-back is issued at
    step c and waited at step c + out_depth before the slot's next store, so
    any out_depth >= 1 is legal (1 serializes the write-back behind the next
    tile's store).

    The depths only reschedule DMAs: outputs are bitwise identical across
    legal configurations. The weight-blocked path runs its own rings and
    ignores the w1 and w2 depths.
    """

    lhs_depth: int = LHS_DEPTH
    lhs_ahead: int = LHS_AHEAD
    w1_depth: int = W1_DEPTH
    w1_ahead: int = W1_AHEAD
    w2_depth: int = W2_DEPTH
    w2_ahead: int = W2_AHEAD
    out_depth: int = OUT_DEPTH

    def __post_init__(self) -> None:
        for name, depth, ahead, margin in (
            ("lhs", self.lhs_depth, self.lhs_ahead, LHS_DEPTH_MARGIN),
            ("w1", self.w1_depth, self.w1_ahead, WEIGHT_DEPTH_MARGIN),
            ("w2", self.w2_depth, self.w2_ahead, WEIGHT_DEPTH_MARGIN),
        ):
            if ahead < 0:
                raise ValueError(f"{name}_ahead must be >= 0, got {ahead}")
            if depth < ahead + margin:
                raise ValueError(
                    f"the {name} stream needs depth >= ahead + {margin}"
                    " (the write-after-read order, see PipelineDepths), got"
                    f" depth={depth} ahead={ahead}"
                )
        if self.out_depth < 1:
            raise ValueError(f"out_depth must be >= 1, got {self.out_depth}")

    def short_name(self) -> str:
        return (
            f"l{self.lhs_depth}.{self.lhs_ahead}"
            f"_w1{self.w1_depth}.{self.w1_ahead}"
            f"_w2{self.w2_depth}.{self.w2_ahead}"
            f"_o{self.out_depth}"
        )
