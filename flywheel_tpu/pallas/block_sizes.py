"""Shared tile-size and layout types and lane constants for Pallas kernels."""

import dataclasses
import enum
import functools
import math
from collections.abc import Sequence

import jax
import numpy as np
from jax.experimental.pallas import tpu as pltpu

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8
# Note (david): Mosaic gives a bf16 VMEM scratch whose row count is a multiple
# of 16 the large (16, 128) tiling (v7x does; v6e may keep (8, 128)), and a
# dynamic vector load or store must start on that tile. Dynamic row accesses
# into bf16 stages therefore step in whole 16-row tiles, which is legal under
# either tiling; HBM DMAs stay on the NUM_SUBLANES grid.
BF16_TILE_ROWS = 2 * NUM_SUBLANES
# Note (david): d64 is the one sub-tile head_dim token-major supports; a
# 64-wide head fills half a lane tile, so heads are DMA'd and folded in pairs.
PAIRED_HEAD_DIM = NUM_LANES // 2
NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))
BF16_BYTES = 2
F32_BYTES = 4
# Kernels on v6e's 128 MiB cores are built against 100 MiB of VMEM; a core with
# less is built against all of it (64 MiB on v7x).
MAX_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


@functools.cache
def vmem_limit_bytes() -> int:
  """The scoped VMEM limit kernels are planned and compiled against.

  Asked on first use, not at import: get_tpu_info() starts the TPU backend,
  which jax.distributed.initialize() has to precede. Off TPU (the Pallas
  interpreter) there is no chip to ask, so builds keep the v6e limit.
  """
  if jax.default_backend() != "tpu":
    return MAX_VMEM_LIMIT_BYTES
  return min(MAX_VMEM_LIMIT_BYTES, pltpu.get_tpu_info().vmem_capacity_bytes)

MIN_NUM_STAGES = 2
# Note (david): the KV-cache kernel issues the next block's DMA into slot
# (s + 1) % stages before it computes on slot s, so a single stage has the
# prefetch overwrite the block being read (measured max_abs_error ~3e-2 vs 1e-3
# for stages >= 2).
STAGES = (2, 3, 4)
DEFAULT_BLOCK_Q_COMPUTE = 256
# Note (david): descending, since pick_tile takes the first candidate that
# divides the axis.
FWD_BLOCKS = (2048, 1024, 512, 256, 128)
FWD_KV_COMPUTE_BLOCKS = (512, 384, 256, 128)
FWD_Q_COMPUTE_BLOCKS = (256, 128)

# Note (david): exp(x) == exp2(x * log2(e)). Folding log2(e) into the score
# scale and the loaded lse lets the softmax use exp2, dropping the per-element
# vmul jnp.exp lowers to on TPU. Gradient matmuls keep the true scale.
LOG2E = math.log2(math.e)


def next_pow2(value: int) -> int:
  return 1 << max(0, int(value) - 1).bit_length()


def round_up(value: int, alignment: int) -> int:
  """Smallest multiple of alignment that is >= value."""
  return -(-value // alignment) * alignment


def pick_tile(candidates: Sequence[int], block: int) -> int:
  """Largest of the descending candidates dividing block, else block itself."""
  return next((tile for tile in candidates if block % tile == 0), block)


def default_block(
    seqlen: int, candidates: Sequence[int], max_seqlen: int | None = None
) -> int:
  """Analytic physical block for a sequence axis, shared by fwd and bwd.

  seqlen <= candidates[0] is one whole-axis block; a longer axis takes the
  largest candidate dividing it. max_seqlen (packed schedules only) caps the
  block at max(candidates[-1], next_pow2(max_seqlen)), relaxed to the largest
  candidate under the cap that divides the axis, since a block spanning several
  sequences leaves the block-diagonal schedule nothing to skip.
  """
  if seqlen <= candidates[0]:
    block = seqlen
  else:
    block = pick_tile(candidates, seqlen)
  if max_seqlen is None:
    return block
  else:
    block_cap = max(candidates[-1], next_pow2(max_seqlen))
    if block <= block_cap:
      return block
    else:
      return pick_tile(
          [tile for tile in candidates if tile <= block_cap], seqlen)


class QKVLayout(enum.IntEnum):
  """Physical q/k/v axis order.

  HEAD_DIM_MINOR is [..., seq_len, head_dim]; SEQ_MINOR is
  [..., head_dim, seq_len].
  """
  HEAD_DIM_MINOR = enum.auto()
  SEQ_MINOR = enum.auto()


def from_head_minor(
    shape: tuple[int, ...], layout: QKVLayout
) -> tuple[int, ...]:
  """Physical shape under layout of a logical (..., seq, head_dim) shape."""
  if layout == QKVLayout.HEAD_DIM_MINOR:
    return shape
  else:
    return (*shape[:-2], shape[-1], shape[-2])


@dataclasses.dataclass(frozen=True, slots=True)
class TokenMajorInfo:
  """Static addressing facts for token-major refs; None means head-major.

  Token-major q/k/v are (batch, seq, heads * head_dim) refs, or
  (total_tokens, heads * head_dim) refs when batch is None, so head h is a
  lane-axis column window. num_q_heads/num_kv_heads are per-row head counts;
  head_dim_qk/head_dim_v are the per-stream column-window widths.
  """
  batch: int | None
  num_q_heads: int
  num_kv_heads: int
  head_dim_qk: int
  head_dim_v: int


@dataclasses.dataclass(frozen=True, slots=True)
class PagedKVInfo:
  """Static addressing facts for a paged KV cache; None means packed K/V.

  The cache is one merged (num_pages, page_size, 2 * num_kv_heads, head_dim)
  operand whose token rows interleave each head's K and V rows, [k0, v0, k1,
  v1, ...], so one head's (K, V) row pair is one 2-row window of the head
  axis. is_bitcast_load reads a staged pair as one u32 word per lane (K in the
  low half, V in the high half), which only the TPU build supports; the
  Pallas interpreter indexes the two rows instead.
  """
  num_kv_heads: int
  page_size: int
  pages_per_seq: int
  is_bitcast_load: bool


@dataclasses.dataclass(frozen=True, slots=True)
class BlockSizes:
  """Tile sizes and physical layout of one attention kernel build.

  block_kv_compute defaults to block_kv; block_q_compute defaults to
  DEFAULT_BLOCK_Q_COMPUTE and is read only by the forward. qkv_layout is shared
  by q/k/v because head_fold allocates one scratch shape per fold group.
  """
  block_q: int
  block_kv: int
  block_kv_compute: int | None = None
  block_q_compute: int | None = None
  num_stages: int = MIN_NUM_STAGES
  qkv_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR

  def __post_init__(self) -> None:
    if type(self.num_stages) is not int or self.num_stages < MIN_NUM_STAGES:
      raise ValueError(
          f"num_stages must be at least {MIN_NUM_STAGES}; got"
          f" {self.num_stages!r}."
      )
    # Note (david): a frozen dataclass can only derive a field default from
    # another field through object.__setattr__ in __post_init__.
    if self.block_kv_compute is None:
      object.__setattr__(self, "block_kv_compute", self.block_kv)
    if self.block_q_compute is None:
      object.__setattr__(self, "block_q_compute", DEFAULT_BLOCK_Q_COMPUTE)
