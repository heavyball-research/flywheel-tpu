"""Shared tile-size and layout types and lane constants for Pallas kernels."""

import dataclasses
import enum
import math
from collections.abc import Sequence

import numpy as np

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8
# Note (david): d64 is the one sub-tile head_dim token-major supports; a
# 64-wide head fills half a lane tile, so heads are DMA'd and folded in pairs.
PAIRED_HEAD_DIM = NUM_LANES // 2
NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))
VMEM_LIMIT_BYTES = 100 * 1024 * 1024
BF16_BYTES = 2
F32_BYTES = 4

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

  The cache holds (num_pages, page_size, heads, head_dim) token rows, either
  merged (one operand whose token row is the num_kv_heads K heads, then the V
  heads) or a K/V pair of operands. is_cache_head_major marks a one-head pair
  passed as the same bytes viewed (num_pages, 1, page_size, head_dim). A kv
  block stages whole token rows, so each head group of kv_heads_per_group
  KV heads (and their q heads) reads its heads out of the same staging.
  is_bitcast_load packs two bf16 heads into one u32 word (TPU builds only).
  """
  num_kv_heads: int
  kv_heads_per_group: int
  page_size: int
  pages_per_seq: int
  is_merged: bool
  is_cache_head_major: bool
  is_bitcast_load: bool

  @property
  def staged_kv_heads(self) -> int:
    """Heads per staged token row of one load part."""
    return 2 * self.num_kv_heads if self.is_merged else self.num_kv_heads


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
