"""Static traits of the forward builds.

A traits object holds what one build fixes at trace time: its tile sizes and
layout, head counts and widths, head folds and rotary layout, plus the
constants derived from them as properties. forward_common sizes its scratch
from it and fwd_body reads it; schedules, refs and traced scalars stay
arguments.
"""

import dataclasses
import math

from jax.experimental import pallas as pl

from ..core.tpu import BF16_TILE_ROWS, NUM_LANES, NUM_SUBLANES
from .config import BlockSizes, PagedKVInfo, QKVLayout, TokenMajorInfo
from .rotary import ROTARY_DIM_MULTIPLE


@dataclasses.dataclass(frozen=True, kw_only=True)
class FwdTraits:
    """Static configuration of a dense or varlen forward build.

    num_q_heads and num_kv_heads count flat heads (token-major: batch rows
    times the per-row heads). head_fold consecutive q heads share one physical
    block. is_per_seq marks a per-seq (packed) schedule. guard_threshold is the
    overflow guard's log2 bound for the build's kv length. rotary_dim 0 means no
    fused RoPE; otherwise rotary_batch is the coefficient planes' batch, and
    rotary_k False means K arrives rotated.
    """

    block_sizes: BlockSizes
    num_q_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_v: int
    head_fold: int
    transposed_pv: bool
    token_major: TokenMajorInfo | None
    is_per_seq: bool
    guard_threshold: float
    rotary_dim: int = 0
    rotary_batch: int = 1
    rotary_interleaved: bool = False
    rotary_k: bool = True
    # A dense or varlen build reads separate K and V operands and stages one q
    # head per pipeline head; PagedFwdTraits sets both fields.
    paged: PagedKVInfo | None = dataclasses.field(default=None, init=False)
    gqa_fold: int = dataclasses.field(default=1, init=False)

    def __post_init__(self) -> None:
        bq, bkv = self.bq, self.bkv
        bq_compute, bkv_compute = self.bq_compute, self.bkv_compute
        head_fold = self.head_fold
        head_dim_qk, head_dim_v = self.head_dim_qk, self.head_dim_v
        if bq % bq_compute:
            raise ValueError(f"{bq=} must be a multiple of {bq_compute=}.")
        if bq_compute % NUM_SUBLANES:
            raise ValueError(
                f"{bq_compute=} must be a multiple of {NUM_SUBLANES}."
            )
        if self.is_per_seq and bq % BF16_TILE_ROWS:
            # Note (david): the per-seq head blend steps the output stage in
            # whole 16-row tiles.
            raise ValueError(
                f"per-seq {bq=} must be a multiple of {BF16_TILE_ROWS}."
            )
        if (
            self.token_major is not None
            and head_dim_v % NUM_LANES
            and (head_fold * head_dim_v) % NUM_LANES
        ):
            # Note (david): a sub-tile head_dim (d64) relies on an even
            # head_fold so every DMA group spans whole 128-lane tiles.
            raise ValueError(
                "token-major requires head_dim_v % 128 == 0, or a head_fold"
                " that pads the DMA group to a 128-lane multiple; got"
                f" {head_dim_v=}, {head_fold=}."
            )
        if head_fold > 1 and head_dim_v != head_dim_qk:
            # Note (david): v_buf, o_stage and the f32 o_scratch all scale with
            # head_dim_v, which no tuned fold was measured against; a fold sized
            # for head_dim_qk can miss the scoped VMEM budget by tens of MB.
            raise ValueError(
                "head_fold > 1 requires head_dim_v == head_dim_qk; got"
                f" {head_dim_v} vs {head_dim_qk}."
            )
        if bkv % bkv_compute:
            raise ValueError(f"{bkv=} must be a multiple of {bkv_compute=}.")
        if bkv_compute % NUM_LANES:
            raise ValueError(
                f"{bkv_compute=} must be a multiple of {NUM_LANES}."
            )
        if self.rotary_dim:
            if type(self.rotary_interleaved) is not bool:
                raise ValueError("rotary_interleaved must be a static bool.")
            # Note (david): rotate_tile rotates every dim of the head, so only
            # full rotary is supported.
            if (
                self.rotary_dim % ROTARY_DIM_MULTIPLE
                or self.rotary_dim != head_dim_qk
            ):
                raise ValueError(
                    "Invalid rotary dimension or coefficient dtype."
                )
            if (
                self.num_q_heads % self.rotary_batch
                or self.num_kv_heads % self.rotary_batch
            ):
                raise ValueError(
                    "rotary batch must divide the flattened head counts."
                )
            if self.token_major is not None and self.rotary_batch not in (
                1,
                self.token_major.batch,
            ):
                raise ValueError(
                    "rotary batch must match token-major batch or be shared."
                )
        # A per-seq schedule indexes its kv compute tiles at runtime, which
        # only a sublane token axis allows (see fwd_body's
        # skips_past_frontier).
        assert not self.is_per_seq or (
            self.is_head_dim_minor and not self.transposed_pv
        ), (self.qkv_layout, self.transposed_pv)

    @property
    def bq(self) -> int:
        return self.block_sizes.block_q

    @property
    def bkv(self) -> int:
        return self.block_sizes.block_kv

    @property
    def bq_compute(self) -> int:
        return self.block_sizes.block_q_compute

    @property
    def bkv_compute(self) -> int:
        return self.block_sizes.block_kv_compute

    @property
    def num_stages(self) -> int:
        return self.block_sizes.num_stages

    @property
    def qkv_layout(self) -> QKVLayout:
        return self.block_sizes.qkv_layout

    @property
    def is_paged(self) -> bool:
        return self.paged is not None

    @property
    def is_head_dim_minor(self) -> bool:
        return self.qkv_layout == QKVLayout.HEAD_DIM_MINOR

    @property
    def o_layout(self) -> QKVLayout:
        return (
            QKVLayout.SEQ_MINOR
            if self.transposed_pv
            else QKVLayout.HEAD_DIM_MINOR
        )

    @property
    def head_dim_v_repeats(self) -> int:
        return pl.cdiv(self.head_dim_v, NUM_LANES)

    @property
    def guard_detect_bound(self) -> float:
        return math.exp2(self.guard_threshold)

    @property
    def bkv_repeats(self) -> int:
        return self.bkv_compute // NUM_LANES

    @property
    def num_k_compute_tiles(self) -> int:
        return self.bkv // self.bkv_compute

    @property
    def num_q_compute_tiles(self) -> int:
        return self.bq // self.bq_compute

    @property
    def num_compute_tiles(self) -> int:
        return self.num_k_compute_tiles * self.num_q_compute_tiles

    # Note (david): a group stages group_heads q heads. head_fold runs one
    # fragment pipeline per head, while gqa_fold stacks its heads over the rows
    # of one fragment, so each K and V tile the MXU holds serves frag_rows rows.
    # The row state of pipeline head g is its group_bq rows, ordered (q tile,
    # folded head, token).
    @property
    def group_heads(self) -> int:
        return self.head_fold * self.gqa_fold

    @property
    def group_bq(self) -> int:
        return self.gqa_fold * self.bq

    @property
    def frag_rows(self) -> int:
        return self.gqa_fold * self.bq_compute

    @property
    def token_tile_shape(self) -> tuple[int, int]:
        return (self.bq_compute, self.bkv_compute)

    @property
    def defers_row_finish(self) -> bool:
        # Note (david): a per-seq row is often one or two blocks, so its row-end
        # work is paid every block or two: on v6e (32:8 d128, 512 x 512,
        # one-block rows) each row cost 0.97 us on top of its 1.89 us block,
        # 0.50 of it the unoverlapped last PV. Dense rows are long and keep the
        # row-end drain.
        return self.is_per_seq

    @property
    def width_qk(self) -> int | None:
        if self.token_major is None:
            return None
        else:
            return self.token_major.head_dim_qk

    @property
    def width_v(self) -> int | None:
        if self.token_major is None:
            return None
        else:
            return self.token_major.head_dim_v

    @property
    def q_heads_per_row(self) -> int | None:
        if self.token_major is None:
            return None
        else:
            return self.token_major.num_q_heads

    @property
    def kv_heads_per_row(self) -> int | None:
        if self.token_major is None:
            return None
        else:
            return self.token_major.num_kv_heads

    @property
    def is_lane_fold(self) -> bool:
        return self.token_major is not None and self.head_fold > 1

    @property
    def lane_width_qk(self) -> int | None:
        return self.width_qk if self.is_lane_fold else None

    @property
    def lane_width_v(self) -> int | None:
        return self.width_v if self.is_lane_fold else None

    @property
    def is_lane_paired(self) -> bool:
        # Note (david): a sub-tile lane width (d64) breaks two Mosaic rules: a
        # slot-sliced scratch ref only takes lane windows at whole-tile offset
        # and size, and the head's half of the loaded tile must then be a static
        # value slice. So scratch is sliced per 128-lane pair and g must be a
        # Python int.
        return self.is_lane_fold and self.lane_width_qk % NUM_LANES != 0

    @property
    def heads_per_lane_tile(self) -> int:
        return NUM_LANES // self.lane_width_qk if self.is_lane_paired else 1

    @property
    def rotary_q_heads(self) -> int:
        return self.num_q_heads // self.rotary_batch

    @property
    def rotary_k_heads(self) -> int:
        return self.num_kv_heads // self.rotary_batch

    @property
    def rotary_groups(self) -> int:
        if not self.rotary_dim:
            return 1
        # Note (david): a dense head-major fold may span batch rows, and only
        # then does each folded head need its own coefficient plane.
        if (
            self.rotary_q_heads % self.head_fold
            or self.rotary_k_heads % self.head_fold
        ):
            return self.head_fold
        else:
            return 1


@dataclasses.dataclass(frozen=True, kw_only=True)
class PagedFwdTraits(FwdTraits):
    """Static configuration of a forward build over a paged KV cache.

    K and V come from the one merged interleaved cache paged describes. A group
    is the gqa_fold q heads of one kv head, which gqa_fold > 1 stacks over the
    rows of every score fragment.
    """

    paged: PagedKVInfo = dataclasses.field()
    gqa_fold: int = dataclasses.field()

    def __post_init__(self) -> None:
        gqa_fold, head_fold = self.gqa_fold, self.head_fold
        transposed_pv, qkv_layout = self.transposed_pv, self.qkv_layout
        bq_compute = self.bq_compute
        num_q_heads, num_kv_heads = self.num_q_heads, self.num_kv_heads
        if gqa_fold > 1 and (
            head_fold != 1 or transposed_pv or gqa_fold > NUM_LANES
        ):
            # Note (david): a folded group's lse rides one 128-lane row per
            # token.
            raise ValueError(
                f"{gqa_fold=} needs a paged build with head_fold == 1, no"
                f" transposed pv and at most {NUM_LANES} folded heads; got"
                f" {head_fold=}, {transposed_pv=}."
            )
        if gqa_fold > 1 and bq_compute % BF16_TILE_ROWS:
            # Note (david): a folded q tile collapses (gqa_fold, bq_compute)
            # into fragment rows, which moves no data only on whole bf16 tiles.
            raise ValueError(
                f"a gqa fold needs {bq_compute=} to be a multiple of"
                f" {BF16_TILE_ROWS}."
            )
        if qkv_layout != QKVLayout.HEAD_DIM_MINOR:
            raise ValueError(
                "a paged KV cache needs a HEAD_DIM_MINOR layout; got"
                f" {qkv_layout=}."
            )
        if gqa_fold > 1 and num_q_heads != num_kv_heads * gqa_fold:
            raise ValueError(
                "a gqa fold stacks all of a kv head's q heads; got"
                f" {num_q_heads=}, {num_kv_heads=}, {gqa_fold=}."
            )
        assert head_fold == 1, head_fold
        super().__post_init__()
