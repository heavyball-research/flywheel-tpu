"""Static traits of the forward, decode and backward builds.

A traits object holds what one build fixes at trace time: its tile sizes and
layout, head counts and widths, head folds and rotary layout, plus the
constants derived from them as properties. Each forward kernel's scratch
shape function (dense_scratch_shapes, varlen_scratch_shapes,
paged_scratch_shapes) sizes its scratch from it and the forward main loops read
it; schedules, refs and traced scalars stay arguments. DecodeTraits does the
same for the KV-cache decode kernel (kernels/decode.py), and BwdTraits for the
backward kernels (kernels/bwd.py, bwd_pipeline.py).
"""

import dataclasses
import math

import jax.numpy as jnp
from jax.experimental import pallas as pl

from flywheel_tpu.core.tpu import BF16_TILE_ROWS, NUM_LANES, NUM_SUBLANES
from flywheel_tpu.softmax_attention.config import (
    BlockSizes,
    KVCacheConfig,
    PagedKVInfo,
    QKVLayout,
    TokenMajorInfo,
)
from flywheel_tpu.softmax_attention.kv_layout import (
    MIN_BLOCK_KV,
    kv_heads_per_word,
    kv_word_rows,
    merged_cache_heads,
)
from flywheel_tpu.softmax_attention.rotary import ROTARY_DIM_MULTIPLE
from flywheel_tpu.softmax_attention.softmax import overflow_guard_threshold

DECODE_HEAD_DIMS = (64, 128, 256)


@dataclasses.dataclass(frozen=True, kw_only=True)
class QKVTraits:
    """Statics a forward and a backward build share: the tiles, the q and kv
    head counts and widths, the head fold and the q/k/v layout.

    num_q_heads and num_kv_heads count flat heads (token-major: batch rows
    times the per-row heads). head_fold consecutive q heads share one physical
    block. token_major is the fused (.., seq, heads * head_dim) layout, None
    for head-major q/k/v; width_qk / width_v are then its per-head widths.
    """

    block_sizes: BlockSizes
    num_q_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_v: int
    head_fold: int
    token_major: TokenMajorInfo | None

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
    def group_heads(self) -> int:
        """The q heads one schedule step stages."""
        return self.head_fold

    @property
    def num_head_groups(self) -> int:
        return self.num_q_heads // self.group_heads

    @property
    def q_heads_per_kv_head(self) -> int:
        return self.num_q_heads // self.num_kv_heads

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
    def is_lane_paired(self) -> bool:
        # Note (david): a sub-tile lane width (d64) breaks two Mosaic rules: a
        # slot-sliced scratch ref only takes lane windows at whole-tile offset
        # and size, and the head's half of the loaded tile must then be a static
        # value slice. So scratch is sliced per 128-lane pair and g must be a
        # Python int.
        # Note (david): Mosaic takes a lane window of a scratch ref only at
        # whole-tile offset and size once another axis is sliced too, so
        # 64-wide heads are addressed a lane tile (a head pair) at a time:
        # loads take the head's half out of the value and stores go through a
        # tile-wide read-modify-write. The head loop is sequential Python, so
        # the partner half is never stale.
        return self.is_lane_fold and self.width_qk % NUM_LANES != 0

    @property
    def heads_per_lane_tile(self) -> int:
        return NUM_LANES // self.width_qk if self.is_lane_paired else 1


@dataclasses.dataclass(frozen=True, kw_only=True)
class FwdTraits(QKVTraits):
    """Static configuration of a dense or varlen forward build.

    transposed_pv computes PV as o^T. is_per_seq marks a per-seq (packed)
    schedule. guard_threshold is the overflow guard's log2 bound for the
    build's kv length. rotary_dim 0 means no fused RoPE; otherwise
    rotary_batch is the coefficient planes' batch, and rotary_k False means K
    arrives rotated.
    """

    transposed_pv: bool
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
        # only a sublane token axis allows (see the live-tile copy of
        # kernels/fwd_varlen.py's varlen_fwd_body).
        assert not self.is_per_seq or (
            self.is_head_dim_minor and not self.transposed_pv
        ), (self.qkv_layout, self.transposed_pv)

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
    def lane_width_qk(self) -> int | None:
        return self.width_qk if self.is_lane_fold else None

    @property
    def lane_width_v(self) -> int | None:
        return self.width_v if self.is_lane_fold else None

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
        if num_q_heads != num_kv_heads * gqa_fold:
            raise ValueError(
                "a gqa fold stacks all of a kv head's q heads; got"
                f" {num_q_heads=}, {num_kv_heads=}, {gqa_fold=}."
            )
        assert head_fold == 1, head_fold
        super().__post_init__()


@dataclasses.dataclass(frozen=True, kw_only=True)
class DecodeTraits:
    """Static configuration of a KV-cache decode build (kernels/decode.py).

    num_q_heads counts every query row of a cache row: query_tokens > 1
    folds a verify window's tokens into the query heads. kv_config is
    dispatch's (block_kv, block_kv_compute, num_kv_stages, items_per_step);
    items_per_step > 1 packs that many (row, block) items into one loop step,
    on a merged cache only. is_merged_cache
    reads one cache whose token rows interleave the KV heads' K and V rows,
    else a K / V pair of cache_dtype. is_cache_head_major stages tokens on the
    sublane axis, and is_bitcast_load reads staged head pairs as u32 words,
    which only the TPU build supports. has_new appends one token per row.
    capacity is a row's token capacity, which every row's keys stay within.
    page_size and pages_per_seq, the flat block table's pages per row padded
    to whole blocks, are None for an unpaged cache.
    """

    batch: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    query_tokens: int
    kv_config: KVCacheConfig
    cache_dtype: jnp.dtype
    is_merged_cache: bool
    is_cache_head_major: bool
    is_bitcast_load: bool
    has_new: bool
    capacity: int
    page_size: int | None
    pages_per_seq: int | None

    def __post_init__(self) -> None:
        num_kv_heads = self.num_kv_heads
        if len(self.kv_config) != 4:
            raise ValueError(
                "kv_config is dispatch's (block_kv, block_kv_compute,"
                f" num_kv_stages, items_per_step); got {self.kv_config}."
            )
        if self.is_packed and not self.is_merged_cache:
            raise NotImplementedError(
                "items_per_step > 1 packs merged-cache decode only; got a K/V"
                " cache pair."
            )
        if not self.is_merged_cache:
            if self.cache_dtype != jnp.bfloat16:
                raise NotImplementedError(
                    "the K/V cache pair is bfloat16 only; got"
                    f" {self.cache_dtype}."
                )
            if num_kv_heads != 1 and num_kv_heads % 2:
                # Note (david): the packed load bitcasts two bf16 heads into one
                # u32 lane. The plain strided read an odd axis would need leaves
                # a singleton minor-but-one buffer dim that Mosaic pads to a
                # full 8-sublane tile, measured 5-10x slower at batch >= 32 and
                # behind the RPA v3 baseline.
                raise ValueError(
                    "token-major cache needs an even head axis (the packed load"
                    " carries two bf16 heads per u32 lane); got"
                    f" {num_kv_heads=}."
                )
        if self.head_dim not in DECODE_HEAD_DIMS:
            raise ValueError(
                f"head_dim must be one of {DECODE_HEAD_DIMS}; got"
                f" {self.head_dim}."
            )
        if self.capacity % MIN_BLOCK_KV:
            raise ValueError(
                f"capacity must be a multiple of {MIN_BLOCK_KV}; got"
                f" {self.capacity}."
            )

    @property
    def block_kv(self) -> int:
        return self.kv_config[0]

    @property
    def block_kv_compute(self) -> int:
        return self.kv_config[1]

    @property
    def num_kv_stages(self) -> int:
        return self.kv_config[2]

    @property
    def items_per_step(self) -> int:
        return self.kv_config[3]

    @property
    def is_packed(self) -> bool:
        return self.items_per_step > 1

    @property
    def guard_threshold(self) -> float:
        # Note (david): a row's keys stay within the caller's capacity, which
        # the padded table does not raise, so the guard is sized on it.
        return overflow_guard_threshold(self.capacity)

    @property
    def query_heads_per_kv_head(self) -> int:
        return self.num_q_heads // self.num_kv_heads

    @property
    def staged_kv_heads(self) -> int:
        # Note (david): a merged build stages whole token rows, every KV head's
        # (K, V) pair (and an fp8 cache's zero pad head).
        if self.is_merged_cache:
            return merged_cache_heads(self.num_kv_heads, self.cache_dtype)
        else:
            return self.num_kv_heads

    @property
    def word_rows(self) -> int:
        return kv_word_rows(self.cache_dtype)

    @property
    def num_kv_compute_fragments(self) -> int:
        return self.block_kv // self.block_kv_compute

    # Note (david): a bitcast head-pair tile interleaves its two heads per
    # token, so both heads' query groups ride one MXU pass with the off-parity
    # columns masked. That is free while the pair's query rows fit one 8-sublane
    # f32 tile; past it the masked half doubles the softmax work, so larger
    # groups unpack one head per fragment. A merged row's u32 word pairs a
    # head's K with its V instead, so a merged build always unpacks one head per
    # fragment.
    # Note: an fp8 merged word holds a head pair, each head's K byte then its
    # V byte in one 16-bit half, so the pair rides one fragment like the bf16
    # head-pair bitcast.
    @property
    def is_pair_word_fragment(self) -> bool:
        return (
            self.is_merged_cache
            and kv_heads_per_word(self.cache_dtype) == 2
            and NUM_SUBLANES // 2
            < 2 * self.query_heads_per_kv_head
            <= NUM_SUBLANES
            and self.num_kv_heads % 2 == 0
        )

    @property
    def kv_heads_per_fragment(self) -> int:
        if (
            self.is_bitcast_load
            and 2 * self.query_heads_per_kv_head <= NUM_SUBLANES
            and not self.is_merged_cache
        ) or self.is_pair_word_fragment:
            return 2
        else:
            return 1

    @property
    def kv_heads_per_group(self) -> int:
        # Note (david): a merged build whose KV head serves fewer than 8 query
        # heads scores several KV heads through one 8-row query tile: each
        # head's keys are pushed once, as before, but the LHS carries the whole
        # group's query rows, and the head keeps its own rows of the result. The
        # group then runs one softmax over a full f32 tile instead of one per
        # head over a mostly empty one, and its running max, sum and accumulator
        # are whole tiles, which drops the per-head cross-lane broadcasts,
        # sublane rotates and single-row state stores.
        query_heads_per_kv_head = self.query_heads_per_kv_head
        if (
            self.is_merged_cache
            and not self.is_pair_word_fragment
            and NUM_SUBLANES % query_heads_per_kv_head == 0
            and self.num_kv_heads % (NUM_SUBLANES // query_heads_per_kv_head)
            == 0
        ):
            return NUM_SUBLANES // query_heads_per_kv_head
        else:
            return 1

    @property
    def num_head_groups(self) -> int:
        return (
            self.num_kv_heads
            // self.kv_heads_per_fragment
            // self.kv_heads_per_group
        )

    @property
    def rows_per_fragment(self) -> int:
        return (
            self.query_heads_per_kv_head
            * self.kv_heads_per_fragment
            * self.kv_heads_per_group
        )

    @property
    def fragment_keys(self) -> int:
        return self.kv_heads_per_fragment * self.block_kv_compute

    @property
    def cache_tile_rows(self) -> int:
        # Note (david): every cache DMA on the token axis moves whole
        # cache_tile_rows tiles. A head-major stage puts tokens on its bf16
        # sublane axis, which v7x tiles 16 rows deep (BF16_TILE_ROWS; v6e 8),
        # and Mosaic sizes a DMA and its wait between the 8-row HBM tiles and
        # that stage in whole 16-row tiles, so an 8-row remainder was dropped
        # from the copy and the wait returned before the DMA landed (v7x read
        # stale tokens; the leftover semaphore count then released later waits
        # early). A token-major stage puts tokens on a leading axis, where each
        # token is whole tiles and the 8-row granule is exact.
        return BF16_TILE_ROWS if self.is_cache_head_major else NUM_SUBLANES


@dataclasses.dataclass(frozen=True, kw_only=True)
class BwdTraits(QKVTraits):
    """Static configuration of a backward build (kernels/bwd.py).

    q_seq_len / kv_seq_len are the build's q and kv lengths. head_fold q heads
    share one schedule step: an MHA fold, a d64 lane pair, or under GQA/MQA
    one kv head's whole q group. dtype is q's, which dq, dk and dv are written
    in.
    """

    q_seq_len: int
    kv_seq_len: int
    dtype: jnp.dtype

    def __post_init__(self) -> None:
        bq, bkv = self.bq, self.bkv
        bq_compute, bkv_compute = self.bq_compute, self.bkv_compute
        q_seq_len, kv_seq_len = self.q_seq_len, self.kv_seq_len
        head_fold = self.head_fold
        # Note (david): lse/di put seq on the lane axis, so a per-block slice at
        # qi * bq is only tile-aligned when bq is a multiple of NUM_LANES.
        if bq % NUM_LANES:
            raise ValueError(f"block {bq} must be a multiple of {NUM_LANES}.")
        if q_seq_len % bq:
            raise ValueError(f"block {bq} must divide q_seq_len {q_seq_len}.")
        if bq_compute % NUM_LANES or bq % bq_compute:
            raise ValueError(
                f"block_q_compute {bq_compute} must be a multiple of"
                f" {NUM_LANES} and divide block {bq}."
            )
        if bkv % NUM_SUBLANES:
            raise ValueError(
                f"block_kv {bkv} must be a multiple of {NUM_SUBLANES}."
            )
        if kv_seq_len % bkv:
            raise ValueError(
                f"block_kv {bkv} must divide kv_seq_len {kv_seq_len}."
            )
        if (
            not self.is_gqa
            and self.token_major is not None
            and head_fold > 1
            and self.token_major.num_q_heads % head_fold
        ):
            raise ValueError(
                f"token-major head_fold {head_fold} must also divide the"
                f" per-row head count {self.token_major.num_q_heads} (a fold"
                " group cannot cross a batch row boundary)."
            )
        if (
            self.token_major is not None
            and self.width_qk % NUM_LANES
            and head_fold % 2
        ):
            raise ValueError(
                "token-major d64 requires an even head_fold (a DMA group must"
                f" span whole 128-lane tiles); got {head_fold}."
            )
        if bkv_compute % NUM_SUBLANES or bkv % bkv_compute:
            raise ValueError(
                f"block_kv_compute {bkv_compute} must be a multiple of"
                f" {NUM_SUBLANES} and divide block_kv {bkv}."
            )

    @property
    def num_q_blocks(self) -> int:
        return self.q_seq_len // self.bq

    @property
    def num_kv_blocks(self) -> int:
        return self.kv_seq_len // self.bkv

    @property
    def num_q_tiles(self) -> int:
        return self.bq // self.bq_compute

    @property
    def num_kv_tiles(self) -> int:
        return self.bkv // self.bkv_compute

    @property
    def is_gqa(self) -> bool:
        return self.num_kv_heads != self.num_q_heads

    @property
    def is_rectangular(self) -> bool:
        return self.q_seq_len != self.kv_seq_len

    @property
    def kv_fold(self) -> int:
        # Note (david): under GQA/MQA the group's kv side is one physical head,
        # so only the q side folds.
        return 1 if self.is_gqa else self.head_fold

    @property
    def is_single_block_fold(self) -> bool:
        return self.head_fold > 1 and not (self.is_gqa or self.is_lane_paired)
