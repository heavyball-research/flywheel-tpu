"""Building blocks of the forward fragment pipeline.

A forward main loop (kernels/fwd_dense.py's dense_fwd_body,
kernels/fwd_varlen.py's varlen_fwd_body or kernels/fwd_paged.py's
paged_fwd_body) runs these at fixed points: the Q and rotary staging copies
and their in-place rotation, the K and V staging copies, the QK fragment and
its masks, the online softmax, the PV with its pending-PV hand-off across
blocks, and a row's output and lse finish. Each takes the build's FwdTraits
(and Variant, where it calls a hook) plus a small frozen group of the refs it
reads, so a loop hands over its scratch rather than a helper closing over it:

- QRefs: the Q operand, its staging ring and the rotary Q coefficients.
- KVRefs / PagedKVRefs: the contiguous K and V streams, or the paged merged
  cache with its word staging ring and unpacked K and V planes. Both give a
  fragment its V operand through v_tile.
- RotaryStream: a Q or K coefficient stream, as QRefs and KVRefs build it.
- RowStateRefs: row max, row sum, o accumulator, block parity, guard flags.
- PendingPVRefs: the PV a block leaves to the next one.
- OutRefs: the o and lse operands, their stages and the DMA semaphores.

More frozen groups carry values rather than refs: ScoreMask, how a build
masks its scores, KVBlock, the kv block a fragment belongs to, and
SoftmaxPass, how online_softmax treats a copy's fragments. A fragment
travels through a pipeline as a Fragment (its coordinates) and a FragmentPV
(what its PV needs), both NamedTuples a loop can carry. The loop
itself (block descriptors and prefetch, the fragment pipeline copies, the
overflow replay, row parity and the deferred row finish) stays with each
kernel's driver.
"""

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..core.tpu import BF16_TILE_ROWS, NUM_LANES, NUM_SUBLANES
from .config import (
    DEFAULT_MASK_VALUE,
    NN_DIM_NUMBERS,
    NT_DIM_NUMBERS,
    QKVLayout,
    hbm_bf16_tile_rows,
)
from .dma import fold_row, hbm_window
from .kv_layout import (
    kv_heads_per_word,
    kv_word_halves,
    kv_word_rows,
    load_staged_words,
    staged_words,
)
from .mask import apply_packed_bounds_mask
from .rotary import rotate_tile
from .schedule import FwdSchedule, ScheduleRow, WindowFragment
from .softmax import (
    MASKED_ROW_BOUND,
    UNANCHORED_ROW_BOUND,
    UNANCHORED_ROW_MAX,
)
from .traits import FwdTraits
from .variant import Variant, VariantOperands

SEM_Q = 0
SEM_K = 1
SEM_V = 2
SEM_OUT = 3
SEM_LSE = 4
NUM_SEMS = 5
SEM_ROTARY_Q = NUM_SEMS
SEM_ROTARY_K = NUM_SEMS + 1
PENDING_WRITE_PARITY = 0
PENDING_READ_PARITY = 1
PENDING_ROW_START = 2
NUM_PENDING_FIELDS = 3
# Note (david): an empty pending PV reads and writes this trash o slot, so the
# next block issues the pending PV without a branch.
TRASH_PARITY = 2
NUM_O_SLOTS = 3
# Note (david): the kv stream prefetches two blocks ahead, so its ring holds
# the running block, the next one and the one in flight.
NUM_KV_STAGES = 3


class RowFinish(NamedTuple):
    """A q block row's output finish, deferred until its last PV has run.

    prev_out_tok is the previous output block's q_tok, out_count the row's
    output copy index and parity that of its final row state. is_live 0 marks
    no row.
    """

    is_live: jax.Array
    si: jax.Array
    q_tok: jax.Array
    base_head: jax.Array
    num_blend_rows: jax.Array
    prev_out_tok: jax.Array
    out_count: jax.Array
    parity: jax.Array


def fold_lane(
    traits: FwdTraits,
    ref: jax.Array,
    g: jax.Array | int,
    block: int,
    width: int | None,
    layout: QKVLayout,
) -> tuple[jax.Array, jax.Array | int | None]:
    """Pipeline head g's window of a per-slot staging ref, and the half of its
    lane pair's tile the head holds (None unless the fold is lane-paired).

    width is the head's lane width under a lane fold, else None, and layout
    the ref's token axis order.
    """
    if traits.is_lane_paired:
        tile_start = (g // traits.heads_per_lane_tile) * NUM_LANES
        return (
            ref.at[:, pl.ds(tile_start, NUM_LANES)],
            g % traits.heads_per_lane_tile,
        )
    else:
        return fold_row(ref, g, block, traits.head_fold, layout, width), None


def take_half(
    value: jax.Array, half: jax.Array | int | None, width: int | None
) -> jax.Array:
    """value's lane half half of width lanes; value itself for half None."""
    if half is None:
        return value
    else:
        return value[..., half * width : (half + 1) * width]


def set_half(
    value: jax.Array, half: int, width: int, new_half: jax.Array
) -> jax.Array:
    """value with its lane half half of width lanes replaced by new_half."""
    # Note (david): jnp .at[].set lowers to scatter, which Pallas TPU lacks,
    # so replace the half with a static split + concatenate.
    halves = [
        value[..., i * width : (i + 1) * width]
        for i in range(value.shape[-1] // width)
    ]
    halves[half] = new_half
    return jnp.concatenate(halves, axis=-1)


@dataclasses.dataclass(frozen=True, eq=False)
class RotaryStream:
    """A rotary coefficient stream: the coefficient operand, its staging
    ring, the rows a block stages, the heads that share one coefficient
    batch, and the kernel's DMA semaphores with the plane its DMAs signal."""

    hbm: jax.Array
    buf: jax.Array
    block: int
    heads_per_batch: int
    sems: jax.Array
    sem_plane: int


@dataclasses.dataclass(frozen=True, eq=False)
class QRefs:
    """The Q stream: the HBM operand, its staging ring, and under rotary the
    Q coefficient operand and its staging ring (else None), plus the kernel's
    DMA semaphores."""

    hbm: jax.Array
    buf: jax.Array
    rotary_hbm: jax.Array | None
    rotary_buf: jax.Array | None
    sems: jax.Array

    def rotary_stream(self, traits: FwdTraits) -> RotaryStream:
        """The Q coefficient stream, one bq-row block per q block."""
        return RotaryStream(
            hbm=self.rotary_hbm,
            buf=self.rotary_buf,
            block=traits.bq,
            heads_per_batch=traits.rotary_q_heads,
            sems=self.sems,
            sem_plane=SEM_ROTARY_Q,
        )


def q_copy(
    traits: FwdTraits,
    q_refs: QRefs,
    head: jax.Array | int,
    tok: jax.Array | int,
    slot: jax.Array | int,
):
    """The DMA of the q block at (head, tok) into Q staging slot slot."""
    return pltpu.make_async_copy(
        hbm_window(
            q_refs.hbm,
            head,
            tok,
            traits.bq,
            width=traits.width_qk,
            heads_per_row=traits.q_heads_per_row,
            layout=traits.qkv_layout,
            token_major=traits.token_major,
            fold=traits.group_heads,
        ),
        q_refs.buf.at[slot],
        q_refs.sems.at[SEM_Q, slot],
    )


def rotary_copies(
    traits: FwdTraits,
    stream: RotaryStream,
    head: jax.Array | int,
    tok: jax.Array | int,
    slot: jax.Array | int,
):
    """The DMAs of stream's coefficient rows [tok, tok + stream.block) of
    every coefficient group of the group at head into staging slot slot."""
    for g in range(traits.rotary_groups):
        batch_idx = (
            0
            if traits.rotary_batch == 1
            else (head + g) // stream.heads_per_batch
        )
        rows = pl.ds(tok, stream.block)
        coeff_window = (
            stream.hbm.at[batch_idx, rows, :]
            if traits.is_head_dim_minor
            else stream.hbm.at[batch_idx, :, rows]
        )
        yield pltpu.make_async_copy(
            coeff_window,
            stream.buf.at[slot, g],
            stream.sems.at[stream.sem_plane, slot],
        )


def rotate_staged(
    traits: FwdTraits,
    buf: jax.Array,
    coeff_buf: jax.Array,
    slot: jax.Array | int,
    block: int,
) -> None:
    """Rotate every folded head of staging slot slot of buf in place."""
    lane_width_qk = traits.lane_width_qk
    for g in range(traits.head_fold):
        head_ref, half = fold_lane(
            traits, buf.at[slot], g, block, lane_width_qk, traits.qkv_layout
        )
        coeff_group = 0 if traits.rotary_groups == 1 else g
        staged = head_ref[...]
        head_value = take_half(staged, half, lane_width_qk)
        rotated = rotate_tile(
            head_value,
            coeff_buf[slot, coeff_group],
            interleaved=traits.rotary_interleaved,
            head_dim_minor=traits.is_head_dim_minor,
        )
        head_ref[...] = (
            rotated
            if half is None
            else set_half(staged, half, lane_width_qk, rotated)
        )


def rotate_gqa_folded_q(
    traits: FwdTraits, q_refs: QRefs, slot: jax.Array | int
) -> None:
    """Rotate the gqa_fold heads of Q staging slot slot in place."""

    # Note (david): the gqa-folded heads share the paged build's one Q
    # coefficient plane, and a rolled loop keeps one copy of the rotation.
    def _rotate_head(h, carry):
        head_ref = q_refs.buf.at[slot, h]
        head_ref[...] = rotate_tile(
            head_ref[...],
            q_refs.rotary_buf[slot, 0],
            interleaved=traits.rotary_interleaved,
            head_dim_minor=traits.is_head_dim_minor,
        )
        return carry

    lax.fori_loop(0, traits.gqa_fold, _rotate_head, None)


def start_q_row(
    traits: FwdTraits,
    q_refs: QRefs,
    head: jax.Array | int,
    tok: jax.Array | int,
    slot: jax.Array | int,
) -> None:
    """Start the Q (and Q coefficient) DMAs of the row at (head, tok)."""
    q_copy(traits, q_refs, head, tok, slot).start()
    if traits.rotary_dim:
        for dma in rotary_copies(
            traits, q_refs.rotary_stream(traits), head, tok, slot
        ):
            dma.start()


def wait_q_row(
    traits: FwdTraits,
    variant: Variant,
    operands: VariantOperands,
    q_refs: QRefs,
    slot: jax.Array | int,
) -> None:
    """Wait for staging slot slot's Q row, then rotate and scale it in
    place."""
    q_copy(traits, q_refs, 0, 0, slot).wait()
    if traits.rotary_dim:
        for dma in rotary_copies(
            traits, q_refs.rotary_stream(traits), 0, 0, slot
        ):
            dma.wait()
        if traits.gqa_fold > 1:
            rotate_gqa_folded_q(traits, q_refs, slot)
        else:
            rotate_staged(
                traits, q_refs.buf, q_refs.rotary_buf, slot, traits.bq
            )
    # Note (david): each staged Q tile is waited on exactly once per
    # row, so scaling it here replaces an HBM pre-scaled Q copy with the
    # same f32 multiply and cast, leaving the logits unchanged.
    if variant.scales_q(operands):
        q_ref = q_refs.buf.at[slot]
        q_ref[...] = variant.transform_q(q_ref[...], operands)


@dataclasses.dataclass(frozen=True, eq=False)
class KVRefs:
    """The contiguous K and V streams: the HBM operands, their staging rings
    and, under a rotary K, the K coefficient operand and its staging ring
    (else None), plus the kernel's DMA semaphores."""

    k_hbm: jax.Array
    v_hbm: jax.Array
    k_buf: jax.Array
    v_buf: jax.Array
    rotary_hbm: jax.Array | None
    rotary_buf: jax.Array | None
    sems: jax.Array

    def rotary_stream(self, traits: FwdTraits) -> RotaryStream:
        """The K coefficient stream, one bkv-row block per kv block."""
        return RotaryStream(
            hbm=self.rotary_hbm,
            buf=self.rotary_buf,
            block=traits.bkv,
            heads_per_batch=traits.rotary_k_heads,
            sems=self.sems,
            sem_plane=SEM_ROTARY_K,
        )

    def v_tile(
        self,
        traits: FwdTraits,
        g: jax.Array | int,
        kv_compute_index: jax.Array | int,
        kv_slot: jax.Array | int,
    ) -> jax.Array:
        """A fragment's V operand as loaded, before a lane pair's half is
        picked."""
        v_ref, _ = fold_lane(
            traits,
            self.v_buf.at[kv_slot],
            g,
            traits.bkv,
            traits.lane_width_v,
            traits.qkv_layout,
        )
        bkv_compute = traits.bkv_compute
        slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
        return (
            v_ref[slice_k, :] if traits.is_head_dim_minor else v_ref[:, slice_k]
        )


def kv_copies(
    traits: FwdTraits,
    kv_refs: KVRefs,
    kv_head: jax.Array | int,
    tok: jax.Array | int,
    slot: jax.Array | int,
) -> list:
    """The K and V DMAs of the kv block at (kv_head, tok) into staging slot
    slot."""
    return [
        pltpu.make_async_copy(
            hbm_window(
                kv_refs.k_hbm,
                kv_head,
                tok,
                traits.bkv,
                width=traits.width_qk,
                heads_per_row=traits.kv_heads_per_row,
                layout=traits.qkv_layout,
                token_major=traits.token_major,
                fold=traits.head_fold,
            ),
            kv_refs.k_buf.at[slot],
            kv_refs.sems.at[SEM_K, slot],
        ),
        pltpu.make_async_copy(
            hbm_window(
                kv_refs.v_hbm,
                kv_head,
                tok,
                traits.bkv,
                width=traits.width_v,
                heads_per_row=traits.kv_heads_per_row,
                layout=traits.qkv_layout,
                token_major=traits.token_major,
                fold=traits.head_fold,
            ),
            kv_refs.v_buf.at[slot],
            kv_refs.sems.at[SEM_V, slot],
        ),
    ]


def start_kv_block(
    traits: FwdTraits,
    kv_refs: KVRefs,
    kv_head: jax.Array,
    kv_tok: jax.Array,
    slot: jax.Array | int,
) -> None:
    """Start the K and V (and K coefficient) DMAs of the kv block at (kv_head,
    kv_tok) into staging slot slot."""
    for dma in kv_copies(traits, kv_refs, kv_head, kv_tok, slot):
        dma.start()
    if traits.rotary_dim and traits.rotary_k:
        for dma in rotary_copies(
            traits, kv_refs.rotary_stream(traits), kv_head, kv_tok, slot
        ):
            dma.start()


def wait_kv_block(
    traits: FwdTraits, kv_refs: KVRefs, slot: jax.Array | int
) -> None:
    """Wait for staging slot slot's K and V (and K coefficient) block."""
    for dma in kv_copies(traits, kv_refs, 0, 0, slot):
        dma.wait()
    if traits.rotary_dim and traits.rotary_k:
        for dma in rotary_copies(
            traits, kv_refs.rotary_stream(traits), 0, 0, slot
        ):
            dma.wait()


# Note (david): a paged group is the gqa_fold q heads of one kv head, which
# read its (K, V) rows; those of one page are the u32 word window of the
# interleaved cache's head axis that holds them, one DMA: a bf16 head's
# [2 * kv_head, 2 * kv_head + 2), an fp8 head pair's 4 rows.
@dataclasses.dataclass(frozen=True, eq=False)
class PagedKVRefs:
    """The paged K and V stream: the merged interleaved cache (num_pages,
    page_size, cache_heads, head_dim), the ring staging one kv head's word
    rows of a block as (NUM_KV_STAGES, bkv, 1, kv_word_rows, head_dim), the
    (2, bkv, head_dim) K and V planes a staged block unpacks into, and the
    kernel's DMA semaphores."""

    cache_hbm: jax.Array
    pair_buf: jax.Array
    planes: jax.Array
    sems: jax.Array

    @property
    def word_rows(self) -> int:
        return kv_word_rows(self.pair_buf.dtype)

    @property
    def heads_per_word(self) -> int:
        return kv_heads_per_word(self.pair_buf.dtype)

    @property
    def num_cache_pages(self) -> int:
        return self.cache_hbm.shape[0]

    def v_tile(
        self,
        traits: FwdTraits,
        g: jax.Array | int,
        kv_compute_index: jax.Array | int,
        kv_slot: jax.Array | int,
    ) -> jax.Array:
        """A fragment's V operand: the unpacked block's V plane tile, which
        every pipeline head and staging slot reads."""
        del g, kv_slot
        return paged_fragment(traits, self, kv_compute_index, 1)


def paged_buffer_tile(
    kv_refs: PagedKVRefs,
    slot: jax.Array | int,
    token_start: int,
    num_tokens: jax.Array | int,
) -> jax.Array:
    """Rows [token_start, token_start + num_tokens) of staging slot slot."""
    return kv_refs.pair_buf.at[slot, pl.ds(token_start, num_tokens), 0, :, :]


def paged_kv_source(
    kv_refs: PagedKVRefs,
    physical_page: jax.Array,
    num_tokens: jax.Array,
    kv_head: jax.Array,
) -> jax.Array:
    """The word rows of kv_head in the first num_tokens tokens of a cache
    page."""
    word_rows = kv_refs.word_rows
    head_row = pl.multiple_of(
        kv_head // kv_refs.heads_per_word * word_rows, word_rows
    )
    return kv_refs.cache_hbm.at[
        physical_page, pl.ds(0, num_tokens), pl.ds(head_row, word_rows), :
    ]


def paged_page_rows(
    traits: FwdTraits, kv_load_end: jax.Array, kv: jax.Array
) -> list[jax.Array]:
    """The rows each page DMA of kv block kv moves."""
    page_size = traits.paged.page_size
    bkv = traits.bkv
    # Note (david): every page DMA of a block is issued, so its wait needs
    # no branch. A page past the load extent moves one sublane tile of a
    # clamped page, whose rows the unpack zeroes.
    num_load_rows = jnp.clip(kv_load_end - kv * bkv, 0, bkv)
    return [
        pl.multiple_of(
            jnp.clip(
                num_load_rows - page * page_size,
                NUM_SUBLANES,
                page_size,
            ),
            NUM_SUBLANES,
        )
        for page in range(bkv // page_size)
    ]


def start_paged_kv_block(
    traits: FwdTraits,
    schedule: FwdSchedule,
    kv_refs: PagedKVRefs,
    page_base: jax.Array,
    kv_head: jax.Array,
    kv_load_end: jax.Array,
    kv: jax.Array,
    slot: jax.Array | int,
) -> None:
    """Start the page DMAs of kv_head's word rows of kv block kv, whose pages
    the block table row at page_base names, into staging slot slot."""
    page_size = traits.paged.page_size
    for page, num_rows in enumerate(paged_page_rows(traits, kv_load_end, kv)):
        physical_page = jnp.clip(
            schedule.kv_page(page_base, kv, page),
            0,
            kv_refs.num_cache_pages - 1,
        )
        pltpu.make_async_copy(
            paged_kv_source(kv_refs, physical_page, num_rows, kv_head),
            paged_buffer_tile(kv_refs, slot, page * page_size, num_rows),
            kv_refs.sems.at[SEM_K, slot],
        ).start()


def wait_paged_kv_block(
    traits: FwdTraits,
    kv_refs: PagedKVRefs,
    kv_load_end: jax.Array,
    kv: jax.Array,
    slot: jax.Array | int,
) -> None:
    """Wait for the page DMAs of kv block kv into staging slot slot."""
    # Note (david): every page signals the one semaphore by its size,
    # and a DMA wait only needs a destination of the awaited size, so
    # one wait on a window as long as all the pages together covers the
    # block.
    num_rows = pl.multiple_of(
        sum(paged_page_rows(traits, kv_load_end, kv)), NUM_SUBLANES
    )
    staged_window = paged_buffer_tile(kv_refs, slot, 0, num_rows)
    pltpu.make_async_copy(
        staged_window, staged_window, kv_refs.sems.at[SEM_K, slot]
    ).wait()


def paged_fragment(
    traits: FwdTraits,
    kv_refs: PagedKVRefs,
    kv_compute_index: jax.Array | int,
    kv_part: int,
) -> jax.Array:
    """The (bkv_compute, head_dim) K (kv_part 0) or V (kv_part 1) plane tile
    of the unpacked paged block."""
    bkv_compute = traits.bkv_compute
    tile_start = pl.multiple_of(kv_compute_index * bkv_compute, NUM_LANES)
    return kv_refs.planes[kv_part, pl.ds(tile_start, bkv_compute), :]


def unpack_paged_block(
    traits: FwdTraits,
    kv_refs: PagedKVRefs,
    kv_slot: jax.Array,
    kv_head: jax.Array,
    kv_load_end: jax.Array,
    kv_used_end: jax.Array,
    kv: jax.Array,
) -> None:
    """Unpack kv_head's K and V rows of kv block kv, staged in slot kv_slot,
    into the K and V planes; V rows from kv_used_end (or past the load extent
    kv_load_end) on unpack as zero."""
    bkv, bkv_compute = traits.bkv, traits.bkv_compute
    pair_buf, planes = kv_refs.pair_buf, kv_refs.planes
    heads_per_word = kv_refs.heads_per_word
    valid_rows = jnp.clip(
        jnp.minimum(
            kv_used_end - kv * bkv,
            jnp.clip(kv_load_end - kv * bkv, 0, bkv),
        ),
        0,
        bkv,
    )
    # Note (david): a staged word holds the kv head's K and V rows,
    # split into planes once per compute tile, not once per q tile.
    # Staged rows from valid_rows on may hold NaN (cache slots past
    # seqused_k or stale VMEM), and a zero probability times a NaN
    # value is still NaN, so they unpack as zero values; the keys
    # are masked and skip the select, which keeps it off the path to
    # the first QK.
    for tile in range(traits.num_k_compute_tiles):
        tile_rows = pl.ds(tile * bkv_compute, bkv_compute)

        def _valid_only(rows, tile=tile):
            row_ids = tile * bkv_compute + lax.broadcasted_iota(
                jnp.int32, rows.shape, 0
            )
            return jnp.where(row_ids < valid_rows, rows, jnp.zeros_like(rows))

        if traits.paged.is_bitcast_load:
            words = load_staged_words(
                staged_words(pair_buf, kv_slot),
                tile * bkv_compute,
                bkv_compute,
                1,
            )
            k_rows, _ = kv_word_halves(words, kv_head, pair_buf.dtype)
            _, v_rows = kv_word_halves(
                _valid_only(words), kv_head, pair_buf.dtype
            )
        else:
            k_rows, v_rows = (
                pair_buf[
                    kv_slot,
                    tile_rows,
                    0,
                    kv_part * heads_per_word + kv_head % heads_per_word,
                    :,
                ]
                for kv_part in range(2)
            )
            v_rows = _valid_only(v_rows)
        # Note (david): an fp8 block is widened to bf16 once per
        # staged block rather than fed to the MXU as fp8, which
        # makes Mosaic rescale every f32 score and output tile for
        # e4m3fn's bias (v6e: 0.356 against 0.349 ms a 1024-token
        # chunk).
        for kv_part, plane_rows in enumerate((k_rows, v_rows)):
            planes[kv_part, tile_rows, :] = plane_rows.astype(planes.dtype)


@dataclasses.dataclass(frozen=True, eq=False)
class RowStateRefs:
    """The softmax and output state of the q block row running.

    row_max holds each row's anchor and row_sum its (2, ...) parity pair of
    sums, every lane a row's value; o is the f32 output accumulator's
    NUM_O_SLOTS slots, the parity pair plus TRASH_PARITY. A pipeline head g
    owns rows [g * group_bq, (g + 1) * group_bq). parity is the SMEM (1,)
    new parity of the block running and guard the SMEM (head_fold,) overflow
    flags of its pipeline heads.
    """

    row_max: jax.Array
    row_sum: jax.Array
    o: jax.Array
    parity: jax.Array
    guard: jax.Array

    def block_parities(self) -> tuple[jax.Array, jax.Array]:
        """The (new, old) row-state parity of the block running, read from
        SMEM at every use."""
        new_parity = self.parity[0]
        return new_parity, 1 - new_parity


@dataclasses.dataclass(frozen=True, eq=False)
class KVBlock:
    """One kv block of a q block row, as its fragments read it.

    q_slot and q_tok are the row's Q staging slot and first token, kv_slot
    and kv_tok the block's K and V staging slot and first token, and is_first
    marks the row's first block.
    """

    q_slot: jax.Array
    q_tok: jax.Array
    kv_slot: jax.Array
    kv_tok: jax.Array
    is_first: jax.Array


@dataclasses.dataclass(frozen=True, eq=False)
class ScoreMask:
    """How a build masks its score fragments.

    causal_offset shifts the query positions of the variant's static window
    (Variant.mask_offset). bounds is a per-seq build's (num_k_compute_tiles,
    2, bkv_compute) int32 bounds rows (None on a dense build), whose second
    row is a span row when has_q_span. defers_row_start: a row may score
    masked fragments before its first visible key, so it anchors on that key
    rather than on its first fragment.
    """

    causal_offset: int | jax.Array
    bounds: jax.Array | None
    has_q_span: bool
    defers_row_start: bool


class Fragment(NamedTuple):
    """One score fragment of the block running: pipeline head g's q compute
    tile q_tile against kv compute tile kv_tile. Each is a Python int where
    the copy knows it, else an int32 scalar; a pipeline carries it."""

    g: jax.Array | int
    kv_tile: jax.Array | int
    q_tile: jax.Array | int


class FragmentPV(NamedTuple):
    """What a fragment's online softmax leaves its PV: the probs, the
    row-start flags and the recovery rescale (None on the ordinary pass). A
    pipeline carries it from a fragment's softmax to its PV."""

    probs: jax.Array
    is_row_start: jax.Array
    rescale: jax.Array | None


@dataclasses.dataclass(frozen=True, eq=False)
class SoftmaxPass:
    """How online_softmax treats a copy's fragments.

    needs_mask applies the variant's window mask, and apply_bounds (static)
    the per-seq bounds rows. guard_recovery runs the overflow replay, which
    rebases the anchor and returns the fragment's rescale. last_kv_tile is the
    block's last kv compute tile that runs, where the ordinary pass checks the
    guard; None leaves the check to the caller. is_tile_zero is a static bool
    when the caller knows whether the fragment's kv tile is 0, else None.
    window_fragment is the fragment's static plan entry on a window block
    copy, None on the generic copies.
    """

    needs_mask: jax.Array | bool
    guard_recovery: bool
    apply_bounds: bool
    last_kv_tile: int | jax.Array | None
    is_tile_zero: bool | None = None
    window_fragment: WindowFragment | None = None


def qk(
    traits: FwdTraits,
    variant: Variant,
    q_refs: QRefs,
    kv_refs: KVRefs | PagedKVRefs,
    block: KVBlock,
    frag: Fragment,
) -> jax.Array:
    """The (frag_rows, bkv_compute) log2-unit logits of fragment frag of the
    staged block."""
    bq_compute, bkv_compute = traits.bq_compute, traits.bkv_compute
    lane_width_qk = traits.lane_width_qk
    frag_rows = traits.frag_rows
    q_buf = q_refs.buf
    g, kv_compute_index, q_compute_index = frag
    q_slot, kv_slot = block.q_slot, block.kv_slot
    slice_q = pl.ds(q_compute_index * bq_compute, bq_compute)
    if traits.is_paged:
        if traits.gqa_fold > 1:
            # Note (david): the q tile's tokens of every folded head,
            # stacked head-major over the fragment rows; bq_compute is
            # whole bf16 tiles, so the reshape moves no data.
            q = q_buf[q_slot, :, slice_q, :].reshape(frag_rows, q_buf.shape[-1])
        else:
            q_ref, _ = fold_lane(
                traits,
                q_buf.at[q_slot],
                g,
                traits.bq,
                lane_width_qk,
                traits.qkv_layout,
            )
            q = q_ref[slice_q, :]
        k = paged_fragment(traits, kv_refs, kv_compute_index, 0)
        dim_numbers = NT_DIM_NUMBERS
    else:
        q_ref, q_half = fold_lane(
            traits,
            q_buf.at[q_slot],
            g,
            traits.bq,
            lane_width_qk,
            traits.qkv_layout,
        )
        k_ref, k_half = fold_lane(
            traits,
            kv_refs.k_buf.at[kv_slot],
            g,
            traits.bkv,
            lane_width_qk,
            traits.qkv_layout,
        )
        slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
        if traits.is_head_dim_minor:
            q = take_half(q_ref[slice_q, :], q_half, lane_width_qk)
            k = take_half(k_ref[slice_k, :], k_half, lane_width_qk)
            dim_numbers = NT_DIM_NUMBERS
        else:
            q = q_ref[:, slice_q].T
            k = k_ref[:, slice_k]
            dim_numbers = NN_DIM_NUMBERS
    raw_logits = lax.dot_general(
        q, k, dim_numbers, preferred_element_type=jnp.float32
    )
    assert raw_logits.shape == (frag_rows, bkv_compute)
    # Note (david): the softcap is applied before masking.
    return variant.transform_logits(raw_logits)


def masked_scores(
    traits: FwdTraits,
    variant: Variant,
    mask: ScoreMask,
    block: KVBlock,
    frag: Fragment,
    logits: jax.Array,
    softmax_pass: SoftmaxPass,
) -> jax.Array:
    """Fragment frag's logits with the variant's window mask applied where
    softmax_pass.needs_mask holds, and the per-seq bounds rows where
    softmax_pass.apply_bounds."""
    bq_compute, bkv_compute = traits.bq_compute, traits.bkv_compute
    token_tile_shape = traits.token_tile_shape
    gqa_fold = traits.gqa_fold
    kv_compute_index, q_compute_index = frag.kv_tile, frag.q_tile
    needs_mask = softmax_pass.needs_mask
    apply_bounds = softmax_pass.apply_bounds
    # Note (david): per-seq bounds live in unshifted q coordinates,
    # while the static window lives in coordinates shifted by
    # causal_offset.
    q_unshifted = block.q_tok + q_compute_index * bq_compute
    q_abs = q_unshifted + mask.causal_offset
    k_abs = block.kv_tok + kv_compute_index * bkv_compute

    def _apply_window_mask(scores):
        return variant.mask_logits(scores, q_abs, k_abs, token_tile_shape)

    # Note (david): a mask depends on the row's token alone, so a gqa
    # fold masks the (gqa_fold, bq_compute, bkv_compute) view of the
    # fragment, and one token-plane comparison broadcasts over its
    # heads.
    if gqa_fold > 1:
        token_logits = logits.reshape(gqa_fold, *token_tile_shape)
    else:
        token_logits = logits
    window_masked = lax.cond(
        needs_mask,
        _apply_window_mask,
        lambda scores: scores,
        token_logits,
    )

    if apply_bounds:
        # Note (david): apply_bounds is static so interior blocks never
        # trace these VPU ops per score tile. The bounds rows sit in one
        # slot per kv compute tile, since the tile loop indexes the tile
        # at runtime and Mosaic slices lanes only at static offsets; the
        # span row exists only under a left window, so every other build
        # keeps a single comparison.
        q_ids = q_unshifted + lax.broadcasted_iota(
            jnp.int32, token_tile_shape, 0
        )
        staged_bounds = mask.bounds.at[kv_compute_index]
        if mask.has_q_span:
            scores = apply_packed_bounds_mask(
                window_masked,
                DEFAULT_MASK_VALUE,
                q_ids,
                staged_bounds[0:1, :],
                staged_bounds[1:2, :],
            )
        else:
            scores = jnp.where(
                q_ids >= staged_bounds[0:1, :],
                window_masked,
                DEFAULT_MASK_VALUE,
            )
    else:
        scores = window_masked
    return scores.reshape(logits.shape)


def tile_parity(
    kv_compute_index: jax.Array | int,
    is_tile_zero: bool | None,
    new_parity: jax.Array,
    old_parity: jax.Array,
) -> jax.Array:
    """The parity a kv tile reads the row state from, the old one at tile 0;
    is_tile_zero is a static bool where the caller knows, else None."""
    if is_tile_zero is None:
        return jnp.where(kv_compute_index == 0, old_parity, new_parity)
    else:
        return old_parity if is_tile_zero else new_parity


def flag_overflow_risk(
    traits: FwdTraits,
    state: RowStateRefs,
    g: jax.Array | int,
    row_sum: jax.Array,
    is_checked: jax.Array | bool,
) -> None:
    """Raise pipeline head g's guard flag if is_checked and a row of row_sum
    could hold a score past the overflow guard."""
    # Note (david): row_sum upper-bounds every single score of the row,
    # so this detector may request a needless replay but never misses a
    # score above the guard bound. Checking at the block's last live
    # tile is enough, since a tile past a q tile's own frontier adds
    # exactly 0 to every anchored row's sum.
    has_overflow_risk = jnp.any(
        jnp.logical_not(row_sum <= traits.guard_detect_bound)
    )
    state.guard[g] = jnp.maximum(
        state.guard[g],
        jnp.where(is_checked, has_overflow_risk.astype(jnp.int32), 0),
    )


def online_softmax(
    traits: FwdTraits,
    variant: Variant,
    state: RowStateRefs,
    mask: ScoreMask,
    block: KVBlock,
    frag: Fragment,
    logits: jax.Array,
    softmax_pass: SoftmaxPass,
) -> FragmentPV:
    """Mask fragment frag's logits and fold them into its pipeline head's row
    state; returns what the fragment's PV needs.

    softmax_pass says which masks apply, whether this is the overflow replay
    (which also returns the fragment's rescale) and where the guard is
    checked.
    """
    group_bq, frag_rows = traits.group_bq, traits.frag_rows
    g, kv_compute_index, q_compute_index = frag
    guard_recovery = softmax_pass.guard_recovery
    window_fragment = softmax_pass.window_fragment
    is_tile_zero = softmax_pass.is_tile_zero
    is_first = block.is_first
    defers_row_start = mask.defers_row_start
    head_rows = pl.ds(g * group_bq, group_bq)
    new_parity, old_parity = state.block_parities()
    row_sum_new_ref = state.row_sum.at[new_parity, head_rows]
    row_max_scratch_g = state.row_max.at[head_rows]

    slice_q = pl.ds(q_compute_index * frag_rows, frag_rows)
    scores = masked_scores(
        traits, variant, mask, block, frag, logits, softmax_pass
    )

    read_parity = tile_parity(
        kv_compute_index, is_tile_zero, new_parity, old_parity
    )
    row_sum_read_ref = state.row_sum.at[read_parity, head_rows]

    state_shape = row_max_scratch_g[slice_q, :].shape
    frag_row_max = lax.broadcast_in_dim(scores.max(axis=-1), state_shape, (0,))
    if window_fragment is None:
        is_row_start = jnp.logical_and(
            is_first,
            kv_compute_index == 0 if is_tile_zero is None else is_tile_zero,
        )
    else:
        # Note (david): a band clipped at kv 0 starts at the row's first
        # block.
        is_row_start = jnp.logical_or(
            window_fragment.starts_band,
            jnp.logical_and(is_first, kv_compute_index == 0),
        )
    row_max_prev = row_max_scratch_g[slice_q, :]
    if defers_row_start:
        # Note (david): a row whose fragments so far all lay before its
        # window holds UNANCHORED_ROW_MAX, which zeroes every
        # probability it scores, so its row_sum and o stay exactly 0
        # until its first visible key anchors it, and no rescale is due
        # then.
        was_unanchored = row_max_prev >= UNANCHORED_ROW_BOUND
        takes_frag_anchor = jnp.logical_or(is_row_start, was_unanchored)
        frag_anchor = jnp.where(
            frag_row_max > MASKED_ROW_BOUND,
            frag_row_max,
            UNANCHORED_ROW_MAX,
        )
    else:
        takes_frag_anchor = is_row_start
        frag_anchor = frag_row_max
    if guard_recovery:
        # Note (david): the recovery replay rebases the anchor onto the
        # overflowing fragment; the ordinary pass keeps the anchor it
        # first picked.
        should_rebase = frag_row_max - row_max_prev > traits.guard_threshold
        row_max = jnp.where(
            takes_frag_anchor,
            frag_anchor,
            jnp.where(
                should_rebase,
                jnp.maximum(row_max_prev, frag_row_max),
                row_max_prev,
            ),
        )
    else:
        row_max = jnp.where(takes_frag_anchor, frag_anchor, row_max_prev)
    # Note (david): logits are in log2 units, so exp2 drops the
    # per-element multiply jnp.exp lowers to on TPU.
    probs = jnp.exp2(scores - jnp.tile(row_max, (1, traits.bkv_repeats)))
    frag_row_sum = lax.broadcast_in_dim(probs.sum(axis=-1), state_shape, (0,))
    if guard_recovery and defers_row_start:
        # Note (david): an unanchored row's state is exactly 0, but
        # exp2(UNANCHORED_ROW_MAX - anchor) is inf, and inf * 0 is NaN.
        rescale = jnp.where(
            was_unanchored, 0.0, jnp.exp2(row_max_prev - row_max)
        )
        row_sum_prev = rescale * row_sum_read_ref[slice_q, :]
    elif guard_recovery:
        rescale = jnp.exp2(row_max_prev - row_max)
        row_sum_prev = rescale * row_sum_read_ref[slice_q, :]
    else:
        rescale = None
        row_sum_prev = row_sum_read_ref[slice_q, :]
    row_sum_new = jnp.where(
        is_row_start, frag_row_sum, row_sum_prev + frag_row_sum
    )
    row_sum_new_ref[slice_q, :] = row_sum_new
    row_max_scratch_g[slice_q, :] = row_max
    if guard_recovery:
        # Note (david): the rescale rides the pipeline carry to this
        # fragment's PV. A per-q-tile scratch would race: once skipped
        # fragments make two fragments of one q tile adjacent, the later
        # softmax overwrites it before the earlier fragment's PV reads
        # it.
        rescale = jnp.where(is_row_start, 1.0, rescale)
    elif window_fragment is not None:
        flag_overflow_risk(
            traits, state, g, row_sum_new, window_fragment.checks_guard
        )
    elif softmax_pass.last_kv_tile is not None:
        flag_overflow_risk(
            traits,
            state,
            g,
            row_sum_new,
            kv_compute_index == softmax_pass.last_kv_tile,
        )
    return FragmentPV(probs, is_row_start, rescale)


def pv(
    traits: FwdTraits,
    state: RowStateRefs,
    g: jax.Array | int,
    q_compute_index: jax.Array | int,
    probs: jax.Array,
    v_tile: jax.Array,
    is_row_start: jax.Array,
    rescale: jax.Array | None,
    *,
    write_parity: jax.Array,
    read_parity: jax.Array,
) -> None:
    """Accumulate a fragment's probs @ V into pipeline head g's o rows of q
    compute tile q_compute_index.

    v_tile is the kv refs' v_tile operand, and rescale the fragment's
    recovery rescale (None on the ordinary pass). The PV of a fragment reads
    o from read_parity and writes it to write_parity.
    """
    group_bq, frag_rows = traits.group_bq, traits.frag_rows
    transposed_pv = traits.transposed_pv
    o_scratch = state.o
    head_rows = pl.ds(g * group_bq, group_bq)
    v_half = g % traits.heads_per_lane_tile if traits.is_lane_paired else None

    if traits.is_paged:
        o_curr = lax.dot_general(
            probs,
            v_tile,
            NN_DIM_NUMBERS,
            preferred_element_type=jnp.float32,
        )
    elif traits.is_head_dim_minor:
        o_curr = lax.dot_general(
            probs,
            take_half(v_tile, v_half, traits.lane_width_v),
            NN_DIM_NUMBERS,
            preferred_element_type=jnp.float32,
        )
    elif transposed_pv:
        # Note (david): contracting the SEQ_MINOR (head_dim, bkv_compute) v
        # slab against probs yields o^T and fills the MXU's K and N dims
        # with bkv_compute and bq_compute instead of capping N at head_dim.
        o_curr = lax.dot_general(
            v_tile,
            probs,
            NT_DIM_NUMBERS,
            preferred_element_type=jnp.float32,
        )
    else:
        o_curr = lax.dot_general(
            probs,
            v_tile,
            NT_DIM_NUMBERS,
            preferred_element_type=jnp.float32,
        )

    slice_q = pl.ds(q_compute_index * frag_rows, frag_rows)
    if transposed_pv:
        # Note (david): q is the lane axis of the transposed o scratch, and
        # Mosaic only slices lanes dynamically at an offset that is static
        # per unrolled copy, as q_compute_index is.
        o_new_ref = o_scratch.at[write_parity, :, head_rows]
        o_read_ref = o_scratch.at[read_parity, :, head_rows]
        tile_index = (slice(None), slice_q)
    else:
        o_new_ref = o_scratch.at[write_parity, head_rows]
        o_read_ref = o_scratch.at[read_parity, head_rows]
        tile_index = (slice_q, slice(None))
    o_read = o_read_ref[tile_index]
    if rescale is None:
        o_prev = o_read
    elif transposed_pv:
        # Note (david): every lane of the rescale holds the row value, so
        # lane 0 transposed to (1, bq_compute) broadcasts over the
        # head_dim_v sublanes without any lane tiling.
        o_prev = rescale[:, 0:1].T * o_read
    else:
        o_prev = (
            jnp.tile(rescale, (1, traits.head_dim_v_repeats))[
                ..., : o_scratch.shape[-1]
            ]
            * o_read
        )
    o_new_ref[tile_index] = jnp.where(is_row_start, o_curr, o_prev + o_curr)


def fragment_pv(
    traits: FwdTraits,
    state: RowStateRefs,
    kv_refs: KVRefs | PagedKVRefs,
    block: KVBlock,
    frag: Fragment,
    frag_pv: FragmentPV,
    is_tile_zero: bool | None = None,
) -> None:
    """The PV of block's fragment frag, from what its online_softmax returned;
    is_tile_zero as in SoftmaxPass."""
    g, kv_compute_index, q_compute_index = frag
    new_parity, old_parity = state.block_parities()
    pv(
        traits,
        state,
        g,
        q_compute_index,
        frag_pv.probs,
        kv_refs.v_tile(traits, g, kv_compute_index, block.kv_slot),
        frag_pv.is_row_start,
        frag_pv.rescale,
        write_parity=new_parity,
        read_parity=tile_parity(
            kv_compute_index, is_tile_zero, new_parity, old_parity
        ),
    )


@dataclasses.dataclass(frozen=True, eq=False)
class PendingPVRefs:
    """The PV a block leaves to the next one: its (frag_rows, bkv_compute)
    f32 probs, its V tile and the SMEM (NUM_PENDING_FIELDS,) int32 fields
    (write parity, read parity, row start), whose parities are TRASH_PARITY
    when no PV is pending."""

    probs: jax.Array
    v: jax.Array
    fields: jax.Array


# Note (david): the mask-free copy leaves its last fragment's PV pending in
# scratch (a pl.when copy returns no values), and the next block's copy
# issues it after its own first QK and softmax, so the MXU never drains
# between blocks. Two row-state parities still suffice: block k's pending PV
# writes block k's new parity before block k + 1 reads it, and block k's
# replay runs inside block k's copy; a replay of the last head redoes the
# pending fragment, so it cancels the pending PV. Under defers_row_finish
# every copy carries its last PV out and a row's output finish runs after
# the next row's first block, so rows alternate parity to keep the finished
# row's state intact until then.
def pending_fragment(traits: FwdTraits) -> Fragment:
    """A block's last fragment, whose PV a copy leaves pending."""
    return Fragment(
        g=traits.head_fold - 1,
        kv_tile=traits.num_k_compute_tiles - 1,
        q_tile=traits.num_q_compute_tiles - 1,
    )


def issue_pending_pv(
    traits: FwdTraits, state: RowStateRefs, pending: PendingPVRefs
) -> None:
    """Run the pending PV; an empty one reads and writes the trash o slot."""
    pending_g, _, pending_q = pending_fragment(traits)
    fields = pending.fields
    pv(
        traits,
        state,
        pending_g,
        pending_q,
        pending.probs[...],
        pending.v[...],
        fields[PENDING_ROW_START] > 0,
        None,
        write_parity=fields[PENDING_WRITE_PARITY],
        read_parity=fields[PENDING_READ_PARITY],
    )


def cancel_pending_pv(pending: PendingPVRefs) -> None:
    """Empty the pending PV: point both of its parities at the trash slot."""
    pending.fields[PENDING_WRITE_PARITY] = TRASH_PARITY
    pending.fields[PENDING_READ_PARITY] = TRASH_PARITY


def carry_out_pv(
    traits: FwdTraits,
    state: RowStateRefs,
    pending: PendingPVRefs,
    kv_refs: KVRefs | PagedKVRefs,
    block: KVBlock,
    frag: Fragment,
    frag_pv: FragmentPV,
    is_tile_zero: bool | None,
) -> None:
    """Leave the PV of block's last fragment frag to the next block: frag has
    pending_fragment's head and q tile, and its kv tile is the block's last
    that ran. is_tile_zero as in SoftmaxPass."""
    pending_g, _, pending_q = pending_fragment(traits)
    assert (frag.g, frag.q_tile) == (pending_g, pending_q), frag
    # Only the ordinary pass carries its last PV out, so it has no rescale.
    assert frag_pv.rescale is None, frag_pv.rescale
    kv_compute_index = frag.kv_tile
    fields = pending.fields
    new_parity, old_parity = state.block_parities()
    pending.probs[...] = frag_pv.probs
    pending.v[...] = kv_refs.v_tile(
        traits, pending_g, kv_compute_index, block.kv_slot
    )
    fields[PENDING_WRITE_PARITY] = new_parity
    fields[PENDING_READ_PARITY] = tile_parity(
        kv_compute_index, is_tile_zero, new_parity, old_parity
    )
    fields[PENDING_ROW_START] = jnp.asarray(frag_pv.is_row_start, jnp.int32)


def stage_bounds(
    traits: FwdTraits,
    schedule: FwdSchedule,
    bounds_buf: jax.Array,
    ctx: ScheduleRow,
    kv: jax.Array,
    kv_tok: jax.Array,
) -> None:
    """Per-seq: write the bounds rows of kv block kv, whose first token is
    kv_tok, into bounds_buf, one slot per kv compute tile."""
    bkv, bkv_compute = traits.bkv, traits.bkv_compute
    # Note (david): synthesize the bounds rows once for the whole (1,
    # bkv) kv window; evaluated per fragment they spilled vector
    # registers the MXU then waited on (+13.4% on a 2048/256 prefill).
    k_ids = kv_tok + lax.broadcasted_iota(jnp.int32, (1, bkv), 1)
    q_lower, q_span = schedule.q_bounds(ctx, kv, k_ids)
    for tile in range(traits.num_k_compute_tiles):
        tile_lanes = slice(tile * bkv_compute, (tile + 1) * bkv_compute)
        bounds_buf[tile, 0:1, :] = q_lower[:, tile_lanes]
        if schedule.has_q_span:
            bounds_buf[tile, 1:2, :] = q_span[:, tile_lanes]


@dataclasses.dataclass(frozen=True, eq=False)
class OutRefs:
    """The output streams: the o operand and its staging ring, the lse
    operand and its stage when the variant returns lse (else None), and the
    kernel's DMA semaphores."""

    o_hbm: jax.Array
    o_stage: jax.Array
    lse_hbm: jax.Array | None
    lse_stage: jax.Array | None
    sems: jax.Array


def o_copy(
    traits: FwdTraits,
    out_refs: OutRefs,
    head: jax.Array | int,
    tok: jax.Array | int,
    slot: jax.Array | int,
):
    """The DMA of output staging slot slot to the o block at (head, tok)."""
    return pltpu.make_async_copy(
        out_refs.o_stage.at[slot],
        hbm_window(
            out_refs.o_hbm,
            head,
            tok,
            traits.bq,
            width=traits.width_v,
            heads_per_row=traits.q_heads_per_row,
            layout=traits.o_layout,
            token_major=traits.token_major,
            fold=traits.group_heads,
        ),
        out_refs.sems.at[SEM_OUT, slot],
    )


def blend_stage_rows(
    traits: FwdTraits, stage_ref: jax.Array, finish: RowFinish
) -> None:
    """Per-seq: copy the rows of the row's stage slot that the previous
    output block owns from that block's slot."""
    num_stages = traits.num_stages
    # Note (david): a per-seq window sits on the hbm_bf16_tile_rows() grid
    # and may reach below blend_start into rows the previous output block
    # computed correctly. That stage is still resident (num_stages >= 2 and
    # only its outbound DMA reads it), so the rows are copied over from it.
    # The bf16 stage is accessed in whole 16-row tiles, but the two windows
    # can be only 8-aligned apart, so each own tile reads the two source
    # tiles it straddles and picks their 8-row halves in f32, where 8 rows
    # are one whole (8, 128) tile.
    out_slot = lax.rem(finish.out_count, num_stages)
    num_blend_rows = finish.num_blend_rows

    # Note (david): rows below a head group's first block belong to no
    # sequence, and the previous output block is the last group's, so the
    # first block never blends.
    @pl.when(jnp.logical_and(num_blend_rows > 0, finish.si > 0))
    def _blend_head_rows():
        prev_slot = lax.rem(finish.out_count + num_stages - 1, num_stages)
        src_row_offset = finish.q_tok - finish.prev_out_tok
        src_offset_in_tile = src_row_offset & (BF16_TILE_ROWS - 1)
        half_rows = BF16_TILE_ROWS // 2

        def _tile_index(slot, row_start):
            return (
                (slot,)
                + (slice(None),) * (len(stage_ref.shape) - 3)
                + (
                    pl.ds(
                        pl.multiple_of(row_start, BF16_TILE_ROWS),
                        BF16_TILE_ROWS,
                    ),
                    slice(None),
                )
            )

        def _blend_tile(tile, carry):
            own_index = _tile_index(out_slot, tile * BF16_TILE_ROWS)
            own_rows = stage_ref[own_index].astype(jnp.float32)
            src_lower_start = (
                src_row_offset - src_offset_in_tile + tile * BF16_TILE_ROWS
            )
            # Note (david): the upper source tile is used only when
            # src_offset_in_tile is 8 and this tile's upper half blends, and
            # then it lies inside the source stage; the clamp keeps the
            # unused read in bounds otherwise.
            src_upper_start = jnp.minimum(
                src_lower_start + BF16_TILE_ROWS,
                stage_ref.shape[-2] - BF16_TILE_ROWS,
            )
            src_lower = stage_ref[
                _tile_index(prev_slot, src_lower_start)
            ].astype(jnp.float32)
            src_upper = stage_ref[
                _tile_index(prev_slot, src_upper_start)
            ].astype(jnp.float32)
            half_ids = lax.broadcasted_iota(
                jnp.int32,
                src_lower[..., :half_rows, :].shape,
                src_lower.ndim - 2,
            )
            is_src_on_tile = half_ids + src_offset_in_tile < half_rows
            prev_rows = jnp.concatenate(
                [
                    jnp.where(
                        is_src_on_tile,
                        src_lower[..., :half_rows, :],
                        src_lower[..., half_rows:, :],
                    ),
                    jnp.where(
                        is_src_on_tile,
                        src_lower[..., half_rows:, :],
                        src_upper[..., :half_rows, :],
                    ),
                ],
                axis=-2,
            )
            row_ids = tile * BF16_TILE_ROWS + lax.broadcasted_iota(
                jnp.int32, own_rows.shape, own_rows.ndim - 2
            )
            stage_ref[own_index] = jnp.where(
                row_ids < num_blend_rows, prev_rows, own_rows
            ).astype(stage_ref.dtype)
            return carry

        lax.fori_loop(
            0,
            (num_blend_rows + BF16_TILE_ROWS - 1) // BF16_TILE_ROWS,
            _blend_tile,
            None,
        )


def fill_packed_padding(
    traits: FwdTraits,
    packed_q_end: jax.Array,
    stage_ref: jax.Array,
    finish: RowFinish,
    fill_value: float,
) -> None:
    """Per-seq: write fill_value into the rows of the row's stage slot at or
    past packed_q_end, cu_seqlens_q[-1]."""
    num_stages = traits.num_stages

    # Note (david): rows at or past cu_seqlens_q[-1] belong to no sequence
    # and read out = 0 (lse = -inf). A window reaching them fills them in
    # its stage, which covers the partial tile at the packed end; the fill
    # after the loop covers the whole tiles past it.
    @pl.when(finish.q_tok + traits.bq > packed_q_end)
    def _fill_rows():
        stage_slot = stage_ref.at[lax.rem(finish.out_count, num_stages)]
        staged = stage_slot[...]
        rows = finish.q_tok + lax.broadcasted_iota(
            jnp.int32, staged.shape, staged.ndim - 2
        )
        stage_slot[...] = jnp.where(
            rows < packed_q_end,
            staged,
            jnp.asarray(fill_value, staged.dtype),
        )


def finish_row_out(
    traits: FwdTraits,
    variant: Variant,
    operands: VariantOperands,
    state: RowStateRefs,
    out_refs: OutRefs,
    finish: RowFinish,
    packed_q_end: jax.Array | None,
) -> None:
    """Normalize the row's output into its output stage slot and start its
    write; its last PV has run, and its final o and row_sum sit at
    finish.parity. packed_q_end is a per-seq build's cu_seqlens_q[-1], None
    on a dense one."""
    num_stages = traits.num_stages
    group_bq, frag_rows = traits.group_bq, traits.frag_rows
    bq_compute, gqa_fold = traits.bq_compute, traits.gqa_fold
    transposed_pv = traits.transposed_pv
    lane_width_v = traits.lane_width_v
    o_scratch, o_stage = state.o, out_refs.o_stage
    out_slot = lax.rem(finish.out_count, num_stages)

    # Note (david): o_stage is one output ring shared by the whole fold
    # group, so wait out the slot's previous DMA before any head writes into
    # it.
    @pl.when(finish.out_count >= num_stages)
    def _wait_prev_out():
        o_copy(traits, out_refs, 0, 0, out_slot).wait()

    def _finalize_head(g):
        head_rows = pl.ds(g * group_bq, group_bq)
        row_sum_ref = state.row_sum.at[finish.parity, head_rows]
        # Note (david): o_stage keeps an explicit fold axis (leading on
        # head-major, lanes on token-major) while o_scratch stays flat-row
        # addressed, so only o_stage goes through fold_lane. A gqa fold
        # writes all of its heads at once.
        o_ref = (
            o_scratch.at[finish.parity, :, head_rows]
            if transposed_pv
            else o_scratch.at[finish.parity, head_rows]
        )
        if gqa_fold > 1:
            o_stage_g, o_half = o_stage.at[out_slot], None
        else:
            o_stage_g, o_half = fold_lane(
                traits,
                o_stage.at[out_slot],
                g,
                traits.bq,
                lane_width_v,
                traits.o_layout,
            )

        def _finalize_fragment(q_compute_index, carry):
            slice_q = pl.ds(q_compute_index * frag_rows, frag_rows)
            row_sum = row_sum_ref[slice_q, :]
            if transposed_pv:
                tile_index = (slice(None), slice_q)
                inverse_row_sum = variant.output_scale(
                    row_sum[:, 0:1], operands
                ).T
            else:
                tile_index = (slice_q, slice(None))
                inverse_row_sum = jnp.tile(
                    variant.output_scale(row_sum, operands),
                    (1, traits.head_dim_v_repeats),
                )[..., : o_scratch.shape[-1]]
            o_normalized = (o_ref[tile_index] * inverse_row_sum).astype(
                o_stage.dtype
            )
            if gqa_fold > 1:
                o_stage_g[
                    :, pl.ds(q_compute_index * bq_compute, bq_compute), :
                ] = o_normalized.reshape(
                    gqa_fold, bq_compute, o_normalized.shape[-1]
                )
            elif o_half is None:
                o_stage_g[tile_index] = o_normalized
            else:
                # Note (david): the stage tile holds both heads of the pair,
                # and the head loop is sequential, so the other half is
                # either untouched or already this step's final value.
                o_stage_g[tile_index] = set_half(
                    o_stage_g[tile_index],
                    o_half,
                    lane_width_v,
                    o_normalized,
                )
            return carry

        lax.fori_loop(
            0,
            traits.num_q_compute_tiles,
            _finalize_fragment,
            None,
            unroll=True,
        )

    for g in range(traits.head_fold):
        _finalize_head(g)

    if traits.is_per_seq:
        blend_stage_rows(traits, o_stage, finish)
        fill_packed_padding(traits, packed_q_end, o_stage, finish, 0.0)
    # Note (david): the output copy starts only after the boundary blend has
    # repaired the rows shared with the previous block.
    o_copy(traits, out_refs, finish.base_head, finish.q_tok, out_slot).start()


def head_sinks(
    traits: FwdTraits,
    sinks_ref: jax.Array,
    heads_outer_batch: int | None,
    base_head: jax.Array,
) -> list[jax.Array]:
    """The sink of each pipeline head of the group at base_head."""
    gqa_fold = traits.gqa_fold

    # Note (deepolar): the sinks hold one logit per sequence head: flat q
    # head f reads sinks[f % nheads], or sinks[f // B] heads-outer. Pipeline
    # head g is flat q head base_head + g * gqa_fold, and the rows of a q
    # tile run (folded head, token), so the tile below, built once per head,
    # gives each folded head's rows its own sink.
    def sink_of(flat_head):
        if heads_outer_batch is None:
            return sinks_ref[lax.rem(flat_head, sinks_ref.shape[0])]
        return sinks_ref[flat_head // heads_outer_batch]

    def head_sink(g):
        first_head = base_head + g * gqa_fold
        if gqa_fold == 1:
            return sink_of(first_head)
        return jnp.concatenate([
            jnp.full(
                (traits.bq_compute, NUM_LANES),
                sink_of(first_head + h),
                jnp.float32,
            )
            for h in range(gqa_fold)
        ])

    return [head_sink(g) for g in range(traits.head_fold)]


def fold_out_sinks(
    traits: FwdTraits,
    variant: Variant,
    state: RowStateRefs,
    finish: RowFinish,
    sinks: list[jax.Array],
) -> None:
    """Fold the sinks into the row's final row_sum at finish.parity, which
    finish_row_out divides by; finish_row_lse reads it sink-free first."""
    group_bq, frag_rows = traits.group_bq, traits.frag_rows
    # Note (david): the row state stays sink-free while the row runs, as FA4
    # folds into its register row state only at the end. A deferred output
    # finish runs after the next row's first block has re-anchored
    # row_max_scratch, so the fold that needs this row's row_max runs at its
    # row end. The slot takes 1 / out_scale: an overflowed sink term gives
    # inf, the output's limit 0, and so does a row with no key.
    for g in range(traits.head_fold):
        head_rows = pl.ds(g * group_bq, group_bq)
        row_sum_ref = state.row_sum.at[finish.parity, head_rows]
        for q_compute_index in range(traits.num_q_compute_tiles):
            tile_rows = pl.ds(q_compute_index * frag_rows, frag_rows)
            _, _, out_scale = variant.fold_sink(
                state.row_max.at[head_rows][tile_rows, :],
                row_sum_ref[tile_rows, :],
                sinks[g],
            )
            row_sum_ref[tile_rows, :] = 1.0 / out_scale


def finish_row_lse(
    traits: FwdTraits,
    variant: Variant,
    state: RowStateRefs,
    out_refs: OutRefs,
    group: jax.Array,
    finish: RowFinish,
    sinks: list[jax.Array] | None,
    packed_q_end: jax.Array | None,
) -> None:
    """Stage and write the lse of head group group's row; its last softmax
    has run, and its final row_sum sits at finish.parity. sinks is
    head_sinks's list, or None; packed_q_end as in finish_row_out."""
    bq, bq_compute = traits.bq, traits.bq_compute
    head_fold, gqa_fold = traits.head_fold, traits.gqa_fold
    group_bq, frag_rows = traits.group_bq, traits.frag_rows
    is_per_seq = traits.is_per_seq
    row_max_scratch = state.row_max
    lse_hbm, lse_stage = out_refs.lse_hbm, out_refs.lse_stage
    out_slot = lax.rem(finish.out_count, traits.num_stages)
    for g in range(head_fold):
        head_rows = pl.ds(g * group_bq, group_bq)
        row_sum_ref = state.row_sum.at[finish.parity, head_rows]
        sink = None if sinks is None else sinks[g]
        if gqa_fold > 1:
            # Note (david): every lane of a row state holds the row's value,
            # and the rows of a q tile run (folded head, token), so each
            # head's token rows reach its own lse lane through one lane
            # select per head.
            lane_ids = lax.broadcasted_iota(
                jnp.int32, (bq_compute, NUM_LANES), 1
            )
            for q_compute_index in range(traits.num_q_compute_tiles):
                tile_rows = pl.ds(q_compute_index * frag_rows, frag_rows)
                row_max = row_max_scratch[tile_rows, :]
                row_sum = row_sum_ref[tile_rows, :]
                tile_lse = variant.lse(row_max, row_sum, sink).reshape(
                    gqa_fold, bq_compute, NUM_LANES
                )
                head_lanes = tile_lse[0]
                for h in range(1, gqa_fold):
                    head_lanes = jnp.where(
                        lane_ids == h, tile_lse[h], head_lanes
                    )
                lse_stage[
                    out_slot,
                    pl.ds(q_compute_index * bq_compute, bq_compute),
                    :,
                ] = head_lanes
        else:
            row_max = row_max_scratch.at[head_rows][:, 0:1]
            row_sum = row_sum_ref[:, 0:1]
            lse_row = variant.lse(row_max, row_sum, sink)
            if is_per_seq:
                lse_stage[out_slot, :, g] = lse_row[:, 0]
            else:
                lse_stage[g, :] = lse_row[:, 0]

    if is_per_seq:
        blend_stage_rows(traits, lse_stage, finish)
        fill_packed_padding(traits, packed_q_end, lse_stage, finish, -jnp.inf)
        # Note (david): the head group is a leading untiled axis and its
        # 128-lane minor axis is copied whole, avoiding the rejected dynamic
        # minor-axis slice at the head offset.
        lse_copies = [(
            lse_stage.at[out_slot],
            lse_hbm.at[group, pl.ds(finish.q_tok, bq), :],
        )]
    elif head_fold % NUM_SUBLANES == 0:
        # Note (david): HBM's leading f32 tile is 8 rows, so an 8-aligned
        # fold is one group-wide copy; a smaller fold copies row by row,
        # since later groups start at h = 2 or 4 and would break tile
        # alignment.
        lse_copies = [(
            lse_stage,
            lse_hbm.at[
                pl.ds(finish.base_head, head_fold), pl.ds(finish.q_tok, bq)
            ],
        )]
    else:
        lse_copies = [
            (
                lse_stage.at[pl.ds(local_head, 1), :],
                lse_hbm.at[
                    pl.ds(finish.base_head + local_head, 1),
                    pl.ds(finish.q_tok, bq),
                ],
            )
            for local_head in range(head_fold)
        ]
    for lse_src, lse_dst in lse_copies:
        lse_copy = pltpu.make_async_copy(
            lse_src, lse_dst, out_refs.sems.at[SEM_LSE, 0]
        )
        lse_copy.start()
        lse_copy.wait()


def fill_packed_tail(
    traits: FwdTraits,
    variant: Variant,
    out_refs: OutRefs,
    packed_q_end: jax.Array,
    num_head_groups: int,
) -> None:
    """Per-seq: once every output DMA has landed, write out = 0 (lse = -inf)
    into the packed padding rows past packed_q_end's HBM tile, for every
    head group."""
    bq = traits.bq
    token_major, group_heads = traits.token_major, traits.group_heads
    o_hbm, o_stage = out_refs.o_hbm, out_refs.o_stage
    lse_hbm, lse_stage = out_refs.lse_hbm, out_refs.lse_stage
    return_lse = variant.return_lse
    # Note (david): packed padding rows own no q block, so once every output
    # DMA has landed they are written from a cleared stage. The last block
    # already zeroed its window's padding rows, so the fill starts at the
    # next HBM tile; with no block at all, cu_seqlens_q is all 0 and so is
    # that tile.
    padded_total_q = o_hbm.shape[-2]
    tile_rows = hbm_bf16_tile_rows()
    fill_start = (packed_q_end + tile_rows - 1) // tile_rows * tile_rows
    num_fills = pl.cdiv(jnp.maximum(padded_total_q - fill_start, 0), bq)

    @pl.when(num_fills > 0)
    def _clear_fill_stage():
        o_stage[0] = jnp.zeros(o_stage.shape[1:], o_stage.dtype)
        if return_lse:
            lse_stage[0] = jnp.full(lse_stage.shape[1:], -jnp.inf, jnp.float32)

    def _fill_rows(fill_index, carry):
        row_start = pl.multiple_of(fill_start + fill_index * bq, tile_rows)
        num_fill_rows = pl.multiple_of(
            jnp.minimum(padded_total_q - row_start, bq), tile_rows
        )
        fill_rows = pl.ds(0, num_fill_rows)
        # Note (david): only a head-major fold stages a leading fold axis; a
        # token-major fold rides the lanes.
        if token_major is None and group_heads > 1:
            o_fill = o_stage.at[0, :, fill_rows, :]
        else:
            o_fill = o_stage.at[0, fill_rows, :]
        for group in range(num_head_groups):
            o_fill_copy = (
                o_fill,
                hbm_window(
                    o_hbm,
                    group * group_heads,
                    row_start,
                    num_fill_rows,
                    width=traits.width_v,
                    heads_per_row=traits.q_heads_per_row,
                    layout=traits.o_layout,
                    token_major=token_major,
                    fold=group_heads,
                ),
                SEM_OUT,
            )
            if return_lse:
                fill_copies = [
                    o_fill_copy,
                    (
                        lse_stage.at[0, fill_rows, :],
                        lse_hbm.at[group, pl.ds(row_start, num_fill_rows), :],
                        SEM_LSE,
                    ),
                ]
            else:
                fill_copies = [o_fill_copy]
            for fill_src, fill_dst, sem_index in fill_copies:
                fill_copy = pltpu.make_async_copy(
                    fill_src, fill_dst, out_refs.sems.at[sem_index, 0]
                )
                fill_copy.start()
                fill_copy.wait()
        return carry

    lax.fori_loop(0, num_fills, _fill_rows, None)
