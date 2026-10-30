"""Building blocks of the forward fragment pipeline.

A forward main loop (fwd_pipeline.fwd_body) runs these at fixed points: the
Q and rotary staging copies and their in-place rotation, the K and V staging
copies, the QK fragment and its masks, the online softmax, the PV with its
pending-PV hand-off across blocks, and a row's output and lse finish. Each
takes the build's FwdTraits (and Variant, where it calls a hook) plus a small
frozen group of the refs it reads, so a loop hands over its scratch rather
than a helper closing over it.
"""

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..core.tpu import NUM_LANES, NUM_SUBLANES
from .config import QKVLayout
from .dma import fold_row, hbm_window
from .kv_layout import (
    kv_heads_per_word,
    kv_word_halves,
    kv_word_rows,
    load_staged_words,
    staged_words,
)
from .rotary import rotate_tile
from .schedule import FwdSchedule
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
class QRefs:
    """The Q stream: the HBM operand, its staging ring, and under rotary the
    Q coefficient operand and its staging ring (else None), plus the kernel's
    DMA semaphores."""

    hbm: jax.Array
    buf: jax.Array
    rotary_hbm: jax.Array | None
    rotary_buf: jax.Array | None
    sems: jax.Array


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
    sems: jax.Array,
    coeff_hbm: jax.Array,
    coeff_buf: jax.Array,
    head: jax.Array | int,
    tok: jax.Array | int,
    block: int,
    slot: jax.Array | int,
    heads_per_batch: int,
    sem_plane: int,
):
    """The DMAs of the rotary coefficient rows [tok, tok + block) of every
    coefficient group of the group at head into staging slot slot."""
    for g in range(traits.rotary_groups):
        batch_idx = (
            0 if traits.rotary_batch == 1 else (head + g) // heads_per_batch
        )
        rows = pl.ds(tok, block)
        coeff_window = (
            coeff_hbm.at[batch_idx, rows, :]
            if traits.is_head_dim_minor
            else coeff_hbm.at[batch_idx, :, rows]
        )
        yield pltpu.make_async_copy(
            coeff_window, coeff_buf.at[slot, g], sems.at[sem_plane, slot]
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
            traits,
            q_refs.sems,
            q_refs.rotary_hbm,
            q_refs.rotary_buf,
            head,
            tok,
            traits.bq,
            slot,
            traits.rotary_q_heads,
            SEM_ROTARY_Q,
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
            traits,
            q_refs.sems,
            q_refs.rotary_hbm,
            q_refs.rotary_buf,
            0,
            0,
            traits.bq,
            slot,
            traits.rotary_q_heads,
            SEM_ROTARY_Q,
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
            traits,
            kv_refs.sems,
            kv_refs.rotary_hbm,
            kv_refs.rotary_buf,
            kv_head,
            kv_tok,
            traits.bkv,
            slot,
            traits.rotary_k_heads,
            SEM_ROTARY_K,
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
            traits,
            kv_refs.sems,
            kv_refs.rotary_hbm,
            kv_refs.rotary_buf,
            0,
            0,
            traits.bkv,
            slot,
            traits.rotary_k_heads,
            SEM_ROTARY_K,
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
