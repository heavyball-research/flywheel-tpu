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

from ..core.tpu import NUM_LANES
from .config import QKVLayout
from .dma import fold_row, hbm_window
from .rotary import rotate_tile
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
