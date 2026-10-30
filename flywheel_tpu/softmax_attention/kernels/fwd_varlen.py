"""Varlen forward kernel driven by a schedule table built from the cu_seqlens.

The per-seq schedule cuts each sequence into its own q blocks, one SMEM table
row each, and synthesizes its masks in-kernel: the sequence bounds plus the
causal or window edges around each sequence's own bottom-right diagonal.
"""

import dataclasses
from collections.abc import Sequence
from functools import partial, reduce

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from ..config import hbm_bf16_tile_rows
from ..dma import advance, drain
from ..fwd_blocks import (
    NUM_KV_STAGES,
    PENDING_WRITE_PARITY,
    TRASH_PARITY,
    KVBlock,
    KVRefs,
    OutRefs,
    PendingPVRefs,
    QRefs,
    RowFinish,
    RowStateRefs,
    ScoreMask,
    cancel_pending_pv,
    carry_out_pv,
    fill_packed_tail,
    finish_row_lse,
    finish_row_out,
    flag_overflow_risk,
    fold_out_sinks,
    fragment_pv,
    head_sinks,
    issue_pending_pv,
    o_copy,
    online_softmax,
    pending_fragment,
    qk,
    rotate_staged,
    stage_bounds,
    start_kv_block,
    start_q_row,
    wait_kv_block,
    wait_q_row,
)
from ..fwd_call import rotary_fields
from ..fwd_pipeline import forward_common
from ..schedule import (
    FwdSchedule,
    ScheduleTableLayout,
    VarlenSchedule,
    make_per_seq_fwd_schedule,
    varlen_schedule_table,
)
from ..seqlen import check_cu_seqlens_pair
from ..traits import FwdTraits
from ..variant import Variant, VariantOperands


def flash_fwd_varlen_kernel(
    table_ref: jax.Array,
    *refs: jax.Array,
    traits: FwdTraits,
    variant: Variant,
    varlen_sched: VarlenSchedule,
    layout: ScheduleTableLayout,
) -> None:
    if variant.has_sinks:
        sinks_ref, q_hbm, k_hbm, v_hbm, *refs = refs
    else:
        sinks_ref = None
        q_hbm, k_hbm, v_hbm, *refs = refs
    left, right = variant.window
    schedule = make_per_seq_fwd_schedule(
        table_ref,
        layout=layout,
        num_head_groups=traits.num_head_groups,
        q_heads_per_kv_head=traits.q_heads_per_kv_head,
        padded_total_k=varlen_sched.padded_total_k,
        bq=traits.bq,
        bkv=traits.bkv,
        bkv_compute=traits.bkv_compute,
        left=left,
        right=right,
    )
    # The schedule masks each sequence around its own diagonal, so the loop
    # runs without the Variant's static mask.
    varlen_fwd_body(
        traits,
        variant.without_mask(),
        schedule,
        q_hbm,
        k_hbm,
        v_hbm,
        refs,
        operands=VariantOperands(sinks_ref=sinks_ref),
    )


def varlen_fwd_body(
    traits: FwdTraits,
    variant: Variant,
    schedule: FwdSchedule,
    q_hbm: jax.Array,
    k_hbm: jax.Array,
    v_hbm: jax.Array,
    refs: Sequence[jax.Array],
    *,
    operands: VariantOperands,
) -> None:
    """Run the varlen fragment pipeline over every block the per-seq schedule
    yields.

    traits is the build's static configuration, variant its elementwise math
    and operands the variant's traced values, and refs are forward_common's
    rotary operands, outputs and scratch, in its allocation order. The logits
    run in log2 units: Q reaches the QK matmul pre-multiplied by log2(e). K
    and V are separate packed operands, head-major or token-major. The
    per-seq schedule writes the packed padding rows past
    schedule.packed_q_end as out = 0, lse = -inf, and applies the mask
    itself, so its variant has none.

    operands.sinks_ref is None or an SMEM (nheads,) float32 of per-head sink
    logits in log2 units, one per sequence head (flat q head f reads
    f % nheads), each added to its head's softmax denominator.
    """
    assert traits.is_per_seq and not traits.is_paged, traits
    assert schedule.q_token is not None and schedule.kv_page is None
    assert schedule.num_live_tiles is not None
    assert schedule.heads_outer_batch is None, schedule.heads_outer_batch
    # The schedule masks each sequence around its own diagonal through the
    # bounds rows, so no fragment applies a static mask (needs_mask False).
    assert variant.window == (None, None), variant.window
    # Every row finishes inside the next row's first block.
    assert traits.defers_row_finish, traits
    bkv = traits.bkv
    num_stages = traits.num_stages
    num_k_compute_tiles = traits.num_k_compute_tiles
    num_q_compute_tiles = traits.num_q_compute_tiles
    num_compute_tiles = traits.num_compute_tiles
    head_fold = traits.head_fold
    group_heads, group_bq = traits.group_heads, traits.group_bq
    is_lane_paired = traits.is_lane_paired
    rotary_dim, rotary_k = traits.rotary_dim, traits.rotary_k
    variant.check_operands(operands)
    return_lse = variant.return_lse
    sinks_ref = operands.sinks_ref

    remaining_refs = list(refs)
    if rotary_dim:
        rotary_q_hbm = remaining_refs.pop(0)
        rotary_k_hbm = remaining_refs.pop(0) if rotary_k else None
    else:
        rotary_q_hbm = rotary_k_hbm = None
    o_hbm = remaining_refs.pop(0)
    lse_hbm = remaining_refs.pop(0) if return_lse else None
    q_buf = remaining_refs.pop(0)
    k_buf, v_buf = remaining_refs.pop(0), remaining_refs.pop(0)
    bounds_buf = remaining_refs.pop(0)
    if rotary_dim:
        rotary_q_buf = remaining_refs.pop(0)
        rotary_k_buf = remaining_refs.pop(0) if rotary_k else None
    else:
        rotary_q_buf = rotary_k_buf = None
    if return_lse:
        *pipeline_refs, lse_stage = remaining_refs
    else:
        pipeline_refs, lse_stage = remaining_refs, None
    # Note (david): the exact unpack pins this variable-arity tail against
    # forward_common's scratch list; a silent mismatch would shift every ref.
    (
        o_stage,
        row_max_scratch,
        row_sum_scratch,
        o_scratch,
        pending_probs,
        pending_v,
        guard_smem,
        parity_smem,
        pending_smem,
        sems,
    ) = pipeline_refs
    state = RowStateRefs(
        row_max=row_max_scratch,
        row_sum=row_sum_scratch,
        o=o_scratch,
        parity=parity_smem,
        guard=guard_smem,
    )
    pending = PendingPVRefs(
        probs=pending_probs, v=pending_v, fields=pending_smem
    )
    out_refs = OutRefs(
        o_hbm=o_hbm,
        o_stage=o_stage,
        lse_hbm=lse_hbm,
        lse_stage=lse_stage,
        sems=sems,
    )
    q_refs = QRefs(
        hbm=q_hbm,
        buf=q_buf,
        rotary_hbm=rotary_q_hbm,
        rotary_buf=rotary_q_buf,
        sems=sems,
    )
    kv_refs = KVRefs(
        k_hbm=k_hbm,
        v_hbm=v_hbm,
        k_buf=k_buf,
        v_buf=v_buf,
        rotary_hbm=rotary_k_hbm,
        rotary_buf=rotary_k_buf,
        sems=sems,
    )
    if return_lse:
        lse_stage[...] = jnp.zeros(lse_stage.shape, jnp.float32)

    num_head_groups = schedule.num_head_groups
    num_rows = schedule.num_rows
    q_heads_per_kv_head = schedule.q_heads_per_kv_head
    # Note (david): only a left window can mask a row's whole first fragment
    # while later keys stay visible; anchoring on that masked score would push
    # the row's first visible key past the guard into a replay.
    defers_row_start = schedule.has_q_span
    mask = ScoreMask(
        causal_offset=variant.mask_offset(operands),
        bounds=bounds_buf,
        has_q_span=schedule.has_q_span,
        defers_row_start=defers_row_start,
    )

    # Note (david): the kv stream prefetches two blocks ahead in a
    # NUM_KV_STAGES ring. Mosaic's scheduler sinks a DMA start towards its
    # wait, so a start and wait in one basic block leave the DMA no time to
    # land, and a wait inside a copy splits the copy into two basic blocks with
    # their own pipeline fill and drain (v6e, 512 x 512: 1574 + 1553 cycles
    # against 2330 for the whole copy). Blocks j and j + 1 hold at most two
    # slots, so the slot block j + 2 takes is free for its start anywhere in
    # block j's step.
    def _block_desc(head, kv, ctx):
        """The DMA inputs of kv block kv of head's group in the row of ctx."""
        # Note (david): the heads-outer fold flattens (heads, batch) with
        # B = heads_outer_batch, so q index n * B + b reads kv index
        # (n // q_heads_per_kv_head) * B + b; the batch-outer fold flattens
        # (batch, heads), so consecutive q indices share one kv index.
        kv_head = head // q_heads_per_kv_head
        desc = (kv_head, schedule.kv_token(kv, ctx))
        # Note (david): the descriptors ride the loop carry, which needs strong
        # int32 scalars.
        return tuple(jnp.asarray(field, jnp.int32) for field in desc)

    def _select_desc(predicate, desc_true, desc_false):
        return tuple(
            jnp.where(predicate, field_true, field_false)
            for field_true, field_false in zip(desc_true, desc_false)
        )

    # Note (david): a block whose DMA inputs equal the previous block's holds
    # the same staged data, so it runs from the previous block's slot; dense GQA
    # at one kv block per row otherwise re-staged each block once per q head
    # (v6e, 32:4 d256, 1024 tokens: 63.4 against 60.6 us). Paged blocks start
    # their DMAs inside the copy and a rotary K is rotated in place, so both
    # load every block.
    reuses_blocks = not (rotary_dim and rotary_k)

    def _is_fresh(desc, prev_desc):
        if reuses_blocks:
            differs = [
                field != prev_field
                for field, prev_field in zip(desc, prev_desc)
            ]
            return reduce(jnp.logical_or, differs)
        else:
            return jnp.bool_(True)

    def _start_block_data(desc, slot):
        kv_head, kv_tok = desc
        start_kv_block(traits, kv_refs, kv_head, kv_tok, slot)

    def _wait_block_data(desc, slot):
        del desc
        wait_kv_block(traits, kv_refs, slot)

    pending_g, pending_k, _ = pending_fragment(traits)

    def _block_step(si, kv, lo, hi, ctx, row, ahead, carry):
        """One kv block of row si. ahead holds the descriptors of the next row's
        first two blocks in the flattened block order, and carry this block's
        and the next block's ring slot, freshness and descriptor."""
        q_tok, base_head, q_slot, row_parity, prev_row_finish = row
        (
            kv_slot,
            next_slot,
            is_fresh,
            is_next_fresh,
            block_desc,
            next_block_desc,
        ) = carry
        kv_tok = schedule.kv_token(kv, ctx)
        is_first = kv == lo
        is_partial, needs_bounds = schedule.block_flags(si, kv, ctx)
        num_live_tiles = schedule.num_live_tiles(kv, ctx)
        # Note (david): row_sum and o scratch ping-pong, so a block's first kv
        # tile reads the previous block's exit state while later tiles update in
        # place; each head keeps its own guard flag, so one head's overflow
        # never replays another. The parity is re-read from SMEM at every use:
        # held as a loop value it lengthened every copy's schedule by about 30%
        # (v6e, 512 x 512 paged: 3537 against 2872 bundles).
        parity_smem[0] = lax.rem(row_parity + kv - lo, 2)

        first_next_row_desc, second_ahead_desc = ahead
        ahead_desc = _select_desc(
            kv + 2 <= hi,
            _block_desc(base_head, kv + 2, ctx),
            _select_desc(kv + 1 == hi, first_next_row_desc, second_ahead_desc),
        )
        is_ahead_fresh = _is_fresh(ahead_desc, next_block_desc).astype(
            jnp.int32
        )
        ahead_slot = jnp.where(
            is_ahead_fresh > 0, lax.rem(next_slot + 1, NUM_KV_STAGES), next_slot
        )
        for g in range(head_fold):
            guard_smem[g] = 0

        @pl.when(is_fresh > 0)
        def _wait_fresh_block():
            _wait_block_data(block_desc, kv_slot)

        if reuses_blocks:

            @pl.when(is_ahead_fresh > 0)
            def _start_fresh_ahead():
                _start_block_data(ahead_desc, ahead_slot)

        block = KVBlock(
            q_slot=q_slot,
            q_tok=q_tok,
            kv_slot=kv_slot,
            kv_tok=kv_tok,
            is_first=is_first,
        )

        def _qk(g, kv_compute_index, q_compute_index):
            return qk(
                traits,
                variant,
                q_buf,
                kv_refs,
                block,
                g,
                kv_compute_index,
                q_compute_index,
            )

        def _online_softmax(
            g,
            kv_compute_index,
            q_compute_index,
            logits,
            *,
            last_kv_tile=num_k_compute_tiles - 1,
            **flags,
        ):
            """online_softmax of one of this block's fragments, which no
            static mask touches, flags its keyword flags."""
            return online_softmax(
                traits,
                variant,
                state,
                mask,
                block,
                g,
                kv_compute_index,
                q_compute_index,
                logits,
                False,
                last_kv_tile=last_kv_tile,
                **flags,
            )

        def _block_pv(*fragment):
            fragment_pv(traits, state, kv_refs, block, *fragment)

        def _carry_out_pv(
            probs,
            is_row_start,
            kv_compute_index=pending_k,
            is_tile_zero=num_k_compute_tiles == 1,
        ):
            carry_out_pv(
                traits,
                state,
                pending,
                kv_refs,
                block,
                probs,
                is_row_start,
                kv_compute_index,
                is_tile_zero,
            )

        def _run_schedule(
            guard_recovery,
            static_g,
            apply_bounds,
            *,
            skips_dead_tiles=False,
            takes_pending=False,
            carries_out=False,
        ):
            """takes_pending issues the previous block's pending PV after the
            first softmax, and carries_out leaves the last fragment's PV
            pending; a replay passes neither. skips_dead_tiles runs a
            bounds-masked block's live tiles alone."""

            def _after_first_softmax():
                if takes_pending:
                    issue_pending_pv(traits, state, pending)
                    if not carries_out:
                        cancel_pending_pv(pending)

            # Note (david): static_g None is the fast path, one distance-1
            # pipeline across every folded head so head g + 1's first QK
            # overlaps head g's last PV. An int static_g is the rare per-head
            # overflow replay, kept simple on purpose.
            heads = range(head_fold) if static_g is None else (static_g,)
            if skips_dead_tiles:
                # Note (david): the live tiles run in a loop whose body is the q
                # tiles of one kv tile, straight-line, with the pending PV in
                # the carry; a cond around every fragment instead ran 17.4
                # against 3.78 ms (v6e, 32:8 d128, paged (512, 2048, 128, 512)).
                # A live tile past a q tile's own frontier scores fully masked
                # fragments, which leave every anchored row's state as it was.
                # Tile 0 is peeled so the carry always holds a pending PV, and
                # with two tiles per block only tile 0 is live.
                last_q_tile = num_q_compute_tiles - 1
                has_tile_loop = num_k_compute_tiles > 2
                last_live_tile = num_live_tiles - 1 if has_tile_loop else 0

                def _tile_fragments(
                    g, kv_compute_index, is_tile_zero, pending_pv
                ):
                    for q_compute_index in range(num_q_compute_tiles):
                        logits = _qk(g, kv_compute_index, q_compute_index)
                        probs, is_row_start, rescale = _online_softmax(
                            g,
                            kv_compute_index,
                            q_compute_index,
                            logits,
                            guard_recovery=guard_recovery,
                            apply_bounds=True,
                            is_tile_zero=is_tile_zero,
                            last_kv_tile=None if has_tile_loop else 0,
                        )
                        if pending_pv is None:
                            _after_first_softmax()
                        else:
                            _block_pv(*pending_pv)
                        pending_pv = (
                            g,
                            kv_compute_index,
                            q_compute_index,
                            probs,
                            is_row_start,
                            rescale,
                            is_tile_zero,
                        )
                    return pending_pv

                pending_pv = None
                for g in heads:
                    pending_pv = _tile_fragments(g, 0, True, pending_pv)
                    if has_tile_loop:

                        def _tile_step(kv_compute_index, carry, g=g):
                            # Note (david): the carried PV is tile 0's on the
                            # first iteration, so its parity stays runtime.
                            carried_pv = (
                                g,
                                kv_compute_index - 1,
                                last_q_tile,
                                *carry,
                                None,
                            )
                            *_, probs, is_row_start, rescale, _ = (
                                _tile_fragments(
                                    g, kv_compute_index, False, carried_pv
                                )
                            )
                            return probs, is_row_start, rescale

                        carry = lax.fori_loop(
                            1,
                            num_live_tiles,
                            _tile_step,
                            tuple(pending_pv[3:6]),
                        )
                        pending_pv = (
                            g,
                            last_live_tile,
                            last_q_tile,
                            *carry,
                            None,
                        )
                if carries_out:
                    assert not guard_recovery
                    _, last_k, _, probs, is_row_start, _, is_tile_zero = (
                        pending_pv
                    )
                    _carry_out_pv(probs, is_row_start, last_k, is_tile_zero)
                else:
                    _block_pv(*pending_pv)
                if has_tile_loop and not guard_recovery:
                    # Note (david): once the loop ends each head's new-parity
                    # row sums are its last live tile's, so one check replaces a
                    # runtime tile compare in every fragment.
                    for g in heads:
                        new_parity, _ = state.block_parities()
                        flag_overflow_risk(
                            traits,
                            state,
                            g,
                            row_sum_scratch[
                                new_parity, pl.ds(g * group_bq, group_bq), :
                            ],
                            True,
                        )
            else:
                if static_g is None:
                    num_fragments = head_fold * num_compute_tiles

                    def _fragment_coords(frag_idx):
                        g = frag_idx // num_compute_tiles
                        tile_idx = frag_idx % num_compute_tiles
                        return (
                            g,
                            tile_idx // num_q_compute_tiles,
                            tile_idx % num_q_compute_tiles,
                        )

                else:
                    num_fragments = num_compute_tiles

                    def _fragment_coords(frag_idx):
                        return (
                            static_g,
                            frag_idx // num_q_compute_tiles,
                            frag_idx % num_q_compute_tiles,
                        )

                # Note (david): a pinned g replaces the loop-carried copy as a
                # Python int, because sub-tile (d64) lane windows need static
                # offsets; a static g costs nothing on the other builds.
                def _resolve_g(g):
                    return static_g if static_g is not None else g

                first_g, first_k, first_q = (
                    jnp.int32(coord) for coord in _fragment_coords(0)
                )
                logits = _qk(_resolve_g(first_g), first_k, first_q)
                probs_first, is_row_start_first, rescale_first = (
                    _online_softmax(
                        _resolve_g(first_g),
                        first_k,
                        first_q,
                        logits,
                        guard_recovery=guard_recovery,
                        apply_bounds=apply_bounds,
                    )
                )
                _after_first_softmax()

                def _pipeline_step(frag_idx, carry):
                    (
                        probs_prev,
                        is_row_start_prev,
                        rescale_prev,
                        prev_g,
                        prev_k,
                        prev_q,
                    ) = carry
                    current_g, current_k, current_q = _fragment_coords(frag_idx)
                    logits = _qk(_resolve_g(current_g), current_k, current_q)
                    probs_curr, is_row_start_curr, rescale_curr = (
                        _online_softmax(
                            _resolve_g(current_g),
                            current_k,
                            current_q,
                            logits,
                            guard_recovery=guard_recovery,
                            apply_bounds=apply_bounds,
                        )
                    )
                    _block_pv(
                        _resolve_g(prev_g),
                        prev_k,
                        prev_q,
                        probs_prev,
                        is_row_start_prev,
                        rescale_prev,
                    )
                    return (
                        probs_curr,
                        is_row_start_curr,
                        rescale_curr,
                        current_g,
                        current_k,
                        current_q,
                    )

                (
                    probs_last,
                    is_row_start_last,
                    rescale_last,
                    last_g,
                    last_k,
                    last_q,
                ) = lax.fori_loop(
                    1,
                    num_fragments,
                    _pipeline_step,
                    (
                        probs_first,
                        is_row_start_first,
                        rescale_first,
                        first_g,
                        first_k,
                        first_q,
                    ),
                    unroll=True,
                )
                if carries_out:
                    # Note (david): this copy runs every fragment, so its last
                    # fragment is the static pending one.
                    assert not guard_recovery
                    _carry_out_pv(probs_last, is_row_start_last)
                else:
                    _block_pv(
                        _resolve_g(last_g),
                        last_k,
                        last_q,
                        probs_last,
                        is_row_start_last,
                        rescale_last,
                    )

        def _run_group(apply_bounds, skips_dead_tiles=False):
            """Run one copy of the block's fragment pipeline, whose last PV
            stays pending for the next block (every row defers its finish),
            then replay each head the guard flagged."""
            # Note (david): the unpack (or K rotation) leads the copy in program
            # order, where the scheduler overlaps it with the pending PV, which
            # reads only scratch.
            if rotary_dim and rotary_k:
                rotate_staged(
                    traits, kv_refs.k_buf, kv_refs.rotary_buf, kv_slot, bkv
                )
            # Note (david): a path that loads every block starts the block two
            # ahead inside the copy, where the page DMAs' scalar work rides the
            # copy's idle scalar slots.
            if not reuses_blocks:
                _start_block_data(ahead_desc, ahead_slot)
            if is_lane_paired:
                # Note (david): one static-g pipeline per head, since the fused
                # pipeline's loop-carried g would make the 64-lane window
                # offsets dynamic; this gives up the overlap across head
                # boundaries, while the first head still takes the pending PV in
                # and the last carries out.
                for g in range(head_fold):
                    _run_schedule(
                        guard_recovery=False,
                        static_g=g,
                        apply_bounds=apply_bounds,
                        skips_dead_tiles=skips_dead_tiles,
                        takes_pending=g == 0,
                        carries_out=g == head_fold - 1,
                    )
            else:
                _run_schedule(
                    guard_recovery=False,
                    static_g=None,
                    apply_bounds=apply_bounds,
                    skips_dead_tiles=skips_dead_tiles,
                    takes_pending=True,
                    carries_out=True,
                )

            for g in range(head_fold):

                @pl.when(guard_smem[g] > 0)
                def _recover(g=g):
                    _run_schedule(
                        guard_recovery=True,
                        static_g=g,
                        apply_bounds=apply_bounds,
                        skips_dead_tiles=skips_dead_tiles,
                    )
                    if g == pending_g:
                        cancel_pending_pv(pending)

        # Note (david): a mask-free copy of the fragment pipeline and masked
        # ones, because the masks cost about as many VPU ops as the softmax
        # itself: one per partial-block offset under window_plans, else one
        # applying every mask the build has. One copy per (causal, bounds) pair
        # measured 2.6x slower at block 2048. A build carries either the static
        # window (dense) or the per-seq bounds (varlen), never both, and
        # has_static_mask still gates the former so a full-attention build never
        # loses its upper triangle.
        needs_any_mask = jnp.logical_or(is_partial, needs_bounds)

        @pl.when(jnp.logical_not(needs_any_mask))
        def _run_unmasked_block():
            _run_group(apply_bounds=False)

        # Note (david): a per-seq q block's kv frontier is a runtime scalar, so
        # a masked block runs only its kv compute tiles below it, in a tile
        # loop that indexes the tile at runtime; only a sublane token axis
        # allows that.
        # Note (david): each tile loop step ends a basic block the fragment
        # pipeline cannot overlap across, so only a block with dead tiles takes
        # the loop: running every masked block through it slowed the packed
        # varlen kernel by 7% (v6e, 32:8 d128: 7.35 against 6.85 ms).
        if num_k_compute_tiles > 1:
            skips_dead_tiles = num_live_tiles < num_k_compute_tiles
            runs_every_tile = jnp.logical_and(
                needs_any_mask, jnp.logical_not(skips_dead_tiles)
            )

            @pl.when(jnp.logical_and(needs_any_mask, skips_dead_tiles))
            def _run_live_tile_block():
                stage_bounds(traits, schedule, bounds_buf, ctx, kv, kv_tok)
                _run_group(apply_bounds=True, skips_dead_tiles=True)

        else:
            runs_every_tile = needs_any_mask

        @pl.when(runs_every_tile)
        def _run_masked_block():
            stage_bounds(traits, schedule, bounds_buf, ctx, kv, kv_tok)
            _run_group(apply_bounds=True)

        # Note (david): the previous row's last PV ran inside this block's
        # copy, and its final parity stays untouched until this row's second
        # block.
        @pl.when(jnp.logical_and(is_first, prev_row_finish.is_live > 0))
        def _finish_previous_row():
            _finish_row_out(prev_row_finish)

        return (
            next_slot,
            ahead_slot,
            is_next_fresh,
            is_ahead_fresh,
            next_block_desc,
            ahead_desc,
        )

    def _finish_row_out(finish):
        finish_row_out(
            traits,
            variant,
            operands,
            state,
            out_refs,
            finish,
            schedule.packed_q_end,
        )

    def _row_after(g, si):
        """The (group, row) after row si of group g; the grid's last row is its
        own successor."""
        is_row_end = si == num_rows - 1
        is_grid_end = jnp.logical_and(g == num_head_groups - 1, is_row_end)
        next_g = jnp.where(
            is_row_end, jnp.minimum(g + 1, num_head_groups - 1), g
        )
        next_si = jnp.where(is_grid_end, si, jnp.where(is_row_end, 0, si + 1))
        return next_g, next_si

    def _first_two_blocks(g, si):
        """The descriptors of the first two blocks of the flattened order from
        row si of group g on, and that row's (lo, hi, ctx)."""
        lo, hi, ctx = schedule.row_interval(si)
        after_g, after_si = _row_after(g, si)
        after_lo, _, after_ctx = schedule.row_interval(after_si)
        first_desc = _block_desc(g * group_heads, lo, ctx)
        second_desc = _select_desc(
            lo < hi,
            _block_desc(g * group_heads, lo + 1, ctx),
            _block_desc(after_g * group_heads, after_lo, after_ctx),
        )
        return first_desc, second_desc, (lo, hi, ctx)

    def _carried_finish(row_state):
        # Note (david): a loop carry drops q_token's pl.multiple_of, and the
        # output DMA window must start on the hbm_bf16_tile_rows() grid, where
        # q_token put it.
        row_parity, finish = row_state
        return row_parity, finish._replace(
            q_tok=pl.multiple_of(finish.q_tok, hbm_bf16_tile_rows())
        )

    def _row_body(g, si, carry):
        """carry ends with the row's first block's new parity and the previous
        row's RowFinish."""
        (
            q_loaded,
            q_inflight,
            q_slot,
            kv_carry,
            prev_out_tok,
            num_out_copies,
            row_state,
        ) = carry
        row_parity, prev_row_finish = _carried_finish(row_state)
        lo, hi, ctx = schedule.row_interval(si)
        # Note (david): past the grid's last row the lookahead is that row
        # itself, so the q stream prefetches nothing, and the kv stream reloads
        # blocks the kernel waits out after its loop.
        next_g, next_si = _row_after(g, si)
        first_next_desc, second_next_desc, (_, _, next_ctx) = _first_two_blocks(
            next_g, next_si
        )
        base_head = g * group_heads
        next_head = next_g * group_heads
        q_tok = schedule.q_token(ctx)
        next_q_tok = schedule.q_token(next_ctx)

        q_loaded, q_inflight, q_slot = advance(
            g * num_rows + si,
            next_g * num_rows + next_si,
            lambda slot: start_q_row(traits, q_refs, base_head, q_tok, slot),
            lambda slot: start_q_row(
                traits, q_refs, next_head, next_q_tok, slot
            ),
            partial(wait_q_row, traits, variant, operands, q_refs),
            q_loaded,
            q_inflight,
            q_slot,
            num_stages,
        )
        kv_carry = lax.fori_loop(
            lo,
            hi + 1,
            lambda kv, block_carry: _block_step(
                si,
                kv,
                lo,
                hi,
                ctx,
                (q_tok, base_head, q_slot, row_parity, prev_row_finish),
                (first_next_desc, second_next_desc),
                block_carry,
            ),
            kv_carry,
        )

        finish = RowFinish(
            is_live=jnp.int32(1),
            si=si,
            q_tok=q_tok,
            base_head=base_head,
            num_blend_rows=schedule.blend_start(ctx) - q_tok,
            prev_out_tok=prev_out_tok,
            out_count=num_out_copies,
            parity=lax.rem(row_parity + hi - lo, 2),
        )
        sinks = (
            head_sinks(traits, sinks_ref, None, base_head)
            if variant.has_sinks
            else None
        )
        if return_lse:
            finish_row_lse(
                traits,
                variant,
                state,
                out_refs,
                g,
                finish,
                sinks,
                schedule.packed_q_end,
            )
        if sinks is not None:
            fold_out_sinks(traits, variant, state, finish, sinks)
        next_row_state = (1 - finish.parity, finish)
        return (
            q_loaded,
            q_inflight,
            q_slot,
            kv_carry,
            q_tok,
            num_out_copies + 1,
            next_row_state,
        )

    cancel_pending_pv(pending)
    first_desc, second_desc, _ = _first_two_blocks(jnp.int32(0), jnp.int32(0))
    _start_block_data(first_desc, 0)
    is_second_fresh = _is_fresh(second_desc, first_desc).astype(jnp.int32)
    second_slot = is_second_fresh

    @pl.when(is_second_fresh > 0)
    def _start_second():
        _start_block_data(second_desc, second_slot)

    minus_one = jnp.int32(-1)
    zero = jnp.int32(0)
    init_carry = (
        minus_one,
        minus_one,
        zero,
        (
            zero,
            second_slot,
            jnp.int32(1),
            is_second_fresh,
            first_desc,
            second_desc,
        ),
        zero,
        zero,
        (zero, RowFinish(*([zero] * len(RowFinish._fields)))),
    )
    *_, kv_carry, _, num_out_copies, row_state = lax.fori_loop(
        0,
        num_head_groups,
        lambda g, group_carry: lax.fori_loop(
            0,
            num_rows,
            lambda si, row_carry: _row_body(g, si, row_carry),
            group_carry,
        ),
        init_carry,
    )
    (
        last_slot,
        after_last_slot,
        is_last_fresh,
        is_after_last_fresh,
        last_desc,
        after_last_desc,
    ) = kv_carry

    # Note (david): the last row has no next row to run its pending PV and
    # finish its output.
    _, last_row_finish = _carried_finish(row_state)

    @pl.when(pending_smem[PENDING_WRITE_PARITY] != TRASH_PARITY)
    def _drain_last_pending_pv():
        issue_pending_pv(traits, state, pending)

    cancel_pending_pv(pending)

    @pl.when(last_row_finish.is_live > 0)
    def _finish_last_row():
        _finish_row_out(last_row_finish)

    @pl.when(is_last_fresh > 0)
    def _wait_last():
        _wait_block_data(last_desc, last_slot)

    @pl.when(is_after_last_fresh > 0)
    def _wait_after_last():
        _wait_block_data(after_last_desc, after_last_slot)

    drain(
        num_out_copies,
        num_stages,
        lambda slot: o_copy(traits, out_refs, 0, 0, slot).wait(),
    )

    fill_packed_tail(
        traits, variant, out_refs, schedule.packed_q_end, num_head_groups
    )


def flash_attn_forward_varlen(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cu_seqlens_q: jax.Array,
    cu_seqlens_k: jax.Array,
    sinks: jax.Array | None = None,
    *,
    varlen_sched: VarlenSchedule,
    traits: FwdTraits,
    variant: Variant,
    interpret: bool,
    rotary: tuple[jax.Array, jax.Array | None] | None = None,
    rotary_interleaved: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    check_cu_seqlens_pair(cu_seqlens_k, cu_seqlens_q)
    assert traits.is_per_seq, traits
    traits = dataclasses.replace(
        traits, **rotary_fields(rotary, rotary_interleaved)
    )
    # Note (david): padding rows past cu_seqlens_q[-1] own no q block and come
    # back out = 0, lse = -inf, so they cost no attention; kv tokens past
    # cu_seqlens_k[-1] are masked wherever a block loads them. Resolving the
    # schedule once per call spares every q block a search of the cu_seqlens.
    left, right = variant.window
    table, layout = varlen_schedule_table(
        cu_seqlens_q,
        cu_seqlens_k,
        varlen_sched=varlen_sched,
        bq=traits.bq,
        bkv=traits.bkv,
        bkv_compute=traits.bkv_compute,
        left=left,
        right=right,
    )
    smem_operands = [table]
    variant.check_sinks_operand(sinks)
    if sinks is not None:
        # Note (deepolar): packed rows carry no batch, so the flat q head is the
        # sequence head itself.
        num_seq_heads = (
            q.shape[0]
            if traits.token_major is None
            else traits.token_major.num_q_heads
        )
        if sinks.shape != (num_seq_heads,):
            raise ValueError(
                f"sinks must be ({num_seq_heads},), one per q head; got"
                f" {sinks.shape}."
            )
        smem_operands.append(variant.sink_logits(sinks))

    kernel = partial(
        flash_fwd_varlen_kernel, varlen_sched=varlen_sched, layout=layout
    )
    return forward_common(
        kernel,
        smem_operands,
        q,
        k,
        v,
        traits,
        variant,
        kernel_name="flash_attn_mha_static_t1_guarded_fwd_varlen_per_seq",
        interpret=interpret,
        rotary=rotary,
    )
