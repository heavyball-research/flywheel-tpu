"""Forward fragment pipeline shared by the batch and varlen flash kernels."""

from collections.abc import Callable, Sequence
from functools import partial, reduce

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..core.tiling import round_up
from ..core.tpu import NUM_LANES
from .config import (
    QKVLayout,
    from_head_minor,
    hbm_bf16_tile_rows,
    vmem_limit_bytes,
)
from .dma import advance, drain
from .fwd_blocks import (
    NUM_KV_STAGES,
    NUM_O_SLOTS,
    NUM_PENDING_FIELDS,
    NUM_SEMS,
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
    flag_overflow_risk,
    fill_packed_tail,
    finish_row_lse,
    finish_row_out,
    fold_out_sinks,
    fragment_pv,
    head_sinks,
    issue_pending_pv,
    o_copy,
    online_softmax,
    pending_fragment,
    qk,
    rotate_staged,
    start_kv_block,
    stage_bounds,
    start_q_row,
    wait_kv_block,
    wait_q_row,
)
from .kv_layout import kv_word_rows
from .schedule import FwdSchedule, window_fragment_plans
from .traits import FwdTraits
from .variant import Variant, VariantOperands


def fwd_body(
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
    """Run the fragment pipeline over every block the schedule yields.

    traits is the build's static configuration, variant its elementwise math
    and operands the variant's traced values, and refs are forward_common's
    rotary operands, outputs and scratch, in its allocation order. The logits
    run in log2 units: Q reaches the QK matmul pre-multiplied by log2(e). A
    per-seq schedule writes the packed padding rows past schedule.packed_q_end
    as out = 0, lse = -inf, and applies the mask itself, so its variant has
    none. A paged build runs kernels/fwd_paged.py's own loop instead.

    operands.sinks_ref is None or an SMEM (nheads,) float32 of per-head sink
    logits in log2 units, one per sequence head (flat q head f reads
    f % nheads, or f // heads_outer_batch heads-outer), each added to its
    head's softmax denominator.
    """
    assert not traits.is_paged, traits
    is_per_seq = traits.is_per_seq
    assert (schedule.q_token is not None) == is_per_seq, is_per_seq
    bq, bkv = traits.bq, traits.bkv
    bq_compute, bkv_compute = traits.bq_compute, traits.bkv_compute
    num_stages = traits.num_stages
    num_k_compute_tiles = traits.num_k_compute_tiles
    num_q_compute_tiles = traits.num_q_compute_tiles
    num_compute_tiles = traits.num_compute_tiles
    head_fold = traits.head_fold
    group_heads, group_bq = traits.group_heads, traits.group_bq
    defers_row_finish = traits.defers_row_finish
    is_lane_paired = traits.is_lane_paired
    rotary_dim, rotary_k = traits.rotary_dim, traits.rotary_k
    variant.check_operands(operands)
    return_lse = variant.return_lse
    window = variant.window
    causal_offset = variant.mask_offset(operands)
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
    bounds_buf = remaining_refs.pop(0) if is_per_seq else None
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
    if return_lse and is_per_seq:
        lse_stage[...] = jnp.zeros(lse_stage.shape, jnp.float32)

    has_static_mask = window != (None, None)
    num_head_groups = schedule.num_head_groups
    num_rows = schedule.num_rows
    q_heads_per_kv_head = schedule.q_heads_per_kv_head
    heads_outer_batch = schedule.heads_outer_batch
    # Note (david): a partial block of a dense window (causal included) masks
    # every fragment of the generic copy, about as many VPU ops as the softmax.
    # Under bq % bkv == 0 and a static offset its fragment flags are static per
    # block offset, so each offset runs its own branch-free copy over the band
    # alone; runtime fragment flags measured 2x slower than masking everything,
    # since a branch per fragment breaks the QK/PV overlap. Any other build
    # masks every fragment of a partial block, which costs compute, never
    # correctness. A runtime causal_offset is a traced scalar, so the isinstance
    # keeps it off.
    if (
        has_static_mask
        and not is_per_seq
        and isinstance(causal_offset, int)
        and bq % bkv == 0
    ):
        window_plans = window_fragment_plans(
            bq=bq,
            bkv=bkv,
            bq_compute=bq_compute,
            bkv_compute=bkv_compute,
            left=window[0],
            right=window[1],
            causal_offset=causal_offset,
        )
    else:
        window_plans = None
    # Note (david): a per-seq q block's kv frontier is a runtime scalar, so a
    # masked block runs only its kv compute tiles below it, in a tile loop that
    # indexes the tile at runtime; only a sublane token axis allows that.
    skips_past_frontier = schedule.num_live_tiles is not None
    assert skips_past_frontier == is_per_seq, (skips_past_frontier, is_per_seq)
    # Note (david): only a left window can mask a row's whole first fragment
    # while later keys stay visible; anchoring on that masked score would push
    # the row's first visible key past the guard into a replay.
    defers_row_start = window[0] is not None or schedule.has_q_span
    mask = ScoreMask(
        causal_offset=causal_offset,
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
        if heads_outer_batch is None:
            kv_head = head // q_heads_per_kv_head
        else:
            kv_head = (
                head // heads_outer_batch // q_heads_per_kv_head
            ) * heads_outer_batch + head % heads_outer_batch
        if is_per_seq:
            desc = (kv_head, schedule.kv_token(kv, ctx))
        else:
            desc = (kv_head, kv * bkv)
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
        if is_per_seq:
            kv_tok = schedule.kv_token(kv, ctx)
        else:
            kv_tok = kv * bkv
        is_first = kv == lo
        is_partial, needs_bounds = schedule.block_flags(si, kv, ctx)
        if skips_past_frontier:
            num_live_tiles = schedule.num_live_tiles(kv, ctx)
        else:
            num_live_tiles = None
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
            needs_mask,
            *,
            last_kv_tile=num_k_compute_tiles - 1,
            **flags,
        ):
            """online_softmax of one of this block's fragments, flags its
            keyword flags."""
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
                needs_mask,
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
            is_causal,
            guard_recovery,
            static_g,
            apply_bounds,
            window_fragments=None,
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
                            is_causal,
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
                            pending = (
                                g,
                                kv_compute_index - 1,
                                last_q_tile,
                                *carry,
                                None,
                            )
                            *_, probs, is_row_start, rescale, _ = (
                                _tile_fragments(
                                    g, kv_compute_index, False, pending
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
            elif window_fragments is not None:
                assert not carries_out
                pending_pv = None
                for g in heads:
                    for fragment in window_fragments:
                        logits = _qk(g, fragment.kv_tile, fragment.q_tile)
                        probs, is_row_start, rescale = _online_softmax(
                            g,
                            fragment.kv_tile,
                            fragment.q_tile,
                            logits,
                            fragment.needs_mask,
                            guard_recovery=guard_recovery,
                            apply_bounds=False,
                            window_fragment=fragment,
                        )
                        if pending_pv is None:
                            _after_first_softmax()
                        else:
                            _block_pv(*pending_pv)
                        pending_pv = (
                            g,
                            fragment.kv_tile,
                            fragment.q_tile,
                            probs,
                            is_row_start,
                            rescale,
                        )
                _block_pv(*pending_pv)
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
                        needs_mask=is_causal,
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
                            is_causal,
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

        def _run_group(
            is_causal,
            apply_bounds,
            window_fragments=None,
            skips_dead_tiles=False,
            carries_out=False,
        ):
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
                        is_causal,
                        guard_recovery=False,
                        static_g=g,
                        apply_bounds=apply_bounds,
                        window_fragments=window_fragments,
                        skips_dead_tiles=skips_dead_tiles,
                        takes_pending=g == 0,
                        carries_out=carries_out and g == head_fold - 1,
                    )
            else:
                _run_schedule(
                    is_causal,
                    guard_recovery=False,
                    static_g=None,
                    apply_bounds=apply_bounds,
                    window_fragments=window_fragments,
                    skips_dead_tiles=skips_dead_tiles,
                    takes_pending=True,
                    carries_out=carries_out,
                )

            for g in range(head_fold):

                @pl.when(guard_smem[g] > 0)
                def _recover(g=g):
                    _run_schedule(
                        is_causal,
                        guard_recovery=True,
                        static_g=g,
                        apply_bounds=apply_bounds,
                        window_fragments=window_fragments,
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
        if is_per_seq:
            needs_any_mask = jnp.logical_or(is_partial, needs_bounds)
        else:
            needs_any_mask = is_partial

        @pl.when(jnp.logical_not(needs_any_mask))
        def _run_unmasked_block():
            _run_group(is_causal=False, apply_bounds=False, carries_out=True)

        # Note (david): each tile loop step ends a basic block the fragment
        # pipeline cannot overlap across, so only a block with dead tiles takes
        # the loop: running every masked block through it slowed the packed
        # varlen kernel by 7% (v6e, 32:8 d128: 7.35 against 6.85 ms).
        if skips_past_frontier and num_k_compute_tiles > 1:
            skips_dead_tiles = num_live_tiles < num_k_compute_tiles
            runs_every_tile = jnp.logical_and(
                needs_any_mask, jnp.logical_not(skips_dead_tiles)
            )

            @pl.when(jnp.logical_and(needs_any_mask, skips_dead_tiles))
            def _run_live_tile_block():
                stage_bounds(traits, schedule, bounds_buf, ctx, kv, kv_tok)
                _run_group(
                    is_causal=has_static_mask,
                    apply_bounds=True,
                    skips_dead_tiles=True,
                    carries_out=defers_row_finish,
                )

        else:
            runs_every_tile = needs_any_mask

        if window_plans is not None:
            for block_offset, fragments in window_plans.items():

                @pl.when(
                    jnp.logical_and(
                        runs_every_tile, kv_tok - q_tok == block_offset
                    )
                )
                def _run_window_block(fragments=fragments):
                    _run_group(
                        is_causal=True,
                        apply_bounds=False,
                        window_fragments=fragments,
                    )

        else:

            @pl.when(runs_every_tile)
            def _run_masked_block():
                if is_per_seq:
                    stage_bounds(traits, schedule, bounds_buf, ctx, kv, kv_tok)
                _run_group(
                    is_causal=has_static_mask,
                    apply_bounds=is_per_seq,
                    carries_out=defers_row_finish,
                )

        if defers_row_finish:
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
        """carry ends with (the row's first block's new parity, the previous
        row's RowFinish) under defers_row_finish, else with ()."""
        (
            q_loaded,
            q_inflight,
            q_slot,
            kv_carry,
            prev_out_tok,
            num_out_copies,
            row_state,
        ) = carry
        if defers_row_finish:
            row_parity, prev_row_finish = _carried_finish(row_state)
        else:
            row_parity, prev_row_finish = 0, None
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
        if is_per_seq:
            q_tok = schedule.q_token(ctx)
            next_q_tok = schedule.q_token(next_ctx)
        else:
            q_tok = (schedule.first_row + si) * bq
            next_q_tok = (schedule.first_row + next_si) * bq

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
            num_blend_rows=(
                schedule.blend_start(ctx) - q_tok
                if is_per_seq
                else jnp.int32(0)
            ),
            prev_out_tok=prev_out_tok,
            out_count=num_out_copies,
            parity=lax.rem(row_parity + hi - lo, 2),
        )
        sinks = (
            head_sinks(traits, sinks_ref, heads_outer_batch, base_head)
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
        if defers_row_finish:
            next_row_state = (1 - finish.parity, finish)
        else:

            @pl.when(pending_smem[PENDING_WRITE_PARITY] != TRASH_PARITY)
            def _drain_pending_pv():
                issue_pending_pv(traits, state, pending)

            cancel_pending_pv(pending)
            _finish_row_out(finish)
            next_row_state = ()
        next_prev_out_tok = q_tok if is_per_seq else prev_out_tok
        return (
            q_loaded,
            q_inflight,
            q_slot,
            kv_carry,
            next_prev_out_tok,
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
    if defers_row_finish:
        init_row_state = (zero, RowFinish(*([zero] * len(RowFinish._fields))))
    else:
        init_row_state = ()
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
        init_row_state,
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

    if defers_row_finish:
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

    if is_per_seq:
        fill_packed_tail(
            traits, variant, out_refs, schedule.packed_q_end, num_head_groups
        )


def rotary_fields(
    rotary: tuple[jax.Array, jax.Array | None] | None,
    rotary_interleaved: bool,
) -> dict[str, int | bool]:
    """The FwdTraits rotary fields of prepare_rotary's packed (batch, tokens,
    head_dim) Q and K coefficient pair (K None when K arrives rotated); none
    without rotary."""
    if rotary is None:
        return {}
    if len(rotary) != 2:
        raise ValueError("rotary must contain Q and K coefficient arrays.")
    rotary_q_coeff, rotary_k_coeff = rotary
    rotary_k = rotary_k_coeff is not None
    if (
        rotary_q_coeff.ndim != 3
        or rotary_q_coeff.shape[0] == 0
        or (
            rotary_k
            and (
                rotary_k_coeff.ndim != 3
                or rotary_q_coeff.shape[0] != rotary_k_coeff.shape[0]
                or rotary_q_coeff.shape[2] != rotary_k_coeff.shape[2]
            )
        )
    ):
        raise ValueError(
            "rotary coefficients must be (batch, tokens, head_dim)."
        )
    if rotary_q_coeff.shape[2] <= 0 or (
        rotary_k and rotary_k_coeff.dtype != rotary_q_coeff.dtype
    ):
        raise ValueError("Invalid rotary dimension or coefficient dtype.")
    return {
        "rotary_dim": rotary_q_coeff.shape[2],
        "rotary_batch": rotary_q_coeff.shape[0],
        "rotary_interleaved": rotary_interleaved,
        "rotary_k": rotary_k,
    }


def forward_common(
    kernel: Callable[..., None],
    smem_operands: list[jax.Array],
    q: jax.Array,
    k: jax.Array,
    v: jax.Array | None,
    traits: FwdTraits,
    variant: Variant,
    *,
    kernel_name: str,
    interpret: bool,
    rotary: tuple[jax.Array, jax.Array | None] | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Validate q/k/v against traits, allocate scratch and run a fwd_body
    kernel.

    kernel is the mode kernel with its schedule already bound; traits and
    variant are bound here. smem_operands lead the kernel's inputs, among them
    the variant's traced values, which the kernel reads into VariantOperands.
    rotary is prepare_rotary's packed (batch, tokens, head_dim) Q and K pair
    (K None when K arrives rotated) that traits' rotary fields describe.
    Returns o shaped like q (head_dim_v wide), or (o, lse) with a (num_q_heads,
    q_seq_len) float32 natural-log lse.

    A PagedFwdTraits build (per-seq, head-major q only) makes k the merged
    interleaved cache traits.paged describes and v None; the kernel then takes
    (q, k, *rotary).
    """
    bq, bkv = traits.bq, traits.bkv
    bkv_compute = traits.bkv_compute
    bq_compute = traits.bq_compute
    num_stages = traits.num_stages
    qkv_layout = traits.qkv_layout
    num_q_heads, num_kv_heads = traits.num_q_heads, traits.num_kv_heads
    head_dim_qk, head_dim_v = traits.head_dim_qk, traits.head_dim_v
    head_fold, gqa_fold = traits.head_fold, traits.gqa_fold
    transposed_pv = traits.transposed_pv
    token_major, paged = traits.token_major, traits.paged
    is_per_seq = traits.is_per_seq
    rotary_dim, rotary_k = traits.rotary_dim, traits.rotary_k
    return_lse = variant.return_lse
    assert (rotary is None) == (rotary_dim == 0), rotary_dim

    if paged is None and k.shape[:-1] != v.shape[:-1]:
        raise ValueError(
            f"'key' {k.shape} and 'value' {v.shape} must have the same leading"
            " dimensions."
        )

    if paged is not None:
        q_seq_len = q.shape[1]
        # Note (david): seqused_k bounds every sequence's kv blocks, so the kv
        # axis only needs its block count; the padded capacity stands in for its
        # length.
        kv_seq_len = round_up(paged.pages_per_seq * paged.page_size, bkv)
    elif token_major is not None:
        batch = token_major.batch
        q_heads_per_row = token_major.num_q_heads
        kv_heads_per_row = token_major.num_kv_heads
        expected_rank = 2 if batch is None else 3
        if (
            q.ndim != expected_rank
            or k.ndim != expected_rank
            or v.ndim != expected_rank
        ):
            raise ValueError(
                f"token-major with batch={batch} expects rank-{expected_rank}"
                f" q/k/v; got {q.shape=}, {k.shape=}, {v.shape=}."
            )
        if batch is not None and (q.shape[0] != batch or k.shape[0] != batch):
            raise ValueError(
                f"q/k lead with batch={batch}; got {q.shape=}, {k.shape=}."
            )
        if q.shape[-1] != q_heads_per_row * head_dim_qk:
            raise ValueError(
                f"q's fused width {q.shape[-1]} != num_heads {q_heads_per_row}"
                f" * head_dim {head_dim_qk}."
            )
        if k.shape[-1] != kv_heads_per_row * head_dim_qk:
            raise ValueError(
                f"k's fused width {k.shape[-1]} != num_kv_heads"
                f" {kv_heads_per_row} * head_dim {head_dim_qk}."
            )
        if v.shape[-1] != kv_heads_per_row * head_dim_v:
            raise ValueError(
                f"v's fused width {v.shape[-1]} != num_kv_heads"
                f" {kv_heads_per_row} * head_dim_v {head_dim_v}."
            )
        q_seq_len = q.shape[-2]
        kv_seq_len = k.shape[-2]
    else:
        q_seq_len = q.shape[1]

        if (
            q.ndim != 3
            or q.shape[0] != num_q_heads
            or q.shape[2] != head_dim_qk
        ):
            raise ValueError(
                f"'query' {q.shape} must be (num_q_heads={num_q_heads}, seq,"
                f" head_dim={head_dim_qk})."
            )
        if v.ndim != 3 or v.shape[2] != head_dim_v:
            raise ValueError(
                f"'value' {v.shape} must be (num_kv_heads, seq,"
                f" head_dim_v={head_dim_v})."
            )
        if k.ndim != 3:
            raise ValueError(
                f"Expected 3-dim 'key' tensor. Instead got a {k.ndim}-dim one."
            )
        if k.shape[2] != head_dim_qk:
            raise ValueError(
                f"Expected 'key' head dimension to be: {head_dim_qk}. Instead"
                f" got: {k.shape[2]}."
            )
        if k.shape[0] != num_kv_heads:
            raise ValueError(
                f"'key' has {k.shape[0]} heads but the schedule was built for"
                f" {num_kv_heads} kv heads."
            )

        kv_seq_len = k.shape[1]

    # Note (david): a per-seq schedule clamps its windows to the axis end and
    # checks that each block fits, so only the dense grid needs the
    # divisibility.
    if not is_per_seq and (q_seq_len % bq or kv_seq_len % bkv):
        raise ValueError(f"{bq=} and {bkv=} must divide the sequence lengths.")
    if rotary is None:
        rotary_dtype = q.dtype
        rotary_operands = []
    else:
        rotary_q_coeff, rotary_k_coeff = rotary
        if rotary_q_coeff.shape[1] != q_seq_len or (
            rotary_k and rotary_k_coeff.shape[1] != kv_seq_len
        ):
            raise ValueError(
                "rotary coefficients must be (batch, tokens, head_dim)."
            )
        # Note (david): float32 coefficients keep the in-kernel f32 rotation
        # identical to an f32 RoPE computed outside the kernel.
        rotary_dtype = rotary_q_coeff.dtype
        if rotary_dtype not in (q.dtype, jnp.dtype(jnp.float32)):
            raise ValueError("Invalid rotary dimension or coefficient dtype.")
        rotary_operands = [coeff for coeff in rotary if coeff is not None]
    num_rotary_streams = (1 + int(rotary_k)) if rotary_dim else 0

    fwd_kernel = partial(kernel, traits=traits, variant=variant)

    group_heads = traits.group_heads
    folded_bq = group_heads * bq
    o_layout = traits.o_layout
    vmem = pltpu.VMEM

    def _staged_shape(rows, head_dim, layout, stages=num_stages):
        if group_heads == 1:
            return (stages, *from_head_minor((rows, head_dim), layout))
        elif token_major is not None:
            # Note (david): a token-major group is head_fold * head_dim
            # consecutive lanes of one row, so the fold rides the lane axis and
            # one DMA per stream lands the whole group.
            return (stages, rows, head_fold * head_dim)
        else:
            # Note (david): an explicit fold axis rank-matches the 3-D HBM head
            # slices, so XLA never materializes a reshaped q/k/v copy, which
            # dominated wall time at short sequences when head_dim < 128 needed
            # lane padding.
            return (
                stages,
                group_heads,
                *from_head_minor((rows, head_dim), layout),
            )

    q_buf = vmem(_staged_shape(bq, head_dim_qk, qkv_layout), q.dtype)
    if paged is None:
        k_buf = vmem(
            _staged_shape(bkv, head_dim_qk, qkv_layout, NUM_KV_STAGES), k.dtype
        )
        v_buf = vmem(
            _staged_shape(bkv, head_dim_v, qkv_layout, NUM_KV_STAGES), v.dtype
        )
        scratch_shapes = [q_buf, k_buf, v_buf]
    else:
        # Note (david): a paged block stages the u32 word rows holding one kv
        # head's K and V, then unpacks them into one bf16 K and one bf16 V
        # plane; a word's (kv_word_rows, head_dim) rows pad to nothing, so the
        # staging costs what two dense planes of the cache dtype would.
        scratch_shapes = [
            q_buf,
            vmem(
                (NUM_KV_STAGES, bkv, 1, kv_word_rows(k.dtype), head_dim_qk),
                k.dtype,
            ),
            vmem((2, bkv, head_dim_qk), q.dtype),
        ]
    if is_per_seq:
        # Note (david): the per-seq kernel synthesizes its bounds rows (the q
        # lower bound, plus the span under a left window) once per block
        # in-kernel, so it needs one block's rows, not a kv-stream ring. Both
        # rows share one (8, 128) int32 tile, so the span row costs no extra
        # VMEM.
        scratch_shapes.append(
            vmem((bkv // bkv_compute, 2, bkv_compute), jnp.int32)
        )
    if rotary_dim:
        scratch_shapes += [
            vmem(
                (
                    stages,
                    traits.rotary_groups,
                    *from_head_minor((block, rotary_dim), qkv_layout),
                ),
                rotary_dtype,
            )
            for block, stages in (
                ((bq, num_stages), (bkv, NUM_KV_STAGES))
                if rotary_k
                else ((bq, num_stages),)
            )
        ]
    o_stage = vmem(_staged_shape(bq, head_dim_v, o_layout), q.dtype)
    row_max_scratch = vmem((folded_bq, NUM_LANES), jnp.float32)
    row_sum_scratch = vmem((2, folded_bq, NUM_LANES), jnp.float32)
    o_scratch = vmem(
        (NUM_O_SLOTS, *from_head_minor((folded_bq, head_dim_v), o_layout)),
        jnp.float32,
    )
    # Note (david): a lane-paired fold's pending V tile is its pair's 128 lanes,
    # as fwd_body loads it.
    pending_probs = vmem((gqa_fold * bq_compute, bkv_compute), jnp.float32)
    v_tile_width = NUM_LANES if traits.is_lane_paired else head_dim_v
    pending_v = vmem(
        from_head_minor((bkv_compute, v_tile_width), qkv_layout),
        q.dtype if paged is not None else v.dtype,
    )
    guard_smem = pltpu.SMEM((head_fold,), jnp.int32)
    parity_smem = pltpu.SMEM((1,), jnp.int32)
    pending_smem = pltpu.SMEM((NUM_PENDING_FIELDS,), jnp.int32)
    sems = pltpu.SemaphoreType.DMA(
        (NUM_SEMS + num_rotary_streams, max(num_stages, NUM_KV_STAGES))
    )
    scratch_shapes += [
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
    ]
    if return_lse and is_per_seq:
        scratch_shapes.append(vmem((num_stages, bq, NUM_LANES), jnp.float32))
    elif return_lse:
        scratch_shapes.append(vmem((head_fold, bq), jnp.float32))

    if qkv_layout == QKVLayout.SEQ_MINOR:
        q_in, k_in, v_in, *rotary_in = (
            operand.swapaxes(-1, -2) for operand in (q, k, v, *rotary_operands)
        )
    else:
        q_in, k_in, v_in, *rotary_in = (q, k, v, *rotary_operands)
    if paged is not None:
        # Note (david): the pool is pinned to HBM because memory space
        # assignment otherwise moves a pool that fits its VMEM budget there and
        # back around every call, the whole pool however few pages the call
        # reads (v6e, 16 MB e5m2 pool: 0.369 against 0.339 ms a chunk). The spec
        # stays pl.ANY: a layout copy XLA inserts before the call escapes the
        # constraint and can land in VMEM, where DMAs compiled for an HBM
        # operand halt the chip.
        kernel_inputs = [
            q_in,
            pltpu.with_memory_space_constraint(k_in, pltpu.HBM),
            *rotary_in,
        ]
    else:
        kernel_inputs = [q_in, k_in, v_in, *rotary_in]
    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    in_specs = [smem_spec] * len(smem_operands) + [any_spec] * len(
        kernel_inputs
    )

    if token_major is None:
        o_shape = (
            num_q_heads,
            *from_head_minor((q_seq_len, head_dim_v), o_layout),
        )
    else:
        o_shape = (
            *q.shape[:-2],
            q_seq_len,
            token_major.num_q_heads * head_dim_v,
        )
    # Note (david): per-seq LSE gives every head group an explicit minor axis of
    # whole 128-lane tiles, so fwd_body DMAs complete tiles at offset zero and
    # groups never alias in the physical layout; only the first group_heads
    # lanes are returned.
    lse_shape = (
        (num_q_heads // group_heads, q_seq_len, NUM_LANES)
        if is_per_seq
        else (num_q_heads, q_seq_len)
    )
    if return_lse:
        out_specs = [any_spec, any_spec]
        out_shape = [
            jax.ShapeDtypeStruct(o_shape, q.dtype),
            jax.ShapeDtypeStruct(lse_shape, jnp.float32),
        ]
    else:
        out_specs = any_spec
        out_shape = jax.ShapeDtypeStruct(o_shape, q.dtype)

    with jax.named_scope(kernel_name):
        outputs = pl.pallas_call(
            fwd_kernel,
            in_specs=in_specs,
            out_specs=out_specs,
            out_shape=out_shape,
            scratch_shapes=scratch_shapes,
            name=kernel_name,
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=vmem_limit_bytes()
            ),
            interpret=pltpu.InterpretParams() if interpret else False,
        )(*smem_operands, *kernel_inputs)
    o_kernel, lse_kernel = outputs if return_lse else (outputs, None)
    # Note (david): transposed_pv emits (heads, head_dim_v, seq); for d64 the
    # swap back is the {1,2,0} layout XLA already prefers, so it folds into a
    # bitcast.
    o = o_kernel.swapaxes(-1, -2) if transposed_pv else o_kernel
    if not return_lse:
        return o
    elif is_per_seq:
        return o, lse_kernel[:, :, :group_heads].swapaxes(1, 2).reshape(
            num_q_heads, q_seq_len
        )
    else:
        return o, lse_kernel
