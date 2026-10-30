"""Dense (batch) forward kernels: the static schedule and the runtime causal
offset one, and their main loop."""

import dataclasses
from collections.abc import Callable, Sequence
from functools import partial, reduce

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...core.tpu import NUM_LANES
from ..config import (
    QKVLayout,
    TokenMajorInfo,
    from_head_minor,
    vmem_limit_bytes,
)
from ..dma import advance, drain
from ..fwd_blocks import (
    NUM_KV_STAGES,
    NUM_O_SLOTS,
    NUM_PENDING_FIELDS,
    NUM_SEMS,
    PENDING_WRITE_PARITY,
    TRASH_PARITY,
    Fragment,
    KVBlock,
    KVRefs,
    OutRefs,
    PendingPVRefs,
    QRefs,
    RowFinish,
    RowStateRefs,
    ScoreMask,
    SoftmaxPass,
    cancel_pending_pv,
    carry_out_pv,
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
    start_q_row,
    wait_kv_block,
    wait_q_row,
)
from ..fwd_call import (
    check_qkv_operands,
    check_rotary_operands,
    rotary_fields,
    staged_shape,
)
from ..schedule import (
    DenseFwdParams,
    FwdSchedule,
    make_dense_fwd_schedule,
    make_runtime_offset_dense_fwd_schedule,
    window_fragment_plans,
)
from ..traits import FwdTraits
from ..variant import Variant, VariantOperands


def flash_fwd_kernel(
    *refs: jax.Array,
    traits: FwdTraits,
    variant: Variant,
    dense_params: DenseFwdParams,
) -> None:
    if variant.has_sinks:
        sinks_ref, q_hbm, k_hbm, v_hbm, *refs = refs
    else:
        sinks_ref = None
        q_hbm, k_hbm, v_hbm, *refs = refs
    dense_fwd_body(
        traits,
        variant,
        make_dense_fwd_schedule(dense_params),
        q_hbm,
        k_hbm,
        v_hbm,
        refs,
        operands=VariantOperands(sinks_ref=sinks_ref),
    )


def dense_fwd_body(
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
    """Run the dense fragment pipeline over every block the schedule yields.

    traits is the build's static configuration, variant its elementwise math
    and operands the variant's traced values, and refs are the rotary
    operands, the outputs and the scratch, in dense_scratch_shapes's order.
    The logits run in log2 units: Q reaches the QK matmul pre-multiplied by
    log2(e). The varlen and paged builds run their own loops
    (kernels/fwd_varlen.py, kernels/fwd_paged.py).

    operands.sinks_ref is None or an SMEM (nheads,) float32 of per-head sink
    logits in log2 units, one per sequence head (flat q head f reads
    f % nheads, or f // heads_outer_batch heads-outer), each added to its
    head's softmax denominator.
    """
    assert not (traits.is_paged or traits.is_per_seq), traits
    assert schedule.q_token is None and schedule.num_live_tiles is None
    bq, bkv = traits.bq, traits.bkv
    bq_compute, bkv_compute = traits.bq_compute, traits.bkv_compute
    num_stages = traits.num_stages
    num_k_compute_tiles = traits.num_k_compute_tiles
    num_q_compute_tiles = traits.num_q_compute_tiles
    num_compute_tiles = traits.num_compute_tiles
    head_fold = traits.head_fold
    group_heads = traits.group_heads
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
    # dense_scratch_shapes's scratch list; a silent mismatch would shift every
    # ref.
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
    if has_static_mask and isinstance(causal_offset, int) and bq % bkv == 0:
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
    # Note (david): only a left window can mask a row's whole first fragment
    # while later keys stay visible; anchoring on that masked score would push
    # the row's first visible key past the guard into a replay.
    defers_row_start = window[0] is not None
    # A dense build has no per-seq bounds rows.
    mask = ScoreMask(
        causal_offset=causal_offset,
        bounds=None,
        has_q_span=False,
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
    def _block_desc(head, kv):
        """The DMA inputs of kv block kv of head's group."""
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

    pending_frag = pending_fragment(traits)

    def _block_step(si, kv, lo, hi, ctx, row, ahead, carry):
        """One kv block of row si. ahead holds the descriptors of the next row's
        first two blocks in the flattened block order, and carry this block's
        and the next block's ring slot, freshness and descriptor."""
        q_tok, base_head, q_slot, row_parity = row
        (
            kv_slot,
            next_slot,
            is_fresh,
            is_next_fresh,
            block_desc,
            next_block_desc,
        ) = carry
        kv_tok = kv * bkv
        is_first = kv == lo
        is_partial, _ = schedule.block_flags(si, kv, ctx)
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
            _block_desc(base_head, kv + 2),
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

        def _qk(frag):
            return qk(traits, variant, q_refs, kv_refs, block, frag)

        def _online_softmax(frag, logits, softmax_pass):
            return online_softmax(
                traits, variant, state, mask, block, frag, logits, softmax_pass
            )

        def _block_pv(frag, frag_pv):
            fragment_pv(traits, state, kv_refs, block, frag, frag_pv)

        def _run_schedule(
            is_causal,
            guard_recovery,
            static_g,
            window_fragments=None,
            *,
            takes_pending=False,
            carries_out=False,
        ):
            """takes_pending issues the previous block's pending PV after the
            first softmax, and carries_out leaves the last fragment's PV
            pending; a replay passes neither."""

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
            if window_fragments is not None:
                assert not carries_out
                pending_pv = None
                for g in heads:
                    for fragment in window_fragments:
                        frag = Fragment(g, fragment.kv_tile, fragment.q_tile)
                        logits = _qk(frag)
                        frag_pv = _online_softmax(
                            frag,
                            logits,
                            SoftmaxPass(
                                needs_mask=fragment.needs_mask,
                                guard_recovery=guard_recovery,
                                apply_bounds=False,
                                last_kv_tile=num_k_compute_tiles - 1,
                                window_fragment=fragment,
                            ),
                        )
                        if pending_pv is None:
                            _after_first_softmax()
                        else:
                            _block_pv(*pending_pv)
                        pending_pv = (frag, frag_pv)
                _block_pv(*pending_pv)
            else:
                if static_g is None:
                    num_fragments = head_fold * num_compute_tiles

                    def _fragment_coords(frag_idx):
                        g = frag_idx // num_compute_tiles
                        tile_idx = frag_idx % num_compute_tiles
                        return Fragment(
                            g,
                            tile_idx // num_q_compute_tiles,
                            tile_idx % num_q_compute_tiles,
                        )

                else:
                    num_fragments = num_compute_tiles

                    def _fragment_coords(frag_idx):
                        return Fragment(
                            static_g,
                            frag_idx // num_q_compute_tiles,
                            frag_idx % num_q_compute_tiles,
                        )

                # Note (david): a pinned g replaces the loop-carried copy as a
                # Python int, because sub-tile (d64) lane windows need static
                # offsets; a static g costs nothing on the other builds.
                def _resolve_g(frag):
                    if static_g is None:
                        return frag
                    return frag._replace(g=static_g)

                softmax_pass = SoftmaxPass(
                    needs_mask=is_causal,
                    guard_recovery=guard_recovery,
                    apply_bounds=False,
                    last_kv_tile=num_k_compute_tiles - 1,
                )
                first = Fragment(
                    *(jnp.int32(coord) for coord in _fragment_coords(0))
                )
                logits = _qk(_resolve_g(first))
                first_pv = _online_softmax(
                    _resolve_g(first), logits, softmax_pass
                )
                _after_first_softmax()

                def _pipeline_step(frag_idx, carry):
                    prev_pv, prev = carry
                    current = _fragment_coords(frag_idx)
                    logits = _qk(_resolve_g(current))
                    current_pv = _online_softmax(
                        _resolve_g(current), logits, softmax_pass
                    )
                    _block_pv(_resolve_g(prev), prev_pv)
                    return current_pv, current

                last_pv, last = lax.fori_loop(
                    1,
                    num_fragments,
                    _pipeline_step,
                    (first_pv, first),
                    unroll=True,
                )
                if carries_out:
                    # Note (david): this copy runs every fragment, so its last
                    # fragment is the static pending one.
                    assert not guard_recovery
                    carry_out_pv(
                        traits,
                        state,
                        pending,
                        kv_refs,
                        block,
                        pending_frag,
                        last_pv,
                        num_k_compute_tiles == 1,
                    )
                else:
                    _block_pv(_resolve_g(last), last_pv)

        def _run_group(is_causal, window_fragments=None, carries_out=False):
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
                        window_fragments=window_fragments,
                        takes_pending=g == 0,
                        carries_out=carries_out and g == head_fold - 1,
                    )
            else:
                _run_schedule(
                    is_causal,
                    guard_recovery=False,
                    static_g=None,
                    window_fragments=window_fragments,
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
                        window_fragments=window_fragments,
                    )
                    if g == pending_frag.g:
                        cancel_pending_pv(pending)

        # Note (david): a mask-free copy of the fragment pipeline and masked
        # ones, because the masks cost about as many VPU ops as the softmax
        # itself: one per partial-block offset under window_plans, else one
        # applying every mask the build has. One copy per (causal, bounds) pair
        # measured 2.6x slower at block 2048. A build carries either the static
        # window (dense) or the per-seq bounds (varlen), never both, and
        # has_static_mask still gates the former so a full-attention build never
        # loses its upper triangle.
        @pl.when(jnp.logical_not(is_partial))
        def _run_unmasked_block():
            _run_group(is_causal=False, carries_out=True)

        if window_plans is not None:
            for block_offset, fragments in window_plans.items():

                @pl.when(
                    jnp.logical_and(is_partial, kv_tok - q_tok == block_offset)
                )
                def _run_window_block(fragments=fragments):
                    _run_group(is_causal=True, window_fragments=fragments)

        else:

            @pl.when(is_partial)
            def _run_masked_block():
                _run_group(is_causal=has_static_mask)

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
        after_lo, _, _ = schedule.row_interval(after_si)
        first_desc = _block_desc(g * group_heads, lo)
        second_desc = _select_desc(
            lo < hi,
            _block_desc(g * group_heads, lo + 1),
            _block_desc(after_g * group_heads, after_lo),
        )
        return first_desc, second_desc, (lo, hi, ctx)

    def _row_body(g, si, carry):
        """A dense row drains its own last PV and finishes at its end, so its
        first block starts at parity 0."""
        (
            q_loaded,
            q_inflight,
            q_slot,
            kv_carry,
            prev_out_tok,
            num_out_copies,
        ) = carry
        row_parity = 0
        lo, hi, ctx = schedule.row_interval(si)
        # Note (david): past the grid's last row the lookahead is that row
        # itself, so the q stream prefetches nothing, and the kv stream reloads
        # blocks the kernel waits out after its loop.
        next_g, next_si = _row_after(g, si)
        first_next_desc, second_next_desc, _ = _first_two_blocks(
            next_g, next_si
        )
        base_head = g * group_heads
        next_head = next_g * group_heads
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
                (q_tok, base_head, q_slot, row_parity),
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
            num_blend_rows=jnp.int32(0),
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

        @pl.when(pending_smem[PENDING_WRITE_PARITY] != TRASH_PARITY)
        def _drain_pending_pv():
            issue_pending_pv(traits, state, pending)

        cancel_pending_pv(pending)
        _finish_row_out(finish)
        return (
            q_loaded,
            q_inflight,
            q_slot,
            kv_carry,
            prev_out_tok,
            num_out_copies + 1,
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
    )
    *_, kv_carry, _, num_out_copies = lax.fori_loop(
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


def dense_scratch_shapes(
    traits: FwdTraits,
    *,
    q_dtype: jnp.dtype,
    k_dtype: jnp.dtype,
    v_dtype: jnp.dtype,
    rotary_dtype: jnp.dtype | None,
    return_lse: bool,
) -> list[pl.MemoryRef]:
    """Every scratch buffer of one dense build, in the order dense_fwd_body
    unpacks them: Q, K and V stages, Q and (under rotary_k) K rotary
    coefficients, out stage, row max, row sum, f32 accumulator, pending
    probs, pending V tile, the SMEM guard flags, block parity and pending-PV
    fields, the DMA semaphores and the lse rows of each folded head.

    Q, K, V, their rotary coefficients and the pending V tile are staged in
    traits.qkv_layout, the out stage and the accumulator in traits.o_layout.
    rotary_dtype is None without rotary."""
    bq, bkv = traits.bq, traits.bkv
    bq_compute, bkv_compute = traits.bq_compute, traits.bkv_compute
    num_stages = traits.num_stages
    qkv_layout, o_layout = traits.qkv_layout, traits.o_layout
    head_dim_qk, head_dim_v = traits.head_dim_qk, traits.head_dim_v
    head_fold = traits.head_fold
    rotary_dim, rotary_k = traits.rotary_dim, traits.rotary_k
    num_rotary_streams = (1 + int(rotary_k)) if rotary_dim else 0
    folded_bq = traits.group_heads * bq
    vmem = pltpu.VMEM

    scratch_shapes = [
        vmem(
            staged_shape(traits, bq, head_dim_qk, qkv_layout, num_stages),
            q_dtype,
        ),
        vmem(
            staged_shape(traits, bkv, head_dim_qk, qkv_layout, NUM_KV_STAGES),
            k_dtype,
        ),
        vmem(
            staged_shape(traits, bkv, head_dim_v, qkv_layout, NUM_KV_STAGES),
            v_dtype,
        ),
    ]
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
    # Note (david): a lane-paired fold's pending V tile is its pair's 128 lanes,
    # as the forward loop loads it.
    v_tile_width = NUM_LANES if traits.is_lane_paired else head_dim_v
    scratch_shapes += [
        vmem(
            staged_shape(traits, bq, head_dim_v, o_layout, num_stages), q_dtype
        ),
        vmem((folded_bq, NUM_LANES), jnp.float32),
        vmem((2, folded_bq, NUM_LANES), jnp.float32),
        vmem(
            (NUM_O_SLOTS, *from_head_minor((folded_bq, head_dim_v), o_layout)),
            jnp.float32,
        ),
        vmem((bq_compute, bkv_compute), jnp.float32),
        vmem(from_head_minor((bkv_compute, v_tile_width), qkv_layout), v_dtype),
        pltpu.SMEM((head_fold,), jnp.int32),
        pltpu.SMEM((1,), jnp.int32),
        pltpu.SMEM((NUM_PENDING_FIELDS,), jnp.int32),
        pltpu.SemaphoreType.DMA(
            (NUM_SEMS + num_rotary_streams, max(num_stages, NUM_KV_STAGES))
        ),
    ]
    if return_lse:
        scratch_shapes.append(vmem((head_fold, bq), jnp.float32))
    return scratch_shapes


def dense_pallas_call(
    kernel: Callable[..., None],
    smem_operands: list[jax.Array],
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    traits: FwdTraits,
    variant: Variant,
    *,
    kernel_name: str,
    interpret: bool,
    rotary: tuple[jax.Array, jax.Array | None] | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Validate q/k/v against traits, allocate the dense scratch and run a
    dense kernel.

    kernel is the mode kernel with its schedule already bound; traits and
    variant are bound here. smem_operands lead the kernel's inputs, among them
    the variant's traced values, which the kernel reads into VariantOperands.
    rotary is prepare_rotary's packed (batch, tokens, head_dim) Q and K pair
    (K None when K arrives rotated) that traits' rotary fields describe.
    Returns o shaped like q (head_dim_v wide), or (o, lse) with a (num_q_heads,
    q_seq_len) float32 natural-log lse.
    """
    assert not (traits.is_paged or traits.is_per_seq), traits
    bq, bkv = traits.bq, traits.bkv
    qkv_layout, o_layout = traits.qkv_layout, traits.o_layout
    num_q_heads, head_dim_v = traits.num_q_heads, traits.head_dim_v
    transposed_pv = traits.transposed_pv
    token_major = traits.token_major
    return_lse = variant.return_lse
    assert (rotary is None) == (traits.rotary_dim == 0), traits.rotary_dim

    q_seq_len, kv_seq_len = check_qkv_operands(q, k, v, traits)

    # Note (david): a per-seq schedule clamps its windows to the axis end and
    # checks that each block fits, so only the dense grid needs the
    # divisibility.
    if q_seq_len % bq or kv_seq_len % bkv:
        raise ValueError(f"{bq=} and {bkv=} must divide the sequence lengths.")
    rotary_dtype, rotary_operands = check_rotary_operands(
        rotary, traits, q.dtype, q_seq_len, kv_seq_len
    )
    scratch_shapes = dense_scratch_shapes(
        traits,
        q_dtype=q.dtype,
        k_dtype=k.dtype,
        v_dtype=v.dtype,
        rotary_dtype=rotary_dtype,
        return_lse=return_lse,
    )

    if qkv_layout == QKVLayout.SEQ_MINOR:
        q_in, k_in, v_in, *rotary_in = (
            operand.swapaxes(-1, -2) for operand in (q, k, v, *rotary_operands)
        )
    else:
        q_in, k_in, v_in, *rotary_in = (q, k, v, *rotary_operands)
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
    lse_shape = (num_q_heads, q_seq_len)
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
            partial(kernel, traits=traits, variant=variant),
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
    else:
        return o, lse_kernel


def check_dense_sinks(
    sinks: jax.Array,
    num_flat_heads: int,
    heads_outer_batch: int | None,
    token_major: TokenMajorInfo | None,
) -> None:
    """Raise unless sinks is (nheads,), one logit per sequence head."""
    # Note (deepolar): a head-major kernel sees batch * nheads flat heads, so
    # only heads-outer and token-major pin nheads. Batch-outer cannot tell
    # nheads from batch, so it only checks that the length divides the flat
    # heads: every flat head then reads inside the sinks, but a wrong length
    # that divides them passes.
    if token_major is not None:
        is_valid = sinks.shape == (token_major.num_q_heads,)
    elif heads_outer_batch is not None:
        is_valid = sinks.shape == (num_flat_heads // heads_outer_batch,)
    else:
        is_valid = (
            sinks.ndim == 1
            and sinks.shape[0] > 0
            and num_flat_heads % sinks.shape[0] == 0
        )
    if not is_valid:
        raise ValueError(
            "sinks must be (nheads,), one logit per sequence head, for"
            f" {num_flat_heads} flat q heads; got {sinks.shape}."
        )


def flash_attn_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sinks: jax.Array | None = None,
    *,
    dense_params: DenseFwdParams,
    traits: FwdTraits,
    variant: Variant,
    interpret: bool,
    rotary: tuple[jax.Array, jax.Array | None] | None = None,
    rotary_interleaved: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    variant.check_sinks_operand(sinks)
    if sinks is not None:
        check_dense_sinks(
            sinks,
            q.shape[0],
            dense_params.heads_outer_batch,
            traits.token_major,
        )
    smem_operands = [] if sinks is None else [variant.sink_logits(sinks)]
    return dense_pallas_call(
        partial(flash_fwd_kernel, dense_params=dense_params),
        smem_operands,
        q,
        k,
        v,
        dataclasses.replace(
            traits, **rotary_fields(rotary, rotary_interleaved)
        ),
        variant,
        kernel_name="flash_attn_mha_static_t1_guarded_fwd",
        interpret=interpret,
        rotary=rotary,
    )


def flash_fwd_runtime_offset_kernel(
    offset_ref: jax.Array,
    q_hbm: jax.Array,
    k_hbm: jax.Array,
    v_hbm: jax.Array,
    *refs: jax.Array,
    traits: FwdTraits,
    variant: Variant,
    dense_params: DenseFwdParams,
) -> None:
    runtime_offset = offset_ref[0]
    dense_fwd_body(
        traits,
        variant,
        make_runtime_offset_dense_fwd_schedule(dense_params, runtime_offset),
        q_hbm,
        k_hbm,
        v_hbm,
        refs,
        operands=VariantOperands(causal_offset=runtime_offset),
    )


def flash_attn_forward_runtime_offset(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    runtime_offset: jax.Array,
    *,
    dense_params: DenseFwdParams,
    traits: FwdTraits,
    variant: Variant,
    interpret: bool,
    rotary: tuple[jax.Array, jax.Array | None] | None = None,
    rotary_interleaved: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    variant.check_sinks_operand(None)
    offset = jnp.asarray(runtime_offset, jnp.int32).reshape(1)
    return dense_pallas_call(
        partial(flash_fwd_runtime_offset_kernel, dense_params=dense_params),
        [offset],
        q,
        k,
        v,
        dataclasses.replace(
            traits, **rotary_fields(rotary, rotary_interleaved)
        ),
        variant,
        kernel_name="flash_attn_mha_runtime_offset_fwd",
        interpret=interpret,
        rotary=rotary,
    )
