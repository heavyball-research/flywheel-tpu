"""Read-only varlen forward kernel over a paged KV cache.

Each sequence attends its own cache prefix [0, seqused_k[r]), staged page by
page through its block table row. The cache is one merged (num_pages,
page_size, cache_heads, head_dim) pool whose token rows interleave the kv
heads' K and V rows in kv_layout's merged order. Every head group is one kv
head, which stages only the u32 word rows of each page that hold its K and
V, and whose q heads are stacked over the rows of every score fragment.
"""

import math
from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...core.tiling import round_up
from ...core.tpu import F32_BYTES, NUM_LANES, NUM_SUBLANES
from ..config import (
    MIN_Q_TILES_PER_BLOCK,
    PAGE_SIZE_MULTIPLE,
    BlockSizes,
    PagedKVInfo,
    hbm_bf16_tile_rows,
    next_pow2,
    vmem_limit_bytes,
)
from ..dma import advance, drain
from ..fwd_blocks import (
    NUM_KV_STAGES,
    NUM_O_SLOTS,
    NUM_PENDING_FIELDS,
    NUM_SEMS,
    PENDING_WRITE_PARITY,
    SEM_K,
    TRASH_PARITY,
    Fragment,
    KVBlock,
    OutRefs,
    PagedKVRefs,
    PendingPVRefs,
    QRefs,
    RowFinish,
    RowStateRefs,
    ScoreMask,
    SoftmaxPass,
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
    stage_bounds,
    start_q_row,
    wait_q_row,
)
from ..fwd_call import check_rotary_operands, rotary_fields
from ..kv_layout import (
    kv_word_halves,
    kv_word_rows,
    load_staged_words,
    resolve_merged_kv_heads,
    staged_words,
)
from ..schedule import (
    FwdSchedule,
    ScheduleTableLayout,
    make_paged_fwd_schedule,
    paged_schedule_table,
)
from ..softmax import overflow_guard_threshold
from ..traits import PagedFwdTraits
from ..variant import Variant, VariantOperands

# Note (david): the distance-1 fragment pipeline keeps five score-shaped f32
# temporaries live, plus a one-byte bounds mask per score.
SCORE_TEMPORARY_BYTES = 5 * F32_BYTES + 1
# Note (david): no per-head fold. The fragment pipeline is fully unrolled over
# head_fold * compute tiles, and folding a kv head's q heads that way measured
# 2.1x slower on v6e (h32 k4 d256, 1024-token chunks over 16K: 22.6 against
# 10.6 ms) with a 15x longer compile. The kv head's q heads are folded into the
# fragment rows instead (paged_fwd_body's gqa_fold).
PAGED_HEAD_FOLD = 1


def flash_fwd_varlen_paged_kernel(
    table_ref: jax.Array,
    block_table_ref: jax.Array,
    kv_scales_ref: jax.Array,
    *refs: jax.Array,
    traits: PagedFwdTraits,
    variant: Variant,
    layout: ScheduleTableLayout,
) -> None:
    if variant.has_sinks:
        sinks_ref, q_hbm, kv_hbm, *refs = refs
    else:
        sinks_ref = None
        q_hbm, kv_hbm, *refs = refs
    window_left, window_right = variant.window
    schedule = make_paged_fwd_schedule(
        table_ref,
        block_table_ref,
        layout=layout,
        num_head_groups=traits.num_head_groups,
        q_heads_per_kv_head=traits.q_heads_per_kv_head,
        bq=traits.bq,
        bkv=traits.bkv,
        bkv_compute=traits.bkv_compute,
        page_size=traits.paged.page_size,
        pages_per_seq=traits.paged.pages_per_seq,
        causal=window_right == 0,
        left=window_left,
    )
    # Note (david): K and V share the one merged cache operand. The K and V
    # descales arrive in SMEM, so a new descale value compiles no new kernel.
    q_scale = variant.descaled_q_scale(kv_scales_ref[0])
    v_scale = kv_scales_ref[1]
    # The schedule masks each sequence around its own diagonal, so the loop
    # runs without the Variant's static mask.
    paged_fwd_body(
        traits,
        variant.without_mask(),
        schedule,
        q_hbm,
        kv_hbm,
        refs,
        operands=VariantOperands(
            q_scale=q_scale, v_scale=v_scale, sinks_ref=sinks_ref
        ),
    )


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
    traits: PagedFwdTraits, kv_load_end: jax.Array, kv: jax.Array
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
    traits: PagedFwdTraits,
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
    traits: PagedFwdTraits,
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


def unpack_paged_block(
    traits: PagedFwdTraits,
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


def paged_fwd_body(
    traits: PagedFwdTraits,
    variant: Variant,
    schedule: FwdSchedule,
    q_hbm: jax.Array,
    cache_hbm: jax.Array,
    refs: Sequence[jax.Array],
    *,
    operands: VariantOperands,
) -> None:
    """Run the paged fragment pipeline over every block the schedule yields.

    traits is the build's static configuration, variant its elementwise math
    and operands the variant's traced values, and refs are the rotary Q
    operand, the outputs and the scratch, in paged_scratch_shapes's order. The
    logits run in log2 units: Q reaches the QK matmul pre-multiplied by
    log2(e). K and V come from the one merged interleaved cache operand
    cache_hbm. The per-seq schedule writes the packed padding rows past
    schedule.packed_q_end as out = 0, lse = -inf, and applies the mask
    itself, so its variant has none.

    A group is the gqa_fold q heads of one kv head and runs one pipeline head
    (head_fold is PAGED_HEAD_FOLD). gqa_fold > 1 stacks the group's heads over
    the rows of every score fragment: fragment row r holds head r //
    bq_compute at token r % bq_compute of the q tile, so bq and bq_compute
    count tokens per head. A paged fp8 cache is widened to bf16 K and V
    planes, and V's descale multiplies the output.

    operands.sinks_ref is None or an SMEM (nheads,) float32 of per-head sink
    logits in log2 units, one per sequence head (flat q head f reads
    f % nheads), each added to its head's softmax denominator.
    """
    assert schedule.q_token is not None and schedule.kv_page is not None
    assert schedule.heads_outer_batch is None, schedule.heads_outer_batch
    # The schedule masks each sequence around its own diagonal through the
    # bounds rows, so no fragment applies a static mask (needs_mask False).
    assert variant.window == (None, None), variant.window
    num_stages = traits.num_stages
    num_k_compute_tiles = traits.num_k_compute_tiles
    num_q_compute_tiles = traits.num_q_compute_tiles
    num_compute_tiles = traits.num_compute_tiles
    head_fold = traits.head_fold
    group_heads, group_bq = traits.group_heads, traits.group_bq
    rotary_dim = traits.rotary_dim
    # A paged build rotates Q only; its cache holds rotated keys.
    assert not (rotary_dim and traits.rotary_k), traits
    variant.check_operands(operands)
    return_lse = variant.return_lse
    sinks_ref = operands.sinks_ref

    remaining_refs = list(refs)
    rotary_q_hbm = remaining_refs.pop(0) if rotary_dim else None
    o_hbm = remaining_refs.pop(0)
    lse_hbm = remaining_refs.pop(0) if return_lse else None
    q_buf = remaining_refs.pop(0)
    # Note (david): a paged block stages its kv head's K and V rows as a
    # (bkv, 1, kv_word_rows, head_dim) block and unpacks it once into dense
    # K and V planes, so every q tile of the block reads plain tiles.
    kv_pair_buf, kv_planes = remaining_refs.pop(0), remaining_refs.pop(0)
    bounds_buf = remaining_refs.pop(0)
    rotary_q_buf = remaining_refs.pop(0) if rotary_dim else None
    if return_lse:
        *pipeline_refs, lse_stage = remaining_refs
    else:
        pipeline_refs, lse_stage = remaining_refs, None
    # Note (david): the exact unpack pins this variable-arity tail against
    # paged_scratch_shapes's scratch list; a silent mismatch would shift every
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
    kv_refs = PagedKVRefs(
        cache_hbm=cache_hbm, pair_buf=kv_pair_buf, planes=kv_planes, sems=sems
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
        desc = (
            ctx.page_base,
            kv_head,
            ctx.kv_load_end,
            ctx.kv_used_end,
            kv,
        )
        # Note (david): the descriptors ride the loop carry, which needs strong
        # int32 scalars.
        return tuple(jnp.asarray(field, jnp.int32) for field in desc)

    def _select_desc(predicate, desc_true, desc_false):
        return tuple(
            jnp.where(predicate, field_true, field_false)
            for field_true, field_false in zip(desc_true, desc_false)
        )

    def _start_block_data(desc, slot):
        page_base, kv_head, kv_load_end, _, kv = desc
        start_paged_kv_block(
            traits,
            schedule,
            kv_refs,
            page_base,
            kv_head,
            kv_load_end,
            kv,
            slot,
        )

    def _wait_block_data(desc, slot):
        _, _, kv_load_end, _, kv = desc
        wait_paged_kv_block(traits, kv_refs, kv_load_end, kv, slot)

    pending_frag = pending_fragment(traits)

    def _block_step(si, kv, lo, hi, ctx, row, ahead, carry):
        """One kv block of row si. ahead holds the descriptors of the next row's
        first two blocks in the flattened block order, and carry this block's
        and the next block's ring slot and descriptor."""
        q_tok, base_head, q_slot, row_parity, prev_row_finish = row
        kv_slot, next_slot, block_desc, next_block_desc = carry
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
        ahead_slot = lax.rem(next_slot + 1, NUM_KV_STAGES)
        for g in range(head_fold):
            guard_smem[g] = 0

        _wait_block_data(block_desc, kv_slot)

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

        def _block_pv(frag, frag_pv, is_tile_zero=None):
            fragment_pv(
                traits, state, kv_refs, block, frag, frag_pv, is_tile_zero
            )

        def _carry_out_pv(frag, frag_pv, is_tile_zero):
            carry_out_pv(
                traits,
                state,
                pending,
                kv_refs,
                block,
                frag,
                frag_pv,
                is_tile_zero,
            )

        def _run_schedule(apply_bounds, skips_dead_tiles, static_g=None):
            """static_g None is the ordinary pass, which issues the previous
            block's pending PV after its first softmax and leaves its last
            fragment's PV pending; an int static_g replays that pipeline
            head's overflow (guard_recovery) and does neither.
            skips_dead_tiles runs a bounds-masked block's live tiles alone."""
            guard_recovery = static_g is not None

            def _after_first_softmax():
                if not guard_recovery:
                    issue_pending_pv(traits, state, pending)

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
                    """Run kv tile kv_compute_index's fragments of head g after
                    pending_pv, a (Fragment, FragmentPV, is_tile_zero) whose PV
                    is still due (None before the first softmax); returns the
                    last fragment's, now due."""
                    softmax_pass = SoftmaxPass(
                        needs_mask=False,
                        guard_recovery=guard_recovery,
                        apply_bounds=True,
                        last_kv_tile=None if has_tile_loop else 0,
                        is_tile_zero=is_tile_zero,
                    )
                    for q_compute_index in range(num_q_compute_tiles):
                        frag = Fragment(g, kv_compute_index, q_compute_index)
                        logits = _qk(frag)
                        frag_pv = _online_softmax(frag, logits, softmax_pass)
                        if pending_pv is None:
                            _after_first_softmax()
                        else:
                            _block_pv(*pending_pv)
                        pending_pv = (frag, frag_pv, is_tile_zero)
                    return pending_pv

                pending_pv = None
                for g in heads:
                    pending_pv = _tile_fragments(g, 0, True, pending_pv)
                    if has_tile_loop:

                        def _tile_step(kv_compute_index, carry, g=g):
                            # Note (david): the carried PV is tile 0's on the
                            # first iteration, so its parity stays runtime.
                            carried_pv = (
                                Fragment(g, kv_compute_index - 1, last_q_tile),
                                carry,
                                None,
                            )
                            _, frag_pv, _ = _tile_fragments(
                                g, kv_compute_index, False, carried_pv
                            )
                            return frag_pv

                        _, tile_zero_pv, _ = pending_pv
                        carry = lax.fori_loop(
                            1, num_live_tiles, _tile_step, tile_zero_pv
                        )
                        pending_pv = (
                            Fragment(g, last_live_tile, last_q_tile),
                            carry,
                            None,
                        )
                if guard_recovery:
                    _block_pv(*pending_pv)
                else:
                    _carry_out_pv(*pending_pv)
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
                    needs_mask=False,
                    guard_recovery=guard_recovery,
                    apply_bounds=apply_bounds,
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
                if guard_recovery:
                    _block_pv(_resolve_g(last), last_pv)
                else:
                    # Note (david): this copy runs every fragment, so its last
                    # fragment is the static pending one.
                    _carry_out_pv(
                        pending_frag, last_pv, num_k_compute_tiles == 1
                    )

        def _run_group(apply_bounds, skips_dead_tiles=False):
            # Note (david): the unpack (or K rotation) leads the copy in program
            # order, where the scheduler overlaps it with the pending PV, which
            # reads only scratch.
            _, staged_kv_head, kv_load_end, kv_used_end, staged_kv = block_desc
            unpack_paged_block(
                traits,
                kv_refs,
                kv_slot,
                staged_kv_head,
                kv_load_end,
                kv_used_end,
                staged_kv,
            )
            # Note (david): a block whose DMA inputs equal the previous block's
            # holds the same staged data, so it runs from the previous block's
            # slot; dense GQA at one kv block per row otherwise re-staged each
            # block once per q head (v6e, 32:4 d256, 1024 tokens: 63.4 against
            # 60.6 us). Paged blocks start their DMAs inside the copy and a
            # rotary K is rotated in place, so both load every block.
            # Note (david): a path that loads every block starts the block two
            # ahead inside the copy, where the page DMAs' scalar work rides the
            # copy's idle scalar slots.
            _start_block_data(ahead_desc, ahead_slot)
            _run_schedule(apply_bounds, skips_dead_tiles)
            for g in range(head_fold):

                @pl.when(guard_smem[g] > 0)
                def _recover(g=g):
                    _run_schedule(apply_bounds, skips_dead_tiles, static_g=g)
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

        return next_slot, ahead_slot, next_block_desc, ahead_desc

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
    second_slot = jnp.int32(1)
    _start_block_data(second_desc, second_slot)

    minus_one = jnp.int32(-1)
    zero = jnp.int32(0)
    init_carry = (
        minus_one,
        minus_one,
        zero,
        (zero, second_slot, first_desc, second_desc),
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
    last_slot, after_last_slot, last_desc, after_last_desc = kv_carry

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

    _wait_block_data(last_desc, last_slot)
    _wait_block_data(after_last_desc, after_last_slot)

    drain(
        num_out_copies,
        num_stages,
        lambda slot: o_copy(traits, out_refs, 0, 0, slot).wait(),
    )

    fill_packed_tail(
        traits, variant, out_refs, schedule.packed_q_end, num_head_groups
    )


def vmem_buffer_bytes(shape: tuple[int, ...], dtype: jnp.dtype) -> int:
    """Bytes of one VMEM scratch buffer under Mosaic's v6 memref tiling.

    A second-minor axis shorter than one large tile pads only to a power of two,
    so a (2, head_dim) bf16 head pair costs no pad.
    """
    itemsize = jnp.dtype(dtype).itemsize
    *leading, rows, lanes = shape
    packing = F32_BYTES // itemsize
    large_tile_rows = NUM_SUBLANES * packing
    if rows >= large_tile_rows:
        tile_rows = large_tile_rows
    else:
        tile_rows = max(packing, next_pow2(min(rows, NUM_SUBLANES)))
    return (
        math.prod(leading)
        * round_up(rows, tile_rows)
        * round_up(lanes, NUM_LANES)
        * itemsize
    )


def paged_scratch_shapes(
    block_sizes: BlockSizes,
    *,
    head_dim: int,
    return_lse: bool,
    rotary_dtype: jnp.dtype | None,
    gqa_fold: int,
    kv_dtype: jnp.dtype,
) -> list[pl.MemoryRef]:
    """Every scratch buffer of one paged build, in the order the kernel
    unpacks them: Q stage, (K, V) word staging, K and V planes, bounds rows
    per kv compute tile, Q rotary coefficients, out stage, row max, row sum,
    f32 accumulator, pending probs, pending V tile, the SMEM guard flags, block
    parity and pending-PV fields, the DMA semaphores and the lse stage.

    The kernel allocates them all, and estimate_vmem_bytes sums the VMEM
    ones. The Q, out, plane and pending V buffers are bf16, q's dtype
    (check_paged_inputs)."""
    bq, bkv = block_sizes.block_q, block_sizes.block_kv
    bkv_compute = block_sizes.block_kv_compute
    num_stages = block_sizes.num_stages
    bf16, f32 = jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)
    vmem, smem = pltpu.VMEM, pltpu.SMEM
    if gqa_fold > 1:
        stage_shape = (num_stages, gqa_fold, bq, head_dim)
    else:
        stage_shape = (num_stages, bq, head_dim)
    group_bq = gqa_fold * bq
    if rotary_dtype is None:
        num_rotary_streams = 0
        rotary_buffers = []
    else:
        num_rotary_streams = 1
        # Note (david): prepare_rotary's one batch plane is shared by every
        # head, so one Q coefficient group is staged.
        rotary_buffers = [
            vmem((num_stages, 1, bq, head_dim), jnp.dtype(rotary_dtype))
        ]
    if return_lse:
        lse_buffers = [vmem((num_stages, bq, NUM_LANES), f32)]
    else:
        lse_buffers = []
    # Note (david): a paged block stages the u32 word rows holding one kv
    # head's K and V, then unpacks them into one bf16 K and one bf16 V
    # plane; a word's (kv_word_rows, head_dim) rows pad to nothing, so the
    # staging costs what two dense planes of the cache dtype would.
    kv_word_buffers = [
        vmem(
            (NUM_KV_STAGES, bkv, 1, kv_word_rows(kv_dtype), head_dim), kv_dtype
        ),
        vmem((2, bkv, head_dim), bf16),
    ]
    # Note (david): the per-seq kernel synthesizes its bounds rows (the q
    # lower bound, plus the span under a left window) once per block
    # in-kernel, so it needs one block's rows, not a kv-stream ring. Both
    # rows share one (8, 128) int32 tile, so the span row costs no extra
    # VMEM.
    bounds_buffer = vmem((bkv // bkv_compute, 2, bkv_compute), jnp.int32)
    return [
        vmem(stage_shape, bf16),
        *kv_word_buffers,
        bounds_buffer,
        *rotary_buffers,
        vmem(stage_shape, bf16),
        vmem((group_bq, NUM_LANES), f32),
        vmem((2, group_bq, NUM_LANES), f32),
        vmem((NUM_O_SLOTS, group_bq, head_dim), f32),
        vmem((gqa_fold * block_sizes.block_q_compute, bkv_compute), f32),
        vmem((bkv_compute, head_dim), bf16),
        smem((PAGED_HEAD_FOLD,), jnp.int32),
        smem((1,), jnp.int32),
        smem((NUM_PENDING_FIELDS,), jnp.int32),
        pltpu.SemaphoreType.DMA(
            (NUM_SEMS + num_rotary_streams, max(num_stages, NUM_KV_STAGES))
        ),
        *lse_buffers,
    ]


def estimate_vmem_bytes(
    block_sizes: BlockSizes,
    *,
    head_dim: int,
    return_lse: bool,
    rotary_dtype: jnp.dtype | None,
    gqa_fold: int,
    kv_dtype: jnp.dtype,
) -> int:
    """Scoped VMEM bytes of one paged build: its VMEM scratch plus the score
    temporaries of one compute fragment."""
    scratch_bytes = sum(
        vmem_buffer_bytes(buffer.shape, buffer.dtype)
        for buffer in paged_scratch_shapes(
            block_sizes,
            head_dim=head_dim,
            return_lse=return_lse,
            rotary_dtype=rotary_dtype,
            gqa_fold=gqa_fold,
            kv_dtype=kv_dtype,
        )
        if buffer.memory_space == pltpu.VMEM
    )
    temporary_bytes = (
        gqa_fold
        * block_sizes.block_q_compute
        * block_sizes.block_kv_compute
        * SCORE_TEMPORARY_BYTES
    )
    return scratch_bytes + temporary_bytes


def check_paged_inputs(
    q: jax.Array,
    kv_cache: jax.Array,
    cu_seqlens_q: jax.Array,
    seqused_k: jax.Array,
    block_table: jax.Array,
    sinks: jax.Array | None,
    *,
    nheads_k: int | None,
    block_sizes: BlockSizes | None,
) -> int:
    """Check flash_attn_varlen_paged's operands, and a pinned block_sizes,
    before dispatch picks the build's tiles; returns the pool's kv head count.

    The arguments are flash_attn_varlen_paged's; block_sizes is the caller's
    pinned one, None when dispatch picks it.
    """
    num_q_heads, total_q, head_dim = q.shape
    if q.dtype != jnp.bfloat16:
        raise NotImplementedError(
            f"paged attention takes a bfloat16 q; got {q.dtype}."
        )
    if kv_cache.ndim != 4:
        raise ValueError(
            "kv_cache must be one merged (num_pages, page_size, cache_heads,"
            f" head_dim) cache; got {kv_cache.shape}."
        )
    _, page_size, cache_heads, cache_head_dim = kv_cache.shape
    num_kv_heads = resolve_merged_kv_heads(
        cache_heads, kv_cache.dtype, nheads_k
    )
    if cache_head_dim != head_dim or head_dim % NUM_LANES:
        raise ValueError(
            f"head_dim must be a multiple of {NUM_LANES} shared by q and the"
            f" cache; got q {head_dim}, cache {cache_head_dim}."
        )
    # Note (david): each page is its own DMA, so smaller pages would leave the
    # kv stream descriptor-bound.
    if page_size % PAGE_SIZE_MULTIPLE:
        raise ValueError(
            f"page_size must be a multiple of {PAGE_SIZE_MULTIPLE}; got"
            f" {page_size}."
        )
    if num_q_heads % num_kv_heads:
        raise ValueError(
            f"nheads={num_q_heads} must be a multiple of"
            f" nheads_k={num_kv_heads}."
        )
    if total_q == 0:
        raise ValueError("q needs at least one token row.")
    if sinks is not None and sinks.shape != (num_q_heads,):
        raise ValueError(
            f"sinks must be ({num_q_heads},), one per q head; got"
            f" {sinks.shape}."
        )

    cu_seqlens_q = jnp.asarray(cu_seqlens_q)
    seqused_k = jnp.asarray(seqused_k)
    block_table = jnp.asarray(block_table)
    batch = cu_seqlens_q.shape[0] - 1
    if seqused_k.shape != (batch,) or not jnp.issubdtype(
        seqused_k.dtype, jnp.integer
    ):
        raise ValueError(
            f"seqused_k must be a ({batch},) integer array; got"
            f" {seqused_k.shape} {seqused_k.dtype}."
        )
    if (
        block_table.ndim != 2
        or block_table.shape[0] != batch
        or block_table.shape[1] == 0
        or not jnp.issubdtype(block_table.dtype, jnp.integer)
    ):
        raise ValueError(
            f"block_table must be a (batch={batch}, max_pages_per_seq >= 1)"
            f" integer array; got {block_table.shape} {block_table.dtype}."
        )

    if block_sizes is not None and block_sizes.block_kv % page_size:
        raise ValueError(
            f"block_kv={block_sizes.block_kv} must be whole pages of"
            f" page_size={page_size}."
        )
    if block_sizes is not None and (
        block_sizes.block_q % block_sizes.block_q_compute
        or block_sizes.block_q // block_sizes.block_q_compute
        < MIN_Q_TILES_PER_BLOCK
    ):
        raise ValueError(
            "the static-anchor softmax needs block_q to hold >= 2 whole q"
            f" compute tiles; got block_q={block_sizes.block_q},"
            f" block_q_compute={block_sizes.block_q_compute}."
        )
    return num_kv_heads


def paged_pallas_call(
    smem_operands: list[jax.Array],
    q: jax.Array,
    kv_cache: jax.Array,
    traits: PagedFwdTraits,
    variant: Variant,
    *,
    layout: ScheduleTableLayout,
    interpret: bool,
    rotary: tuple[jax.Array, None] | None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Run the paged kernel over q, its rows padded to whole q blocks.

    smem_operands are the kernel's SMEM inputs (schedule table, flat block
    table, K and V descales, then the sink logits under has_sinks), and
    rotary the padded (q coefficients, None) pair traits' rotary fields
    describe. Returns o shaped like q, or (o, lse) with a (num_q_heads,
    q_seq_len) float32 natural-log lse.
    """
    num_q_heads, q_seq_len, _ = q.shape
    assert q.dtype == jnp.bfloat16, q.dtype
    assert (rotary is None) == (traits.rotary_dim == 0), traits.rotary_dim
    if rotary is not None:
        _, rotary_k_coeff = rotary
        assert rotary_k_coeff is None and not traits.rotary_k, traits
    # A paged build rotates Q only, so the check never reads the kv length and
    # q_seq_len stands in for it.
    rotary_dtype, rotary_operands = check_rotary_operands(
        rotary, traits, q.dtype, q_seq_len, q_seq_len
    )
    scratch_shapes = paged_scratch_shapes(
        traits.block_sizes,
        head_dim=traits.head_dim_qk,
        return_lse=variant.return_lse,
        rotary_dtype=rotary_dtype,
        gqa_fold=traits.gqa_fold,
        kv_dtype=kv_cache.dtype,
    )
    # Note (david): the pool is pinned to HBM because memory space
    # assignment otherwise moves a pool that fits its VMEM budget there and
    # back around every call, the whole pool however few pages the call
    # reads (v6e, 16 MB e5m2 pool: 0.369 against 0.339 ms a chunk). The spec
    # stays pl.ANY: a layout copy XLA inserts before the call escapes the
    # constraint and can land in VMEM, where DMAs compiled for an HBM
    # operand halt the chip.
    kernel_inputs = [
        q,
        pltpu.with_memory_space_constraint(kv_cache, pltpu.HBM),
        *rotary_operands,
    ]
    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    in_specs = [smem_spec] * len(smem_operands) + [any_spec] * len(
        kernel_inputs
    )
    o_shape = jax.ShapeDtypeStruct(
        (num_q_heads, q_seq_len, traits.head_dim_v), q.dtype
    )
    # Note (david): per-seq LSE gives every head group an explicit minor axis of
    # whole 128-lane tiles, so the kernel DMAs complete tiles at offset zero
    # and groups never alias in the physical layout; only the first
    # group_heads lanes are returned.
    group_heads = traits.group_heads
    lse_shape = jax.ShapeDtypeStruct(
        (num_q_heads // group_heads, q_seq_len, NUM_LANES), jnp.float32
    )
    if variant.return_lse:
        out_specs = [any_spec, any_spec]
        out_shape = [o_shape, lse_shape]
    else:
        out_specs = any_spec
        out_shape = o_shape

    kernel_name = "flash_attn_varlen_paged_fwd"
    with jax.named_scope(kernel_name):
        outputs = pl.pallas_call(
            partial(
                flash_fwd_varlen_paged_kernel,
                traits=traits,
                variant=variant,
                layout=layout,
            ),
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
    if not variant.return_lse:
        return outputs
    o, lse_kernel = outputs
    return o, lse_kernel[:, :, :group_heads].swapaxes(1, 2).reshape(
        num_q_heads, q_seq_len
    )


def flash_attn_varlen_paged(
    q: jax.Array,
    kv_cache: jax.Array,
    cu_seqlens_q: jax.Array,
    seqused_k: jax.Array,
    block_table: jax.Array,
    sinks: jax.Array | None = None,
    *,
    block_sizes: BlockSizes,
    variant: Variant,
    interpret: bool,
    rotary: tuple[jax.Array, None] | None = None,
    rotary_interleaved: bool = False,
    nheads_k: int | None = None,
    k_scale: float | jax.Array = 1.0,
    v_scale: float | jax.Array = 1.0,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Packed head-major attention over a paged KV cache, read-only.

    q is (nheads, total_q, head_dim) bf16 packed by cu_seqlens_q, whose first
    entry is 0. kv_cache is the merged (num_pages, page_size,
    merged_cache_heads(nheads_k), head_dim) pool, bf16 or fp8, whose token rows
    interleave the kv heads' K and V rows (kv_layout's merged order); nheads_k
    defaults to half its head axis, and an fp8 pool, which pads an odd head
    count, needs it. An fp8 pool is widened to bf16 once per staged block;
    k_scale multiplies every score and v_scale the output, both scalars that
    may be traced (variant.has_kv_scales). seqused_k is each sequence's kv
    length with its new tokens, which the cache must already hold; a
    block_table entry past those tokens' pages is only read through a clamp
    into the pool, and the rows it names are discarded.
    variant's mask aligns to the end of each sequence's keys, with a right
    bound of 0 (causal) or none, and no causal_offset; a left bound also hides
    the keys more than that many positions before that diagonal. Rows past
    cu_seqlens_q[-1] are out = 0, lse = -inf. rotary is prepare_rotary's (q
    coefficients, None) pair. sinks (None or (nheads,) float) adds
    exp(sinks[h]) to head h's softmax denominator, sink logits on the scale of
    the softcapped scores (gpt-oss attention sinks).

    The caller runs check_paged_inputs on the operands first, then takes
    block_sizes from dispatch.resolve_paged_block_sizes.
    """
    variant.check_paged_window()
    window_left, window_right = variant.window
    if variant.causal_offset or variant.runtime_causal_offset:
        raise ValueError(
            "a paged forward aligns its mask to each sequence's own keys; a"
            " kernel-wide causal_offset would be applied on top of it."
        )
    if not variant.has_kv_scales:
        raise ValueError(
            "the paged kernel always reads K and V descales; build its Variant"
            " with has_kv_scales=True."
        )
    variant.check_sinks_operand(sinks)
    causal = window_right == 0
    num_q_heads, total_q, head_dim = q.shape
    _, page_size, cache_heads, _ = kv_cache.shape
    num_kv_heads = resolve_merged_kv_heads(
        cache_heads, kv_cache.dtype, nheads_k
    )
    cu_seqlens_q = jnp.asarray(cu_seqlens_q)
    seqused_k = jnp.asarray(seqused_k)
    block_table = jnp.asarray(block_table)
    pages_per_seq = block_table.shape[1]
    capacity = pages_per_seq * page_size
    gqa_fold = num_q_heads // num_kv_heads
    bq = block_sizes.block_q
    paged = PagedKVInfo(
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        is_bitcast_load=not interpret,
    )

    padded_total_q = round_up(total_q, bq)
    num_pad_q = padded_total_q - total_q
    padded_q = jnp.pad(q, ((0, 0), (0, num_pad_q), (0, 0)))
    if rotary is None:
        padded_rotary = None
    else:
        padded_rotary = (
            jnp.pad(rotary[0], ((0, 0), (0, num_pad_q), (0, 0))),
            None,
        )

    # Note (david): resolving the schedule once per call spares every q block
    # a search of the cu_seqlens and its window arithmetic.
    table, layout = paged_schedule_table(
        cu_seqlens_q,
        seqused_k,
        padded_total_q=padded_total_q,
        bq=bq,
        bkv=block_sizes.block_kv,
        bkv_compute=block_sizes.block_kv_compute,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        causal=causal,
        left=window_left,
    )
    traits = PagedFwdTraits(
        block_sizes=block_sizes,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        head_dim_v=head_dim,
        head_fold=PAGED_HEAD_FOLD,
        transposed_pv=False,
        token_major=None,
        is_per_seq=True,
        guard_threshold=overflow_guard_threshold(capacity),
        **rotary_fields(padded_rotary, rotary_interleaved),
        paged=paged,
        gqa_fold=gqa_fold,
    )
    smem_operands = [
        table,
        block_table.astype(jnp.int32).reshape(-1),
        jnp.stack([
            jnp.asarray(scale, jnp.float32).reshape(())
            for scale in (k_scale, v_scale)
        ]),
    ]
    if sinks is not None:
        smem_operands.append(variant.sink_logits(sinks))
    outputs = paged_pallas_call(
        smem_operands,
        padded_q,
        kv_cache,
        traits,
        # Note (david): as in make_flash_attn_mha, Q carries softmax_scale /
        # softcap, so log2(e) rides in the tanh output scale.
        variant,
        layout=layout,
        interpret=interpret,
        rotary=padded_rotary,
    )
    if variant.return_lse:
        out, lse = outputs
        return out[:, :total_q], lse[:, :total_q]
    else:
        return outputs[:, :total_q]
