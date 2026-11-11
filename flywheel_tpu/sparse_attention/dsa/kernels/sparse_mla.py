"""TensorCore attention of each query token over its own gathered top-k
latent rows: MQA with one key head of the latent and the rope, the latent as
the value."""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from flywheel_tpu.core.tpu import NUM_LANES, NUM_SUBLANES
from flywheel_tpu.softmax_attention.config import (
    LOG2E,
    NT_DIM_NUMBERS,
    vmem_limit_bytes,
)
from flywheel_tpu.sparse_attention.dsa.kernels.gather import (
    BYTE_BITS,
    CHUNK_SLOTS,
    FLAG_ROW_SLOTS,
    HALF_BITS,
    NOPE_WORDS,
    QUAD_TOKENS,
    ROPE_BLOCK_SLOTS,
    ROPE_HALF_SLOTS,
    ROPE_ROW_SLOTS,
)

FP8_DTYPE = jnp.dtype(jnp.float8_e4m3fn)
# Note (david): the max is taken over all of a token's keys first, so no key
# block rescales another and each block's P @ V overlaps the next block's exp.
NUM_KEY_BLOCKS = 4
# Note (david): a step attends its tokens in one basic block, so the MXU
# streams one token's matmuls while the vector units take another's softmax.
MAX_TOKENS_PER_STEP = 4
# Note (david): an unused slot's score is pushed below any real score by far
# more than exp2 resolves, so its probability is exactly 0.
UNUSED_SCORE = -(2.0**72)
# Note (david): a token whose max score, in q . k units, is under this saw no
# real key; real scores stay within 576 * 448 * |q|.
NO_KEY_SCORE = -(2.0**60)
# Note (david): a step attends the first slots of its tokens: all top_k in one
# basic block, or up to MAX_BLOCKED_BLOCKS blocks of BLOCK_SLOTS with a
# running max. More blocks would cost more rescaling than they save.
BLOCK_SLOTS = 2 * CHUNK_SLOTS
MAX_BLOCKED_BLOCKS = 3
# Halves of a rope block, each scored against rope queries of its own.
ROPE_HALVES = ROPE_BLOCK_SLOTS // ROPE_HALF_SLOTS
# Note (david): the rings of NUM_STAGES slots keep NUM_STAGES steps' copies in
# flight; a step's copies start once the step NUM_STAGES before it has read
# their slot.
NUM_STAGES = 4
# Note (david): the padding steps' zero rows go out from a slot of their own,
# while the last attended steps' rows still copy out of the ring.
PADDING_SLOT = NUM_STAGES
# Words of the attention's SMEM scalars: the f32 bits of the cache's scale,
# then the slots of each step of the call.
KV_SCALE_WORD = 0
FIRST_SLOTS_WORD = 1


def max_blocked_slots(top_k: int) -> int:
    """The most slots a step attends a block at a time; a step with more
    attends all top_k."""
    return min(MAX_BLOCKED_BLOCKS * BLOCK_SLOTS, top_k - BLOCK_SLOTS)


def attention_scalars(kv_scale: jax.Array, step_slots: jax.Array) -> jax.Array:
    """The attention's (1 + T / step tokens,) int32 SMEM scalars from the
    (1,) f32 cache scale and each step's slots."""
    return jnp.concatenate(
        [lax.bitcast_convert_type(kv_scale, jnp.int32), step_slots]
    )


def dsa_sparse_mla_kernel(
    first_token_ref: jax.Array,
    scalars_hbm: jax.Array,
    cu_q_lens_hbm: jax.Array,
    q_nope_hbm: jax.Array,
    q_pe_hbm: jax.Array,
    nope_hbm: jax.Array,
    rope_hbm: jax.Array,
    flag_hbm: jax.Array,
    *refs: jax.Array,
    num_previous: int,
    num_steps: int,
    sm_scale_log2: float,
    return_lse: bool,
) -> None:
    """Attend the chunk's steps that hold request tokens, each over its first
    slots of gathered rows, then write the zero outputs of its steps of
    padding.

    refs holds the earlier chunks' operands, which only carry their rows
    through and are skipped, then out_hbm, [lse_hbm], the rings of the
    gathered nope, rope and flag rows, of the queries and of out and [lse],
    the scalars and cu_q_lens in SMEM and the DMA semaphores.
    """
    refs = refs[num_previous:]
    if return_lse:
        out_hbm, lse_hbm, *refs = refs
        *rings, lse_ring, scalars_ref, cu_q_lens_ref, sems = refs
    else:
        out_hbm, *refs = refs
        *rings, scalars_ref, cu_q_lens_ref, sems = refs
        lse_hbm = lse_ring = None
    nope_ring, rope_ring, flag_ring, q_nope_ring, q_pe_ring, out_ring = rings
    step_tokens, _, num_heads = q_pe_ring.shape[1:]
    step_rows = nope_ring.shape[1]
    top_k = step_rows // step_tokens
    # Note (david): the rings of token blocks come in blocks of NUM_SUBLANES
    # rows, see token_ring; the kernel reads them as token blocks.
    token_block = (step_tokens, num_heads, q_nope_hbm.shape[2])
    q_nope_ring = q_nope_ring.reshape(-1, *token_block)
    out_ring = out_ring.reshape(-1, *token_block)
    first_step = first_token_ref[0] // step_tokens

    def _input_copies(step, slot):
        """The copies of the chunk's step's gathered rows and queries into
        the ring slot."""
        # Note (david): a step's copies are whole, so the rows of its padding
        # tokens come in as the gather left them; the step masks those
        # tokens by their place.
        first_row = pl.multiple_of(step * step_rows, CHUNK_SLOTS)
        first_rope_row = pl.multiple_of(
            first_row // ROPE_ROW_SLOTS, CHUNK_SLOTS // ROPE_ROW_SLOTS
        )
        tokens = pl.ds((first_step + step) * step_tokens, step_tokens)
        return (
            pltpu.make_async_copy(
                nope_hbm.at[pl.ds(first_row, step_rows)],
                nope_ring.at[slot],
                sems.at[slot],
            ),
            pltpu.make_async_copy(
                rope_hbm.at[pl.ds(first_rope_row, step_rows // ROPE_ROW_SLOTS)],
                rope_ring.at[slot],
                sems.at[slot],
            ),
            pltpu.make_async_copy(
                flag_hbm.at[
                    pl.ds(
                        first_row // FLAG_ROW_SLOTS,
                        step_rows // FLAG_ROW_SLOTS,
                    )
                ],
                flag_ring.at[slot],
                sems.at[slot],
            ),
            pltpu.make_async_copy(
                q_nope_hbm.at[tokens], q_nope_ring.at[slot], sems.at[slot]
            ),
            pltpu.make_async_copy(
                q_pe_hbm.at[tokens], q_pe_ring.at[slot], sems.at[slot]
            ),
        )

    def _start(copies):
        for copy in copies:
            copy.start()

    def _wait(copies):
        for copy in copies:
            copy.wait()

    # Note (david): cu_q_lens' last word says which steps are attended; the
    # first step's copies start with the scalars' copies.
    scalar_copies = (
        pltpu.make_async_copy(scalars_hbm, scalars_ref, sems.at[PADDING_SLOT]),
        pltpu.make_async_copy(
            cu_q_lens_hbm, cu_q_lens_ref, sems.at[PADDING_SLOT]
        ),
    )
    _start(scalar_copies)
    _start(_input_copies(0, 0))
    _wait(scalar_copies)
    num_request_tokens = cu_q_lens_ref[cu_q_lens_ref.shape[0] - 1]
    # Note (david): the request tokens are the call's first rows, so the
    # chunk's steps that hold one, the attended ones, are its first.
    num_attended = jnp.clip(
        pl.cdiv(num_request_tokens, step_tokens) - first_step, 0, num_steps
    )

    def _request_tokens(step):
        """Tokens of the chunk's attended step before the call's padding
        rows."""
        return jnp.minimum(
            num_request_tokens - (first_step + step) * step_tokens, step_tokens
        )

    def _output_copies(step, slot):
        """The copies of the ring slot's out and lse rows to the chunk's
        step."""
        copies = [
            pltpu.make_async_copy(
                out_ring.at[slot],
                out_hbm.at[
                    pl.ds((first_step + step) * step_tokens, step_tokens)
                ],
                sems.at[slot],
            )
        ]
        if return_lse:
            copies.append(
                pltpu.make_async_copy(
                    lse_ring.at[slot],
                    lse_hbm.at[pl.ds(first_step + step, 1)],
                    sems.at[slot],
                )
            )
        return copies

    def _attend(step, carry):
        slot = lax.rem(step, NUM_STAGES)
        _wait(_input_copies(step, slot))
        # Note (david): the step writes the slot's out rows, which the step
        # NUM_STAGES before it may still copy out.
        pl.when(step >= NUM_STAGES)(
            lambda: _wait(_output_copies(step - NUM_STAGES, slot))
        )
        kv_scale = lax.bitcast_convert_type(
            scalars_ref[KV_SCALE_WORD], jnp.float32
        )
        slots = scalars_ref[FIRST_SLOTS_WORD + first_step + step]
        operands = (
            kv_scale,
            q_nope_ring.at[slot],
            q_pe_ring.at[slot],
            nope_ring.at[slot],
            rope_ring.at[slot],
            flag_ring.at[slot],
            out_ring.at[slot],
            None if lse_ring is None else lse_ring.at[slot],
            _request_tokens(step),
        )

        @pl.when(slots == top_k)
        def _all_slots():
            _attend_step(*operands, sm_scale_log2=sm_scale_log2)

        @pl.when(slots < top_k)
        def _some_slots():
            # Note (david): the blocked body's scratch lives in its branch
            # alone, so the full body keeps all of the VMEM it had.
            stats = pltpu.VMEM((step_tokens, num_heads, 1), jnp.float32)
            pl.run_scoped(
                lambda *scratch: _attend_blocks(
                    *operands,
                    *scratch,
                    slots // BLOCK_SLOTS,
                    sm_scale_log2=sm_scale_log2,
                ),
                pltpu.VMEM(
                    (step_tokens, ROPE_HALVES, num_heads, NUM_LANES),
                    q_pe_ring.dtype,
                ),
                stats,
                stats,
                pltpu.VMEM(token_block, jnp.float32),
            )

        _start(_output_copies(step, slot))
        # Note (david): the copies overwrite the slot the step has just read,
        # so they start after its last load.
        pl.when(step + NUM_STAGES < num_attended)(
            lambda: _start(_input_copies(step + NUM_STAGES, slot))
        )
        return carry

    def _copy_padding(copy_action, step, carry):
        copy_action(_output_copies(step, PADDING_SLOT))
        return carry

    def _for_padding_steps(copy_action):
        lax.fori_loop(
            num_attended,
            num_steps,
            functools.partial(_copy_padding, copy_action),
            0,
        )

    num_first_steps = min(NUM_STAGES, num_steps)
    for step in range(1, num_first_steps):
        pl.when(step < num_attended)(
            lambda step=step: _start(_input_copies(step, step))
        )
    lax.fori_loop(0, num_attended, _attend, 0)

    @pl.when(num_attended < num_steps)
    def _start_padding():
        _write_no_keys(
            out_ring.at[PADDING_SLOT],
            None if lse_ring is None else lse_ring.at[PADDING_SLOT],
        )
        _for_padding_steps(_start)

    # Note (david): a kernel must not exit with DMAs in flight; a chunk of
    # padding alone has not waited for its first step's.
    pl.when(num_attended == 0)(lambda: _wait(_input_copies(0, 0)))
    for age in range(1, num_first_steps + 1):
        last_step = num_attended - age
        pl.when(last_step >= 0)(
            lambda last_step=last_step: _wait(
                _output_copies(last_step, lax.rem(last_step, NUM_STAGES))
            )
        )
    pl.when(num_attended < num_steps)(lambda: _for_padding_steps(_wait))


def _attend_step(
    kv_scale: jax.Array,
    q_nope_ref: jax.Array,
    q_pe_ref: jax.Array,
    nope_ref: jax.Array,
    rope_ref: jax.Array,
    flag_ref: jax.Array,
    out_ref: jax.Array,
    lse_ref: jax.Array | None,
    request_tokens: jax.Array,
    *,
    sm_scale_log2: float,
) -> None:
    """Attend the step's tokens, each over all its gathered rows, all in one
    basic block; the tokens from request_tokens on are padding.

    Every token's scores come first and its softmax and P @ V after, so the
    list scheduler sees all of the step's matmuls before any softmax waits on
    them.
    """
    step_tokens = q_nope_ref.shape[0]
    top_k = nope_ref.shape[0] // step_tokens
    score_scale = kv_scale * sm_scale_log2
    staged = [
        _token_scores(
            q_nope_ref,
            _rope_queries(q_pe_ref[token]),
            nope_ref,
            rope_ref,
            flag_ref,
            token,
            token * top_k,
            top_k,
        )
        for token in range(step_tokens)
    ]
    for token, (keys, scores) in enumerate(staged):
        row_max, row_sum, accumulator = _softmax_pv(scores, keys, score_scale)
        _write_token(
            out_ref,
            lse_ref,
            token,
            row_max,
            row_sum,
            accumulator,
            kv_scale,
            _no_key_bound(token, request_tokens, score_scale),
        )


def _attend_blocks(
    kv_scale: jax.Array,
    q_nope_ref: jax.Array,
    q_pe_ref: jax.Array,
    nope_ref: jax.Array,
    rope_ref: jax.Array,
    flag_ref: jax.Array,
    out_ref: jax.Array,
    lse_ref: jax.Array | None,
    request_tokens: jax.Array,
    rope_query_ref: jax.Array,
    max_ref: jax.Array,
    sum_ref: jax.Array,
    accumulator_ref: jax.Array,
    num_blocks: jax.Array,
    *,
    sm_scale_log2: float,
) -> None:
    """Attend the step's tokens, each over its first num_blocks *
    BLOCK_SLOTS gathered rows, a block of every token per basic block: each
    block rescales the running sum and P @ V of the blocks before it to the
    running max. The tokens from request_tokens on are padding."""
    step_tokens = q_nope_ref.shape[0]
    top_k = nope_ref.shape[0] // step_tokens
    score_scale = kv_scale * sm_scale_log2
    for token in range(step_tokens):
        for half, query in enumerate(_rope_queries(q_pe_ref[token])):
            rope_query_ref[token, half] = query
    max_ref[...] = jnp.full(max_ref.shape, -jnp.inf, max_ref.dtype)
    sum_ref[...] = jnp.zeros(sum_ref.shape, sum_ref.dtype)
    accumulator_ref[...] = jnp.zeros(
        accumulator_ref.shape, accumulator_ref.dtype
    )

    @pl.loop(0, num_blocks)
    def _block(block):
        first_slot = pl.multiple_of(block * BLOCK_SLOTS, BLOCK_SLOTS)
        staged = [
            _token_scores(
                q_nope_ref,
                tuple(
                    rope_query_ref[token, half] for half in range(ROPE_HALVES)
                ),
                nope_ref,
                rope_ref,
                flag_ref,
                token,
                token * top_k + first_slot,
                BLOCK_SLOTS,
            )
            for token in range(step_tokens)
        ]
        for token, (keys, scores) in enumerate(staged):
            running_max = max_ref[token]
            new_max = jnp.maximum(
                running_max, scores.max(axis=1, keepdims=True) * score_scale
            )
            rescale = jnp.exp2(running_max - new_max)
            probabilities = jnp.exp2(scores * score_scale - new_max)
            # Note (david): the probabilities round to the cache's fp8, as in
            # _softmax_pv; shifted by the running max, none is above 1.
            pv = jnp.dot(
                probabilities.astype(FP8_DTYPE),
                keys,
                preferred_element_type=jnp.float32,
            )
            max_ref[token] = new_max
            sum_ref[token] = sum_ref[token] * rescale + probabilities.sum(
                axis=1, keepdims=True
            )
            accumulator_ref[token] = accumulator_ref[token] * rescale + pv

    for token in range(step_tokens):
        _write_token(
            out_ref,
            lse_ref,
            token,
            max_ref[token],
            sum_ref[token],
            accumulator_ref[token],
            kv_scale,
            _no_key_bound(token, request_tokens, score_scale),
        )


def _write_token(
    out_ref: jax.Array,
    lse_ref: jax.Array | None,
    token: int,
    row_max: jax.Array,
    row_sum: jax.Array,
    accumulator: jax.Array,
    kv_scale: jax.Array,
    no_key_bound: jax.Array,
) -> None:
    """The step token's output and lse row from its (H, 1) max and sum in
    score_scale log2 units and its (H, LATENT_DIM) P @ V."""
    # Note (david): a token that sees no key gets a zero output and an lse of
    # +inf whatever its scores hold.
    has_keys = row_max > no_key_bound
    inverse_row_sum = jnp.where(has_keys, kv_scale / row_sum, 0.0)
    out_ref[token] = jnp.where(
        has_keys, accumulator * inverse_row_sum, 0.0
    ).astype(out_ref.dtype)
    if lse_ref is not None:
        lse = jnp.where(
            has_keys, (row_max + jnp.log2(row_sum)) * math.log(2.0), jnp.inf
        )
        lse_ref[0, pl.ds(token, 1)] = _lane_row(lse)


def _no_key_bound(
    token: int, request_tokens: jax.Array, score_scale: jax.Array
) -> jax.Array:
    """The score, in score_scale log2 units, that the max of a token that saw
    a key passes: none does for a padding token, whose rows hold what the
    gather left in them."""
    return jnp.where(
        token < request_tokens, NO_KEY_SCORE * score_scale, jnp.inf
    )


def _write_no_keys(out_ref: jax.Array, lse_ref: jax.Array | None) -> None:
    """A step of padding tokens: zero outputs and lse rows of +inf."""
    out_ref[...] = jnp.zeros(out_ref.shape, out_ref.dtype)
    if lse_ref is not None:
        lse_ref[...] = jnp.full(lse_ref.shape, jnp.inf, lse_ref.dtype)


def _token_scores(
    q_nope_ref: jax.Array,
    rope_queries: tuple[jax.Array, jax.Array],
    nope_ref: jax.Array,
    rope_ref: jax.Array,
    flag_ref: jax.Array,
    token: int,
    first_row: int | jax.Array,
    num_keys: int,
) -> tuple[jax.Array, jax.Array]:
    """The step token's keys (num_keys, LATENT_DIM) fp8 of the step's
    gathered rows from first_row on and its q . k scores (H, num_keys) f32,
    an unused slot's near UNUSED_SCORE.

    Each rope block's halves are scored apart against their own rope
    queries, then interleaved back into slot order.
    """
    keys, rope_keys = _token_keys(nope_ref, rope_ref, first_row, num_keys)
    half_scores = [
        lax.dot_general(
            q_nope_ref[token],
            _half_keys(keys, half),
            NT_DIM_NUMBERS,
            preferred_element_type=jnp.float32,
        )
        + lax.dot_general(
            rope_query,
            rope_keys,
            NT_DIM_NUMBERS,
            preferred_element_type=jnp.float32,
        )
        for half, rope_query in enumerate(rope_queries)
    ]
    return keys, _slot_scores(half_scores, flag_ref, first_row)


def _half_keys(keys: jax.Array, half: int) -> jax.Array:
    """The keys of half `half` of each rope block of keys, block after
    block."""
    return jnp.concatenate(
        [
            keys[first : first + ROPE_HALF_SLOTS]
            for first in range(
                half * ROPE_HALF_SLOTS, keys.shape[0], ROPE_BLOCK_SLOTS
            )
        ],
        axis=0,
    )


def _slot_scores(
    half_scores: list[jax.Array],
    flag_ref: jax.Array,
    first_row: int | jax.Array,
) -> jax.Array:
    """The (H, num_keys) scores of the step's slots from first_row on, in
    slot order, from the scores of each rope block half, an unused slot's
    plus UNUSED_SCORE."""
    num_pieces = ROPE_HALVES * half_scores[0].shape[1] // ROPE_HALF_SLOTS
    first_flag_row = first_row // FLAG_ROW_SLOTS
    pieces = []
    for piece in range(num_pieces):
        block, half = divmod(piece, ROPE_HALVES)
        first_key = block * ROPE_HALF_SLOTS
        flags = flag_ref[pl.ds(first_flag_row + piece, 1)]
        pieces.append(
            half_scores[half][:, first_key : first_key + ROPE_HALF_SLOTS]
            + jnp.where(flags != 0, UNUSED_SCORE, 0.0)
        )
    return jnp.concatenate(pieces, axis=1)


def _token_keys(
    nope_ref: jax.Array,
    rope_ref: jax.Array,
    first_row: int | jax.Array,
    num_keys: int,
) -> tuple[jax.Array, jax.Array]:
    """The num_keys fp8 keys (num_keys, LATENT_DIM) of the step's gathered
    rows from first_row on, which are their values too, and their rope keys.

    Nope rows are int32 banded words (byte b of word w is dim 128 b + w); a
    byte transpose of four stride-4 loads gives the keys' byte planes. Rope
    rows are 4-token words that bitcast to fp8 rows, see ROPE_ROW_SLOTS.
    """
    quarter = num_keys // QUAD_TOKENS
    words = [
        nope_ref[pl.ds(first_row + j, quarter, stride=QUAD_TOKENS)]
        for j in range(QUAD_TOKENS)
    ]
    keys = jnp.concatenate(
        [pltpu.bitcast(plane, FP8_DTYPE) for plane in byte_planes(words)],
        axis=1,
    )
    rope_keys = pltpu.bitcast(
        rope_ref[
            pl.ds(first_row // ROPE_ROW_SLOTS, num_keys // ROPE_ROW_SLOTS)
        ],
        FP8_DTYPE,
    )
    return keys, rope_keys


def byte_planes(words: list[jax.Array]) -> tuple[jax.Array, ...]:
    """Four word arrays -> four byte planes: plane b word byte j is
    words[j]'s byte b, by a 16-bit then an 8-bit pack round."""
    w0, w1, w2, w3 = words
    w0, w2 = _pack_pair(w0, w2, HALF_BITS)
    w1, w3 = _pack_pair(w1, w3, HALF_BITS)
    p0, p1 = _pack_pair(w0, w1, BYTE_BITS)
    p2, p3 = _pack_pair(w2, w3, BYTE_BITS)
    return p0, p1, p2, p3


def _pack_pair(
    x: jax.Array, y: jax.Array, bits: int
) -> tuple[jax.Array, jax.Array]:
    """Interleave x's and y's low and high `bits`-wide fields of each
    2 * bits-wide field: (x low, y low) words and (x high, y high) words."""
    return _pack_low(x, y, bits), _pack_low(x >> bits, y >> bits, bits)


def _pack_low(x: jax.Array, y: jax.Array, bits: int) -> jax.Array:
    """Words whose 2 * bits-wide fields hold x's low `bits` bits, then y's."""
    if bits == HALF_BITS:
        return pltpu.pack_elementwise([x, y], packed_dtype=jnp.int16)
    packed = pltpu.pack_elementwise(
        [pltpu.bitcast(x, jnp.int16), pltpu.bitcast(y, jnp.int16)],
        packed_dtype=jnp.int8,
    )
    return pltpu.bitcast(packed, jnp.uint32)


def _rope_queries(q_pe_t: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The token's (ROPE_DIM, H) rope queries, heads minor, as (H, NUM_LANES)
    rows that match the rope keys' lane tile, for each half of a rope block:
    the rope then zeros, and zeros then the rope."""
    rope_dim, num_heads = q_pe_t.shape
    zeros = jnp.zeros((NUM_LANES - rope_dim, num_heads), jnp.float32)
    padded = jnp.concatenate([q_pe_t.astype(jnp.float32), zeros], axis=0).T
    return tuple(
        query.astype(q_pe_t.dtype)
        for query in (padded, pltpu.roll(padded, rope_dim, 1))
    )


def _softmax_pv(
    scores: jax.Array, values: jax.Array, score_scale: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """(row_max, row_sum, P @ V) of the q . k scores (H, top_k) at
    score_scale log2 units: one exact max over all keys, then the exp, the
    row sum and P @ V a key block at a time."""
    block_keys = scores.shape[1] // NUM_KEY_BLOCKS
    # Note (david): score_scale is positive, so the max of the raw scores
    # scales to the max of the scaled ones.
    row_max = scores.max(axis=1, keepdims=True) * score_scale
    row_sum = None
    accumulator = None
    for block in range(NUM_KEY_BLOCKS):
        keys_in_block = pl.ds(block * block_keys, block_keys)
        probabilities = jnp.exp2(
            scores[:, keys_in_block] * score_scale - row_max
        )
        # Note (david): the probabilities round to the cache's fp8 for P @ V;
        # a max-shifted one is at most 1, so none overflows fp8.
        pv = jnp.dot(
            probabilities.astype(FP8_DTYPE),
            values[keys_in_block],
            preferred_element_type=jnp.float32,
        )
        block_sum = probabilities.sum(axis=1, keepdims=True)
        row_sum = block_sum if row_sum is None else row_sum + block_sum
        accumulator = pv if accumulator is None else accumulator + pv
    return row_max, row_sum, accumulator


def _lane_row(column: jax.Array) -> jax.Array:
    """An (H, 1) column as a lane-dense (1, NUM_LANES) row, lane h its row h
    and zeros past H."""
    num_rows = column.shape[0]
    row = lax.broadcasted_iota(jnp.int32, (num_rows, NUM_LANES), 0)
    lane = lax.broadcasted_iota(jnp.int32, (num_rows, NUM_LANES), 1)
    return jnp.where(row == lane, column, 0.0).sum(axis=0, keepdims=True)


def token_ring(
    num_slots: int, token_block: tuple[int, int, int], dtype: jnp.dtype
) -> pltpu.VMEM:
    """A VMEM ring of num_slots (tokens, H, LATENT_DIM) token blocks."""
    # Note (david): a bf16 scratch of whole token blocks gets other tiles
    # than the HBM operands, and every copy between them then converts the
    # tiling; in blocks of NUM_SUBLANES rows it keeps the operands' tiles.
    step_tokens, num_heads, latent_dim = token_block
    num_blocks = num_slots * step_tokens * num_heads // NUM_SUBLANES
    return pltpu.VMEM((num_blocks, NUM_SUBLANES, latent_dim), dtype)


def tokens_per_step(num_tokens: int) -> int:
    """Tokens a step attends: MAX_TOKENS_PER_STEP, or all of a call of fewer
    tokens."""
    return min(MAX_TOKENS_PER_STEP, num_tokens)


def dsa_sparse_mla_pallas(
    q_nope: jax.Array,
    q_pe_t: jax.Array,
    nope_rows: jax.Array,
    rope_rows: jax.Array,
    flag_rows: jax.Array,
    scalars: jax.Array,
    previous: tuple[jax.Array, ...] | None,
    *,
    cu_q_lens: jax.Array,
    first_token: jax.Array,
    num_chunk_tokens: int,
    sm_scale: float,
    return_lse: bool,
) -> tuple[jax.Array, ...]:
    """Attend tokens [first_token, first_token + num_chunk_tokens) of the
    call, whose gathered rows nope_rows, rope_rows and flag_rows hold.

    Args:
      q_nope: (T, H, LATENT_DIM) bf16, every token of the call; T a multiple
        of the step's tokens.
      q_pe_t: (T, ROPE_DIM, H) bf16, the (T, H, ROPE_DIM) rope queries with
        the heads minor, the layout XLA gives them at H = 128.
      nope_rows, rope_rows, flag_rows: the gather's outputs for the chunk's
        tokens.
      scalars: attention_scalars of the fp8 cache's positive scale and the
        slots each step of the call attends: top_k, or up to
        max_blocked_slots(top_k) in whole BLOCK_SLOTS, or 0.
      previous: (out, [lse]) that this chunk's rows are written into, in
        place: the call's earlier chunks', or any arrays of the returned
        shapes; None for a chunk that makes them.
      cu_q_lens: (B + 1,) int32, read for its last word alone: the call's
        tokens before its padding rows, which are its last. The gathered
        rows of a step of padding are not read, but for the chunk's first
        step's.
      first_token: (1,) int32, the chunk's first token, a multiple of the
        step's tokens; traced, so chunks of one size share one kernel.
      num_chunk_tokens: the chunk's tokens, a multiple of the step's tokens.
      sm_scale: positive scale of q_nope . latent + q_pe . rope.
      return_lse: also return the lse rows.

    Returns:
      (out, [lse]): out (T, H, LATENT_DIM) bf16 and lse (T / G, G, NUM_LANES)
      f32, G the step's tokens and lane h of a token's row its head h. Only
      the chunk's rows are written; a padding token's are zeros and +inf.
    """
    num_tokens, num_heads, latent_dim = q_nope.shape
    rope_dim = q_pe_t.shape[1]
    top_k = nope_rows.shape[0] // num_chunk_tokens
    step_tokens = tokens_per_step(num_tokens)
    step_rows = step_tokens * top_k
    token_block = (step_tokens, num_heads, latent_dim)
    out_shapes = [jax.ShapeDtypeStruct(q_nope.shape, q_nope.dtype)]
    scratch_shapes = [
        # nope_ring
        pltpu.VMEM((NUM_STAGES, step_rows, NOPE_WORDS), nope_rows.dtype),
        # rope_ring
        pltpu.VMEM(
            (NUM_STAGES, step_rows // ROPE_ROW_SLOTS, NOPE_WORDS),
            rope_rows.dtype,
        ),
        # flag_ring
        pltpu.VMEM(
            (NUM_STAGES, step_rows // FLAG_ROW_SLOTS, NOPE_WORDS),
            flag_rows.dtype,
        ),
        # q_nope_ring
        token_ring(NUM_STAGES, token_block, q_nope.dtype),
        # q_pe_ring
        pltpu.VMEM(
            (NUM_STAGES, step_tokens, rope_dim, num_heads), q_pe_t.dtype
        ),
        # out_ring
        token_ring(NUM_STAGES + 1, token_block, q_nope.dtype),
    ]
    if return_lse:
        lse_block = (1, step_tokens, NUM_LANES)
        out_shapes.append(
            jax.ShapeDtypeStruct(
                (num_tokens // step_tokens, *lse_block[1:]), jnp.float32
            )
        )
        # lse_ring
        scratch_shapes.append(
            pltpu.VMEM((NUM_STAGES + 1, *lse_block), jnp.float32)
        )
    # scalars_ref, cu_q_lens_ref
    scratch_shapes.append(pltpu.SMEM(scalars.shape, scalars.dtype))
    scratch_shapes.append(pltpu.SMEM(cu_q_lens.shape, cu_q_lens.dtype))
    # Note (david): a ring slot's copies share one DMA semaphore: a step waits
    # for all of them, its rows and queries and the out rows the slot last
    # held, before it reads or writes the slot. The padding slot's also takes
    # the scalars' copies, which land before any padding row starts.
    scratch_shapes.append(pltpu.SemaphoreType.DMA((NUM_STAGES + 1,)))
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    # Note (david): the kernel copies the scalars and cu_q_lens to SMEM
    # itself, under its first step's copies; cu_q_lens comes as the caller
    # holds it, so no XLA op stands before the kernel for the count.
    operands = [
        scalars,
        cu_q_lens,
        q_nope,
        q_pe_t,
        nope_rows,
        rope_rows,
        flag_rows,
    ]
    in_specs = [hbm_spec] * len(operands)
    # Note (david): the kernel's refs start with the scalar first_token.
    num_inputs = 1 + len(operands)
    if previous is None:
        previous = ()
        aliases = {}
    else:
        # Note (david): the chunk writes only its own steps' rows, so its
        # outputs alias the earlier chunks' to keep their rows.
        in_specs += [hbm_spec] * len(previous)
        aliases = {num_inputs + i: i for i in range(len(previous))}
    kernel = functools.partial(
        dsa_sparse_mla_kernel,
        num_previous=len(previous),
        num_steps=num_chunk_tokens // step_tokens,
        sm_scale_log2=sm_scale * LOG2E,
        return_lse=return_lse,
    )
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=in_specs,
            out_specs=[hbm_spec] * len(out_shapes),
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shapes,
        input_output_aliases=aliases,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_bytes(),
            disable_bounds_checks=True,
        ),
        name="dsa_sparse_mla",
    )(first_token, *operands, *previous)
