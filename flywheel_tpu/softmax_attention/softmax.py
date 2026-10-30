"""Static-anchor softmax row-state helpers shared by the prefill and decode
kernels."""

import math

import jax
import jax.numpy as jnp

from .config import DEFAULT_MASK_VALUE

# Note (david): the guard keeps score <= e^T, row_sum <= kv_seq_len * e^T and
# |o| <= row_sum * max|v| under the f32 max (e^88.7), assuming |v| <= 2^12 and
# keeping a 2-nat margin. Below a 20-nat threshold the static anchor cannot be
# guarded safely.
F32_MAX_LOG = 88.7
V_ABS_MAX_LOG2 = 12
GUARD_MARGIN_NATS = 2.0
MIN_GUARD_THRESHOLD_NATS = 20.0

# Note (david): a masked score is exactly DEFAULT_MASK_VALUE and real logits
# sit far inside half of it, so these bounds never mistake one for the other;
# exp2(score - UNANCHORED_ROW_MAX) is exactly 0 for every score.
MASKED_ROW_BOUND = 0.5 * DEFAULT_MASK_VALUE
UNANCHORED_ROW_MAX = -DEFAULT_MASK_VALUE
UNANCHORED_ROW_BOUND = -MASKED_ROW_BOUND


def overflow_guard_threshold(kv_seq_len: int) -> float:
    """Static-anchor overflow-guard threshold in log2 units."""
    threshold = (
        F32_MAX_LOG
        - math.log(kv_seq_len)
        - V_ABS_MAX_LOG2 * math.log(2.0)
        - GUARD_MARGIN_NATS
    )
    if threshold < MIN_GUARD_THRESHOLD_NATS:
        raise ValueError(
            f"overflow-guard threshold {threshold:.1f} nats is below the 20-nat"
            f" floor for kv_seq_len={kv_seq_len}; the static anchor cannot be"
            " guarded safely at this sequence length."
        )
    # Note (david): the kernels compare anchors in log2 units, while the floor
    # above is checked in nats.
    return threshold / math.log(2.0)


def fold_sink(
    row_max: jax.Array, row_sum: jax.Array, sink: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Join a sink logit (log2 units) to finished static-anchor row state.

    Returns (anchor, row_sum, out_scale): lse is anchor + log2(row_sum) in log2
    units, and the output is the accumulator times out_scale (before any V
    descale). sink broadcasts against the row state.
    """
    # Note (deepolar): a sink is one more softmax term with no value row, so it
    # only joins the denominator. The output scale 1 / (row_sum + exp2(sink -
    # row_max)) needs no new anchor: an overflowed term gives 0, the output's
    # limit. lse re-anchors at max(row_max, sink), so exp2(sink - anchor) stays
    # finite when the sink sits far above the static anchor. A row that saw no
    # key has row_sum 0 (decode) or a masked-score anchor (the fragment pipeline
    # counts its masked scores against themselves). Its only term is the sink,
    # so it takes the sink as its anchor with row_sum 1: out = 0 and lse = sink,
    # and a -inf sink gives lse = -inf with no -inf - -inf.
    has_keys = jnp.logical_and(row_sum > 0, row_max > MASKED_ROW_BOUND)
    out_scale = jnp.where(
        has_keys, 1.0 / (row_sum + jnp.exp2(sink - row_max)), 0.0
    )
    anchor = jnp.where(has_keys, jnp.maximum(row_max, sink), sink)
    row_sum = jnp.where(
        has_keys,
        row_sum * jnp.exp2(row_max - anchor) + jnp.exp2(sink - anchor),
        1.0,
    )
    return anchor, row_sum, out_scale
