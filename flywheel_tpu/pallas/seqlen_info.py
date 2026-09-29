"""Packed-sequence (cu_seqlens) preparation for the varlen forward kernel."""

import jax
import jax.numpy as jnp


def extend_cu_seqlens(cu_seqlens: jax.Array, padded_len: int) -> jax.Array:
  """Append padded_len so the pad tail past cu_seqlens[-1] is its own sequence.

  Pad rows then always have a reachable key (the softmax denominator stays
  nonzero) and attend only other pad tokens. Already-padded input gains an
  empty trailing sequence.
  """
  return jnp.concatenate(
      [jnp.asarray(cu_seqlens, jnp.int32), jnp.array([padded_len], jnp.int32)]
  )


def split_pad_tail(
    cu_q_ext: jax.Array, cu_k_ext: jax.Array
) -> tuple[jax.Array, jax.Array]:
  """Split the extended pad sequence into a square plus a zero-query sequence.

  Pad rows only need a nonempty kv window, and an under-filled buffer would
  otherwise give them a tail_q x tail_kv trapezoid over every unclaimed kv
  token (a 4067 x 36236 tail measured 2.8x the real batch's flops on v6e). The
  pad keeps a tail_q-wide window and a sequence with zero query rows, which gets
  no q block, owns the leftover kv, so the pad costs tail_q^2 / 2. Real
  sequences are untouched. Both arrays grow by one entry; values may be tracers.
  """
  tail_q = cu_q_ext[-1] - cu_q_ext[-2]
  tail_kv = cu_k_ext[-1] - cu_k_ext[-2]
  # Note (david): min() keeps the kv split inside the axis when tail_kv <
  # tail_q, which is as degenerate as the unsplit tail, never worse.
  kv_split = cu_k_ext[-2] + jnp.minimum(tail_q, tail_kv)
  return (
      jnp.concatenate([cu_q_ext, cu_q_ext[-1:]]),
      jnp.concatenate([cu_k_ext[:-1], kv_split[None], cu_k_ext[-1:]]),
  )


def check_cu_seqlens_pair(
    cu_seqlens_k: jax.Array, cu_seqlens_q: jax.Array
) -> None:
  """Shape rules every packed-sequence build shares; values may be tracers."""
  cu_seqlens_k = jnp.asarray(cu_seqlens_k)
  cu_seqlens_q = jnp.asarray(cu_seqlens_q)
  if cu_seqlens_k.ndim != 1 or cu_seqlens_k.shape != cu_seqlens_q.shape:
    raise ValueError(
        "cu_seqlens_q and cu_seqlens_k must be equal-shaped 1-D arrays (one"
        f" entry per sequence boundary); got {cu_seqlens_q.shape},"
        f" {cu_seqlens_k.shape}."
    )
  if cu_seqlens_k.shape[0] < 2:
    raise ValueError(
        f"cu_seqlens needs at least 2 entries (0 and the total); got"
        f" {cu_seqlens_k.shape[0]}."
    )
