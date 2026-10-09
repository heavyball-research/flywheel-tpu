"""Packed-sequence (cu_seqlens) preparation for the varlen forward kernel."""

import jax
import jax.numpy as jnp


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
