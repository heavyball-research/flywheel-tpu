"""Packed-sequence and causal fragment mask appliers."""

import jax
import jax.numpy as jnp
from jax import lax


def apply_packed_bounds_mask(
    qk: jax.Array,
    mask_value: float,
    ids: jax.Array,
    lo: jax.Array,
    span: jax.Array,
) -> jax.Array:
  """Mask qk entries whose major-axis id lies outside [lo, lo + span).

  ids is qk-shaped: the absolute index along qk's sublane axis. lo and span are
  (1, minor) rows indexed by qk's lane axis and broadcast over the sublanes.
  """
  # Note (david): one unsigned comparison checks both bounds, since ids below
  # lo wrap around to large unsigned values.
  relative_ids = (ids - lo).astype(jnp.uint32)
  return jnp.where(relative_ids < span.astype(jnp.uint32), qk, mask_value)


def apply_causal_kv_major(
    qk: jax.Array,
    mask_value: float,
    q_offset: jax.Array | int,
    k_offset: jax.Array | int,
) -> jax.Array:
  """Causal mask on a kv-major (bkv, bq) score tile.

  q_offset and k_offset are the tile's absolute (causal_offset-adjusted) first
  q and kv positions.
  """
  k_ids = k_offset + lax.broadcasted_iota(jnp.int32, qk.shape, 0)
  q_ids = q_offset + lax.broadcasted_iota(jnp.int32, qk.shape, 1)
  return jnp.where(q_ids >= k_ids, qk, mask_value)

