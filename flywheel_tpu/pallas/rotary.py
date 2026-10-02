"""RoPE table preparation and VMEM tile rotation for prefill forward."""

import jax
import jax.numpy as jnp
from jax import lax

# Note (david): GPU FlashAttention requires rotary_dim divisible by 16; keep the
# same table contract.
ROTARY_DIM_MULTIPLE = 16


def prepare_rotary(
    cos: jax.Array | None,
    sin: jax.Array | None,
    *,
    head_dim: int,
    dtype: jnp.dtype,
    seqlen_q: int,
    seqlen_k: int,
    interleaved: bool,
    cu_q: jax.Array | None = None,
    cu_k: jax.Array | None = None,
    max_seqlen_k: int | None = None,
    rotate_k: bool = True,
    seqused_k: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | None] | None:
  """Gather per-token cos/sin coefficients without materializing rotated Q or K.

  cos and sin are (seqlen_ro, head_dim / 2) tables in Q's dtype or float32;
  float32 keeps the in-kernel f32 rotation identical to an f32 RoPE outside the
  kernel. Prefill positions are 0..len_k-1 for K and len_k-len_q..len_k-1 for Q
  per sequence. Returns (q_coefficients, k_coefficients), each
  (2, 1, tokens, head_dim / 2) holding cos then sin; the singleton axis is
  shared by all batch rows or packed heads. k_coefficients is None when
  rotate_k is False (K arrives already rotated). Returns None without tables.

  seqused_k (with cu_q, in place of cu_k) gives each sequence's len_k directly,
  as a paged KV cache does; its K is already rotated, so rotate_k must be
  False.
  """
  if type(interleaved) is not bool:
    raise ValueError("rotary_interleaved must be a static bool.")
  if type(rotate_k) is not bool:
    raise ValueError("rotary_k must be a static bool.")
  if seqused_k is not None and (cu_q is None or cu_k is not None or rotate_k):
    raise ValueError(
        "seqused_k replaces cu_k for packed Q over an already rotated K: pass"
        " it with cu_q, cu_k=None and rotate_k=False.")
  if (cos is None) != (sin is None):
    raise ValueError("rotary_cos and rotary_sin must be provided together.")
  if cos is None and not rotate_k:
    raise ValueError("rotary_k=False needs rotary_cos and rotary_sin.")
  if cos is None:
    return None
  else:
    cos, sin = jnp.asarray(cos), jnp.asarray(sin)
    if cos.ndim != 2 or sin.shape != cos.shape:
      raise ValueError(
          "rotary_cos and rotary_sin must have the same 2-D shape.")
    rotary_dim = 2 * cos.shape[1]
    if rotary_dim == 0 or rotary_dim % ROTARY_DIM_MULTIPLE:
      raise ValueError(
          "rotary_dim must be positive and divisible by"
          f" {ROTARY_DIM_MULTIPLE}.")
    if rotary_dim != head_dim:
      raise ValueError(
          f"rotary_dim ({rotary_dim}) must equal head_dim ({head_dim}); partial"
          " rotary is not supported.")
    if (cos.dtype != sin.dtype
        or cos.dtype not in (jnp.dtype(dtype), jnp.dtype(jnp.float32))):
      raise ValueError(
          "rotary_cos and rotary_sin must share Q's dtype or be float32.")
    if cu_q is None:
      kv_len_bound = seqlen_k
    else:
      kv_len_bound = max_seqlen_k
    if cos.shape[0] < kv_len_bound:
      raise ValueError("rotary table is shorter than the KV sequence bound.")

    def _packed_positions(cu_seqlens, num_tokens, seq_offsets):
      token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
      seq_ids = jnp.minimum(
          jnp.searchsorted(cu_seqlens, token_ids, side="right") - 1,
          cu_seqlens.shape[0] - 2)
      positions = token_ids - cu_seqlens[seq_ids] + seq_offsets[seq_ids]
      # Note (david): unused packed capacity is a kernel-internal padding
      # sequence whose rotation is irrelevant; position 0 keeps it in the table.
      return jnp.where(token_ids < cu_seqlens[-1], positions, 0)

    if cu_q is None and seqlen_q > seqlen_k:
      raise ValueError("RoPE prefill requires seqlen_q <= seqlen_k.")
    elif cu_q is None:
      q_positions = jnp.arange(seqlen_q) + seqlen_k - seqlen_q
      k_positions = jnp.arange(seqlen_k)
    else:
      q_lens = jnp.diff(cu_q)
      if seqused_k is None:
        k_lens = jnp.diff(cu_k)
      else:
        k_lens = jnp.asarray(seqused_k, q_lens.dtype)
      are_lengths_valid = jnp.all(
          (q_lens >= 0) & (k_lens >= q_lens) & (k_lens <= cos.shape[0]))

      def _raise_invalid_lengths():
        raise ValueError(
            "RoPE requires 0 <= len_q <= len_k <= rotary table length.")

      # Note (david): traced cu_seqlens cannot be checked at trace time, so the
      # check runs on device and raises through an ordered debug callback.
      if isinstance(are_lengths_valid, jax.core.Tracer):
        lax.cond(
            are_lengths_valid, lambda: None,
            lambda: jax.debug.callback(_raise_invalid_lengths, ordered=True))
      elif not bool(are_lengths_valid):
        _raise_invalid_lengths()
      q_positions = _packed_positions(cu_q, seqlen_q, k_lens - q_lens)
      if seqused_k is None:
        k_positions = _packed_positions(
            cu_k, seqlen_k, jnp.zeros_like(k_lens))
      else:
        k_positions = None

    q_coefficients = jnp.stack((cos[q_positions], sin[q_positions]))[:, None]
    if rotate_k:
      k_coefficients = jnp.stack((cos[k_positions], sin[k_positions]))[:, None]
    else:
      k_coefficients = None
    return q_coefficients, k_coefficients


def rotate_tile(
    tile: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    interleaved: bool,
    head_dim_minor: bool,
) -> jax.Array:
  """Rotate one head tile in f32, then round once to its dtype.

  cos and sin axes already match the tile's physical Q/K layout.
  """
  head_dim_axis = 1 if head_dim_minor else 0
  half_dim = cos.shape[head_dim_axis]
  dim_ids = lax.broadcasted_iota(jnp.int32, tile.shape, head_dim_axis)
  if interleaved:
    is_first_of_pair = dim_ids % 2 == 0
    partner_ids = dim_ids ^ 1
    frequency_ids = dim_ids // 2
  else:
    is_first_of_pair = dim_ids < half_dim
    partner_ids = jnp.where(
        is_first_of_pair, dim_ids + half_dim, dim_ids - half_dim)
    frequency_ids = dim_ids % half_dim
  tile_f32 = tile.astype(jnp.float32)
  # Note (david): take_along_axis lowers to Mosaic's native dynamic_gather, so
  # no lane-axis reshape, unaligned bf16 Ref slice or cross-head permutation.
  partner = jnp.take_along_axis(tile_f32, partner_ids, axis=head_dim_axis,
                                mode="promise_in_bounds")
  cosine = jnp.take_along_axis(cos.astype(jnp.float32), frequency_ids,
                               axis=head_dim_axis, mode="promise_in_bounds")
  sine = jnp.take_along_axis(sin.astype(jnp.float32), frequency_ids,
                             axis=head_dim_axis, mode="promise_in_bounds")
  signed_partner = jnp.where(is_first_of_pair, -partner, partner)
  rotated = tile_f32 * cosine + signed_partner * sine
  return rotated.astype(tile.dtype)
