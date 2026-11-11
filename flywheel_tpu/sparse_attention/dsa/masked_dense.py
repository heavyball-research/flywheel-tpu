"""DSA masked-dense MLA: each query token attends the cache positions up to its
own, read densely from the split fp8 cache and masked to its top-k selection.

A request of at most top_k positions needs no mask past causality, since its
top-k selects every position it sees.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from flywheel_tpu.sparse_attention.dsa.cache_write import token_positions
from flywheel_tpu.sparse_attention.dsa.kernels.masked_dense import (
    SLAB_POSITIONS,
    MaskedDenseConfig,
    bitmap_rows,
    default_masked_dense_config,
    dsa_masked_dense_pallas,
    dsa_selection_bitmap_pallas,
    masked_dense_plan,
)
from flywheel_tpu.sparse_attention.dsa.sparse_mla import (
    check_indices,
    check_operands,
)

SUPPORTED_HEADS = (16, 32, 64, 128)


@functools.partial(
    jax.jit,
    static_argnames=("sm_scale", "max_seqlen_q", "max_seqlen_k", "return_lse"),
)
def dsa_masked_dense(
    q_nope: jax.Array,
    q_pe: jax.Array,
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    topk_indices: jax.Array,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
    *,
    sm_scale: float,
    kv_scale: jax.Array,
    max_seqlen_k: int,
    max_seqlen_q: int | None = None,
    return_lse: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """dsa_sparse_mla's attention, each token's selected positions read
    densely from the cache and masked, not gathered.

    Args:
      q_nope, q_pe, nope_cache, rope_cache, cu_q_lens, kv_lens, block_table:
        as dsa_sparse_mla takes them; H in SUPPORTED_HEADS.
      topk_indices: (T, top_k) int32 as dsa_sparse_mla takes them; only a
        token's first min(top_k, position + 1) slots are read.
      sm_scale: scale of q_nope . latent + q_pe . rope.
      kv_scale: f32 scalar scale of the fp8 cache; traced.
      max_seqlen_k: a static bound on kv_lens. At most top_k, no bitmap is
        built.
      max_seqlen_q: a static bound on the q_lens, which sizes the q blocks;
        T if None.
      return_lse: also return the natural-log lse.

    Returns:
      out (T, H, LATENT_DIM) bf16, or (out, lse) with return_lse, lse (H, T)
      f32, as dsa_sparse_mla returns them.
    """
    num_tokens, num_heads, _ = q_nope.shape
    if max_seqlen_q is None:
        max_seqlen_q = num_tokens
    check_operands(
        q_nope, q_pe, nope_cache, rope_cache, cu_q_lens, kv_lens, block_table
    )
    check_indices(topk_indices, num_tokens)
    check_bounds(
        num_heads,
        num_tokens,
        block_table,
        nope_cache,
        max_seqlen_q,
        max_seqlen_k,
    )
    top_k = topk_indices.shape[1]
    has_bitmap = max_seqlen_k > top_k
    config = masked_dense_config_of(
        num_heads=num_heads,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        has_bitmap=has_bitmap,
    )
    if has_bitmap:
        bitmap = selection_bitmap(
            topk_indices, cu_q_lens, kv_lens, max_seqlen_k=max_seqlen_k
        )
    else:
        bitmap = None
    return masked_dense_attention(
        q_nope,
        q_pe,
        nope_cache,
        rope_cache,
        bitmap,
        cu_q_lens,
        kv_lens,
        block_table,
        attended_cu_q_lens=cu_q_lens,
        config=config,
        max_seqlen_q=max_seqlen_q,
        sm_scale=sm_scale,
        kv_scale=kv_scale,
        top_k=top_k,
        max_seqlen_k=max_seqlen_k,
        return_lse=return_lse,
    )


def selection_bitmap(
    topk_indices: jax.Array,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    *,
    max_seqlen_k: int,
) -> jax.Array:
    """Each token's selection bitmap from its first min(top_k, position + 1)
    slots, whatever the slots after them hold."""
    _, position = token_positions(cu_q_lens, kv_lens, topk_indices.shape[0])
    return dsa_selection_bitmap_pallas(
        topk_indices, position + 1, num_slabs=bitmap_slabs(max_seqlen_k)
    )


def bitmap_slabs(max_seqlen_k: int) -> int:
    """Slabs of a selection bitmap that covers every position."""
    return pl.cdiv(max_seqlen_k, SLAB_POSITIONS)


def masked_dense_config_of(
    *,
    num_heads: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    has_bitmap: bool,
) -> MaskedDenseConfig:
    """The default blocks of a call, with bitmaps or without."""
    if has_bitmap:
        rows_per_token = bitmap_rows(bitmap_slabs(max_seqlen_k))
    else:
        rows_per_token = 0
    return default_masked_dense_config(
        num_heads=num_heads,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        rows_per_token=rows_per_token,
    )


def masked_dense_attention(
    q_nope: jax.Array,
    q_pe: jax.Array,
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    bitmap: jax.Array | None,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
    *,
    attended_cu_q_lens: jax.Array,
    config: MaskedDenseConfig,
    max_seqlen_q: int,
    sm_scale: float,
    kv_scale: jax.Array,
    top_k: int,
    max_seqlen_k: int,
    return_lse: bool,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """dsa_masked_dense on checked operands at config, its bitmap given:
    None when every request it attends holds at most top_k positions.

    It attends the requests that attended_cu_q_lens gives their own q_len,
    cu_q_lens itself for all of them; the rows of a request it gives 0 are
    unspecified, its lse too, and rows past cu_q_lens[B] are zero.
    """
    num_tokens, num_heads, _ = q_nope.shape
    plan = masked_dense_plan(
        attended_cu_q_lens,
        kv_lens,
        num_tokens=num_tokens,
        max_seqlen_q=max_seqlen_q,
        config=config,
    )
    # Note (david): the kernel reads q_pe with the heads minor, as XLA lays
    # (T, 128, ROPE_DIM) out, so the swap is a bitcast.
    outputs = dsa_masked_dense_pallas(
        q_nope,
        jnp.swapaxes(q_pe, 1, 2),
        nope_cache,
        rope_cache,
        bitmap,
        cu_q_lens,
        kv_lens,
        block_table,
        jnp.asarray(kv_scale, jnp.float32).reshape(1),
        plan=plan,
        sm_scale=sm_scale,
        top_k=top_k,
        num_slabs=bitmap_slabs(max_seqlen_k),
        return_lse=return_lse,
    )
    if not return_lse:
        return outputs[0]
    else:
        out, lse_blocks = outputs
        request, position = token_positions(cu_q_lens, kv_lens, num_tokens)
        return out, token_lse(
            lse_blocks,
            plan.request_first_q_block,
            cu_q_lens,
            request,
            position,
            num_heads,
        )


def token_lse(
    lse_blocks: jax.Array,
    request_first_q_block: jax.Array,
    cu_q_lens: jax.Array,
    request: jax.Array,
    position: jax.Array,
    num_heads: int,
) -> jax.Array:
    """(H, T) lse from the kernel's lse blocks, q block j holding token i's
    at [j, i, head]; +inf for padding tokens."""
    max_q_blocks, block_q, _ = lse_blocks.shape
    token = jnp.arange(request.shape[0], dtype=jnp.int32)
    token_in_request = token - cu_q_lens[request]
    ordinal = jnp.clip(
        request_first_q_block[request] + token_in_request // block_q,
        0,
        max_q_blocks - 1,
    )
    lse = lse_blocks[ordinal, token_in_request % block_q, :num_heads]
    return jnp.where(position[:, None] >= 0, lse, jnp.inf).T


def check_bounds(
    num_heads: int,
    num_tokens: int,
    block_table: jax.Array,
    nope_cache: jax.Array,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> None:
    """Fails unless H is supported and the static bounds fit the call."""
    capacity = block_table.shape[1] * nope_cache.shape[1]
    if num_heads not in SUPPORTED_HEADS:
        raise NotImplementedError(
            f"The masked-dense MLA supports {SUPPORTED_HEADS} heads; got"
            f" {num_heads}."
        )
    elif not 1 <= max_seqlen_q <= num_tokens:
        raise ValueError(
            f"max_seqlen_q must be in [1, T={num_tokens}]; got {max_seqlen_q}."
        )
    elif not 1 <= max_seqlen_k <= capacity:
        raise ValueError(
            f"max_seqlen_k must be in [1, the {capacity} positions block_table"
            f" holds]; got {max_seqlen_k}."
        )
