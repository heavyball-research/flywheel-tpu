"""Naive attention and the random padding masks the flash attn tests share."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from tests.padding import pad_input, unpad_input

# Note (david): masking with a finite value instead of -inf keeps a fully masked
# row finite, so jax.vjp through it does not leak NaN into every input; such
# rows are zeroed explicitly.
MASK_VALUE = -1e30


def randn(rng, shape, dtype=jnp.bfloat16):
    """Standard normal samples drawn in f32 from a numpy Generator, cast to dtype."""
    return jnp.asarray(rng.standard_normal(shape, dtype=np.float32)).astype(dtype)


def generate_random_padding_mask(max_seqlen, batch_size, rng):
    """(batch_size, max_seqlen) bool prefix masks whose lengths are drawn from
    [max(1, max_seqlen - 20), max_seqlen]."""
    lengths = rng.integers(max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1))
    return jnp.asarray(np.arange(max_seqlen)[None, :] < lengths)


def generate_qkv(q, k, v, query_padding_mask, key_padding_mask):
    """Packs the valid tokens of padded q/k/v.

    q: (batch_size, seqlen_q, nheads, d); k: (batch_size, seqlen_k, nheads_k, d);
    v: (batch_size, seqlen_k, nheads_k, d_v); the masks are (batch_size, seqlen) bool.
    Returns (q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
    max_seqlen_k, output_pad_fn), where output_pad_fn scatters packed q-side rows
    back to (batch_size, seqlen_q, ...).
    """
    batch_size, seqlen_q = q.shape[:2]
    q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q, query_padding_mask)
    k_unpad, _, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k, key_padding_mask)
    v_unpad, *_ = unpad_input(v, key_padding_mask)

    def output_pad_fn(output_unpad):
        return pad_input(output_unpad, indices_q, batch_size, seqlen_q)

    return (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        output_pad_fn,
    )


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size,
    query_padding_mask=None,
    key_padding_mask=None,
):
    """True where the bottom-right aligned window hides a key: (seqlen_q, seqlen_k),
    or (batch, 1, seqlen_q, seqlen_k) with padding masks. Each bound is unbounded
    when negative, so (left, -1) is a left-only window."""
    row_idx = jnp.arange(seqlen_q)[:, None]
    col_idx = jnp.arange(seqlen_k)[None, :]
    sk = (
        seqlen_k
        if key_padding_mask is None
        else key_padding_mask.sum(-1)[:, None, None, None]
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else query_padding_mask.sum(-1)[:, None, None, None]
    )
    diagonal = row_idx + sk - sq
    mask = jnp.zeros(jnp.broadcast_shapes(jnp.shape(diagonal), col_idx.shape), bool)
    if window_size[1] >= 0:
        mask = mask | (col_idx > diagonal + window_size[1])
    if window_size[0] >= 0:
        mask = mask | (col_idx < diagonal - window_size[0])
    return mask


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
):
    """Naive softmax(q @ k^T / sqrt(d)) @ v on padded inputs.

    q: (batch_size, seqlen_q, nheads, d); k: (batch_size, seqlen_k, nheads_k, d);
    v: (batch_size, seqlen_k, nheads_k, d_v); the padding masks are (batch_size,
    seqlen) bool. causal and window_size are bottom-right aligned on each sequence's
    valid lengths, and causal forces the right bound to 0. upcast computes in f32;
    without it the math runs in the input dtype, and reorder_ops scales k instead of
    q, so the pair measures how far plain bf16 arithmetic drifts.

    Returns (out, lse): out (batch_size, seqlen_q, nheads, d_v) in q's dtype, zero
    on padded queries and on rows with no visible key; lse (batch_size, nheads,
    seqlen_q) f32, -inf on rows with no visible key.
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = (x.astype(jnp.float32) for x in (q, k, v))
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = jnp.repeat(k, q.shape[2] // k.shape[2], axis=2)
    v = jnp.repeat(v, q.shape[2] // v.shape[2], axis=2)
    softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    # Note (david): the default TPU matmul precision truncates f32 operands to one
    # bf16 pass, which would make the upcast reference as coarse as the kernel.
    if not reorder_ops:
        scores = jnp.einsum(
            "bthd,bshd->bhts", q * softmax_scale, k, precision=jax.lax.Precision.HIGHEST
        )
    else:
        scores = jnp.einsum(
            "bthd,bshd->bhts", q, k * softmax_scale, precision=jax.lax.Precision.HIGHEST
        )
    if softcap > 0:
        scores = jnp.tanh(scores / softcap) * softcap
    masked = jnp.zeros((1, 1, seqlen_q, seqlen_k), bool)
    if key_padding_mask is not None:
        masked = masked | ~key_padding_mask[:, None, None, :]
    if window_size[0] >= 0 or window_size[1] >= 0:
        masked = masked | construct_local_mask(
            seqlen_q, seqlen_k, window_size, query_padding_mask, key_padding_mask
        )
    scores = jnp.where(masked, MASK_VALUE, scores)
    empty_row = masked.all(axis=-1, keepdims=True)
    # Note (david): the softmax accumulates in f32 even for bf16 scores, then rounds
    # to v's dtype.
    attention = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(v.dtype)
    attention = jnp.where(empty_row, 0.0, attention)
    lse = jnp.where(
        empty_row[..., 0], -jnp.inf, jax.nn.logsumexp(scores.astype(jnp.float32), axis=-1)
    )
    output = jnp.einsum("bhts,bshd->bthd", attention, v, precision=jax.lax.Precision.HIGHEST)
    if query_padding_mask is not None:
        output = jnp.where(~query_padding_mask[:, :, None, None], 0.0, output)
    return output.astype(dtype_og), lse
