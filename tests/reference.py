"""Naive attention references shared by the test modules."""

import math

import jax
import jax.numpy as jnp

from flywheel_tpu import flash_attn_func, flash_attn_varlen_func

# Note (david): masking with -1e30 instead of -inf keeps a fully masked row
# finite, so its softmax gradient does not leak NaN into every input.
MASK_VALUE = -1e30


def seq_index(cu_seqlens, seqlen):
    return jnp.searchsorted(
        jnp.asarray(cu_seqlens), jnp.arange(seqlen), side="right") - 1


def attention_ref(
    q,
    k,
    v,
    causal=False,
    window_size=(-1, -1),
    cu_seqlens=None,
    softmax_scale=None,
    softcap=0.0,
    upcast=True,
    return_lse=False,
):
    """Naive attention on (batch, seqlen, nheads, headdim) with flash_attn masks.

    causal and window_size are bottom-right aligned: query i sees key j iff
    j <= i + seqlen_kv - seqlen_q. A row with no visible key returns out = 0
    and lse = -inf. cu_seqlens is a (cu_q, cu_kv) pair covering the padded axes.
    """
    input_dtype = q.dtype
    compute_dtype = jnp.float32 if upcast else input_dtype
    q, k, v = (x.astype(compute_dtype) for x in (q, k, v))
    _, seqlen_q, nheads, headdim = q.shape
    seqlen_kv, nheads_k = k.shape[1:3]
    scale = 1.0 / math.sqrt(headdim) if softmax_scale is None else softmax_scale
    k, v = (jnp.repeat(x, nheads // nheads_k, axis=2) for x in (k, v))
    # Note (david): the default TPU matmul precision truncates f32 operands to
    # one bf16 pass, which would make the reference as coarse as the kernel.
    raw_scores = jnp.einsum(
        "bshd,bthd->bhst", q, k, precision=jax.lax.Precision.HIGHEST) * scale
    scores = (
        raw_scores if softcap == 0.0
        else jnp.tanh(raw_scores / softcap) * softcap
    )

    offset = seqlen_kv - seqlen_q
    row = jnp.arange(seqlen_q)[:, None]
    col = jnp.arange(seqlen_kv)[None, :]
    left = window_size[0]
    right = 0 if causal else window_size[1]
    left_visible = True if left == -1 else row + offset - left <= col
    right_visible = True if right == -1 else col <= row + offset + right
    same_seq = (
        True if cu_seqlens is None
        else seq_index(cu_seqlens[0], seqlen_q)[:, None]
        == seq_index(cu_seqlens[1], seqlen_kv)[None, :]
    )
    visible = left_visible & right_visible & same_seq
    scores = jnp.where(visible, scores, MASK_VALUE)

    probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
    out = jnp.einsum(
        "bhst,bthd->bshd", probs.astype(scores.dtype), v,
        precision=jax.lax.Precision.HIGHEST).astype(input_dtype)
    lse = jax.scipy.special.logsumexp(scores.astype(jnp.float32), axis=-1)
    num_empty_rows = 0 if right == -1 else max(0, -offset - right)
    empty_row = jnp.arange(seqlen_q) < num_empty_rows
    out = jnp.where(empty_row[None, :, None, None], 0, out)
    lse = jnp.where(empty_row[None, None, :], -jnp.inf, lse)
    return (out, lse) if return_lse else out


def head_major_flash_attn(q, k, v, *, heads_outer_fold=False, **kwargs):
    """flash_attn_func's head-major layout run on the tests' (batch, seqlen,
    nheads, headdim) arrays, which attention_ref and random_qkv share.

    The API takes (batch, nheads, seqlen, headdim), or (nheads, batch, seqlen,
    headdim) with heads_outer_fold; out comes back in the tests' layout and lse
    as is.
    """
    axes = (2, 0, 1, 3) if heads_outer_fold else (0, 2, 1, 3)
    inverse = (1, 2, 0, 3) if heads_outer_fold else (0, 2, 1, 3)
    result = flash_attn_func(
        *(x.transpose(axes) for x in (q, k, v)), token_major=False,
        heads_outer_fold=heads_outer_fold, **kwargs)
    if kwargs.get("return_softmax_lse", False):
        out, lse = result
        return out.transpose(inverse), lse
    return result.transpose(inverse)


def head_major_varlen_flash_attn(q, k, v, *args, **kwargs):
    """flash_attn_varlen_func's head-major layout run on the tests' packed
    (total, nheads, headdim) arrays.

    The API takes (nheads, total, headdim); out comes back in the tests'
    layout and lse as is.
    """
    result = flash_attn_varlen_func(
        *(x.transpose(1, 0, 2) for x in (q, k, v)), *args, token_major=False,
        **kwargs)
    if kwargs.get("return_softmax_lse", False):
        out, lse = result
        return out.transpose(1, 0, 2), lse
    return result.transpose(1, 0, 2)


def random_qkv(seed, batch, seqlen, nheads, nheads_k, headdim, dtype,
               headdim_v=None, seqlen_kv=None):
    headdim_v = headdim if headdim_v is None else headdim_v
    seqlen_kv = seqlen if seqlen_kv is None else seqlen_kv
    key_q, key_k, key_v = jax.random.split(jax.random.PRNGKey(seed), 3)
    q = jax.random.normal(key_q, (batch, seqlen, nheads, headdim), dtype=dtype)
    k = jax.random.normal(
        key_k, (batch, seqlen_kv, nheads_k, headdim), dtype=dtype)
    v = jax.random.normal(
        key_v, (batch, seqlen_kv, nheads_k, headdim_v), dtype=dtype)
    return q, k, v


def rel_err(actual, expected):
    actual, expected = (x.astype(jnp.float32) for x in (actual, expected))
    return float(
        jnp.linalg.norm(actual - expected) / (jnp.linalg.norm(expected) + 1e-12))


def head_major_ref(q, k, v, causal, scale, causal_offset=0):
    """attention_ref's math on the kernel's (nheads, seq, head_dim) layout.

    Returns (o, lse) in f32 with lse in nats. causal is bottom-right aligned by
    causal_offset (key j visible to query i iff j <= i + causal_offset).
    """
    q, k, v = (x.astype(jnp.float32) for x in (q, k, v))
    seqlen_q, seqlen_kv = q.shape[1], k.shape[1]
    logits = scale * jnp.einsum("hqd,hkd->hqk", q, k)
    row = jnp.arange(seqlen_q)[:, None]
    col = jnp.arange(seqlen_kv)[None, :]
    causal_visible = col <= row + causal_offset if causal else True
    logits = jnp.where(causal_visible, logits, MASK_VALUE)
    lse = jax.scipy.special.logsumexp(logits, axis=-1)
    o = jnp.einsum("hqk,hkd->hqd", jnp.exp(logits - lse[..., None]), v)
    # Note (david): the jnp.where gradient is exactly zero on the empty rows,
    # so the -inf lse laid on top stays safe to differentiate through.
    num_empty_rows = max(0, -causal_offset) if causal else 0
    empty_row = jnp.arange(seqlen_q) < num_empty_rows
    o = jnp.where(empty_row[None, :, None], 0, o)
    lse = jnp.where(empty_row[None, :], -jnp.inf, lse)
    return o, lse
