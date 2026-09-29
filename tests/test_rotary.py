"""Fused prefill RoPE against the same attention on independently rotated q/k."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flywheel_tpu import flash_attn_func, flash_attn_varlen_func
from flywheel_tpu.pallas.block_sizes import BlockSizes, TokenMajorInfo
from flywheel_tpu.pallas.flash_fwd import make_flash_attn_mha
from tests.reference import (
    attention_ref,
    head_major_flash_attn,
    head_major_varlen_flash_attn,
    random_qkv,
)

INTERPRET = jax.default_backend() != "tpu"
HEAD_DIM = 128


def rotary_tables(length, rotary_dim, dtype=jnp.bfloat16, theta=10000.0):
    frequency = theta ** (
        -jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim)
    angle = jnp.arange(length, dtype=jnp.float32)[:, None] * frequency
    return jnp.cos(angle).astype(dtype), jnp.sin(angle).astype(dtype)


def rotate_reference(x, cos, sin, positions, interleaved):
    """Full-head RoPE in f32 on (..., heads, head_dim), cast back to x's dtype."""
    value = np.asarray(x, dtype=np.float32)
    cosine = np.asarray(cos, dtype=np.float32)[positions][..., None, :]
    sine = np.asarray(sin, dtype=np.float32)[positions][..., None, :]
    result = np.empty_like(value)
    if interleaved:
        first, second = value[..., 0::2], value[..., 1::2]
        result[..., 0::2] = first * cosine - second * sine
        result[..., 1::2] = first * sine + second * cosine
    else:
        half = value.shape[-1] // 2
        first, second = value[..., :half], value[..., half:]
        result[..., :half] = first * cosine - second * sine
        result[..., half:] = first * sine + second * cosine
    return jnp.asarray(result, dtype=x.dtype)


def layout_inputs(q, k, v, token_major):
    return (
        tuple(x.reshape(*x.shape[:-2], -1) for x in (q, k, v)) if token_major
        else (q, k, v)
    )


def assert_rope_close(actual, expected, atol=2e-3, rtol=1e-2):
    np.testing.assert_allclose(actual.astype(jnp.float32),
                               expected.astype(jnp.float32), atol=atol,
                               rtol=rtol)


def dense_attn(q, k, v, token_major, **kwargs):
    """flash_attn_func on the tests' (batch, seqlen, nheads, headdim) arrays,
    token-major or head-major."""
    if token_major:
        return flash_attn_func(*layout_inputs(q, k, v, True), token_major=True,
                               head_dim=q.shape[-1], **kwargs)
    return head_major_flash_attn(q, k, v, **kwargs)


def varlen_attn(q, k, v, *args, token_major, **kwargs):
    """flash_attn_varlen_func on the tests' packed (total, nheads, headdim)
    arrays, token-major or head-major."""
    if token_major:
        return flash_attn_varlen_func(
            *layout_inputs(q, k, v, True), *args, token_major=True,
            head_dim=q.shape[-1], **kwargs)
    return head_major_varlen_flash_attn(q, k, v, *args, **kwargs)


dense_head_major = functools.partial(head_major_flash_attn,
                                     interpret=INTERPRET)


@pytest.mark.parametrize("token_major", [False, True])
@pytest.mark.parametrize("interleaved", [False, True])
def test_dense_rotary(token_major, interleaved):
    q, k, v = random_qkv(71, 2, 128, 4, 4, HEAD_DIM, jnp.bfloat16)
    cos, sin = rotary_tables(128, HEAD_DIM)
    positions = np.broadcast_to(np.arange(128), (2, 128))
    q_rotated = rotate_reference(q, cos, sin, positions, interleaved)
    k_rotated = rotate_reference(k, cos, sin, positions, interleaved)
    kwargs = dict(causal=True, interpret=INTERPRET, return_softmax_lse=True)
    expected, expected_lse = dense_attn(q_rotated, k_rotated, v, token_major,
                                        **kwargs)
    actual, actual_lse = dense_attn(
        q, k, v, token_major, **kwargs, rotary_cos=cos, rotary_sin=sin,
        rotary_interleaved=interleaved)
    assert_rope_close(actual, expected, rtol=8e-3)
    np.testing.assert_allclose(actual_lse, expected_lse, atol=2e-3, rtol=2e-3)
    naive = attention_ref(q_rotated, k_rotated, v, causal=True)
    if token_major:
        naive = naive.reshape(*naive.shape[:-2], -1)
    assert_rope_close(actual, naive, atol=6e-3, rtol=3e-2)


@pytest.mark.parametrize("token_major", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_dense_rotary_rectangular(token_major, causal, kv_heads):
    # Note (david): K sits at positions 0..255 and the 93 queries at the last
    # 93 of them.
    q, k, v = random_qkv(72, 1, 93, 4, kv_heads, HEAD_DIM, jnp.bfloat16,
                         seqlen_kv=256, headdim_v=256)
    cos, sin = rotary_tables(256, HEAD_DIM)
    q_rotated = rotate_reference(q, cos, sin, np.arange(163, 256)[None], True)
    k_rotated = rotate_reference(k, cos, sin, np.arange(256)[None], True)
    kwargs = dict(causal=causal, interpret=INTERPRET, softcap=10.0)
    expected = dense_attn(q_rotated, k_rotated, v, token_major, **kwargs)
    actual = dense_attn(q, k, v, token_major, **kwargs, rotary_cos=cos,
                        rotary_sin=sin)
    assert_rope_close(actual, expected, rtol=8e-3)


@pytest.mark.parametrize("token_major", [False, True])
@pytest.mark.parametrize("interleaved", [False, True])
def test_varlen_rotary_dynamic_packing(token_major, interleaved):
    # Note (david): positions come from the runtime cu_seqlens, so one trace
    # must serve every packing.
    keys = jax.random.split(jax.random.key(73), 3)
    q, k, v = (jax.random.normal(key, shape, jnp.bfloat16)
               for key, shape in zip(keys, [(173, 4, HEAD_DIM),
                                            (277, 2, HEAD_DIM),
                                            (277, 2, HEAD_DIM)]))
    cos, sin = rotary_tables(256, HEAD_DIM)
    kwargs = dict(max_seqlen_q=256, max_seqlen_k=256, causal=True,
                  token_major=token_major, interpret=INTERPRET,
                  return_softmax_lse=True, block_sizes=(128, 128, 64, 128))
    fused = jax.jit(functools.partial(varlen_attn, **kwargs,
                                      rotary_interleaved=interleaved))
    for q_lengths, k_lengths in [([37, 0, 136], [83, 0, 194]),
                                 ([81, 0, 92], [112, 0, 165])]:
        cu_q = jnp.asarray(np.cumsum([0, *q_lengths]), jnp.int32)
        cu_k = jnp.asarray(np.cumsum([0, *k_lengths]), jnp.int32)
        q_positions = np.concatenate(
            [np.arange(len_k - len_q, len_k)
             for len_q, len_k in zip(q_lengths, k_lengths)])
        k_positions = np.concatenate([np.arange(len_k) for len_k in k_lengths])
        q_rotated = rotate_reference(q, cos, sin, q_positions, interleaved)
        k_rotated = rotate_reference(k, cos, sin, k_positions, interleaved)
        expected, expected_lse = varlen_attn(
            q_rotated, k_rotated, v, cu_q, cu_k, **kwargs)
        actual, actual_lse = fused(q, k, v, cu_q, cu_k, rotary_cos=cos,
                                   rotary_sin=sin)
        assert_rope_close(actual, expected, atol=3e-3)
        np.testing.assert_allclose(actual_lse, expected_lse, atol=3e-3,
                                   rtol=2e-3)
    assert fused._cache_size() == 1


@pytest.mark.parametrize("dim,head_fold,seqlen,num_stages",
                         [(64, 2, 384, 3), (128, 4, 128, 2)])
@pytest.mark.parametrize("interleaved", [False, True])
def test_rotary_token_major_kernel(dim, head_fold, seqlen, num_stages,
                                   interleaved):
    q, k, v = random_qkv(74, 1, seqlen, 4, 4, dim, jnp.bfloat16)
    cos, sin = rotary_tables(seqlen, dim)
    positions = np.arange(seqlen)[None]
    q_rotated = rotate_reference(q, cos, sin, positions, interleaved)
    k_rotated = rotate_reference(k, cos, sin, positions, interleaved)
    kernel = make_flash_attn_mha(
        4, seqlen, seqlen, causal=True,
        block_sizes=BlockSizes(128, 128, 128, 64, num_stages=num_stages),
        head_fold=head_fold, interpret=INTERPRET, q_scale=0.125,
        token_major=TokenMajorInfo(1, 4, 4, dim, dim))
    coeff = jnp.stack((cos, sin))[:, None]
    fused_layout = lambda x: x.reshape(1, seqlen, -1)
    expected = kernel(*map(fused_layout, (q_rotated, k_rotated, v)))
    actual = kernel(*map(fused_layout, (q, k, v)), rotary=(coeff, coeff),
                    rotary_interleaved=interleaved)
    assert_rope_close(actual, expected)


def test_rotary_head_fold_crosses_batch():
    # Note (david): a fold of 2 over 1 head per batch row puts two batch rows,
    # at different positions, in one fold group.
    q, k, v = random_qkv(76, 2, 128, 1, 1, HEAD_DIM, jnp.bfloat16)
    cos, sin = rotary_tables(256, HEAD_DIM)
    positions = np.arange(256).reshape(2, 128)
    q_rotated = rotate_reference(q, cos, sin, positions, False)
    k_rotated = rotate_reference(k, cos, sin, positions, False)
    coeff = jnp.stack((cos[positions], sin[positions]))
    kernel = make_flash_attn_mha(
        2, 128, 128, causal=False, block_sizes=BlockSizes(128, 128, 128, 64),
        head_fold=2, interpret=INTERPRET, q_scale=0.125)
    head_major = lambda x: x[:, :, 0, :]
    expected = kernel(*map(head_major, (q_rotated, k_rotated, v)))
    actual = kernel(*map(head_major, (q, k, v)), rotary=(coeff, coeff),
                    rotary_interleaved=False)
    assert_rope_close(actual, expected)


@pytest.mark.parametrize("token_major", [False, True])
@pytest.mark.parametrize("window", [(-1, -1), (32, 16)])
def test_varlen_rotary_d64_window_and_unused_capacity(token_major, window):
    # Note (david): the last 15 buffer rows belong to no sequence, so only the
    # first 241 rows are defined.
    q, k, v = (x[0] for x in random_qkv(77, 1, 256, 4, 4, 64, jnp.bfloat16))
    cu = jnp.array([0, 37, 151, 241], jnp.int32)
    positions = np.concatenate([np.arange(n) for n in (37, 114, 90)]
                               + [np.zeros(15, dtype=int)])
    cos, sin = rotary_tables(128, 64)
    q_rotated = rotate_reference(q, cos, sin, positions, False)
    k_rotated = rotate_reference(k, cos, sin, positions, False)
    kwargs = dict(max_seqlen_q=128, max_seqlen_k=128, causal=False,
                  window_size=window, token_major=token_major,
                  interpret=INTERPRET)
    expected = varlen_attn(q_rotated, k_rotated, v, cu, cu, **kwargs)
    actual = varlen_attn(q, k, v, cu, cu, **kwargs, rotary_cos=cos,
                         rotary_sin=sin, rotary_interleaved=False)
    assert_rope_close(actual[:241], expected[:241], atol=3e-3)


@pytest.mark.parametrize("batch,seqlen", [(1, 256), (2, 1024)])
@pytest.mark.parametrize("nheads,nheads_k", [(8, 2), (4, 1)])
def test_dense_rotary_q_only(batch, seqlen, nheads, nheads_k):
    # Note (david): external and fused RoPE can round differently at bf16
    # boundaries under interpret mode, hence attention-level tolerances.
    q, k, v = random_qkv(101, batch, seqlen, nheads, nheads_k, HEAD_DIM,
                         jnp.bfloat16)
    cos, sin = rotary_tables(seqlen, HEAD_DIM, jnp.float32, theta=1.0e6)
    positions = np.broadcast_to(np.arange(seqlen), (batch, seqlen))
    q_rotated = rotate_reference(q, cos, sin, positions, False)
    k_rotated = rotate_reference(k, cos, sin, positions, False)
    expected = dense_head_major(q_rotated, k_rotated, v, causal=True)
    actual = dense_head_major(
        q, k_rotated, v, causal=True, rotary_cos=cos, rotary_sin=sin,
        rotary_interleaved=False, rotary_k=False)
    assert_rope_close(actual, expected, atol=8e-3, rtol=2e-2)


def test_dense_rotary_float32_tables():
    q, k, v = random_qkv(102, 2, 256, 8, 2, HEAD_DIM, jnp.bfloat16)
    cos, sin = rotary_tables(256, HEAD_DIM, jnp.float32, theta=1.0e6)
    positions = np.broadcast_to(np.arange(256), (2, 256))
    q_rotated = rotate_reference(q, cos, sin, positions, False)
    k_rotated = rotate_reference(k, cos, sin, positions, False)
    expected = dense_head_major(q_rotated, k_rotated, v, causal=True)
    actual = dense_head_major(q, k, v, causal=True, rotary_cos=cos,
                              rotary_sin=sin, rotary_interleaved=False)
    assert_rope_close(actual, expected, atol=8e-3, rtol=2e-2)


def test_varlen_rotary_q_only():
    lengths = [300, 180, 32]
    total = sum(lengths)
    keys = jax.random.split(jax.random.key(103), 3)
    q = jax.random.normal(keys[0], (total, 8, HEAD_DIM), jnp.bfloat16)
    k = jax.random.normal(keys[1], (total, 2, HEAD_DIM), jnp.bfloat16)
    v = jax.random.normal(keys[2], (total, 2, HEAD_DIM), jnp.bfloat16)
    cu = jnp.asarray(np.cumsum([0, *lengths]), jnp.int32)
    positions = np.concatenate([np.arange(n) for n in lengths])
    cos, sin = rotary_tables(512, HEAD_DIM, jnp.float32, theta=1.0e6)
    q_rotated = rotate_reference(q, cos, sin, positions, False)
    k_rotated = rotate_reference(k, cos, sin, positions, False)
    kwargs = dict(max_seqlen_q=512, max_seqlen_k=512, causal=True,
                  interpret=INTERPRET, block_sizes=(128, 128, 64, 128))
    expected = head_major_varlen_flash_attn(q_rotated, k_rotated, v, cu, cu,
                                            **kwargs)
    actual = head_major_varlen_flash_attn(
        q, k_rotated, v, cu, cu, **kwargs, rotary_cos=cos, rotary_sin=sin,
        rotary_interleaved=False, rotary_k=False)
    assert_rope_close(actual, expected, atol=8e-3, rtol=2e-2)


@pytest.mark.parametrize("batch,seqlen", [(1, 1024), (4, 256)])
def test_heads_outer_fold_rotary_q_only(batch, seqlen):
    q, k, v = random_qkv(105, batch, seqlen, 8, 2, HEAD_DIM, jnp.bfloat16)
    cos, sin = rotary_tables(seqlen, HEAD_DIM, jnp.float32, theta=1.0e6)
    positions = np.broadcast_to(np.arange(seqlen), (batch, seqlen))
    q_rotated = rotate_reference(q, cos, sin, positions, False)
    k_rotated = rotate_reference(k, cos, sin, positions, False)
    expected = dense_head_major(q_rotated, k_rotated, v, causal=True)
    actual = dense_head_major(
        q, k_rotated, v, causal=True, heads_outer_fold=True, rotary_cos=cos,
        rotary_sin=sin, rotary_interleaved=False, rotary_k=False)
    assert_rope_close(actual, expected, atol=8e-3, rtol=2e-2)
