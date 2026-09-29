"""Token-major (batch, seq, heads * head_dim) layout against head-major.

Both layouts run the same kernel math, so forward results must be bitwise
equal; accuracy against a reference is the head-major suites' job.
"""

import functools

import jax
import jax.numpy as jnp
import pytest

from flywheel_tpu import flash_attn_func, flash_attn_varlen_func
from flywheel_tpu.flash_attn_interface import default_block_sizes
from flywheel_tpu.pallas import flash_bwd
from flywheel_tpu.pallas.block_sizes import BlockSizes, TokenMajorInfo
from flywheel_tpu.pallas.flash_bwd import flash_attn_bwd
from flywheel_tpu.pallas.flash_fwd import make_flash_attn_mha
from tests.reference import (
    head_major_flash_attn,
    head_major_varlen_flash_attn,
    random_qkv,
)

INTERPRET = jax.default_backend() != "tpu"


def token_major_info(batch, nheads, nheads_k=None, head_dim=128,
                     head_dim_v=None):
    return TokenMajorInfo(
        batch=batch, num_q_heads=nheads,
        num_kv_heads=nheads if nheads_k is None else nheads_k,
        head_dim_qk=head_dim,
        head_dim_v=head_dim if head_dim_v is None else head_dim_v)


def to_token_major(x):
    return x.reshape(x.shape[0], x.shape[1], -1)


def fold_heads(x):
    """(batch, seq, heads, dim) -> the head-major kernel's (batch * heads, seq,
    dim)."""
    return x.transpose(0, 2, 1, 3).reshape(-1, x.shape[1], x.shape[3])


def assert_grad_parity(grad_tm, grad_hm):
    # Note (david): token-major reduces di over its own fused layout, so its f32
    # accumulation order differs from head-major's; on v6e under 0.3% of
    # elements move, by at most 4.9e-4, all at bf16 rounding boundaries.
    assert grad_tm.shape == grad_hm.shape
    assert jnp.allclose(grad_tm.astype(jnp.float32),
                        grad_hm.astype(jnp.float32), rtol=2 ** -7,
                        atol=2 ** -9)


def dense_forward_parity(q4, k4, v4, *, causal, head_fold=1, window=None,
                         causal_offset=0, softcap=0.0, return_lse=False,
                         tm_head_fold=None, block_sizes=None):
    """Head-major and token-major dense kernels on the same (batch, seq, heads,
    dim) data must agree bitwise.

    tm_head_fold overrides the token-major fold alone: d64 token-major needs an
    even fold, which the head-major build only accepts on a single block.
    """
    batch, seqlen_q, nheads, headdim = q4.shape
    seqlen_kv, nheads_k, headdim_v = k4.shape[1], k4.shape[2], v4.shape[3]
    common = dict(
        causal=causal, causal_offset=causal_offset, window=window,
        softcap=softcap, return_lse=return_lse, interpret=INTERPRET,
        block_sizes=(default_block_sizes(seqlen_q, seqlen_kv)
                     if block_sizes is None else block_sizes))
    head_major = make_flash_attn_mha(
        batch * nheads, seqlen_q, seqlen_kv, num_kv_heads=batch * nheads_k,
        head_fold=head_fold, **common)
    token_major = make_flash_attn_mha(
        nheads, seqlen_q, seqlen_kv, num_kv_heads=nheads_k,
        token_major=token_major_info(batch, nheads, nheads_k, headdim,
                                     headdim_v),
        head_fold=head_fold if tm_head_fold is None else tm_head_fold,
        **common)
    outputs_hm = head_major(fold_heads(q4), fold_heads(k4), fold_heads(v4))
    outputs_tm = token_major(
        to_token_major(q4), to_token_major(k4), to_token_major(v4))
    if return_lse:
        (out_hm, lse_hm), (out_tm, lse_tm) = outputs_hm, outputs_tm
        assert jnp.array_equal(lse_tm, lse_hm)
    else:
        out_hm, out_tm = outputs_hm, outputs_tm
    out_hm4 = out_hm.reshape(batch, nheads, seqlen_q, headdim_v).transpose(
        0, 2, 1, 3)
    assert jnp.array_equal(
        out_tm.reshape(batch, seqlen_q, nheads, headdim_v), out_hm4)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("nheads,nheads_k", [(4, 4), (4, 2), (4, 1)])
@pytest.mark.parametrize("return_lse", [False, True])
def test_tm_dense_fwd_parity(causal, nheads, nheads_k, return_lse):
    q, k, v = random_qkv(0, 2, 256, nheads, nheads_k, 128, jnp.bfloat16)
    dense_forward_parity(q, k, v, causal=causal, return_lse=return_lse)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_fold", [2, 4])
@pytest.mark.parametrize("return_lse", [False, True])
def test_tm_dense_fwd_parity_head_fold(causal, head_fold, return_lse):
    # Note (david): with 4 heads per row, a fold of 2 also starts a group mid
    # row, and a fold of 4 takes a whole row, so both batch rows decode.
    q, k, v = random_qkv(4, 2, 256, 4, 4, 128, jnp.bfloat16)
    dense_forward_parity(
        q, k, v, causal=causal, head_fold=head_fold, return_lse=return_lse)


D64_MULTIBLOCK = BlockSizes(block_q=128, block_kv=128, block_kv_compute=128,
                            block_q_compute=64)


@pytest.mark.parametrize(
    "seed,shape,qkv_extra,parity_kwargs",
    [
        pytest.param(1, (2, 256, 4, 128), dict(headdim_v=256),
                     dict(causal=True), id="head_dim_v"),
        pytest.param(2, (2, 128, 4, 128), dict(seqlen_kv=256),
                     dict(causal=True, causal_offset=128), id="rect_offset"),
        pytest.param(3, (2, 256, 4, 128), {},
                     dict(causal=False, window=(64, 32)), id="window"),
        pytest.param(3, (2, 256, 4, 128), {},
                     dict(causal=True, softcap=30.0), id="softcap"),
        # Note (david): a fold of 16 needs 16 heads per batch row.
        *[pytest.param(5, (1, 128, 16, 128), {},
                       dict(causal=causal, head_fold=16),
                       id=f"fold16-causal{causal}")
          for causal in (False, True)],
        # Note (david): head_dim 64 runs paired: an even fold keeps every HBM DMA
        # group on whole 128-lane tiles.
        *[pytest.param(40, (2, 256, 4, 64), {},
                       dict(causal=causal, return_lse=return_lse,
                            tm_head_fold=2),
                       id=f"d64-causal{causal}-lse{return_lse}")
          for causal in (False, True) for return_lse in (False, True)],
        pytest.param(41, (2, 256, 4, 64), {},
                     dict(causal=True, tm_head_fold=4), id="d64_full_row"),
        *[pytest.param(42, (2, 256, 4, 64), {},
                       dict(causal=causal, tm_head_fold=2,
                            block_sizes=D64_MULTIBLOCK),
                       id=f"d64_multiblock-causal{causal}")
          for causal in (False, True)],
        pytest.param(43, (2, 128, 4, 64), dict(seqlen_kv=256),
                     dict(causal=True, causal_offset=128, tm_head_fold=2),
                     id="d64_rect_offset"),
    ],
)
def test_tm_dense_fwd_parity_cases(seed, shape, qkv_extra, parity_kwargs):
    batch, seqlen, nheads, headdim = shape
    q, k, v = random_qkv(seed, batch, seqlen, nheads, nheads, headdim,
                         jnp.bfloat16, **qkv_extra)
    dense_forward_parity(q, k, v, **parity_kwargs)


@pytest.mark.parametrize(
    "seed,nheads_k,headdim,causal,cu,head_fold,return_lse",
    [
        *[(6, nheads_k, 128, causal, [0, 100, 228, 256], 1, False)
          for nheads_k in (4, 2) for causal in (False, True)],
        (7, 4, 128, True, [0, 128, 256], 2, True),
        *[(44, 4, 64, causal, [0, 100, 228, 256], 2, False)
          for causal in (False, True)],
        (45, 4, 64, True, [0, 128, 256], 2, True),
    ],
)
def test_tm_varlen_fwd_parity(seed, nheads_k, headdim, causal, cu, head_fold,
                              return_lse):
    total, nheads, max_seqlen = 256, 4, 128
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    q = jax.random.normal(keys[0], (total, nheads, headdim), jnp.bfloat16)
    k = jax.random.normal(keys[1], (total, nheads_k, headdim), jnp.bfloat16)
    v = jax.random.normal(keys[2], (total, nheads_k, headdim), jnp.bfloat16)
    cu = jnp.asarray(cu, jnp.int32)
    common = dict(
        causal=causal, head_fold=head_fold, return_lse=return_lse,
        interpret=INTERPRET, varlen_max_seqlen_kv=max_seqlen,
        block_sizes=default_block_sizes(total, total, max_seqlen))
    head_major = make_flash_attn_mha(
        nheads, total, total, num_kv_heads=nheads_k, **common)
    token_major = make_flash_attn_mha(
        nheads, total, total, num_kv_heads=nheads_k,
        token_major=token_major_info(None, nheads, nheads_k, headdim),
        **common)
    outputs_hm = head_major(*(x.transpose(1, 0, 2) for x in (q, k, v)), cu, cu)
    outputs_tm = token_major(*(x.reshape(total, -1) for x in (q, k, v)), cu, cu)
    if return_lse:
        (out_hm, lse_hm), (out_tm, lse_tm) = outputs_hm, outputs_tm
        assert jnp.array_equal(lse_tm, lse_hm)
    else:
        out_hm, out_tm = outputs_hm, outputs_tm
    assert jnp.array_equal(out_tm.reshape(total, nheads, headdim),
                           out_hm.transpose(1, 0, 2))


@pytest.mark.parametrize("causal", [False, True])
def test_tm_hybrid_fwd_parity(causal):
    # Note (david): the runtime schedule on 3-D storage is the padded-dense
    # combination: one real 200-token sequence per row, the padded tail its
    # own kernel-internal sequence.
    batch, seqlen, nheads, headdim = 2, 256, 4, 128
    q, k, v = random_qkv(8, batch, seqlen, nheads, nheads, headdim,
                         jnp.bfloat16)
    cu = jnp.array([0, 200], jnp.int32)
    common = dict(causal=causal, interpret=INTERPRET,
                  varlen_max_seqlen_kv=256,
                  block_sizes=default_block_sizes(seqlen, seqlen, 256))
    head_major = make_flash_attn_mha(
        batch * nheads, seqlen, seqlen, num_kv_heads=batch * nheads, **common)
    token_major = make_flash_attn_mha(
        nheads, seqlen, seqlen, num_kv_heads=nheads,
        token_major=token_major_info(batch, nheads, nheads, headdim), **common)
    out_hm = head_major(fold_heads(q), fold_heads(k), fold_heads(v), cu, cu)
    out_tm = token_major(to_token_major(q), to_token_major(k),
                         to_token_major(v), cu, cu)
    out_hm4 = out_hm.reshape(batch, nheads, seqlen, headdim).transpose(
        0, 2, 1, 3)
    assert jnp.array_equal(out_tm.reshape(q.shape), out_hm4)


def interface_parity(q4, k4, v4, **kwargs):
    outputs4 = head_major_flash_attn(q4, k4, v4, interpret=INTERPRET, **kwargs)
    outputs3 = flash_attn_func(
        to_token_major(q4), to_token_major(k4), to_token_major(v4),
        head_dim=q4.shape[-1], token_major=True, interpret=INTERPRET,
        **kwargs)
    if kwargs.get("return_softmax_lse", False):
        (out4, lse4), (out3, lse3) = outputs4, outputs3
        assert jnp.array_equal(lse3, lse4)
    else:
        out4, out3 = outputs4, outputs3
    assert jnp.array_equal(out3, to_token_major(out4))


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("nheads,nheads_k", [(4, 4), (4, 2), (4, 1)])
def test_tm_interface_fwd_parity(causal, nheads, nheads_k):
    q, k, v = random_qkv(9, 2, 256, nheads, nheads_k, 128, jnp.bfloat16)
    interface_parity(q, k, v, causal=causal)


@pytest.mark.parametrize(
    "seed,seqlen,headdim_v,kwargs",
    [
        pytest.param(10, 200, 128, dict(causal=True), id="unaligned"),
        pytest.param(11, 256, 128, dict(causal=True, return_softmax_lse=True),
                     id="lse"),
        pytest.param(12, 256, 256, dict(causal=False), id="head_dim_v"),
    ],
)
def test_tm_interface_fwd_parity_cases(seed, seqlen, headdim_v, kwargs):
    q, k, v = random_qkv(seed, 2, seqlen, 4, 4, 128, jnp.bfloat16,
                         headdim_v=headdim_v)
    interface_parity(q, k, v, **kwargs)


def test_tm_interface_varlen_fwd_parity():
    total, nheads, headdim = 256, 4, 128
    keys = jax.random.split(jax.random.PRNGKey(13), 3)
    q, k, v = (jax.random.normal(key, (total, nheads, headdim), jnp.bfloat16)
               for key in keys)
    cu = jnp.array([0, 100, 256], jnp.int32)
    kwargs = dict(cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=156,
                  max_seqlen_k=156, causal=True, interpret=INTERPRET)
    out3 = head_major_varlen_flash_attn(q, k, v, **kwargs)
    out2 = flash_attn_varlen_func(
        q.reshape(total, -1), k.reshape(total, -1), v.reshape(total, -1),
        head_dim=headdim, token_major=True, **kwargs)
    assert jnp.array_equal(out2, out3.reshape(total, -1))


def test_tm_interface_cache_isolation():
    # Note (david): the token-major kernel cache keys per-row heads plus a
    # separate batch field, so batches 2 and 4 at 8 heads differ only in that
    # field and must still get their own kernels.
    qa, ka, va = random_qkv(15, 2, 256, 8, 8, 128, jnp.bfloat16)
    qb, kb, vb = random_qkv(16, 4, 256, 8, 8, 128, jnp.bfloat16)
    interface_parity(qa, ka, va, causal=True)
    interface_parity(qb, kb, vb, causal=True)


def backward_parity(q4, k4, v4, *, causal, causal_offset=0, head_fold=1,
                    tm_head_fold=1, block=None, block_kv=None):
    """flash_attn_bwd on both layouts from the same head-major forward.

    An explicit head_fold of 1 runs the same kernel on both sides; None lets
    each side pick its own fold.
    """
    batch, seqlen_q, nheads, _ = q4.shape
    out4, lse = head_major_flash_attn(
        q4, k4, v4, causal=causal, return_softmax_lse=True,
        interpret=INTERPRET)
    do4 = jax.random.normal(jax.random.PRNGKey(99), out4.shape, out4.dtype)
    lse = lse.reshape(batch * nheads, seqlen_q)
    kwargs = dict(causal=causal, causal_offset=causal_offset, block=block,
                  block_kv=block_kv, interpret=INTERPRET)
    grads_hm = flash_attn_bwd(
        *map(fold_heads, (q4, k4, v4, out4)), lse, fold_heads(do4),
        head_fold=head_fold, **kwargs)
    grads_tm = flash_attn_bwd(
        *map(to_token_major, (q4, k4, v4, out4)), lse, to_token_major(do4),
        token_major=True, head_dim=q4.shape[-1], head_fold=tm_head_fold,
        **kwargs)
    for grad_tm, grad_hm, primal in zip(grads_tm, grads_hm, (q4, k4, v4)):
        _, seqlen, heads, dim = primal.shape
        assert_grad_parity(
            grad_tm.reshape(primal.shape),
            grad_hm.reshape(batch, heads, seqlen, dim).transpose(0, 2, 1, 3))


@pytest.mark.parametrize(
    "seed,shape,qkv_extra,parity_kwargs",
    [
        *[pytest.param(20, (2, 256, 4, 4, 128), {}, dict(causal=causal),
                       id=f"mha-causal{causal}")
          for causal in (False, True)],
        pytest.param(21, (2, 256, 4, 4, 128), {},
                     dict(causal=True, block=128), id="multi_block"),
        pytest.param(22, (2, 256, 4, 4, 128), dict(headdim_v=256),
                     dict(causal=False), id="head_dim_v"),
        # Note (david): k/v carry the kv length while o/do carry the q length,
        # so the shape checks must not require the two to match.
        pytest.param(31, (2, 128, 4, 4, 128), dict(seqlen_kv=256),
                     dict(causal=False), id="rect"),
        *[pytest.param(23, (2, 256, 4, nheads_k, 128), {},
                       dict(causal=True, head_fold=None, tm_head_fold=None),
                       id=f"gqa{nheads_k}")
          for nheads_k in (2, 1)],
        # Note (david): the leading q rows are wholly below the diagonal, so the
        # post-kernel dq zero-fill runs on the 3-D token-major dq.
        pytest.param(23, (2, 256, 4, 4, 128), {},
                     dict(causal=True, causal_offset=-64), id="negative_offset"),
        # Note (david): the auto fold is picked from the flat head count (8 here)
        # and must clamp to the 2 heads of one batch row.
        pytest.param(34, (4, 256, 2, 2, 128), {},
                     dict(causal=True, head_fold=None, tm_head_fold=None),
                     id="auto_fold_clamp"),
        pytest.param(23, (2, 128, 4, 4, 128), {},
                     dict(causal=True, head_fold=2, tm_head_fold=2),
                     id="single_block_fold"),
        *[pytest.param(46, (2, 256, 4, 4, 64), {},
                       dict(causal=causal, tm_head_fold=None),
                       id=f"d64-causal{causal}")
          for causal in (False, True)],
        pytest.param(47, (2, 256, 4, 4, 64), {},
                     dict(causal=True, block=128, tm_head_fold=None),
                     id="d64_multi_block"),
    ],
)
def test_tm_bwd_parity(seed, shape, qkv_extra, parity_kwargs):
    q, k, v = random_qkv(seed, *shape, jnp.bfloat16, **qkv_extra)
    backward_parity(q, k, v, **parity_kwargs)


def test_tm_bwd_parity_gqa_q_chunked(monkeypatch):
    # Note (david): the dq_acc gate then fits two of the three q blocks, so the
    # token-major GQA build must run as q-row chunks sliced on the token axis,
    # or it raises.
    batch, seqlen, nheads, nheads_k, headdim = 2, 384, 4, 2, 128
    q, k, v = random_qkv(24, batch, seqlen, nheads, nheads_k, headdim,
                         jnp.bfloat16)
    monkeypatch.setattr(flash_bwd, "DQ_ACC_VMEM_LIMIT_BYTES",
                        2 * 128 * (nheads // nheads_k) * headdim * 4)
    backward_parity(q, k, v, causal=True, head_fold=None, tm_head_fold=None,
                    block=128, block_kv=128)


def dense_loss(q, k, v, *, attn, causal, **kwargs):
    return attn(q, k, v, causal=causal, interpret=INTERPRET,
                **kwargs).astype(jnp.float32).sum()


@pytest.mark.parametrize(
    "seed,seqlen,headdim,causal",
    [
        (30, 256, 128, False),
        (30, 256, 128, True),
        # Note (david): 200 tokens take the padded dense gradient path.
        (31, 200, 128, True),
        (49, 256, 64, False),
        (49, 256, 64, True),
    ],
)
def test_tm_grad_parity_dense(seed, seqlen, headdim, causal):
    q, k, v = random_qkv(seed, 2, seqlen, 4, 4, headdim, jnp.bfloat16)
    grads4 = jax.grad(
        functools.partial(dense_loss, attn=head_major_flash_attn,
                          causal=causal),
        argnums=(0, 1, 2))(q, k, v)
    grads3 = jax.grad(
        functools.partial(dense_loss, attn=flash_attn_func, causal=causal,
                          head_dim=headdim, token_major=True),
        argnums=(0, 1, 2))(*map(to_token_major, (q, k, v)))
    for grad3, grad4 in zip(grads3, grads4):
        assert_grad_parity(grad3, to_token_major(grad4))
