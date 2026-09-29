"""Head-major flash_attn_func / flash_attn_varlen_func against a naive reference."""

import functools
import itertools
import math

import jax
import jax.numpy as jnp
import pytest

from flywheel_tpu import flash_attn_interface, tuned_block_sizes
from flywheel_tpu.pallas.block_sizes import BlockSizes
from flywheel_tpu.pallas.flash_fwd import make_flash_attn_mha
from tests.reference import (
    attention_ref,
    head_major_flash_attn,
    head_major_varlen_flash_attn,
    random_qkv,
    rel_err,
)

# Note (david): token-major has its own suite in test_token_major.py. The
# dense cases keep the reference's (batch, seqlen, nheads, headdim) arrays and
# the packed ones its (total, nheads, headdim) arrays, and reach the head-major
# APIs through head_major_flash_attn / head_major_varlen_flash_attn.
flash_attn_func = head_major_flash_attn
flash_attn_varlen_func = head_major_varlen_flash_attn

ON_TPU = jax.default_backend() == "tpu"
INTERPRET = not ON_TPU
# Note (david): interpret mode emulates the kernel on CPU and only affords
# short sequences.
SEQLENS = [128, 256, 1024, 2048] if ON_TPU else [128, 256]
# Note (david): the boundaries sit off every block edge, so blocks straddle
# sequences and each q block reaches a different number of kv blocks.
PACKED_MANY = (100, 128, 52, 90, 128, 76, 110, 128, 84, 128)


def assert_close_to_reference(out, q, k, v, rows=slice(None), **mask_kwargs):
    # Note (david): flash-attention's rule: the kernel may drift at most twice
    # as far from the f32 reference as the same math run in bf16.
    out_ref = attention_ref(q, k, v, upcast=True, **mask_kwargs)
    out_lp = attention_ref(q, k, v, upcast=False, **mask_kwargs)
    assert out.shape == out_ref.shape and out.dtype == q.dtype
    # Note (david): an empty packed sequence reduces over zero rows.
    err = jnp.abs(out[:, rows].astype(jnp.float32)
                  - out_ref[:, rows].astype(jnp.float32)).max(initial=0.0)
    err_lp = jnp.abs(out_lp[:, rows].astype(jnp.float32)
                     - out_ref[:, rows].astype(jnp.float32)).max(initial=0.0)
    assert err <= 2 * err_lp + 1e-5, (err, err_lp)


def assert_lse_close(lse, q, k, v, **mask_kwargs):
    _, lse_ref = attention_ref(q, k, v, return_lse=True, **mask_kwargs)
    finite = jnp.isfinite(lse_ref)
    assert lse.dtype == jnp.float32
    assert jnp.array_equal(jnp.isfinite(lse), finite)
    err = jnp.abs(jnp.where(finite, lse - lse_ref, 0.0)).max(initial=0.0)
    assert err <= 1e-2, err


def input_grads(fn, q, k, v, dout):
    return jax.grad(
        lambda *qkv: (fn(*qkv) * dout).sum(), argnums=(0, 1, 2))(q, k, v)


def assert_grads_close_to_reference(q, k, v, causal):
    dout = jax.random.normal(jax.random.PRNGKey(7), q.shape, dtype=q.dtype)
    grads = input_grads(
        functools.partial(flash_attn_func, causal=causal, interpret=INTERPRET),
        q, k, v, dout)
    grads_ref = input_grads(
        functools.partial(attention_ref, causal=causal), q, k, v, dout)
    for name, grad, grad_ref in zip("qkv", grads, grads_ref):
        assert grad.shape == grad_ref.shape and grad.dtype == q.dtype, name
        assert rel_err(grad, grad_ref) < 3e-2, (name, rel_err(grad, grad_ref))


def cu_seqlens_of(seqlens):
    return jnp.array([0, *itertools.accumulate(seqlens)], jnp.int32)


def packed_qkv(seqlens_q, nheads, nheads_k, headdim, seqlens_kv=None):
    """Sequences packed end to end, plus each one as its own batch-1 input."""
    seqlens_kv = seqlens_q if seqlens_kv is None else seqlens_kv
    segments = [
        random_qkv(index, 1, len_q, nheads, nheads_k, headdim, jnp.bfloat16,
                   seqlen_kv=len_kv)
        for index, (len_q, len_kv) in enumerate(zip(seqlens_q, seqlens_kv))
    ]
    q, k, v = (
        jnp.concatenate([segment[axis][0] for segment in segments])
        for axis in range(3)
    )
    return q, k, v, segments


def run_head_major_kernel(kernel, q, k, v, *cu_seqlens):
    # Note (david): make_flash_attn_mha works in exp2, so q is pre-scaled by
    # log2(e) and the reference runs with softmax_scale 1.
    q_scaled = (q.astype(jnp.float32) * math.log2(math.e)).astype(q.dtype)
    out = kernel(
        *(x[0].transpose(1, 0, 2) for x in (q_scaled, k, v)), *cu_seqlens)
    return out.transpose(1, 0, 2)[None]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("nheads,nheads_k", [(4, 4), (4, 2), (4, 1)])
@pytest.mark.parametrize("headdim", [64, 128, 256])
@pytest.mark.parametrize("seqlen", SEQLENS)
def test_flash_attn_output(seqlen, headdim, nheads, nheads_k, causal):
    q, k, v = random_qkv(0, 2, seqlen, nheads, nheads_k, headdim, jnp.bfloat16)
    out = flash_attn_func(q, k, v, causal=causal, interpret=INTERPRET)
    assert_close_to_reference(out, q, k, v, causal=causal)


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_overflow_guard_recovery(causal):
    # Note (david): scaled keys in the last quarter push late row maxima far past
    # the guard threshold anchored on the first kv block, so the recovery
    # replay runs, while early causal rows never see them and stay on the fast
    # path.
    seqlen = 1024 if ON_TPU else 256
    q, k, v = random_qkv(0, 2, seqlen, 4, 4, 64, jnp.bfloat16)
    k = k.at[:, 3 * seqlen // 4:].multiply(40.0)
    out = flash_attn_func(q, k, v, causal=causal, interpret=INTERPRET)
    assert_close_to_reference(out, q, k, v, causal=causal)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("nheads,nheads_k", [(4, 4), (4, 2), (4, 1)])
@pytest.mark.parametrize("seqlen", SEQLENS)
def test_flash_attn_softmax_lse(seqlen, nheads, nheads_k, causal):
    q, k, v = random_qkv(0, 2, seqlen, nheads, nheads_k, 64, jnp.bfloat16)
    out, lse = flash_attn_func(
        q, k, v, causal=causal, return_softmax_lse=True, interpret=INTERPRET)
    assert_close_to_reference(out, q, k, v, causal=causal)
    assert_lse_close(lse, q, k, v, causal=causal)


@pytest.mark.parametrize("headdim,headdim_v", [(128, 64), (64, 128)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seqlen", SEQLENS)
def test_flash_attn_head_dim_v(seqlen, causal, headdim, headdim_v):
    q, k, v = random_qkv(
        0, 2, seqlen, 4, 2, headdim, jnp.bfloat16, headdim_v=headdim_v)
    out = flash_attn_func(q, k, v, causal=causal, interpret=INTERPRET)
    assert_close_to_reference(out, q, k, v, causal=causal)


@pytest.mark.parametrize("window_size", [(16, 0), (0, 16), (16, 16), (16, -1)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seqlen", SEQLENS)
def test_flash_attn_window_size(seqlen, causal, window_size):
    q, k, v = random_qkv(0, 2, seqlen, 4, 2, 64, jnp.bfloat16)
    out = flash_attn_func(
        q, k, v, causal=causal, window_size=window_size, interpret=INTERPRET)
    assert_close_to_reference(
        out, q, k, v, causal=causal, window_size=window_size)


def test_flash_attn_window_size_crosses_kv_compute_tiles():
    # Note (david): seqlen 768 splits kv into two compute tiles, and a right
    # bound reaching across them takes the fragment-skip path that a causal
    # diagonal never reaches.
    q, k, v = random_qkv(0, 1, 768, 2, 2, 64, jnp.bfloat16)
    out = flash_attn_func(
        q, k, v, causal=False, window_size=(0, 400), interpret=INTERPRET)
    assert_close_to_reference(out, q, k, v, causal=False, window_size=(0, 400))


def test_flash_attn_head_major_layout():
    # Note (david): the API itself, without the test adapter: (batch, nheads,
    # seqlen, headdim) in and out.
    q, k, v = random_qkv(0, 2, 200, 4, 2, 128, jnp.bfloat16)
    q_bhsd, k_bhsd, v_bhsd = (x.transpose(0, 2, 1, 3) for x in (q, k, v))
    out, lse = flash_attn_interface.flash_attn_func(
        q_bhsd, k_bhsd, v_bhsd, causal=True, return_softmax_lse=True,
        token_major=False, interpret=INTERPRET)
    assert out.shape == q_bhsd.shape and lse.shape == (2, 4, 200)
    assert_close_to_reference(out.transpose(0, 2, 1, 3), q, k, v, causal=True)
    assert_lse_close(lse, q, k, v, causal=True)


def test_flash_attn_attention_chunk_unsupported():
    q, k, v = random_qkv(0, 1, 256, 2, 2, 128, jnp.bfloat16)
    with pytest.raises(NotImplementedError, match="attention_chunk"):
        flash_attn_func(q, k, v, causal=True, attention_chunk=128,
                        interpret=INTERPRET)


@pytest.mark.parametrize("softcap", [30.0, 50.0])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seqlen", SEQLENS)
def test_flash_attn_softcap(seqlen, causal, softcap):
    q, k, v = random_qkv(0, 2, seqlen, 4, 2, 64, jnp.bfloat16)
    out = flash_attn_func(
        q, k, v, causal=causal, softcap=softcap, interpret=INTERPRET)
    assert_close_to_reference(out, q, k, v, causal=causal, softcap=softcap)


@pytest.mark.parametrize("seqlen", [120, 300, 1000])
def test_flash_attn_unaligned_seqlen(seqlen):
    # Note (david): only a right bound of 0 keeps real queries off the padded kv
    # tail, so unaligned dense attention is causal-only.
    q, k, v = random_qkv(0, 2, seqlen, 4, 2, 64, jnp.bfloat16)
    out = flash_attn_func(q, k, v, causal=True, interpret=INTERPRET)
    assert_close_to_reference(out, q, k, v, causal=True)


@pytest.mark.parametrize(
    "seqlen_q,seqlen_kv,causal",
    [
        (256, 128, False),
        (256, 128, True),
        (128, 384, False),
        (128, 384, True),
        (200, 120, True),
        (120, 333, True),
    ],
)
def test_flash_attn_rect(seqlen_q, seqlen_kv, causal):
    q, k, v = random_qkv(
        0, 2, seqlen_q, 4, 2, 64, jnp.bfloat16, seqlen_kv=seqlen_kv)
    out = flash_attn_func(q, k, v, causal=causal, interpret=INTERPRET)
    assert_close_to_reference(out, q, k, v, causal=causal)


@pytest.mark.parametrize(
    "seqlen_q,seqlen_kv,causal,window_size",
    [
        (256, 128, False, (-1, -1)),
        (256, 128, True, (-1, -1)),
        (128, 384, False, (-1, -1)),
        (128, 384, True, (-1, -1)),
        (200, 120, True, (-1, -1)),
        (256, 128, False, (32, 32)),
    ],
)
def test_flash_attn_rect_softmax_lse(seqlen_q, seqlen_kv, causal, window_size):
    q, k, v = random_qkv(
        0, 2, seqlen_q, 4, 2, 64, jnp.bfloat16, seqlen_kv=seqlen_kv)
    out, lse = flash_attn_func(
        q, k, v, causal=causal, window_size=window_size,
        return_softmax_lse=True, interpret=INTERPRET)
    assert_close_to_reference(
        out, q, k, v, causal=causal, window_size=window_size)
    assert_lse_close(lse, q, k, v, causal=causal, window_size=window_size)
    # Note (david): rows with no visible key are defined as exact zeros.
    right = 0 if causal else window_size[1]
    num_empty_rows = 0 if right == -1 else max(0, seqlen_q - seqlen_kv - right)
    assert jnp.all(out[:, :num_empty_rows] == 0)


@pytest.mark.parametrize("causal,window_size", [(True, (64, 0)), (False, (32, 32))])
@pytest.mark.parametrize("seqlen_q,seqlen_kv", [(256, 128), (128, 384)])
def test_flash_attn_rect_window_size(seqlen_q, seqlen_kv, causal, window_size):
    q, k, v = random_qkv(
        0, 2, seqlen_q, 4, 2, 64, jnp.bfloat16, seqlen_kv=seqlen_kv)
    out = flash_attn_func(
        q, k, v, causal=causal, window_size=window_size, interpret=INTERPRET)
    assert_close_to_reference(
        out, q, k, v, causal=causal, window_size=window_size)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("nheads,nheads_k", [(4, 4), (4, 2), (4, 1)])
def test_flash_attn_rect_gqa(nheads, nheads_k, causal):
    q, k, v = random_qkv(
        0, 2, 256, nheads, nheads_k, 64, jnp.bfloat16, seqlen_kv=128)
    out = flash_attn_func(q, k, v, causal=causal, interpret=INTERPRET)
    assert_close_to_reference(out, q, k, v, causal=causal)


@pytest.mark.parametrize("headdim,headdim_v", [(128, 64), (64, 128)])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_rect_head_dim_v(causal, headdim, headdim_v):
    q, k, v = random_qkv(
        0, 2, 256, 4, 2, headdim, jnp.bfloat16, headdim_v=headdim_v,
        seqlen_kv=128)
    out = flash_attn_func(q, k, v, causal=causal, interpret=INTERPRET)
    assert_close_to_reference(out, q, k, v, causal=causal)


def test_flash_attn_seqlen_above_2048():
    # Note (david): 3072 is not a multiple of 2048, so the default block drops
    # to 1024 and the sequence spans several physical blocks.
    q, k, v = random_qkv(0, 1, 3072, 2, 2, 64, jnp.bfloat16)
    out = flash_attn_func(q, k, v, causal=True, interpret=INTERPRET)
    assert_close_to_reference(out, q, k, v, causal=True)


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_rect_short_q_long_kv(causal):
    # Note (david): the static-anchor softmax needs two q compute tiles, which
    # a 128-row q only gets if the default blocks halve the q axis alone.
    q, k, v = random_qkv(0, 2, 128, 4, 2, 64, jnp.bfloat16, seqlen_kv=1024)
    out = flash_attn_func(q, k, v, causal=causal, interpret=INTERPRET)
    assert_close_to_reference(out, q, k, v, causal=causal)


def test_flash_attn_rect_fully_masked_q_block():
    # Note (david): q 4096 against kv 128 leaves the whole first q block without
    # a visible key, so the schedule must skip that block row.
    q, k, v = random_qkv(0, 1, 4096, 2, 2, 64, jnp.bfloat16, seqlen_kv=128)
    out = flash_attn_func(q, k, v, causal=True, interpret=INTERPRET)
    assert_close_to_reference(out, q, k, v, causal=True)
    assert jnp.all(out[:, :4096 - 128] == 0)


def test_flash_attn_causal_unequal_physical_blocks():
    # Note (david): fragment skipping assumes block_q == block_kv, so an
    # explicit block_q != block_kv must mask every fragment of a partial block.
    q, k, v = random_qkv(0, 1, 256, 2, 2, 64, jnp.bfloat16)
    kernel = make_flash_attn_mha(
        2, 256, 256, causal=True,
        block_sizes=BlockSizes(block_q=256, block_kv=128, block_kv_compute=128,
                               block_q_compute=128),
        num_kv_heads=2, interpret=INTERPRET)
    out = run_head_major_kernel(kernel, q, k, v)
    assert_close_to_reference(out, q, k, v, causal=True, softmax_scale=1.0)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("batch,seqlen", [(1, 1024), (2, 256), (4, 512)])
@pytest.mark.parametrize("nheads,nheads_k", [(8, 2), (8, 8), (4, 1)])
def test_flash_attn_heads_outer_fold(batch, seqlen, nheads, nheads_k, causal):
    # Note (david): the fold only renumbers the kernel's leading axis, so the
    # result must be bitwise the batch-outer one.
    q, k, v = random_qkv(104, batch, seqlen, nheads, nheads_k, 128,
                         jnp.bfloat16)
    expected = flash_attn_func(q, k, v, causal=causal, interpret=INTERPRET)
    actual = flash_attn_func(
        q, k, v, causal=causal, heads_outer_fold=True, interpret=INTERPRET)
    assert jnp.array_equal(actual, expected)


@pytest.mark.parametrize("seqlen,causal", [(512, False), (512, True), (1000, True)])
def test_flash_attn_tuned_head_fold(monkeypatch, seqlen, causal):
    # Note (david): the tuned key carries the flat head count and the TPU v6
    # table has no d256 entry for 4 heads, so the lookup misses and the analytic
    # defaults run; seqlen 1000 looks up its padded length 1024.
    monkeypatch.setattr(tuned_block_sizes, "get_device_name", lambda: "TPU v6")
    flash_attn_interface.get_kernel.cache_clear()
    q, k, v = random_qkv(0, 2, seqlen, 2, 2, 256, jnp.bfloat16)
    out = flash_attn_func(q, k, v, causal=causal, interpret=INTERPRET)
    assert_close_to_reference(out, q, k, v, causal=causal)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("headdim", [64, 128])
@pytest.mark.parametrize("seqlen", SEQLENS)
def test_flash_attn_backward(seqlen, headdim, causal):
    q, k, v = random_qkv(0, 2, seqlen, 4, 4, headdim, jnp.bfloat16)
    assert_grads_close_to_reference(q, k, v, causal)


@pytest.mark.parametrize("seqlen", [192, 1000] if ON_TPU else [192])
def test_flash_attn_backward_unaligned(seqlen):
    q, k, v = random_qkv(0, 2, seqlen, 4, 4, 64, jnp.bfloat16)
    assert_grads_close_to_reference(q, k, v, causal=True)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("nheads_k", [2, 1])
def test_flash_attn_backward_gqa(causal, nheads_k):
    q, k, v = random_qkv(0, 2, 256, 4, nheads_k, 64, jnp.bfloat16)
    assert_grads_close_to_reference(q, k, v, causal)


@pytest.mark.parametrize(
    "seqlen_q,seqlen_kv,causal",
    [
        (256, 128, False),
        (256, 128, True),
        (128, 384, False),
        (128, 384, True),
        (100, 150, True),
        # Note (david): only q pads here, so non-causal stays dense; the padded q
        # rows are harmless only because the output slice hands them a zero
        # cotangent.
        (100, 128, False),
        (100, 128, True),
    ],
)
def test_flash_attn_rect_backward(seqlen_q, seqlen_kv, causal):
    q, k, v = random_qkv(
        0, 2, seqlen_q, 4, 4, 64, jnp.bfloat16, seqlen_kv=seqlen_kv)
    assert_grads_close_to_reference(q, k, v, causal)


@pytest.mark.parametrize(
    "seqlens_q,seqlens_kv,block_sizes",
    [
        pytest.param([37, 53, 22], [37, 53, 22], None, id="unaligned"),
        pytest.param([37, 53, 22], [37, 53, 22], (256, 256, 128, 128),
                     id="wide_blocks"),
        # Note (david): 64-wide q compute tiles keep the two tiles per block the
        # fragment pipeline needs at 128-wide blocks.
        pytest.param([128, 0, 37, 256], [128, 0, 37, 256], (128, 128, 64, 128),
                     id="empty_and_block_multiple"),
        pytest.param([37, 5, 64, 22], [90, 64, 64, 130], None,
                     id="chunked_prefill"),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_varlen_func(seqlens_q, seqlens_kv, block_sizes, causal):
    # Note (david): each packed sequence must match itself run alone, which
    # catches any leak across a sequence boundary.
    q, k, v, segments = packed_qkv(seqlens_q, 4, 2, 64, seqlens_kv)
    cu_q = cu_seqlens_of(seqlens_q)
    out = flash_attn_varlen_func(
        q, k, v, cu_q, cu_seqlens_of(seqlens_kv), max(seqlens_q),
        max(seqlens_kv), causal=causal, interpret=INTERPRET,
        block_sizes=block_sizes)
    for index, segment in enumerate(segments):
        rows = slice(int(cu_q[index]), int(cu_q[index + 1]))
        assert_close_to_reference(out[rows][None], *segment, causal=causal)


@pytest.mark.parametrize(
    "seqlens,nheads_k,block_sizes",
    [
        pytest.param([37, 53, 22], 2, None, id="gqa"),
        # Note (david): starts off the 8-token grid make every first q block
        # reach back into its predecessor, so lse goes through the boundary
        # blend exactly as O does.
        pytest.param([37, 53, 22, 100], 4, None, id="unaligned"),
        # Note (david): an empty sequence owns no q block, so the next block
        # blends from two sequences back.
        pytest.param([45, 0, 61, 0, 30], 4, None, id="empty_sequence"),
        # Note (david): the last short sequence clamps its q window to the padded
        # end, the widest blend the per-seq schedule can need.
        pytest.param([384, 96, 32], 4, (256, 256, 128, 256), id="clamped_tail"),
        # Note (david): cu_q on the block grid starts every sequence on its own
        # block, so no staged window carries rows of the previous sequence and
        # the boundary blend never runs.
        pytest.param([128, 256, 128], 2, (128, 128, 64, 128),
                     id="block_aligned"),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_varlen_softmax_lse(seqlens, nheads_k, block_sizes, causal):
    q, k, v, _ = packed_qkv(seqlens, 4, nheads_k, 64)
    cu = cu_seqlens_of(seqlens)
    out, lse = flash_attn_varlen_func(
        q, k, v, cu, cu, max(seqlens), max(seqlens), causal=causal,
        return_softmax_lse=True, interpret=INTERPRET, block_sizes=block_sizes)
    assert_close_to_reference(
        out[None], q[None], k[None], v[None], causal=causal,
        cu_seqlens=(cu, cu))
    assert_lse_close(
        lse[None], q[None], k[None], v[None], causal=causal,
        cu_seqlens=(cu, cu))


@pytest.mark.parametrize(
    "seqlens_q,seqlens_kv",
    [
        # Note (david): the q and kv totals pad to different 128-multiples, in
        # both directions, and a q pad longer than the kv pad leaves pad rows
        # with no reachable key.
        ([17, 32, 5], [40, 32, 60]),
        ([0, 12, 1], [24, 30, 100]),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_varlen_unequal_totals(seqlens_q, seqlens_kv, causal):
    q, k, v, segments = packed_qkv(seqlens_q, 4, 2, 64, seqlens_kv)
    cu_q = cu_seqlens_of(seqlens_q)
    out, lse = flash_attn_varlen_func(
        q, k, v, cu_q, cu_seqlens_of(seqlens_kv), max(seqlens_q),
        max(seqlens_kv), causal=causal, return_softmax_lse=True,
        interpret=INTERPRET)
    for index, segment in enumerate(segments):
        rows = slice(int(cu_q[index]), int(cu_q[index + 1]))
        assert_close_to_reference(out[rows][None], *segment, causal=causal)
        assert_lse_close(lse[None, :, rows], *segment, causal=causal)


def test_flash_attn_varlen_differing_q_kv_splits():
    # Note (david): equal totals but cu_q != cu_k make the packed mask a general
    # block-diagonal; non-causal because sequence 0 has more queries than keys.
    q, k, v = (x[0] for x in random_qkv(0, 1, 128, 2, 2, 64, jnp.bfloat16))
    cu_q = jnp.array([0, 64, 128], jnp.int32)
    cu_k = jnp.array([0, 32, 128], jnp.int32)
    out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, 64, 96, interpret=INTERPRET)
    assert_close_to_reference(
        out[None], q[None], k[None], v[None], cu_seqlens=(cu_q, cu_k))


@pytest.mark.parametrize(
    "seqlens_q,seqlens_kv,block_sizes",
    [
        pytest.param([37, 53, 22], [37, 53, 22], None, id="unaligned"),
        pytest.param([37, 5, 64, 22], [90, 64, 64, 130], None,
                     id="chunked_prefill"),
        # Note (david): 128-wide blocks under 300-token sequences put a left
        # edge several kv blocks past each sequence's first block.
        pytest.param([300, 45, 200], [300, 60, 330], (128, 128, 64, 128),
                     id="left_edge_skips_blocks"),
    ],
)
@pytest.mark.parametrize(
    "causal,window_size",
    [(True, (16, 0)), (True, (160, 0)), (False, (8, 8)), (False, (24, -1)),
     (False, (-1, 4)), (False, (0, 0))],
)
def test_flash_attn_varlen_window(
        seqlens_q, seqlens_kv, block_sizes, causal, window_size):
    # Note (david): the window is bottom-right aligned per sequence, so each
    # packed sequence must match itself run alone under the same window.
    q, k, v, segments = packed_qkv(seqlens_q, 4, 2, 64, seqlens_kv)
    cu_q = cu_seqlens_of(seqlens_q)
    out, lse = flash_attn_varlen_func(
        q, k, v, cu_q, cu_seqlens_of(seqlens_kv), max(seqlens_q),
        max(seqlens_kv), causal=causal, window_size=window_size,
        return_softmax_lse=True, interpret=INTERPRET, block_sizes=block_sizes)
    for index, segment in enumerate(segments):
        rows = slice(int(cu_q[index]), int(cu_q[index + 1]))
        assert_close_to_reference(
            out[rows][None], *segment, causal=causal, window_size=window_size)
        assert_lse_close(
            lse[None, :, rows], *segment, causal=causal,
            window_size=window_size)


def test_flash_attn_varlen_is_forward_only():
    seqlens = [37, 53, 22]
    cu = cu_seqlens_of(seqlens)
    q, k, v, _ = packed_qkv(seqlens, 2, 2, 64)

    def loss(q):
        return flash_attn_varlen_func(
            q, k, v, cu, cu, max(seqlens), max(seqlens), causal=True,
            interpret=INTERPRET).astype(jnp.float32).sum()

    with pytest.raises(NotImplementedError, match="flash_attn_varlen_func"):
        jax.grad(loss)(q)


def test_flash_attn_varlen_folded_softmax_lse(monkeypatch):
    monkeypatch.setattr(
        flash_attn_interface, "get_varlen_head_fold",
        lambda key, num_heads, head_dim, head_dim_v: 4)
    flash_attn_interface.get_kernel.cache_clear()
    q, k, v = random_qkv(19, 1, 512, 8, 8, 128, jnp.bfloat16)
    cu = jnp.array([0, 256, 512], jnp.int32)
    out, lse = flash_attn_varlen_func(
        q[0], k[0], v[0], cu, cu, 256, 256, return_softmax_lse=True,
        interpret=INTERPRET)
    assert_close_to_reference(out[None], q, k, v, cu_seqlens=(cu, cu))
    assert_lse_close(lse[None], q, k, v, cu_seqlens=(cu, cu))


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_varlen_many_blocks(causal):
    blocks = BlockSizes(block_q=128, block_kv=128, block_kv_compute=128,
                        block_q_compute=64, num_stages=3)
    total = sum(PACKED_MANY)
    cu = cu_seqlens_of(PACKED_MANY)
    q, k, v = random_qkv(0, 1, total, 2, 2, 64, jnp.bfloat16)
    kernel = make_flash_attn_mha(
        2, total, total, causal=causal, block_sizes=blocks, num_kv_heads=2,
        interpret=INTERPRET, varlen_max_seqlen_kv=max(PACKED_MANY))
    out = run_head_major_kernel(kernel, q, k, v, cu, cu)
    assert_close_to_reference(
        out, q, k, v, causal=causal, cu_seqlens=(cu, cu), softmax_scale=1.0)


def test_flash_attn_varlen_causal_below_diagonal_packed_block():
    # Note (david): one 500-token sequence in a 512 buffer puts q block 1 across
    # the real/pad boundary while strictly below the diagonal from kv block 0;
    # the masked pipeline copy must not apply the diagonal's fragment skip
    # there. The skip only fires with more than one kv compute tile per block.
    real = 500
    cu = jnp.array([0, real], jnp.int32)
    q, k, v = random_qkv(3, 1, 512, 2, 2, 64, jnp.bfloat16)
    kernel = make_flash_attn_mha(
        2, 512, 512, causal=True,
        block_sizes=BlockSizes(block_q=256, block_kv=256, block_kv_compute=128,
                               block_q_compute=128),
        num_kv_heads=2, interpret=INTERPRET, varlen_max_seqlen_kv=512)
    out = run_head_major_kernel(kernel, q, k, v, cu, cu)
    assert_close_to_reference(
        out, q, k, v, rows=slice(0, real), causal=True, cu_seqlens=(cu, cu),
        softmax_scale=1.0)
