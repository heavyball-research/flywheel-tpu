"""Regression tests for kernel paths the random grids of test_flash_attn.py do
not reach: dense and packed flash_attn_func / flash_attn_varlen_func, and
single-token decode through flash_attn_with_kvcache."""

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flywheel_tpu import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
from flywheel_tpu.flash_attn_interface import (
    default_block_sizes,
    fused_q_scale,
    pad_seqlen,
)
from flywheel_tpu.pallas.block_sizes import next_pow2
from flywheel_tpu.pallas.flash_fwd import make_flash_attn_mha
from tests.test_util import attention_ref, randn

INTERPRET = jax.default_backend() != "tpu"


def to_head_major(x):
    """(batch, seqlen, nheads, d) <-> (batch, nheads, seqlen, d), or packed
    (total, nheads, d) <-> (nheads, total, d)."""
    return jnp.swapaxes(x, -3, -2)


def max_abs(x):
    return jnp.abs(x.astype(jnp.float32)).max(initial=0.0).item()


def assert_output_close(out, q, k, v, rows=slice(None), **mask_kwargs):
    """test_flash_attn.py's rule: at most twice the error of the bf16 reference
    with reordered ops, plus the rounding of arithmetic on the f32 reference."""
    out_ref, _ = attention_ref(q, k, v, **mask_kwargs)
    out_bf16, _ = attention_ref(
        q, k, v, upcast=False, reorder_ops=True, **mask_kwargs
    )
    assert out.shape == out_ref.shape and out.dtype == q.dtype
    out, out_ref, out_bf16 = out[:, rows], out_ref[:, rows], out_bf16[:, rows]
    atol = 2 * max_abs(out_ref + 0.3 - 0.3 - out_ref)
    assert max_abs(out - out_ref) <= 2 * max_abs(out_bf16 - out_ref) + atol


def dense_attn(q, k, v, **kwargs):
    """flash_attn_func on (batch, seqlen, nheads, d) arrays."""
    return to_head_major(
        flash_attn_func(
            *map(to_head_major, (q, k, v)), interpret=INTERPRET, **kwargs
        )
    )


def random_qkv(seed, batch, seqlen_q, nheads, nheads_k, d, seqlen_k=None):
    rng = np.random.default_rng(seed)
    seqlen_k = seqlen_q if seqlen_k is None else seqlen_k
    return (
        randn(rng, (batch, seqlen_q, nheads, d)),
        randn(rng, (batch, seqlen_k, nheads_k, d)),
        randn(rng, (batch, seqlen_k, nheads_k, d)),
    )


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_overflow_guard_recovery(causal):
    # Note (david): scaled keys in the last quarter push late row maxima far
    # past the guard threshold anchored on the first kv block, so the recovery
    # replay runs, while early causal rows never see them and stay on the fast
    # path.
    seqlen = 256 if INTERPRET else 1024
    q, k, v = random_qkv(0, 2, seqlen, 4, 4, 64)
    k = k.at[:, 3 * seqlen // 4 :].multiply(40.0)
    out = dense_attn(q, k, v, causal=causal)
    assert_output_close(out, q, k, v, causal=causal)


def test_flash_attn_window_size_crosses_kv_compute_tiles():
    # Note (david): seqlen 768 splits kv into two compute tiles, and a right
    # bound reaching across them ends each q tile's band past the diagonal,
    # where a causal plan never ends one.
    q, k, v = random_qkv(0, 1, 768, 2, 2, 64)
    out = dense_attn(q, k, v, causal=False, window_size=(0, 400))
    assert_output_close(out, q, k, v, causal=False, window_size=(0, 400))


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_rect_short_q_long_kv(causal):
    # Note (david): the static-anchor softmax needs two q compute tiles, which
    # a 128-row q only gets if the default blocks halve the q axis alone.
    q, k, v = random_qkv(0, 2, 128, 4, 2, 64, seqlen_k=1024)
    out = dense_attn(q, k, v, causal=causal)
    assert_output_close(out, q, k, v, causal=causal)


def test_flash_attn_rect_fully_masked_q_block():
    # Note (david): q 4096 against kv 128 leaves the whole first q block without
    # a visible key, so the schedule must skip that block row.
    q, k, v = random_qkv(0, 1, 4096, 2, 2, 64, seqlen_k=128)
    out = dense_attn(q, k, v, causal=True)
    assert_output_close(out, q, k, v, causal=True)
    assert jnp.all(out[:, : 4096 - 128] == 0)


def cu_seqlens_of(seqlens):
    return jnp.array([0, *itertools.accumulate(seqlens)], jnp.int32)


def packed_qkv(seqlens, nheads, nheads_k, d):
    """Sequences packed end to end, plus each one as its own batch-1 input."""
    segments = [
        random_qkv(index, 1, seqlen, nheads, nheads_k, d)
        for index, seqlen in enumerate(seqlens)
    ]
    q, k, v = (
        jnp.concatenate([segment[axis][0] for segment in segments])
        for axis in range(3)
    )
    return q, k, v, segments


def varlen_attn(q, k, v, *args, **kwargs):
    """flash_attn_varlen_func on packed (total, nheads, d) arrays; lse as is."""
    result = flash_attn_varlen_func(
        *map(to_head_major, (q, k, v)), *args, interpret=INTERPRET, **kwargs
    )
    if kwargs.get("return_softmax_lse", False):
        out, lse = result
        return to_head_major(out), lse
    return to_head_major(result)


@pytest.mark.parametrize(
    "seqlens",
    [
        pytest.param([50, 78], id="one_block"),
        pytest.param([37, 53, 22, 100, 300], id="several_blocks"),
    ],
)
def test_flash_attn_varlen_func_feeds_the_program(seqlens):
    # On v7x XLA lays a bf16 kernel output out in 16-row tiles when further ops
    # of the same program read it, as a model layer's do, and in 8-row tiles
    # when it is the program's output, as in every other test here. The
    # program has to compile and hand on what the kernel alone returns.
    q, k, v, _ = packed_qkv(seqlens, 4, 2, 128)
    cu = cu_seqlens_of(seqlens)

    def attend(q, k, v, cu):
        return varlen_attn(
            q, k, v, cu, cu, max(seqlens), max(seqlens), causal=True
        )

    @jax.jit
    def layer(q, k, v, cu):
        merged = attend(q, k, v, cu).reshape(q.shape[0], -1)
        return merged @ jnp.eye(merged.shape[1], dtype=merged.dtype)

    expected = attend(q, k, v, cu)
    assert jnp.array_equal(layer(q, k, v, cu).reshape(expected.shape), expected)


@pytest.mark.parametrize(
    "seqlens_q,num_pad",
    [
        ([17, 1, 0, 40], 23),
        ([5], 600),
        ([0, 0], 9),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_varlen_padding_rows_are_zero(seqlens_q, num_pad, causal):
    # Note (david): buffer rows past cu_seqlens[-1] own no q block, so they
    # come back out = 0 and lse = -inf even when their queries are NaN. A
    # 600-row pad takes several fill windows, and an all-empty batch runs no
    # block at all.
    q, k, v, segments = packed_qkv(seqlens_q, 4, 2, 128)
    pad_q, pad_k, pad_v = random_qkv(99, 1, num_pad, 4, 2, 128)
    q = jnp.concatenate([q, jnp.full_like(pad_q[0], jnp.nan)])
    k = jnp.concatenate([k, pad_k[0]])
    v = jnp.concatenate([v, pad_v[0]])
    cu = cu_seqlens_of(seqlens_q)
    max_seqlen = max(*seqlens_q, 1)
    out, lse = varlen_attn(
        q,
        k,
        v,
        cu,
        cu,
        max_seqlen,
        max_seqlen,
        causal=causal,
        return_softmax_lse=True,
    )
    packed_q_end = int(cu[-1])
    assert jnp.array_equal(
        out[packed_q_end:], jnp.zeros_like(out[packed_q_end:])
    )
    assert jnp.all(lse[:, packed_q_end:] == -jnp.inf)
    for index, segment in enumerate(segments):
        rows = slice(int(cu[index]), int(cu[index + 1]))
        assert_output_close(out[rows][None], *segment, causal=causal)


@pytest.mark.parametrize(
    "seqlens",
    [
        pytest.param([200, 184], id="two_seqs"),
        pytest.param([250, 6, 3, 125], id="short_seqs_in_clamped_window"),
        pytest.param(
            [3000, 5000],
            id="long_two_seqs",
            marks=pytest.mark.skipif(
                INTERPRET, reason="interpret mode affords short sequences only"
            ),
        ),
        pytest.param(
            [5500],
            id="long_one_seq",
            marks=pytest.mark.skipif(
                INTERPRET, reason="interpret mode affords short sequences only"
            ),
        ),
        pytest.param(
            [2500, 100, 50, 30],
            id="long_short_tail",
            marks=pytest.mark.skipif(
                INTERPRET, reason="interpret mode affords short sequences only"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "causal,window_size",
    [
        (False, (-1, -1)),
        (True, (-1, -1)),
        (True, (96, 0)),
    ],
)
def test_flash_attn_varlen_ragged_last_block(seqlens, causal, window_size):
    # Note (david): a packed total of 128 times an odd count keeps the
    # max_seqlen-capped block, which does not divide it, so the last q and kv
    # windows clamp to the padded end and overlap the previous block. Short
    # sequences inside the clamped window chain the head blend over several
    # identical windows, and one 5500-token sequence needs the ragged last kv
    # block that a floored block count would drop.
    max_seqlen = max(seqlens)
    padded_total = pad_seqlen(sum(seqlens))
    max_seqlen_bucket = min(padded_total, next_pow2(max_seqlen))
    blocks = default_block_sizes(
        padded_total,
        padded_total,
        max_seqlen_bucket,
        max_seqlen_bucket,
        window_size=window_size,
        head_dim=128,
    )
    assert padded_total % blocks.block_q and padded_total % blocks.block_kv
    q, k, v, segments = packed_qkv(seqlens, 4, 2, 128)
    cu = cu_seqlens_of(seqlens)
    out, lse = varlen_attn(
        q,
        k,
        v,
        cu,
        cu,
        max_seqlen,
        max_seqlen,
        causal=causal,
        window_size=window_size,
        return_softmax_lse=True,
    )
    mask_kwargs = {"causal": causal, "window_size": window_size}
    for index, segment in enumerate(segments):
        rows = slice(int(cu[index]), int(cu[index + 1]))
        assert_output_close(out[rows][None], *segment, **mask_kwargs)
        _, lse_ref = attention_ref(*segment, **mask_kwargs)
        _, lse_bf16 = attention_ref(
            *segment, upcast=False, reorder_ops=True, **mask_kwargs
        )
        lse_error = max_abs(lse[:, rows] - lse_ref[0])
        assert lse_error <= 2 * max_abs(lse_bf16 - lse_ref) + 1e-3


def random_sinks(nheads):
    return jax.random.uniform(
        jax.random.PRNGKey(5), (nheads,), jnp.float32, -2.0, 8.0
    )


def assert_lse_close(lse, q, k, v, **mask_kwargs):
    """lse (batch, nheads, seqlen_q) against the f32 reference, by the out rule
    plus test_flash_attn.py's fixed 1e-2."""
    _, lse_ref = attention_ref(q, k, v, **mask_kwargs)
    _, lse_bf16 = attention_ref(
        q, k, v, upcast=False, reorder_ops=True, **mask_kwargs
    )
    assert max_abs(lse - lse_ref) <= 2 * max_abs(lse_bf16 - lse_ref) + 1e-2


@pytest.mark.parametrize("layout", ["head_major", "token_major", "heads_outer"])
@pytest.mark.parametrize("nheads,nheads_k", [(8, 2), (4, 4)])
@pytest.mark.parametrize(
    "mask_kwargs",
    [
        pytest.param(dict(causal=True), id="causal"),
        pytest.param(dict(causal=True, window_size=(127, 0)), id="window"),
        pytest.param(dict(causal=False, softcap=30.0), id="softcap"),
    ],
)
def test_flash_attn_func_attention_sinks(layout, nheads, nheads_k, mask_kwargs):
    # Note (deepolar): q 384 against kv 256 leaves the first 128 causal rows
    # with no key, so they attend only their sink: out = 0, lse = sink. Batch
    # 2 checks that the kernel maps each flat q head back to its sequence head
    # in every layout, batch-outer and heads-outer.
    seqlen_k = 256 if mask_kwargs["causal"] else 384
    q, k, v = random_qkv(0, 2, 384, nheads, nheads_k, 128, seqlen_k=seqlen_k)
    sinks = random_sinks(nheads)
    if layout == "head_major":
        out, lse = flash_attn_func(
            *map(to_head_major, (q, k, v)),
            learnable_sink=sinks,
            return_softmax_lse=True,
            interpret=INTERPRET,
            **mask_kwargs,
        )
        out = to_head_major(out)
    elif layout == "token_major":
        out, lse = flash_attn_func(
            *(x.reshape(*x.shape[:2], -1) for x in (q, k, v)),
            learnable_sink=sinks,
            token_major=True,
            head_dim=128,
            return_softmax_lse=True,
            interpret=INTERPRET,
            **mask_kwargs,
        )
        out = out.reshape(q.shape)
    else:
        out = flash_attn_func(
            *(x.transpose(2, 0, 1, 3) for x in (q, k, v)),
            learnable_sink=sinks,
            heads_outer_fold=True,
            interpret=INTERPRET,
            **mask_kwargs,
        )
        out, lse = out.transpose(1, 2, 0, 3), None
    assert_output_close(out, q, k, v, sinks=sinks, **mask_kwargs)
    if lse is not None:
        assert_lse_close(lse, q, k, v, sinks=sinks, **mask_kwargs)
    if mask_kwargs["causal"]:
        assert jnp.all(out[:, :128] == 0)
        if lse is not None:
            np.testing.assert_allclose(
                lse[:, :, :128],
                jnp.broadcast_to(sinks[None, :, None], lse[:, :, :128].shape),
            )


@pytest.mark.parametrize(
    "mask_kwargs",
    [
        pytest.param(dict(causal=True), id="causal"),
        pytest.param(dict(causal=True, window_size=(127, 0)), id="window"),
    ],
)
def test_flash_attn_func_attention_sinks_head_pairs(mask_kwargs):
    # Note (deepolar): token-major head_dim 64 pairs two heads per block on
    # every backend, so the second pipeline head must read its own sink.
    q, k, v = random_qkv(0, 2, 256, 4, 4, 64)
    sinks = random_sinks(4)
    out, lse = flash_attn_func(
        *(x.reshape(*x.shape[:2], -1) for x in (q, k, v)),
        learnable_sink=sinks,
        token_major=True,
        head_dim=64,
        return_softmax_lse=True,
        interpret=INTERPRET,
        **mask_kwargs,
    )
    assert_output_close(
        out.reshape(q.shape), q, k, v, sinks=sinks, **mask_kwargs
    )
    assert_lse_close(lse, q, k, v, sinks=sinks, **mask_kwargs)


@pytest.mark.parametrize("nheads,nheads_k", [(8, 2), (4, 4)])
@pytest.mark.parametrize(
    "mask_kwargs",
    [
        pytest.param(dict(causal=True), id="causal"),
        pytest.param(dict(causal=True, window_size=(127, 0)), id="window"),
        pytest.param(dict(causal=False, softcap=30.0), id="softcap"),
    ],
)
def test_flash_attn_varlen_attention_sinks(nheads, nheads_k, mask_kwargs):
    # Note (deepolar): MHA with no window folds several heads per block, so a
    # pipeline head past the first reads its own sink.
    seqlens = [37, 300, 128, 1]
    q, k, v, segments = packed_qkv(seqlens, nheads, nheads_k, 128)
    cu = cu_seqlens_of(seqlens)
    sinks = random_sinks(nheads)
    out, lse = varlen_attn(
        q,
        k,
        v,
        cu,
        cu,
        max(seqlens),
        max(seqlens),
        learnable_sink=sinks,
        return_softmax_lse=True,
        **mask_kwargs,
    )
    for index, segment in enumerate(segments):
        rows = slice(int(cu[index]), int(cu[index + 1]))
        assert_output_close(
            out[rows][None], *segment, sinks=sinks, **mask_kwargs
        )
        assert_lse_close(
            lse[None, :, rows], *segment, sinks=sinks, **mask_kwargs
        )


@pytest.mark.parametrize("entry", ["dense", "varlen"])
def test_attention_sinks_are_forward_only(entry):
    q, k, v = random_qkv(0, 1, 128, 4, 2, 128)
    sinks = random_sinks(4)
    if entry == "dense":

        def loss(q):
            return dense_attn(q, k, v, causal=True, learnable_sink=sinks).sum()

    else:
        cu = cu_seqlens_of([128])

        def loss(q):
            return varlen_attn(
                q[0],
                k[0],
                v[0],
                cu,
                cu,
                128,
                128,
                causal=True,
                learnable_sink=sinks,
            ).sum()

    with pytest.raises(NotImplementedError, match="forward-only"):
        jax.grad(loss)(q)


@pytest.mark.parametrize("entry", ["dense", "varlen", "decode"])
def test_attention_sinks_far_above_the_scores(entry):
    # Note (deepolar): q = -2.5 against k = +2.5 scores about -70.7 at d 128, so
    # sinks of 20 and 40 sit more than 128 log2 units above the static anchor;
    # folded against that anchor, exp2(sink - anchor) overflowed and lse came
    # back +inf.
    sinks = jnp.array([20.0, 40.0, 20.0, 40.0], jnp.float32)
    if entry == "decode":
        q, k_cache, v_cache, _, _ = random_decode_inputs(
            29, 2, 2, 256, 4, 2, 128
        )
        q, k_cache = jnp.full_like(q, -2.5), jnp.full_like(k_cache, 2.5)
        cache_seqlens = jnp.array([100, 256], jnp.int32)
        out, lse, _, _ = decode_with_cache_copies(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            return_softmax_lse=True,
            interpret=INTERPRET,
            learnable_sink=sinks,
        )
        assert_decode_matches(
            out,
            lse,
            q,
            k_cache,
            v_cache,
            cache_seqlens,
            jnp.arange(2, dtype=jnp.int32),
            sinks=sinks,
        )
        return
    q, k, v = random_qkv(0, 1, 256, 4, 2, 128)
    q, k = jnp.full_like(q, -2.5), jnp.full_like(k, 2.5)
    if entry == "dense":
        out, lse = flash_attn_func(
            *map(to_head_major, (q, k, v)),
            causal=True,
            learnable_sink=sinks,
            return_softmax_lse=True,
            interpret=INTERPRET,
        )
        out = to_head_major(out)
    else:
        cu = cu_seqlens_of([256])
        out, lse = varlen_attn(
            q[0],
            k[0],
            v[0],
            cu,
            cu,
            256,
            256,
            causal=True,
            learnable_sink=sinks,
            return_softmax_lse=True,
        )
        out, lse = out[None], lse[None]
    assert jnp.all(jnp.isfinite(lse))
    assert_output_close(out, q, k, v, causal=True, sinks=sinks)
    assert_lse_close(lse, q, k, v, causal=True, sinks=sinks)


@pytest.mark.parametrize("entry", ["varlen", "decode"])
def test_attention_minus_inf_sinks_are_no_sink(entry):
    # Note (deepolar): a -inf sink adds nothing, so its head matches a call
    # without sinks, and a row with no key returns out = 0, lse = -inf, not
    # NaN. The finite heads' empty rows return lse = sink. Packed causal rows
    # end on their bottom-right diagonal, so a 64-row query against 16 keys
    # leaves its first 48 rows with no key.
    sinks = jnp.array([-jnp.inf, 2.0, -jnp.inf, 4.0], jnp.float32)
    none_heads, sink_heads = np.array([0, 2]), np.array([1, 3])
    if entry == "decode":
        q, k_cache, v_cache, _, _ = random_decode_inputs(
            29, 2, 2, 256, 4, 2, 128
        )
        kwargs = dict(
            cache_seqlens=jnp.array([0, 200], jnp.int32),
            return_softmax_lse=True,
            interpret=INTERPRET,
        )
        out, lse, _, _ = decode_with_cache_copies(
            q, k_cache, v_cache, learnable_sink=sinks, **kwargs
        )
        plain_out, plain_lse, _, _ = decode_with_cache_copies(
            q, k_cache, v_cache, **kwargs
        )
        out, plain_out = out[:, 0].swapaxes(0, 1), plain_out[:, 0].swapaxes(
            0, 1
        )
        lse, plain_lse = lse[..., 0].T, plain_lse[..., 0].T
        empty, keyed = np.array([0]), np.array([1])
    else:
        q, k, v = random_qkv(0, 1, 128, 4, 2, 128, seqlen_k=80)
        cu_q, cu_k = cu_seqlens_of([64, 64]), cu_seqlens_of([16, 64])
        kwargs = dict(causal=True, return_softmax_lse=True)
        out, lse = varlen_attn(
            q[0], k[0], v[0], cu_q, cu_k, 64, 64, learnable_sink=sinks, **kwargs
        )
        plain_out, plain_lse = varlen_attn(
            q[0], k[0], v[0], cu_q, cu_k, 64, 64, **kwargs
        )
        out, plain_out = out.swapaxes(0, 1), plain_out.swapaxes(0, 1)
        empty, keyed = np.arange(48), np.arange(48, 128)
    out, plain_out = np.asarray(out, np.float32), np.asarray(
        plain_out, np.float32
    )
    lse, plain_lse = np.asarray(lse), np.asarray(plain_lse)
    assert not np.isnan(out).any() and not np.isnan(lse).any()
    assert np.all(out[:, empty] == 0)
    assert np.all(lse[np.ix_(none_heads, empty)] == -np.inf)
    np.testing.assert_allclose(
        lse[np.ix_(sink_heads, empty)],
        np.broadcast_to(
            np.asarray(sinks)[sink_heads, None], (len(sink_heads), len(empty))
        ),
    )
    np.testing.assert_allclose(
        out[np.ix_(none_heads, keyed)],
        plain_out[np.ix_(none_heads, keyed)],
        atol=1e-2,
    )
    np.testing.assert_allclose(
        lse[np.ix_(none_heads, keyed)],
        plain_lse[np.ix_(none_heads, keyed)],
        atol=1e-3,
    )


@pytest.mark.parametrize("entry", ["dense", "varlen"])
def test_kernel_rejects_misshaped_sinks(entry):
    # Note (deepolar): a direct make_flash_attn_mha caller skips the API's
    # check, and a sinks array that does not cover the flat q heads would read
    # SMEM past its end.
    batch, nheads, seqlen = 2, 4, 128
    blocks = default_block_sizes(seqlen, seqlen, head_dim=128)
    if entry == "dense":
        kernel = make_flash_attn_mha(
            batch * nheads,
            seqlen,
            seqlen,
            causal=True,
            block_sizes=blocks,
            interpret=INTERPRET,
        )
        q = jnp.zeros((batch * nheads, seqlen, 128), jnp.bfloat16)
        args = (q, q, q)
    else:
        kernel = make_flash_attn_mha(
            nheads,
            seqlen,
            seqlen,
            causal=True,
            block_sizes=blocks,
            interpret=INTERPRET,
            varlen_max_seqlen_kv=seqlen,
        )
        q = jnp.zeros((nheads, seqlen, 128), jnp.bfloat16)
        cu = cu_seqlens_of([seqlen])
        args = (q, q, q, cu, cu)
    with pytest.raises(ValueError, match="sinks must be"):
        kernel(*args, jnp.zeros((3,), jnp.float32))


@pytest.mark.parametrize("causal", [False, True])
def test_attention_sinks_after_guard_recovery(causal):
    # Note (deepolar): the last quarter of the keys scores far above the first
    # block's anchor, so the guard replays and rebases the row state before the
    # sink fold reads it; the sinks span below, near and far above the rebased
    # row max.
    q, k, v = random_qkv(0, 2, 1024, 4, 4, 64)
    k = k.at[:, 768:].multiply(40.0)
    sinks = jnp.array([0.0, 30.0, 300.0, -5.0], jnp.float32)
    out, lse = flash_attn_func(
        *map(to_head_major, (q, k, v)),
        causal=causal,
        learnable_sink=sinks,
        return_softmax_lse=True,
        interpret=INTERPRET,
    )
    assert_output_close(to_head_major(out), q, k, v, causal=causal, sinks=sinks)
    assert_lse_close(lse, q, k, v, causal=causal, sinks=sinks)


def test_decode_attention_sinks_after_guard_recovery():
    # Note (deepolar): the second half of the cache scores 2048 against the
    # first half's 0, so decode's anchor rebases before the sinks join.
    capacity, heads, head_dim = 4096, 2, 64
    q = jnp.ones((1, 1, heads, head_dim), jnp.bfloat16)
    half = (1, capacity // 2, heads, head_dim)
    k_cache = jnp.concatenate(
        (jnp.zeros(half, jnp.bfloat16), jnp.full(half, 32.0, jnp.bfloat16)),
        axis=1,
    )
    v_cache = jnp.concatenate(
        (jnp.ones(half, jnp.bfloat16), jnp.full(half, 3.0, jnp.bfloat16)),
        axis=1,
    )
    sinks = jnp.array([260.0, 100.0], jnp.float32)
    out, lse, _, _ = decode_with_cache_copies(
        q,
        k_cache,
        v_cache,
        cache_seqlens=capacity,
        return_softmax_lse=True,
        interpret=INTERPRET,
        learnable_sink=sinks,
    )
    assert_decode_matches(
        out,
        lse,
        q,
        k_cache,
        v_cache,
        jnp.array([capacity], jnp.int32),
        jnp.array([0], jnp.int32),
        sinks=sinks,
    )


@pytest.mark.parametrize("window_size", [(127, 0), (-1, 0)])
def test_flash_attn_varlen_attention_sinks_gpt_oss(window_size):
    # Note (deepolar): gpt-oss attention is 64 q heads on 8 kv heads at
    # head_dim 64, with a 128-token window on the sliding layers.
    seqlens = [37, 300, 128, 1]
    q, k, v, segments = packed_qkv(seqlens, 64, 8, 64)
    cu = cu_seqlens_of(seqlens)
    sinks = random_sinks(64)
    mask_kwargs = dict(causal=True, window_size=window_size)
    out, lse = varlen_attn(
        q,
        k,
        v,
        cu,
        cu,
        max(seqlens),
        max(seqlens),
        learnable_sink=sinks,
        return_softmax_lse=True,
        **mask_kwargs,
    )
    for index, segment in enumerate(segments):
        rows = slice(int(cu[index]), int(cu[index + 1]))
        assert_output_close(
            out[rows][None], *segment, sinks=sinks, **mask_kwargs
        )
        assert_lse_close(
            lse[None, :, rows], *segment, sinks=sinks, **mask_kwargs
        )


def test_varlen_padding_rows_ignore_sinks():
    # Note (deepolar): padding rows past cu_seqlens[-1] never attend, so they
    # keep out = 0 and lse = -inf with sinks, even when their q is NaN.
    seqlens = [17, 40]
    q, k, v, _ = packed_qkv(seqlens, 4, 2, 128)
    pad_q, pad_k, pad_v = random_qkv(99, 1, 23, 4, 2, 128)
    q = jnp.concatenate([q, jnp.full_like(pad_q[0], jnp.nan)])
    k = jnp.concatenate([k, pad_k[0]])
    v = jnp.concatenate([v, pad_v[0]])
    cu = cu_seqlens_of(seqlens)
    out, lse = varlen_attn(
        q,
        k,
        v,
        cu,
        cu,
        40,
        40,
        causal=True,
        learnable_sink=random_sinks(4),
        return_softmax_lse=True,
    )
    assert jnp.all(out[57:] == 0)
    assert jnp.all(lse[:, 57:] == -jnp.inf)


def test_traced_sinks_compile_once():
    q, k, v = map(to_head_major, random_qkv(0, 1, 256, 4, 2, 128))
    attend = jax.jit(
        lambda sinks: flash_attn_func(
            q, k, v, causal=True, learnable_sink=sinks, interpret=INTERPRET
        )
    )
    attend(random_sinks(4)).block_until_ready()
    attend(random_sinks(4) + 1.0).block_until_ready()
    assert attend._cache_size() == 1  # pylint: disable=protected-access


def decode_reference(
    q,
    k_cache,
    v_cache,
    cache_seqlens,
    cache_batch_idx,
    window_size=(-1, -1),
    softcap=0.0,
    sinks=None,
):
    """(out, lse) of one decode step over the mapped contiguous cache rows.

    softcap > 0 caps the scaled scores at tanh(s / softcap) * softcap. sinks
    (num_query_heads,) joins each head's softmax as one more logit with no
    value row.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    keys = k_cache[cache_batch_idx].astype(jnp.float32)
    values = v_cache[cache_batch_idx].astype(jnp.float32)
    group = q.shape[2] // keys.shape[2]
    keys, values = (
        jnp.repeat(cache, group, axis=2) for cache in (keys, values)
    )
    # The decode wrapper folds softmax_scale * log2(e) into q and rounds it to
    # bf16 before the kernel, so the reference scores start from that same
    # rounded q. Scaling the unrounded q instead is off by the rounding: with
    # one visible key the lse is a single dot product, the error has a std of
    # 1.6e-3 at head_dim 128 or 256, and it passes the lse tolerance below for
    # 1-2% of the elements, whatever the kernel does.
    # With a softcap the folded scale is softmax_scale / softcap, so the raw dot
    # product is already s / softcap.
    q_scaled = (q.astype(jnp.float32) * fused_q_scale(scale, softcap)).astype(
        q.dtype
    )
    raw_scores = jnp.einsum(
        "bqhd,bkhd->bhqk", q_scaled.astype(jnp.float32), keys
    )
    if softcap:
        scores = jnp.tanh(raw_scores) * softcap
    else:
        scores = math.log(2.0) * raw_scores
    key_positions = jnp.arange(k_cache.shape[1])[None, :]
    query_positions = (cache_seqlens - 1)[:, None]
    left, right = window_size
    left_visible = (
        True if left == -1 else key_positions >= query_positions - left
    )
    right_visible = (
        True if right == -1 else key_positions <= query_positions + right
    )
    visible = (
        (key_positions < cache_seqlens[:, None]) & left_visible & right_visible
    )
    scores = jnp.where(visible[:, None, None, :], scores, -jnp.inf)
    if sinks is not None:
        sink_logits = jnp.broadcast_to(
            sinks.astype(jnp.float32)[None, :, None, None],
            (*scores.shape[:3], 1),
        )
        scores = jnp.concatenate([scores, sink_logits], axis=-1)
    probabilities = jnp.where(
        (cache_seqlens > 0)[:, None, None, None],
        jax.nn.softmax(scores, axis=-1),
        0.0,
    )
    lse = jax.scipy.special.logsumexp(scores, axis=-1)
    if sinks is not None:
        probabilities = probabilities[..., :-1]
    out = jnp.einsum("bhqk,bkhd->bqhd", probabilities, values)
    return out.astype(q.dtype), lse


def assert_decode_matches(
    out,
    lse,
    q,
    k_cache,
    v_cache,
    total_seqlens,
    cache_batch_idx,
    window_size=(-1, -1),
    out_tol=(2e-2, 3e-2),
    softcap=0.0,
    sinks=None,
):
    out_ref, lse_ref = decode_reference(
        q,
        k_cache,
        v_cache,
        total_seqlens,
        cache_batch_idx,
        window_size,
        softcap,
        sinks,
    )
    np.testing.assert_allclose(
        out.astype(jnp.float32),
        out_ref.astype(jnp.float32),
        rtol=out_tol[0],
        atol=out_tol[1],
    )
    finite = jnp.isfinite(lse_ref)
    np.testing.assert_allclose(
        lse[finite], lse_ref[finite], rtol=2e-3, atol=3e-3
    )
    np.testing.assert_array_equal(jnp.isneginf(lse), jnp.isneginf(lse_ref))


def random_decode_inputs(
    seed, batch, cache_batch, capacity, num_query_heads, num_kv_heads, head_dim
):
    """(q, k_cache, v_cache, k, v) for one contiguous-cache decode step."""
    keys = jax.random.split(jax.random.PRNGKey(seed), 5)
    cache_shape = (cache_batch, capacity, num_kv_heads, head_dim)
    new_shape = (batch, 1, num_kv_heads, head_dim)
    return (
        jax.random.normal(
            keys[0], (batch, 1, num_query_heads, head_dim), jnp.bfloat16
        ),
        jax.random.normal(keys[1], cache_shape, jnp.bfloat16),
        jax.random.normal(keys[2], cache_shape, jnp.bfloat16),
        jax.random.normal(keys[3], new_shape, jnp.bfloat16),
        jax.random.normal(keys[4], new_shape, jnp.bfloat16),
    )


def decode_with_cache_copies(q, k_cache, v_cache, *args, **kwargs):
    # Note (david): the API donates the caches, which deletes the caller's
    # arrays, and the reference checks still need them.
    return flash_attn_with_kvcache(
        q, jnp.copy(k_cache), jnp.copy(v_cache), *args, **kwargs
    )


def test_flash_attn_with_kvcache_recovers_when_later_block_changes_anchor():
    # Note (david): the second half's keys lift the row max from 0 to 256 after
    # the first block has frozen the softmax anchor.
    capacity, heads, head_dim = 4096, 2, 64
    q = jnp.ones((1, 1, heads, head_dim), jnp.bfloat16)
    half_shape = (1, capacity // 2, heads, head_dim)
    k_cache = jnp.concatenate(
        (
            jnp.zeros(half_shape, jnp.bfloat16),
            jnp.full(half_shape, 32.0, jnp.bfloat16),
        ),
        axis=1,
    )
    v_cache = jnp.concatenate(
        (
            jnp.ones(half_shape, jnp.bfloat16),
            jnp.full(half_shape, 3.0, jnp.bfloat16),
        ),
        axis=1,
    )
    out, lse, _, _ = decode_with_cache_copies(
        q,
        k_cache,
        v_cache,
        cache_seqlens=capacity,
        return_softmax_lse=True,
        interpret=INTERPRET,
    )
    assert_decode_matches(
        out,
        lse,
        q,
        k_cache,
        v_cache,
        jnp.array([capacity], jnp.int32),
        jnp.array([0], jnp.int32),
    )


def test_kvcache_compute_fragments_handle_masked_prefix_and_boundary_append():
    # Note (david): a 255-token window leaves whole compute fragments masked in
    # front of the live range, and 1799 + 1 = 1800 fills the last row of an
    # 8-sublane cache tile, the granule the cache copy rounds up to.
    q, k_cache, v_cache, k, v = random_decode_inputs(91, 2, 3, 2048, 4, 2, 64)
    cache_seqlens = jnp.array([512, 1799], jnp.int32)
    cache_batch_idx = jnp.array([2, 0], jnp.int32)
    out, lse, updated_k, updated_v = decode_with_cache_copies(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        window_size=(255, 0),
        return_softmax_lse=True,
        interpret=INTERPRET,
    )
    expected_k = k_cache.at[cache_batch_idx, cache_seqlens].set(k[:, 0])
    expected_v = v_cache.at[cache_batch_idx, cache_seqlens].set(v[:, 0])
    np.testing.assert_array_equal(updated_k, expected_k)
    np.testing.assert_array_equal(updated_v, expected_v)
    assert_decode_matches(
        out,
        lse,
        q,
        expected_k,
        expected_v,
        cache_seqlens + 1,
        cache_batch_idx,
        (255, 0),
        out_tol=(3e-2, 4e-2),
    )


@pytest.mark.parametrize(
    ("head_dim", "capacity", "cache_seqlen"),
    [(128, 2048, 1537), (256, 1024, 900)],
)
def test_kvcache_contiguous_cache_across_compute_fragments(
    head_dim, capacity, cache_seqlen
):
    q, k_cache, v_cache, _, _ = random_decode_inputs(
        head_dim, 1, 1, capacity, 2, 2, head_dim
    )
    cache_seqlens = jnp.array([cache_seqlen], jnp.int32)
    out, lse, updated_k, updated_v = decode_with_cache_copies(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        return_softmax_lse=True,
        interpret=INTERPRET,
    )
    np.testing.assert_array_equal(updated_k, k_cache)
    np.testing.assert_array_equal(updated_v, v_cache)
    assert_decode_matches(
        out,
        lse,
        q,
        k_cache,
        v_cache,
        cache_seqlens,
        jnp.array([0], jnp.int32),
        out_tol=(3e-2, 4e-2),
    )


@pytest.mark.parametrize("window_size", [(-1, -1), (255, 0)])
@pytest.mark.parametrize(
    ("num_query_heads", "num_kv_heads", "head_dim"),
    [(8, 1, 128), (8, 2, 128), (16, 2, 64)],
)
def test_kvcache_contiguous_cache_attention_sinks(
    window_size, num_query_heads, num_kv_heads, head_dim
):
    # Note (deepolar): one KV head stages head-major, two pack a head pair per
    # u32 lane and 8 query heads per KV head unpack one head per fragment. The
    # empty row attends only its sink: out = 0, lse = sink.
    q, k_cache, v_cache, _, _ = random_decode_inputs(
        29, 3, 3, 1024, num_query_heads, num_kv_heads, head_dim
    )
    sinks = jax.random.uniform(
        jax.random.PRNGKey(5), (num_query_heads,), jnp.float32, -2.0, 8.0
    )
    cache_seqlens = jnp.array([0, 517, 1024], jnp.int32)
    rows = jnp.arange(3, dtype=jnp.int32)
    out, lse, _, _ = decode_with_cache_copies(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        window_size=window_size,
        return_softmax_lse=True,
        interpret=INTERPRET,
        learnable_sink=sinks,
    )
    assert_decode_matches(
        out,
        lse,
        q,
        k_cache,
        v_cache,
        cache_seqlens,
        rows,
        window_size,
        sinks=sinks,
    )
    np.testing.assert_array_equal(out[0], 0.0)
    np.testing.assert_allclose(lse[0, :, 0], sinks, rtol=1e-6)


@pytest.mark.parametrize(
    ("num_query_heads", "num_kv_heads", "head_dim"),
    [(8, 1, 128), (6, 1, 256), (8, 2, 128)],
)
def test_flash_attn_with_kvcache_appends_step_after_step(
    num_query_heads, num_kv_heads, head_dim
):
    # Note (david): one executable serves every position, and each step's
    # append must land on the cache the previous step returned: the positions
    # cross the 8-row and 16-row cache tiles (one KV head stages head-major,
    # whose token axis v7x tiles 16 deep) and end at the last slot.
    batch, capacity = 2, 256
    positions = [*range(0, 40), *range(capacity - 20, capacity)]
    q, k_cache, v_cache, _, _ = random_decode_inputs(
        17, batch, batch, capacity, num_query_heads, num_kv_heads, head_dim
    )
    k_cache = k_cache.at[:, positions].set(0.0)
    v_cache = v_cache.at[:, positions].set(0.0)
    expected_k, expected_v = k_cache, v_cache
    updated_k, updated_v = jnp.copy(k_cache), jnp.copy(v_cache)
    rows = jnp.arange(batch, dtype=jnp.int32)
    new_shape = (batch, 1, num_kv_heads, head_dim)
    for step, position in enumerate(positions):
        keys = jax.random.split(jax.random.PRNGKey(100 + step), 2)
        k = jax.random.normal(keys[0], new_shape, jnp.bfloat16)
        v = jax.random.normal(keys[1], new_shape, jnp.bfloat16)
        cache_seqlens = jnp.full((batch,), position, jnp.int32)
        out, lse, updated_k, updated_v = flash_attn_with_kvcache(
            q,
            updated_k,
            updated_v,
            k,
            v,
            cache_seqlens=cache_seqlens,
            return_softmax_lse=True,
            interpret=INTERPRET,
        )
        expected_k = expected_k.at[rows, position].set(k[:, 0])
        expected_v = expected_v.at[rows, position].set(v[:, 0])
        np.testing.assert_array_equal(updated_k, expected_k)
        np.testing.assert_array_equal(updated_v, expected_v)
        assert_decode_matches(
            out, lse, q, expected_k, expected_v, cache_seqlens + 1, rows
        )


def test_flash_attn_with_kvcache_block_table_changes_under_one_executable():
    # Note (david): the table and the lengths are runtime operands, so one
    # executable must serve both page layouts, each append landing where its
    # table says.
    batch, page_size, pages_per_seq, heads, head_dim = 2, 128, 2, 2, 128
    q = jnp.ones((batch, 1, heads, head_dim), jnp.bfloat16)
    kv_pages = jnp.zeros(
        (batch * pages_per_seq, page_size, 2 * heads, head_dim), jnp.bfloat16
    )
    k = jnp.ones((batch, 1, heads, head_dim), jnp.bfloat16)
    v = jnp.full_like(k, 2.0)
    table_a = jnp.array([[0, 1], [2, 3]], jnp.int32)
    table_b = jnp.array([[3, 2], [1, 0]], jnp.int32)
    lens_a = jnp.array([0, 129], jnp.int32)
    lens_b = jnp.array([130, 5], jnp.int32)
    compiled = flash_attn_with_kvcache.lower(
        q,
        kv_pages,
        None,
        k,
        v,
        cache_seqlens=lens_a,
        block_table=table_a,
        interpret=INTERPRET,
    ).compile()
    rows = jnp.arange(batch)
    for table, lens in ((table_a, lens_a), (table_b, lens_b)):
        _, kv_pages = compiled(
            q, kv_pages, None, k, v, cache_seqlens=lens, block_table=table
        )
        slots = (table[rows, lens // page_size], lens % page_size)
        slot_kv = kv_pages[slots]
        np.testing.assert_array_equal(slot_kv[..., 0::2, :], k[:, 0])
        np.testing.assert_array_equal(slot_kv[..., 1::2, :], v[:, 0])
    assert float(jnp.abs(kv_pages).sum()) == 4 * float(
        jnp.abs(k[0, 0]).sum() + jnp.abs(v[0, 0]).sum()
    )


def test_flash_attn_with_kvcache_pads_block_table_to_whole_blocks():
    # Note (david): 9 pages of 128 tokens divide only into 384-token blocks, so
    # the kernel takes 1024-token blocks over a table padded to 16 pages. The
    # rows end inside the first block, on it, one token past it and on the last
    # real page, so second blocks read padded entries next to real ones.
    batch, page_size, pages_per_seq = 4, 128, 9
    heads, heads_k, head_dim = 4, 2, 128
    num_pages = batch * pages_per_seq
    keys = jax.random.split(jax.random.PRNGKey(0), 5)
    q = jax.random.normal(keys[0], (batch, 1, heads, head_dim), jnp.bfloat16)
    kv_pages = jax.random.normal(
        keys[1], (num_pages, page_size, 2 * heads_k, head_dim), jnp.bfloat16
    )
    k = jax.random.normal(keys[2], (batch, 1, heads_k, head_dim), jnp.bfloat16)
    v = jax.random.normal(keys[3], (batch, 1, heads_k, head_dim), jnp.bfloat16)
    table = (
        jax.random.permutation(keys[4], num_pages)
        .astype(jnp.int32)
        .reshape(batch, pages_per_seq)
    )
    cache_seqlens = jnp.array([129, 1023, 1024, 1151], jnp.int32)
    rows = jnp.arange(batch)
    row_kv = kv_pages[table].reshape(
        batch, pages_per_seq * page_size, 2 * heads_k, head_dim
    )
    k_cache = row_kv[..., 0::2, :].at[rows, cache_seqlens].set(k[:, 0])
    v_cache = row_kv[..., 1::2, :].at[rows, cache_seqlens].set(v[:, 0])
    out, lse, kv_pages = flash_attn_with_kvcache(
        q,
        kv_pages,
        None,
        k,
        v,
        cache_seqlens=cache_seqlens,
        block_table=table,
        return_softmax_lse=True,
        interpret=INTERPRET,
    )
    assert_decode_matches(
        out, lse, q, k_cache, v_cache, cache_seqlens + 1, rows
    )
    slot_kv = kv_pages[
        table[rows, cache_seqlens // page_size], cache_seqlens % page_size
    ]
    np.testing.assert_array_equal(slot_kv[..., 0::2, :], k[:, 0])
    np.testing.assert_array_equal(slot_kv[..., 1::2, :], v[:, 0])
