# Copyright 2026 heavyball-research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for kernel paths the random grids of test_prefill.py do
not reach: dense and packed flash_attn_func / flash_attn_varlen_func, and
single-token decode through flash_attn_with_kvcache."""

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flywheel_tpu import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
from flywheel_tpu.softmax_attention import dispatch, fwd_blocks
from flywheel_tpu.softmax_attention.config import BlockSizes, next_pow2
from flywheel_tpu.softmax_attention.dispatch import (
    default_block_sizes,
    get_kernel,
    make_flash_attn_mha,
)
from flywheel_tpu.softmax_attention.inputs import make_variant, pad_seqlen
from flywheel_tpu.softmax_attention.kernels import fwd_varlen
from flywheel_tpu.softmax_attention.kernels.fwd_dense_mxu import (
    dense_has_checked_units,
)
from flywheel_tpu.softmax_attention.mxu_pipeline import FRAGMENT_ROWS
from flywheel_tpu.softmax_attention.variant import Variant, fused_q_scale
from tests.softmax_attention.util import (
    HAS_MXU_ACCUMULATORS,
    attention_ref,
    randn,
    requires_mxu_accumulators,
)

INTERPRET = jax.default_backend() != "tpu"


def to_head_major(x):
    """(batch, seqlen, nheads, d) <-> (batch, nheads, seqlen, d), or packed
    (total, nheads, d) <-> (nheads, total, d)."""
    return jnp.swapaxes(x, -3, -2)


def max_abs(x):
    return jnp.abs(x.astype(jnp.float32)).max(initial=0.0).item()


def assert_output_close(out, q, k, v, rows=slice(None), **mask_kwargs):
    """test_prefill.py's rule: at most twice the error of the bf16 reference
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


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_varlen_guard_replay_unroll(causal):
    # Note (deepolar): the second sequence's last keys score far above the
    # first kv block's anchor, so its heads replay; every unroll factor of the
    # replay loop must give the rolled replay's output.
    seqlens = [384, 1024]
    q, k, v, segments = packed_qkv(seqlens, 4, 2, 128)
    k = k.at[384 + 768 :].multiply(40.0)
    segments[1] = (segments[1][0], k[None, 384:], segments[1][2])
    cu = cu_seqlens_of(seqlens)
    outs = [
        varlen_attn(
            q,
            k,
            v,
            cu,
            cu,
            max(seqlens),
            max(seqlens),
            causal=causal,
            guard_replay_unroll=unroll,
        )
        for unroll in (1, 2, 64)
    ]
    for index, segment in enumerate(segments):
        rows = slice(int(cu[index]), int(cu[index + 1]))
        assert_output_close(outs[0][rows][None], *segment, causal=causal)
    for out in outs[1:]:
        assert jnp.array_equal(out, outs[0])


def test_guard_replay_unroll_rejects_bad_values():
    q, k, v, _ = packed_qkv([128], 2, 2, 128)
    cu = cu_seqlens_of([128])
    with pytest.raises(ValueError, match="guard_replay_unroll must be >= 1"):
        varlen_attn(q, k, v, cu, cu, 128, 128, guard_replay_unroll=0)
    with pytest.raises(
        ValueError, match="guard_replay_unroll applies to packed"
    ):
        flash_attn_varlen_func(
            to_head_major(q),
            jnp.zeros((4, 128, 2, 128), jnp.bfloat16),
            None,
            cu,
            None,
            128,
            128,
            block_table=jnp.zeros((1, 1), jnp.int32),
            seqused_k=jnp.array([128], jnp.int32),
            guard_replay_unroll=2,
            interpret=INTERPRET,
        )


@pytest.mark.parametrize("entry", ["dense", "varlen"])
@pytest.mark.parametrize("array_type", ["jax", "numpy"])
def test_flash_attn_array_softmax_scale(entry, array_type):
    # A concrete 0-d array scale, such as 1 / jnp.sqrt(d), keys the kernel
    # cache as the Python float it holds.
    d = 128
    array_scale = 1.0 / jnp.sqrt(jnp.float32(d))
    if array_type == "numpy":
        array_scale = np.asarray(array_scale)
    seqlens = [100, 156]
    q, k, v, _ = packed_qkv(seqlens, 2, 2, d)
    if entry == "dense":
        q, k, v = (operand[None] for operand in (q, k, v))

    def attend(softmax_scale):
        if entry == "dense":
            return dense_attn(q, k, v, causal=True, softmax_scale=softmax_scale)
        cu = cu_seqlens_of(seqlens)
        return varlen_attn(
            q,
            k,
            v,
            cu,
            cu,
            max(seqlens),
            max(seqlens),
            causal=True,
            softmax_scale=softmax_scale,
        )

    expected = attend(float(array_scale))
    assert jnp.array_equal(attend(array_scale), expected)


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


@pytest.mark.parametrize(
    "seqlens",
    [
        # (1024, 1024, 256, 512) blocks, two kv tiles: the last sequence's kv
        # window clamps to 1408 - 1024 = 384, so tile 0 is all sequence 0.
        pytest.param([1024, 300], id="two_tiles"),
        pytest.param(
            # (2048, 2048, 256, 512) blocks, four kv tiles: the window clamps
            # to 4992 - 2048 = 2944, so tiles 0 and 1 are all sequence 0.
            [4096, 817],
            id="four_tiles",
            marks=pytest.mark.skipif(
                INTERPRET, reason="interpret mode affords short sequences only"
            ),
        ),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_varlen_clamped_window_first_pass(
    seqlens, causal, monkeypatch
):
    # A short last sequence's kv window clamps to the padded end and starts
    # inside the previous sequence, so its leading kv tiles hold no key of
    # its own. Anchoring the static row max on such a fully masked tile
    # overflows every later tile into NaN. The overflow guard would replay
    # such a block, so it is switched off here and the first pass alone must
    # be right.
    def no_flag(*args, **kwargs):
        del args, kwargs

    monkeypatch.setattr(fwd_blocks, "flag_overflow_risk", no_flag)
    monkeypatch.setattr(fwd_varlen, "flag_overflow_risk", no_flag)
    jax.clear_caches()
    q, k, v, segments = packed_qkv(seqlens, 4, 2, 128)
    cu = cu_seqlens_of(seqlens)
    out = varlen_attn(
        q, k, v, cu, cu, max(seqlens), max(seqlens), causal=causal
    )
    jax.clear_caches()
    # max_abs alone misses the NaN on CPU, whose vectorized max reduction
    # drops NaN in large arrays.
    assert np.isfinite(np.asarray(out, np.float32)).all()
    for index, segment in enumerate(segments):
        rows = slice(int(cu[index]), int(cu[index + 1]))
        assert_output_close(out[rows][None], *segment, causal=causal)


def random_sinks(nheads):
    return jax.random.uniform(
        jax.random.PRNGKey(5), (nheads,), jnp.float32, -2.0, 8.0
    )


def assert_lse_close(lse, q, k, v, **mask_kwargs):
    """lse (batch, nheads, seqlen_q) against the f32 reference, by the out rule
    plus test_prefill.py's fixed 1e-2."""
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
    variant = Variant(softmax_scale=1.0, causal=True, has_sinks=True)
    if entry == "dense":
        kernel = make_flash_attn_mha(
            batch * nheads,
            seqlen,
            seqlen,
            variant=variant,
            block_sizes=blocks,
            head_dim=128,
            interpret=INTERPRET,
        )
        q = jnp.zeros((batch * nheads, seqlen, 128), jnp.bfloat16)
        args = (q, q, q)
    else:
        kernel = make_flash_attn_mha(
            nheads,
            seqlen,
            seqlen,
            variant=variant,
            block_sizes=blocks,
            head_dim=128,
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


# Note (david): the head width whose dense and packed varlen builds run the
# explicit MXU loops (kernels/fwd_dense_mxu.py, kernels/fwd_varlen_mxu.py) on
# a chip with MXU accumulators; everywhere else these shapes take the
# dot_general loops that test_prefill.py's grid covers.
MXU_HEAD_DIM = 256


@requires_mxu_accumulators
@pytest.mark.parametrize(
    "nheads,seqlen,causal",
    [
        (4, 4096, False),
        (4, 4096, True),
        (4, 1024, True),
        (32, 1024, True),
        (4, 512, True),
        (4, 384, False),
        (2, 128, True),
        (1, 128, False),
    ],
)
def test_dense_mxu_dispatch(nheads, seqlen, causal):
    # Note (david): every block size runs the explicit loop: the tuned
    # 32-head 1024-token causal build's head fold becomes one head a row,
    # and blocks below the loop's bodies group into one kv block.
    variant = make_variant(
        MXU_HEAD_DIM**-0.5,
        softcap=0.0,
        causal=causal,
        window_size=(-1, -1),
        learnable_sink=None,
        return_lse=False,
    )
    kernel = get_kernel(
        nheads,
        nheads,
        seqlen,
        seqlen,
        variant,
        False,
        MXU_HEAD_DIM,
        MXU_HEAD_DIM,
    )
    assert kernel.keywords["traits"].uses_mxu_pipeline


@requires_mxu_accumulators
@pytest.mark.parametrize(
    "block_q,block_kv,seqlen,causal",
    [
        (32, 256, 512, False),
        (128, 128, 384, False),
        (16, 128, 128, False),
        (256, 256, 1024, True),
        (128, 128, 128, True),
        (512, 512, 2176, False),
        (512, 512, 4224, True),
    ],
)
def test_dense_mxu_small_blocks(block_q, block_kv, seqlen, causal):
    # Note (david): blocks below the loop's bodies group into larger ones, or
    # into one kv block whose row runs as one segment; a kv axis of no whole
    # number of blocks pads to them, a full row ending at the last body with
    # a valid key and a causal row never meeting the padded keys.
    q, k, v = random_qkv(0, 1, seqlen, 4, 2, MXU_HEAD_DIM)
    variant = make_variant(
        MXU_HEAD_DIM**-0.5,
        softcap=0.0,
        causal=causal,
        window_size=(-1, -1),
        learnable_sink=None,
        return_lse=False,
    )
    kernel = make_flash_attn_mha(
        4,
        seqlen,
        seqlen,
        variant=variant,
        block_sizes=BlockSizes(
            block_q=block_q, block_kv=block_kv, block_q_compute=block_q // 2
        ),
        head_dim=MXU_HEAD_DIM,
        num_kv_heads=2,
        interpret=INTERPRET,
    )
    assert kernel.keywords["traits"].uses_mxu_pipeline
    out = kernel(*(x[0].swapaxes(0, 1) for x in (q, k, v)))
    assert_output_close(out.swapaxes(0, 1)[None], q, k, v, causal=causal)


@requires_mxu_accumulators
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k,window_size",
    [
        (1280, 2048, (-1, -1)),
        (1792, 1024, (-1, -1)),
        (1200, 4096, (-1, -1)),
        (640, 4096, (-1, -1)),
        (1152, 8192, (-1, -1)),
        (1040, 2048, (256, 256)),
        (1664, 8192, (1024, -1)),
        (1040, 1024, (256, -1)),
    ],
)
def test_dense_mxu_q_blocks_of_any_fragment_count(
    seqlen_q, seqlen_k, window_size
):
    # Note (david): a q block whose kv tile's items fill no whole slot lap in
    # LOOP_BODY_TILES tiles runs longer loop bodies; a masked build whose kv
    # block holds fewer than two of them keeps the dot_general loop.
    q, k, v = random_qkv(0, 1, seqlen_q, 4, 4, MXU_HEAD_DIM, seqlen_k=seqlen_k)
    out = dense_attn(q, k, v, window_size=window_size)
    assert_output_close(out, q, k, v, window_size=window_size)


@requires_mxu_accumulators
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k,causal,window_size",
    [
        (64, 128, True, (-1, -1)),
        (113, 203, True, (-1, -1)),
        (640, 128, True, (-1, -1)),
        (1023, 1024, True, (-1, -1)),
        (1024, 1023, True, (-1, -1)),
        (256, 256, False, (174, 14)),
        (512, 256, False, (44, 255)),
        (128, 256, False, (76, 0)),
    ],
)
def test_dense_mxu_masked_single_block(seqlen_q, seqlen_k, causal, window_size):
    # Note (david): a mask without a static causal plan, over blocks too small
    # for masked loop bodies: a kv axis of one block runs every row as one
    # segment with every item masked.
    variant = make_variant(
        MXU_HEAD_DIM**-0.5,
        softcap=0.0,
        causal=causal,
        window_size=window_size,
        learnable_sink=None,
        return_lse=False,
        causal_offset=seqlen_k - seqlen_q,
    )
    kernel = get_kernel(
        4,
        4,
        pad_seqlen(seqlen_q),
        pad_seqlen(seqlen_k),
        variant,
        False,
        MXU_HEAD_DIM,
        MXU_HEAD_DIM,
    )
    assert kernel.keywords["traits"].uses_mxu_pipeline
    q, k, v = random_qkv(0, 1, seqlen_q, 4, 4, MXU_HEAD_DIM, seqlen_k=seqlen_k)
    out = dense_attn(q, k, v, causal=causal, window_size=window_size)
    assert_output_close(out, q, k, v, causal=causal, window_size=window_size)


@requires_mxu_accumulators
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k,window_size",
    [
        (4224, 4224, (429, 3167)),
        (4224, 4224, (512, 512)),
        (2176, 4224, (686, 2555)),
        (128, 4224, (512, 512)),
    ],
)
def test_dense_mxu_masked_blocks_of_one_slot_lap(
    seqlen_q, seqlen_k, window_size
):
    # Note (david): a window over several kv blocks runs masked loop bodies
    # and edge segments in q blocks of one slot lap of fragments, fewer than
    # the pipeline holds in flight.
    variant = make_variant(
        MXU_HEAD_DIM**-0.5,
        softcap=0.0,
        causal=False,
        window_size=window_size,
        learnable_sink=None,
        return_lse=False,
        causal_offset=seqlen_k - seqlen_q,
    )
    kernel = get_kernel(
        4,
        4,
        pad_seqlen(seqlen_q),
        pad_seqlen(seqlen_k),
        variant,
        False,
        MXU_HEAD_DIM,
        MXU_HEAD_DIM,
    )
    traits = kernel.keywords["traits"]
    assert traits.uses_mxu_pipeline
    assert traits.bq == dispatch.DENSE_MXU_MIN_Q_BLOCK
    q, k, v = random_qkv(0, 1, seqlen_q, 4, 4, MXU_HEAD_DIM, seqlen_k=seqlen_k)
    out = dense_attn(q, k, v, window_size=window_size)
    assert_output_close(out, q, k, v, window_size=window_size)


@requires_mxu_accumulators
@pytest.mark.parametrize(
    "seqlens,nheads_k,blocks",
    [
        ([300, 37], 8, (128, 256, 16, 256)),
        ([128] * 6, 4, (128, 256, 16, 256)),
        ([1, 130, 64], 2, (128, 512, 32, 256)),
        # Runs of one-tile rows share start segments (row groups), between
        # lone rows of several tiles.
        ([64, 37, 100, 1, 250, 900, 17, 64, 64, 300], 4, (128, 512, 16, 256)),
        ([40, 90, 1, 128, 60, 700, 30], 8, (128, 256, 16, 256)),
    ],
)
def test_varlen_mxu_small_blocks(seqlens, nheads_k, blocks, monkeypatch):
    # Note (david): pinned blocks of few fragments a tile run the explicit
    # loop as they are: a row's tiles in whole slot laps, those past its live
    # ones hiding every key.
    explicit_bodies = []
    explicit_body = fwd_varlen.varlen_mxu_fwd_body

    def counted_body(*args, **kwargs):
        explicit_bodies.append(args[0])
        return explicit_body(*args, **kwargs)

    monkeypatch.setattr(fwd_varlen, "varlen_mxu_fwd_body", counted_body)
    jax.clear_caches()
    q, k, v, segments = packed_qkv(seqlens, 8, nheads_k, MXU_HEAD_DIM)
    cu = cu_seqlens_of(seqlens)
    out = varlen_attn(
        q,
        k,
        v,
        cu,
        cu,
        max(seqlens),
        max(seqlens),
        causal=True,
        block_sizes=blocks,
    )
    jax.clear_caches()
    assert explicit_bodies
    assert_segments_close(out, segments, cu, causal=True)


@requires_mxu_accumulators
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k,causal,window_size",
    [
        pytest.param(1024, 8192, True, (-1, -1), id="causal-q1k-k8k"),
        pytest.param(3072, 8192, True, (-1, -1), id="causal-q3k-k8k"),
        pytest.param(6144, 7168, True, (-1, -1), id="causal-q6k-k7k"),
        pytest.param(8192, 8192, False, (127, 0), id="window-127"),
        pytest.param(8192, 8192, False, (4095, 0), id="window-4095"),
        pytest.param(8192, 8192, False, (-1, 128), id="right-128"),
        pytest.param(8192, 8192, False, (127, 127), id="both-bounds"),
        pytest.param(8192, 8192, False, (1000, -1), id="left-only"),
    ],
)
def test_dense_mxu_dispatch_takes_every_mask(
    seqlen_q, seqlen_k, causal, window_size
):
    variant = make_variant(
        MXU_HEAD_DIM**-0.5,
        softcap=0.0,
        causal=causal,
        window_size=window_size,
        learnable_sink=None,
        return_lse=False,
        causal_offset=seqlen_k - seqlen_q,
    )
    kernel = get_kernel(
        4, 2, seqlen_q, seqlen_k, variant, False, MXU_HEAD_DIM, MXU_HEAD_DIM
    )
    assert kernel.keywords["traits"].uses_mxu_pipeline


@requires_mxu_accumulators
def test_dense_mxu_dispatch_takes_a_runtime_offset():
    variant = make_variant(
        MXU_HEAD_DIM**-0.5,
        softcap=0.0,
        causal=True,
        window_size=(-1, -1),
        learnable_sink=None,
        return_lse=False,
        runtime_causal_offset=True,
    )
    kernel = make_flash_attn_mha(
        4,
        4096,
        16384,
        variant=variant,
        block_sizes=BlockSizes(block_q=2048, block_kv=2048),
        head_dim=MXU_HEAD_DIM,
        num_kv_heads=2,
    )
    assert kernel.keywords["traits"].uses_mxu_pipeline


@requires_mxu_accumulators
@pytest.mark.parametrize(
    "causal,seqlen_q,seqlen,nheads_k,scaled_quarter",
    [
        (True, 6144, 6144, 4, 0),
        (True, 2048, 2048, 1, 0),
        (False, 4096, 4096, 4, 3),
        (False, 1152, 8192, 4, 3),
    ],
)
def test_dense_mxu_guard_replay(
    causal, seqlen_q, seqlen, nheads_k, scaled_quarter
):
    # Note (david): keys scaled far up in a block a row visits after the one
    # its anchors come from push its row sums past the guard, so the row
    # replays: a causal row anchors on its diagonal block and meets block 0
    # later, a full row anchors on block 0. The sinks and lse read the
    # replayed state; the one-block MQA rows run from one staged block, and
    # the 1152-row q block's longer loop bodies leave no check unit.
    q, k, v = random_qkv(
        0, 1, seqlen_q, 4, nheads_k, MXU_HEAD_DIM, seqlen_k=seqlen
    )
    quarter = seqlen // 4
    keys = slice(scaled_quarter * quarter, (scaled_quarter + 1) * quarter)
    k = k.at[:, keys].multiply(40.0)
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


# A dense guard case runs 2048-token blocks, one check unit each: the last q
# block of a 12288-token causal grid visits blocks 0 to 5 in order (units 0 to
# 5), the last its diagonal block. Units 0 to 3 are checked at the end of the
# next block's first body; units 4 and 5 make the row's final unit.
GUARD_SEQLEN = 12288
GUARD_BLOCK = 2048
GUARD_TILE_KEYS = 256
# A spiked key's score in nats against rows that score 0 everywhere else.
OVERFLOW_SCORE = 200.0
LAST_Q_BLOCK = slice(GUARD_SEQLEN - GUARD_BLOCK, GUARD_SEQLEN)
# The last fragment's rows of the last q block: 16 rows.
DENSE_LAST_FRAGMENT_ROWS = slice(GUARD_SEQLEN - 16, GUARD_SEQLEN)
# Note (david): 2^-64 flags every check unit and every row.
FORCED_GUARD_THRESHOLD = -64.0


def guard_qkv(spikes, rows, *, seqlen=GUARD_SEQLEN, nheads=4, nheads_k=2):
    """random_qkv's (1, seqlen, heads, d) arrays where the q rows rows point
    along one direction and every other q row is zero: every key scores 0
    against them but those of each (start, stop, score) of spikes, which
    score score nats."""
    q, k, v = random_qkv(0, 1, seqlen, nheads, nheads_k, MXU_HEAD_DIM)
    scale = MXU_HEAD_DIM**-0.5
    magnitude = math.sqrt(OVERFLOW_SCORE / scale)
    q, k = (np.array(x, np.float32) for x in (q, k))
    q[:] = 0.0
    q[:, rows, :, 0] = magnitude
    k[..., 0] = 0.0
    for start, stop, score in spikes:
        k[:, start:stop, :, 0] = score / (magnitude * scale)
    return jnp.asarray(q, jnp.bfloat16), jnp.asarray(k, jnp.bfloat16), v


def dense_unit_keys(unit, first_tile, end_tile, *, block=GUARD_BLOCK):
    """(start, stop) of the keys of tiles [first_tile, end_tile) of check
    unit unit of the last q block's row of a causal grid: unit u is its kv
    block u, the last its diagonal block."""
    first_key = unit * block
    return (
        first_key + first_tile * GUARD_TILE_KEYS,
        first_key + end_tile * GUARD_TILE_KEYS,
    )


def assert_checked_build(nheads, nheads_k, seqlen, causal):
    """Assert that flash_attn_func's build of the shape runs the explicit MXU
    loop with checked units."""
    variant = make_variant(
        MXU_HEAD_DIM**-0.5,
        softcap=0.0,
        causal=causal,
        window_size=(-1, -1),
        learnable_sink=None,
        return_lse=False,
    )
    kernel = get_kernel(
        nheads,
        nheads_k,
        seqlen,
        seqlen,
        variant,
        False,
        MXU_HEAD_DIM,
        MXU_HEAD_DIM,
    )
    traits = kernel.keywords["traits"]
    assert traits.uses_mxu_pipeline
    assert dense_has_checked_units(
        seqlen // traits.bkv, traits.bq // FRAGMENT_ROWS
    )


def explicit_dense_attn(
    q, k, v, *, block_q, block_kv, causal, window_size=(-1, -1), sinks=None
):
    """make_flash_attn_mha's dense build of block_q x block_kv blocks on
    (1, seqlen, nheads, d) arrays; the build must run the explicit MXU loop
    with checked units. With sinks it returns (out, lse)."""
    _, seqlen, nheads, head_dim = q.shape
    variant = make_variant(
        head_dim**-0.5,
        softcap=0.0,
        causal=causal,
        window_size=window_size,
        learnable_sink=sinks,
        return_lse=sinks is not None,
    )
    kernel = make_flash_attn_mha(
        nheads,
        seqlen,
        seqlen,
        variant=variant,
        block_sizes=BlockSizes(block_q=block_q, block_kv=block_kv),
        head_dim=head_dim,
        num_kv_heads=k.shape[2],
        interpret=INTERPRET,
    )
    traits = kernel.keywords["traits"]
    assert traits.uses_mxu_pipeline
    assert dense_has_checked_units(
        seqlen // traits.bkv, traits.bq // FRAGMENT_ROWS
    )
    sink_args = () if sinks is None else (sinks,)
    outputs = kernel(*(x[0].swapaxes(0, 1) for x in (q, k, v)), *sink_args)
    if sinks is None:
        return outputs.swapaxes(0, 1)[None]
    out, lse = outputs
    return out.swapaxes(0, 1)[None], lse[None]


DENSE_GUARD_CASES = [
    pytest.param(
        dict(spikes=[(*dense_unit_keys(0, 0, 1), OVERFLOW_SCORE)]),
        id="unit0",
    ),
    pytest.param(
        dict(spikes=[(*dense_unit_keys(2, 1, 2), OVERFLOW_SCORE)]),
        id="middle-unit",
    ),
    pytest.param(
        dict(
            spikes=[(*dense_unit_keys(1, 7, 8), OVERFLOW_SCORE)],
            rows=DENSE_LAST_FRAGMENT_ROWS,
        ),
        id="last-fragments",
    ),
    pytest.param(
        dict(
            spikes=[
                (*dense_unit_keys(4, 3, 4), OVERFLOW_SCORE),
                (*dense_unit_keys(5, 1, 2), 2 * OVERFLOW_SCORE),
            ]
        ),
        id="final-unit",
    ),
    pytest.param(
        dict(
            spikes=[
                (*dense_unit_keys(1, 1, 2), OVERFLOW_SCORE),
                (*dense_unit_keys(3, 2, 3), 2 * OVERFLOW_SCORE),
            ]
        ),
        id="two-units",
    ),
    # Unit 1's replay runs through unit 2's first body, whose keys raise the
    # anchor past the one unit 2's checkpoint was written on; unit 2's last
    # tile then flags unit 2, which replays from that checkpoint.
    pytest.param(
        dict(
            spikes=[
                (*dense_unit_keys(1, 1, 2), OVERFLOW_SCORE),
                (*dense_unit_keys(2, 0, 1), 2 * OVERFLOW_SCORE),
                (*dense_unit_keys(2, 7, 8), 3 * OVERFLOW_SCORE),
            ]
        ),
        id="anchor-raised-by-replay",
    ),
]


@requires_mxu_accumulators
@pytest.mark.parametrize("spec", DENSE_GUARD_CASES)
def test_dense_mxu_guard_replays_flagged_units(spec):
    # Note (david): the spiked keys score ~200 nats against rows anchored at
    # 0, so only a replay keeps those rows finite.
    assert_checked_build(4, 2, GUARD_SEQLEN, causal=True)
    q, k, v = guard_qkv(spec["spikes"], spec.get("rows", LAST_Q_BLOCK))
    out = dense_attn(q, k, v, causal=True)
    assert np.all(np.isfinite(np.asarray(out, np.float32)))
    assert_output_close(out, q, k, v, causal=True)


@requires_mxu_accumulators
def test_dense_mxu_guard_replays_a_middle_unit_with_sinks():
    spikes = [(*dense_unit_keys(2, 1, 2), OVERFLOW_SCORE)]
    q, k, v = guard_qkv(spikes, LAST_Q_BLOCK)
    sinks = random_sinks(4)
    out, lse = flash_attn_func(
        *map(to_head_major, (q, k, v)),
        causal=True,
        learnable_sink=sinks,
        return_softmax_lse=True,
        interpret=INTERPRET,
    )
    assert_output_close(to_head_major(out), q, k, v, causal=True, sinks=sinks)
    assert_lse_close(lse, q, k, v, causal=True, sinks=sinks)


@requires_mxu_accumulators
def test_dense_mxu_guard_replays_full_row_units():
    # Note (david): full rows anchor on block 0's first tile; blocks 0 and 1
    # of the four are checked units, the first replaying from the row start.
    seqlen = 8192
    assert_checked_build(4, 2, seqlen, causal=False)
    spikes = [
        (2 * GUARD_TILE_KEYS, 3 * GUARD_TILE_KEYS, OVERFLOW_SCORE),
        (
            GUARD_BLOCK + 5 * GUARD_TILE_KEYS,
            GUARD_BLOCK + 6 * GUARD_TILE_KEYS,
            2 * OVERFLOW_SCORE,
        ),
    ]
    q, k, v = guard_qkv(spikes, slice(0, seqlen), seqlen=seqlen)
    out = dense_attn(q, k, v, causal=False)
    assert np.all(np.isfinite(np.asarray(out, np.float32)))
    assert_output_close(out, q, k, v, causal=False)


@requires_mxu_accumulators
def test_dense_mxu_guard_replays_units_of_1024_token_blocks():
    # Note (david): the diagonal block's static items close the row, in its
    # final unit with block 4.
    seqlen, block = 6144, 1024
    spikes = [
        (*dense_unit_keys(0, 0, 1, block=block), OVERFLOW_SCORE),
        (
            *dense_unit_keys(2, 1, 2, block=block),
            2 * OVERFLOW_SCORE,
        ),
        (
            *dense_unit_keys(5, 3, 4, block=block),
            3 * OVERFLOW_SCORE,
        ),
    ]
    q, k, v = guard_qkv(spikes, slice(seqlen - block, seqlen), seqlen=seqlen)
    out = explicit_dense_attn(
        q, k, v, block_q=block, block_kv=block, causal=True
    )
    assert np.all(np.isfinite(np.asarray(out, np.float32)))
    assert_output_close(out, q, k, v, causal=True)


@requires_mxu_accumulators
@pytest.mark.parametrize(
    "causal,seqlen,block_q,block_kv",
    [
        pytest.param(True, 8192, 2048, 2048, id="causal"),
        pytest.param(True, 4096, 1024, 1024, id="causal-short-diagonal"),
        pytest.param(False, 8192, 2048, 2048, id="full"),
        pytest.param(False, 6144, 1024, 2048, id="full-two-body-q"),
    ],
)
def test_dense_mxu_forced_guard_replays_units(
    causal, seqlen, block_q, block_kv, monkeypatch
):
    # Note (david): every check unit and every row is flagged: each checked
    # unit replays at the end of the next block's first body, and each row's
    # final unit after the row's drain.
    monkeypatch.setattr(
        dispatch,
        "overflow_guard_threshold",
        lambda kv_seq_len: FORCED_GUARD_THRESHOLD,
    )
    q, k, v = random_qkv(0, 1, seqlen, 4, 2, MXU_HEAD_DIM)
    out = explicit_dense_attn(
        q, k, v, block_q=block_q, block_kv=block_kv, causal=causal
    )
    assert_output_close(out, q, k, v, causal=causal)


# A left window over guard_qkv rows: the last q block's rows first see keys
# 2049 to 4096, where they take their anchors.
DENSE_GUARD_WINDOW = (8191, -1)
DENSE_WINDOW_GUARD_CASES = [
    # Unit 0 (block 1 of 2048 keys) holds the rows' anchor tiles; the rows
    # anchored before its tile 5 meet the spike there.
    pytest.param(2048, 2048, [(3500, 3510, OVERFLOW_SCORE)], id="anchor-unit"),
    # Unit 2 (block 3), after every anchor.
    pytest.param(
        2048, 2048, [(7000, 7010, OVERFLOW_SCORE)], id="after-anchors"
    ),
    # Unit 0 (block 2 of 1024 keys) ends before the anchors of the rows past
    # 11262, which enter unit 1's checkpoint unanchored; unit 1 replays from
    # it.
    pytest.param(
        2048,
        1024,
        [(2900, 2910, OVERFLOW_SCORE), (3900, 3910, 2 * OVERFLOW_SCORE)],
        id="before-anchors",
    ),
    # block_q 1024 against 2048-key blocks runs the masked runtime plan.
    pytest.param(
        1024,
        2048,
        [(3500, 3510, OVERFLOW_SCORE), (7000, 7010, 2 * OVERFLOW_SCORE)],
        id="runtime-plan",
    ),
]


@requires_mxu_accumulators
@pytest.mark.parametrize("block_q,block_kv,spikes", DENSE_WINDOW_GUARD_CASES)
def test_dense_mxu_guard_replays_flagged_units_in_a_window(
    block_q, block_kv, spikes
):
    q, k, v = guard_qkv(spikes, LAST_Q_BLOCK)
    out = explicit_dense_attn(
        q,
        k,
        v,
        block_q=block_q,
        block_kv=block_kv,
        causal=True,
        window_size=DENSE_GUARD_WINDOW,
    )
    assert np.all(np.isfinite(np.asarray(out, np.float32)))
    assert_output_close(
        out, q, k, v, causal=True, window_size=DENSE_GUARD_WINDOW
    )


@requires_mxu_accumulators
@pytest.mark.parametrize(
    "block_q,block_kv",
    [
        pytest.param(2048, 1024, id="static-plan"),
        pytest.param(1024, 2048, id="runtime-plan"),
    ],
)
def test_dense_mxu_forced_guard_replays_window_units_with_sinks(
    block_q, block_kv, monkeypatch
):
    # Note (david): every check unit and every row is flagged under a left
    # window: the replays start before, at and after the rows' anchor tiles.
    monkeypatch.setattr(
        dispatch,
        "overflow_guard_threshold",
        lambda kv_seq_len: FORCED_GUARD_THRESHOLD,
    )
    window = (5000, -1)
    q, k, v = random_qkv(0, 1, 8192, 4, 2, MXU_HEAD_DIM)
    sinks = random_sinks(4)
    out, lse = explicit_dense_attn(
        q,
        k,
        v,
        block_q=block_q,
        block_kv=block_kv,
        causal=True,
        window_size=window,
        sinks=sinks,
    )
    mask_kwargs = dict(causal=True, window_size=window, sinks=sinks)
    assert_output_close(out, q, k, v, **mask_kwargs)
    assert_lse_close(lse, q, k, v, **mask_kwargs)


skip_interpret = pytest.mark.skipif(
    INTERPRET, reason="interpret mode affords short sequences only"
)


def assert_segments_close(out, segments, cu_q, lse=None, **mask_kwargs):
    """Each packed sequence's out (and lse) against its own reference."""
    for index, segment in enumerate(segments):
        rows = slice(int(cu_q[index]), int(cu_q[index + 1]))
        if rows.start == rows.stop:
            continue
        assert_output_close(out[rows][None], *segment, **mask_kwargs)
        if lse is not None:
            assert_lse_close(lse[None, :, rows], *segment, **mask_kwargs)


def unequal_packed_qkv(seqlens_q, seqlens_k, nheads, nheads_k, d):
    """packed_qkv with each sequence's own q and kv lengths."""
    segments = [
        random_qkv(index, 1, seqlen_q, nheads, nheads_k, d, seqlen_k=seqlen_k)
        for index, (seqlen_q, seqlen_k) in enumerate(zip(seqlens_q, seqlens_k))
    ]
    q, k, v = (
        jnp.concatenate([segment[axis][0] for segment in segments])
        for axis in range(3)
    )
    return q, k, v, segments


@pytest.mark.parametrize(
    "seqlens,causal,window_size",
    [
        ([300, 37], True, (-1, -1)),
        ([300, 37], True, (127, -1)),
        ([300, 37], False, (300, 200)),
        ([300, 37], False, (-1, 100)),
        ([16] * 8, True, (-1, -1)),
        ([1, 130], True, (-1, -1)),
        ([16] * 8, True, (127, -1)),
        ([1, 130], False, (300, 200)),
    ],
)
def test_varlen_mxu_takes_the_explicit_path(
    monkeypatch, seqlens, causal, window_size
):
    # Note (david): short sequences take dispatch's small blocks, which the
    # explicit loop runs as they are.
    explicit_bodies = []
    explicit_body = fwd_varlen.varlen_mxu_fwd_body

    def counted_body(*args, **kwargs):
        explicit_bodies.append(args[0])
        return explicit_body(*args, **kwargs)

    monkeypatch.setattr(fwd_varlen, "varlen_mxu_fwd_body", counted_body)
    jax.clear_caches()
    q, k, v, segments = packed_qkv(seqlens, 8, 2, MXU_HEAD_DIM)
    cu = cu_seqlens_of(seqlens)
    out = varlen_attn(
        q,
        k,
        v,
        cu,
        cu,
        max(seqlens),
        max(seqlens),
        causal=causal,
        window_size=window_size,
    )
    jax.clear_caches()
    assert bool(explicit_bodies) == HAS_MXU_ACCUMULATORS
    assert_segments_close(
        out, segments, cu, causal=causal, window_size=window_size
    )


@skip_interpret
@pytest.mark.parametrize("causal", [False, True])
def test_varlen_mxu_guard_replay(causal):
    # Note (david): the second sequence's last keys score far above its rows'
    # first-tile anchors, so its rows replay; the replay loop's unroll factor
    # leaves the output as it is.
    seqlens = [384, 2048, 100]
    q, k, v, segments = packed_qkv(seqlens, 8, 2, MXU_HEAD_DIM)
    k = k.at[384 + 1536 : 384 + 2048].multiply(40.0)
    segments[1] = (segments[1][0], k[None, 384 : 384 + 2048], segments[1][2])
    cu = cu_seqlens_of(seqlens)
    outs = [
        varlen_attn(
            q,
            k,
            v,
            cu,
            cu,
            max(seqlens),
            max(seqlens),
            causal=causal,
            return_softmax_lse=True,
            guard_replay_unroll=unroll,
        )
        for unroll in (1, 2)
    ]
    out, lse = outs[0]
    assert_segments_close(out, segments, cu, lse, causal=causal)
    assert jnp.array_equal(outs[1][0], out)


# A chunk after a long prefix, its 512-token q blocks against kv blocks of 4
# tiles: the chunk's two rows run 26 and 28 live tiles, whose check units 0 to
# 3 and 0 to 4 are checked at the end of the next unit's first body; the rest
# replay with the row.
GUARD_PREFIX = 6144
GUARD_CHUNK = 1024
GUARD_BLOCK_SIZES = (512, 1024, 128, 128)
MXU_TILE_KEYS = 256
GUARD_UNIT_KEYS = 4 * MXU_TILE_KEYS


@pytest.fixture(name="explicit_varlen_builds")
def fixture_explicit_varlen_builds(monkeypatch):
    """The traits of the packed builds traced on the explicit MXU pipeline
    while the test runs; the kernel caches are cleared around it, so every
    call builds and traces its kernel."""
    builds = []
    mxu_body = fwd_varlen.varlen_mxu_fwd_body

    def recorded_body(traits, *args, **kwargs):
        builds.append(traits)
        return mxu_body(traits, *args, **kwargs)

    monkeypatch.setattr(fwd_varlen, "varlen_mxu_fwd_body", recorded_body)
    get_kernel.cache_clear()
    jax.clear_caches()
    yield builds
    get_kernel.cache_clear()
    jax.clear_caches()


def guard_case(nheads, nheads_k, spikes, *, rows=None):
    """(q, k, v, segments) of one GUARD_CHUNK-token chunk whose keys are a
    GUARD_PREFIX-token prefix and itself; its q rows (all when None) point
    along one direction, against which every key scores 0 but those of each
    (start, stop, score) of spikes, which score score nats."""
    seqlen_k = GUARD_PREFIX + GUARD_CHUNK
    q, k, v, _ = unequal_packed_qkv(
        [GUARD_CHUNK], [seqlen_k], nheads, nheads_k, MXU_HEAD_DIM
    )
    scale = MXU_HEAD_DIM**-0.5
    magnitude = math.sqrt(OVERFLOW_SCORE / scale)
    q = np.zeros(q.shape, np.float32)
    q[slice(None) if rows is None else rows, :, 0] = magnitude
    k = np.array(k, np.float32)
    k[..., 0] = 0.0
    for start, stop, score in spikes:
        k[start:stop, :, 0] = score / (magnitude * scale)
    q, k = (jnp.asarray(x, jnp.bfloat16) for x in (q, k))
    return q, k, v, [(q[None], k[None], v[None])]


def run_guard_case(q, k, v, **kwargs):
    """varlen_attn on a guard_case chunk, with GUARD_BLOCK_SIZES."""
    return varlen_attn(
        q,
        k,
        v,
        cu_seqlens_of([GUARD_CHUNK]),
        cu_seqlens_of([GUARD_PREFIX + GUARD_CHUNK]),
        GUARD_CHUNK,
        GUARD_PREFIX + GUARD_CHUNK,
        causal=True,
        return_softmax_lse=True,
        block_sizes=GUARD_BLOCK_SIZES,
        **kwargs,
    )


def varlen_unit_keys(unit, first_tile, end_tile):
    """(start, stop) of the keys of tiles [first_tile, end_tile) of check
    unit unit of a guard_case row."""
    return (
        unit * GUARD_UNIT_KEYS + first_tile * MXU_TILE_KEYS,
        unit * GUARD_UNIT_KEYS + end_tile * MXU_TILE_KEYS,
    )


# The last fragments of a fold-4 512-token q block's tile: its last token chunk
# of 16 rows for every folded head.
VARLEN_LAST_FRAGMENT_ROWS = np.concatenate(
    [np.arange(496, 512), np.arange(1008, 1024)]
)
VARLEN_GUARD_CASES = [
    pytest.param(
        dict(spikes=[(*varlen_unit_keys(0, 2, 3), OVERFLOW_SCORE)]), id="unit0"
    ),
    pytest.param(
        dict(spikes=[(*varlen_unit_keys(2, 1, 2), OVERFLOW_SCORE)]),
        id="middle-unit",
    ),
    pytest.param(
        dict(
            spikes=[(*varlen_unit_keys(1, 3, 4), OVERFLOW_SCORE)],
            rows=VARLEN_LAST_FRAGMENT_ROWS,
        ),
        id="last-fragments",
    ),
    pytest.param(
        dict(
            spikes=[
                (*varlen_unit_keys(5, 1, 2), OVERFLOW_SCORE),
                (GUARD_PREFIX, GUARD_PREFIX + 57, OVERFLOW_SCORE),
            ]
        ),
        id="final-unit",
    ),
    pytest.param(
        dict(
            spikes=[
                (*varlen_unit_keys(1, 1, 2), OVERFLOW_SCORE),
                (*varlen_unit_keys(3, 2, 3), 2 * OVERFLOW_SCORE),
            ]
        ),
        id="two-units",
    ),
    # Unit 1's replay runs through unit 2's first tiles, whose keys raise the
    # anchor past the one unit 2's checkpoint was written on; unit 2's last
    # tile then flags unit 2, which replays from that checkpoint.
    pytest.param(
        dict(
            spikes=[
                (*varlen_unit_keys(1, 1, 2), OVERFLOW_SCORE),
                (*varlen_unit_keys(2, 0, 1), 2 * OVERFLOW_SCORE),
                (*varlen_unit_keys(2, 3, 4), 3 * OVERFLOW_SCORE),
            ]
        ),
        id="anchor-raised-by-replay",
    ),
]


@skip_interpret
@requires_mxu_accumulators
@pytest.mark.parametrize("spec", VARLEN_GUARD_CASES)
def test_varlen_mxu_guard_replays_flagged_units(spec, explicit_varlen_builds):
    # Note (david): the spiked keys score ~200 nats against rows that anchor on
    # their first kv tile at 0, so only a replay keeps those rows finite.
    q, k, v, segments = guard_case(8, 2, **spec)
    out, lse = run_guard_case(q, k, v)
    assert {
        traits.num_k_compute_tiles for traits in explicit_varlen_builds
    } == {GUARD_BLOCK_SIZES[1] // MXU_TILE_KEYS}
    assert jnp.all(jnp.isfinite(out.astype(jnp.float32)))
    assert_segments_close(
        out, segments, cu_seqlens_of([GUARD_CHUNK]), lse, causal=True
    )


# A left window over guard_case rows: q block 0's rows first see keys 2049
# to 2560, so its unit 0 starts at key 2048 and holds every row's anchor.
GUARD_WINDOW = (4095, -1)
VARLEN_WINDOW_GUARD_CASES = [
    # Rows anchored on tile 0 meet the spike on tile 1, where later rows
    # take their anchors: the flagged unit holds anchor tiles.
    pytest.param([(2300, 2311, OVERFLOW_SCORE)], id="anchor-unit"),
    # Unit 2 of q block 0, unit 1's last tile of q block 1.
    pytest.param([(4352, 4401, OVERFLOW_SCORE)], id="after-anchors"),
]


@skip_interpret
@requires_mxu_accumulators
@pytest.mark.parametrize("spikes", VARLEN_WINDOW_GUARD_CASES)
def test_varlen_mxu_guard_replays_flagged_units_in_a_window(
    spikes, explicit_varlen_builds
):
    q, k, v, segments = guard_case(8, 2, spikes)
    out, lse = run_guard_case(q, k, v, window_size=GUARD_WINDOW)
    assert explicit_varlen_builds
    assert jnp.all(jnp.isfinite(out.astype(jnp.float32)))
    assert_segments_close(
        out,
        segments,
        cu_seqlens_of([GUARD_CHUNK]),
        lse,
        causal=True,
        window_size=GUARD_WINDOW,
    )


@skip_interpret
@requires_mxu_accumulators
@pytest.mark.parametrize("nheads,nheads_k", [(8, 2), (4, 4)])
def test_varlen_mxu_guard_replays_a_middle_unit_with_sinks(
    nheads, nheads_k, explicit_varlen_builds
):
    q, k, v, segments = guard_case(
        nheads, nheads_k, [(*varlen_unit_keys(2, 1, 2), OVERFLOW_SCORE)]
    )
    sinks = random_sinks(nheads)
    out, lse = run_guard_case(q, k, v, learnable_sink=sinks)
    assert explicit_varlen_builds
    assert_segments_close(
        out,
        segments,
        cu_seqlens_of([GUARD_CHUNK]),
        lse,
        causal=True,
        sinks=sinks,
    )


@skip_interpret
@requires_mxu_accumulators
@pytest.mark.parametrize(
    "nheads,nheads_k,block_kv",
    [
        pytest.param(8, 2, 512, id="fold4-block2"),
        pytest.param(8, 2, 1024, id="fold4-block4"),
        pytest.param(8, 2, 2048, id="fold4-block8"),
        pytest.param(4, 4, 1024, id="fold1-block4"),
    ],
)
def test_varlen_mxu_forced_guard_replays_units(
    nheads, nheads_k, block_kv, monkeypatch, explicit_varlen_builds
):
    # Note (david): every check unit and every row is flagged: each checked
    # unit replays at the end of the next unit's first body, and each row's
    # final unit as the row is completed, while the next row runs; the
    # 2500-token sequence's rows hold no checked unit and replay whole.
    monkeypatch.setattr(
        dispatch,
        "overflow_guard_threshold",
        lambda seqlen_kv: FORCED_GUARD_THRESHOLD,
    )
    seqlens_q, seqlens_k = [1024, 200, 2500], [6024, 3201, 2500]
    q, k, v, segments = unequal_packed_qkv(
        seqlens_q, seqlens_k, nheads, nheads_k, MXU_HEAD_DIM
    )
    cu_q, cu_k = cu_seqlens_of(seqlens_q), cu_seqlens_of(seqlens_k)
    out, lse = varlen_attn(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max(seqlens_q),
        max(seqlens_k),
        causal=True,
        return_softmax_lse=True,
        block_sizes=(512, block_kv, 128, 128),
    )
    assert {
        traits.num_k_compute_tiles for traits in explicit_varlen_builds
    } == {block_kv // MXU_TILE_KEYS}
    assert_segments_close(out, segments, cu_q, lse, causal=True)


@skip_interpret
@requires_mxu_accumulators
def test_varlen_mxu_check_units_over_six_tile_blocks(explicit_varlen_builds):
    # Note (david): with 1536-key kv blocks the first sequence's check unit 1
    # opens kv block 1 on its last body.
    seqlens = [6144, 300]
    q, k, v, segments = packed_qkv(seqlens, 8, 2, MXU_HEAD_DIM)
    cu = cu_seqlens_of(seqlens)
    out = varlen_attn(
        q,
        k,
        v,
        cu,
        cu,
        max(seqlens),
        max(seqlens),
        causal=True,
        block_sizes=(512, 1536, 128, 128),
    )
    assert {
        traits.num_k_compute_tiles for traits in explicit_varlen_builds
    } == {1536 // MXU_TILE_KEYS}
    assert_segments_close(out, segments, cu, causal=True)


@skip_interpret
@requires_mxu_accumulators
@pytest.mark.parametrize("left", [700, 2500])
def test_varlen_mxu_forced_guard_replays_window_units_with_sinks(
    left, monkeypatch, explicit_varlen_builds
):
    # Note (david): every check unit and row is flagged under a left window:
    # the replays start before, at and after the rows' anchor tiles.
    monkeypatch.setattr(
        dispatch,
        "overflow_guard_threshold",
        lambda seqlen_kv: FORCED_GUARD_THRESHOLD,
    )
    nheads, nheads_k = 8, 2
    seqlens_q, seqlens_k = [1024, 200, 2500], [6024, 3201, 2500]
    q, k, v, segments = unequal_packed_qkv(
        seqlens_q, seqlens_k, nheads, nheads_k, MXU_HEAD_DIM
    )
    sinks = random_sinks(nheads)
    cu_q, cu_k = cu_seqlens_of(seqlens_q), cu_seqlens_of(seqlens_k)
    window = (left, -1)
    out, lse = varlen_attn(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max(seqlens_q),
        max(seqlens_k),
        causal=True,
        window_size=window,
        learnable_sink=sinks,
        return_softmax_lse=True,
        block_sizes=(512, 1024, 128, 128),
    )
    assert explicit_varlen_builds
    assert_segments_close(
        out, segments, cu_q, lse, causal=True, window_size=window, sinks=sinks
    )
