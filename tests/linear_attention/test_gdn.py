"""fused_conv1d_gdn and fused_conv1d_gdn_replay against a naive gated delta
rule.

The chunk tests run one prefill call and the fused_recurrent tests decode
steps, through the fused kernel and its ReplaySSM variant (see harness).
q, k, v are the input of a random short conv, whose output the reference
sees; log decays g enter through encode_gate, and beta is drawn as the
logits the kernel takes. The gate_in_kernel tests check the in-kernel gate
against the same kernel on naive_gdn_gate's log decay. Rows whose conv
width (2 * H + HV) * D is no multiple of 128 skip.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.linear_attention.harness import (
    DECODE,
    GDN,
    TPU_ONLY,
    Fused,
    bf16_values,
    encode_gate,
    param_grid,
)
from tests.linear_attention.reference import (
    assert_close,
    logsigmoid,
    naive_gdn_gate,
    naive_recurrent_gated_delta_rule,
    normalize,
    rand,
    randn,
    repeat_heads,
    uniform,
)

pytestmark = TPU_ONLY


def reference(fused: Fused, beta: jax.Array, g: jax.Array,
              initial_state: jax.Array | None
              ) -> tuple[jax.Array, jax.Array]:
    """The naive recurrence on the conv's activated q, k, v, sequence by
    sequence when packed; beta is the kernel's logits. o is rounded to the
    kernel's bf16 output dtype."""
    num_groups = fused.n_v // fused.n_kq
    inputs = (normalize(repeat_heads(fused.q, num_groups)),
              normalize(repeat_heads(fused.k, num_groups)), fused.v,
              jax.nn.sigmoid(beta), g)
    if not fused.varlen:
        out, state = naive_recurrent_gated_delta_rule(
            *inputs, initial_state=initial_state, output_final_state=True)
        return bf16_values(out), state
    outs, states = [], []
    for seq, (bos, eos) in enumerate(
            zip(fused.cu_seqlens[:-1], fused.cu_seqlens[1:])):
        out, state = naive_recurrent_gated_delta_rule(
            *(x[:, bos:eos] for x in inputs),
            initial_state=initial_state[seq:seq + 1],
            output_final_state=True)
        outs.append(out)
        states.append(state)
    return bf16_values(jnp.concatenate(outs, axis=1)), jnp.concatenate(states)


@DECODE
@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "D", "gate_logit_normalizer"),
    param_grid(
        "B{}-T{}-H{}-HV{}-D{}-gate_logit_normalizer{}",
        [
            (1, 63, 1, 1, 64, 1),
            (2, 500, 4, 4, 60, 1),
            (2, 1000, 2, 8, 128, 0.1),
            (3, 1024, 2, 2, 128, 1),
            (4, 1024, 3, 3, 128, 10),
            (4, 2048, 4, 4, 64, 1),
            (2, 1024, 4, 4, 128, 0.1),
            (2, 1024, 4, 8, 128, 10),
        ],
        lambda B, T, H, HV, D, *_: (H, HV, D),
    ),
)
def test_fused_recurrent(B: int, T: int, H: int, HV: int, D: int,
                         gate_logit_normalizer: float,
                         record_dtype: jnp.dtype | None) -> None:
    rng = np.random.default_rng(42)
    q = randn(rng, (B, T, H, D))
    k = randn(rng, (B, T, H, D))
    v = randn(rng, (B, T, HV, D))
    beta = rand(rng, (B, T, HV))
    g = logsigmoid(rand(rng, (B, T, HV))) / gate_logit_normalizer
    h0 = randn(rng, (B, HV, D, D))
    fused = Fused(GDN, rng, q, k, v, initial_state=h0)
    raw_g, g = encode_gate(GDN, g)

    ref, ref_ht = reference(fused, beta, g, h0)
    tri, tri_ht = fused.recurrent(raw_g, beta, record_dtype)
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


# Note (david): ratio bounds both o and ht. Rows above 0.005 carry their
# measured error plus margin; that error is the kernel's bf16 rounding of
# conv tap products, of the dots that invert T and of its MXU operands.
# mask_p = 1.0 keeps g = 0, so no decay damps T and its inversion loses most.
@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "D", "gate_logit_normalizer", "mask_p", "ratio"),
    param_grid(
        "B{}-T{}-H{}-HV{}-D{}-gate_logit_normalizer{}-mask_p{}",
        [
            (4, 1024, 4, 4, 128, 1, 1.0, 0.012),
            (2, 75, 4, 4, 64, 0.01, 0, 0.005),
            (2, 500, 3, 3, 60, 1, 0, 0.005),
            (2, 1000, 3, 3, 64, 1, 0.5, 0.005),
            (3, 1024, 4, 4, 100, 0.1, 0, 0.005),
            (4, 1024, 4, 4, 128, 1, 0, 0.006),
            (2, 1500, 4, 4, 128, 10, 0, 0.005),
            (4, 2048, 8, 8, 64, 1, 0, 0.005),
            (2, 256, 2, 4, 64, 1, 0, 0.007),
            (2, 512, 2, 8, 64, 0.1, 0, 0.0065),
            (2, 1024, 4, 8, 128, 1, 0, 0.005),
        ],
        lambda B, T, H, HV, D, *_: (H, HV, D),
    ),
)
def test_chunk(B: int, T: int, H: int, HV: int, D: int,
               gate_logit_normalizer: float, mask_p: float,
               ratio: float) -> None:
    rng = np.random.default_rng(42)
    q = rand(rng, (B, T, H, D))
    k = rand(rng, (B, T, H, D))
    v = rand(rng, (B, T, HV, D))
    beta = rand(rng, (B, T, HV))
    g = logsigmoid(rand(rng, (B, T, HV))) / gate_logit_normalizer
    g = g * (rand(rng, g.shape) > mask_p)
    h0 = jnp.zeros((B, HV, D, D))
    fused = Fused(GDN, rng, q, k, v, initial_state=h0)
    raw_g, g = encode_gate(GDN, g)

    tri, tri_ht = fused.chunk(raw_g, beta)
    ref, ref_ht = reference(fused, beta, g, h0)
    assert_close("o", ref, tri, ratio)
    assert_close("ht", ref_ht, tri_ht, ratio)


@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "D", "gate_logit_normalizer", "chunk_size"),
    param_grid(
        "B{}-T{}-H{}-HV{}-D{}-gate{}-chunk{}",
        [(1, 64, 2, 4, 32, 1.0, chunk_size) for chunk_size in [16, 32, 64]],
        lambda B, T, H, HV, D, *_: (H, HV, D),
    ),
)
def test_chunk_with_chunk_size(B: int, T: int, H: int, HV: int, D: int,
                               gate_logit_normalizer: float,
                               chunk_size: int) -> None:
    rng = np.random.default_rng(42)
    q = rand(rng, (B, T, H, D))
    k = rand(rng, (B, T, H, D))
    v = rand(rng, (B, T, HV, D))
    beta = rand(rng, (B, T, HV))
    g = logsigmoid(rand(rng, (B, T, HV))) / gate_logit_normalizer
    h0 = jnp.zeros((B, HV, D, D))
    fused = Fused(GDN, rng, q, k, v, initial_state=h0)
    raw_g, g = encode_gate(GDN, g)

    ref_o, ref_ht = reference(fused, beta, g, h0)
    tri_o, tri_ht = fused.chunk(raw_g, beta, chunk_size=chunk_size)
    assert_close(f"o@{chunk_size}", ref_o, tri_o, 0.005)
    assert_close(f"ht@{chunk_size}", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "D"),
    param_grid(
        "B{}-T{}-H{}-HV{}-D{}",
        [
            (2, 500, 4, 4, 64),
            (4, 2048, 8, 8, 64),
        ],
        lambda B, T, H, HV, D: (H, HV, D),
    ),
)
def test_chunk_beta_sigmoid_in_kernel(B: int, T: int, H: int, HV: int,
                                      D: int) -> None:
    """The kernel's in-kernel sigmoid of raw beta logits matches the
    reference's manual sigmoid."""
    rng = np.random.default_rng(42)
    q = rand(rng, (B, T, H, D))
    k = rand(rng, (B, T, H, D))
    v = rand(rng, (B, T, HV, D))
    beta = randn(rng, (B, T, HV))
    g = logsigmoid(rand(rng, (B, T, HV)))
    h0 = jnp.zeros((B, HV, D, D))
    fused = Fused(GDN, rng, q, k, v, initial_state=h0)
    raw_g, g = encode_gate(GDN, g)

    tri, tri_ht = fused.chunk(raw_g, beta)
    ref, ref_ht = reference(fused, beta, g, h0)
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@DECODE
@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "D", "has_dt_bias"),
    param_grid(
        "B{}-T{}-H{}-HV{}-D{}-has_dt_bias{}",
        [
            (1, 64, 1, 1, 64, False),
            (2, 256, 2, 2, 64, True),
            (2, 512, 2, 4, 64, True),
            (3, 1000, 2, 8, 128, False),
            (4, 1024, 4, 4, 128, True),
        ],
        lambda B, T, H, HV, D, *_: (H, HV, D),
    ),
)
def test_fused_recurrent_gate_in_kernel(B: int, T: int, H: int, HV: int,
                                        D: int, has_dt_bias: bool,
                                        record_dtype: jnp.dtype | None
                                        ) -> None:
    """The in-kernel gate matches the manual gate path."""
    rng = np.random.default_rng(42)
    q = randn(rng, (B, T, H, D))
    k = randn(rng, (B, T, H, D))
    v = randn(rng, (B, T, HV, D))
    beta = rand(rng, (B, T, HV))
    g_raw = randn(rng, (B, T, HV))
    A_log = jnp.log(uniform(rng, (HV, ), 1, 16))
    dt_bias = randn(rng, (HV, )) if has_dt_bias else None
    h0 = randn(rng, (B, HV, D, D))
    fused = Fused(GDN, rng, q, k, v, initial_state=h0)

    g_ref, _ = encode_gate(GDN, naive_gdn_gate(g_raw, A_log, dt_bias))
    ref, ref_ht = fused.recurrent(g_ref, beta, record_dtype)
    tri, tri_ht = fused.recurrent(g_raw, beta, record_dtype, A_log=A_log,
                                  dt_bias=dt_bias)
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@DECODE
@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "D"),
    param_grid(
        "B{}-T{}-H{}-HV{}-D{}",
        [
            (1, 64, 1, 1, 64),
            (4, 1024, 4, 4, 128),
        ],
        lambda B, T, H, HV, D: (H, HV, D),
    ),
)
def test_fused_recurrent_beta_sigmoid_in_kernel(
        B: int, T: int, H: int, HV: int, D: int,
        record_dtype: jnp.dtype | None) -> None:
    """The in-kernel sigmoid of raw beta logits matches the reference's
    manual sigmoid."""
    rng = np.random.default_rng(42)
    q = randn(rng, (B, T, H, D))
    k = randn(rng, (B, T, H, D))
    v = randn(rng, (B, T, HV, D))
    beta = randn(rng, (B, T, HV))
    g = logsigmoid(rand(rng, (B, T, HV)))
    h0 = randn(rng, (B, HV, D, D))
    fused = Fused(GDN, rng, q, k, v, initial_state=h0)
    raw_g, g = encode_gate(GDN, g)

    ref, ref_ht = reference(fused, beta, g, h0)
    tri, tri_ht = fused.recurrent(raw_g, beta, record_dtype)
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@DECODE
@pytest.mark.parametrize(
    ("H", "HV", "D", "has_dt_bias", "cu_seqlens"),
    param_grid(
        "H{}-HV{}-D{}-has_dt_bias{}-cu_seqlens{}",
        [
            (2, 2, 64, True, [0, 15, 100, 300]),
            (2, 4, 64, False, [0, 256, 500, 1000]),
            (4, 4, 128, True, [0, 15, 100, 300, 1200, 2000]),
        ],
        lambda H, HV, D, *_: (H, HV, D),
    ),
)
def test_fused_recurrent_gate_in_kernel_varlen(
        H: int, HV: int, D: int, has_dt_bias: bool, cu_seqlens: list[int],
        record_dtype: jnp.dtype | None) -> None:
    """Varlen decode with the in-kernel gate."""
    rng = np.random.default_rng(42)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = randn(rng, (1, T, H, D))
    k = randn(rng, (1, T, H, D))
    v = randn(rng, (1, T, HV, D))
    beta = rand(rng, (1, T, HV))
    g_raw = randn(rng, (1, T, HV))
    A_log = jnp.log(uniform(rng, (HV, ), 1, 16))
    dt_bias = randn(rng, (HV, )) if has_dt_bias else None
    h0 = randn(rng, (N, HV, D, D))
    fused = Fused(GDN, rng, q, k, v, cu_seqlens=cu_seqlens, initial_state=h0)

    g_ref, _ = encode_gate(GDN, naive_gdn_gate(g_raw, A_log, dt_bias))
    ref, ref_ht = fused.recurrent(g_ref, beta, record_dtype)
    tri, tri_ht = fused.recurrent(g_raw, beta, record_dtype, A_log=A_log,
                                  dt_bias=dt_bias)
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ("H", "HV", "D", "mask_p", "cu_seqlens"),
    param_grid(
        "H{}-HV{}-D{}-mask_p{}-cu_seqlens{}",
        [
            (4, 4, 60, 0, [0, 15]),
            (4, 4, 64, 0, [0, 256, 500, 1000]),
            (4, 4, 64, 0.5, [0, 256, 500, 1000]),
            (4, 4, 100, 0, [0, 15, 100, 300, 1200, 2000]),
            (2, 4, 64, 0, [0, 256, 500, 1000]),
            (2, 8, 64, 0, [0, 256, 500, 1000]),
        ],
        lambda H, HV, D, *_: (H, HV, D),
    ),
)
def test_chunk_varlen(H: int, HV: int, D: int, mask_p: float,
                      cu_seqlens: list[int]) -> None:
    rng = np.random.default_rng(42)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = randn(rng, (1, T, H, D))
    k = normalize(randn(rng, (1, T, H, D)))
    v = randn(rng, (1, T, HV, D))
    g = logsigmoid(rand(rng, (1, T, HV)))
    g = g * (rand(rng, g.shape) > mask_p)
    beta = rand(rng, (1, T, HV))
    h0 = randn(rng, (N, HV, D, D))
    fused = Fused(GDN, rng, q, k, v, cu_seqlens=cu_seqlens, initial_state=h0)
    raw_g, g = encode_gate(GDN, g)

    tri, tri_ht = fused.chunk(raw_g, beta)
    ref, ref_ht = reference(fused, beta, g, h0)
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ("H", "D", "mask_p", "cu_seqlens"),
    param_grid(
        "H{}-D{}-mask_p{}-cu_seqlens{}",
        [
            (4, 60, 0, [0, 8192]),
            (4, 60, 0, [0, 15]),
            (4, 64, 0, [0, 256, 500, 1000]),
            (4, 64, 0.5, [0, 256, 500, 1000]),
            (4, 100, 0, [0, 15, 100, 300, 1200, 2000]),
        ],
        lambda H, D, *_: (H, H, D),
    ),
)
def test_chunk_varlen_prefill(H: int, D: int, mask_p: float,
                              cu_seqlens: list[int]) -> None:
    rng = np.random.default_rng(42)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = randn(rng, (1, T, H, D))
    k = normalize(randn(rng, (1, T, H, D)))
    v = randn(rng, (1, T, H, D))
    g = logsigmoid(rand(rng, (1, T, H)))
    g = g * (rand(rng, g.shape) > mask_p)
    beta = rand(rng, (1, T, H))
    h0 = randn(rng, (N, H, D, D))
    fused = Fused(GDN, rng, q, k, v, cu_seqlens=cu_seqlens, initial_state=h0)
    raw_g, g = encode_gate(GDN, g)

    tri, tri_ht = fused.chunk(raw_g, beta)
    ref, ref_ht = reference(fused, beta, g, h0)
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "has_dt_bias"),
    param_grid(
        "B{}-T{}-H{}-D{}-has_dt_bias{}",
        [
            (2, 75, 4, 64, True),
            (2, 500, 3, 60, False),
            (2, 1000, 3, 64, True),
            (3, 1024, 4, 100, True),
            (4, 1024, 4, 128, False),
            (4, 2048, 8, 64, True),
        ],
        lambda B, T, H, D, _: (H, H, D),
    ),
)
def test_chunk_gate_in_kernel(B: int, T: int, H: int, D: int,
                              has_dt_bias: bool) -> None:
    """The in-kernel gate matches the manual gate path."""
    rng = np.random.default_rng(42)
    q = rand(rng, (B, T, H, D))
    k = rand(rng, (B, T, H, D))
    v = rand(rng, (B, T, H, D))
    beta = rand(rng, (B, T, H))
    g_raw = randn(rng, (B, T, H))
    A_log = randn(rng, (H, ))
    dt_bias = randn(rng, (H, )) if has_dt_bias else None
    h0 = jnp.zeros((B, H, D, D))
    fused = Fused(GDN, rng, q, k, v, initial_state=h0)

    tri, tri_ht = fused.chunk(g_raw, beta, A_log=A_log, dt_bias=dt_bias)
    g_ref, _ = encode_gate(GDN, naive_gdn_gate(g_raw, A_log, dt_bias))
    ref, ref_ht = fused.chunk(g_ref, beta)
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ("B", "T", "Hq", "H", "D", "has_dt_bias"),
    param_grid(
        "B{}-T{}-Hq{}-H{}-D{}-has_dt_bias{}",
        [
            (2, 256, 2, 4, 64, True),
            (2, 512, 1, 4, 64, False),
            (2, 512, 2, 8, 64, True),
            (2, 1024, 4, 8, 128, True),
        ],
        lambda B, T, Hq, H, D, _: (Hq, H, D),
    ),
)
def test_chunk_gate_in_kernel_gqa(B: int, T: int, Hq: int, H: int, D: int,
                                  has_dt_bias: bool) -> None:
    """The in-kernel gate with grouped value heads (H > Hq)."""
    rng = np.random.default_rng(42)
    q = rand(rng, (B, T, Hq, D))
    k = rand(rng, (B, T, Hq, D))
    v = rand(rng, (B, T, H, D))
    beta = rand(rng, (B, T, H))
    g_raw = randn(rng, (B, T, H))
    A_log = randn(rng, (H, ))
    dt_bias = randn(rng, (H, )) if has_dt_bias else None
    h0 = jnp.zeros((B, H, D, D))
    fused = Fused(GDN, rng, q, k, v, initial_state=h0)

    tri, tri_ht = fused.chunk(g_raw, beta, A_log=A_log, dt_bias=dt_bias)
    g_ref, _ = encode_gate(GDN, naive_gdn_gate(g_raw, A_log, dt_bias))
    ref, ref_ht = fused.chunk(g_ref, beta)
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ("H", "D", "has_dt_bias", "cu_seqlens"),
    param_grid(
        "H{}-D{}-has_dt_bias{}-cu_seqlens{}",
        [
            (4, 60, True, [0, 15]),
            (4, 64, False, [0, 256, 500, 1000]),
            (4, 64, True, [0, 256, 500, 1000]),
            (4, 100, True, [0, 15, 100, 300, 1200, 2000]),
        ],
        lambda H, D, *_: (H, H, D),
    ),
)
def test_chunk_gate_in_kernel_varlen(H: int, D: int, has_dt_bias: bool,
                                     cu_seqlens: list[int]) -> None:
    """The in-kernel gate with variable-length sequences."""
    rng = np.random.default_rng(42)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = randn(rng, (1, T, H, D))
    k = randn(rng, (1, T, H, D))
    v = randn(rng, (1, T, H, D))
    beta = rand(rng, (1, T, H))
    g_raw = randn(rng, (1, T, H))
    A_log = randn(rng, (H, ))
    dt_bias = randn(rng, (H, )) if has_dt_bias else None
    h0 = randn(rng, (N, H, D, D))
    fused = Fused(GDN, rng, q, k, v, cu_seqlens=cu_seqlens, initial_state=h0)

    tri, tri_ht = fused.chunk(g_raw, beta, A_log=A_log, dt_bias=dt_bias)
    g_ref, _ = encode_gate(GDN, naive_gdn_gate(g_raw, A_log, dt_bias))
    ref, ref_ht = fused.chunk(g_ref, beta)
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
