"""fused_conv1d_kda and fused_conv1d_kda_replay against a naive KDA.

The chunk tests run one prefill call and the fused_recurrent tests decode
steps, through the fused kernel and its ReplaySSM variant (see harness).
q, k, v are the input of a random short conv, whose output the reference
sees. An activated beta enters with beta_is_activated; a log decay g enters
through encode_gate, and a raw gate (use_gate_in_kernel) as the kernel's
bf16 gate. test_fused_recurrent_gate_in_kernel checks the in-kernel gate
against the same kernel on the naive gate's log decay, and
test_fused_recurrent_use_beta_sigmoid_in_kernel checks raw beta logits
against the activated beta. The grids hold only rows whose conv width
(2 * H + HV) * D is a multiple of 128, since the kernels' conv works on
whole 128-lane slabs of qkv.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.linear_attention.harness import (
    DECODE,
    KDA,
    TPU_ONLY,
    Fused,
    bf16_values,
    encode_gate,
    logit,
    param_grid,
)
from tests.linear_attention.reference import (
    assert_close,
    logsigmoid,
    naive_kda_gate,
    naive_kda_lowerbound_gate,
    naive_recurrent_kda,
    normalize,
    rand,
    randn,
    uniform,
)

pytestmark = TPU_ONLY


def reference(fused: Fused, g: jax.Array, beta: jax.Array,
              initial_state: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The naive KDA on the conv's activated q, k, v, sequence by sequence
    when packed. o is rounded to the kernel's bf16 output dtype."""
    inputs = (normalize(fused.q), normalize(fused.k), fused.v, g, beta)
    if not fused.varlen:
        out, state = naive_recurrent_kda(*inputs,
                                         initial_state=initial_state,
                                         output_final_state=True)
        return bf16_values(out), state
    outs, states = [], []
    for seq, (bos, eos) in enumerate(
            zip(fused.cu_seqlens[:-1], fused.cu_seqlens[1:])):
        out, state = naive_recurrent_kda(
            *(x[:, bos:eos] for x in inputs),
            initial_state=initial_state[seq:seq + 1],
            output_final_state=True)
        outs.append(out)
        states.append(state)
    return bf16_values(jnp.concatenate(outs, axis=1)), jnp.concatenate(states)


def gate_inputs(
    g: jax.Array,
    use_gate_in_kernel: bool,
    lower_bound: float | None = None,
    A_log: jax.Array | None = None,
    dt_bias: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, dict]:
    """The kernel's bf16 raw gate and gate kwargs for g, and the log decay
    the reference sees.

    With use_gate_in_kernel g is the raw gate, which the reference activates
    with naive_kda_gate, or naive_kda_lowerbound_gate under lower_bound;
    otherwise g is the log decay itself.
    """
    if not use_gate_in_kernel:
        raw_g, g_ref = encode_gate(KDA, g, lower_bound)
        return raw_g, g_ref, dict(lower_bound=lower_bound)
    raw_g = g.astype(jnp.bfloat16)
    if lower_bound is None:
        g_ref = naive_kda_gate(raw_g, A_log, dt_bias)
    else:
        g_ref = naive_kda_lowerbound_gate(raw_g, A_log, dt_bias, lower_bound)
    return raw_g, g_ref, dict(A_log=A_log, dt_bias=dt_bias,
                              lower_bound=lower_bound)


def test_chunk_invalid_chunk_size() -> None:
    rng = np.random.default_rng(42)
    B, T, H, D = 1, 64, 1, 64
    q = randn(rng, (B, T, H, D))
    k = randn(rng, (B, T, H, D))
    v = randn(rng, (B, T, H, D))
    g = randn(rng, (B, T, H, D))
    beta = randn(rng, (B, T, H))
    fused = Fused(KDA, rng, q, k, v)

    with pytest.raises(
            ValueError,
            match="compute_chunk_size must be a positive multiple of 16"):
        fused.chunk(g.astype(jnp.bfloat16), beta, chunk_size=24)


@DECODE
@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "D", "gate_logit_normalizer"),
    param_grid(
        "B{}-T{}-H{}-HV{}-D{}-gate_logit_normalizer{}",
        [
            (4, 1024, 4, 4, 128, 1),
            (2, 1024, 2, 8, 128, 1),
        ],
    ),
)
def test_fused_recurrent(B: int, T: int, H: int, HV: int, D: int,
                         gate_logit_normalizer: float,
                         record_dtype: jnp.dtype | None) -> None:
    rng = np.random.default_rng(42)
    q = rand(rng, (B, T, H, D))
    k = rand(rng, (B, T, H, D))
    v = rand(rng, (B, T, HV, D))
    g = logsigmoid(randn(rng, (B, T, HV, D))) / gate_logit_normalizer
    beta = jax.nn.sigmoid(randn(rng, (B, T, HV)))
    h0 = randn(rng, (B, HV, D, D))
    fused = Fused(KDA, rng, q, k, v, initial_state=h0)
    raw_g, g = encode_gate(KDA, g)

    ref, ref_ht = reference(fused, g, beta, h0)
    tri, tri_ht = fused.recurrent(raw_g, beta, record_dtype,
                                  beta_is_activated=True)
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@DECODE
@pytest.mark.parametrize(
    ("B", "T", "H", "D"),
    param_grid(
        "B{}-T{}-H{}-D{}",
        [
            (2, 256, 4, 64),
        ],
    ),
)
def test_fused_recurrent_use_beta_sigmoid_in_kernel(
        B: int, T: int, H: int, D: int,
        record_dtype: jnp.dtype | None) -> None:
    """Raw beta logits with the in-kernel sigmoid match the activated beta."""
    rng = np.random.default_rng(42)
    q = rand(rng, (B, T, H, D))
    k = rand(rng, (B, T, H, D))
    v = rand(rng, (B, T, H, D))
    g = logsigmoid(randn(rng, (B, T, H, D)))
    beta_post = jax.nn.sigmoid(randn(rng, (B, T, H)))
    beta_raw = logit(jnp.clip(beta_post, 1e-4, 1 - 1e-4))
    h0 = randn(rng, (B, H, D, D))
    fused = Fused(KDA, rng, q, k, v, initial_state=h0)
    raw_g, _ = encode_gate(KDA, g)

    ref, ref_ht = fused.recurrent(raw_g, beta_post, record_dtype,
                                  beta_is_activated=True)
    tri, tri_ht = fused.recurrent(raw_g, beta_raw, record_dtype)
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@DECODE
@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "D", "has_a_log", "has_dt_bias", "safe_gate"),
    param_grid(
        "B{}-T{}-H{}-HV{}-D{}-has_a_log{}-has_dt_bias{}-safe_gate{}",
        [
            (2, 256, 2, 2, 64, True, True, False),
            (2, 512, 2, 4, 64, True, True, True),
            (3, 1000, 2, 8, 128, True, False, False),
            (4, 1024, 4, 4, 128, True, True, True),
            (2, 256, 2, 4, 64, False, True, True),
        ],
    ),
)
def test_fused_recurrent_gate_in_kernel(B: int, T: int, H: int, HV: int,
                                        D: int, has_a_log: bool,
                                        has_dt_bias: bool, safe_gate: bool,
                                        record_dtype: jnp.dtype | None
                                        ) -> None:
    """The in-kernel gate matches the manual gate path."""
    rng = np.random.default_rng(42)
    q = rand(rng, (B, T, H, D))
    k = rand(rng, (B, T, H, D))
    v = rand(rng, (B, T, HV, D))
    beta = jax.nn.sigmoid(rand(rng, (B, T, HV)))
    g_raw = randn(rng, (B, T, HV, D))
    A_log = jnp.log(uniform(rng, (HV, ), 1, 16)) if has_a_log else None
    dt_bias = randn(rng, (HV * D, )) if has_dt_bias else None
    h0 = randn(rng, (B, HV, D, D))
    fused = Fused(KDA, rng, q, k, v, initial_state=h0)

    lower_bound = -5.0 if safe_gate else None
    raw_g, g_ref, gate_kwargs = gate_inputs(g_raw, True, lower_bound, A_log,
                                            dt_bias)
    raw_g_ref, _ = encode_gate(KDA, g_ref)
    ref, ref_ht = fused.recurrent(raw_g_ref, beta, record_dtype,
                                  beta_is_activated=True)
    tri, tri_ht = fused.recurrent(raw_g, beta, record_dtype,
                                  beta_is_activated=True, **gate_kwargs)
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@DECODE
@pytest.mark.parametrize(
    ("B", "H", "D", "gate_logit_normalizer", "use_gate_in_kernel",
     "safe_gate"),
    param_grid(
        "B{}-H{}-D{}-norm{}-gate{}-safe_gate{}",
        [
            (16, 16, 128, 1.0, False, False),
            (32, 8, 64, 1.0, False, False),
            (7, 32, 128, 0.5, False, False),
            (16, 16, 128, 1.0, True, False),
            (32, 8, 64, 1.0, True, False),
            (7, 32, 128, 0.5, True, True),
        ],
    ),
)
def test_fused_recurrent_vllm_decode(B: int, H: int, D: int,
                                     gate_logit_normalizer: float,
                                     use_gate_in_kernel: bool,
                                     safe_gate: bool,
                                     record_dtype: jnp.dtype | None) -> None:
    """vLLM-style decoding with continuous batching and paged state
    storage."""
    rng = np.random.default_rng(42)
    max_cache_slots = B * 3
    # Note (david): slot 0 is the kernels' null block, so the pool keeps it
    # in front of the max_cache_slots slots.
    state_indices = 1 + rng.permutation(max_cache_slots)[:B]
    h0 = randn(rng, (B, H, D, D))

    # fill unaccessed slots with a huge value to detect out-of-bounds access.
    HUGE_VALUE = 1e30
    dim = 3 * H * D
    state_pool = np.full((1 + max_cache_slots, H, D, D), HUGE_VALUE,
                         np.float32)
    conv_pool = np.full((1 + max_cache_slots, 3, dim), HUGE_VALUE,
                        np.float32)

    T = 1
    total_tokens = B * T
    q = rand(rng, (1, total_tokens, H, D))
    k = rand(rng, (1, total_tokens, H, D))
    v = rand(rng, (1, total_tokens, H, D))
    g = randn(rng, (1, total_tokens, H, D))
    if use_gate_in_kernel:
        A_log = jnp.log(uniform(rng, (H, ), 1, 16))
        dt_bias = randn(rng, (H * D, ))
        lower_bound = -5.0 if safe_gate else None
    else:
        g = logsigmoid(g) / gate_logit_normalizer
        A_log = None
        dt_bias = None
        lower_bound = None
    beta = jax.nn.sigmoid(randn(rng, (1, total_tokens, H)))

    cu_seqlens = list(range(0, total_tokens + 1, T))
    fused = Fused(KDA, rng, q, k, v, cu_seqlens=cu_seqlens, initial_state=h0,
                  state_indices=state_indices, state_pool=state_pool,
                  conv_pool=conv_pool)
    raw_g, g_ref, gate_kwargs = gate_inputs(g, use_gate_in_kernel,
                                            lower_bound, A_log, dt_bias)
    ref_out, ref_ht = reference(fused, g_ref, beta, h0)
    tri_out, tri_ht = fused.recurrent(raw_g, beta, record_dtype,
                                      beta_is_activated=True, **gate_kwargs)

    assert_close("o", ref_out, tri_out, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)

    mask = np.ones(1 + max_cache_slots, dtype=bool)
    mask[state_indices] = False
    mask[0] = False
    assert_close("Untouched ht", fused.state_pool[mask],
                 fused.final_state_pool[mask], 0.0)
    assert_close("Untouched conv", fused.conv_pool[mask],
                 fused.final_conv_pool[mask], 0.0)


@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "D", "gate_logit_normalizer", "mask_p",
     "use_gate_in_kernel", "safe_gate", "chunk_size"),
    param_grid(
        "B{}-T{}-H{}-HV{}-D{}-gate_logit_normalizer{}-mask_p{}"
        "-use_gate{}-safe_gate{}-chunk_size{}",
        [
            (4, 1024, 4, 4, 128, 1, 0, False, True, 64),
            (2, 1500, 4, 4, 128, 10, 0, True, False, 64),
            (4, 2048, 8, 8, 64, 1, 0, True, True, 64),
            (2, 1024, 2, 4, 64, 1, 0, False, False, 64),
            (2, 1024, 2, 8, 64, 1, 0, True, False, 64),
            (2, 1024, 4, 8, 128, 1, 0, True, False, 64),
            (2, 160, 2, 4, 64, 1, 0, True, True, 32),
            (2, 1024, 2, 4, 64, 1, 0, True, True, 64),
            (2, 1024, 2, 8, 128, 1, 0, True, False, 64),
        ],
    ),
)
def test_chunk(B: int, T: int, H: int, HV: int, D: int,
               gate_logit_normalizer: float, mask_p: float,
               use_gate_in_kernel: bool, safe_gate: bool,
               chunk_size: int) -> None:
    rng = np.random.default_rng(42)
    q = rand(rng, (B, T, H, D))
    k = rand(rng, (B, T, H, D))
    v = rand(rng, (B, T, HV, D))
    g = randn(rng, (B, T, HV, D))
    if use_gate_in_kernel:
        A_log = randn(rng, (HV, ))
        dt_bias = randn(rng, (HV * D, ))
    else:
        A_log = dt_bias = None
        g = logsigmoid(g) / gate_logit_normalizer
        g = g * (rand(rng, g.shape) > mask_p)
    if safe_gate:
        lower_bound = -5.0
        if not use_gate_in_kernel:
            g = jnp.clip(g, -5, 0)
    else:
        lower_bound = None

    beta = jax.nn.sigmoid(randn(rng, (B, T, HV)))
    h0 = randn(rng, (B, HV, D, D))
    fused = Fused(KDA, rng, q, k, v, initial_state=h0)
    raw_g, g_ref, gate_kwargs = gate_inputs(g, use_gate_in_kernel,
                                            lower_bound, A_log, dt_bias)

    ref, ref_ht = reference(fused, g_ref, beta, h0)
    tri, tri_ht = fused.chunk(raw_g, beta, chunk_size=chunk_size,
                              beta_is_activated=True, **gate_kwargs)
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ("T", "P", "H", "D", "chunk_size", "use_gate_in_kernel"),
    param_grid(
        "T{}-P{}-H{}-D{}-chunk_size{}-use_gate{}",
        [(*test, use_gate_in_kernel)
         for use_gate_in_kernel in [False, True] for test in [
             (16, 6, 2, 64, 64),
             (32, 6, 2, 64, 32),
             (22, 21, 4, 128, 64),
             (40, 33, 2, 64, 32),
             (70, 65, 2, 64, 64),
         ]],
    ),
)
def test_chunk_causality(T: int, P: int, H: int, D: int, chunk_size: int,
                         use_gate_in_kernel: bool) -> None:
    """Future gates must leave prefix outputs unchanged."""
    rng = np.random.default_rng(42)
    shape = (1, T, H, D)
    q = normalize(randn(rng, shape))
    k = normalize(randn(rng, shape))
    v = 0.5 * randn(rng, shape)
    beta = jnp.ones(shape[:-1])
    g = bf16_values(-0.7 * (0.75 + 0.25 * rand(rng, shape)))
    A_log = jnp.log(uniform(rng, (H, ), 1.0, 16.0))
    dt_bias = 0.1 * rand(rng, (H * D, )) - 0.05
    fused = Fused(KDA, rng, q, k, v)

    if use_gate_in_kernel:
        gate_logit = math.log(0.99 / 0.01)
        g_edited = g.at[:, P:].set(
            bf16_values(gate_logit / jnp.exp(A_log).reshape(1, 1, H, 1) -
                        dt_bias.reshape(1, 1, H, D)))
    else:
        g_edited = g.at[:, P:].set(-4.95)
    assert not np.array_equal(g[:, P:], g_edited[:, P:])

    outputs = []
    for g_i in (g, g_edited):
        raw_g, _, gate_kwargs = gate_inputs(g_i, use_gate_in_kernel, -5.0,
                                            A_log, dt_bias)
        o, _ = fused.chunk(raw_g, beta, chunk_size=chunk_size,
                           beta_is_activated=True, **gate_kwargs)
        outputs.append(np.asarray(o[:, :P]))

    ref, tri = outputs
    assert np.isfinite(ref).all() and np.isfinite(tri).all()
    assert np.array_equal(ref, tri), "future gates changed prefix outputs"


@pytest.mark.parametrize(
    ("lower_bound", "use_gate_in_kernel", "cu_seqlens", "gate_pattern"),
    [
        pytest.param(
            lower_bound,
            use_gate_in_kernel,
            cu_seqlens,
            gate_pattern,
            id=f"lower_bound{lower_bound}-use_gate{use_gate_in_kernel}"
            f"-cu_seqlens{cu_seqlens}-{gate_pattern}",
        ) for cu_seqlens in [None, [0, 1, 17, 38]]
        for use_gate_in_kernel in [False, True]
        for lower_bound in [-5.0, -2.0, -0.01]
        for gate_pattern in (["random", "lower_bound", "zero", "mixed"]
                             if not use_gate_in_kernel and lower_bound == -5.0
                             else ["random"])
    ],
)
def test_chunk_lower_bound(lower_bound: float, use_gate_in_kernel: bool,
                           cu_seqlens: list[int] | None,
                           gate_pattern: str) -> None:
    rng = np.random.default_rng(42)
    B, T, H, D = 1, 38, 2, 64
    shape = (B, T, H, D)
    q = normalize(randn(rng, shape))
    k = normalize(randn(rng, shape))
    v = randn(rng, shape)
    g = bf16_values(10 * randn(rng, shape))
    if not use_gate_in_kernel:
        g = lower_bound * jax.nn.sigmoid(g)
        if gate_pattern == "lower_bound":
            g = jnp.full_like(g, lower_bound)
        elif gate_pattern == "zero":
            g = jnp.zeros_like(g)
        elif gate_pattern == "mixed":
            g = g.at[..., ::2].set(lower_bound).at[..., 1::2].set(0.)
    beta = jax.nn.sigmoid(randn(rng, (B, T, H)))
    A_log = jnp.zeros((H, ))
    dt_bias = 0.1 * randn(rng, (H * D, ))
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    h0 = randn(rng, (N, H, D, D))
    fused = Fused(KDA, rng, q, k, v, cu_seqlens=cu_seqlens, initial_state=h0)
    raw_g, g_ref, gate_kwargs = gate_inputs(g, use_gate_in_kernel,
                                            lower_bound, A_log, dt_bias)

    ref, ref_ht = reference(fused, g_ref, beta, h0)
    tri, tri_ht = fused.chunk(raw_g, beta, beta_is_activated=True,
                              **gate_kwargs)
    for name, ref_value, tri_value in (("o", ref, tri), ("ht", ref_ht,
                                                         tri_ht)):
        assert np.isfinite(np.asarray(tri_value)).all(), name
        assert_close(name, ref_value, tri_value, 0.005)


def varlen_inputs(
    rng: np.random.Generator, H: int, D: int, mask_p: float,
    cu_seqlens: list[int], use_gate_in_kernel: bool, safe_gate: bool
) -> tuple[Fused, jax.Array, jax.Array, jax.Array, dict, jax.Array]:
    """The varlen chunk inputs: the Fused run, the raw gate, the reference's
    log decay, beta, the gate kwargs and h0."""
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = randn(rng, (1, T, H, D))
    k = normalize(randn(rng, (1, T, H, D)))
    v = randn(rng, (1, T, H, D))
    g = randn(rng, (1, T, H, D))
    if use_gate_in_kernel:
        A_log = jnp.log(uniform(rng, (1, 1, H, 1), 1, 16))
        dt_bias = randn(rng, (H * D, ))
    else:
        A_log = dt_bias = None
        g = logsigmoid(g)
        g = g * (rand(rng, g.shape) > mask_p)
    mask = rand(rng, g.shape) > mask_p
    g = jnp.where(mask, g, -1000.)
    lower_bound = None
    if safe_gate:
        assert use_gate_in_kernel is False
        g = jnp.clip(g, -5, 0)
        lower_bound = -5.0

    beta = jax.nn.sigmoid(rand(rng, (1, T, H)))
    h0 = randn(rng, (N, H, D, D))
    fused = Fused(KDA, rng, q, k, v, cu_seqlens=cu_seqlens, initial_state=h0)
    raw_g, g_ref, gate_kwargs = gate_inputs(g, use_gate_in_kernel,
                                            lower_bound, A_log, dt_bias)
    return fused, raw_g, g_ref, beta, gate_kwargs, h0


# Note (david): ratio bounds both o and ht. Rows above 0.005 carry the
# kernel's measured bf16 error plus margin; mask_p = 0.9 leaves most of g
# zero, the case decay damps least.
@pytest.mark.parametrize(
    ("H", "D", "mask_p", "cu_seqlens", "use_gate_in_kernel", "safe_gate",
     "chunk_size", "ratio"),
    param_grid(
        "H{}-D{}-mask_p{}-cu_seqlens{}-gate{}-safe_gate{}-chunk_size{}",
        [
            (4, 64, 0.9, [0, 256, 500, 1000], True, False, 64, 0.0065),
            (4, 128, 0.5, [0, 256, 500, 1000], False, False, 64, 0.005),
            (4, 256, 0, [0, 100, 300, 1200, 3000, 4096], False, True, 64,
             0.005),
        ],
    ),
)
def test_chunk_varlen(H: int, D: int, mask_p: float, cu_seqlens: list[int],
                      use_gate_in_kernel: bool, safe_gate: bool,
                      chunk_size: int, ratio: float) -> None:
    rng = np.random.default_rng(42)
    fused, raw_g, g_ref, beta, gate_kwargs, h0 = varlen_inputs(
        rng, H, D, mask_p, cu_seqlens, use_gate_in_kernel, safe_gate)

    tri, tri_ht = fused.chunk(raw_g, beta, chunk_size=chunk_size,
                              beta_is_activated=True, **gate_kwargs)
    ref, ref_ht = reference(fused, g_ref, beta, h0)
    assert_close("o", ref, tri, ratio)
    assert_close("ht", ref_ht, tri_ht, ratio)


# Note (david): ratio bounds both o and ht. Rows above 0.005 carry the
# kernel's measured bf16 error plus margin; mask_p = 0.9 leaves most of g
# zero, the case decay damps least.
@pytest.mark.parametrize(
    ("H", "D", "mask_p", "cu_seqlens", "use_gate_in_kernel", "safe_gate",
     "ratio"),
    param_grid(
        "H{}-D{}-mask_p{}-cu_seqlens{}-gate{}-safe_gate{}",
        [
            (4, 64, 0.9, [0, 256, 500, 1000], True, False, 0.0065),
            (4, 128, 0.5, [0, 256, 500, 1000], False, False, 0.005),
            (4, 256, 0, [0, 100, 300, 1200, 3000, 4096], False, True, 0.005),
        ],
    ),
)
def test_chunk_varlen_prefill(H: int, D: int, mask_p: float,
                              cu_seqlens: list[int],
                              use_gate_in_kernel: bool, safe_gate: bool,
                              ratio: float) -> None:
    rng = np.random.default_rng(42)
    fused, raw_g, g_ref, beta, gate_kwargs, h0 = varlen_inputs(
        rng, H, D, mask_p, cu_seqlens, use_gate_in_kernel, safe_gate)

    tri, tri_ht = fused.chunk(raw_g, beta, beta_is_activated=True,
                              **gate_kwargs)
    ref, ref_ht = reference(fused, g_ref, beta, h0)
    assert_close("o", ref, tri, ratio)
    assert_close("ht", ref_ht, tri_ht, ratio)
