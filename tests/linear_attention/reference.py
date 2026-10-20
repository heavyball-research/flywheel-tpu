"""Naive GDN and KDA references, their gates and the error check of the
linear attention tests. Tensors are [B, T, H, D]."""

import jax
import jax.numpy as jnp
import numpy as np

# Note (david): TPU's default f32 dot rounds its operands to bf16, which the
# reference must not.
HIGHEST = jax.lax.Precision.HIGHEST


def rand(rng: np.random.Generator, shape: tuple[int, ...]) -> jax.Array:
    """U[0, 1) samples."""
    return jnp.asarray(rng.random(shape, dtype=np.float32))


def randn(rng: np.random.Generator, shape: tuple[int, ...]) -> jax.Array:
    """Standard normal samples."""
    return jnp.asarray(rng.standard_normal(shape, dtype=np.float32))


def uniform(rng: np.random.Generator, shape: tuple[int, ...], low: float,
            high: float) -> jax.Array:
    """U[low, high) samples."""
    return jnp.asarray(rng.uniform(low, high, shape).astype(np.float32))


def logsigmoid(x: jax.Array) -> jax.Array:
    return jax.nn.log_sigmoid(x)


def normalize(x: jax.Array) -> jax.Array:
    """x over its L2 norm along the last axis, the norm clamped at 1e-12."""
    norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True))
    return x / jnp.maximum(norm, 1e-12)


def repeat_heads(x: jax.Array, num_groups: int) -> jax.Array:
    """Each head of [B, T, H, D] x repeated num_groups times in place."""
    return jnp.repeat(x, num_groups, axis=2)


def get_abs_err(x: jax.Array, y: jax.Array) -> float:
    diff = np.asarray(x, np.float64) - np.asarray(y, np.float64)
    return float(np.abs(diff).max())


def get_err_ratio(x: jax.Array, y: jax.Array) -> float:
    x = np.asarray(x, np.float64)
    err = np.sqrt(np.mean(np.square(x - np.asarray(y, np.float64))))
    base = np.sqrt(np.mean(np.square(x)))
    return float(err / (base + 1e-8))


def assert_close(prefix: str,
                 ref: jax.Array,
                 tri: jax.Array,
                 ratio: float,
                 err_atol: float = 1e-6) -> None:
    """tri's RMS error relative to ref's RMS stays below ratio, unless no
    element is further than err_atol off."""
    abs_atol = get_abs_err(ref, tri)
    error_rate = get_err_ratio(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {error_rate:.6f}"
    print(msg)
    if abs_atol <= err_atol:
        return
    assert not np.isnan(np.asarray(ref, np.float32)).any(), (
        f"{prefix}: NaN detected in ref")
    assert not np.isnan(np.asarray(tri, np.float32)).any(), (
        f"{prefix}: NaN detected in tri")
    assert error_rate < ratio, msg


@jax.jit
def _gated_delta_rule_scan(q: jax.Array, k: jax.Array, v: jax.Array,
                           beta: jax.Array, g: jax.Array,
                           h: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Time-major [T, B, H, ...] inputs, q already scaled."""

    def _step(h, token):
        b_q, b_k, b_v, b_beta, b_g = token
        h = h * jnp.exp(b_g)[..., None, None]
        b_v = b_v - jnp.einsum("bhk,bhkv->bhv", b_k, h, precision=HIGHEST)
        b_v = b_v * b_beta[..., None]
        h = h + b_k[..., :, None] * b_v[..., None, :]
        return h, jnp.einsum("bhk,bhkv->bhv", b_q, h, precision=HIGHEST)

    return jax.lax.scan(_step, h, (q, k, v, beta, g))


def naive_recurrent_gated_delta_rule(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    beta: jax.Array,
    g: jax.Array,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    """Gated delta rule, token by token, in f32.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        beta: [B, T, H]
        g: [B, T, H]
        scale: Defaults to K**-0.5.
        initial_state: [B, H, K, V]
        output_final_state: Whether to return the final state.

    Returns:
        o: [B, T, H, V]
        final_state: [B, H, K, V] if output_final_state else None
    """
    q, k, v, beta, g = (jnp.swapaxes(x, 0, 1).astype(jnp.float32)
                        for x in (q, k, v, beta, g))
    _, batch, heads, key_dim = k.shape
    if scale is None:
        scale = key_dim**-0.5
    h = jnp.zeros((batch, heads, key_dim, v.shape[-1]), jnp.float32)
    if initial_state is not None:
        h = initial_state.astype(jnp.float32)
    h, o = _gated_delta_rule_scan(q * scale, k, v, beta, g, h)
    return jnp.swapaxes(o, 0, 1), h if output_final_state else None


@jax.jit
def _kda_scan(q: jax.Array, k: jax.Array, v: jax.Array, g: jax.Array,
              beta: jax.Array, s: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Time-major [T, B, HV, ...] inputs, q already scaled."""

    def _step(s, token):
        q_i, k_i, v_i, g_i, b_i = token
        s = s * jnp.exp(g_i)[..., None]
        v_i = v_i - jnp.einsum("bhk,bhkv->bhv", k_i, s, precision=HIGHEST)
        s = s + jnp.einsum("bhk,bhv->bhkv", b_i[..., None] * k_i, v_i,
                           precision=HIGHEST)
        return s, jnp.einsum("bhk,bhkv->bhv", q_i, s, precision=HIGHEST)

    return jax.lax.scan(_step, s, (q, k, v, g, beta))


def naive_recurrent_kda(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    """KDA, token by token, in f32.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, HV, V], HV divisible by H.
        g: Per-channel log decay [B, T, HV, K].
        beta: [B, T, HV]
        scale: Defaults to K**-0.5.
        initial_state: [B, HV, K, V]
        output_final_state: Whether to return the final state.

    Returns:
        o: [B, T, HV, V]
        final_state: [B, HV, K, V] if output_final_state else None
    """
    batch, _, heads, key_dim = q.shape
    num_v_heads = v.shape[2]
    if scale is None:
        scale = key_dim**-0.5
    q = repeat_heads(q, num_v_heads // heads) * scale
    k = repeat_heads(k, num_v_heads // heads)
    q, k, v, g, beta = (jnp.swapaxes(x, 0, 1).astype(jnp.float32)
                        for x in (q, k, v, g, beta))
    s = jnp.zeros((batch, num_v_heads, key_dim, v.shape[-1]), jnp.float32)
    if initial_state is not None:
        s = s + initial_state
    s, o = _kda_scan(q, k, v, g, beta, s)
    return jnp.swapaxes(o, 0, 1), s if output_final_state else None


def naive_gdn_gate(g: jax.Array,
                   A_log: jax.Array,
                   dt_bias: jax.Array | None = None) -> jax.Array:
    """-exp(A_log) * softplus(g + dt_bias) for g [..., HV]."""
    g = g.astype(jnp.float32)
    if dt_bias is not None:
        g = g + dt_bias.astype(jnp.float32)
    return -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(g)


def naive_kda_gate(g: jax.Array,
                   A_log: jax.Array,
                   dt_bias: jax.Array | None = None) -> jax.Array:
    """-exp(A_log)[:, None] * softplus(g + dt_bias) for g [..., H, K]."""
    heads = g.shape[-2]
    g = g.astype(jnp.float32)
    if dt_bias is not None:
        g = g + dt_bias.reshape(heads, -1)
    return -jnp.exp(A_log.reshape(heads, 1).astype(
        jnp.float32)) * jax.nn.softplus(g)


def naive_kda_lowerbound_gate(g: jax.Array,
                              A_log: jax.Array | None = None,
                              dt_bias: jax.Array | None = None,
                              lower_bound: float = -5.0) -> jax.Array:
    """lower_bound * sigmoid(exp(A_log) * (g + dt_bias)) for g [..., H, K]."""
    heads = g.shape[-2]
    g = g.astype(jnp.float32)
    if dt_bias is not None:
        g = g + dt_bias.reshape(heads, -1)
    if A_log is not None:
        g = jnp.exp(A_log.reshape(heads, 1).astype(jnp.float32)) * g
    return lower_bound * jax.nn.sigmoid(g)
