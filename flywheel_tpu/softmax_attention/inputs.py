"""User arguments -> kernel inputs: shape and keyword checks, padding and
the Variant."""

import jax
import jax.numpy as jnp

from ..core.tiling import round_up
from ..core.tpu import NUM_LANES
from .config import PAIRED_HEAD_DIM
from .variant import Variant

# Note (david): seq is axis 1 of the token-major dense layout (batch, seq,
# heads * head_dim) and axis 2 of both head-major ones, (batch, heads, seq,
# head_dim) and heads-outer (heads, batch, seq, head_dim); tokens are axis 0 of
# the token-major packed layout (total, heads * head_dim) and axis 1 of the
# head-major one (heads, total, head_dim).
TOKEN_MAJOR_SEQ_AXIS = 1
HEAD_MAJOR_SEQ_AXIS = 2
TOKEN_MAJOR_PACKED_AXIS = 0
HEAD_MAJOR_PACKED_AXIS = 1
AXIS_NAMES = {
    "b": "batch",
    "h": "nheads",
    "s": "seqlen",
    "t": "total",
    "d": "headdim",
}
# Note (david): prepare_rotary returns (batch, tokens, head_dim) packed cos/sin
# planes.
ROTARY_TOKEN_AXIS = 1


def pad_seqlen(seqlen: int) -> int:
    return round_up(seqlen, NUM_LANES)


def pad_axis(operand: jax.Array, axis: int, num_pad: int) -> jax.Array:
    if num_pad == 0:
        return operand
    else:
        return jnp.pad(
            operand,
            [(0, num_pad if dim == axis else 0) for dim in range(operand.ndim)],
        )


def descale_scalar(
    descale: float | jax.Array | None, arg_name: str
) -> jax.Array:
    """One per-tensor KV descale as an f32 scalar, possibly traced; None
    is 1."""
    descale = jnp.asarray(1.0 if descale is None else descale, jnp.float32)
    if descale.size != 1:
        raise ValueError(
            f"{arg_name} must be one per-tensor value; got shape"
            f" {descale.shape}."
        )
    return descale.reshape(())


def check_sinks(sinks: jax.Array | None, nheads: int) -> jax.Array | None:
    """sinks as a (nheads,) float32 array, or None."""
    if sinks is None:
        return None
    sinks = jnp.asarray(sinks)
    if sinks.shape != (nheads,) or not jnp.issubdtype(
        sinks.dtype, jnp.floating
    ):
        raise ValueError(
            f"learnable_sink must be a ({nheads},) float array, one logit per"
            f" query head; got {sinks.shape} {sinks.dtype}."
        )
    return sinks.astype(jnp.float32)


def parse_token_major(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    head_dim: int | None,
    expected_rank: int,
) -> tuple[int, int, int, int]:
    if head_dim is None:
        raise ValueError(
            "token-major (fused head_dim) inputs require head_dim=; the fused"
            f" width alone cannot determine the head split (got {q.shape=})."
        )
    if type(head_dim) is not int:
        raise ValueError(
            f"head_dim must be a static python int; got {head_dim!r}."
        )
    if head_dim <= 0 or (head_dim % NUM_LANES and head_dim != PAIRED_HEAD_DIM):
        raise ValueError(
            "token-major inputs require head_dim % 128 == 0 or head_dim == 64"
            f" (paired); got {head_dim}."
        )
    if k.ndim != expected_rank or v.ndim != expected_rank:
        raise ValueError(
            f"q/k/v ranks must match; got {q.shape=}, {k.shape=}, {v.shape=}."
        )
    if q.shape[-1] % head_dim or k.shape[-1] % head_dim:
        raise ValueError(
            f"head_dim {head_dim} must divide q/k fused widths; got"
            f" {q.shape=}, {k.shape=}."
        )
    nheads = q.shape[-1] // head_dim
    nheads_k = k.shape[-1] // head_dim
    if v.shape[-1] % nheads_k:
        raise ValueError(
            f"v fused width {v.shape[-1]} must be a multiple of"
            f" nheads_k {nheads_k}."
        )
    head_dim_v = v.shape[-1] // nheads_k
    if head_dim_v % NUM_LANES and head_dim_v != PAIRED_HEAD_DIM:
        raise ValueError(
            "token-major requires head_dim_v % 128 == 0 or head_dim_v == 64"
            f" (paired); got {head_dim_v}."
        )
    # Note (david): paired d64 DMA groups span two heads, which needs MHA (a
    # GQA/MQA kv stream cannot be paired), equal q and v head dims and an even
    # head count.
    if head_dim % NUM_LANES or head_dim_v % NUM_LANES:
        if head_dim_v != head_dim:
            raise NotImplementedError(
                "token-major d64 requires head_dim_v == head_dim (no MLA); got"
                f" {head_dim} vs {head_dim_v}."
            )
        if nheads_k != nheads:
            raise NotImplementedError(
                "token-major d64 is MHA-only (GQA/MQA's kv stream cannot be"
                " paired); pass 4-D head-major inputs instead."
            )
        if nheads % 2:
            raise ValueError(
                f"token-major d64 requires an even head count; got {nheads}."
            )
    return nheads, nheads_k, head_dim, head_dim_v


def parse_qkv(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    head_dim: int | None,
    token_major: bool,
    expected_rank: int,
    axes: str,
) -> tuple[int, int, int, int]:
    """nheads, nheads_k, head_dim_qk, head_dim_v of q/k/v.

    axes names the head-major layout's axes ("bhsd", "hbsd" or "thd": batch,
    heads, seqlen, total tokens, head_dim); token-major inputs fuse the heads
    into the last axis instead.
    """
    if token_major:
        if q.ndim != expected_rank:
            raise ValueError(
                f"token_major=True expects {expected_rank}-D (..., seqlen,"
                f" nheads * head_dim) q/k/v; got {q.shape=}. Pass"
                f" token_major=False for {expected_rank + 1}-D inputs."
            )
        nheads, nheads_k, head_dim_qk, head_dim_v = parse_token_major(
            q, k, v, head_dim, expected_rank
        )
        batch_axes = range(expected_rank - 2)
    else:
        if head_dim is not None:
            raise ValueError(
                "head_dim is only meaningful for token-major (..., seqlen,"
                f" nheads * head_dim) inputs; {expected_rank}-D inputs carry"
                " the split in their shape."
            )
        if any(operand.ndim != expected_rank for operand in (q, k, v)):
            layout = ", ".join(AXIS_NAMES[name] for name in axes)
            raise ValueError(
                f"q/k/v must be {expected_rank}-D ({layout}); got {q.shape=},"
                f" {k.shape=}, {v.shape=}."
            )
        if q.shape[-1] != k.shape[-1]:
            raise ValueError(f"Incompatible shapes: {q.shape=} vs {k.shape=}.")
        head_axis = axes.index("h")
        nheads, head_dim_qk = q.shape[head_axis], q.shape[-1]
        nheads_k, head_dim_v = k.shape[head_axis], v.shape[-1]
        batch_axes = [axis for axis, name in enumerate(axes) if name == "b"]
    if any(q.shape[axis] != k.shape[axis] for axis in batch_axes):
        raise ValueError(f"Incompatible shapes: {q.shape=} vs {k.shape=}.")
    # Note (david): v may differ from k in head_dim only (MLA-style), since the
    # kernel sizes the qk and v head dims independently.
    if k.shape[:-1] != v.shape[:-1]:
        raise ValueError(
            f"k and v must share every axis but head_dim; got {k.shape=},"
            f" {v.shape=}."
        )
    if nheads % nheads_k:
        raise ValueError(f"{nheads=} must be a multiple of {nheads_k=}.")
    if not q.dtype == k.dtype == v.dtype == jnp.bfloat16:
        raise NotImplementedError(
            f"Only bfloat16 is supported on TPU (v0); got {q.dtype=},"
            f" {k.dtype=}, {v.dtype=}."
        )
    return nheads, nheads_k, head_dim_qk, head_dim_v


def check_common_kwargs(
    window_size: tuple[int, int],
    dropout_p: float = 0.0,
    softcap: float = 0.0,
    alibi_slopes: jax.Array | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> tuple[int, int]:
    if dropout_p != 0.0:
        raise NotImplementedError("dropout_p is not supported on TPU (v0).")
    # Note (david): window_size comes back as a tuple so it can key the kernel
    # cache.
    window_size = tuple(window_size)
    if len(window_size) != 2 or any(
        bound != -1 and bound < 0 for bound in window_size
    ):
        raise ValueError(
            "window_size must be a 2-tuple of -1 (unbounded) or non-negative"
            f" ints; got {window_size}."
        )
    if softcap < 0.0:
        raise ValueError(f"softcap must be non-negative; got {softcap}.")
    if alibi_slopes is not None:
        raise NotImplementedError("alibi_slopes is not supported on TPU (v0).")
    if deterministic:
        raise NotImplementedError(
            "deterministic is not supported on TPU (v0); the forward kernel is"
            " deterministic already."
        )
    if return_attn_probs:
        raise NotImplementedError(
            "return_attn_probs is not supported on TPU (v0)."
        )
    return window_size


def make_variant(
    softmax_scale: float,
    *,
    softcap: float,
    causal: bool,
    window_size: tuple[int, int],
    learnable_sink: jax.Array | None,
    return_lse: bool,
    causal_offset: int = 0,
    runtime_causal_offset: bool = False,
    has_kv_scales: bool = False,
) -> Variant:
    """The Variant of one call from its user kwargs: softmax_scale resolved,
    window_size as check_common_kwargs returns it, and learnable_sink, which
    stays a kernel operand, only marking that the build takes sinks.

    softmax_scale may be any concrete scalar, a 0-d array included; it is
    taken as a Python float, since the Variant keys the kernel cache.
    """
    return Variant(
        softmax_scale=float(softmax_scale),
        softcap=softcap,
        causal=causal,
        window_size=window_size,
        causal_offset=causal_offset,
        runtime_causal_offset=runtime_causal_offset,
        has_sinks=learnable_sink is not None,
        has_kv_scales=has_kv_scales,
        return_lse=return_lse,
    )
