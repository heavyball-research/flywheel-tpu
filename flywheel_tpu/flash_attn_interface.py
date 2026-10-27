"""Flash-attention entry points served by the TPU Pallas kernels.

Arguments the kernels do not implement are accepted but raise
NotImplementedError on any non-default value.
"""

import functools
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp

from .pallas.block_sizes import (
    FWD_BLOCKS,
    FWD_KV_COMPUTE_BLOCKS,
    FWD_Q_COMPUTE_BLOCKS,
    LOG2E,
    NUM_LANES,
    PAIRED_HEAD_DIM,
    BlockSizes,
    QKVLayout,
    TokenMajorInfo,
    default_block,
    next_pow2,
    pick_tile,
    round_up,
)
from .pallas.copy_utils import resolve_merged_kv_heads
from .pallas.flash_bwd import flash_attn_bwd
from .pallas.flash_fwd import MIN_Q_TILES_PER_BLOCK, make_flash_attn_mha
from .pallas.flash_fwd_kvcache import flash_attn_kvcache_pallas
from .pallas.flash_fwd_kvcache_varlen import (
    append_ragged,
    flash_attn_kvcache_varlen,
)
from .pallas.flash_fwd_varlen_paged import flash_attn_varlen_paged
from .pallas.rotary import prepare_rotary
from .tuned_block_sizes import get_tuned_config, get_varlen_head_fold

# Note (david): each page is its own DMA, so smaller pages would leave the
# decode kernel descriptor-bound.
PAGE_SIZE_MULTIPLE = 128
# Note (david): the paged varlen kernel pads every request to a 128-token q
# block, so a short window folded into the decode kernel's query heads reads
# the cache once instead; the decode kernel keeps all of a row's query heads
# in VMEM, which caps the fold at this many rows per KV head.
MAX_FOLDED_QUERY_ROWS = 128
# Note (david): a packed (or right-only) window caps both physical forward
# blocks. A q block row loads the window plus bq + bkv keys of partial blocks,
# all through the masked fragment pipeline, so large blocks waste about
# (bq + bkv) / W, and small ones pay a pipeline step per few fragments. On v6e
# varlen 16K prefill (2 x 8K, 16:8 heads of 256, 32:8 of 128, W 1024 and
# 4096), 1024 beat 512 by 1.25-1.64x and 2048 by up to 1.44x, except at W 4096,
# where 2048 was 2-7% faster.
WINDOW_BLOCK = 1024
# Note (david): a dense window with a left bound runs fwd_pipeline's static
# band copies, which skip and mask per compute tile, so its blocks stay at the
# full-attention size. On v6e dense 16K causal prefill (16:8 heads of 256,
# 32:8 of 128, W 128-4096), 2048 blocks with 256-key kv tiles beat 512-key ones
# by 7-26%, while 128-key tiles fell off a code-size cliff (5.5-12.7 ms at
# W 512). A d256 window wider than a block pair runs the 64-fragment mask-free
# copy, which ran 9.63 ms at 256-key tiles (W 4096) against 1.98 at 512.
BAND_KV_COMPUTE_BLOCKS = (256, 128)
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


def default_block_sizes(
    q_seqlen: int,
    kv_seqlen: int,
    max_seqlen: int | None = None,
    max_seqlen_q: int | None = None,
    *,
    window_size: tuple[int, int] = (-1, -1),
    head_dim: int = NUM_LANES,
) -> BlockSizes:
    # Note (david): the kernel masks in absolute coordinates, so block_q and
    # block_kv come from their own axes. Chunked-prefill q chunks are far shorter
    # than their kv contexts, and a q block sized to the kv cap straddles several
    # sequences and leaves the block-diagonal schedule little to skip, so
    # max_seqlen_q caps the q axis on its own.
    seq_q_cap = max_seqlen if max_seqlen_q is None else max_seqlen_q
    is_dense_band = max_seqlen is None and window_size[0] != -1
    # Note (david): only the dense grid needs blocks that divide both axes. The
    # per-seq schedule clamps every window to the padded end, so a packed total
    # of 128 times an odd count keeps the max_seqlen-capped block, not 128 rows.
    must_divide = max_seqlen is None
    if is_dense_band:
        # Note (david): the band copies need bq % bkv == 0.
        block_q = default_block(q_seqlen, FWD_BLOCKS)
        block_kv = default_block(kv_seqlen, FWD_BLOCKS, block_q)
    elif window_size != (-1, -1):
        block_q = default_block(
            q_seqlen,
            FWD_BLOCKS,
            WINDOW_BLOCK if seq_q_cap is None else min(seq_q_cap, WINDOW_BLOCK),
            must_divide=must_divide,
        )
        block_kv = default_block(
            kv_seqlen,
            FWD_BLOCKS,
            WINDOW_BLOCK
            if max_seqlen is None
            else min(max_seqlen, WINDOW_BLOCK),
            must_divide=must_divide,
        )
    else:
        block_q = default_block(
            q_seqlen, FWD_BLOCKS, seq_q_cap, must_divide=must_divide
        )
        block_kv = default_block(
            kv_seqlen, FWD_BLOCKS, max_seqlen, must_divide=must_divide
        )
    q_compute_tile = pick_tile(FWD_Q_COMPUTE_BLOCKS, block_q)
    if block_q // q_compute_tile < MIN_Q_TILES_PER_BLOCK:
        block_q_compute = q_compute_tile // MIN_Q_TILES_PER_BLOCK
    else:
        block_q_compute = q_compute_tile
    if is_dense_band and (
        head_dim <= NUM_LANES or window_size[0] < block_q + block_kv - 1
    ):
        kv_compute_blocks = BAND_KV_COMPUTE_BLOCKS
    else:
        kv_compute_blocks = FWD_KV_COMPUTE_BLOCKS
    return BlockSizes(
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=pick_tile(kv_compute_blocks, block_kv),
        block_q_compute=block_q_compute,
    )


@functools.cache
def get_kernel(
    num_heads: int,
    num_kv_heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    causal: bool,
    window_size: tuple[int, int],
    interpret: bool,
    softcap: float,
    head_dim: int,
    head_dim_v: int,
    causal_offset: int,
    varlen_max_seqlen_kv: int | None = None,
    varlen_max_seqlen_q: int | None = None,
    token_major: bool = False,
    kernel_batch: int | None = None,
    q_scale: float = 1.0,
    blocks_override: tuple[int, int, int, int] | None = None,
    return_lse: bool = False,
    heads_outer_batch: int | None = None,
) -> functools.partial:
    # Note (david): seqlen_q and seqlen_kv are padded lengths while causal_offset
    # comes from the original ones. Different original pairs can pad to the same
    # multiples of 128, so a rectangular mask is detected by causal_offset too.
    is_rectangular = seqlen_q != seqlen_kv or causal_offset != 0
    is_varlen = varlen_max_seqlen_kv is not None
    has_window = window_size != (-1, -1)
    # Note (david): a tuned config fixes the blocks and the head fold together,
    # so a hit replaces the analytic block sizes wholesale. Windows, softcap and
    # rectangular masks were never measured and never consult it.
    if (
        blocks_override is None
        and not has_window
        and softcap == 0.0
        and not is_rectangular
    ):
        tuned_config = get_tuned_config(
            "fwd",
            token_major=token_major,
            num_heads=num_heads,
            batch=kernel_batch,
            seq_len=seqlen_q,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            causal=causal,
            max_seqlen=varlen_max_seqlen_kv,
            return_lse=return_lse,
            num_kv_heads=num_kv_heads,
            heads_outer_batch=heads_outer_batch,
        )
    else:
        tuned_config = None
    if tuned_config is not None:
        (
            block_q,
            block_kv,
            block_q_compute,
            block_kv_compute,
            num_stages,
            head_fold,
            qkv_layout_name,
            tuned_transposed_pv,
        ) = tuned_config
        # Note (david): the per-sequence varlen schedule DMAs at arbitrary token
        # offsets, which needs the token axis second-minor, so it overrides a tuned
        # SEQ_MINOR layout and transposed_pv.
        if is_varlen:
            qkv_layout = QKVLayout.HEAD_DIM_MINOR
            transposed_pv = False
        else:
            qkv_layout = QKVLayout[qkv_layout_name.upper()]
            transposed_pv = tuned_transposed_pv
        block_sizes = BlockSizes(
            block_q=block_q,
            block_kv=block_kv,
            block_kv_compute=block_kv_compute,
            block_q_compute=block_q_compute,
            num_stages=num_stages,
            qkv_layout=qkv_layout,
        )
    else:
        transposed_pv = False
        if blocks_override is None:
            block_sizes = default_block_sizes(
                seqlen_q,
                seqlen_kv,
                varlen_max_seqlen_kv,
                varlen_max_seqlen_q,
                window_size=window_size,
                head_dim=head_dim,
            )
        else:
            # Note (david): the caller already padded both axes to multiples of these
            # blocks, and the kernel build rejects infeasible tiles.
            block_q, block_kv, block_q_compute, block_kv_compute = (
                blocks_override
            )
            block_sizes = BlockSizes(
                block_q=block_q,
                block_kv=block_kv,
                block_kv_compute=block_kv_compute,
                block_q_compute=block_q_compute,
            )
        # Note (david): only varlen has a fold table of its own, so a dense build
        # that missed the tuned configs runs unfolded. The kernel folds varlen only
        # on square physical blocks, which chunked prefill breaks while keeping the
        # padded axes equal. The table was measured head-major but serves
        # token-major too: the fold amortizes per-head step costs the layout does
        # not change, and halving it until it divides the per-row num_heads keeps a
        # token-major group inside one batch row.
        if (
            not is_varlen
            or num_heads != num_kv_heads
            or has_window
            or is_rectangular
            or block_sizes.block_q != block_sizes.block_kv
        ):
            head_fold = 1
        else:
            head_fold = get_varlen_head_fold(
                (head_dim, block_sizes.block_q, causal),
                num_heads,
                head_dim,
                head_dim_v,
            )
    # Note (david): token-major d64 pairs heads, so the fold is the DMA alignment
    # unit and an odd fold is bumped to the minimal pair.
    if token_major and head_dim % NUM_LANES and head_fold % 2:
        kernel_head_fold = 2
    else:
        kernel_head_fold = head_fold
    if has_window:
        window_left = None if window_size[0] == -1 else window_size[0]
        # Note (david): window semantics: causal forces the right bound to 0, and
        # -1 means unbounded.
        if causal:
            window_right = 0
        elif window_size[1] == -1:
            window_right = None
        else:
            window_right = window_size[1]
        window = (window_left, window_right)
    else:
        window = None
    if token_major:
        token_major_info = TokenMajorInfo(
            batch=kernel_batch,
            num_q_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_v=head_dim_v,
        )
    else:
        token_major_info = None
    return make_flash_attn_mha(
        num_heads,
        seqlen_q,
        seqlen_kv,
        causal=causal,
        causal_offset=causal_offset,
        window=window,
        block_sizes=block_sizes,
        num_kv_heads=num_kv_heads,
        interpret=interpret,
        softcap=softcap,
        return_lse=return_lse,
        head_fold=kernel_head_fold,
        transposed_pv=transposed_pv,
        varlen_max_seqlen_kv=varlen_max_seqlen_kv,
        token_major=token_major_info,
        q_scale=q_scale,
        heads_outer_batch=heads_outer_batch,
    )


def pad_axis(operand: jax.Array, axis: int, num_pad: int) -> jax.Array:
    if num_pad == 0:
        return operand
    else:
        return jnp.pad(
            operand,
            [(0, num_pad if dim == axis else 0) for dim in range(operand.ndim)],
        )


def fused_q_scale(softmax_scale: float, softcap: float) -> float:
    # Note (david): the kernel softmax runs in log2 units, so log2(e) rides in
    # the fused Q scale. With softcap, Q takes softmax_scale / softcap instead
    # and the kernel applies log2(e) after its tanh.
    if softcap == 0.0:
        return softmax_scale * LOG2E
    else:
        return softmax_scale / softcap


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


def attn_with_vjp(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    build_kernel: Callable[..., functools.partial],
    softmax_scale: float,
    causal: bool,
    causal_offset: int,
    interpret: bool,
    bwd_unsupported: str,
    token_major: bool,
    head_dim: int | None,
) -> jax.Array:
    # Note (david): the forward kernel fuses q_scale into Q, so the backward
    # differentiates the raw q/k/v against the true softmax_scale. Only the
    # custom_vjp forward asks the kernel for lse, so the primal skips its compute
    # and DMA.
    @jax.custom_vjp
    def _attn(q, k, v):
        return build_kernel(return_lse=False)(q, k, v)

    def _attn_fwd(q, k, v):
        out, lse = build_kernel(return_lse=True)(q, k, v)
        return out, (q, k, v, out, lse)

    def _attn_bwd(residuals, dout):
        # Note (david): raise only when a gradient is requested, so forward-only
        # configurations still run their forward.
        if bwd_unsupported:
            raise NotImplementedError(
                f"The TPU backward kernel does not support {bwd_unsupported};"
                " that configuration is forward-only."
            )
        return flash_attn_bwd(
            *residuals,
            dout,
            causal=causal,
            causal_offset=causal_offset,
            softmax_scale=softmax_scale,
            interpret=interpret,
            token_major=token_major,
            head_dim=head_dim,
        )

    _attn.defvjp(_attn_fwd, _attn_bwd)
    return _attn(q, k, v)


def forward_only_attn(
    kernel: Callable[..., jax.Array | tuple[jax.Array, jax.Array]],
    reason: str,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array | None,
    kernel_args: tuple[jax.Array, ...],
    rotary: tuple[jax.Array, jax.Array | None] | None,
    rotary_interleaved: bool,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    # Note (david): a bare pallas_call has no differentiation rule, so jax.grad
    # would fail deep inside the tracer instead of naming the unsupported
    # feature. The rotary coefficients are primal inputs so that table gradients
    # raise as well.
    @jax.custom_vjp
    def _forward(q, k, v, rotary):
        if rotary is None:
            return kernel(q, k, v, *kernel_args)
        else:
            return kernel(
                q,
                k,
                v,
                *kernel_args,
                rotary=rotary,
                rotary_interleaved=rotary_interleaved,
            )

    def _raise_unsupported(*unused_args):
        raise NotImplementedError(
            f"{reason} is forward-only; backward is unsupported."
        )

    _forward.defvjp(_raise_unsupported, _raise_unsupported)
    return _forward(q, k, v, rotary)


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
    if not (q.dtype == k.dtype == v.dtype == jnp.bfloat16):
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


def paged_window(
    causal: bool, window_size: tuple[int, int]
) -> tuple[bool, int | None]:
    """(is_right_bounded, window_left) for the paged varlen kernel.

    Its schedule bounds a row's keys on the right only at the causal diagonal,
    so the right bound must be 0 or unbounded; causal forces 0, as in
    flash_attn_func.
    """
    if causal or window_size[1] == 0:
        is_right_bounded = True
    elif window_size[1] == -1:
        is_right_bounded = False
    else:
        raise NotImplementedError(
            "a paged KV cache takes a window right bound of -1 (unbounded) or 0"
            f" (causal) only; got window_size={window_size}."
        )
    window_left = None if window_size[0] == -1 else window_size[0]
    return is_right_bounded, window_left


def flash_attn_func(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    attention_chunk: int | None = None,
    softcap: float = 0.0,
    alibi_slopes: jax.Array | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    return_softmax_lse: bool = False,
    interpret: bool = False,
    head_dim: int | None = None,
    token_major: bool = False,
    *,
    rotary_cos: jax.Array | None = None,
    rotary_sin: jax.Array | None = None,
    rotary_interleaved: bool = False,
    rotary_k: bool = True,
    heads_outer_fold: bool = False,
    learnable_sink: jax.Array | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """flash_attn.flash_attn_func on TPU, JAX arrays in and out.

    Head-major (default): q (batch, nheads, seqlen_q, headdim_qk), k (batch,
    nheads_k, seqlen_k, headdim_qk), v (batch, nheads_k, seqlen_k, headdim_v),
    out (batch, nheads, seqlen_q, headdim_v) -- the kernel's own head-major
    layout with the batch split off, so no operand is relaid out. Token-major
    (token_major=True) fuses the heads into the last axis, (batch, seqlen,
    nheads * head_dim), and head_dim gives the split: a multiple of 128, or 64
    for MHA with an even head count. nheads % nheads_k == 0, and all inputs are
    bfloat16.

    causal is bottom-right aligned on the original lengths: query i attends key
    j iff j <= i + seqlen_k - seqlen_q. seqlen_q may be any length; seqlen_k must
    be a multiple of 128 unless the mask's right bound is 0. Rows with no
    reachable key return out = 0 and lse = -inf. return_softmax_lse adds lse
    (batch, nheads, seqlen_q) float32, the natural-log logsumexp of the scaled,
    masked scores. softcap > 0 applies
    tanh(softmax_scale * q @ k^T / softcap) * softcap.

    rotary_cos / rotary_sin: fused full-head RoPE tables (seqlen_ro,
    headdim_qk / 2) in q's dtype or float32, headdim_qk divisible by 16. K is
    rotated at positions 0..seqlen_k - 1 and Q from seqlen_k - seqlen_q on, so
    seqlen_q <= seqlen_k. rotary_interleaved pairs adjacent dimensions instead
    of the two halves; rotary_k=False rotates Q only.

    heads_outer_fold (head-major only): q/k/v and out are heads-outer, (nheads,
    batch, seqlen, headdim), again with no relayout; supports Q-only RoPE and no
    lse. attention_chunk is not implemented. interpret runs the Pallas TPU
    interpreter.

    learnable_sink: None or (nheads,) float sink logits, on the scale of the
    softcapped scores (gpt-oss attention sinks, FlashAttention's
    learnable_sink): exp(learnable_sink[h]) joins head h's softmax denominator
    with no value row, and lse includes it. A row with no reachable key then
    returns out = 0 and lse = learnable_sink[h]. Each logit must be finite or
    -inf (no sink); +inf and NaN are not checked and give undefined results.

    Gradients cover MHA and GQA/MQA with causal or full masks; window_size,
    softcap, RoPE, heads_outer_fold and learnable_sink are forward-only.
    """
    window_size = check_common_kwargs(
        window_size,
        dropout_p=dropout_p,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )
    if attention_chunk is not None:
        raise NotImplementedError("attention_chunk is not implemented on TPU.")
    if type(heads_outer_fold) is not bool:
        raise TypeError("heads_outer_fold must be a bool.")
    has_rotary_tables = rotary_cos is not None or rotary_sin is not None
    if heads_outer_fold and (
        token_major or return_softmax_lse or (has_rotary_tables and rotary_k)
    ):
        raise NotImplementedError(
            "heads_outer_fold is a forward-only head-major layout; it supports"
            " fused RoPE only as Q-only rotation (rotary_k=False) and no"
            " return_softmax_lse."
        )
    expected_rank = 3 if token_major else 4
    nheads, nheads_k, head_dim_qk, head_dim_v = parse_qkv(
        q,
        k,
        v,
        head_dim,
        token_major,
        expected_rank,
        "hbsd" if heads_outer_fold else "bhsd",
    )
    if token_major:
        seq_axis = TOKEN_MAJOR_SEQ_AXIS
        batch = q.shape[0]
    else:
        seq_axis = HEAD_MAJOR_SEQ_AXIS
        batch = q.shape[1] if heads_outer_fold else q.shape[0]
    seqlen_q, seqlen_k = q.shape[seq_axis], k.shape[seq_axis]
    rotary = prepare_rotary(
        rotary_cos,
        rotary_sin,
        head_dim=head_dim_qk,
        dtype=q.dtype,
        rotate_k=rotary_k,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        interleaved=rotary_interleaved,
    )

    padded_seqlen_q = pad_seqlen(seqlen_q)
    padded_seqlen_k = pad_seqlen(seqlen_k)
    num_pad_q = padded_seqlen_q - seqlen_q
    num_pad_kv = padded_seqlen_k - seqlen_k
    # Note (david): a right bound of 0 keeps every real query off the padded kv
    # tail, so seqlen_k pads for free; without it the dense kernel cannot hide
    # that tail. Padded q rows need no mask: they are sliced off, and the zero
    # cotangent of that slice cancels their k/v gradient contributions.
    is_right_bounded = causal or window_size[1] == 0
    if num_pad_kv and not is_right_bounded:
        raise ValueError(
            f"flash_attn_func is dense-only: seqlen_k={seqlen_k} is not a"
            " multiple of 128 and the mask has no right bound at 0, so the"
            " padded kv tail would be attended. Align seqlen_k to a multiple of"
            " 128, or use flash_attn_varlen_func."
        )

    q = pad_axis(q, seq_axis, num_pad_q)
    k, v = (pad_axis(operand, seq_axis, num_pad_kv) for operand in (k, v))
    if rotary is None:
        rotary_coeffs = None
    else:
        q_coeff, k_coeff = rotary
        rotary_coeffs = (
            pad_axis(q_coeff, ROTARY_TOKEN_AXIS, num_pad_q),
            None
            if k_coeff is None
            else pad_axis(k_coeff, ROTARY_TOKEN_AXIS, num_pad_kv),
        )

    softmax_scale = (
        1.0 / math.sqrt(head_dim_qk) if softmax_scale is None else softmax_scale
    )
    # Note (david): bottom-right alignment anchors the causal diagonal on the
    # original lengths, not the padded ones.
    causal_offset = seqlen_k - seqlen_q
    # Note (david): token-major kernels are built from per-row heads plus the
    # batch, while head-major folds the batch into the head axis.
    if token_major:
        kernel_num_heads, kernel_num_kv_heads = nheads, nheads_k
        token_major_batch = batch
    else:
        kernel_num_heads = batch * nheads
        kernel_num_kv_heads = batch * nheads_k
        token_major_batch = None
    build_kernel = functools.partial(
        get_kernel,
        num_heads=kernel_num_heads,
        num_kv_heads=kernel_num_kv_heads,
        seqlen_q=padded_seqlen_q,
        seqlen_kv=padded_seqlen_k,
        causal=bool(causal),
        window_size=window_size,
        interpret=bool(interpret),
        softcap=float(softcap),
        head_dim=head_dim_qk,
        head_dim_v=head_dim_v,
        causal_offset=causal_offset,
        token_major=bool(token_major),
        kernel_batch=token_major_batch,
        q_scale=float(fused_q_scale(softmax_scale, softcap)),
    )
    # Note (david): token-major kernels address heads as lane-axis column windows
    # of the fused refs, and either head-major layout is the kernel's leading
    # axis split in two, so no operand is relaid out.
    if token_major:
        kernel_q, kernel_k, kernel_v = q, k, v
    else:
        kernel_q, kernel_k, kernel_v = (
            operand.reshape(-1, *operand.shape[2:]) for operand in (q, k, v)
        )
    sinks = check_sinks(learnable_sink, nheads)
    sink_args = () if sinks is None else (sinks,)
    if heads_outer_fold:
        # Note (david): flash_attn_bwd has no heads-outer kv map, so this layout
        # is forward-only. Q-only RoPE stays valid: the dense coefficients carry a
        # singleton batch shared by every kernel row, and each heads-outer row is
        # one sequence at positions 0..seqlen-1.
        reason = "heads_outer_fold" if rotary_coeffs is None else "Fused RoPE"
        kernel_output = forward_only_attn(
            build_kernel(return_lse=False, heads_outer_batch=batch),
            reason,
            kernel_q,
            kernel_k,
            kernel_v,
            sink_args,
            rotary_coeffs,
            rotary_interleaved,
        )
    elif rotary_coeffs is not None or sinks is not None:
        reason = "learnable_sink" if rotary_coeffs is None else "Fused RoPE"
        kernel_output = forward_only_attn(
            build_kernel(return_lse=bool(return_softmax_lse)),
            reason,
            kernel_q,
            kernel_k,
            kernel_v,
            sink_args,
            rotary_coeffs,
            rotary_interleaved,
        )
    elif return_softmax_lse:
        # Note (david): lse has no gradient of its own, so this path stays the
        # bare kernel call.
        kernel_output = build_kernel(return_lse=True)(
            kernel_q, kernel_k, kernel_v
        )
    else:
        if window_size != (-1, -1):
            bwd_unsupported = "window_size"
        elif softcap != 0.0:
            bwd_unsupported = "softcap"
        else:
            bwd_unsupported = ""
        kernel_output = attn_with_vjp(
            kernel_q,
            kernel_k,
            kernel_v,
            build_kernel=build_kernel,
            softmax_scale=softmax_scale,
            causal=bool(causal),
            causal_offset=causal_offset,
            interpret=bool(interpret),
            bwd_unsupported=bwd_unsupported,
            token_major=token_major,
            head_dim=head_dim_qk if token_major else None,
        )
    if return_softmax_lse:
        kernel_out, kernel_lse = kernel_output
    else:
        kernel_out = kernel_output

    if token_major:
        out_padded = kernel_out
    else:
        out_padded = kernel_out.reshape(
            *q.shape[:2], padded_seqlen_q, head_dim_v
        )
    out_sliced = jax.lax.slice_in_dim(out_padded, 0, seqlen_q, axis=seq_axis)
    # Note (david): a right bound leaves the first -causal_offset - right rows
    # with no reachable key. The kernel skips their blocks, so its output there
    # is unspecified; match flash_attn with out = 0 and lse = -inf.
    if causal:
        num_masked_rows = -causal_offset
    elif window_size[1] != -1:
        num_masked_rows = -causal_offset - window_size[1]
    else:
        num_masked_rows = 0
    if num_masked_rows > 0:
        is_valid_row = jnp.arange(seqlen_q) >= num_masked_rows
        row_broadcast_index = tuple(
            slice(None) if axis == seq_axis else None
            for axis in range(out_sliced.ndim)
        )
        out = jnp.where(is_valid_row[row_broadcast_index], out_sliced, 0)
    else:
        out = out_sliced
    if return_softmax_lse:
        # Note (david): the kernel lse is (batch * nheads, padded_seqlen_q)
        # batch-major in both layouts.
        lse = kernel_lse.reshape(batch, nheads, padded_seqlen_q)[
            :, :, :seqlen_q
        ]
        if num_masked_rows > 0:
            # Note (deepolar): a masked row's only softmax term is its sink.
            empty_lse = -jnp.inf if sinks is None else sinks[None, :, None]
            return out, jnp.where(is_valid_row[None, None, :], lse, empty_lse)
        else:
            return out, lse
    else:
        return out


def _flash_attn_func_pcp_causal_offset(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    causal_offset: jax.Array | int,
    *,
    softmax_scale: float | None = None,
    return_softmax_lse: bool = False,
    heads_outer_fold: bool = False,
    interpret: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Internal dense PCP forward with a device-side Q-position offset.

    Q/K/V are flash_attn_func's head-major (1, nheads, seqlen, headdim), or
    (nheads, 1, seqlen, headdim) with heads_outer_fold. The PCP caller
    guarantees 0 <= offset and offset + q_len <= kv_len before padding. This is
    not a general arbitrary-offset attention API.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            "runtime-offset attention requires 4-D head-major Q/K/V."
        )
    batch_axis, head_axis = (1, 0) if heads_outer_fold else (0, 1)
    batch, num_heads = q.shape[batch_axis], q.shape[head_axis]
    q_len, head_dim = q.shape[HEAD_MAJOR_SEQ_AXIS], q.shape[3]
    if batch != 1 or k.shape[batch_axis] != 1 or v.shape[batch_axis] != 1:
        raise ValueError("runtime-offset attention requires batch_size == 1.")
    if k.shape != v.shape or k.shape[3] != head_dim:
        raise ValueError(
            "K/V must share shape and Q/K head dimensions must match."
        )
    num_kv_heads = k.shape[head_axis]
    kv_len = k.shape[HEAD_MAJOR_SEQ_AXIS]
    if num_heads % num_kv_heads:
        raise ValueError("Q heads must be divisible by KV heads.")
    if q_len <= 0 or kv_len < q_len:
        raise ValueError(
            "runtime-offset attention requires a positive Q length, a"
            " positive KV length, and kv_len >= q_len."
        )
    if any(operand.dtype != jnp.bfloat16 for operand in (q, k, v)):
        raise NotImplementedError(
            "runtime-offset attention supports bf16 only."
        )
    offset_array = jnp.asarray(causal_offset)
    if offset_array.size != 1 or not jnp.issubdtype(
        offset_array.dtype, jnp.integer
    ):
        raise ValueError("causal_offset must be one integer scalar.")
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    padded_q_len = pad_seqlen(q_len)
    padded_kv_len = pad_seqlen(kv_len)
    q = pad_axis(q, HEAD_MAJOR_SEQ_AXIS, padded_q_len - q_len)
    k = pad_axis(k, HEAD_MAJOR_SEQ_AXIS, padded_kv_len - kv_len)
    v = pad_axis(v, HEAD_MAJOR_SEQ_AXIS, padded_kv_len - kv_len)
    kernel = make_flash_attn_mha(
        num_heads,
        padded_q_len,
        padded_kv_len,
        causal=True,
        causal_offset=0,
        block_sizes=default_block_sizes(padded_q_len, padded_kv_len),
        num_kv_heads=num_kv_heads,
        interpret=interpret,
        return_lse=return_softmax_lse,
        q_scale=float(fused_q_scale(softmax_scale, 0.0)),
        heads_outer_batch=1 if heads_outer_fold else None,
        runtime_causal_offset=True,
    )
    kernel_output = kernel(
        *(operand.reshape(-1, *operand.shape[2:]) for operand in (q, k, v)),
        offset_array.reshape(()),
    )
    if return_softmax_lse:
        kernel_out, kernel_lse = kernel_output
    else:
        kernel_out = kernel_output
    out = kernel_out.reshape(*q.shape[:2], padded_q_len, head_dim)[:, :, :q_len]
    if return_softmax_lse:
        return (
            out,
            kernel_lse.reshape(batch, num_heads, padded_q_len)[:, :, :q_len],
        )
    else:
        return out


def paged_varlen_attn(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array | None,
    cu_seqlens_q: jax.Array,
    cu_seqlens_k: jax.Array | None,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    softmax_scale: float | None,
    causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    return_softmax_lse: bool,
    interpret: bool,
    head_dim: int | None,
    token_major: bool,
    blocks_override: tuple[int, int, int, int] | None,
    rotary_cos: jax.Array | None,
    rotary_sin: jax.Array | None,
    rotary_interleaved: bool,
    rotary_k: bool,
    block_table: jax.Array,
    seqused_k: jax.Array | None,
    num_active: jax.Array | int | None,
    k_descale: float | jax.Array | None,
    v_descale: float | jax.Array | None,
    nheads_k: int | None,
    sinks: jax.Array | None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """flash_attn_varlen_func's paged KV cache branch (block_table given)."""
    if token_major:
        raise NotImplementedError(
            "a paged KV cache takes head-major q (nheads, total_q, headdim);"
            " token_major=True is not supported."
        )
    if head_dim is not None:
        raise ValueError(
            "head_dim is only meaningful for token-major inputs; a paged call"
            " takes head-major q."
        )
    is_right_bounded, window_left = paged_window(causal, window_size)
    if cu_seqlens_k is not None:
        raise ValueError(
            "with block_table, pass cu_seqlens_k=None: seqused_k gives each"
            " sequence's kv length."
        )
    if v is not None:
        raise ValueError(
            "with block_table, pass one merged (num_pages, page_size,"
            " cache_heads, headdim) cache as k and v=None; separate paged K and"
            " V caches are not supported."
        )
    if seqused_k is None:
        raise ValueError(
            "block_table needs seqused_k, the (batch,) kv length of each"
            " sequence (its new tokens included)."
        )
    if rotary_cos is not None and rotary_k:
        raise NotImplementedError(
            "a paged KV cache holds K already rotated; pass rotary_k=False to"
            " rotate Q only."
        )
    for arg_name, seqlen_bound in (
        ("max_seqlen_q", max_seqlen_q),
        ("max_seqlen_k", max_seqlen_k),
    ):
        if isinstance(seqlen_bound, jax.core.Tracer):
            raise ValueError(
                f"{arg_name} must be a concrete int (it picks the kernel's"
                " block sizes); pass a static bucket bound, not a traced"
                " value."
            )
        if int(seqlen_bound) <= 0:
            raise ValueError(
                f"{arg_name} must be positive; got {seqlen_bound}."
            )
    if q.ndim != 3:
        raise ValueError(
            "with block_table, q must be (nheads, total_q, headdim); got"
            f" {q.shape=}."
        )

    cu_seqlens_q = jnp.asarray(cu_seqlens_q)
    seqused_k = jnp.asarray(seqused_k)
    if (
        cu_seqlens_q.ndim != 1
        or cu_seqlens_q.shape[0] < 2
        or not jnp.issubdtype(cu_seqlens_q.dtype, jnp.integer)
    ):
        raise ValueError(
            "cu_seqlens_q must be a 1-D integer array of batch + 1 boundaries;"
            f" got {cu_seqlens_q.shape} {cu_seqlens_q.dtype}."
        )
    batch = cu_seqlens_q.shape[0] - 1
    cu_seqlens_q = cu_seqlens_q.astype(jnp.int32)
    # Note (david): under a trace even a closed-over concrete array indexes to
    # a tracer, so the check runs on the indexed entry.
    first_boundary = cu_seqlens_q[0]
    if (
        not isinstance(first_boundary, jax.core.Tracer)
        and int(first_boundary) != 0
    ):
        raise ValueError(
            "with block_table, cu_seqlens_q must start at 0; got"
            f" cu_seqlens_q[0]={int(first_boundary)}."
        )
    if num_active is None:
        rotary_seqused_k = seqused_k
    else:
        num_active = jnp.asarray(num_active)
        if num_active.size != 1 or not jnp.issubdtype(
            num_active.dtype, jnp.integer
        ):
            raise ValueError(
                "num_active must be one integer; got"
                f" {num_active.shape} {num_active.dtype}."
            )
        num_active = num_active.reshape(())
        # Note (david): a traced num_active is the caller's contract, like the
        # other traced metadata; out of range, the cu_seqlens_q lookup below
        # would clamp or wrap it instead of raising.
        if (
            not isinstance(num_active, jax.core.Tracer)
            and not 0 <= int(num_active) <= batch
        ):
            raise ValueError(
                f"num_active must be in [0, {batch}]; got {int(num_active)}."
            )
        num_active = num_active.astype(jnp.int32)
        # Note (david): pinning every boundary past num_active to
        # cu_seqlens_q[num_active] empties the inactive sequences, so they own no
        # q block, the kernel never reads their seqused_k or block_table entries,
        # and their rows become packed padding.
        cu_seqlens_q = jnp.where(
            jnp.arange(batch + 1, dtype=jnp.int32) <= num_active,
            cu_seqlens_q,
            cu_seqlens_q[num_active],
        )
        rotary_seqused_k = jnp.where(
            jnp.arange(batch, dtype=jnp.int32) < num_active, seqused_k, 0
        )

    total_q, head_dim_qk = q.shape[1], q.shape[2]
    rotary = prepare_rotary(
        rotary_cos,
        rotary_sin,
        head_dim=head_dim_qk,
        dtype=q.dtype,
        rotate_k=rotary_k,
        seqlen_q=total_q,
        seqlen_k=int(max_seqlen_k),
        interleaved=rotary_interleaved,
        cu_q=cu_seqlens_q,
        seqused_k=None if rotary_cos is None else rotary_seqused_k,
        max_seqlen_k=max_seqlen_k,
    )
    if blocks_override is None:
        paged_block_sizes = None
    else:
        block_q, block_kv, block_q_compute, block_kv_compute = blocks_override
        paged_block_sizes = BlockSizes(
            block_q=block_q,
            block_kv=block_kv,
            block_kv_compute=block_kv_compute,
            block_q_compute=block_q_compute,
        )
    softmax_scale = (
        1.0 / math.sqrt(head_dim_qk) if softmax_scale is None else softmax_scale
    )
    kernel = functools.partial(
        flash_attn_varlen_paged,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        causal=is_right_bounded,
        window_left=window_left,
        softcap=float(softcap),
        q_scale=float(fused_q_scale(softmax_scale, softcap)),
        return_lse=bool(return_softmax_lse),
        interpret=bool(interpret),
        block_sizes=paged_block_sizes,
        nheads_k=nheads_k,
        k_scale=descale_scalar(k_descale, "k_descale"),
        v_scale=descale_scalar(v_descale, "v_descale"),
    )
    # Note (david): K and V share the one merged cache operand k.
    return forward_only_attn(
        lambda q, kv_cache, unused_v, *kernel_args, **rotary_kwargs: kernel(
            q, kv_cache, *kernel_args, **rotary_kwargs
        ),
        "flash_attn_varlen_func",
        q,
        k,
        None,
        (cu_seqlens_q, seqused_k, block_table, check_sinks(sinks, q.shape[0])),
        rotary,
        rotary_interleaved,
    )


def flash_attn_varlen_func(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array | None,
    cu_seqlens_q: jax.Array,
    cu_seqlens_k: jax.Array | None,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: jax.Array | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    return_softmax_lse: bool = False,
    interpret: bool = False,
    head_dim: int | None = None,
    token_major: bool = False,
    block_sizes: tuple[int, int, int, int] | None = None,
    *,
    rotary_cos: jax.Array | None = None,
    rotary_sin: jax.Array | None = None,
    rotary_interleaved: bool = False,
    rotary_k: bool = True,
    block_table: jax.Array | None = None,
    seqused_k: jax.Array | None = None,
    num_active: jax.Array | int | None = None,
    k_descale: float | jax.Array | None = None,
    v_descale: float | jax.Array | None = None,
    nheads_k: int | None = None,
    learnable_sink: jax.Array | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """flash_attn.flash_attn_varlen_func on TPU for packed sequences.

    Head-major (default): q (nheads, total_q, headdim_qk), k (nheads_k,
    total_k, headdim_qk), v (nheads_k, total_k, headdim_v), out (nheads,
    total_q, headdim_v) -- the kernel's own head-major layout, so no operand is
    relaid out. Token-major (token_major=True) fuses the heads into the last
    axis, (total, nheads * headdim), and takes head_dim as flash_attn_func
    does. cu_seqlens_q / cu_seqlens_k are
    (num_seqs + 1,) int cumulative offsets and may be tracers; buffer rows past
    cu_seqlens[-1] are padding that never attends or is attended. Padding q
    rows return out = 0 and lse = -inf, as with a paged KV cache.

    causal and window_size are bottom-right aligned per sequence, as
    flash_attn_func aligns them on one sequence's own lengths; a row with no
    visible key is unspecified, so causal needs len_k(s) >= len_q(s).
    max_seqlen_q / max_seqlen_k must be static ints: they only cap the block
    sizes of the block-skipping schedule and are bucketed to powers of two, so
    an underestimate is slow, not wrong.

    Forward-only: jax.grad / jax.vjp raise NotImplementedError.
    return_softmax_lse adds lse (nheads, total_q) float32 in the varlen
    layout. RoPE follows flash_attn_func with per-sequence positions; the table
    must cover max_seqlen_k.

    block_sizes (TPU extension): (block_q, block_kv, block_q_compute,
    block_kv_compute) pins the forward tiles instead of the tuned or analytic
    ones. block_q and block_kv must be multiples of 128, and each packed total
    then pads to a multiple of its own block.

    Paged KV cache (block_table given; read-only, as in FlashAttention): k is
    one merged (num_pages, page_size, cache_heads, headdim) pool laid out as
    flash_attn_with_kvcache's (bf16 [k0, v0, k1, v1, ...], or fp8 with
    k_descale / v_descale and an explicit nheads_k), and v is None; headdim and
    page_size are multiples of 128, any nheads_k. block_table: (batch,
    max_pages_per_seq) int, never read past a sequence's own pages.
    cu_seqlens_k must be None and seqused_k ((batch,) int) is each sequence's
    kv length, its new tokens included, which the cache must already hold;
    max_seqlen_k bounds it. q / out stay head-major; token_major, rotary_k=True
    and a window right bound other than -1 or 0 raise, and RoPE rotates Q
    only, at positions seqused_k - len_q + t. window_size and softcap act as on
    packed K/V. cu_seqlens_q[0] must be 0; a concrete one is checked, a traced
    one is the caller's contract. A one-token sequence is an ordinary sequence.
    num_active (int or int32 scalar, possibly traced) keeps only the first
    num_active sequences; rows past the last kept one return out = 0 and lse =
    -inf. A concrete num_active outside [0, batch] raises; a traced one must
    lie in that range, which is not checked on device.
    learnable_sink: None or (nheads,) float sink logits, on the scale of the
    softcapped scores; exp(learnable_sink[h]) joins head h's softmax
    denominator with no value row (gpt-oss attention sinks), and lse includes
    it. Each logit must be finite or -inf (no sink); +inf and NaN are not
    checked and give undefined results. Padding rows and rows past num_active
    still return out = 0 and lse = -inf.
    """
    window_size = check_common_kwargs(
        window_size,
        dropout_p=dropout_p,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )
    if block_sizes is None:
        blocks_override = None
    else:
        blocks_override = tuple(block_sizes)
        if len(blocks_override) != 4 or not all(
            type(block) is int and block > 0 for block in blocks_override
        ):
            raise ValueError(
                "block_sizes must be 4 positive ints (block_q, block_kv,"
                f" block_q_compute, block_kv_compute); got {blocks_override!r}."
            )
        if blocks_override[0] % NUM_LANES or blocks_override[1] % NUM_LANES:
            raise ValueError(
                "block_q and block_kv must be multiples of 128 (the kernel's"
                f" native tile); got {blocks_override!r}."
            )
    if block_table is not None:
        return paged_varlen_attn(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            return_softmax_lse=return_softmax_lse,
            interpret=interpret,
            head_dim=head_dim,
            token_major=token_major,
            blocks_override=blocks_override,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            rotary_interleaved=rotary_interleaved,
            rotary_k=rotary_k,
            block_table=block_table,
            seqused_k=seqused_k,
            num_active=num_active,
            k_descale=k_descale,
            v_descale=v_descale,
            nheads_k=nheads_k,
            sinks=learnable_sink,
        )
    elif seqused_k is not None or num_active is not None:
        raise ValueError(
            "seqused_k and num_active describe a paged KV cache and need"
            " block_table; packed K/V takes cu_seqlens_k."
        )
    elif any(arg is not None for arg in (k_descale, v_descale, nheads_k)):
        raise ValueError(
            "k_descale, v_descale and nheads_k describe an fp8 paged pool and"
            " need block_table; packed K/V is bf16."
        )
    else:
        expected_rank = 2 if token_major else 3
        nheads, nheads_k, head_dim_qk, head_dim_v = parse_qkv(
            q, k, v, head_dim, token_major, expected_rank, "htd"
        )
        if token_major:
            token_axis = TOKEN_MAJOR_PACKED_AXIS
        else:
            token_axis = HEAD_MAJOR_PACKED_AXIS
        total_q, total_k = q.shape[token_axis], k.shape[token_axis]

        cu_seqlens_q = jnp.asarray(cu_seqlens_q)
        cu_seqlens_k = jnp.asarray(cu_seqlens_k)
        if cu_seqlens_q.ndim != 1 or cu_seqlens_q.shape != cu_seqlens_k.shape:
            raise ValueError(
                "cu_seqlens_q and cu_seqlens_k must be equal-shaped 1-D"
                f" arrays; got {cu_seqlens_q.shape=}, {cu_seqlens_k.shape=}."
            )
        for arg_name, seqlen_bound in (
            ("max_seqlen_q", max_seqlen_q),
            ("max_seqlen_k", max_seqlen_k),
        ):
            if isinstance(seqlen_bound, jax.core.Tracer):
                # Note (david): a non-concrete bound is a ValueError in the public
                # contract, not a TypeError.
                raise ValueError(
                    f"{arg_name} must be a concrete int (it picks the kernel's"
                    " block sizes); pass a static bucket bound, not a traced"
                    " value."
                )
            if int(seqlen_bound) <= 0:
                raise ValueError(
                    f"{arg_name} must be positive; got {seqlen_bound}."
                )

        rotary = prepare_rotary(
            rotary_cos,
            rotary_sin,
            head_dim=head_dim_qk,
            dtype=q.dtype,
            rotate_k=rotary_k,
            seqlen_q=total_q,
            seqlen_k=total_k,
            interleaved=rotary_interleaved,
            cu_q=cu_seqlens_q,
            cu_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
        )

        # Note (david): cu_seqlens needs no adjustment for the padding, because rows
        # past cu_seqlens_q[-1] own no q block and the kernel writes them as zeros.
        if blocks_override is None:
            padded_total_q = pad_seqlen(total_q)
            padded_total_k = pad_seqlen(total_k)
        else:
            padded_total_q = round_up(total_q, blocks_override[0])
            padded_total_k = round_up(total_k, blocks_override[1])
        num_pad_q = padded_total_q - total_q
        num_pad_kv = padded_total_k - total_k
        q = pad_axis(q, token_axis, num_pad_q)
        k, v = (pad_axis(operand, token_axis, num_pad_kv) for operand in (k, v))
        if rotary is None:
            rotary_coeffs = None
        else:
            q_coeff, k_coeff = rotary
            rotary_coeffs = (
                pad_axis(q_coeff, ROTARY_TOKEN_AXIS, num_pad_q),
                None
                if k_coeff is None
                else pad_axis(k_coeff, ROTARY_TOKEN_AXIS, num_pad_kv),
            )

        softmax_scale = (
            1.0 / math.sqrt(head_dim_qk)
            if softmax_scale is None
            else softmax_scale
        )
        # Note (david): the max_seqlen bounds only cap the block sizes but ride in
        # the kernel cache key, so power-of-two buckets keep a wobbling longest
        # sequence on one compiled kernel.
        max_seqlen_k_bucket = min(padded_total_k, next_pow2(max_seqlen_k))
        max_seqlen_q_bucket = min(padded_total_q, next_pow2(max_seqlen_q))
        # Note (david): causal_offset stays 0 because packed causal and window
        # masking is per-sequence bottom-right, derived from the cu_seqlens pair in
        # the kernel.
        build_kernel = functools.partial(
            get_kernel,
            num_heads=nheads,
            num_kv_heads=nheads_k,
            seqlen_q=padded_total_q,
            seqlen_kv=padded_total_k,
            causal=bool(causal),
            window_size=window_size,
            interpret=bool(interpret),
            softcap=float(softcap),
            head_dim=head_dim_qk,
            head_dim_v=head_dim_v,
            causal_offset=0,
            varlen_max_seqlen_kv=max_seqlen_k_bucket,
            varlen_max_seqlen_q=max_seqlen_q_bucket,
            token_major=bool(token_major),
            q_scale=float(fused_q_scale(softmax_scale, softcap)),
            blocks_override=blocks_override,
        )
        sinks = check_sinks(learnable_sink, nheads)
        sink_args = () if sinks is None else (sinks,)
        # Note (david): varlen has no backward kernel, so every path, lse included,
        # is forward-only and jax.grad names it instead of failing in the tracer.
        kernel_output = forward_only_attn(
            build_kernel(return_lse=bool(return_softmax_lse)),
            "flash_attn_varlen_func",
            q,
            k,
            v,
            (cu_seqlens_q, cu_seqlens_k, *sink_args),
            rotary_coeffs,
            rotary_interleaved,
        )
        if return_softmax_lse:
            out, kernel_lse = kernel_output
        else:
            out = kernel_output
        out = jax.lax.slice_in_dim(out, 0, total_q, axis=token_axis)
        if return_softmax_lse:
            # Note (david): the kernel emits lse as (nheads, padded_total_q) in both
            # layouts.
            return out, kernel_lse[:, :total_q]
        else:
            return out


@functools.partial(
    jax.jit,
    static_argnames=(
        "max_seqlen_q",
        "softmax_scale",
        "causal",
        "window_size",
        "softcap",
        "return_softmax_lse",
        "interpret",
        "nheads_k",
    ),
    donate_argnames=("k_cache", "v_cache"),
)
def flash_attn_with_kvcache(
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array | None,
    k: jax.Array | None = None,
    v: jax.Array | None = None,
    *,
    cache_seqlens: jax.Array | int,
    cache_batch_idx: jax.Array | None = None,
    block_table: jax.Array | None = None,
    num_active: jax.Array | int | None = None,
    cu_seqlens_q: jax.Array | None = None,
    max_seqlen_q: int | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    return_softmax_lse: bool = False,
    interpret: bool = False,
    k_descale: float | jax.Array | None = None,
    v_descale: float | jax.Array | None = None,
    nheads_k: int | None = None,
    learnable_sink: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
    """Attention against a contiguous or paged KV cache, with optional append.

    block_table picks one of two cache layouts; anything else raises
    ValueError:

    - block_table=None, contiguous pair (FlashAttention's layout): k_cache and
      v_cache are separate (batch_cache, capacity, nheads_k, head_dim) arrays,
      capacity a multiple of 128. cache_batch_idx: optional (batch,) cache row
      per query row, identity by default.
    - block_table given, paged merged: k_cache is one (num_pages, page_size,
      cache_heads, head_dim) pool whose token rows interleave the KV heads' K
      and V rows, page_size a multiple of 128, and v_cache is None. A bf16 pool
      holds each KV head's K then its V, [k0, v0, k1, v1, ...] (RPA v3's head
      order), cache_heads = 2 * nheads_k. A float8_e4m3fn or float8_e5m2 pool
      holds each head pair's two K rows then its two V rows, [k0, k1, v0, v1,
      ...], an odd nheads_k padded with one zero head (cache_heads = 2 *
      nheads_k rounded up to a multiple of 4). block_table: (batch,
      max_pages_per_seq) int mapping each row's positions to pages;
      cache_batch_idx must be None. Pages an append writes must be private to
      the request.

    An fp8 pool stores K / k_descale and V / v_descale (FlashAttention-3's
    descales, here one per-tensor scalar each, default 1, which may be traced:
    a new value compiles no new kernel) and feeds them to the MXU as stored; q
    stays bf16, and k / v appends must already be in the pool's dtype. nheads_k
    (TPU extension) is the pool's KV head count; it defaults to k's heads, else
    to half the head axis, so an fp8 pool, which pads an odd count, needs it
    without k.

    In both, nheads % nheads_k == 0. The contiguous pair needs nheads_k even or
    1 because its cache load packs two bf16 heads into one u32 lane, and it
    serves single-token decode only; the paged merged cache takes any nheads_k,
    and multi-token and packed calls (chunked prefill) need it: they append the
    new tokens, then run flash_attn_varlen_func's paged kernel over the pages.
    A (batch, seqlen_q, ...) call with no left window and at most
    MAX_FOLDED_QUERY_ROWS query rows per KV head (a speculative verify window)
    runs the decode kernel instead, the window folded into the query heads.

    q: (batch, seqlen_q, nheads, head_dim), or packed (total_q, nheads,
    head_dim) with cu_seqlens_q, (batch + 1,) nondecreasing int boundaries from
    0 to at most total_q. Packed buffer padding returns out = 0 and lse = -inf.
    With cu_seqlens_q, max_seqlen_q (required, a static positive int) is the
    longest request's query length; as in flash_attn_varlen_func it only picks
    the q tile, so a bound off the true length is slow, not wrong.
    Without cu_seqlens_q, seqlen_q is the bound and max_seqlen_q must be None.

    k / v: new tokens in q's token layout with nheads_k heads, appended after
    each row's cache_seqlens (a paged append writes each token's whole K/V
    row); pass both or neither. cache_seqlens: int or (batch,) valid entries
    per row before the append. num_active: the leading rows that are real
    requests; later rows are skipped, and their out / lse are 0 / -inf on
    multi-token and packed calls but unspecified on single-token decode. All of
    these may be tracers.

    causal and window_size are bottom-right aligned per request; multi-token and
    packed calls take a window right bound of -1 or 0 only (causal forces 0).
    softcap > 0 applies tanh(s / softcap) * softcap to the scaled scores. A
    decode row with nothing to attend returns out = 0 and lse = -inf; on
    multi-token and packed calls, as in flash_attn_varlen_func, a row with no
    visible key is unspecified. Forward-only.

    learnable_sink: None or (nheads,) float sink logits, on the scale of the
    softcapped scores (gpt-oss attention sinks): exp(learnable_sink[h]) joins
    head h's softmax denominator with no value row, and lse includes it. A
    decode row with nothing to attend then returns out = 0 and lse =
    learnable_sink[h]. Each logit must be finite or -inf (no sink); +inf and
    NaN are not checked and give undefined results.

    Returns (out, k_cache, v_cache) for the contiguous pair and (out, kv_cache)
    for the paged merged cache, with lse after out when return_softmax_lse:
    (batch, nheads, seqlen_q), or (nheads, total_q) when packed. The caches are
    donated and updated in place, so rebind them; under an outer jax.jit,
    donate them on that jit instead.
    """
    window_size = check_common_kwargs(window_size, softcap=softcap)
    is_ragged = cu_seqlens_q is not None
    if is_ragged:
        cu_seqlens_q = jnp.asarray(cu_seqlens_q)
        if (
            cu_seqlens_q.ndim != 1
            or cu_seqlens_q.shape[0] < 2
            or not jnp.issubdtype(cu_seqlens_q.dtype, jnp.integer)
        ):
            raise ValueError(
                "cu_seqlens_q must be a 1-D integer array of batch + 1"
                " boundaries."
            )
        if q.ndim != 3:
            raise ValueError(
                "With cu_seqlens_q, q must be (total_q, nheads, head_dim)."
            )
        # Note (david): max_seqlen_q picks the q tile; q.shape[0] counts every
        # packed request, so tiles sized by it compute mostly padding.
        if type(max_seqlen_q) is not int or max_seqlen_q <= 0:
            raise ValueError(
                "With cu_seqlens_q, max_seqlen_q must be a positive static"
                " python int, the longest request's query length; got"
                f" {max_seqlen_q!r}."
            )
        cu_seqlens_q = cu_seqlens_q.astype(jnp.int32)
        batch = cu_seqlens_q.shape[0] - 1
        nheads, head_dim = q.shape[-2:]
        seqlen_q = None
    elif max_seqlen_q is not None:
        raise ValueError(
            "max_seqlen_q is only for a packed q with cu_seqlens_q; a (batch,"
            " seqlen_q, nheads, head_dim) q takes seqlen_q from its shape; got"
            f" {max_seqlen_q=}."
        )
    elif q.ndim != 4:
        raise ValueError(
            f"q must be (batch, seqlen_q, nheads, head_dim); got {q.shape=}."
        )
    else:
        batch, seqlen_q, nheads, head_dim = q.shape
        if batch <= 0 or seqlen_q <= 0:
            raise ValueError("batch and seqlen_q must be positive.")
    is_extend = is_ragged or seqlen_q != 1
    # Note (david): a paged cache is always the merged one, a contiguous cache
    # always the K/V pair.
    is_paged = block_table is not None
    if is_extend and not is_paged:
        raise NotImplementedError(
            "Multi-token and packed KV-cache attention needs the paged merged"
            " cache (block_table); the contiguous K/V pair serves single-token"
            " decode only."
        )
    if is_paged:
        if v_cache is not None:
            raise ValueError(
                "With block_table, pass one merged (num_pages, page_size,"
                " cache_heads, head_dim) cache as k_cache and v_cache=None;"
                " separate paged K and V caches are not supported."
            )
        if cache_batch_idx is not None:
            raise ValueError(
                "block_table and cache_batch_idx are mutually exclusive: the"
                " block table already maps each query row to its pages."
            )
        if k_cache.ndim != 4:
            raise ValueError(
                "With block_table, k_cache must be (num_pages, page_size,"
                " cache_heads, head_dim), the KV heads' K and V rows"
                f" interleaved; got {k_cache.shape=}."
            )
        if nheads_k is None and k is not None:
            nheads_k = k.shape[-2]
        pool_heads = resolve_merged_kv_heads(
            k_cache.shape[2], k_cache.dtype, nheads_k
        )
        if k_cache.shape[1] % PAGE_SIZE_MULTIPLE:
            raise ValueError(
                "page_size (k_cache.shape[1]) must be a multiple of"
                f" {PAGE_SIZE_MULTIPLE}; got {k_cache.shape[1]}."
            )
        block_table = jnp.asarray(block_table)
        if not jnp.issubdtype(block_table.dtype, jnp.integer):
            raise ValueError(
                "block_table must be an integer array; got"
                f" {block_table.dtype}."
            )
        if block_table.ndim != 2 or block_table.shape[0] != batch:
            raise ValueError(
                f"block_table must be (batch={batch}, max_pages_per_seq); got"
                f" {block_table.shape}."
            )
        flat_block_table = block_table.astype(jnp.int32).reshape(-1)
        cache_shape = (*k_cache.shape[:2], pool_heads, k_cache.shape[3])
        cache_operands = (k_cache,)
    else:
        if v_cache is None:
            raise ValueError(
                "Without block_table, k_cache and v_cache must be separate"
                " (batch_cache, capacity, nheads_k, head_dim) caches; got"
                " v_cache=None (a merged cache must be paged)."
            )
        if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
            raise ValueError(
                "k_cache and v_cache must share the shape (batch_cache,"
                f" capacity, nheads_k, head_dim); got {k_cache.shape=},"
                f" {v_cache.shape=}."
            )
        if nheads_k is not None and nheads_k != k_cache.shape[2]:
            raise ValueError(
                f"{nheads_k=} must match the contiguous caches'"
                f" {k_cache.shape[2]}."
            )
        flat_block_table = None
        cache_shape = k_cache.shape
        cache_operands = (k_cache, v_cache)
    nheads_k = cache_shape[2]
    if cache_shape[3] != head_dim:
        raise ValueError(
            f"cache head_dim {cache_shape[3]} must match q's {head_dim}."
        )
    if nheads_k <= 0 or nheads <= 0 or nheads % nheads_k:
        raise ValueError(f"{nheads=} must be a multiple of {nheads_k=}.")
    has_new = k is not None
    if has_new != (v is not None):
        raise ValueError("k and v must be passed together, or both left out.")
    kv_operands = (*cache_operands, k, v) if has_new else cache_operands
    if (
        q.dtype != jnp.bfloat16
        or any(operand.dtype != k_cache.dtype for operand in kv_operands)
        or (not is_paged and k_cache.dtype != jnp.bfloat16)
    ):
        kv_dtypes = tuple(str(operand.dtype) for operand in kv_operands)
        raise NotImplementedError(
            "q must be bfloat16 and the KV operands share one dtype, bfloat16"
            f" or (paged pool) float8_e4m3fn / float8_e5m2; got q {q.dtype}, KV"
            f" {kv_dtypes}."
        )
    k_descale = descale_scalar(k_descale, "k_descale")
    v_descale = descale_scalar(v_descale, "v_descale")
    expected_new_shape = (*q.shape[:-2], nheads_k, head_dim)
    if has_new and (
        k.shape != expected_new_shape or v.shape != expected_new_shape
    ):
        raise ValueError(
            f"k and v must be {expected_new_shape} (matching q's token layout);"
            f" got {k.shape=}, {v.shape=}."
        )

    cache_seqlens = jnp.asarray(cache_seqlens, jnp.int32)
    if cache_seqlens.ndim == 0:
        row_cache_seqlens = jnp.broadcast_to(cache_seqlens, (batch,))
    else:
        row_cache_seqlens = cache_seqlens
    if cache_batch_idx is None:
        cache_batch_idx = jnp.arange(batch, dtype=jnp.int32)
    else:
        cache_batch_idx = jnp.asarray(cache_batch_idx, jnp.int32)
    for arg_name, row_values in (
        ("cache_seqlens", row_cache_seqlens),
        ("cache_batch_idx", cache_batch_idx),
    ):
        if row_values.shape != (batch,):
            raise ValueError(
                f"{arg_name} must be a scalar or ({batch},) int array; got"
                f" {row_values.shape}."
            )

    softmax_scale = (
        1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale
    )
    sinks = check_sinks(learnable_sink, nheads)
    if is_extend:
        is_right_bounded, window_left = paged_window(causal, window_size)
        is_folded = (
            not is_ragged
            and window_left is None
            and seqlen_q * (nheads // nheads_k) <= MAX_FOLDED_QUERY_ROWS
        )
    else:
        is_folded = False
    is_varlen = is_extend and not is_folded
    if is_varlen:
        if is_ragged:
            packed_cu_seqlens_q = cu_seqlens_q
            packed_max_seqlen_q = max_seqlen_q
        else:
            packed_cu_seqlens_q = (
                jnp.arange(batch + 1, dtype=jnp.int32) * seqlen_q
            )
            packed_max_seqlen_q = seqlen_q
        packed_q = q.reshape(-1, nheads, head_dim)
        if has_new:
            packed_k = k.reshape(-1, nheads_k, head_dim)
            packed_v = v.reshape(-1, nheads_k, head_dim)
        else:
            packed_k = packed_v = None
        kernel_outputs = flash_attn_kvcache_varlen(
            packed_q,
            k_cache,
            packed_k,
            packed_v,
            packed_cu_seqlens_q,
            row_cache_seqlens,
            flat_block_table,
            num_active,
            max_seqlen_q=packed_max_seqlen_q,
            q_scale=float(fused_q_scale(softmax_scale, softcap)),
            causal=is_right_bounded,
            window_left=window_left,
            softcap=float(softcap),
            return_lse=bool(return_softmax_lse),
            interpret=bool(interpret),
            nheads_k=nheads_k,
            k_scale=k_descale,
            v_scale=v_descale,
            sinks=sinks,
        )
    else:
        # Note (david): the decode kernel runs its softmax in log2 units straight
        # off q @ k^T, so softmax_scale * log2(e) (softmax_scale / softcap with a
        # softcap) is folded into Q here, with the same bf16 rounding the dense
        # kernels apply per staged Q tile. The K descale is one per-tensor
        # factor on every score, so it rides in the same scale.
        q_scaled = (
            q.astype(jnp.float32)
            * fused_q_scale(softmax_scale * k_descale, softcap)
        ).astype(q.dtype)
        new_token_shape = (batch, nheads_k, head_dim)
        num_active_array = jnp.asarray(
            batch if num_active is None else num_active, jnp.int32
        ).reshape(1)
        if is_folded:
            # Note (david): the decode kernel appends one token per row, so a
            # window's K/V go in by the ragged append.
            if has_new:
                k_cache = append_ragged(
                    k_cache,
                    k.reshape(-1, nheads_k, head_dim),
                    v.reshape(-1, nheads_k, head_dim),
                    jnp.arange(batch + 1, dtype=jnp.int32) * seqlen_q,
                    row_cache_seqlens,
                    flat_block_table,
                    num_active_array,
                    page_size=k_cache.shape[1],
                    pages_per_seq=flat_block_table.shape[0] // batch,
                    interpret=bool(interpret),
                )
                row_cache_seqlens = row_cache_seqlens + seqlen_q
            kernel_q = q_scaled
            kernel_has_new = False
        else:
            kernel_q = q_scaled.reshape(batch, nheads, head_dim)
            kernel_has_new = has_new
        if kernel_has_new:
            k_new, v_new = k.reshape(new_token_shape), v.reshape(
                new_token_shape
            )
        else:
            # Note (david): without has_new the kernel never touches these, but it
            # still needs correctly shaped operands to build its BlockSpecs.
            k_new = v_new = jnp.zeros(new_token_shape, k_cache.dtype)
        window_left = None if window_size[0] == -1 else window_size[0]
        kernel_outputs = flash_attn_kvcache_pallas(
            kernel_q,
            k_cache,
            v_cache,
            k_new,
            v_new,
            row_cache_seqlens,
            cache_batch_idx,
            cache_head_major=False,
            has_new=kernel_has_new,
            left=window_left,
            return_lse=bool(return_softmax_lse),
            interpret=bool(interpret),
            block_table=flat_block_table,
            num_active=num_active_array,
            merged_cache=is_paged,
            softcap=float(softcap),
            causal=is_folded and is_right_bounded,
            nheads_k=nheads_k if is_paged else None,
            v_scale=v_descale,
            sinks=sinks,
        )
    caches_out = kernel_outputs[1 : 1 + len(cache_operands)]
    if is_folded:
        # Note (david): a window is a multi-token call, whose rows past
        # num_active return out = 0 and lse = -inf, as on the varlen path.
        is_active = jnp.arange(batch) < num_active_array[0]
        out = jnp.where(is_active[:, None, None, None], kernel_outputs[0], 0)
    else:
        out = kernel_outputs[0].reshape(q.shape)
    if return_softmax_lse:
        kernel_lse = kernel_outputs[1 + len(cache_operands)]
        if is_ragged:
            lse = kernel_lse
        elif is_varlen:
            lse = kernel_lse.reshape(nheads, batch, seqlen_q).transpose(1, 0, 2)
        else:
            # Note (david): the decode kernel broadcasts each row's lse across a
            # full 128-lane tile; the first column is the lse.
            decode_lse = (
                kernel_lse[..., 0]
                .reshape(batch, seqlen_q, nheads)
                .transpose(0, 2, 1)
            )
            if is_folded:
                lse = jnp.where(is_active[:, None, None], decode_lse, -jnp.inf)
            else:
                lse = decode_lse
        return (out, lse, *caches_out)
    else:
        return (out, *caches_out)
