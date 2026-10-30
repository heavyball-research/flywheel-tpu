"""flash_attn_with_kvcache served by the TPU Pallas kernels.

Arguments the kernels do not implement are accepted but raise
NotImplementedError on any non-default value.
"""

import functools
import math

import jax
import jax.numpy as jnp

from .config import PAGE_SIZE_MULTIPLE
from .dispatch import resolve_kvcache_config, resolve_paged_block_sizes
from .inputs import (
    check_common_kwargs,
    check_sinks,
    descale_scalar,
    fused_q_scale,
    paged_window,
)
from .kernels.decode import flash_attn_kvcache_pallas
from .kernels.fwd_paged import check_paged_inputs, flash_attn_varlen_paged
from .kernels.kv_append import append_ragged
from .kv_layout import resolve_merged_kv_heads

# Note (david): the paged varlen kernel pads every request to a 128-token q
# block, so a short window folded into the decode kernel's query heads reads
# the cache once instead; the decode kernel keeps all of a row's query heads
# in VMEM, which caps the fold at this many rows per KV head.
MAX_FOLDED_QUERY_ROWS = 128


def flash_attn_kvcache_varlen(
    q: jax.Array,
    kv_cache: jax.Array,
    k: jax.Array | None,
    v: jax.Array | None,
    cu_seqlens_q: jax.Array,
    cache_seqlens: jax.Array,
    block_table: jax.Array,
    num_active: int | jax.Array | None,
    *,
    max_seqlen_q: int,
    q_scale: float,
    causal: bool,
    window_left: int | None,
    softcap: float,
    return_lse: bool,
    interpret: bool,
    nheads_k: int,
    k_scale: jax.Array,
    v_scale: jax.Array,
    sinks: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
    """Append packed new tokens once, then attend with the paged varlen kernel,
    which reads the updated pages only.

    q is (total_q, nheads, head_dim) packed by cu_seqlens_q; kv_cache is the
    merged interleaved (num_pages, page_size, merged_cache_heads(nheads_k),
    head_dim) cache and block_table its flat (batch * pages_per_seq,) page
    table. max_seqlen_q is dispatch.resolve_paged_block_sizes's; causal,
    window_left, softcap, q_scale, nheads_k, k_scale, v_scale and sinks are
    flash_attn_varlen_paged's. Returns (out, kv_cache) plus lse (nheads,
    total_q) float32 when return_lse.
    """
    batch = cu_seqlens_q.shape[0] - 1
    num_query_heads = q.shape[1]
    page_size = kv_cache.shape[1]
    pages_per_seq = block_table.size // batch
    if pages_per_seq == 0:
        raise ValueError(
            "block_table must reserve at least one page per sequence."
        )
    if q.shape[0] == 0 and return_lse:
        return (q, kv_cache, jnp.empty((num_query_heads, 0), jnp.float32))
    elif q.shape[0] == 0:
        return (q, kv_cache)
    else:
        num_active_array = jnp.asarray(
            batch if num_active is None else num_active, jnp.int32
        ).reshape(1)
        has_new = k is not None
        if has_new:
            updated_kv_cache = append_ragged(
                kv_cache,
                k,
                v,
                cu_seqlens_q,
                cache_seqlens,
                block_table,
                num_active_array,
                page_size=page_size,
                pages_per_seq=pages_per_seq,
                interpret=interpret,
            )
        else:
            updated_kv_cache = kv_cache
        # Note (david): pinning every boundary past num_active to
        # cu_seqlens_q[num_active] empties the inactive requests, so they own no
        # q block and the kernel never reads their lengths or table rows; their
        # rows become packed padding (out = 0, lse = -inf).
        num_active_scalar = num_active_array[0]
        kept_cu_seqlens_q = jnp.where(
            jnp.arange(batch + 1, dtype=jnp.int32) <= num_active_scalar,
            cu_seqlens_q,
            cu_seqlens_q[num_active_scalar],
        )
        total_lengths = cache_seqlens + (
            jnp.diff(cu_seqlens_q) if has_new else 0
        )
        head_major_q = q.transpose(1, 0, 2)
        row_block_table = block_table.reshape(batch, pages_per_seq)
        num_kv_heads = check_paged_inputs(
            head_major_q,
            updated_kv_cache,
            kept_cu_seqlens_q,
            total_lengths,
            row_block_table,
            sinks,
            nheads_k=nheads_k,
            block_sizes=None,
        )
        block_sizes = resolve_paged_block_sizes(
            head_major_q,
            updated_kv_cache,
            row_block_table,
            num_kv_heads=num_kv_heads,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=pages_per_seq * page_size,
            return_lse=return_lse,
            rotary=None,
            has_window=window_left is not None,
            block_sizes=None,
        )
        outputs = flash_attn_varlen_paged(
            head_major_q,
            updated_kv_cache,
            kept_cu_seqlens_q,
            total_lengths,
            row_block_table,
            sinks,
            block_sizes=block_sizes,
            causal=causal,
            q_scale=q_scale,
            return_lse=return_lse,
            interpret=interpret,
            window_left=window_left,
            softcap=softcap,
            nheads_k=nheads_k,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        if return_lse:
            out, lse = outputs
            return (out.transpose(1, 0, 2), updated_kv_cache, lse)
        else:
            return (outputs.transpose(1, 0, 2), updated_kv_cache)


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
        # Note (david): each page is its own DMA, so smaller pages would leave
        # the decode kernel descriptor-bound.
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
            causal=is_right_bounded,  # pylint: disable=possibly-used-before-assignment
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
        # Note (david): the decode kernel runs its softmax in log2 units
        # straight off q @ k^T, so softmax_scale * log2(e) (softmax_scale /
        # softcap with a softcap) is folded into Q here, with the same bf16
        # rounding the dense kernels apply per staged Q tile. The K descale is
        # one per-tensor factor on every score, so it rides in the same scale.
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
            # Note (david): without has_new the kernel never touches these, but
            # it still needs correctly shaped operands to build its BlockSpecs.
            k_new = v_new = jnp.zeros(new_token_shape, k_cache.dtype)
        window_left = None if window_size[0] == -1 else window_size[0]
        if is_paged:
            page_size = k_cache.shape[1]
            capacity = flat_block_table.shape[0] // batch * page_size
        else:
            page_size = None
            capacity = k_cache.shape[1]
        # Note (david): softcap stays out of the tuned key because decode is
        # bound by the cache read; the tanh did not move the v6e step time. q is
        # always bf16, so the key's dtype is the cache's.
        kv_config = resolve_kvcache_config(
            page_size=page_size,
            dtype=k_cache.dtype,
            # A folded window's tokens are query heads of the decode kernel.
            num_query_heads=seqlen_q * nheads,
            num_kv_heads=nheads_k,
            head_dim=head_dim,
            capacity=capacity,
            batch=batch,
            append=kernel_has_new,
            sliding_window=window_left,
            return_lse=bool(return_softmax_lse),
        )
        kernel_outputs = flash_attn_kvcache_pallas(
            kernel_q,
            k_cache,
            v_cache,
            k_new,
            v_new,
            row_cache_seqlens,
            cache_batch_idx,
            has_new=kernel_has_new,
            left=window_left,
            return_lse=bool(return_softmax_lse),
            interpret=bool(interpret),
            kv_config=kv_config,
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
