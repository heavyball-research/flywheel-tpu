"""Read-only varlen forward kernel over a paged KV cache.

Each sequence attends its own cache prefix [0, seqused_k[r]), staged page by
page through its block table row. The cache is one merged (num_pages,
page_size, cache_heads, head_dim) pool whose token rows interleave the kv
heads' K and V rows in kv_layout's merged order. Every head group is one kv
head, which stages only the u32 word rows of each page that hold its K and
V, and whose q heads are stacked over the rows of every score fragment.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp

from ...core.tiling import round_up
from ...core.tpu import F32_BYTES, NUM_LANES, NUM_SUBLANES
from ..config import (
    MIN_Q_TILES_PER_BLOCK,
    PAGE_SIZE_MULTIPLE,
    BlockSizes,
    PagedKVInfo,
    next_pow2,
)
from ..fwd_blocks import NUM_KV_STAGES, NUM_O_SLOTS
from ..fwd_pipeline import forward_common, fwd_body, rotary_fields
from ..kv_layout import kv_word_rows, resolve_merged_kv_heads
from ..schedule import (
    ScheduleTableLayout,
    make_paged_fwd_schedule,
    paged_schedule_table,
)
from ..softmax import overflow_guard_threshold
from ..traits import PagedFwdTraits
from ..variant import Variant, VariantOperands

# Note (david): the distance-1 fragment pipeline keeps five score-shaped f32
# temporaries live, plus a one-byte bounds mask per score.
SCORE_TEMPORARY_BYTES = 5 * F32_BYTES + 1
# Note (david): no per-head fold. The fragment pipeline is fully unrolled over
# head_fold * compute tiles, and folding a kv head's q heads that way measured
# 2.1x slower on v6e (h32 k4 d256, 1024-token chunks over 16K: 22.6 against
# 10.6 ms) with a 15x longer compile. The kv head's q heads are folded into the
# fragment rows instead (fwd_body's gqa_fold).
PAGED_HEAD_FOLD = 1


def flash_fwd_varlen_paged_kernel(
    table_ref: jax.Array,
    block_table_ref: jax.Array,
    kv_scales_ref: jax.Array,
    *refs: jax.Array,
    traits: PagedFwdTraits,
    variant: Variant,
    layout: ScheduleTableLayout,
) -> None:
    if variant.has_sinks:
        sinks_ref, q_hbm, kv_hbm, *refs = refs
    else:
        sinks_ref = None
        q_hbm, kv_hbm, *refs = refs
    window_left, window_right = variant.window
    schedule = make_paged_fwd_schedule(
        table_ref,
        block_table_ref,
        layout=layout,
        num_head_groups=traits.num_head_groups,
        q_heads_per_kv_head=traits.q_heads_per_kv_head,
        bq=traits.bq,
        bkv=traits.bkv,
        bkv_compute=traits.bkv_compute,
        page_size=traits.paged.page_size,
        pages_per_seq=traits.paged.pages_per_seq,
        causal=window_right == 0,
        left=window_left,
    )
    # Note (david): K and V share the one merged cache operand. The K and V
    # descales arrive in SMEM, so a new descale value compiles no new kernel.
    q_scale = variant.descaled_q_scale(kv_scales_ref[0])
    v_scale = kv_scales_ref[1]
    # The schedule masks each sequence around its own diagonal, so fwd_body
    # runs without the Variant's static mask.
    fwd_body(
        traits,
        variant.without_mask(),
        schedule,
        q_hbm,
        kv_hbm,
        kv_hbm,
        refs,
        operands=VariantOperands(
            q_scale=q_scale, v_scale=v_scale, sinks_ref=sinks_ref
        ),
    )


def vmem_buffer_bytes(shape: tuple[int, ...], dtype: jnp.dtype) -> int:
    """Bytes of one VMEM scratch buffer under Mosaic's v6 memref tiling.

    A second-minor axis shorter than one large tile pads only to a power of two,
    so a (2, head_dim) bf16 head pair costs no pad.
    """
    itemsize = jnp.dtype(dtype).itemsize
    *leading, rows, lanes = shape
    packing = F32_BYTES // itemsize
    large_tile_rows = NUM_SUBLANES * packing
    if rows >= large_tile_rows:
        tile_rows = large_tile_rows
    else:
        tile_rows = max(packing, next_pow2(min(rows, NUM_SUBLANES)))
    return (
        math.prod(leading)
        * round_up(rows, tile_rows)
        * round_up(lanes, NUM_LANES)
        * itemsize
    )


def paged_scratch_shapes(
    block_sizes: BlockSizes,
    *,
    head_dim: int,
    return_lse: bool,
    rotary_dtype: jnp.dtype | None,
    gqa_fold: int,
    kv_dtype: jnp.dtype,
) -> list[tuple[tuple[int, ...], jnp.dtype]]:
    """(shape, dtype) of each VMEM scratch buffer forward_common allocates for
    one paged build, in its order: Q stage, (K, V) word staging, K and V planes,
    bounds rows per kv compute tile, Q rotary coefficients, out stage, row max,
    row sum, f32 accumulator, pending probs, pending V tile, lse."""
    bq, bkv = block_sizes.block_q, block_sizes.block_kv
    bkv_compute = block_sizes.block_kv_compute
    num_stages = block_sizes.num_stages
    bf16, f32 = jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)
    if gqa_fold > 1:
        stage_shape = (num_stages, gqa_fold, bq, head_dim)
    else:
        stage_shape = (num_stages, bq, head_dim)
    group_bq = gqa_fold * bq
    if rotary_dtype is None:
        rotary_shapes = []
    else:
        # Note (david): prepare_rotary's one batch plane is shared by every
        # head, so one Q coefficient group is staged.
        rotary_shapes = [
            ((num_stages, 1, bq, head_dim), jnp.dtype(rotary_dtype))
        ]
    if return_lse:
        lse_shapes = [((num_stages, bq, NUM_LANES), f32)]
    else:
        lse_shapes = []
    return [
        (stage_shape, bf16),
        ((NUM_KV_STAGES, bkv, 1, kv_word_rows(kv_dtype), head_dim), kv_dtype),
        ((2, bkv, head_dim), bf16),
        ((bkv // bkv_compute, 2, bkv_compute), jnp.dtype(jnp.int32)),
        *rotary_shapes,
        (stage_shape, bf16),
        ((group_bq, NUM_LANES), f32),
        ((2, group_bq, NUM_LANES), f32),
        ((NUM_O_SLOTS, group_bq, head_dim), f32),
        (
            (
                gqa_fold * block_sizes.block_q_compute,
                block_sizes.block_kv_compute,
            ),
            f32,
        ),
        ((block_sizes.block_kv_compute, head_dim), bf16),
        *lse_shapes,
    ]


def estimate_vmem_bytes(
    block_sizes: BlockSizes,
    *,
    head_dim: int,
    return_lse: bool,
    rotary_dtype: jnp.dtype | None,
    gqa_fold: int,
    kv_dtype: jnp.dtype,
) -> int:
    """Scoped VMEM bytes of one paged build: its scratch plus the score
    temporaries of one compute fragment."""
    scratch_bytes = sum(
        vmem_buffer_bytes(shape, dtype)
        for shape, dtype in paged_scratch_shapes(
            block_sizes,
            head_dim=head_dim,
            return_lse=return_lse,
            rotary_dtype=rotary_dtype,
            gqa_fold=gqa_fold,
            kv_dtype=kv_dtype,
        )
    )
    temporary_bytes = (
        gqa_fold
        * block_sizes.block_q_compute
        * block_sizes.block_kv_compute
        * SCORE_TEMPORARY_BYTES
    )
    return scratch_bytes + temporary_bytes


def check_paged_inputs(
    q: jax.Array,
    kv_cache: jax.Array,
    cu_seqlens_q: jax.Array,
    seqused_k: jax.Array,
    block_table: jax.Array,
    sinks: jax.Array | None,
    *,
    nheads_k: int | None,
    block_sizes: BlockSizes | None,
) -> int:
    """Check flash_attn_varlen_paged's operands, and a pinned block_sizes,
    before dispatch picks the build's tiles; returns the pool's kv head count.

    The arguments are flash_attn_varlen_paged's; block_sizes is the caller's
    pinned one, None when dispatch picks it.
    """
    num_q_heads, total_q, head_dim = q.shape
    if q.dtype != jnp.bfloat16:
        raise NotImplementedError(
            f"paged attention takes a bfloat16 q; got {q.dtype}."
        )
    if kv_cache.ndim != 4:
        raise ValueError(
            "kv_cache must be one merged (num_pages, page_size, cache_heads,"
            f" head_dim) cache; got {kv_cache.shape}."
        )
    _, page_size, cache_heads, cache_head_dim = kv_cache.shape
    num_kv_heads = resolve_merged_kv_heads(
        cache_heads, kv_cache.dtype, nheads_k
    )
    if cache_head_dim != head_dim or head_dim % NUM_LANES:
        raise ValueError(
            f"head_dim must be a multiple of {NUM_LANES} shared by q and the"
            f" cache; got q {head_dim}, cache {cache_head_dim}."
        )
    # Note (david): each page is its own DMA, so smaller pages would leave the
    # kv stream descriptor-bound.
    if page_size % PAGE_SIZE_MULTIPLE:
        raise ValueError(
            f"page_size must be a multiple of {PAGE_SIZE_MULTIPLE}; got"
            f" {page_size}."
        )
    if num_q_heads % num_kv_heads:
        raise ValueError(
            f"nheads={num_q_heads} must be a multiple of"
            f" nheads_k={num_kv_heads}."
        )
    if total_q == 0:
        raise ValueError("q needs at least one token row.")
    if sinks is not None and sinks.shape != (num_q_heads,):
        raise ValueError(
            f"sinks must be ({num_q_heads},), one per q head; got"
            f" {sinks.shape}."
        )

    cu_seqlens_q = jnp.asarray(cu_seqlens_q)
    seqused_k = jnp.asarray(seqused_k)
    block_table = jnp.asarray(block_table)
    batch = cu_seqlens_q.shape[0] - 1
    if seqused_k.shape != (batch,) or not jnp.issubdtype(
        seqused_k.dtype, jnp.integer
    ):
        raise ValueError(
            f"seqused_k must be a ({batch},) integer array; got"
            f" {seqused_k.shape} {seqused_k.dtype}."
        )
    if (
        block_table.ndim != 2
        or block_table.shape[0] != batch
        or block_table.shape[1] == 0
        or not jnp.issubdtype(block_table.dtype, jnp.integer)
    ):
        raise ValueError(
            f"block_table must be a (batch={batch}, max_pages_per_seq >= 1)"
            f" integer array; got {block_table.shape} {block_table.dtype}."
        )

    if block_sizes is not None and block_sizes.block_kv % page_size:
        raise ValueError(
            f"block_kv={block_sizes.block_kv} must be whole pages of"
            f" page_size={page_size}."
        )
    if block_sizes is not None and (
        block_sizes.block_q % block_sizes.block_q_compute
        or block_sizes.block_q // block_sizes.block_q_compute
        < MIN_Q_TILES_PER_BLOCK
    ):
        raise ValueError(
            "the static-anchor softmax needs block_q to hold >= 2 whole q"
            f" compute tiles; got block_q={block_sizes.block_q},"
            f" block_q_compute={block_sizes.block_q_compute}."
        )
    return num_kv_heads


def flash_attn_varlen_paged(
    q: jax.Array,
    kv_cache: jax.Array,
    cu_seqlens_q: jax.Array,
    seqused_k: jax.Array,
    block_table: jax.Array,
    sinks: jax.Array | None = None,
    *,
    block_sizes: BlockSizes,
    variant: Variant,
    interpret: bool,
    rotary: tuple[jax.Array, None] | None = None,
    rotary_interleaved: bool = False,
    nheads_k: int | None = None,
    k_scale: float | jax.Array = 1.0,
    v_scale: float | jax.Array = 1.0,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Packed head-major attention over a paged KV cache, read-only.

    q is (nheads, total_q, head_dim) bf16 packed by cu_seqlens_q, whose first
    entry is 0. kv_cache is the merged (num_pages, page_size,
    merged_cache_heads(nheads_k), head_dim) pool, bf16 or fp8, whose token rows
    interleave the kv heads' K and V rows (kv_layout's merged order); nheads_k
    defaults to half its head axis, and an fp8 pool, which pads an odd head
    count, needs it. An fp8 pool is widened to bf16 once per staged block;
    k_scale multiplies every score and v_scale the output, both scalars that
    may be traced (variant.has_kv_scales). seqused_k is each sequence's kv
    length with its new tokens, which the cache must already hold; a
    block_table entry past those tokens' pages is only read through a clamp
    into the pool, and the rows it names are discarded.
    variant's mask aligns to the end of each sequence's keys, with a right
    bound of 0 (causal) or none, and no causal_offset; a left bound also hides
    the keys more than that many positions before that diagonal. Rows past
    cu_seqlens_q[-1] are out = 0, lse = -inf. rotary is prepare_rotary's (q
    coefficients, None) pair. sinks (None or (nheads,) float) adds
    exp(sinks[h]) to head h's softmax denominator, sink logits on the scale of
    the softcapped scores (gpt-oss attention sinks).

    The caller runs check_paged_inputs on the operands first, then takes
    block_sizes from dispatch.resolve_paged_block_sizes.
    """
    variant.check_paged_window()
    window_left, window_right = variant.window
    if variant.causal_offset or variant.runtime_causal_offset:
        raise ValueError(
            "a paged forward aligns its mask to each sequence's own keys; a"
            " kernel-wide causal_offset would be applied on top of it."
        )
    if not variant.has_kv_scales:
        raise ValueError(
            "the paged kernel always reads K and V descales; build its Variant"
            " with has_kv_scales=True."
        )
    variant.check_sinks_operand(sinks)
    causal = window_right == 0
    num_q_heads, total_q, head_dim = q.shape
    _, page_size, cache_heads, _ = kv_cache.shape
    num_kv_heads = resolve_merged_kv_heads(
        cache_heads, kv_cache.dtype, nheads_k
    )
    cu_seqlens_q = jnp.asarray(cu_seqlens_q)
    seqused_k = jnp.asarray(seqused_k)
    block_table = jnp.asarray(block_table)
    pages_per_seq = block_table.shape[1]
    capacity = pages_per_seq * page_size
    gqa_fold = num_q_heads // num_kv_heads
    bq = block_sizes.block_q
    paged = PagedKVInfo(
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        is_bitcast_load=not interpret,
    )

    padded_total_q = round_up(total_q, bq)
    num_pad_q = padded_total_q - total_q
    padded_q = jnp.pad(q, ((0, 0), (0, num_pad_q), (0, 0)))
    if rotary is None:
        padded_rotary = None
    else:
        padded_rotary = (
            jnp.pad(rotary[0], ((0, 0), (0, num_pad_q), (0, 0))),
            None,
        )

    # Note (david): resolving the schedule once per call spares every q block
    # a search of the cu_seqlens and its window arithmetic.
    table, layout = paged_schedule_table(
        cu_seqlens_q,
        seqused_k,
        padded_total_q=padded_total_q,
        bq=bq,
        bkv=block_sizes.block_kv,
        bkv_compute=block_sizes.block_kv_compute,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        causal=causal,
        left=window_left,
    )
    traits = PagedFwdTraits(
        block_sizes=block_sizes,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        head_dim_v=head_dim,
        head_fold=PAGED_HEAD_FOLD,
        transposed_pv=False,
        token_major=None,
        is_per_seq=True,
        guard_threshold=overflow_guard_threshold(capacity),
        **rotary_fields(padded_rotary, rotary_interleaved),
        paged=paged,
        gqa_fold=gqa_fold,
    )
    smem_operands = [
        table,
        block_table.astype(jnp.int32).reshape(-1),
        jnp.stack([
            jnp.asarray(scale, jnp.float32).reshape(())
            for scale in (k_scale, v_scale)
        ]),
    ]
    if sinks is not None:
        smem_operands.append(variant.sink_logits(sinks))
    kernel = partial(flash_fwd_varlen_paged_kernel, layout=layout)
    outputs = forward_common(
        kernel,
        smem_operands,
        padded_q,
        kv_cache,
        None,
        traits,
        # Note (david): as in make_flash_attn_mha, Q carries softmax_scale /
        # softcap, so log2(e) rides in the tanh output scale.
        variant,
        kernel_name="flash_attn_varlen_paged_fwd",
        interpret=interpret,
        rotary=padded_rotary,
    )
    if variant.return_lse:
        out, lse = outputs
        return out[:, :total_q], lse[:, :total_q]
    else:
        return outputs[:, :total_q]
