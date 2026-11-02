"""flash_attn_func / flash_attn_varlen_func / flash_attn_with_kvcache against a
naive reference.

The tests hold (batch, seqlen, nheads, headdim) arrays; flash_attn_func and
flash_attn_varlen_func take head-major operands, so each call transposes in and
out. A kernel passes when its error from the f32 reference is at most rtol times
the error of the same math run in bf16 with reordered ops, plus the rounding of
any arithmetic on the reference.
"""

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flywheel_tpu import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
from tests.padding import pad_input, unpad_input
from tests.test_util import (
    apply_rotary_emb,
    attention_ref,
    generate_qkv,
    generate_random_padding_mask,
    merge_kv_heads,
    randn,
    split_kv_heads,
)

INTERPRET = jax.default_backend() != "tpu"
# Note (david): interpret mode emulates the kernels on CPU and only affords
# short sequences.
MAX_INTERPRET_SEQLEN = 256
COMPILED_HDIMS = [64, 96, 128, 192, 256]
# Note (david): kernel tiles are 128 tokens wide, so the dense kernel and every
# KV cache layout need seqlen_k or the page size to be a multiple of 128.
TILE = 128
# Note (david): (softcap, local, rotary_fraction, rotary_interleaved). RoPE is
# orthogonal to softcap and local windows, so it runs on the plain mask only,
# and the kernels rotate the full head only, so rotary_fraction skips
# FlashAttention's 0.5.
FEATURE_CASES = [
    (softcap, local, 0.0, False)
    for softcap in (0.0, 15.0)
    for local in (False, True)
] + [
    (0.0, False, 1.0, rotary_interleaved)
    for rotary_interleaved in (False, True)
]


def on_backend(seqlens):
    """The (seqlen_q, seqlen_k, ...) cases the backend affords."""
    return [
        case
        for case in seqlens
        if not INTERPRET or max(case[:2]) <= MAX_INTERPRET_SEQLEN
    ]


def dense_cases(seqlens):
    """(seqlen_q, seqlen_k, causal) flash_attn_func supports: the dense kernel
    needs seqlen_k % 128 == 0 unless causal keeps every row off the padded kv
    tail."""
    return on_backend([
        (seqlen_q, seqlen_k, causal)
        for seqlen_q, seqlen_k in seqlens
        for causal in (False, True)
        if causal or seqlen_k % TILE == 0
    ])


def dense_feature_cases(seqlens):
    """dense_cases with (softcap, local, rotary_fraction, rotary_interleaved)
    appended; RoPE puts the queries on the last seqlen_q keys, so it needs
    seqlen_q <= seqlen_k.
    """
    return [
        (
            seqlen_q,
            seqlen_k,
            causal,
            softcap,
            local,
            rotary_fraction,
            rotary_interleaved,
        )
        for seqlen_q, seqlen_k, causal in dense_cases(seqlens)
        for softcap, local, rotary_fraction, rotary_interleaved in FEATURE_CASES
        if rotary_fraction == 0.0 or seqlen_q <= seqlen_k
    ]


def varlen_feature_cases(seqlens):
    """(seqlen_q, seqlen_k, causal, softcap, local, rotary_fraction,
    rotary_interleaved) for flash_attn_varlen_func; the test trims each
    sequence's query length to what the mask and RoPE need."""
    return on_backend([
        (seqlen_q, seqlen_k, causal, *features)
        for seqlen_q, seqlen_k in seqlens
        for causal in (False, True)
        for features in FEATURE_CASES
    ])


def to_head_major(x):
    """(batch, seqlen, nheads, d) -> flash_attn_func's (batch, nheads, seqlen,
    d), or packed (total, nheads, d) -> flash_attn_varlen_func's (nheads, total,
    d); the map is its own inverse."""
    return jnp.swapaxes(x, -3, -2)


def max_abs(x):
    return jnp.abs(x.astype(jnp.float32)).max(initial=0.0).item()


def mean_abs(x):
    return jnp.abs(x.astype(jnp.float32)).mean().item()


def arithmetic_atol(ref):
    """Numerical error if we just do any arithmetic on ref."""
    return 2 * max_abs(ref + 0.3 - 0.3 - ref)


def assert_lse_close(lse, lse_ref, lse_bf16, valid, rtol):
    """lse matches on the valid rows.

    Note (david): the kernels round the scaled q to bf16 once per row, an error
    every score of the row shares, while the bf16 reference rounds each score on
    its own and averages that away over a long row, so lse gets the out rule
    plus a fixed 1e-2.
    """
    err = max_abs(jnp.where(valid, lse - lse_ref, 0.0))
    err_bf16 = max_abs(jnp.where(valid, lse_bf16 - lse_ref, 0.0))
    print(f"LSE max diff: {err}, bf16 reference LSE max diff: {err_bf16}")
    assert err <= rtol * err_bf16 + 1e-2


@pytest.mark.parametrize("dtype", [jnp.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("d", COMPILED_HDIMS)
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k,causal,softcap,local,rotary_fraction,rotary_interleaved",
    dense_feature_cases([
        (1, 1),
        (64, 128),
        (128, 192),
        (256, 256),
        (239, 1),
        (799, 3),
        (113, 203),
        (113, 128),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (384, 256),
        (640, 128),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (4096, 4096),
        (4224, 4224),
    ]),
)
def test_flash_attn_output(
    seqlen_q,
    seqlen_k,
    d,
    causal,
    local,
    softcap,
    rotary_fraction,
    rotary_interleaved,
    mha_type,
    dtype,
):
    rng = np.random.default_rng(0)
    batch_size = 9 if seqlen_k <= 2048 else 2
    nheads = 6
    nheads_kv = nheads if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    # rotary_dim must be a multiple of 16, and must be <= d
    rotary_dim = math.floor(int(rotary_fraction * d) / 16) * 16
    dv_vals = (
        [128, d]
        if d > 128 and d <= 192
        else ([256, 512, d] if d <= 64 else [d])
    )
    for dv in dv_vals:
        print(f"{dv = }")
        q = randn(rng, (batch_size, seqlen_q, nheads, d), dtype)
        if softcap > 0.0:
            # Note (david): lifting q puts the scores in the range the cap
            # bends.
            q = q * softcap / 4
        k = randn(rng, (batch_size, seqlen_k, nheads_kv, d), dtype)
        v = randn(rng, (batch_size, seqlen_k, nheads_kv, dv), dtype)
        window_size = (
            (-1, -1)
            if not local
            else tuple(rng.integers(0, seqlen_k, 2).tolist())
        )
        if rotary_dim > 0:
            angle = rng.random((seqlen_k, rotary_dim // 2)) * 2 * math.pi
            cos = jnp.asarray(np.cos(angle), dtype)
            sin = jnp.asarray(np.sin(angle), dtype)
            # Note (david): flash_attn_func rotates K at positions 0..seqlen_k -
            # 1 and Q at the last seqlen_q of them.
            q_ro = apply_rotary_emb(
                q,
                cos,
                sin,
                seqlen_offsets=seqlen_k - seqlen_q,
                interleaved=rotary_interleaved,
            )
            k_ro = apply_rotary_emb(k, cos, sin, interleaved=rotary_interleaved)
        else:
            cos, sin = None, None
            q_ro, k_ro = q, k
        out_ref, lse_ref = attention_ref(
            q_ro,
            k_ro,
            v,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
        )
        out_bf16, lse_bf16 = attention_ref(
            q_ro,
            k_ro,
            v,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            upcast=False,
            reorder_ops=True,
        )
        print(f"bf16 reference max diff: {max_abs(out_bf16 - out_ref)}")
        print(f"bf16 reference mean diff: {mean_abs(out_bf16 - out_ref)}")
        fwd_atol = arithmetic_atol(out_ref)
        rtol = 2 if softcap == 0.0 else 3

        out, lse = flash_attn_func(
            *map(to_head_major, (q, k, v)),
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            return_softmax_lse=True,
            interpret=INTERPRET,
            rotary_cos=cos,
            rotary_sin=sin,
            rotary_interleaved=rotary_interleaved,
        )
        out = to_head_major(out)
        print(f"Output max diff: {max_abs(out - out_ref)}")
        print(f"Output mean diff: {mean_abs(out - out_ref)}")
        assert (
            max_abs(out - out_ref)
            <= rtol * max_abs(out_bf16 - out_ref) + fwd_atol
        )
        # Note (david): rows with no visible key are exact zeros with lse -inf.
        assert jnp.array_equal(jnp.isneginf(lse), jnp.isneginf(lse_ref))
        assert_lse_close(lse, lse_ref, lse_bf16, jnp.isfinite(lse_ref), rtol)

        # Note (david): flash_attn_func's backward covers causal and full masks
        # only; window_size, softcap and RoPE are forward-only.
        if not local and softcap == 0.0 and rotary_dim == 0:
            g = randn(rng, out.shape, dtype)

            def attn(q, k, v):
                out = flash_attn_func(
                    *map(to_head_major, (q, k, v)),
                    causal=causal,
                    interpret=INTERPRET,
                )
                return to_head_major(out)

            def attn_ref(q, k, v, **kwargs):
                return attention_ref(q, k, v, causal=causal, **kwargs)[0]

            dq, dk, dv_ = jax.vjp(attn, q, k, v)[1](g)
            dq_ref, dk_ref, dv_ref = jax.vjp(attn_ref, q, k, v)[1](g)
            dq_bf16, dk_bf16, dv_bf16 = jax.vjp(
                lambda q, k, v: attn_ref(  # pylint: disable=cell-var-from-loop
                    q, k, v, upcast=False, reorder_ops=True
                ),
                q,
                k,
                v,
            )[1](g)
            for name, grad, grad_ref, grad_bf16 in (
                ("dQ", dq, dq_ref, dq_bf16),
                ("dK", dk, dk_ref, dk_bf16),
                ("dV", dv_, dv_ref, dv_bf16),
            ):
                print(f"{name} max diff: {max_abs(grad - grad_ref)}")
                print(f"{name} mean diff: {mean_abs(grad - grad_ref)}")
                print(
                    f"{name} bf16 reference max diff:"
                    f" {max_abs(grad_bf16 - grad_ref)}"
                )
                print(
                    f"{name} bf16 reference mean diff:"
                    f" {mean_abs(grad_bf16 - grad_ref)}"
                )
                # Note (david): a row with one visible key has a dQ of exactly 0
                # in both references, while the kernel's D = rowsum(dO * O) and
                # dP sum in different orders, so the floor keeps an all-zero
                # reference from demanding a bit-exact 0.
                atol = arithmetic_atol(grad_ref) + 1e-4
                assert (
                    max_abs(grad - grad_ref)
                    <= rtol * max_abs(grad_bf16 - grad_ref) + atol
                ), name


@pytest.mark.parametrize("dtype", [jnp.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("d", COMPILED_HDIMS)
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k,causal,softcap,local,rotary_fraction,rotary_interleaved",
    varlen_feature_cases([
        (1, 1),
        (1, 3),
        (2, 1),
        (511, 1),
        (3, 513),
        (64, 128),
        (128, 128),
        (256, 256),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (307, 256),
        (640, 128),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
        (4096, 4096),
    ]),
)
def test_flash_attn_varlen_output(
    seqlen_q,
    seqlen_k,
    d,
    causal,
    local,
    softcap,
    rotary_fraction,
    rotary_interleaved,
    mha_type,
    dtype,
):
    rng = np.random.default_rng(
        seqlen_q + seqlen_k + d + int(causal) * 2 + int(local)
    )
    batch_size = 9 if seqlen_q <= 2048 else 2
    nheads = 6
    nheads_kv = nheads if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    # rotary_dim must be a multiple of 16, and must be <= d
    rotary_dim = math.floor(int(rotary_fraction * d) / 16) * 16
    dv_vals = (
        [128, d]
        if d > 128 and d <= 192
        else ([256, 512, d] if d <= 64 else [d])
    )
    for dv in dv_vals:
        print(f"{dv = }")
        q = randn(rng, (batch_size, seqlen_q, nheads, d), dtype)
        if softcap > 0.0:
            # Note (david): lifting q puts the scores in the range the cap
            # bends.
            q = q * softcap / 4
        k = randn(rng, (batch_size, seqlen_k, nheads_kv, d), dtype)
        v = randn(rng, (batch_size, seqlen_k, nheads_kv, dv), dtype)
        window_size = (
            (-1, -1)
            if not local
            else tuple(rng.integers(0, seqlen_k, 2).tolist())
        )
        query_padding_mask = generate_random_padding_mask(
            seqlen_q, batch_size, rng
        )
        key_padding_mask = generate_random_padding_mask(
            seqlen_k, batch_size, rng
        )
        # Note (david): flash_attn_varlen_func leaves a row with no visible key
        # unspecified, so no sequence has one: every key length is positive, and
        # under a right bound r each query length stays within its key length +
        # r. RoPE puts each sequence's queries on its last keys, which needs r =
        # 0.
        right = 0 if causal or rotary_dim > 0 else window_size[1]
        if right >= 0:
            query_padding_mask = query_padding_mask & (
                jnp.arange(seqlen_q)[None, :]
                < key_padding_mask.sum(-1, keepdims=True) + right
            )
        (
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            output_pad_fn,
        ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask)
        if rotary_dim > 0:
            angle = rng.random((seqlen_k, rotary_dim // 2)) * 2 * math.pi
            cos = jnp.asarray(np.cos(angle), dtype)
            sin = jnp.asarray(np.sin(angle), dtype)
            q_ro = apply_rotary_emb(
                q,
                cos,
                sin,
                seqlen_offsets=key_padding_mask.sum(-1)
                - query_padding_mask.sum(-1),
                interleaved=rotary_interleaved,
            )
            k_ro = apply_rotary_emb(k, cos, sin, interleaved=rotary_interleaved)
        else:
            cos, sin = None, None
            q_ro, k_ro = q, k
        out_ref, lse_ref = attention_ref(
            q_ro,
            k_ro,
            v,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
        )
        out_bf16, lse_bf16 = attention_ref(
            q_ro,
            k_ro,
            v,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            upcast=False,
            reorder_ops=True,
        )
        print(f"bf16 reference max diff: {max_abs(out_bf16 - out_ref)}")
        print(f"bf16 reference mean diff: {mean_abs(out_bf16 - out_ref)}")
        fwd_atol = arithmetic_atol(out_ref)
        rtol = 2 if softcap == 0.0 else 3

        valid = jnp.broadcast_to(query_padding_mask[:, None, :], lse_ref.shape)
        assert jnp.isfinite(jnp.where(valid, lse_ref, 0.0)).all()

        out_unpad, lse_unpad = flash_attn_varlen_func(
            *map(to_head_major, (q_unpad, k_unpad, v_unpad)),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            return_softmax_lse=True,
            interpret=INTERPRET,
            rotary_cos=cos,
            rotary_sin=sin,
            rotary_interleaved=rotary_interleaved,
        )
        out = output_pad_fn(to_head_major(out_unpad))
        lse = jnp.swapaxes(output_pad_fn(lse_unpad.T), 1, 2)
        print(f"Output max diff: {max_abs(out - out_ref)}")
        print(f"Output mean diff: {mean_abs(out - out_ref)}")
        assert (
            max_abs(out - out_ref)
            <= rtol * max_abs(out_bf16 - out_ref) + fwd_atol
        )
        assert_lse_close(lse, lse_ref, lse_bf16, valid, rtol)


def generate_block_kvcache(
    seqlen_k, page_size, batch_size, nheads_k, d, dv, rng, dtype
):
    num_blocks = math.ceil(seqlen_k / page_size) * batch_size * 3
    k_cache_paged = randn(rng, (num_blocks, page_size, nheads_k, d), dtype)
    v_cache_paged = randn(rng, (num_blocks, page_size, nheads_k, dv), dtype)
    page_table = jnp.asarray(rng.permutation(num_blocks), jnp.int32).reshape(
        batch_size, -1
    )
    k_cache = gather_paged_cache(k_cache_paged, page_table)[:, :seqlen_k]
    v_cache = gather_paged_cache(v_cache_paged, page_table)[:, :seqlen_k]
    return (
        k_cache,
        v_cache,
        page_table,
        k_cache_paged,
        v_cache_paged,
        num_blocks,
    )


def gather_paged_cache(cache_paged, page_table):
    """(num_blocks, page_size, ...) pages -> each row's (nblocks * page_size,
    ...) view."""
    batch_size, nblocks = page_table.shape
    return cache_paged[page_table.reshape(-1)].reshape(
        batch_size, nblocks * cache_paged.shape[1], *cache_paged.shape[2:]
    )


KV_DTYPES = [jnp.bfloat16, jnp.float8_e4m3fn, jnp.float8_e5m2]
# Note (david): power-of-two descales keep every dequantized fp8 value exact in
# bf16, so the bf16 reference sees the very K and V the kernel feeds the MXU.
FP8_K_DESCALE, FP8_V_DESCALE = 2.0**-3, 2.0**-2


def kvcache_cases(seqlens):
    """(seqlen_q, seqlen_k, page_size, has_batch_idx, varlen_q, mha_type,
    kv_dtype) the TPU caches support.

    The contiguous cache (page_size None) serves single-token decode only and
    packs two bf16 KV heads per u32 lane, so gqa's 3 KV heads need the paged
    cache; a paged cache's block_table maps the rows, so cache_batch_idx is
    contiguous-only. Only the paged pool takes an fp8 cache.
    """
    cases = []
    for (seqlen_q, seqlen_k), *case in itertools.product(
        seqlens,
        [None, 128, 256],
        [False, True],
        [False, True],
        ["mha", "mqa", "gqa"],
        KV_DTYPES,
    ):
        page_size, has_batch_idx, varlen_q, mha_type, kv_dtype = case
        if page_size is None:
            supported = (
                seqlen_q == 1
                and not varlen_q
                and mha_type != "gqa"
                and kv_dtype == jnp.bfloat16
            )
        else:
            supported = seqlen_k % page_size == 0 and not has_batch_idx
        if supported:
            cases.append((seqlen_q, seqlen_k, *case))
    return on_backend(cases)


@pytest.mark.parametrize("dtype", [jnp.bfloat16])
@pytest.mark.parametrize("new_kv", [False, True])
@pytest.mark.parametrize(
    "causal,local", [(False, False), (True, False), (False, True)]
)
@pytest.mark.parametrize("softcap", [0.0, 15.0])
@pytest.mark.parametrize("d", [128])
# Note (david): every cache layout holds whole 128-token tiles, so the
# capacities are multiples of 128; cache_seqlens draws unaligned lengths.
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k,page_size,has_batch_idx,varlen_q,mha_type,kv_dtype",
    kvcache_cases([
        (1, 128),
        (3, 1024),
        (64, 256),
        (64, 2048),
        (128, 128),
        (256, 512),  # an append spanning more than one page
    ]),
)
def test_flash_attn_kvcache(
    seqlen_q,
    seqlen_k,
    d,
    varlen_q,
    has_batch_idx,
    page_size,
    softcap,
    causal,
    local,
    new_kv,
    mha_type,
    kv_dtype,
    dtype,
):
    rng = np.random.default_rng(0)
    if kv_dtype == jnp.bfloat16:
        k_descale, v_descale = 1.0, 1.0
    else:
        k_descale, v_descale = FP8_K_DESCALE, FP8_V_DESCALE

    def quantize(x, descale):
        """The stored form of K or V: x / descale cast to the cache dtype."""
        return (x.astype(jnp.float32) / descale).astype(kv_dtype)

    def dequantize(x, descale):
        return (x.astype(jnp.float32) * descale).astype(dtype)

    def cache_rows(pages, descale):
        """Each batch row's dequantized (seqlen_k, ...) view of a paged pool."""
        return gather_paged_cache(dequantize(pages, descale), page_table)[
            :, :seqlen_k
        ]

    batch_size = 5
    batch_size_cache = batch_size if not has_batch_idx else batch_size * 2
    nheads = 6
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    dv = d
    q = randn(rng, (batch_size, seqlen_q, nheads, d), dtype)
    if softcap > 0.0:
        # Note (david): lifting q puts the scores in the range the cap bends.
        q = q * softcap / 4
    if varlen_q:
        query_padding_mask = generate_random_padding_mask(
            seqlen_q, batch_size, rng
        )
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, *_ = unpad_input(
            q, query_padding_mask
        )

        def output_pad_fn(output_unpad):
            return pad_input(output_unpad, indices_q, batch_size, seqlen_q)

    else:
        query_padding_mask = None
        q_unpad = q
        cu_seqlens_q, max_seqlen_q = None, None
    # Note (david): the paged kernel bounds a row's keys on the right only at
    # the causal diagonal, so a local window here is left-only.
    window_size = (
        (-1, -1) if not local else (int(rng.integers(0, seqlen_k)), -1)
    )

    # Note (david): appended k / v share q's token layout, so seqlen_new is
    # seqlen_q and varlen k / v follow q's packing.
    seqlen_new = seqlen_q
    if new_kv:
        # Note (david): an fp8 pool takes appends already quantized; k / v below
        # are the dequantized values the reference attends.
        k_stored = quantize(
            randn(rng, (batch_size, seqlen_new, nheads_k, d), dtype), k_descale
        )
        v_stored = quantize(
            randn(rng, (batch_size, seqlen_new, nheads_k, dv), dtype), v_descale
        )
        k, v = dequantize(k_stored, k_descale), dequantize(v_stored, v_descale)
        if varlen_q:
            key_new_padding_mask = query_padding_mask
            k_unpad, indices_k, *_ = unpad_input(k_stored, key_new_padding_mask)
            v_unpad, *_ = unpad_input(v_stored, key_new_padding_mask)
        else:
            k_unpad, v_unpad = k_stored, v_stored
    else:
        k, v, k_unpad, v_unpad = None, None, None, None
    if page_size is None:
        k_cache = randn(rng, (batch_size_cache, seqlen_k, nheads_k, d), dtype)
        v_cache = randn(rng, (batch_size_cache, seqlen_k, nheads_k, dv), dtype)
        page_table = None
    else:
        (
            k_cache,
            v_cache,
            page_table,
            k_cache_paged,
            v_cache_paged,
            _,
        ) = generate_block_kvcache(
            seqlen_k, page_size, batch_size_cache, nheads_k, d, dv, rng, dtype
        )
        k_cache_paged = quantize(k_cache_paged, k_descale)
        v_cache_paged = quantize(v_cache_paged, v_descale)
        k_cache, v_cache = cache_rows(k_cache_paged, k_descale), cache_rows(
            v_cache_paged, v_descale
        )
    # Note (david): a multi-token row with no visible key is unspecified, so a
    # causal read-only call keeps every query at or past position 0:
    # cache_seqlens >= seqlen_q.
    cache_seqlens = jnp.asarray(
        rng.integers(
            0 if new_kv else (seqlen_q if causal else 1),
            (seqlen_k - seqlen_new + 1) if new_kv else (seqlen_k + 1),
            (batch_size,),
        ),
        jnp.int32,
    )
    if has_batch_idx:
        cache_batch_idx = jnp.asarray(
            rng.permutation(batch_size_cache)[:batch_size], jnp.int32
        )
    else:
        cache_batch_idx = None
    arange = jnp.arange(seqlen_k)[None, :]
    cache_seqlens_expanded = cache_seqlens[:, None]
    if not new_kv:
        key_padding_mask = arange < cache_seqlens_expanded
    else:
        k_new_seqlens = (
            key_new_padding_mask.sum(-1, keepdims=True)
            if varlen_q
            else seqlen_new
        )
        key_padding_mask = arange < cache_seqlens_expanded + k_new_seqlens
    k_cache_ref = k_cache if not has_batch_idx else k_cache[cache_batch_idx]
    v_cache_ref = v_cache if not has_batch_idx else v_cache[cache_batch_idx]
    if new_kv:
        update_mask = (cache_seqlens_expanded <= arange) & (
            arange < cache_seqlens_expanded + k_new_seqlens
        )
        k_to_update = k.reshape(-1, nheads_k, d)
        v_to_update = v.reshape(-1, nheads_k, dv)
        if varlen_q:
            k_to_update = k_to_update[indices_k]
            v_to_update = v_to_update[indices_k]
        update_index = jnp.nonzero(update_mask)
        k_cache_ref = k_cache_ref.at[update_index].set(k_to_update)
        v_cache_ref = v_cache_ref.at[update_index].set(v_to_update)
    out_ref, lse_ref = attention_ref(
        q,
        k_cache_ref,
        v_cache_ref,
        query_padding_mask,
        key_padding_mask,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
    )
    out_bf16, lse_bf16 = attention_ref(
        q,
        k_cache_ref,
        v_cache_ref,
        query_padding_mask,
        key_padding_mask,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        upcast=False,
        reorder_ops=True,
    )
    # Note (david): the caches are donated, so every call gets a fresh copy.
    if page_size is None:
        caches = (jnp.copy(k_cache), jnp.copy(v_cache))
    else:
        caches = (merge_kv_heads(k_cache_paged, v_cache_paged), None)
    out, lse, *caches_out = flash_attn_with_kvcache(
        q if not varlen_q else q_unpad,
        *caches,
        k_unpad,
        v_unpad,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        block_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        return_softmax_lse=True,
        interpret=INTERPRET,
        k_descale=k_descale,
        v_descale=v_descale,
        nheads_k=nheads_k,
    )
    if varlen_q:
        out = output_pad_fn(out)
        lse = jnp.swapaxes(output_pad_fn(lse.T), 1, 2)
    if query_padding_mask is None:
        valid = jnp.ones(lse_ref.shape, bool)
    else:
        valid = jnp.broadcast_to(query_padding_mask[:, None, :], lse_ref.shape)
    assert jnp.isfinite(jnp.where(valid, lse_ref, 0.0)).all()
    print(f"Output max diff: {max_abs(out - out_ref)}")
    print(f"Output mean diff: {mean_abs(out - out_ref)}")
    print(f"bf16 reference max diff: {max_abs(out_bf16 - out_ref)}")
    print(f"bf16 reference mean diff: {mean_abs(out_bf16 - out_ref)}")

    if new_kv:
        if page_size is None:
            k_cache_out, v_cache_out = caches_out
            k_cache_select = (
                k_cache_out
                if not has_batch_idx
                else k_cache_out[cache_batch_idx]
            )
            v_cache_select = (
                v_cache_out
                if not has_batch_idx
                else v_cache_out[cache_batch_idx]
            )
        else:
            (kv_cache_out,) = caches_out
            k_pages_out, v_pages_out = split_kv_heads(kv_cache_out, nheads_k)
            k_cache_select = cache_rows(k_pages_out, k_descale)
            v_cache_select = cache_rows(v_pages_out, v_descale)
        assert jnp.array_equal(v_cache_select, v_cache_ref)
        assert jnp.array_equal(k_cache_select, k_cache_ref)
    rtol = 2 if softcap == 0.0 else 3
    assert max_abs(out - out_ref) <= rtol * max_abs(out_bf16 - out_ref) + 1e-5
    mult_mean = 1.5
    assert mean_abs(out - out_ref) <= mult_mean * mean_abs(out_bf16 - out_ref)
    assert_lse_close(lse, lse_ref, lse_bf16, valid, rtol)


@pytest.mark.skipif(INTERPRET, reason="a race needs the compiled kernels")
@pytest.mark.parametrize("dtype", [jnp.bfloat16])
@pytest.mark.parametrize("d", COMPILED_HDIMS)
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k,causal",
    dense_cases([
        (1, 239),
        (239, 1),
        (3, 799),
        (799, 3),
        (1024, 128),
        (97, 97),
        (128, 128),
        (200, 200),
        (256, 256),
        (257, 257),
        (384, 384),
        (512, 512),
        (768, 768),
        (1024, 1024),
        (2048, 2048),
    ]),
)
def test_flash_attn_race_condition(seqlen_q, seqlen_k, d, causal, dtype):
    rng = np.random.default_rng(0)
    batch_size = 60
    nheads = 4
    q = randn(rng, (batch_size, nheads, seqlen_q, d), dtype)
    k = randn(rng, (batch_size, nheads, seqlen_k, d), dtype)
    v = randn(rng, (batch_size, nheads, seqlen_k, d), dtype)

    @jax.jit
    def fwd_bwd(q, k, v, g):
        out, vjp = jax.vjp(
            lambda q, k, v: flash_attn_func(q, k, v, causal=causal), q, k, v
        )
        return out, *vjp(g)

    g = randn(rng, (batch_size, nheads, seqlen_q, d), dtype)
    out0, dq0, dk0, dv0 = fwd_bwd(q, k, v, g)
    # Note (david): the kernels accumulate in a fixed order, so every repeat on
    # the same inputs must match bitwise; a DMA read before its wait breaks
    # that.
    for _ in range(100):
        out, dq, dk, dv = fwd_bwd(q, k, v, g)
        assert jnp.array_equal(out, out0)
        assert jnp.array_equal(dq, dq0)
        assert jnp.array_equal(dk, dk0)
        assert jnp.array_equal(dv, dv0)
