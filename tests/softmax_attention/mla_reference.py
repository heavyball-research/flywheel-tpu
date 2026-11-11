"""FlashMLA's dense decode test ported to flash_mla_decode: its random cases,
its f32 reference and its closeness check, plus a bitwise check of the cache
the fused append leaves.

The port keeps FlashMLA's distributions, NaN traps, tolerances and anomaly
rules, and changes only what flash_mla_decode's contract changes: a cache row
is the latent, the rope and zero padding lanes; q and the new tokens arrive
split into latent and rope; the kernel appends each request's s_q new tokens
itself, so the cache a case hands over holds kv_lens - s_q valid tokens and
NaN rows where the new tokens go. An fp8 case quantizes the cache and the new
tokens, latent and rope alike, to float8_e4m3fn under one per-tensor
kv_scale, where FlashMLA's fp8 cache scales each lane tile of the latent and
keeps the rope bf16; its reference attends the dequantized values, as
FlashMLA's does.

The prefill cases and reference carry the same draws, traps and attention
over to a ragged chunk: each request's new tokens, packed token-major, follow
its own cached prefix, and causal is bottom-right.
"""

import dataclasses
import math
import random

import jax
import jax.numpy as jnp
import numpy as np

from flywheel_tpu.softmax_attention.mla_decode import (
    CACHE_ROW_DIM,
    FP8_CACHE_DTYPE,
    LATENT_DIM,
    ROPE_DIM,
    mla_cache_shape,
)

QK_DIM = LATENT_DIM + ROPE_DIM
# Note (david): FlashMLA scales the scores by 1 / sqrt(d), d the q / k head
# dim.
SM_SCALE = 1 / math.sqrt(QK_DIM)
# FlashMLA's block_table entry for the pages past a request's last one.
INVALID_PAGE = 2147480000
# FlashMLA rounds the longest request up to a multiple of SEQLEN_ALIGN tokens
# before it sizes every request's pages.
SEQLEN_ALIGN = 256
# Note (david): the reference attends a block of query rows at a time, which
# bounds its scores' memory on long chunks.
REFERENCE_ROWS = 256
# The largest finite fp8 cache value, onto which an fp8 case's kv_scale maps
# the data's max abs.
FP8_MAX = float(jnp.finfo(FP8_CACHE_DTYPE).max)


def cdiv(x, y):
    return (x + y - 1) // y


@dataclasses.dataclass(frozen=True)
class MlaParam:
    """FlashMLA's TestParam for dense decode; block_size is page_size, and
    one KV head of the latent and the rope is what flash_mla_decode fixes."""

    b: int
    s_q: int
    # Every request's kv length after the append, the mean one when is_varlen.
    s_k: int
    is_varlen: bool
    is_causal: bool
    is_fp8: bool = False
    have_zero_seqlen_k: bool = False
    page_size: int = 64
    h_q: int = 128
    seed: int = 0
    # Note (david): FlashMLA draws every length; a cell that must put the new
    # tokens at a given position pins each request's kv length here instead,
    # with b its length, s_k its max, and neither is_varlen nor
    # have_zero_seqlen_k.
    kv_lens: tuple[int, ...] | None = None


@dataclasses.dataclass(frozen=True)
class MlaCase:
    """flash_mla_decode's positional operands as host arrays, in its order,
    then its kv_scale."""

    q_nope: np.ndarray  # (b, s_q, h_q, LATENT_DIM) bf16
    q_pe: np.ndarray  # (b, s_q, h_q, ROPE_DIM) bf16
    new_kv_c: np.ndarray  # (b, s_q, LATENT_DIM) in the cache's dtype
    new_k_pe: np.ndarray  # (b, s_q, ROPE_DIM) in the cache's dtype
    # (num_pages, page_size, CACHE_ROW_DIM) bf16 or fp8, the logical cache;
    # operands hands flash_mla_decode its word layout (mla_cache_shape).
    kv_cache: np.ndarray
    kv_lens: np.ndarray  # (b,) i32
    block_table: np.ndarray  # (b, pages_per_seq) i32
    # An fp8 cache's f32 scale, a real value being the fp8 value times
    # kv_scale; None for a bf16 cache.
    kv_scale: np.float32 | None = None

    def operands(self) -> tuple[jax.Array, ...]:
        """The positional operands as device arrays, ready for
        flash_mla_decode(*..., kv_scale=kv_scale), the cache in its word
        layout."""
        num_pages, page_size, _ = self.kv_cache.shape
        kv_cache = self.kv_cache.reshape(
            mla_cache_shape(num_pages, page_size, self.kv_cache.dtype)
        )
        return tuple(
            jnp.asarray(value)
            for value in (
                self.q_nope,
                self.q_pe,
                self.new_kv_c,
                self.new_k_pe,
                kv_cache,
                self.kv_lens,
                self.block_table,
            )
        )


@dataclasses.dataclass(frozen=True)
class MlaPrefillParam:
    """A ragged prefill chunk: request r appends q_lens[r] tokens after its
    prefix_lens[r] cached ones, and padded_tokens rows follow the chunk."""

    q_lens: tuple[int, ...]
    prefix_lens: tuple[int, ...]
    h_q: int = 16
    page_size: int = 64
    is_fp8: bool = False
    seed: int = 0
    padded_tokens: int = 0


@dataclasses.dataclass(frozen=True)
class MlaPrefillCase:
    """A prefill chunk's operands as host arrays, T tokens packed token-major:
    request r's are rows [cu_q_lens[r], cu_q_lens[r + 1]), the rest padding."""

    q_nope: np.ndarray  # (T, h_q, LATENT_DIM) bf16
    q_pe: np.ndarray  # (T, h_q, ROPE_DIM) bf16
    new_kv_c: np.ndarray  # (T, LATENT_DIM) in the cache's dtype
    new_k_pe: np.ndarray  # (T, ROPE_DIM) in the cache's dtype
    # (num_pages, page_size, CACHE_ROW_DIM) bf16 or fp8, the logical cache
    # before the chunk is written.
    kv_cache: np.ndarray
    cu_q_lens: np.ndarray  # (b + 1,) i32
    kv_lens: np.ndarray  # (b,) i32, the prefix and the chunk
    block_table: np.ndarray  # (b, pages_per_seq) i32
    kv_scale: np.float32 | None = None  # an fp8 cache's f32 scale


def logical_cache(kv_cache) -> np.ndarray:
    """A word-layout cache (mla_cache_shape) as the logical (num_pages,
    page_size, CACHE_ROW_DIM) host array."""
    kv_cache = np.asarray(kv_cache)
    num_pages, words, packing, row_dim = kv_cache.shape
    return kv_cache.reshape(num_pages, words * packing, row_dim)


def clamped_randn(rng, shape, scale):
    """FlashMLA's (randn * scale).clamp(-1, 1), drawn in f32."""
    x = rng.standard_normal(shape, dtype=np.float32)
    x *= np.float32(scale)
    return np.clip(x, -1.0, 1.0, out=x)


def make_case(param: MlaParam) -> MlaCase:
    """FlashMLA's generate_test_data for param, laid out for flash_mla_decode.

    The kv lengths are s_k, or with is_varlen max(normalvariate(s_k, s_k / 2),
    s_q) from Python's random seeded with param.seed, the same draws as
    FlashMLA's; have_zero_seqlen_k then zeroes the length of each request whose
    standard normal draw is positive, making it inactive. A param that pins
    kv_lens gets them as they are. q is randn clamped to [-1, 1]; cache rows
    and new tokens are randn / 10 clamped to [-1, 1], with zero padding lanes.
    Every request owns the same number of pages, sized from the longest one
    rounded up to SEQLEN_ALIGN tokens, through a random permutation of all
    pages. Its rows past kv_lens - s_q are NaN in every lane: the new tokens'
    rows, the tail of its last page and its pages past cdiv(kv_lens,
    page_size), whose block_table entries are INVALID_PAGE. With is_fp8, the
    bf16 cache and new tokens drawn so, the bf16 case's own, are then
    quantized to FP8_CACHE_DTYPE (quantize_case), as FlashMLA quantizes its
    bf16 cache.
    """
    bf16 = jnp.bfloat16
    rng = np.random.default_rng(param.seed)
    kv_lens = draw_kv_lens(param, rng)
    q = clamped_randn(rng, (param.b, param.s_q, param.h_q, QK_DIM), 1.0)
    kv_cache, block_table = paged_cache(
        rng, kv_lens, kv_lens - param.s_q, param.page_size
    )
    new_kv = clamped_randn(rng, (param.b, param.s_q, QK_DIM), 0.1).astype(bf16)
    q = q.astype(bf16)
    case = MlaCase(
        q_nope=q[..., :LATENT_DIM],
        q_pe=q[..., LATENT_DIM:],
        new_kv_c=new_kv[..., :LATENT_DIM],
        new_k_pe=new_kv[..., LATENT_DIM:],
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        block_table=block_table,
    )
    if param.is_fp8:
        return quantize_case(case)
    else:
        return case


def draw_kv_lens(param: MlaParam, rng: np.random.Generator) -> np.ndarray:
    """make_case's kv lengths for param, (b,) i32; have_zero_seqlen_k draws
    from rng."""
    kv_lens = np.full(param.b, param.s_k, np.int32)
    if param.kv_lens is not None:
        kv_lens[:] = param.kv_lens
    if param.is_varlen:
        lengths = random.Random(param.seed)
        for i in range(param.b):
            kv_lens[i] = int(
                max(lengths.normalvariate(param.s_k, param.s_k / 2), param.s_q)
            )
    if param.have_zero_seqlen_k:
        kv_lens[rng.standard_normal(param.b) > 0] = 0
    return kv_lens


def make_prefill_case(param: MlaPrefillParam) -> MlaPrefillCase:
    """make_case's draws for a ragged chunk, in packed order: request r's
    cache holds prefix_lens[r] valid rows, then NaN where its chunk goes, and
    the padded_tokens rows after the chunk are NaN in q and the new tokens."""
    assert len(param.q_lens) == len(param.prefix_lens), param
    bf16 = jnp.bfloat16
    rng = np.random.default_rng(param.seed)
    q_lens = np.array(param.q_lens, np.int32)
    prefix_lens = np.array(param.prefix_lens, np.int32)
    cu_q_lens = np.zeros(len(q_lens) + 1, np.int32)
    cu_q_lens[1:] = np.cumsum(q_lens)
    chunk_tokens = int(cu_q_lens[-1])
    num_tokens = chunk_tokens + param.padded_tokens

    q = np.full((num_tokens, param.h_q, QK_DIM), np.nan, bf16)
    q[:chunk_tokens] = clamped_randn(
        rng, (chunk_tokens, param.h_q, QK_DIM), 1.0
    )
    kv_lens = prefix_lens + q_lens
    kv_cache, block_table = paged_cache(
        rng, kv_lens, prefix_lens, param.page_size
    )
    new_kv = np.full((num_tokens, QK_DIM), np.nan, bf16)
    new_kv[:chunk_tokens] = clamped_randn(rng, (chunk_tokens, QK_DIM), 0.1)
    case = MlaPrefillCase(
        q_nope=q[..., :LATENT_DIM],
        q_pe=q[..., LATENT_DIM:],
        new_kv_c=new_kv[..., :LATENT_DIM],
        new_k_pe=new_kv[..., LATENT_DIM:],
        kv_cache=kv_cache,
        cu_q_lens=cu_q_lens,
        kv_lens=kv_lens,
        block_table=block_table,
    )
    if param.is_fp8:
        return quantize_case(case)
    else:
        return case


def paged_cache(
    rng: np.random.Generator,
    kv_lens: np.ndarray,
    prefix_lens: np.ndarray,
    page_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """make_case's bf16 cache and block_table for requests of kv_lens tokens,
    each with prefix_lens valid rows and NaN past them."""
    bf16 = jnp.bfloat16
    num_requests = len(kv_lens)
    num_pages = num_requests * pages_per_seq(int(kv_lens.max()), page_size)
    block_table = (
        rng.permutation(num_pages).astype(np.int32).reshape(num_requests, -1)
    )
    kv_cache = np.full((num_pages, page_size, CACHE_ROW_DIM), np.nan, bf16)
    for i, (kv_len, prefix_len) in enumerate(
        zip(kv_lens.tolist(), prefix_lens.tolist())
    ):
        num_owned = cdiv(kv_len, page_size)
        rows = np.zeros((num_owned, page_size, CACHE_ROW_DIM), bf16)
        rows[..., :QK_DIM] = clamped_randn(
            rng, (num_owned, page_size, QK_DIM), 0.1
        )
        rows.reshape(-1, CACHE_ROW_DIM)[prefix_len:] = np.nan
        kv_cache[block_table[i, :num_owned]] = rows
        block_table[i, num_owned:] = INVALID_PAGE
    return kv_cache, block_table


def pages_per_seq(max_kv_len: int, page_size: int) -> int:
    """The pages make_case gives every request when the longest holds
    max_kv_len tokens."""
    # Note (david): FlashMLA's SEQLEN_ALIGN rounding can leave a page size that
    # does not divide it with a fraction of a page, so round to both; an all
    # inactive batch still gets one page per request.
    align = math.lcm(SEQLEN_ALIGN, page_size)
    return max(cdiv(max_kv_len, align), 1) * align // page_size


def quantize_case[CaseT: (MlaCase, MlaPrefillCase)](case: CaseT) -> CaseT:
    """case, a bf16 one, with its cache and new tokens quantized to
    FP8_CACHE_DTYPE under one per-tensor kv_scale.

    kv_scale is amax / FP8_MAX, amax the max abs over the cache's valid rows
    and the real new tokens, so the largest value quantizes to FP8_MAX and
    none overflows; a value quantizes to round(value / kv_scale), and NaN rows
    stay NaN.
    """
    page_size = case.kv_cache.shape[1]
    owned_pages = [
        case.block_table[i, : cdiv(kv_len, page_size)]
        for i, kv_len in enumerate(case.kv_lens.tolist())
    ]
    amax = max(
        float(np.nanmax(np.abs(values.astype(np.float32)), initial=0.0))
        for values in (
            case.new_kv_c,
            case.new_k_pe,
            *(case.kv_cache[pages] for pages in owned_pages),
        )
    )
    kv_scale = np.float32(amax / FP8_MAX)

    def quantize(x):
        return (x.astype(np.float32) / kv_scale).astype(FP8_CACHE_DTYPE)

    kv_cache = np.full(case.kv_cache.shape, np.nan, FP8_CACHE_DTYPE)
    for pages in owned_pages:
        kv_cache[pages] = quantize(case.kv_cache[pages])
    return dataclasses.replace(
        case,
        new_kv_c=quantize(case.new_kv_c),
        new_k_pe=quantize(case.new_k_pe),
        kv_cache=kv_cache,
        kv_scale=kv_scale,
    )


def attend_request(
    q: np.ndarray,
    kv_cache: np.ndarray,
    page_row: np.ndarray,
    kv_len: int,
    kv_scale: np.float32 | None,
    *,
    causal: bool,
    sm_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """FlashMLA's reference_torch for one request: its queries q, (n, h_q,
    QK_DIM) f32 at its last n positions, attend its first kv_len rows of
    kv_cache on the pages page_row lists.

    The rows attend in f32 with NaN zeroed: scores sm_scale * q . (latent,
    rope), with causal a tril(diagonal = kv_len - n) mask, V the latent. An
    fp8 cache's rows are dequantized first, to their f32 value times
    kv_scale, so that quantization error is not mistaken for a kernel bug. A
    query that sees no key gets out 0 and lse +inf.

    Returns out (n, h_q, LATENT_DIM) f32 and lse (h_q, n) f32 natural log.
    """
    num_queries, h_q, _ = q.shape
    pages = page_row[: cdiv(kv_len, kv_cache.shape[1])]
    rows = kv_cache[pages].reshape(-1, CACHE_ROW_DIM)[:kv_len, :QK_DIM]
    # Note (david): dequantized rows stay f32, so the reference adds no
    # rounding of its own.
    if kv_scale is None:
        kv = rows.astype(np.float32)
    else:
        kv = rows.astype(np.float32) * kv_scale
    kv[np.isnan(kv)] = 0.0
    out = np.zeros((num_queries, h_q, LATENT_DIM), np.float32)
    lse = np.zeros((h_q, num_queries), np.float32)
    for first in range(0, num_queries, REFERENCE_ROWS):
        block = slice(first, min(first + REFERENCE_ROWS, num_queries))
        num_rows = block.stop - first
        scores = (q[block].reshape(-1, QK_DIM) @ kv.T).reshape(
            num_rows, h_q, kv_len
        )
        if causal:
            visible = (
                np.arange(kv_len)[None, :]
                <= (kv_len - num_queries + np.arange(first, block.stop))[
                    :, None
                ]
            )
        else:
            visible = np.ones((num_rows, kv_len), bool)
        scores = np.where(visible[:, None], scores, -np.inf)
        scores *= np.float32(sm_scale)
        peak = scores.max(axis=-1, keepdims=True, initial=-np.inf)
        peak = np.where(np.isfinite(peak), peak, 0.0)
        with np.errstate(divide="ignore"):
            row_lse = peak + np.log(
                np.exp(scores - peak).sum(axis=-1, keepdims=True)
            )
        lonely = row_lse == -np.inf
        probs = np.exp(scores - np.where(lonely, 0.0, row_lse))
        row_out = probs.reshape(num_rows * h_q, kv_len) @ kv[:, :LATENT_DIM]
        out[block] = np.where(
            lonely, 0.0, row_out.reshape(num_rows, h_q, LATENT_DIM)
        )
        lse[:, block] = np.where(lonely, np.inf, row_lse)[..., 0].T
    return out, lse


def reference(
    case: MlaCase, *, causal: bool, sm_scale: float = SM_SCALE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FlashMLA's reference_torch on flash_mla_decode's operands.

    Writes each active request's new tokens as (latent, rope, zero padding)
    rows at [kv_lens - s_q, kv_lens), then attends each request's kv_lens rows
    as attend_request does; every query of an inactive request sees no key.

    Returns (out, lse, kv_cache): out (b, s_q, h_q, LATENT_DIM) bf16, lse (b,
    h_q, s_q) f32 natural log, kv_cache the cache flash_mla_decode must
    return.
    """
    kv_cache = case.kv_cache.copy()
    b, s_q, h_q, _ = case.q_nope.shape
    page_size = kv_cache.shape[1]
    new_rows = np.zeros((b, s_q, CACHE_ROW_DIM), kv_cache.dtype)
    new_rows[..., :LATENT_DIM] = case.new_kv_c
    new_rows[..., LATENT_DIM:QK_DIM] = case.new_k_pe
    for i, kv_len in enumerate(case.kv_lens.tolist()):
        if kv_len > 0:
            positions = np.arange(kv_len - s_q, kv_len)
            pages = case.block_table[i, positions // page_size]
            kv_cache[pages, positions % page_size] = new_rows[i]
    q = np.concatenate([case.q_nope, case.q_pe], axis=-1).astype(np.float32)
    out = np.zeros((b, s_q, h_q, LATENT_DIM), np.float32)
    lse = np.zeros((b, h_q, s_q), np.float32)
    for i, kv_len in enumerate(case.kv_lens.tolist()):
        out[i], lse[i] = attend_request(
            q[i],
            kv_cache,
            case.block_table[i],
            kv_len,
            case.kv_scale,
            causal=causal,
            sm_scale=sm_scale,
        )
    return out.astype(jnp.bfloat16), lse, kv_cache


def write_chunks(case: MlaPrefillCase) -> np.ndarray:
    """The logical cache with each request's chunk tokens written as (latent,
    rope, zero padding) rows at [kv_lens - q_len, kv_lens)."""
    kv_cache = case.kv_cache.copy()
    num_tokens = case.q_nope.shape[0]
    page_size = kv_cache.shape[1]
    new_rows = np.zeros((num_tokens, CACHE_ROW_DIM), kv_cache.dtype)
    new_rows[:, :LATENT_DIM] = case.new_kv_c
    new_rows[:, LATENT_DIM:QK_DIM] = case.new_k_pe
    for i, (start, end, kv_len) in enumerate(
        zip(
            case.cu_q_lens[:-1].tolist(),
            case.cu_q_lens[1:].tolist(),
            case.kv_lens.tolist(),
        )
    ):
        positions = np.arange(kv_len - (end - start), kv_len)
        pages = case.block_table[i, positions // page_size]
        kv_cache[pages, positions % page_size] = new_rows[start:end]
    return kv_cache


def prefill_reference(
    case: MlaPrefillCase, *, causal: bool, sm_scale: float = SM_SCALE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """reference() for a ragged chunk.

    Attends each request's kv_lens rows of the cache write_chunks leaves as
    attend_request does, so that with causal chunk token i sees positions
    [0, kv_lens - q_len + i]. Padding rows get out 0 and lse +inf.

    Returns (out, lse, kv_cache): out (T, h_q, LATENT_DIM) f32; lse (h_q, T)
    f32 natural log; kv_cache the logical cache after the write.
    """
    kv_cache = write_chunks(case)
    num_tokens, h_q, _ = case.q_nope.shape
    chunks = list(
        zip(
            case.cu_q_lens[:-1].tolist(),
            case.cu_q_lens[1:].tolist(),
            case.kv_lens.tolist(),
        )
    )
    q = np.concatenate([case.q_nope, case.q_pe], axis=-1).astype(np.float32)
    out = np.zeros((num_tokens, h_q, LATENT_DIM), np.float32)
    lse = np.full((h_q, num_tokens), np.inf, np.float32)
    for i, (start, end, kv_len) in enumerate(chunks):
        out[start:end], lse[:, start:end] = attend_request(
            q[start:end],
            kv_cache,
            case.block_table[i],
            kv_len,
            case.kv_scale,
            causal=causal,
            sm_scale=sm_scale,
        )
    return out, lse, kv_cache


def check_is_allclose(
    name: str,
    ans,
    ref,
    abs_tol: float = 1e-5,
    rel_tol: float = 1e-2,
    cos_diff_tol: float = 1e-7,
) -> bool:
    """FlashMLA's check_is_allclose: prints why ans and ref differ and returns
    False, or returns True.

    +inf, -inf and NaN must sit at the same positions, which then count as 0.
    Every other element passes when its abs error is under abs_tol or its
    error relative to |ref| + 1e-6 is under rel_tol, and the whole tensor
    needs 1 - 2 sum(ans * ref) / sum(ans^2 + ref^2) under cos_diff_tol.
    """
    ans = np.asarray(ans).astype(np.float32)
    ref = np.asarray(ref).astype(np.float32)
    assert ans.shape == ref.shape, f"{name} shape: {ans.shape} vs {ref.shape}"

    def deal_with_anomalies(is_anomaly, label):
        ref_mask, ans_mask = is_anomaly(ref), is_anomaly(ans)
        ref[ref_mask] = 0.0
        ans[ans_mask] = 0.0
        if np.array_equal(ref_mask, ans_mask):
            return True
        else:
            print(
                f"{name} anomaly {label} mismatch: {ans_mask.sum()} in ans but"
                f" {ref_mask.sum()} in ref"
            )
            return False

    anomalies_check_passed = True
    for is_anomaly, label in (
        (np.isposinf, "inf"),
        (np.isneginf, "-inf"),
        (np.isnan, "nan"),
    ):
        anomalies_check_passed &= deal_with_anomalies(is_anomaly, label)
    if not anomalies_check_passed:
        return False

    x, y = ans.astype(np.float64), ref.astype(np.float64)
    denominator = (x * x + y * y).sum()
    cos_diff = 0.0 if denominator == 0 else 1 - 2 * (x * y).sum() / denominator
    raw_abs_err = np.abs(ans - ref)
    raw_rel_err = raw_abs_err / (np.abs(ref) + 1e-6)
    rel_err = np.where(raw_abs_err < abs_tol, 0.0, raw_rel_err)
    abs_err = np.where(raw_rel_err < rel_tol, 0.0, raw_abs_err)
    pass_mask = (abs_err < abs_tol) | (rel_err < rel_tol)

    if not pass_mask.all():
        print(f"{name} mismatch")
        for label, err in (("abs", abs_err), ("rel", rel_err)):
            pos = int(np.argmax(err))
            print(
                f"max {label} err: {err.max()}: pos"
                f" {[int(i) for i in np.unravel_index(pos, ans.shape)]},"
                f" {ans.reshape(-1)[pos]} vs {ref.reshape(-1)[pos]}"
            )
        print(
            f"{pass_mask.sum()} out of {pass_mask.size} passed"
            f" ({pass_mask.mean() * 100.0:.2f}%)"
        )
        print(f"Cosine diff: {cos_diff} (threshold: {cos_diff_tol})")
        return False
    elif abs(cos_diff) > cos_diff_tol:
        print(
            f"{name} mismatch: cosine diff too large: {cos_diff} vs"
            f" {cos_diff_tol}"
        )
        return False
    else:
        return True


def check_kv_cache(name: str, ans, ref) -> bool:
    """Whether ans holds ref's bits in every lane, where any NaN matches any
    NaN; prints the first (page, row) pairs that differ and returns False
    otherwise."""
    ans, ref = np.asarray(ans), np.asarray(ref)
    assert (
        ans.shape == ref.shape and ans.dtype == ref.dtype
    ), f"{name}: {ans.shape} {ans.dtype} vs {ref.shape} {ref.dtype}"
    bits = np.dtype(f"u{ans.dtype.itemsize}")
    differ = (ans.view(bits) != ref.view(bits)).any(axis=-1)
    # Note (david): NaN is tested on the rows whose bits differ only, as isnan
    # over a whole cache is slow.
    if differ.any():
        a, r = ans[differ], ref[differ]
        same = (a.view(bits) == r.view(bits)) | (np.isnan(a) & np.isnan(r))
        differ[differ] = ~same.all(axis=-1)
    bad_rows = np.argwhere(differ)
    if len(bad_rows):
        print(
            f"{name} mismatch: {len(bad_rows)} rows differ, the first"
            f" (page, row): {bad_rows[:8].tolist()}"
        )
        return False
    else:
        return True
