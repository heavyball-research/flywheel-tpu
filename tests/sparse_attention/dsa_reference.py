"""Host references of DeepSeek sparse attention (DSA): the lightning indexer
and the sparse MLA, on logical tensors.

The indexer quantizes q and k per block to float8_e4m3fn under UE8M0 scales
and scores key s for token t as

    k_scale[s] * sum_h w[t, h] * relu(q_fp8[t, h] . k_fp8[s])

in f32. A token sees the causal prefix of its request, and its top-k are the
indices of its min(INDEX_TOPK, row length) highest scores.

The paged index-K cache keeps tokens along lanes: cache[page, g, i, slot] is
byte i of key dim 4 * g + i for g < KEY_GROUPS, and group SCALE_GROUP holds
the UE8M0 scale in byte 0.

The sparse MLA reference is mla_reference's attention with each token's
selection as its mask.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np

from flywheel_tpu.softmax_attention.mla_decode import (
    FP8_CACHE_DTYPE,
    LATENT_DIM,
    ROPE_DIM,
)
from flywheel_tpu.sparse_attention.dsa.indexer import (
    CACHE_GROUPS,
    FP8_DTYPE,
    INDEX_HEAD_DIM,
    KEY_GROUPS,
    SCALE_GROUP,
    WORD_BYTES,
)
from tests.softmax_attention.mla_reference import (
    QK_DIM,
    SM_SCALE,
    cdiv,
    clamped_randn,
    paged_cache,
)

INDEX_TOPK = 2048
# Index heads of DeepSeek and of GLM.
DEEPSEEK_V32_INDEX_HEADS = 64
GLM52_INDEX_HEADS = 32
FP8_MAX = 448.0
# A block's max abs is floored here before it sets the scale.
AMAX_FLOOR = 1e-4
F32_MANTISSA_BITS = 23
F32_MANTISSA_MASK = (1 << F32_MANTISSA_BITS) - 1
F32_EXPONENT_MASK = 0xFF
# Token rows scored at a time, which bounds the logits' memory.
REFERENCE_ROWS = 64


def ue8m0_scale_exponent(amax: np.ndarray) -> np.ndarray:
    """The biased exponent (uint8) of the UE8M0 scale of blocks of max abs
    amax: 2 ** ceil(log2(max(amax, AMAX_FLOOR) / FP8_MAX)), the ratio
    rounded to f32 first."""
    ratio = np.maximum(amax.astype(np.float32), np.float32(AMAX_FLOOR))
    ratio = ratio * np.float32(1.0 / FP8_MAX)
    bits = ratio.view(np.uint32)
    exponent = (bits >> F32_MANTISSA_BITS) & F32_EXPONENT_MASK
    has_mantissa = (bits & F32_MANTISSA_MASK) != 0
    return (exponent + has_mantissa).astype(np.uint8)


def exponent_to_scale(exponent: np.ndarray) -> np.ndarray:
    """f32 power of two of a UE8M0 biased exponent."""
    return (exponent.astype(np.uint32) << F32_MANTISSA_BITS).view(np.float32)


def ue8m0_quantize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """UE8M0 quantization over the last axis, one INDEX_HEAD_DIM block:
    (x / scale clamped to +-FP8_MAX as FP8_DTYPE, scale f32)."""
    values = np.asarray(x).astype(np.float32)
    scale = exponent_to_scale(ue8m0_scale_exponent(np.abs(values).max(-1)))
    quantized = np.clip(values / scale[..., None], -FP8_MAX, FP8_MAX)
    return quantized.astype(FP8_DTYPE), scale


def index_weights(weights_proj: np.ndarray, q_scale: np.ndarray) -> np.ndarray:
    """The indexer's per-head weights w (T, H) f32 from the weight
    projection's output and q's UE8M0 scales, both (T, H)."""
    num_heads = weights_proj.shape[-1]
    softmax_scale = np.float32(INDEX_HEAD_DIM**-0.5)
    head_scale = np.float32(num_heads**-0.5)
    return (
        weights_proj.astype(np.float32) * head_scale * q_scale * softmax_scale
    )


def chunk_positions(
    cu_q_lens: np.ndarray, kv_lens: np.ndarray
) -> list[tuple[int, int, int]]:
    """Per request, (first token row, first cache position, q_len) of its
    chunk of new tokens at cache positions [kv_len - q_len, kv_len)."""
    cu_q_lens = np.asarray(cu_q_lens)
    return [
        (int(cu_q_lens[r]), int(kv_len) - q_len, q_len)
        for r, kv_len in enumerate(np.asarray(kv_lens).tolist())
        for q_len in (int(cu_q_lens[r + 1] - cu_q_lens[r]),)
    ]


def write_index_chunks(
    cache: np.ndarray,
    k_fp8: np.ndarray,
    k_scale: np.ndarray,
    cu_q_lens: np.ndarray,
    kv_lens: np.ndarray,
    block_table: np.ndarray,
) -> np.ndarray:
    """A copy of the byte cache with each request's chunk tokens, rows of
    k_fp8 (T, INDEX_HEAD_DIM) and k_scale (T,), written at their positions;
    every other byte keeps its value."""
    cache = np.array(cache, copy=True)
    page_size = cache.shape[-1]
    key_bytes = np.asarray(k_fp8).view(np.uint8)
    exponents = (
        np.asarray(k_scale, np.float32).view(np.uint32) >> F32_MANTISSA_BITS
    ).astype(np.uint8)
    block_table = np.asarray(block_table)
    for r, (row, start, q_len) in enumerate(
        chunk_positions(cu_q_lens, kv_lens)
    ):
        for offset in range(q_len):
            position = start + offset
            page = block_table[r, position // page_size]
            slot = position % page_size
            cache[page, :KEY_GROUPS, :, slot] = key_bytes[row + offset].reshape(
                KEY_GROUPS, WORD_BYTES
            )
            cache[page, SCALE_GROUP, :, slot] = 0
            cache[page, SCALE_GROUP, 0, slot] = exponents[row + offset]
    return cache


def gather_index_keys(
    cache: np.ndarray, kv_len: int, page_row: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One request's first kv_len cached keys: k_fp8 (kv_len, INDEX_HEAD_DIM)
    and k_scale (kv_len,) f32, from the pages page_row lists."""
    page_size = cache.shape[-1]
    pages = np.asarray(page_row)[: -(-kv_len // page_size)]
    # (pages, groups, bytes, slots) -> (positions, groups * bytes)
    tokens = np.asarray(cache)[pages].transpose(0, 3, 1, 2)
    tokens = tokens.reshape(-1, CACHE_GROUPS * WORD_BYTES)[:kv_len]
    k_fp8 = tokens[:, :INDEX_HEAD_DIM].copy().view(FP8_DTYPE)
    k_scale = exponent_to_scale(tokens[:, SCALE_GROUP * WORD_BYTES])
    return k_fp8, k_scale


def indexer_row_lengths(
    cu_q_lens: np.ndarray, kv_lens: np.ndarray, num_tokens: int
) -> np.ndarray:
    """Positions each token row sees, (num_tokens,) int32; rows past
    cu_q_lens[-1] are padding and see none."""
    lengths = np.zeros(num_tokens, np.int32)
    for row, start, q_len in chunk_positions(cu_q_lens, kv_lens):
        lengths[row : row + q_len] = start + 1 + np.arange(q_len)
    return lengths


def score_request(
    q_fp8: np.ndarray,
    weights: np.ndarray,
    k_fp8: np.ndarray,
    k_scale: np.ndarray,
) -> np.ndarray:
    """Index scores of one request's rows: q_fp8 (n, H, D), weights (n, H),
    k_fp8 (S, D), k_scale (S,) -> scores (n, S) f32, every key scored."""
    keys = k_fp8.astype(np.float32)
    scores = np.zeros((q_fp8.shape[0], keys.shape[0]), np.float32)
    for first in range(0, q_fp8.shape[0], REFERENCE_ROWS):
        rows = slice(first, first + REFERENCE_ROWS)
        logits = np.einsum("thd,sd->ths", q_fp8[rows].astype(np.float32), keys)
        scores[rows] = np.einsum(
            "ths,th->ts", np.maximum(logits, 0.0), weights[rows]
        )
    return scores * k_scale[None, :]


def indexer_scores(
    q_fp8: np.ndarray,
    weights: np.ndarray,
    cache: np.ndarray,
    cu_q_lens: np.ndarray,
    kv_lens: np.ndarray,
    block_table: np.ndarray,
    num_columns: int,
) -> np.ndarray:
    """Index scores of every token row against its request's cached keys,
    (T, num_columns) f32: each row's causal prefix scored, -inf past it and
    in padding rows."""
    num_tokens = q_fp8.shape[0]
    q_fp8 = np.asarray(q_fp8)
    weights = np.asarray(weights, np.float32)
    scores = np.full((num_tokens, num_columns), -np.inf, np.float32)
    lengths = indexer_row_lengths(cu_q_lens, kv_lens, num_tokens)
    for r, (row, start, q_len) in enumerate(
        chunk_positions(cu_q_lens, kv_lens)
    ):
        if q_len == 0:
            continue
        kv_len = start + q_len
        k_fp8, k_scale = gather_index_keys(
            cache, kv_len, np.asarray(block_table)[r]
        )
        rows = slice(row, row + q_len)
        request_scores = score_request(
            q_fp8[rows], weights[rows], k_fp8, k_scale
        )
        visible = np.arange(kv_len)[None, :] < lengths[rows, None]
        scores[rows, :kv_len] = np.where(visible, request_scores, -np.inf)
    return scores


def indexer_topk(
    scores: np.ndarray, row_lengths: np.ndarray, topk: int = INDEX_TOPK
) -> np.ndarray:
    """(T, topk) int32 indices of each row's min(topk, row length) highest
    scores among its first row-length columns, by descending score with ties
    to the lower index, then -1."""
    scores = np.asarray(scores, np.float32)
    num_tokens, num_columns = scores.shape
    columns = np.arange(num_columns)[None, :]
    masked = np.where(columns < row_lengths[:, None], scores, -np.inf)
    # A stable sort of the negated scores orders ties by index.
    order = np.argsort(-masked, axis=1, kind="stable")[:, :topk]
    indices = np.full((num_tokens, topk), -1, np.int32)
    width = order.shape[1]
    kept = np.arange(width)[None, :] < np.minimum(row_lengths, topk)[:, None]
    indices[:, :width] = np.where(kept, order, -1)
    return indices


# Note (david): the DSA cache keeps no padding lanes, its nope and rope apart.
DSA_ROPE_START = LATENT_DIM
# Tokens a 128-word rope row of the DSA cache holds.
DSA_ROPE_TOKENS_PER_ROW = 8
# An index past every request, which the kernel must not look up.
FAR_POSITION = 1 << 30


@dataclasses.dataclass(frozen=True)
class DsaParam:
    """A ragged batch for DSA sparse MLA: request r has q_lens[r] query tokens
    after prefix_lens[r] cached ones; each token selects up to top_k of the
    positions it sees, in random order."""

    q_lens: tuple[int, ...]
    prefix_lens: tuple[int, ...]
    h_q: int = 128
    page_size: int = 64
    top_k: int = 2048
    padded_tokens: int = 0
    seed: int = 0
    # A few slots name positions past their token, which it must not see.
    has_stray_indices: bool = False
    # A few selected slots hold -1 or positions past their token instead.
    has_dropped_slots: bool = False
    # Slots after the first min(top_k, p + 1) name seen positions, to ignore.
    has_stale_slots: bool = False


@dataclasses.dataclass(frozen=True)
class DsaCase:
    """A DSA batch as host arrays, T tokens packed token-major."""

    q_nope: np.ndarray  # (T, h_q, LATENT_DIM) bf16
    q_pe: np.ndarray  # (T, h_q, ROPE_DIM) bf16
    # (num_pages, page_size, LATENT_DIM + ROPE_DIM) fp8, the logical cache
    # after the batch's tokens are written; NaN past each request's kv_len.
    kv_cache: np.ndarray
    new_kv: np.ndarray  # (T, LATENT_DIM + ROPE_DIM) fp8, each token's latent
    cu_q_lens: np.ndarray  # (b + 1,) i32
    kv_lens: np.ndarray  # (b,) i32
    block_table: np.ndarray  # (b, pages_per_seq) i32
    topk_indices: np.ndarray  # (T, top_k) i32
    kv_scale: np.float32


def make_dsa_case(param: DsaParam) -> DsaCase:
    """A DSA batch of mla_reference's prefill draws quantized to fp8, with
    each token's top-k selection; padding tokens select far positions."""
    assert len(param.q_lens) == len(param.prefix_lens), param
    rng = np.random.default_rng(param.seed)
    q_lens = np.array(param.q_lens, np.int32)
    kv_lens = np.array(param.prefix_lens, np.int32) + q_lens
    cu_q_lens = np.zeros(len(q_lens) + 1, np.int32)
    cu_q_lens[1:] = np.cumsum(q_lens)
    chunk_tokens = int(cu_q_lens[-1])
    num_tokens = chunk_tokens + param.padded_tokens
    q = np.full((num_tokens, param.h_q, QK_DIM), np.nan, jnp.bfloat16)
    q[:chunk_tokens] = clamped_randn(
        rng, (chunk_tokens, param.h_q, QK_DIM), 1.0
    )
    kv_cache, block_table = paged_cache(rng, kv_lens, kv_lens, param.page_size)
    owned_pages = [
        block_table[i, : cdiv(kv_len, param.page_size)]
        for i, kv_len in enumerate(kv_lens.tolist())
    ]
    amax = max(
        float(
            np.nanmax(np.abs(kv_cache[pages].astype(np.float32)), initial=0.0)
        )
        for pages in owned_pages
    )
    kv_scale = np.float32(amax / FP8_MAX)
    latent = np.full((*kv_cache.shape[:2], QK_DIM), np.nan, FP8_CACHE_DTYPE)
    for pages in owned_pages:
        latent[pages] = (
            kv_cache[pages][..., :QK_DIM].astype(np.float32) / kv_scale
        ).astype(FP8_CACHE_DTYPE)
    topk_indices = rng.integers(
        FAR_POSITION,
        2 * FAR_POSITION,
        (num_tokens, param.top_k),
        dtype=np.int32,
    )
    new_kv = np.zeros((num_tokens, QK_DIM), FP8_CACHE_DTYPE)
    for r, (start, end) in enumerate(zip(cu_q_lens[:-1], cu_q_lens[1:])):
        for token in range(start, end):
            position = kv_lens[r] - end + token
            new_kv[token] = latent[
                block_table[r, position // param.page_size],
                position % param.page_size,
            ]
            topk_indices[token] = select_positions(
                rng, position, int(kv_lens[r]), param
            )
    return DsaCase(
        q_nope=q[..., :LATENT_DIM],
        q_pe=q[..., LATENT_DIM:],
        kv_cache=latent,
        new_kv=new_kv,
        cu_q_lens=cu_q_lens,
        kv_lens=kv_lens,
        block_table=block_table,
        topk_indices=topk_indices,
        kv_scale=kv_scale,
    )


def select_positions(
    rng: np.random.Generator, position: int, kv_len: int, param: DsaParam
) -> np.ndarray:
    """(top_k,) i32: a token's selection among [0, position] in its first
    min(top_k, position + 1) slots, then -1 or stale positions."""
    num_seen = position + 1
    if num_seen <= param.top_k:
        chosen = rng.permutation(num_seen)
    else:
        chosen = rng.choice(num_seen, param.top_k, replace=False)
    num_slots = len(chosen)
    slots = np.full(param.top_k, -1, np.int32)
    slots[:num_slots] = chosen
    # A slot past every request, one past its own cache and one past the
    # token.
    strays = [FAR_POSITION, kv_len + 5]
    if position + 1 < kv_len:
        strays.append(position + 1)
    if param.has_stray_indices:
        unused = np.flatnonzero(slots < 0)[: len(strays)]
        slots[unused] = strays[: len(unused)]
    if param.has_dropped_slots:
        drops = [*strays, -1]
        dropped = rng.choice(num_slots, min(len(drops), num_slots), False)
        slots[dropped] = drops[: len(dropped)]
    slots[:num_slots] = rng.permutation(slots[:num_slots])
    if param.has_stale_slots:
        slots[num_slots:] = rng.integers(0, num_seen, param.top_k - num_slots)
    return slots


def dsa_split_cache(kv_cache: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A logical (num_pages, page_size, LATENT_DIM + ROPE_DIM) fp8 cache as
    the split int32 caches dsa_cache_shapes describes."""
    num_pages, page_size, _ = kv_cache.shape
    nope = band_words(kv_cache[..., :DSA_ROPE_START].reshape(-1, LATENT_DIM))
    rope = band_words(kv_cache[..., DSA_ROPE_START:].reshape(-1, ROPE_DIM))
    return (
        nope.reshape(num_pages, page_size, -1),
        rope.reshape(num_pages, page_size // DSA_ROPE_TOKENS_PER_ROW, -1),
    )


def band_words(values: np.ndarray) -> np.ndarray:
    """(n, 4 m) fp8 rows as (n, m) int32 words, byte b of word w holding dim
    m b + w."""
    num_rows, num_dims = values.shape
    as_bytes = values.view(np.uint8).reshape(num_rows, 4, num_dims // 4)
    return np.ascontiguousarray(as_bytes.transpose(0, 2, 1)).view(np.int32)[
        ..., 0
    ]


def sparse_mla_reference(
    case: DsaCase, *, sm_scale: float = SM_SCALE
) -> tuple[np.ndarray, np.ndarray]:
    """attend_request with each token's selection as its mask: the token at
    position p attends the distinct positions in [0, p] its first
    min(top_k, p + 1) indices name. A padding token, or one that selects
    nothing, gets out 0 and lse +inf.

    Returns out (T, h_q, LATENT_DIM) f32 and lse (h_q, T) f32 natural log.
    """
    num_tokens, h_q, _ = case.q_nope.shape
    page_size = case.kv_cache.shape[1]
    q = np.concatenate([case.q_nope, case.q_pe], axis=-1).astype(np.float32)
    out = np.zeros((num_tokens, h_q, LATENT_DIM), np.float32)
    lse = np.full((h_q, num_tokens), np.inf, np.float32)
    for r, (start, end) in enumerate(
        zip(case.cu_q_lens[:-1].tolist(), case.cu_q_lens[1:].tolist())
    ):
        for token in range(start, end):
            position = case.kv_lens[r] - end + token
            index = case.topk_indices[token, : position + 1]
            selected = np.unique(index[(index >= 0) & (index <= position)])
            if len(selected) == 0:
                continue
            pages = case.block_table[r, selected // page_size]
            kv = case.kv_cache[pages, selected % page_size].astype(np.float32)
            kv *= case.kv_scale
            scores = (q[token] @ kv.T) * np.float32(sm_scale)
            peak = scores.max(axis=-1, keepdims=True)
            row_lse = peak + np.log(
                np.exp(scores - peak).sum(axis=-1, keepdims=True)
            )
            out[token] = np.exp(scores - row_lse) @ kv[:, :LATENT_DIM]
            lse[:, token] = row_lse[:, 0]
    return out, lse
