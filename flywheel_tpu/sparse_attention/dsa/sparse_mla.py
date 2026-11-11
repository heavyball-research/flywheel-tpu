"""DSA sparse MLA: each query token attends its own top-k cached latents.

The token axis runs in chunks of one size: the SparseCore gathers chunk
i + 1's rows from the split cache while the TensorCore attends chunk i.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from flywheel_tpu.core.tpu import NUM_LANES, NUM_SUBLANES
from flywheel_tpu.softmax_attention.kernels.mla_decode import (
    LATENT_DIM,
    ROPE_DIM,
)
from flywheel_tpu.sparse_attention.dsa.cache_write import (
    check_caches,
    check_requests,
    token_positions,
)
from flywheel_tpu.sparse_attention.dsa.kernels.gather import (
    CHUNK_FIRST_INDEX_ROW,
    CHUNK_FIRST_TOKEN,
    CHUNK_SLOTS,
    CHUNK_WORDS,
    META_POSITION,
    META_SLOTS,
    META_TABLE_OFFSET,
    META_WORDS,
    NOPE_WORDS,
    ROPE_WORDS,
    UNUSED_INDEX,
    dsa_gather_pallas,
    dsa_topk_gather_pallas,
    gather_slots,
)
from flywheel_tpu.sparse_attention.dsa.kernels.sparse_mla import (
    BLOCK_SLOTS,
    attention_scalars,
    dsa_sparse_mla_pallas,
    max_blocked_slots,
    tokens_per_step,
)
from flywheel_tpu.topk.kernels.sparse_core_topk import TopkGeometry

QUERY_DTYPE = jnp.dtype(jnp.bfloat16)
MAX_HEADS = NUM_LANES
# Note (david): a call cuts its tokens into about TARGET_CHUNKS chunks, so each
# gather runs under another chunk's attention; MIN_CHUNK_TOKENS keeps per-call
# cost from outweighing that overlap, MAX_CHUNK_TOKENS bounds the gathered HBM.
TARGET_CHUNKS = 8
MIN_CHUNK_TOKENS = 32
MAX_CHUNK_TOKENS = 256


def default_chunk_tokens(num_tokens: int) -> int:
    """Tokens a chunk of a call of num_tokens tokens holds: a whole number of
    attention steps."""
    step_tokens = tokens_per_step(num_tokens)
    target = min(
        MAX_CHUNK_TOKENS,
        max(MIN_CHUNK_TOKENS, pl.cdiv(num_tokens, TARGET_CHUNKS)),
    )
    return pl.cdiv(target, step_tokens) * step_tokens


def chunk_starts(num_tokens: int, chunk_tokens: int) -> list[int]:
    """First tokens of chunks of chunk_tokens tokens that cover num_tokens:
    consecutive, the last one moved back to end at num_tokens, so that every
    chunk has one size and shares one compile; its tokens that the chunk
    before it holds too are computed twice, to the same values."""
    return [
        min(first, num_tokens - chunk_tokens)
        for first in range(0, num_tokens, chunk_tokens)
    ]


def token_metadata(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    num_tokens: int,
    pages_per_seq: int,
    top_k: int,
) -> tuple[jax.Array, jax.Array]:
    """The (T, META_WORDS) int32 rows the gather reads per token, and the
    (T / G,) int32 slots each attention step of G tokens attends.

    A token's row holds its request's block-table offset, the last position
    it may attend (-1 for padding) and its step's slots: the first min(top_k,
    position + 1) slots of each of the step's tokens up to whole
    BLOCK_SLOTS, top_k past max_blocked_slots(top_k), 0 for a step of
    padding. A pure function of cu_q_lens and kv_lens, which layers share.
    """
    request, position = token_positions(cu_q_lens, kv_lens, num_tokens)
    step_tokens = tokens_per_step(num_tokens)
    used = jnp.clip(position + 1, 0, top_k).reshape(-1, step_tokens).max(1)
    blocked = (used + BLOCK_SLOTS - 1) // BLOCK_SLOTS * BLOCK_SLOTS
    step_slots = jnp.where(blocked > max_blocked_slots(top_k), top_k, blocked)
    columns = {
        META_TABLE_OFFSET: request * pages_per_seq,
        META_POSITION: position,
        META_SLOTS: jnp.repeat(step_slots, step_tokens),
    }
    zeros = jnp.zeros_like(position)
    token_meta = jnp.stack(
        [columns.get(lane, zeros) for lane in range(META_WORDS)], axis=1
    )
    return token_meta, step_slots


def chunk_row(first_index_row: int, first_token: int) -> jax.Array:
    """The gather's (CHUNK_WORDS,) int32 chunk row, a constant."""
    row = np.zeros(CHUNK_WORDS, np.int32)
    row[CHUNK_FIRST_INDEX_ROW] = first_index_row
    row[CHUNK_FIRST_TOKEN] = first_token
    return jnp.asarray(row)


class SplitCache(NamedTuple):
    """The gather's view of the split cache: keyword operands of
    dsa_gather_pallas and dsa_topk_gather_pallas, top_k the slots a token
    takes, gather_slots of the call's top_k."""

    block_table: jax.Array
    nope_cache: jax.Array
    rope_cache: jax.Array
    top_k: int
    page_size: int
    pages_per_seq: int


class ChunkAttention(NamedTuple):
    """The operands every chunk's gather and attention share."""

    q_nope: jax.Array
    q_pe: jax.Array
    token_meta: jax.Array
    scalars: jax.Array
    cu_q_lens: jax.Array
    cache: SplitCache
    sm_scale: float
    return_lse: bool

    def gather_rows(
        self, row: jax.Array, indices: jax.Array, num_tokens: int
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """The SparseCore's (nope_rows, rope_rows, flag_rows) of num_tokens
        tokens, their first index row and token in the gather's chunk
        row."""
        return dsa_gather_pallas(
            row,
            indices,
            self.token_meta,
            num_tokens=num_tokens,
            **self.cache._asdict(),
        )

    def select_and_gather_rows(
        self,
        row: jax.Array,
        num_tokens: int,
        row_lengths: jax.Array,
        scores: jax.Array,
        geometry: TopkGeometry,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
        """The top-k of num_tokens tokens' checked scores and row_lengths,
        their first token in the gather's chunk row, and in the same
        SparseCore call gather_rows's rows of their selection; returns the
        rows and the (num_tokens, geometry.k) indices."""
        *rows, indices = dsa_topk_gather_pallas(
            row,
            self.token_meta,
            row_lengths=row_lengths,
            scores=scores,
            num_tokens=num_tokens,
            geometry=geometry,
            **self.cache._asdict(),
        )
        return tuple(rows), indices

    def attend_rows(
        self,
        rows: tuple[jax.Array, jax.Array, jax.Array],
        previous: tuple[jax.Array, ...] | None,
        *,
        first_token: int,
        num_tokens: int,
        scalars: jax.Array,
    ) -> tuple[jax.Array, ...]:
        """Attend the call's tokens [first_token, first_token + num_tokens)
        over their gathered rows, with scalars the call's scalars, which a
        caller may pass through a barrier; returns (out, [lse]) with their
        rows written into previous's."""
        return dsa_sparse_mla_pallas(
            self.q_nope,
            self.q_pe,
            *rows,
            scalars,
            previous,
            cu_q_lens=self.cu_q_lens,
            first_token=jnp.full((1,), first_token, jnp.int32),
            num_chunk_tokens=num_tokens,
            sm_scale=self.sm_scale,
            return_lse=self.return_lse,
        )


def chunk_attention(
    q_nope: jax.Array,
    q_pe: jax.Array,
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
    *,
    top_k: int,
    sm_scale: float,
    kv_scale: jax.Array,
    return_lse: bool,
) -> ChunkAttention:
    """The shared operands of a call's chunks on checked operands, its
    queries padded to whole attention steps and each token's top_k slots
    to gather_slots(top_k)."""
    num_tokens = q_nope.shape[0]
    step_tokens = tokens_per_step(num_tokens)
    num_slots = gather_slots(top_k)
    page_size = nope_cache.shape[1]
    # Note (david): the attention steps over whole groups of tokens, so the
    # padded rows join with no position and a zero output.
    padded_tokens = pl.cdiv(num_tokens, step_tokens) * step_tokens
    token_pad = ((0, padded_tokens - num_tokens), (0, 0), (0, 0))
    # Note (david): the SparseCore DMAs whole 32-byte granules, so a table
    # row starts on one.
    pages_per_seq = pl.cdiv(block_table.shape[1], NUM_SUBLANES) * NUM_SUBLANES
    table = jnp.pad(
        block_table, ((0, 0), (0, pages_per_seq - block_table.shape[1]))
    ).reshape(-1)
    cache = SplitCache(
        block_table=table,
        nope_cache=nope_cache.reshape(-1, NOPE_WORDS),
        rope_cache=rope_cache.reshape(-1, ROPE_WORDS),
        top_k=num_slots,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
    )
    token_meta, step_slots = token_metadata(
        cu_q_lens, kv_lens, padded_tokens, pages_per_seq, num_slots
    )
    return ChunkAttention(
        q_nope=jnp.pad(q_nope, token_pad),
        # Note (david): the kernel reads q_pe with the heads minor, as XLA
        # lays (T, 128, ROPE_DIM) out, so the swap is a bitcast.
        q_pe=jnp.pad(jnp.swapaxes(q_pe, 1, 2), token_pad),
        token_meta=token_meta,
        scalars=attention_scalars(
            jnp.asarray(kv_scale, jnp.float32).reshape(1), step_slots
        ),
        cu_q_lens=cu_q_lens,
        cache=cache,
        sm_scale=sm_scale,
        return_lse=return_lse,
    )


def chunk_outputs(
    outputs: tuple[jax.Array, ...], num_tokens: int, num_heads: int
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """The attention's (out, [lse]) over padded tokens as the call's out, or
    (out, lse) with lse (H, T)."""
    out = outputs[0][:num_tokens]
    if len(outputs) == 1:
        return out
    else:
        lse = outputs[1].reshape(-1, NUM_LANES)
        return out, lse[:num_tokens, :num_heads].T


@functools.partial(
    jax.jit, static_argnames=("sm_scale", "return_lse", "chunk_tokens")
)
def dsa_sparse_mla(
    q_nope: jax.Array,
    q_pe: jax.Array,
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    topk_indices: jax.Array,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
    *,
    sm_scale: float,
    kv_scale: jax.Array,
    return_lse: bool = False,
    chunk_tokens: int | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Each query token attends its selected cache positions as MQA with one
    key head of the latent and the rope, and the latent as the value.

    The probabilities round to fp8 for P @ V, as the dense MLA kernels'
    do. One kernel serves decode and prefill tokens alike.

    Args:
      q_nope: (T, H, LATENT_DIM) bf16 queries with W_UK absorbed, packed
        token-major; H at most 128.
      q_pe: (T, H, ROPE_DIM) bf16 rope queries.
      nope_cache, rope_cache: dsa_cache_shapes(num_pages, page_size) int32
        split fp8 cache, holding every token the indices name; read only.
      topk_indices: (T, top_k) int32 logical positions within each token's
        request, -1 for an unused slot; top_k a multiple of 128. Slots from
        min(top_k, position + 1) on are unused whatever they hold. Read in
        place.
      cu_q_lens: (B + 1,) int32 from 0, nondecreasing: request r's tokens are
        rows [cu_q_lens[r], cu_q_lens[r + 1]); rows past cu_q_lens[B] are
        padding, with a zero output.
      kv_lens: (B,) int32 tokens of each request in the cache, its tokens the
        last q_len of them.
      block_table: (B, pages_per_seq) int32 cache page of each request page.
      sm_scale: scale of q_nope . latent + q_pe . rope.
      kv_scale: f32 scalar, a cached value being its fp8 value times
        kv_scale; traced, so a new scale compiles no new kernel.
      return_lse: also return the natural-log lse.
      chunk_tokens: tokens a chunk holds, a multiple of the attention step's
        tokens; default_chunk_tokens(T) if None.

    Returns:
      out (T, H, LATENT_DIM) bf16, or (out, lse) with return_lse, lse (H, T)
      f32, +inf for a token that sees no key.
    """
    num_tokens, num_heads, _ = q_nope.shape
    check_operands(
        q_nope, q_pe, nope_cache, rope_cache, cu_q_lens, kv_lens, block_table
    )
    check_indices(topk_indices, num_tokens)
    if chunk_tokens is None:
        chunk_tokens = default_chunk_tokens(num_tokens)
    else:
        check_chunk_tokens(chunk_tokens, num_tokens)
    chunk = chunk_attention(
        q_nope,
        q_pe,
        nope_cache,
        rope_cache,
        cu_q_lens,
        kv_lens,
        block_table,
        top_k=topk_indices.shape[1],
        sm_scale=sm_scale,
        kv_scale=kv_scale,
        return_lse=return_lse,
    )
    outputs = attend_in_chunks(chunk, topk_indices, chunk_tokens)
    return chunk_outputs(outputs, num_tokens, num_heads)


def attend_in_chunks(
    chunk: ChunkAttention, topk_indices: jax.Array, chunk_tokens: int
) -> tuple[jax.Array, ...]:
    """Gather and attend every token's (T, top_k) indices chunk after chunk;
    chunk i + 1's gather depends on nothing of chunk i's attention, so the
    SparseCore runs it under that attention. Returns (out, [lse])."""
    padded_tokens = chunk.q_nope.shape[0]
    chunk_tokens = min(chunk_tokens, padded_tokens)
    indices = slot_indices(topk_indices, chunk.cache.top_k).reshape(-1)
    outputs = None
    for first in chunk_starts(padded_tokens, chunk_tokens):
        rows = chunk.gather_rows(chunk_row(first, first), indices, chunk_tokens)
        outputs = chunk.attend_rows(
            rows,
            outputs,
            first_token=first,
            num_tokens=chunk_tokens,
            scalars=chunk.scalars,
        )
    return outputs


def slot_indices(topk_indices: jax.Array, num_slots: int) -> jax.Array:
    """The (T, num_slots) indices the gather reads: the (T, top_k) indices,
    then UNUSED_INDEX in the slots past top_k."""
    top_k = topk_indices.shape[1]
    if top_k == num_slots:
        return topk_indices
    return jnp.pad(
        topk_indices,
        ((0, 0), (0, num_slots - top_k)),
        constant_values=UNUSED_INDEX,
    )


def check_operands(
    q_nope: jax.Array,
    q_pe: jax.Array,
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
) -> None:
    """Fails unless the sparse MLA's queries, caches and requests meet its
    contract."""
    num_tokens, num_heads, _ = q_nope.shape
    check_caches(nope_cache, rope_cache)
    check_requests(cu_q_lens, kv_lens, block_table)
    if q_nope.shape[2] != LATENT_DIM or q_pe.shape != (
        num_tokens,
        num_heads,
        ROPE_DIM,
    ):
        raise ValueError(
            f"q_nope and q_pe must be (T, H, {LATENT_DIM}) and (T, H,"
            f" {ROPE_DIM}); got {q_nope.shape} and {q_pe.shape}."
        )
    elif {q_nope.dtype, q_pe.dtype} != {QUERY_DTYPE}:
        raise NotImplementedError(
            "The sparse MLA takes bf16 queries; got"
            f" {q_nope.dtype} and {q_pe.dtype}."
        )
    elif num_heads > MAX_HEADS or num_heads % NUM_SUBLANES:
        raise NotImplementedError(
            f"The sparse MLA supports a multiple of {NUM_SUBLANES} heads up to"
            f" {MAX_HEADS}; got {num_heads}."
        )


def check_chunk_tokens(chunk_tokens: int, num_tokens: int) -> None:
    """Fails unless chunk_tokens is a multiple of the attention step's tokens
    in a call of num_tokens tokens."""
    step_tokens = tokens_per_step(num_tokens)
    if chunk_tokens % step_tokens:
        raise ValueError(
            f"chunk_tokens must be a multiple of {step_tokens} tokens; got"
            f" {chunk_tokens}."
        )


def check_indices(topk_indices: jax.Array, num_tokens: int) -> None:
    """Fails unless topk_indices is (num_tokens, top_k) int32 with top_k a
    multiple of CHUNK_SLOTS."""
    if topk_indices.ndim != 2 or topk_indices.shape[0] != num_tokens:
        raise ValueError(
            f"topk_indices must be ({num_tokens}, top_k); got"
            f" {topk_indices.shape}."
        )
    elif topk_indices.dtype != jnp.dtype(jnp.int32):
        raise NotImplementedError(
            f"topk_indices must be int32; got {topk_indices.dtype}."
        )
    elif topk_indices.shape[1] % CHUNK_SLOTS:
        raise ValueError(
            f"top_k must be a multiple of {CHUNK_SLOTS}; got"
            f" {topk_indices.shape[1]}."
        )
