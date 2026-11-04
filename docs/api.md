# API Reference

## Softmax attention

The main functions implement scaled dot product attention
(softmax(Q @ K^T * softmax_scale) @ V), with the same names and arguments as
[flash-attn](https://github.com/Dao-AILab/flash-attention), taking and
returning JAX arrays:

```python
from flywheel_tpu import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
```

All inputs must be bfloat16, except a paged KV cache, which may also be
float8_e4m3fn or float8_e5m2. `dropout_p`, `alibi_slopes`, `deterministic`,
`return_attn_probs` and `attention_chunk` are accepted for compatibility and
raise `NotImplementedError` on any non-default value. `interpret=True` runs
the Pallas TPU interpreter, e.g. on CPU.

### flash_attn_func

```python
flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                window_size=(-1, -1), attention_chunk=None, softcap=0.0,
                alibi_slopes=None, deterministic=False, return_attn_probs=False,
                return_softmax_lse=False, interpret=False, head_dim=None,
                token_major=False, *, rotary_cos=None, rotary_sin=None,
                rotary_interleaved=False, rotary_k=True, heads_outer_fold=False,
                learnable_sink=None):
"""Unlike flash-attn, the default layout is head-major, (batch, nheads, seqlen, headdim),
which is the kernel's own layout, so no operand is relaid out.
Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
If the row of the mask is all zero, the output will be zero (and lse -inf).
If window_size != (-1, -1), implements sliding window local attention. Query at position i
will only attend to keys between
[i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.
seqlen_q may be any length. seqlen_k must be a multiple of 128 unless causal=True or
window_size[1] == 0.
jax.grad runs the Pallas backward kernel for MHA and MQA/GQA with causal or full masks.
window_size, softcap, rotary, heads_outer_fold and learnable_sink are forward-only: their
gradient raises NotImplementedError.

Arguments:
    q: (batch_size, nheads, seqlen_q, headdim)
    k: (batch_size, nheads_k, seqlen_k, headdim)
    v: (batch_size, nheads_k, seqlen_k, headdim_v). headdim_v may differ from headdim.
        With token_major=True, q/k/v are (batch_size, seqlen, nheads * headdim).
    softmax_scale: float. The scaling of QK^T before applying softmax.
        Default to 1 / sqrt(headdim).
    causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
    window_size: (left, right). If not (-1, -1), implements sliding window local attention.
    softcap: float. Anything > 0 activates softcapping attention:
        tanh(softmax_scale * q @ k^T / softcap) * softcap.
    return_softmax_lse: bool. Whether to also return the logsumexp of the scaled, masked
        attention scores.
    head_dim: int. Required with token_major=True, where the fused last axis alone does not
        give the head split: a multiple of 128, or 64 for MHA with an even nheads.
    token_major: bool. See above.
    rotary_cos [optional]: (seqlen_ro, headdim / 2), in q's dtype or float32. The cos of the
        rotary embedding applied to q and k inside the kernel. headdim must be divisible by 16.
        k is rotated at positions 0 .. seqlen_k - 1 and q from seqlen_k - seqlen_q on, so
        seqlen_q <= seqlen_k.
    rotary_sin [optional]: (seqlen_ro, headdim / 2). Similar to rotary_cos.
    rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
        If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
        rotary embedding will combine dimensions 0 & headdim / 2, 1 & headdim / 2 + 1
        (i.e. GPT-NeoX style).
    rotary_k: bool. If False, only q is rotated.
    heads_outer_fold: bool. q/k/v and out are (nheads, batch_size, seqlen, headdim) instead.
        Forward-only, with no return_softmax_lse and only q-only rotary (rotary_k=False).
    learnable_sink [optional]: (nheads,), float. One sink logit per query head (gpt-oss
        attention sinks), on the scale of the softcapped scores: exp(learnable_sink[h])
        joins head h's softmax denominator with no value row, and lse includes it. A row
        with no visible key returns out = 0 and lse = learnable_sink[h]. Each logit must be
        finite or -inf (no sink). May be traced.
Return:
    out: (batch_size, nheads, seqlen_q, headdim_v), or (batch_size, seqlen_q,
        nheads * headdim_v) with token_major=True.
    softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen_q), float32.
        The natural-log logsumexp of each row of the scaled, masked attention scores.
"""
```

### flash_attn_varlen_func

```python
flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                       dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1),
                       softcap=0.0, alibi_slopes=None, deterministic=False,
                       return_attn_probs=False, return_softmax_lse=False, interpret=False,
                       head_dim=None, token_major=False, block_sizes=None, *,
                       rotary_cos=None, rotary_sin=None, rotary_interleaved=False,
                       rotary_k=True, block_table=None, seqused_k=None, num_active=None,
                       k_descale=None, v_descale=None, nheads_k=None,
                       learnable_sink=None):
"""Attention over sequences packed along the token axis. Unlike flash-attn, the default layout
is head-major, (nheads, total, headdim), which is the kernel's own layout, so no operand is
relaid out.
causal and window_size are bottom-right aligned per sequence, as in flash_attn_func. A row
with no visible key is unspecified, so causal needs seqlen_k >= seqlen_q for every sequence.
Rotary uses per-sequence positions. Forward-only: jax.grad raises NotImplementedError.

Arguments:
    q: (nheads, total_q, headdim), where total_q = total number of query tokens in the batch.
    k: (nheads_k, total_k, headdim), where total_k = total number of key tokens in the batch.
    v: (nheads_k, total_k, headdim_v).
        With token_major=True, q/k/v are (total, nheads * headdim).
    cu_seqlens_q: (batch_size + 1,), int. The cumulative sequence lengths of the sequences
        in the batch, used to index into q. May be traced. Rows past cu_seqlens_q[-1] are
        padding that never attends or is attended.
    cu_seqlens_k: (batch_size + 1,), int. The cumulative sequence lengths of the sequences
        in the batch, used to index into k and v.
    max_seqlen_q: int. Maximum query sequence length in the batch. Must be static: it only
        caps the kernel's block sizes and is bucketed to a power of two, so an underestimate
        is slow, not wrong.
    max_seqlen_k: int. Maximum key sequence length in the batch. Must be static, as above.
        The rotary tables must cover it.
    block_sizes [optional]: (block_q, block_kv, block_q_compute, block_kv_compute). Pins the
        forward tiles instead of the tuned or analytic ones; block_q and block_kv must be
        multiples of 128.
    block_table [optional]: (batch_size, max_num_blocks_per_seq), int. Reads k as a paged KV
        cache, read-only: one merged (num_blocks, page_block_size, cache_heads, headdim)
        cache laid out as flash_attn_with_kvcache's, bf16 or fp8, with v and cu_seqlens_k
        None. page_block_size and headdim are multiples of 128; any nheads_k. q stays
        head-major; token_major and rotary_k=True are not supported,
        and window_size takes a right bound of -1 or 0 only (causal forces 0).
    seqused_k [optional]: (batch_size,), int. With block_table, each sequence's kv length,
        its new tokens included, which the cache must already hold; max_seqlen_k bounds it.
    num_active [optional]: int, may be traced. With block_table, only the first num_active
        sequences attend; rows past the last of them return out = 0 and lse = -inf.
    k_descale, v_descale, nheads_k [optional]: with block_table, as in
        flash_attn_with_kvcache. An fp8 cache needs nheads_k.
    learnable_sink [optional]: as in flash_attn_func. Padding rows and rows past
        num_active still return out = 0 and lse = -inf.
    The other arguments are as in flash_attn_func.
Return:
    out: (nheads, total_q, headdim_v), or (total_q, nheads * headdim_v) with token_major=True.
    softmax_lse [optional, if return_softmax_lse=True]: (nheads, total_q), float32.
"""
```

### flash_attn_with_kvcache

```python
flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, *, cache_seqlens,
                        cache_batch_idx=None, block_table=None, num_active=None,
                        cu_seqlens_q=None, max_seqlen_q=None, softmax_scale=None,
                        causal=False, window_size=(-1, -1), softcap=0.0,
                        return_softmax_lse=False, interpret=False, k_descale=None,
                        v_descale=None, nheads_k=None, learnable_sink=None):
"""
If k and v are not None, they are appended to the cache after each row's cache_seqlens
entries, and attention runs against the updated cache, all in 1 kernel. This is useful for
incremental decoding: pass in the cache from the previous step and the new keys/values of
the current step.

JAX arrays are immutable, so unlike flash-attn the updated cache is returned. The function
is jitted with k_cache and v_cache donated, so the update happens in place in HBM; rebind
the returned caches and do not reuse the old ones. Under an outer jax.jit, donate the caches
on that jit instead.

If you pass in k / v, you must make sure that the cache is large enough to hold the new
values. For example, the KV cache could be pre-allocated with the max sequence length, and
you can use cache_seqlens to keep track of the current sequence lengths of each sequence in
the batch.

Unlike flash_attn_func, q is token-major, as in flash-attn. Decode is q with seqlen_q = 1;
seqlen_q > 1 or a packed q with cu_seqlens_q runs chunked prefill against the cache, which
needs the paged cache (block_table): the new tokens are appended, then flash_attn_varlen_func's
paged kernel attends the updated pages. A short (batch_size, seqlen_q) window with no left
window and at most 128 query rows per KV head (seqlen_q * nheads / nheads_k), such as a
speculative verify window, runs on the decode kernel instead, folded into the query heads.
The contiguous cache serves single-token decode only.

Supports MQA/GQA as flash_attn_func does. The paged cache takes any nheads_k; the contiguous
cache needs nheads_k even or 1, because its cache load packs two bf16 heads into one u32 lane.

If causal=True, the causal mask is aligned to the bottom right corner of each request's
attention matrix, as in flash_attn_func. A decode row with nothing to attend returns out = 0
and lse = -inf; on multi-token and packed calls, as in flash_attn_varlen_func, a row with no
visible key is unspecified.

If window_size != (-1, -1), implements sliding window local attention. Multi-token and
packed calls take a window right bound of -1 or 0 only (causal forces 0).

Arguments:
    q: (batch_size, seqlen_q, nheads, headdim), or (total_q, nheads, headdim) with
        cu_seqlens_q.
    k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
        seqlen_cache a multiple of 128,
        or (num_blocks, page_block_size, 2 * nheads_k, headdim) if there's a block_table
        (i.e. paged KV cache). The paged cache merges K and V: a bfloat16 cache interleaves
        each KV head's K row and V row, [k0, v0, k1, v1, ...] (RPA v3's head order); a
        float8_e4m3fn or float8_e5m2 cache holds each head pair's two K rows then its two
        V rows, [k0, k1, v0, v1, ...], an odd nheads_k padded with one zero head.
        page_block_size must be a multiple of 128.
    v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
        or None if there's a block_table.
    k [optional]: (batch_size, seqlen_new, nheads_k, headdim), or (total_q, nheads_k, headdim)
        with cu_seqlens_q. If not None, we concatenate k with k_cache, starting at the
        indices specified by cache_seqlens. Pass k and v together.
    v [optional]: Similar to k.
    cache_seqlens: int, or (batch_size,), int. The sequence lengths of the KV cache before
        the append.
    cache_batch_idx [optional]: (batch_size,), int. The indices used to index into the KV
        cache. If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
        Contiguous cache only.
    block_table [optional]: (batch_size, max_num_blocks_per_seq), int. Maps each row's
        positions to cache pages. Pages an append writes must be private to the request.
    num_active [optional]: int. The leading rows that are real requests; later rows are
        skipped, and their out / lse are 0 / -inf on multi-token and packed calls but
        unspecified on single-token decode.
    cu_seqlens_q [optional]: (batch_size + 1,), int. Nondecreasing query boundaries from 0 to
        at most total_q; buffer padding past cu_seqlens_q[-1] returns out = 0 and lse = -inf.
    max_seqlen_q [optional]: int. Maximum query sequence length in the batch, required with
        cu_seqlens_q and rejected without it. Must be static: as in flash_attn_varlen_func it
        only caps the kernel's block sizes, so a bound off the true length is slow, not wrong.
    softmax_scale, causal, window_size, softcap, learnable_sink: as in flash_attn_func.
    k_descale, v_descale [optional]: float. Per-tensor descales of an fp8 cache, default 1
        (FlashAttention-3's): the cache stores K / k_descale and V / v_descale. k / v
        appends must already be in the cache's dtype. q stays bfloat16.
    nheads_k [optional]: int. The cache's KV head count. Defaults to k's heads, else to
        half the cache's head axis, so an fp8 cache without k needs it.
    cache_seqlens, cache_batch_idx, block_table, num_active, cu_seqlens_q, k_descale,
    v_descale and learnable_sink may be traced.

Return:
    out: (batch_size, seqlen_q, nheads, headdim), or (total_q, nheads, headdim) with
        cu_seqlens_q.
    softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen_q), or
        (nheads, total_q) with cu_seqlens_q, float32.
    k_cache, v_cache: the updated caches, or the updated merged kv_cache alone if there's a
        block_table. The full return is (out, [softmax_lse,] k_cache, v_cache) or
        (out, [softmax_lse,] kv_cache).
"""
```

To see the implementation of these functions, check
[prefill.py](../flywheel_tpu/softmax_attention/prefill.py) and
[decode.py](../flywheel_tpu/softmax_attention/decode.py).

## Linear attention

The linear-attention layers of hybrid models, Gated DeltaNet (GDN, e.g.
Qwen3-Next) and Kimi Delta Attention (KDA, e.g. Kimi Linear), each run as one
fused kernel: causal depthwise conv1d + silu over the packed qkv projection,
then the chunked delta rule, over a ragged serving batch whose per-request
conv and recurrent states live in slot caches:

```python
from flywheel_tpu.linear_attention import fused_conv1d_gdn, fused_conv1d_kda
```

Both are inference-only (no backward pass), take bfloat16 activations, and are
jitted with `conv_state` and `recurrent_state` donated: rebind the returned
caches and do not reuse the old ones. Their implementations are in
[gdn/wrapper.py](../flywheel_tpu/linear_attention/gdn/wrapper.py) and
[kda/wrapper.py](../flywheel_tpu/linear_attention/kda/wrapper.py), with tests in
[tests/linear_attention/](../tests/linear_attention/).

### fused_conv1d_gdn

```python
fused_conv1d_gdn(qkv, b, a, conv_state, recurrent_state, conv_weight, conv_bias,
                 a_log, dt_bias, query_start_loc, state_indices, distribution,
                 seq_lens, read_state_indices, read_offsets=None, *, n_kq, n_v,
                 d_k, d_v, kernel_size, num_spec_tokens=0, zero_initialize_out=True,
                 compute_precision=jnp.float32, decode_tile_size=4,
                 mixed_tile_size=128, compute_chunk_size=64):
"""Per token t of a sequence, after the conv1d + silu, with one (d_k, d_v) state S per
value head:
    q_t = l2norm(q_t) / sqrt(d_k),  k_t = l2norm(k_t)
    beta_t = sigmoid(b_t),  g_t = -exp(a_log) * softplus(a_t + dt_bias)
    S_t = exp(g_t) * S_{t-1} + beta_t * k_t (v_t - exp(g_t) * S_{t-1}^T k_t)^T
    o_t = S_t^T q_t
Supports grouped value heads: n_v must be divisible by n_kq, and value head h reads
q/k head h // (n_v / n_kq).
Sequences are packed along the token axis as in flash_attn_varlen_func, decodes first. A
sequence with no prior context (seq_lens equal to its new token count) starts from a zero
state, so its slots may hold garbage.

Arguments:
    qkv: (num_tokens, dim), bfloat16, dim = 2 * n_kq * d_k + n_v * d_v. q, k and v
        concatenated along the last axis in that order, each (nheads * headdim).
    b: (num_tokens, n_v). The input to beta.
    a: (num_tokens, n_v). The input to the decay gate.
    conv_state: (num_slots, kernel_size - 1, dim). The last kernel_size - 1 conv inputs
        of each slot. Slot 0 is the null block for padded or invalid tokens.
    recurrent_state: (num_slots, n_v, d_k, d_v). The state S of each slot, same null block.
    conv_weight: (dim, 1, kernel_size). The depthwise conv weight.
    conv_bias [optional]: (dim,).
    a_log: (n_v,). The per-head log decay scale.
    dt_bias: (n_v,). The bias of the decay gate.
    query_start_loc: (num_seqs + 1,), int. The cumulative token counts of the sequences,
        used to index into qkv, like cu_seqlens_q.
    state_indices: (num_seqs,), int. The slot each sequence writes its final state to.
        With num_spec_tokens > 0, the base of each request's num_spec_tokens + 1
        consecutive slots.
    distribution: (3,), int32. [decode_end, prefill_end, mixed_end]. Sequences
        [0, decode_end) are single-token decodes, or verify windows of up to
        num_spec_tokens + 1 tokens; [decode_end, mixed_end) are prefills or chunked
        prefills; sequences from mixed_end on are skipped.
    seq_lens: (num_seqs,), int. The length of each sequence including its new tokens.
    read_state_indices: (num_seqs,), int. The slot each sequence reads its initial state
        from. Equals state_indices unless prefix caching resumes from a cached state block.
    read_offsets [optional]: (num_seqs,), int32. Required if num_spec_tokens > 0: the
        number of accepted tokens minus 1 from the last verify step. A windowed sequence
        reads its state from read_state_indices[s] + read_offsets[s] and writes the state
        after window position t to state_indices[s] + t.
    n_kq, n_v: int. The number of q/k heads and of value heads.
    d_k, d_v: int. The q/k and value head dims.
    kernel_size: int. The conv kernel size, between 2 and 17.
    num_spec_tokens: int. The speculative draft tokens per request; 0 gives plain
        one-token decode.
    zero_initialize_out: bool. Whether to zero the output rows past the last real token,
        which no sequence owns.
    compute_precision: dtype. The dtype of the in-kernel math.
    decode_tile_size: int. Sequences per decode tile.
    mixed_tile_size: int. Tokens per prefill tile, a multiple of compute_chunk_size.
    compute_chunk_size: int. Tokens per chunked delta-rule step: a multiple of 16 larger
        than kernel_size - 1; above 64 a multiple of 64, above 256 of 256, and so on.
Return:
    (conv_state, recurrent_state): The updated caches, in their input dtypes.
    out: (num_tokens, n_v * d_v), bfloat16.
    The full return is ((conv_state, recurrent_state), out).
"""
```

### fused_conv1d_kda

```python
fused_conv1d_kda(qkv, b, g, conv_state, recurrent_state, conv_weight, conv_bias,
                 a_log, dt_bias, query_start_loc, state_indices, distribution,
                 seq_lens, read_state_indices, read_offsets=None, *, n_kq, n_v,
                 d_k, d_v, kernel_size, num_spec_tokens=0, zero_initialize_out=True,
                 compute_precision=jnp.float32, decode_tile_size=4,
                 mixed_tile_size=128, compute_chunk_size=64, lower_bound=None,
                 beta_is_activated=False):
"""The gated delta rule of fused_conv1d_gdn with one decay gate per key channel: the log
decay d_t is (n_v, d_k) per token, computed from the raw gate g_t, and decays the state
row-wise:
    d_t = -exp(a_log) * softplus(g_t + dt_bias), or, with lower_bound,
          lower_bound * sigmoid(exp(a_log) * (g_t + dt_bias))
    S_t = diag(exp(d_t)) S_{t-1} + beta_t * k_t (v_t - (diag(exp(d_t)) S_{t-1})^T k_t)^T

Arguments:
    g: (num_tokens, n_v * d_k), bfloat16. The raw gate, head-major like q and k. It
        replaces fused_conv1d_gdn's a.
    a_log: (n_v,). The per-head log decay scale.
    dt_bias: (n_v * d_k,). The per-channel bias of the decay gate.
    lower_bound [optional]: float. A finite negative bound on the log decay, switching the
        gate to the sigmoid form above.
    beta_is_activated: bool. Whether b already holds beta, skipping its sigmoid.
    The other arguments and the return are as in fused_conv1d_gdn.
"""
```

### ReplaySSM decode

For decode, `fused_conv1d_gdn_replay` and `fused_conv1d_kda_replay` run
[ReplaySSM](https://arxiv.org/abs/2610.13778) (Liou et al., 2026): a decode
sequence keeps its recurrent state as a checkpoint plus up to `replay_window`
records of its recent tokens, computes its outputs from them without
materializing the state, and writes a new checkpoint only when the records
would overflow. Speculative tokens commit by moving `replay_lens`, so the
recurrent state needs no per-position checkpoints. The paper implements GDN
(and Mamba-2) in vLLM; the KDA variant is our extension. The f32 checkpoint is
stored as two 16-bit planes, so a decode step reads only the bf16 high plane;
`replay_layout`, `split_checkpoint` and `merge_checkpoint` in each wrapper give
the buffer shapes and convert the checkpoint from and to f32. Their arguments
are documented in their docstrings.
