"""fused_moe: the routed expert MLP of a MoE layer on TPU, from routing to the
combined output.

Four steps, three of them Pallas kernels. Where the hardware has SparseCores
that do indirect DMA (v7x), the permute runs on them, and so do the routing
and the combine on batches that fill their vector subcores; elsewhere the
routing and the permute are XLA ops and the combine is a TensorCore kernel.
The permute and combine paths give the same bits.

0. routing (fused_moe_softmax_topk): each token's top_k experts by router
   logit (lax.top_k's choice and order) and the softmax over their logits.
1. permute: sort the (token, expert) pairs by expert. Only the indices and
   the routing weights move; no activation is copied.
2. fused_mlp (kernel): per expert, gather its tokens' rows straight from the
   unsorted activations, run mm1 + the gated activation + mm2 with the
   intermediate kept in VMEM, and scale each row by its routing weight.
3. combine (kernel): per token, gather its top_k scaled rows by the inverse
   sort permutation and sum them in f32.

The expert weights are merged and padded once at load time
(prepare_expert_weights).
"""

import functools
import math

import jax
import jax.numpy as jnp

from flywheel_tpu.core.tiling import round_up
from flywheel_tpu.core.tpu import NUM_LANES
from flywheel_tpu.moe.config import BF16_PACKING, ROW_ALIGNMENT, PipelineDepths
from flywheel_tpu.moe.kernels.combine import combine
from flywheel_tpu.moe.kernels.fused_mlp import fused_mlp
from flywheel_tpu.moe.kernels.sc_combine import (
    fills_gather_pipelines,
    sc_combine,
    sc_combine_index_length,
)
from flywheel_tpu.moe.kernels.sc_permute import (
    sc_permute,
    sparse_core_supported,
)
from flywheel_tpu.moe.kernels.sc_topk import routes_on_sparse_core, sc_topk

# Note (david): the fused MLP's static knobs, which both entry points take
# and pass through.
MLP_OPTION_NAMES = (
    "activation",
    "tile_m",
    "tail_bucket",
    "tail_bucket_m_values",
    "pipeline_depths",
    "weight_block_columns",
    "weight_block_segment_rows",
    "gather_pipeline_rows",
)


def prepare_expert_weights(
    w_gate: jax.Array, w_up: jax.Array, w2: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """The fused_moe w1 and w2 of the experts' gate, up and down projections.

    padded is intermediate rounded up to a multiple of 128. w_gate and w_up
    [E, hidden, intermediate] take zero columns up to padded and merge into
    w1 [E, hidden, 2 * padded]; w2 [E, intermediate, hidden_out] takes zero
    rows up to [E, padded, hidden_out].

    Note (david): the padding is exact. A padded column holds act(0) * 0 = 0
    in the intermediate, and a zero row of w2 adds nothing to the output.

    A load-time weight transform: call it once when the weights are loaded,
    not inside the jitted forward, where it would copy every expert's weights
    on each call.
    """
    if w_gate.ndim != 3 or w_gate.shape != w_up.shape:
        raise ValueError(
            "w_gate and w_up must both be [E, hidden, intermediate], got"
            f" {list(w_gate.shape)} and {list(w_up.shape)}"
        )
    num_experts, _, intermediate = w_gate.shape
    if w2.ndim != 3 or w2.shape[:2] != (num_experts, intermediate):
        raise ValueError(
            "w2 must be [E, intermediate, hidden_out] ="
            f" [{num_experts}, {intermediate}, ...], got {list(w2.shape)}"
        )
    num_pad_columns = round_up(intermediate, NUM_LANES) - intermediate
    if num_pad_columns == 0:
        return jnp.concatenate([w_gate, w_up], axis=-1), w2
    else:
        column_padding = ((0, 0), (0, 0), (0, num_pad_columns))
        w1 = jnp.concatenate(
            [jnp.pad(w_gate, column_padding), jnp.pad(w_up, column_padding)],
            axis=-1,
        )
        return w1, jnp.pad(w2, ((0, 0), (0, num_pad_columns), (0, 0)))


def check_padded_intermediate(w1: jax.Array, w2: jax.Array) -> None:
    """Raises unless the weights' intermediate size is a multiple of 128, as
    prepare_expert_weights pads it."""
    if w1.shape[-1] % (2 * NUM_LANES) != 0 or w2.shape[1] % NUM_LANES != 0:
        raise ValueError(
            f"the intermediate size must be a multiple of {NUM_LANES}, got w1"
            f" {list(w1.shape)} and w2 {list(w2.shape)}; build w1 and w2"
            " with prepare_expert_weights at weight load time, which pads it"
        )


def invert_permutation(perm: jax.Array) -> jax.Array:
    """inv with inv[perm[s]] = s, by a unique-index scatter.

    Note (david): a scatter, not a second argsort, and a scatter-add onto
    zeros, which equals a set for unique indices: XLA offloads add-combiner
    scatters to SparseCore, while a scatter-set lowers to a serializing
    TensorCore fusion.
    """
    num_rows = perm.shape[0]
    return (
        jnp.zeros((num_rows,), jnp.int32)
        .at[perm]
        .add(jnp.arange(num_rows, dtype=jnp.int32))
    )


def xla_permute(
    flat_ids: jax.Array, flat_weights: jax.Array, num_experts: int, top_k: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The permute in XLA, with sc_permute's outputs: (token_indices,
    sorted_weights, group_sizes, inv_perm), the rows padded to a multiple of
    128 that read token 0 with weight 0 and count toward the last expert."""
    num_rows = flat_ids.shape[0]
    positions = jnp.arange(num_rows, dtype=jnp.int32)
    # Note (david): one stable multi-operand sort keyed on the expert ids. The
    # position payload is the permutation and the weight payload lands in
    # sorted order, which replaces a separate per-element XLA gather.
    _, sort_perm, sorted_weights = jax.lax.sort(
        (flat_ids, positions, flat_weights), num_keys=1, is_stable=True
    )
    group_sizes = jnp.bincount(flat_ids, length=num_experts).astype(jnp.int32)
    token_indices = sort_perm // top_k
    num_pad_rows = round_up(num_rows, ROW_ALIGNMENT) - num_rows
    if num_pad_rows == 0:
        return (
            token_indices,
            sorted_weights,
            group_sizes,
            invert_permutation(sort_perm),
        )
    else:
        return (
            jnp.pad(token_indices, (0, num_pad_rows)),
            jnp.pad(sorted_weights, (0, num_pad_rows)),
            group_sizes.at[-1].add(num_pad_rows),
            invert_permutation(sort_perm),
        )


def combine_rows(
    rows: jax.Array,
    inv_perm: jax.Array,
    top_k: int,
    num_tokens: int,
    has_sparse_core: bool,
    interpret: bool,
) -> jax.Array:
    """The token-ordered [num_tokens, hidden] sum of the scaled rows, on
    SparseCore when the batch fills its gather pipelines. Both kernels give
    the same bits."""
    hidden = rows.shape[1] * rows.shape[2]
    if has_sparse_core and fills_gather_pipelines(num_tokens, hidden):
        # Note (david): the last 16-token tile reads whole index windows; the
        # entries past the pairs gather row 0 for tokens that are dropped.
        index_length = sc_combine_index_length(num_tokens, top_k)
        if inv_perm.shape[0] < index_length:
            sc_inv_perm = jnp.pad(
                inv_perm, (0, index_length - inv_perm.shape[0])
            )
        else:
            sc_inv_perm = inv_perm
        out = sc_combine(rows, sc_inv_perm, top_k, num_tokens=num_tokens)
        return out[:num_tokens] if out.shape[0] != num_tokens else out
    else:
        out = combine(
            rows, inv_perm, top_k, num_tokens=num_tokens, interpret=interpret
        )
        return out.reshape(num_tokens, hidden)


def routed_mlp(
    hidden_states: jax.Array,
    flat_ids: jax.Array,
    flat_weights: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    *,
    top_k: int,
    activation: str,
    tile_m: int | None,
    tail_bucket: bool,
    tail_bucket_m_values: tuple[int, ...] | None,
    pipeline_depths: PipelineDepths | None,
    weight_block_columns: int | None,
    weight_block_segment_rows: int | None,
    gather_pipeline_rows: int | None,
    interpret: bool,
) -> jax.Array:
    """permute, fused_mlp and combine of the flat (token, k) pairs:
    bf16 [num_tokens, hidden_out]."""
    num_tokens, hidden = hidden_states.shape
    num_experts = w1.shape[0]
    has_sparse_core = not interpret and sparse_core_supported()
    packed_hidden_states = hidden_states.reshape(
        num_tokens, BF16_PACKING, hidden // BF16_PACKING
    )
    # Note (david): the fused MLP takes a multiple of 128 rows: the padded
    # rows read token 0 with weight 0, count toward the last expert and are
    # never combined. Both permute paths give the same rows bit for bit.
    if has_sparse_core:
        token_indices, sorted_weights_f32, group_sizes, inv_perm = sc_permute(
            flat_ids, flat_weights, num_experts=num_experts, top_k=top_k
        )
        # Note (david): exact, as the kernel carries the weights in f32.
        sorted_weights = sorted_weights_f32.astype(flat_weights.dtype)
    else:
        token_indices, sorted_weights, group_sizes, inv_perm = xla_permute(
            flat_ids, flat_weights, num_experts, top_k
        )
    rows = fused_mlp(
        packed_hidden_states,
        token_indices,
        group_sizes,
        w1,
        w2,
        sorted_weights,
        activation=activation,
        tile_m=tile_m,
        tail_bucket=tail_bucket,
        tail_bucket_m_values=tail_bucket_m_values,
        pipeline_depths=pipeline_depths,
        weight_block_columns=weight_block_columns,
        weight_block_segment_rows=weight_block_segment_rows,
        gather_pipeline_rows=gather_pipeline_rows,
        interpret=interpret,
    )
    return combine_rows(
        rows, inv_perm, top_k, num_tokens, has_sparse_core, interpret
    )


@functools.partial(jax.jit, static_argnames=(*MLP_OPTION_NAMES, "interpret"))
def fused_moe(
    hidden_states: jax.Array,
    topk_weights: jax.Array,
    topk_ids: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    *,
    activation: str = "silu",
    tile_m: int | None = None,
    tail_bucket: bool = True,
    tail_bucket_m_values: tuple[int, ...] | None = None,
    pipeline_depths: PipelineDepths | None = None,
    weight_block_columns: int | None = None,
    weight_block_segment_rows: int | None = None,
    gather_pipeline_rows: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """The routed expert MLP of a MoE layer, weighted and summed per token.

    For each token t,
    out[t] = sum_k topk_weights[t, k] * (act(h @ gate) * (h @ up)) @ w2[e],
    with h = hidden_states[t], e = topk_ids[t, k] and gate | up = w1[e].

    Args:
        hidden_states: bf16 [..., hidden], hidden a multiple of 256.
        topk_weights: float [..., top_k], the routing weights.
        topk_ids: int [..., top_k], the selected experts, in [0, E).
        w1: bf16 merged gate|up weights [E, hidden, 2 * intermediate] from
            prepare_expert_weights, intermediate a multiple of 128.
        w2: bf16 down projection [E, intermediate, hidden] from
            prepare_expert_weights.
        activation: "silu" or "gelu" (tanh approximation).
        tile_m: Expert-sorted rows per fused MLP step. None sizes a tile
            to hold a typical expert's rows (config.default_tile_m) when two
            whole experts fit VMEM, else takes 256 with weight blocks; either
            steps down while the buffers exceed VMEM.
        tail_bucket: Run each expert's last, partial tile at the smallest of
            tail_bucket_m_values that covers it instead of at tile_m, which
            cuts the work spent on rows of other experts.
        tail_bucket_m_values: The static row buckets; values at or above
            tile_m are dropped and tile_m is appended. None derives them
            from the rows per expert (config.default_tail_bucket_m_values),
            keeping on the whole-expert path those whose bodies fit the
            instruction memory.
        pipeline_depths: The fused MLP's DMA stream depths and issue leads;
            None picks the tuned defaults. The output does not depend on
            them.
        weight_block_columns: Force streaming the expert weights in
            intermediate-dim blocks of this many columns, a multiple of 128
            that divides intermediate. None streams whole experts when two of
            w1 and of w2 fit VMEM next to the other buffers, else blocks
            sized to fit.
        weight_block_segment_rows: On the weight-blocked path, run each
            weight block over static segments of this many rows.
        gather_pipeline_rows: On the weight-blocked path, issue the next tile's
            row gathers in segments of this many rows (a divisor of tile_m,
            one segment per weight block), interleaved with the current
            tile's compute.
        interpret: Run the Pallas TPU interpreter, e.g. on CPU.

    Returns:
        bf16 [..., hidden], the shape of hidden_states.
    """
    *lead, hidden = hidden_states.shape
    check_padded_intermediate(w1, w2)
    if hidden_states.dtype != jnp.bfloat16:
        raise ValueError(
            f"hidden_states must be bf16, got {hidden_states.dtype}"
        )
    if (
        topk_ids.ndim != len(lead) + 1
        or tuple(topk_ids.shape[:-1]) != tuple(lead)
        or not jnp.issubdtype(topk_ids.dtype, jnp.integer)
    ):
        raise ValueError(
            f"topk_ids must be int [{', '.join(map(str, lead))}, top_k], got"
            f" {topk_ids.dtype}{list(topk_ids.shape)}"
        )
    if topk_weights.shape != topk_ids.shape or not jnp.issubdtype(
        topk_weights.dtype, jnp.floating
    ):
        raise ValueError(
            f"topk_weights must be float {list(topk_ids.shape)}, got"
            f" {topk_weights.dtype}{list(topk_weights.shape)}"
        )
    num_tokens, top_k = math.prod(lead), topk_ids.shape[-1]
    if num_tokens == 0:
        return jnp.zeros((*lead, w2.shape[-1]), hidden_states.dtype)
    out = routed_mlp(
        hidden_states.reshape(num_tokens, hidden),
        topk_ids.reshape(-1).astype(jnp.int32),
        topk_weights.reshape(-1),
        w1,
        w2,
        top_k=top_k,
        activation=activation,
        tile_m=tile_m,
        tail_bucket=tail_bucket,
        tail_bucket_m_values=tail_bucket_m_values,
        pipeline_depths=pipeline_depths,
        weight_block_columns=weight_block_columns,
        weight_block_segment_rows=weight_block_segment_rows,
        gather_pipeline_rows=gather_pipeline_rows,
        interpret=interpret,
    )
    return out.reshape(*lead, w2.shape[-1])


@functools.partial(
    jax.jit, static_argnames=("top_k", *MLP_OPTION_NAMES, "interpret")
)
def fused_moe_softmax_topk(
    hidden_states: jax.Array,
    router_logits: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    *,
    top_k: int,
    activation: str = "silu",
    tile_m: int | None = None,
    tail_bucket: bool = True,
    tail_bucket_m_values: tuple[int, ...] | None = None,
    pipeline_depths: PipelineDepths | None = None,
    weight_block_columns: int | None = None,
    weight_block_segment_rows: int | None = None,
    gather_pipeline_rows: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """The routed-expert MoE layer from the router logits, with the
    renormalized softmax top-k routing.

    Each token's experts are lax.top_k's choice and order of its logits (ties
    to the lower expert), and its weights the softmax over those top_k
    logits, which equals the softmax over all experts renormalized over the
    selected ones. The output is fused_moe's on that routing, bit for bit.
    Other routings (sigmoid scores, grouped or biased selection) compute
    their own topk_weights and topk_ids and call
    fused_moe(hidden_states, topk_weights, topk_ids, w1, w2).

    Where the routing runs on SparseCore (sc_topk), its exp can differ from
    XLA's in the last f32 bit.

    Args:
        hidden_states: bf16 [num_tokens, hidden], hidden a multiple of 256.
        router_logits: bf16 [num_tokens, E].
        w1, w2: as in fused_moe.
        top_k: Experts per token, at most 16 on SparseCore.
        The remaining arguments are fused_moe's.

    Returns:
        bf16 [num_tokens, hidden_out].
    """
    check_padded_intermediate(w1, w2)
    if hidden_states.ndim != 2 or hidden_states.dtype != jnp.bfloat16:
        raise ValueError(
            "hidden_states must be bf16[num_tokens, hidden], got"
            f" {hidden_states.dtype}{list(hidden_states.shape)}"
        )
    num_tokens = hidden_states.shape[0]
    num_experts = w1.shape[0]
    if (
        router_logits.shape != (num_tokens, num_experts)
        or router_logits.dtype != jnp.bfloat16
    ):
        raise ValueError(
            f"router_logits must be bf16[{num_tokens}, {num_experts}], got"
            f" {router_logits.dtype}{list(router_logits.shape)}"
        )
    if not 1 <= top_k <= num_experts:
        raise ValueError(f"top_k={top_k} must be in [1, {num_experts}]")
    if num_tokens == 0:
        return jnp.zeros((0, w2.shape[-1]), hidden_states.dtype)
    has_sparse_core = not interpret and sparse_core_supported()
    if has_sparse_core and routes_on_sparse_core(num_tokens, num_experts):
        flat_ids, flat_weights = sc_topk(router_logits, top_k=top_k)
    else:
        top_logits, topk_ids = jax.lax.top_k(router_logits, top_k)
        topk_weights = jax.nn.softmax(top_logits.astype(jnp.float32), axis=-1)
        flat_ids = topk_ids.reshape(-1).astype(jnp.int32)
        flat_weights = topk_weights.reshape(-1)
    return routed_mlp(
        hidden_states,
        flat_ids,
        flat_weights,
        w1,
        w2,
        top_k=top_k,
        activation=activation,
        tile_m=tile_m,
        tail_bucket=tail_bucket,
        tail_bucket_m_values=tail_bucket_m_values,
        pipeline_depths=pipeline_depths,
        weight_block_columns=weight_block_columns,
        weight_block_segment_rows=weight_block_segment_rows,
        gather_pipeline_rows=gather_pipeline_rows,
        interpret=interpret,
    )
