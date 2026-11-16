"""MoE routing on SparseCore: softmax over each token's top-k router logits
(the renormalized softmax top-k), with lax.top_k's expert choice and order.

Each token's E logits are split into 16-lane vectors and a bitonic merge
tree of hardware sorts (plsc.sort_key_val) keeps the top 16 keys, so a token
costs 2 * E / 16 - 1 sorts. A key is the logit's order-preserving uint32
(the order the hardware sort compares in) with the expert index in its low
16 bits, inverted so that a lower index wins a tie, as in lax.top_k; bf16
logits have zero low mantissa bits as f32, so the packing loses nothing and
the value comes back exactly from the key.

The logits stay bf16 in HBM, read in the TensorCore tiling: one int32 word
holds rows 2j and 2j + 1 of a column, so a vector load feeds two tokens.
The vector subcores of both SparseCores take row blocks round-robin,
double-buffered.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from flywheel_tpu.core.tiling import round_up
from flywheel_tpu.core.tpu import BF16_BYTES, BF16_TILE_ROWS, F32_BYTES

HIGH_HALF = -65536  # 0xFFFF0000 as int32
LOW_HALF = 0xFFFF
HALF_WORD_BITS = 16
SIGN_BIT = -(2**31)  # 0x80000000 as int32
# Note (david): the order-preserving key flips the sign bit of a positive
# logit and every bit of a negative one (the high halves; the low halves hold
# the expert).
POSITIVE_FLIP = SIGN_BIT
NEGATIVE_FLIP = HIGH_HALF
# Note (david): a block is a whole number of bf16 HBM tiles (BF16_TILE_ROWS
# rows, half as many int32 word rows), at most MAX_BLOCK_ROWS rows.
MAX_BLOCK_ROWS = 64
# Note (david): below this many router logits, sc_topk's launch latency
# makes it slower than XLA's top_k.
SPARSE_CORE_MIN_LOGITS = 2048 * 128
# Note (david): the input blocks and the output blocks double buffer, so a
# block's loads and stores overlap the sorts of its neighbours.
NUM_SLOTS = 2
# Note (david): from this many experts on, a token's merge tree is deep
# enough that building both tokens of a word row in one iteration (their
# trees overlap) outruns one pass per row parity despite the spills; below
# it the spills cost more. The paired loop takes blocks of PAIR_BLOCK_ROWS
# rows and software-pipelines only when a subcore runs at least
# MIN_PAIR_BLOCKS of them.
PAIR_ROWS_MIN_EXPERTS = 512
PAIR_BLOCK_ROWS = 32
MIN_PAIR_BLOCKS = 2
# Note (david): an output slot starts on the 1D VMEM tile.
SLOT_ALIGNMENT = 128
# Note (david): a rough count that lets XLA's scheduler size this call, so it
# can place TensorCore work (the activation relayout) under it.
FLOPS_PER_LOGIT = 8


def output_slot_size(block_rows: int, top_k: int, lanes: int) -> int:
    """VMEM words of one output slot: the block's outputs and room for the
    last row's whole-vector store."""
    return round_up(block_rows * top_k + lanes, SLOT_ALIGNMENT)


def routes_on_sparse_core(num_tokens: int, num_experts: int) -> bool:
    """Whether the batch holds enough router logits for sc_topk to outrun
    XLA's top_k despite its launch latency."""
    return num_tokens * num_experts >= SPARSE_CORE_MIN_LOGITS


def merge_tree(keys: list[jax.Array], descending: bool) -> jax.Array:
    """The top num_lanes of the key vectors, sorted."""
    if len(keys) == 1:
        return plsc.sort_key_val(keys[0], keys[0], descending=descending)[0]
    half = len(keys) // 2
    left = merge_tree(keys[:half], True)
    right = merge_tree(keys[half:], False)
    # Note (david): max(descending, ascending) is the top half of the union,
    # and bitonic.
    merged = jnp.maximum(left, right)
    return plsc.sort_key_val(merged, merged, descending=descending)[0]


def sc_topk_kernel(
    logits_hbm,  # bf16[num_tokens, num_experts], TensorCore tiling
    ids_hbm,  # int32[num_tokens * top_k]
    weights_hbm,  # f32[num_tokens * top_k]
    logits_v,  # VMEM int32[NUM_SLOTS, block_rows // 2, num_experts]
    ids_v,  # VMEM int32[NUM_SLOTS * output_slot_size]
    weights_v,  # VMEM f32[NUM_SLOTS * output_slot_size]
    sem,  # DMA sem[NUM_SLOTS]
    out_sem,  # DMA sem[NUM_SLOTS]
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    num_subcores: int,
    num_workers: int,
    lanes: int,
    should_pair_rows: bool,
    block_rows: int,
):
    worker = lax.axis_index("core") * num_subcores + lax.axis_index("subcore")
    lane_ids = jnp.arange(lanes, dtype=jnp.int32)
    is_kept = lane_ids < top_k
    words = logits_hbm.bitcast(jnp.int32)  # [num_tokens // 2, num_experts]
    word_rows = block_rows // 2
    out_size = block_rows * top_k
    slot_size = output_slot_size(block_rows, top_k, lanes)
    num_blocks = num_tokens // block_rows
    num_worker_blocks = (num_blocks - worker + num_workers - 1) // num_workers

    def _load(i, slot):
        block = worker + i * num_workers
        return pltpu.make_async_copy(
            words.at[
                pl.ds(pl.multiple_of(block * word_rows, word_rows), word_rows)
            ],
            logits_v.at[slot],
            sem.at[slot],
        )

    def _stores(i, slot):
        """The ids and weights write-back of this worker's block i."""
        block = worker + i * num_workers
        return [
            pltpu.make_async_copy(
                out_v.at[pl.ds(slot * slot_size, out_size)],
                out_hbm.at[pl.ds(block * out_size, out_size)],
                out_sem.at[slot],
            )
            for out_v, out_hbm in ((ids_v, ids_hbm), (weights_v, weights_hbm))
        ]

    def _emit(top, slot, row):
        top_words = top.view(jnp.int32)
        ids = LOW_HALF - (top_words & LOW_HALF)
        high = top_words & HIGH_HALF
        # Note (david): a set sign bit marks a positive logit (see _keys_of).
        flip = jnp.where(high < 0, POSITIVE_FLIP, NEGATIVE_FLIP)
        logits = (high ^ flip).view(jnp.float32)
        exp_logits = jnp.where(is_kept, jnp.exp(logits - logits[0]), 0.0)
        start = slot * slot_size + row * top_k
        plsc.store_compressed(ids_v.at[pl.ds(start, lanes)], ids, mask=is_kept)
        plsc.store_compressed(
            weights_v.at[pl.ds(start, lanes)],
            exp_logits / jnp.sum(exp_logits),
            mask=is_kept,
        )

    def _keys_of(slot, j, half):
        """uint32 keys of the row-parity half tokens of word row j: each
        logit's f32 bits (low half zero) flipped to an order-preserving key,
        with the inverted expert index in the low half."""
        keys = []
        for v in range(num_experts // lanes):
            word = logits_v[slot, j, pl.ds(v * lanes, lanes)]
            if half == 0:
                f32_bits = word << HALF_WORD_BITS
            else:
                f32_bits = word & HIGH_HALF
            flip = jnp.where(f32_bits < 0, NEGATIVE_FLIP, POSITIVE_FLIP)
            expert_keys = (LOW_HALF - v * lanes) - lane_ids
            keys.append(((f32_bits ^ flip) | expert_keys).view(jnp.uint32))
        return keys

    @pl.when(num_worker_blocks > 0)
    def _():
        _load(0, 0).start()

    @pl.loop(0, num_worker_blocks)
    def _(i):
        slot = lax.rem(i, NUM_SLOTS)

        @pl.when(i + 1 < num_worker_blocks)
        def _():
            _load(i + 1, 1 - slot).start()

        _load(i, slot).wait()

        @pl.when(i >= NUM_SLOTS)
        def _():
            for copy in _stores(i - NUM_SLOTS, slot):
                copy.wait()

        if should_pair_rows:

            @plsc.parallel_loop(0, word_rows)
            def _(j):
                _emit(merge_tree(_keys_of(slot, j, 0), True), slot, 2 * j)
                _emit(merge_tree(_keys_of(slot, j, 1), True), slot, 2 * j + 1)

        else:
            for half in (0, 1):

                @plsc.parallel_loop(0, word_rows)
                def _(j, half=half):
                    top = merge_tree(_keys_of(slot, j, half), True)
                    _emit(top, slot, 2 * j + half)

        for copy in _stores(i, slot):
            copy.start()

    for lag in range(NUM_SLOTS, 0, -1):

        @pl.when(num_worker_blocks >= lag)
        def _(lag=lag):
            i = num_worker_blocks - lag
            for copy in _stores(i, lax.rem(i, NUM_SLOTS)):
                copy.wait()


@functools.partial(
    jax.jit, static_argnames=("top_k", "should_pair_rows", "block_rows")
)
def sc_topk(
    router_logits: jax.Array,
    *,
    top_k: int,
    should_pair_rows: bool | None = None,
    block_rows: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """softmax(top_k(router_logits)) on SparseCore.

    Args:
        router_logits: bf16[num_tokens, E].
        top_k: Experts per token, at most 16.
        should_pair_rows: Build both tokens of a word row in one loop
            iteration instead of one pass per row parity. None pairs them
            from PAIR_ROWS_MIN_EXPERTS experts on, when every subcore runs
            MIN_PAIR_BLOCKS blocks.
        block_rows: Tokens per DMA block, a multiple of 16; num_tokens is
            padded to a multiple of it. None takes PAIR_BLOCK_ROWS for
            paired rows, else MAX_BLOCK_ROWS or fewer, so that a short batch
            spreads over every vector subcore.

    Returns:
        (topk_ids, topk_weights): int32 and f32 [num_tokens * top_k], each
        token's experts in descending logit order (lax.top_k's order, ties to
        the lower expert), the weights the softmax over the selected logits.
    """
    if router_logits.ndim != 2 or router_logits.dtype != jnp.bfloat16:
        raise ValueError(
            "router_logits must be bf16[num_tokens, E], got"
            f" {router_logits.dtype}{list(router_logits.shape)}"
        )
    num_tokens, num_experts = router_logits.shape
    sparse_core = pltpu.get_tpu_info().sparse_core
    if sparse_core is None:
        raise RuntimeError("sc_topk needs a TPU with SparseCores")
    max_top_k = min(sparse_core.num_lanes, num_experts)
    if not 1 <= top_k <= max_top_k:
        raise ValueError(f"top_k={top_k} must be in [1, {max_top_k}]")
    num_workers = sparse_core.num_cores * sparse_core.num_subcores
    if should_pair_rows is None:
        is_paired = (
            num_experts >= PAIR_ROWS_MIN_EXPERTS
            and num_tokens >= MIN_PAIR_BLOCKS * PAIR_BLOCK_ROWS * num_workers
        )
    else:
        is_paired = should_pair_rows
    if block_rows is not None:
        rows_per_block = block_rows
    elif is_paired:
        rows_per_block = PAIR_BLOCK_ROWS
    else:
        rows_per_block = min(
            MAX_BLOCK_ROWS,
            round_up(pl.cdiv(num_tokens, num_workers), BF16_TILE_ROWS),
        )
    if rows_per_block <= 0 or rows_per_block % BF16_TILE_ROWS:
        raise ValueError(
            f"block_rows={rows_per_block} must be a multiple of"
            f" {BF16_TILE_ROWS}"
        )
    # Note (david): padded experts have logit -inf, so a real expert wins
    # every tie with them (the lower index); padded tokens are dropped.
    padded_experts = round_up(num_experts, sparse_core.num_lanes)
    if padded_experts == num_experts:
        expert_padded_logits = router_logits
    else:
        expert_padded_logits = jnp.pad(
            router_logits,
            ((0, 0), (0, padded_experts - num_experts)),
            constant_values=-jnp.inf,
        )
    padded_tokens = round_up(num_tokens, rows_per_block)
    if padded_tokens == num_tokens:
        padded_logits = expert_padded_logits
    else:
        padded_logits = jnp.pad(
            expert_padded_logits, ((0, padded_tokens - num_tokens), (0, 0))
        )
    slot_size = output_slot_size(rows_per_block, top_k, sparse_core.num_lanes)
    output_bytes_per_pair = jnp.dtype(jnp.int32).itemsize + F32_BYTES
    kernel = pl.kernel(
        functools.partial(
            sc_topk_kernel,
            num_tokens=padded_tokens,
            num_experts=padded_experts,
            top_k=top_k,
            num_subcores=sparse_core.num_subcores,
            num_workers=num_workers,
            lanes=sparse_core.num_lanes,
            should_pair_rows=is_paired,
            block_rows=rows_per_block,
        ),
        out_type=(
            jax.ShapeDtypeStruct((padded_tokens * top_k,), jnp.int32),
            jax.ShapeDtypeStruct((padded_tokens * top_k,), jnp.float32),
        ),
        mesh=plsc.VectorSubcoreMesh(
            core_axis_name="core",
            subcore_axis_name="subcore",
            num_cores=sparse_core.num_cores,
        ),
        scratch_types=[
            pltpu.VMEM(
                (NUM_SLOTS, rows_per_block // 2, padded_experts), jnp.int32
            ),
            pltpu.VMEM((NUM_SLOTS * slot_size,), jnp.int32),
            pltpu.VMEM((NUM_SLOTS * slot_size,), jnp.float32),
            pltpu.SemaphoreType.DMA((NUM_SLOTS,)),
            pltpu.SemaphoreType.DMA((NUM_SLOTS,)),
        ],
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True, needs_layout_passes=False
        ),
        name=f"moe_sc_topk-t_{padded_tokens}-e_{padded_experts}-k_{top_k}",
        cost_estimate=pl.CostEstimate(
            flops=padded_tokens * padded_experts * FLOPS_PER_LOGIT,
            bytes_accessed=padded_tokens
            * (BF16_BYTES * padded_experts + output_bytes_per_pair * top_k),
            transcendentals=padded_tokens * top_k,
        ),
    )
    ids, weights = kernel(padded_logits)
    if padded_tokens == num_tokens:
        return ids, weights
    else:
        num_pairs = num_tokens * top_k
        return ids[:num_pairs], weights[:num_pairs]
