# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import functools
import itertools
import os
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from . import config

NUM_LANES = 128
BF16_SUBLANE_ROWS = 16
L2_NORM_EPS = 1e-6
BASE_BLOCK_SIZE = 4
BLOCKS_PER_LEVEL = 4
NUM_STATE_PHASES = 3


def profile_scope(name: str) -> contextlib.AbstractContextManager[None]:
    """XProf trace region, active only when KDA_PROFILE_SCOPES=1."""
    # Note (david): named scopes are scheduling barriers (about +19% per-tile
    # time), so they are for attribution only and must stay off for benchmarks.
    if os.environ.get("KDA_PROFILE_SCOPES") == "1":
        return jax.named_scope(name)
    else:
        return contextlib.nullcontext()


def l2_norm(vectors: jax.Array) -> jax.Array:
    # Note (david): rsqrt, since a divide by sqrt lowers to rsqrt + rcp and on
    # the lane-padded [heads, chunk, 1] norms every EUP push costs a whole vreg.
    return vectors * jax.lax.rsqrt(
        jnp.sum(vectors * vectors, axis=-1, keepdims=True, dtype=vectors.dtype)
        + L2_NORM_EPS)


def log_decay(
    gate: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    lower_bound: float | None,
) -> jax.Array:
    """Per-key-channel log decay of the KDA gate.

    -exp(a_log) * softplus(gate + dt_bias), or, with lower_bound set,
    lower_bound * sigmoid(exp(a_log) * (gate + dt_bias)).
    """
    biased_gate = gate + dt_bias
    decay_rate = jnp.exp(a_log)
    if lower_bound is None:
        return -decay_rate * jax.nn.softplus(biased_gate)
    else:
        return lower_bound * jax.nn.sigmoid(decay_rate * biased_gate)


def silu(pre_activation: jax.Array) -> jax.Array:
    # Note (david): the tanh form of x * sigmoid(x) is one EUP push, against
    # exp + rcp for jax.nn.silu.
    return 0.5 * pre_activation * (1.0 + jnp.tanh(0.5 * pre_activation))


def get_mask_dtype(dtype: jnp.dtype) -> jnp.dtype:
    """Integer dtype as wide as dtype, for iota masks over dtype data."""
    match jnp.dtype(dtype).itemsize:
        case 4:
            return jnp.int32
        case 2:
            return jnp.int16
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def pack_width(chunk: int, num_v_heads: int) -> int:
    """Heads packed side by side along lanes per [chunk, chunk] block row.

    128 // chunk when 16 | chunk | 128 and it divides num_v_heads, else 1.
    """
    # Note (david): a lone [chunk, chunk] f32 block is lane-padded to 128
    # lanes, so packing 128 // chunk heads halves the vregs at chunk 64 while
    # each dot stays one vreg column wide.
    if chunk % BF16_SUBLANE_ROWS or NUM_LANES % chunk:
        return 1
    elif num_v_heads % (NUM_LANES // chunk):
        return 1
    else:
        return NUM_LANES // chunk


def pack_lanes(per_head: jax.Array) -> jax.Array:
    """[groups, pack, rows, cols] -> [groups, rows, pack * cols]."""
    return jnp.concatenate([per_head[:, i] for i in range(per_head.shape[1])],
                           axis=-1)


def unpack_lanes(packed: jax.Array, pack: int) -> jax.Array:
    """[groups, rows, pack * cols] -> [groups * pack, rows, cols]."""
    groups, rows, width = packed.shape
    cols = width // pack
    per_head = [packed[..., i * cols:(i + 1) * cols] for i in range(pack)]
    return jnp.stack(per_head, axis=1).reshape(groups * pack, rows, cols)


def block_diag(packed: jax.Array, block: int) -> jax.Array:
    """[X_0 | .. | X_{P-1}] -> blockdiag(X_0, .., X_{P-1}).

    packed is [groups, rows, pack * block]; the result is [groups, pack * rows,
    pack * block], block i being one masked select of packed.
    """
    groups, rows, cols = packed.shape
    if cols == block:
        return packed
    else:
        lane = jax.lax.broadcasted_iota(jnp.int32, (groups, rows, cols), 2)
        lane_block = lane >> (block.bit_length() - 1)
        return jnp.concatenate([
            jnp.where(lane_block == i, packed, 0) for i in range(cols // block)
        ], axis=1)


def batched_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    precision: jax.lax.Precision | None = None,
) -> jax.Array:
    """[batch, m, k] @ [batch, k, n] -> [batch, m, n], accumulated in f32."""
    return jax.lax.dot(
        lhs,
        rhs,
        dimension_numbers=(((2, ), (1, )), ((0, ), (0, ))),
        precision=precision,
        preferred_element_type=jnp.float32,
    )


def stacked_dots(
    lhs_parts: list[jax.Array],
    rhs: jax.Array,
    dot: Callable[[jax.Array, jax.Array], jax.Array] = batched_matmul,
) -> list[jax.Array]:
    """dot(part, rhs) for every part, as one stacked dot when rows align."""
    # Note (david): the MXU instruction stream is the critical path and Mosaic
    # shares a weight push only between adjacent dots, so parts sharing rhs are
    # stacked into one dot; the sublane concat needs 16-row bf16 tiles.
    if all(part.shape[1] % BF16_SUBLANE_ROWS == 0 for part in lhs_parts):
        part_rows = [part.shape[1] for part in lhs_parts]
        stacked = dot(jnp.concatenate(lhs_parts, axis=1), rhs)
        return jnp.split(stacked, list(itertools.accumulate(part_rows))[:-1],
                         axis=1)
    else:
        return [dot(part, rhs) for part in lhs_parts]


def dot_per_block(lhs: jax.Array, rhs: jax.Array, block: int) -> jax.Array:
    """Per head lhs_h @ rhs_h, with lhs_h lane block h of a packed row.

    lhs is [groups, rows, pack * block] bf16 and rhs [groups * pack, block, n]
    bf16; returns [groups * pack, rows, n] f32.
    """
    groups, rows, cols = lhs.shape
    pack = cols // block
    rhs_per_head = rhs.reshape(groups, pack, block, -1)
    out_cols = rhs_per_head.shape[-1]
    # Note (david): only lane block 0 is an aligned slice and slicing block
    # i > 0 costs an XLU lane rotate on the dependency chain, so the whole
    # packed row streams against rhs_h zero-padded to rows [i * block,
    # (i + 1) * block), at pack - 1 extra weight pushes per head.
    per_head_outs = [batched_matmul(lhs[:, :, :block], rhs_per_head[:, 0])]
    for i in range(1, pack):
        leading_zeros = jnp.zeros((groups, i * block, out_cols),
                                  rhs_per_head.dtype)
        if i + 1 < pack:
            padded_rhs_parts = [
                leading_zeros,
                rhs_per_head[:, i],
                jnp.zeros((groups, (pack - 1 - i) * block, out_cols),
                          rhs_per_head.dtype),
            ]
        else:
            padded_rhs_parts = [leading_zeros, rhs_per_head[:, i]]
        per_head_outs.append(
            batched_matmul(lhs, jnp.concatenate(padded_rhs_parts, axis=1)))
    return jnp.stack(per_head_outs, axis=1).reshape(groups * pack, rows,
                                                    out_cols)


def invert_triangular_matrix(
    unit_lower: jax.Array,
    pack: int = 1,
    fill: Callable[[int, int], None] | None = None,
) -> jax.Array:
    """Inverse of unit lower triangular blocks, [groups, chunk, pack * chunk].

    pack heads sit side by side along lanes, one [chunk, chunk] block each.
    fill(slot, num_slots), if given, runs after each dot is issued so the
    caller can emit independent MXU work into its result-latency gap.
    """
    groups, chunk, cols = unit_lower.shape
    assert cols == pack * chunk, unit_lower.shape
    dtype = unit_lower.dtype
    # Note (david): with L strictly lower, (I + L)^-1 = (I - L)(I + L^2)
    # (I + L^4) ... runs as batched MXU dots, ~1.0 us per 64-token tile on v6e
    # against 2.0 for a row-serial VPU forward substitution. Its intermediates
    # hold binomial sums of key correlations (~1e17 over 64 repeated keys at
    # beta 1) that cancel only in the last product, losing the small inverse
    # entries to mantissa precision. So the series runs on 4 x 4 diagonal
    # blocks (worst intermediate 2), and each level combines 4 blocks as
    # T^-1 = D^-1 (I - N)(I + N^2) with N = E D^-1 and N^4 == 0. Every operand
    # stays bounded like the inverse, so plain bf16 dots reach 9e-3 max abs
    # error against f64 on repeated keys.
    if chunk % BF16_SUBLANE_ROWS == 0:
        level_sizes = [BASE_BLOCK_SIZE]
        while level_sizes[-1] < chunk:
            level_sizes.append(min(level_sizes[-1] * BLOCKS_PER_LEVEL, chunk))
        operand_dtype, precision = jnp.bfloat16, None
    else:
        # Note (david): a chunk below 16 rows cannot fill a bf16 sublane tile,
        # so it runs one level with input-dtype operands at HIGHEST precision.
        assert chunk < BF16_SUBLANE_ROWS, chunk
        level_sizes = [chunk]
        operand_dtype, precision = dtype, jax.lax.Precision.HIGHEST

    def _chain_levels(size: int) -> int:
        return max((size - 1).bit_length() - 1, 0)

    # Note (david): fill gets one slot per dot, so num_slots counts the base
    # squaring and chain products, then per block level the N dot, the stacked
    # D^-1 N dot and that level's chain products.
    levels = _chain_levels(level_sizes[0])
    num_slots = (levels + 1 if levels else 0) + sum(
        _chain_levels(hi // lo) + 2
        for lo, hi in itertools.pairwise(level_sizes))
    slot_counter = itertools.count()

    def _emit() -> None:
        if fill is not None:
            fill(next(slot_counter), num_slots)

    def _dot(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
        return batched_matmul(lhs.astype(operand_dtype),
                              rhs.astype(operand_dtype),
                              precision).astype(dtype)

    def _product_chain(product: jax.Array, power: jax.Array, num_levels: int,
                       block: int) -> jax.Array:
        # Note (david): the squaring for the next level shares the product's
        # weight, so both ride one stacked dot.
        for level in range(num_levels):
            power = power.astype(operand_dtype)
            power_diag = block_diag(power, block)
            if level + 1 < num_levels:
                product_times_power, power = stacked_dots([product, power],
                                                          power_diag, _dot)
            else:
                (product_times_power, ) = stacked_dots([product], power_diag,
                                                       _dot)
            _emit()
            product = product + product_times_power
        return product

    def _lane_iota(rows: int) -> jax.Array:
        lane = jax.lax.broadcasted_iota(jnp.int32, (groups, rows, cols), 2)
        return lane & (chunk - 1) if pack > 1 else lane

    def _row_iota(rows: int) -> jax.Array:
        return jax.lax.broadcasted_iota(jnp.int32, (groups, rows, cols), 1)

    def _diagonal_blocks(size: int) -> jax.Array:
        if size == chunk:
            return unit_lower
        else:
            lane_block = _lane_iota(size) >> (size.bit_length() - 1)
            blocks = unit_lower[:, :size]
            for i in range(1, chunk // size):
                blocks = jnp.where(lane_block == i,
                                   unit_lower[:, i * size:(i + 1) * size],
                                   blocks)
            return blocks

    def _sub_block_index(rows: int, block: int, enclosing: int) -> jax.Array:
        index = _lane_iota(rows) >> (block.bit_length() - 1)
        return index & (enclosing // block - 1) if enclosing < chunk else index

    # Note (david): the base level holds the chunk's 16 x 16 diagonal blocks in
    # one [groups, 16, cols] slab, a bf16 sublane tile, and masks out the
    # slab's off-diagonal base blocks.
    block_size = level_sizes[0]
    slab_rows = min(chunk, BF16_SUBLANE_ROWS)
    lane = _lane_iota(slab_rows)
    lane_in_slab = lane & (slab_rows - 1) if slab_rows < chunk else lane
    row = _row_iota(slab_rows)
    identity = jnp.where(row == lane_in_slab, 1, 0).astype(dtype)
    slab_lower = _diagonal_blocks(slab_rows) - identity
    if block_size < slab_rows:
        shift = block_size.bit_length() - 1
        strictly_lower = jnp.where(row >> shift == lane_in_slab >> shift,
                                   slab_lower, 0)
    else:
        strictly_lower = slab_lower
    base_inverse = identity - strictly_lower
    if levels:
        strictly_lower_diag = block_diag(strictly_lower.astype(operand_dtype),
                                         slab_rows)
        power = _dot(strictly_lower, strictly_lower_diag)
        _emit()
        inverse = _product_chain(base_inverse, power, levels, slab_rows)
    else:
        inverse = base_inverse

    for next_size in level_sizes[1:]:
        num_blocks = next_size // block_size
        shift = block_size.bit_length() - 1
        lane_block = _sub_block_index(next_size, block_size, next_size)
        row_block = _row_iota(next_size) >> shift
        if inverse.shape[1] == next_size:
            d_inv = inverse
        else:
            inverse_lane_block = _sub_block_index(block_size, block_size,
                                                  next_size)
            d_inv = jnp.concatenate([
                jnp.where(inverse_lane_block == i, inverse, 0)
                for i in range(num_blocks)
            ], axis=1)
        strict_block_lower = jnp.where(row_block > lane_block,
                                       _diagonal_blocks(next_size), 0)
        d_inv_operand = d_inv.astype(operand_dtype)
        nilpotent = _dot(strict_block_lower,
                         block_diag(d_inv_operand,
                                    next_size)).astype(operand_dtype)
        _emit()
        nilpotent_diag = block_diag(nilpotent, next_size)
        block_levels = _chain_levels(num_blocks)
        # Note (david): D^-1 (I - N) and the N^2 the chain needs next share N's
        # weight, so they run as one stacked dot.
        if block_levels:
            d_inv_nilpotent, power = stacked_dots([d_inv_operand, nilpotent],
                                                  nilpotent_diag, _dot)
            _emit()
            inverse = _product_chain(d_inv - d_inv_nilpotent, power,
                                     block_levels, next_size)
        else:
            (d_inv_nilpotent, ) = stacked_dots([d_inv_operand], nilpotent_diag,
                                               _dot)
            _emit()
            inverse = d_inv - d_inv_nilpotent
        block_size = next_size
    return inverse


def cumsum_rows(increments: jax.Array) -> jax.Array:
    """Inclusive cumsum along axis 1, accumulated in f32."""
    # Note (david): a log-depth scan of whole-slab sublane rolls, one XLU
    # rotate per vreg per level; a row-serial add chain is XLU-shuffle bound
    # (~500 bundles per 64-token tile on v6e). The roll needs 32-bit lanes,
    # and the tree order fixes the rounding, so another cumsum need not match
    # it bitwise.
    dtype = increments.dtype
    prefix_sums = increments.astype(jnp.float32)
    chunk = prefix_sums.shape[1]
    row = jax.lax.broadcasted_iota(jnp.int32, prefix_sums.shape, 1)
    shift = 1
    while shift < chunk:
        prefix_sums = prefix_sums + jnp.where(
            row >= shift, pltpu.roll(prefix_sums, shift, 1), 0)
        shift *= 2
    return prefix_sums.astype(dtype)


def fused_transpose_broadcast(values: jax.Array, src_dim: int,
                              dst_dim: int) -> jax.Array:
    """Moves axis src_dim of values into its size-1 axis dst_dim.

    Axis src_dim of the result has size 1; the masked reduction stays in the
    dtype of values.
    """
    assert values.shape[dst_dim] == 1
    dtype = values.dtype
    mask_dtype = get_mask_dtype(dtype)
    mask_shape = list(values.shape)
    mask_shape[dst_dim] = mask_shape[src_dim]
    src_iota = jax.lax.broadcasted_iota(mask_dtype, mask_shape, src_dim)
    dst_iota = jax.lax.broadcasted_iota(mask_dtype, mask_shape, dst_dim)
    is_same_index = src_iota == dst_iota
    return jnp.where(is_same_index, values, 0).sum(axis=src_dim,
                                                   keepdims=True,
                                                   dtype=dtype)


def prepare_chunk(
    row_lo: jax.Array,
    row_hi: jax.Array,
    q_chunked: jax.Array,
    k_chunked: jax.Array,
    v_chunked: jax.Array,
    b_chunked: jax.Array,
    g_chunked: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.KDAConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Masks one chunk to its real rows [row_lo, row_hi) and derives the gates.

    q_chunked and k_chunked are [seq, num_kq_heads, chunk, kq_head_dim],
    v_chunked is [seq, num_v_heads, chunk, v_head_dim], g_chunked is [seq,
    num_v_heads, chunk, kq_head_dim], and b_chunked is [seq, 1, chunk,
    gate_lanes] with the value heads on lanes, padded to
    cfg.aligned_num_v_heads. row_lo is nonzero only for the pad rows before a
    sequence's first token on the block grid. Returns (q, k, v, g, beta) in
    the compute dtype: q, k, v masked but not normalized, the per-key-channel
    log decay g shaped like g_chunked, and beta shaped like b_chunked.
    """
    assert cfg.mode == config.KDAMode.PER_SEQ

    mask_dtype = get_mask_dtype(cfg.dtypes.compute)
    token_iota = jax.lax.broadcasted_iota(
        mask_dtype, (cfg.seq_tile_size, 1, cfg.compute_chunk_size, 1), 2)
    is_real_token = jnp.logical_and(
        token_iota >= row_lo.reshape(-1, 1, 1, 1).astype(mask_dtype),
        token_iota < row_hi.reshape(-1, 1, 1, 1).astype(mask_dtype))

    with profile_scope("kda_mask"):
        q = jnp.where(is_real_token, q_chunked.astype(cfg.dtypes.compute), 0)
        k = jnp.where(is_real_token, k_chunked.astype(cfg.dtypes.compute), 0)
        v = jnp.where(is_real_token, v_chunked.astype(cfg.dtypes.compute), 0)

    b_chunked = b_chunked.astype(cfg.dtypes.compute)
    g_chunked = g_chunked.astype(cfg.dtypes.compute)

    a_log = a_log[None].astype(cfg.dtypes.compute)
    dt_bias = dt_bias[None].astype(cfg.dtypes.compute)

    with profile_scope("gates"):
        beta = b_chunked if cfg.beta_is_activated else jax.nn.sigmoid(b_chunked)
        g = log_decay(g_chunked, a_log, dt_bias, cfg.lower_bound)

        beta = jnp.where(is_real_token, beta, 0)
        # Note (david): pad rows get log decay 0, a decay of exp(0) = 1, so
        # they carry the previous state through unchanged.
        g = jnp.where(is_real_token, g, 0)

    return q, k, v, g, beta


def pack_out(out: jax.Array, cfg: config.KDAConfig) -> jax.Array:
    """[num_v_heads, chunk, v_head_dim] -> [chunk, num_v_heads * v_head_dim]."""
    # Note (david): each head is exactly one 128-lane tile, so gluing heads
    # along lanes is vreg naming, where swapaxes(0, 1) would be a real sublane
    # transpose.
    return jnp.concatenate([out[h] for h in range(cfg.num_v_heads)], axis=-1)


# Note (david): the phases are separate methods because the MXU issues in
# program order, so a dot fills another's latency gap only if emitted there;
# run_pipelined interleaves a tile's chunks through them.
class ChunkKDA:
    """One chunk of the Kimi delta rule for one sequence.

    q_chunked and k_chunked are [num_kq_heads, chunk, kq_head_dim], masked but
    not normalized; v_chunked is [num_v_heads, chunk, v_head_dim]; the
    per-key-channel log decay g is [num_v_heads, chunk, kq_head_dim] and beta
    [1, chunk, gate_lanes], value heads on lanes. With G[i, d] the cumulative
    log decay of key channel d at row i,

        Akk[i, j] = beta_i sum_d k[i,d] k[j,d] exp(G[i,d] - G[j,d])  (i > j)
        Aqk[i, j] =        sum_d q[i,d] k[j,d] exp(G[i,d] - G[j,d])  (i >= j)

    and T = I + Akk gives the WY operands u = T^-1 (v beta) and
    w = T^-1 (k beta exp(G)); the [kq_head_dim, v_head_dim] state decays per
    key row. q carries l2 normalization and kq_head_dim**-0.5, k l2
    normalization. Value head h reads kq head h // cfg.v_per_kq_head, and a
    head group holds pack adjacent value heads. The phases kkt_qk, tinv, uw,
    state and state_out run in that order; the last three take a half-open
    head-group range [lo, hi).
    """

    def __init__(
        self,
        q_chunked: jax.Array,
        k_chunked: jax.Array,
        v_chunked: jax.Array,
        g: jax.Array,
        beta: jax.Array,
        cfg: config.KDAConfig,
    ) -> None:
        self.cfg = cfg
        self.compute_dtype = cfg.dtypes.compute

        # Note (david): k is normalized first because the chunk's first dot
        # waits on it for the weight k^T and the leading LHS rows; q only feeds
        # the trailing rows of that stacked kkt / qk dot.
        with profile_scope("l2norm_k"):
            k_normed = l2_norm(k_chunked)
        # Note (david): repeating the kq heads over their v heads (GQA) along a
        # non lane / sublane dim is free.
        k_repeat = jnp.repeat(k_normed, cfg.v_per_kq_head, axis=0)

        with profile_scope("g_cumsum"):
            # Note (david): the per-channel cumsum is lane-dense as is, so no
            # gate here needs a transpose before its exp.
            g_cumsum = cumsum_rows(g)
            self.decay_from_start = jnp.exp(g_cumsum)
            g_last = g_cumsum[:, -1:]
            decay_to_end = jnp.exp(g_last - g_cumsum)
            # Note (david): the state decays per key row, so the chunk decay
            # exp(G_last) moves to [num_v_heads, kq_head_dim, 1].
            self.decay_chunk = fused_transpose_broadcast(
                jnp.exp(g_last), src_dim=2, dst_dim=1)
            beta_col = fused_transpose_broadcast(
                beta, src_dim=2, dst_dim=0)[:cfg.num_v_heads]

        # Note (david): pair matrices are packed as [groups, chunk, pack *
        # chunk], entry [group, i, p * chunk + j] being head group * pack + p at
        # token pair (i, j); block-diagonal weights keep heads from mixing.
        chunk = self.chunk = cfg.compute_chunk_size
        pack = self.pack = pack_width(chunk, cfg.num_v_heads)
        groups = self.groups = cfg.num_v_heads // pack
        pair_shape = (groups, chunk, pack * chunk)

        with profile_scope("masks"):
            mask_dtype = get_mask_dtype(self.compute_dtype)
            row_idx = jax.lax.broadcasted_iota(mask_dtype, pair_shape, 1)
            lane_idx = jax.lax.broadcasted_iota(mask_dtype, pair_shape, 2)
            col_idx = lane_idx & (chunk - 1) if pack > 1 else lane_idx
            self.is_diagonal = row_idx == col_idx

        # Note (david): bf16 MXU operands pack 16 rows per vreg against f32's 8,
        # and every cast below sets a rounding boundary of the kernel.
        bf16 = jnp.bfloat16

        with profile_scope("k_transpose"):
            # Note (david): k is transposed here, one f32 XLU transpose per
            # group, because a transposed dot RHS lowers to vmatpush.xpose at 7
            # cycles per push against 3.
            self.k_t = self.transposed(k_repeat)
            self.k_t_weight = block_diag(self.k_t.astype(bf16), chunk)
            # Note (david): a per-channel gate cannot be folded in after the
            # transpose, so k is scaled by exp(G_last - G_j) before its own.
            self.k_decayed_t = self.transposed(
                k_repeat * decay_to_end).astype(bf16)

        with profile_scope("l2norm_q"):
            q_normed = l2_norm(q_chunked) * cfg.kq_head_dim**-0.5
        self.q_repeat = jnp.repeat(q_normed, cfg.v_per_kq_head, axis=0)

        k_beta = k_repeat * beta_col
        # Note (david): one [v beta | k beta exp(G)] weight raises MXU
        # utilization when v_head_dim is below the MXU size, and the lane
        # concat / split is free when v_head_dim is a multiple of the lane
        # count.
        self.uw_rhs = jnp.concatenate(
            [v_chunked * beta_col, k_beta * self.decay_from_start],
            axis=-1).astype(bf16)
        with profile_scope("qk_kk_operands"):
            self.qk_kk_operands = self.prepare_qk_kk_operands(
                self.q_repeat, k_beta, k_repeat, g_cumsum)
        self.akk_parts: list[jax.Array] = []
        self.aqk_parts: list[jax.Array] = []
        self.num_kkt_groups_done = 0
        self.uw_parts: dict[tuple[int, int], list[jax.Array]] = {}
        self.state_read_parts: dict[tuple[int, int],
                                    tuple[jax.Array, jax.Array]] = {}
        self.out_parts: dict[tuple[int, int], tuple[jax.Array, jax.Array]] = {}

    def transposed(self, per_head: jax.Array) -> jax.Array:
        """[num_v_heads, chunk, dim] -> [groups, dim, pack * chunk].

        The pack's per-head transposes sit side by side along lanes.
        """
        return jnp.swapaxes(
            per_head.reshape(self.groups, self.pack, self.chunk, -1).reshape(
                self.groups, self.pack * self.chunk, -1), 1, 2)

    def prepare_qk_kk_operands(
        self,
        q: jax.Array,
        k_beta: jax.Array,
        k: jax.Array,
        g_cumsum: jax.Array,
    ) -> list[tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
        """Decayed q / k operands whose masked dots sum to Aqk and Akk.

        q, k_beta = beta * k, k and g_cumsum are [heads, chunk, K]. Every causal
        pair (i, j) is covered once, at one level of the block hierarchy
        1 | 4 | 16 | ... | chunk, by a dot of

            q_decayed = q * exp(G_i - G_ref)
            k_beta_decayed = beta * k * exp(G_i - G_ref)
            k_decayed_t = (k * exp(G_ref - G_j))^T

        with G_ref the last row of the key's sub-chunk. Returns per dot
        (q_decayed, k_beta_decayed, k_decayed_t, is_pair_in_dot): operands
        [groups, chunk, pack * K] bf16, the block-diagonal weight [groups,
        pack * K, pack * chunk] bf16, and the mask [1, chunk, pack * chunk] of
        the pairs the dot contributes.
        """
        # Note (david): a per-channel gate does not factor out of the dot.
        # Splitting at G_ref keeps both exponents nonpositive on contributing
        # pairs, since j <= ref <= i and exact prefix sums of g <= 0 are
        # non-increasing, whereas exp(-G_j) alone can overflow even when the
        # gated product is small. Unused exponents are zeroed before the exp.
        # Gathering only the masked rows would halve the streamed rows but
        # gains just ~7% end to end (not MXU-stream bound) for a harder read.
        chunk, pack = self.chunk, self.pack
        heads, _, key_dim = q.shape
        bf16 = jnp.bfloat16
        pair_shape = (1, chunk, pack * chunk)
        row = jax.lax.broadcasted_iota(jnp.int32, pair_shape, 1)
        lane = jax.lax.broadcasted_iota(jnp.int32, pair_shape, 2)
        col = lane & (chunk - 1) if pack > 1 else lane
        query_rows = jax.lax.broadcasted_iota(jnp.int32, (1, chunk, 1), 1)

        def _block_index(position: jax.Array, block_width: int) -> jax.Array:
            return position >> (block_width.bit_length() - 1)

        def _g_reference(block: int, offset: int) -> jax.Array:
            ref_rows = [
                jnp.broadcast_to(g_cumsum[:, ref_row:ref_row + 1],
                                 (heads, block, key_dim))
                for ref_row in range(offset, chunk, block)
            ]
            if len(ref_rows) == 1:
                return ref_rows[0]
            else:
                return jnp.concatenate(ref_rows, axis=1)

        level_sizes = [1]
        while level_sizes[-1] < chunk:
            level_sizes.append(min(level_sizes[-1] * BLOCKS_PER_LEVEL, chunk))
        operands = []
        for sub_chunk_size, block_size in itertools.pairwise(level_sizes):
            is_bottom_level = sub_chunk_size == 1
            if is_bottom_level:
                # Note (david): the bottom level's exponent is 0, so its weight
                # is plain k.
                k_decayed_t = self.k_t_weight
            else:
                k_decayed = k * jnp.exp(
                    _g_reference(sub_chunk_size, sub_chunk_size - 1) - g_cumsum)
                k_decayed_t = block_diag(
                    self.transposed(k_decayed).astype(bf16), chunk)
            query_sub_chunk_rows = _block_index(query_rows & (block_size - 1),
                                                sub_chunk_size)
            query_sub_chunk = _block_index(row & (block_size - 1),
                                           sub_chunk_size)
            key_sub_chunk = _block_index(col & (block_size - 1), sub_chunk_size)
            is_same_enclosing_block = (_block_index(row, block_size)
                                       == _block_index(col, block_size))
            num_sub_chunks = block_size // sub_chunk_size
            # Note (david): higher levels cover only pairs across sub-chunks,
            # so their last key sub-chunk has no later query sub-chunk; the
            # bottom level also covers i == j, so every causal pair appears
            # once.
            num_key_sub_chunks = (num_sub_chunks
                                  if is_bottom_level else num_sub_chunks - 1)
            for key_sub_chunk_idx in range(num_key_sub_chunks):
                is_valid_query_row = (
                    query_sub_chunk_rows >= key_sub_chunk_idx
                    if is_bottom_level else
                    query_sub_chunk_rows > key_sub_chunk_idx)
                is_pair_in_dot = (
                    is_same_enclosing_block
                    & (key_sub_chunk == key_sub_chunk_idx)
                    & (query_sub_chunk >= key_sub_chunk_idx
                       if is_bottom_level else
                       query_sub_chunk > key_sub_chunk_idx))
                g_ref = _g_reference(
                    block_size, sub_chunk_size * (key_sub_chunk_idx + 1) - 1)
                decay = jnp.exp(
                    jnp.where(is_valid_query_row, g_cumsum - g_ref, 0))
                q_decayed = pack_lanes((q * decay).astype(bf16).reshape(
                    self.groups, pack, chunk, -1))
                k_beta_decayed = pack_lanes(
                    (k_beta * decay).astype(bf16).reshape(
                        self.groups, pack, chunk, -1))
                operands.append(
                    (q_decayed, k_beta_decayed, k_decayed_t, is_pair_in_dot))
        return operands

    def kkt_qk(self, hi: int | None = None) -> None:
        """Emits Akk and Aqk of groups [done, hi) from the decayed operands."""
        lo = self.num_kkt_groups_done
        hi = self.groups if hi is None else hi
        if hi <= lo:
            return
        with profile_scope("kkt_qk"):
            for operand_idx, (q_decayed, k_beta_decayed, k_decayed_t,
                              is_pair_in_dot) in enumerate(
                                  self.qk_kk_operands):
                akk_part, aqk_part = stacked_dots(
                    [k_beta_decayed[lo:hi], q_decayed[lo:hi]],
                    k_decayed_t[lo:hi])
                akk_part = jnp.where(is_pair_in_dot, akk_part, 0)
                aqk_part = jnp.where(is_pair_in_dot, aqk_part, 0)
                if operand_idx == 0:
                    akk, aqk = akk_part, aqk_part
                else:
                    akk, aqk = akk + akk_part, aqk + aqk_part
            self.akk_parts.append(akk)
            self.aqk_parts.append(aqk)
            self.num_kkt_groups_done = hi

    def tinv(self, fill: Callable[[int, int], None] | None = None) -> None:
        """Inverts T = I + Akk and collects Aqk.

        fill(slot, num_slots) runs after each dot of the inversion chain.
        """
        self.kkt_qk()
        with profile_scope("kkt_gate"):
            # Note (david): Akk and Aqk arrive gated and lower triangular, so
            # only Akk's k_i . k_i diagonal gives way to T's unit diagonal.
            akk = jnp.concatenate(self.akk_parts,
                                  axis=0).astype(self.compute_dtype)
            t_matrix = jnp.where(self.is_diagonal, 1, akk)

        with profile_scope("tinv"):
            self.t_inv = invert_triangular_matrix(t_matrix, self.pack, fill)

        with profile_scope("qk_gate"):
            self.aqk = jnp.concatenate(self.aqk_parts,
                                       axis=0).astype(self.compute_dtype)

    def value_head_slice(self, lo: int, hi: int) -> slice:
        """Value heads of head groups [lo, hi)."""
        return slice(lo * self.pack, hi * self.pack)

    def uw(self, lo: int, hi: int) -> None:
        """u = T^-1 (v beta) and w = T^-1 (k beta exp(G)) of groups [lo, hi)."""
        with profile_scope("uw"):
            uw_concat = dot_per_block(
                self.t_inv[lo:hi].astype(jnp.bfloat16),
                self.uw_rhs[self.value_head_slice(lo, hi)],
                self.chunk).astype(self.compute_dtype)
            self.uw_parts[(lo, hi)] = jnp.split(uw_concat,
                                                [self.cfg.v_head_dim],
                                                axis=-1)

    def state(self, state_prev: jax.Array, lo: int, hi: int) -> None:
        """Reads the incoming state S for groups [lo, hi), after their uw.

        out_inter = q exp(G) @ S and v_new = u - w @ S share S's weight push.
        """
        bf16 = jnp.bfloat16
        head_slice = self.value_head_slice(lo, hi)
        self.state_prev = state_prev
        with profile_scope("state"):
            u, w = self.uw_parts[(lo, hi)]
            q_decayed = (self.q_repeat[head_slice]
                         * self.decay_from_start[head_slice])
            # Note (david): only the MXU copy of S is bf16; state_out decays the
            # original S.
            state_reads = batched_matmul(
                jnp.concatenate([w, q_decayed], axis=1).astype(bf16),
                state_prev[head_slice].astype(bf16))
            w_state, out_inter = jnp.split(state_reads, 2, axis=1)
            # Note (david): the residual is formed in the compute dtype before
            # it is rounded to bf16.
            v_new = (u - w_state.astype(self.compute_dtype)).astype(bf16)
            self.state_read_parts[(lo, hi)] = (out_inter, v_new)

    def state_out(self, lo: int, hi: int) -> None:
        """Outgoing state and output of groups [lo, hi), after their state.

        S_out = exp(G_last) S + sum_j (k_j exp(G_last - G_j))^T v_new_j, with
        exp(G_last) per key row, and out_i = out_inter_i
        + sum_{j <= i} Aqk[i, j] v_new_j.
        """
        bf16 = jnp.bfloat16
        head_slice = self.value_head_slice(lo, hi)
        with profile_scope("state_out"):
            out_inter, v_new = self.state_read_parts[(lo, hi)]
            k_decayed_t = self.k_decayed_t[lo:hi]
            aqk_operand = self.aqk[lo:hi].astype(bf16)
            # Note (david): both dots take the weight v_new, so they run as one
            # dot with a stacked LHS against blockdiag(v_new_h), pushing the
            # pack's weights once rather than once per head. A head is a whole
            # vreg wide, so the block diagonal is a lane-tile concat with zeros
            # (vreg naming) rather than block_diag's selects.
            v_new_per_head = v_new.reshape(hi - lo, self.pack, self.chunk, -1)
            zeros = jnp.zeros_like(v_new_per_head[:, 0])
            v_new_block_diag = jnp.concatenate([
                jnp.concatenate([zeros] * i + [v_new_per_head[:, i]]
                                + [zeros] * (self.pack - 1 - i),
                                axis=-1) for i in range(self.pack)
            ], axis=1)
            state_update, out_intra = stacked_dots([k_decayed_t, aqk_operand],
                                                   v_new_block_diag)
            state = (self.state_prev[head_slice] * self.decay_chunk[head_slice]
                     + unpack_lanes(state_update, self.pack))
            out = out_inter + unpack_lanes(out_intra, self.pack)
            self.out_parts[(lo, hi)] = (out, state)


def run_pipelined(
    chunks: list[ChunkKDA],
    state_prev: jax.Array,
) -> tuple[list[jax.Array], jax.Array]:
    """Runs consecutive chunks of one sequence with their phases interleaved.

    Returns every chunk's out [num_v_heads, chunk, v_head_dim] and the final
    state [num_v_heads, kq_head_dim, v_head_dim].
    """
    # Note (david): chunk c's tinv chain waits ~200 cycles between dependent
    # rounds and the MXU issues in program order, so its gaps take chunk
    # c + 1's kkt / qk dots, finished one round early so the next chain never
    # waits on them, and chunk c - 1's uw, state and state_out, each phase
    # sliced by head group so a dependent phase sits rounds behind. Callers
    # emit every chunk's conv1d first, so chunk c + 1's norms and transposes
    # already cover the XLU when the first gap opens.
    chunk_outputs: list[jax.Array] = []
    state_carry = state_prev

    def _build_state_steps(chunk: ChunkKDA,
                           num_slots: int) -> list[Callable[[], None]]:
        num_pieces = max(1, num_slots // NUM_STATE_PHASES)
        group_bounds = sorted({
            piece * chunk.groups // num_pieces
            for piece in range(num_pieces + 1)
        })
        group_ranges = list(itertools.pairwise(group_bounds))
        state_steps = [
            functools.partial(chunk.uw, lo, hi) for lo, hi in group_ranges
        ]
        # Note (david): the incoming state is bound now, since _finish_chunk
        # rebinds state_carry.
        state_steps += [
            functools.partial(chunk.state, state_carry, lo, hi)
            for lo, hi in group_ranges
        ]
        state_steps += [
            functools.partial(chunk.state_out, lo, hi)
            for lo, hi in group_ranges
        ]

        def _finish_chunk() -> None:
            nonlocal state_carry
            done_ranges = sorted(chunk.out_parts)
            assert sum(hi - lo for lo, hi in done_ranges) == chunk.groups, (
                done_ranges)
            chunk_outputs.append(
                jnp.concatenate(
                    [chunk.out_parts[group_range][0]
                     for group_range in done_ranges], axis=0))
            state_carry = jnp.concatenate(
                [chunk.out_parts[group_range][1]
                 for group_range in done_ranges], axis=0)

        return state_steps + [_finish_chunk]

    chunks[0].kkt_qk()
    for chunk_idx, chunk in enumerate(chunks):
        previous_chunk = chunks[chunk_idx - 1] if chunk_idx > 0 else None
        next_chunk = (chunks[chunk_idx + 1]
                      if chunk_idx + 1 < len(chunks) else None)
        pending_state_steps: list[Callable[[], None]] | None = None
        pending_kkt_groups: list[int] = []

        # Note (david): tinv calls _fill synchronously within this iteration,
        # so closing over this iteration's chunks is safe.
        def _fill(slot: int, num_slots: int) -> None:
            nonlocal pending_state_steps, pending_kkt_groups
            # Note (david): the chain reports its gap count only at slot 0, so
            # both queues are sized, and the incoming state bound, there.
            if slot == 0:
                if previous_chunk is None:
                    pending_state_steps = []
                else:
                    pending_state_steps = _build_state_steps(
                        previous_chunk, num_slots)
                if next_chunk is not None and num_slots > 1:
                    pending_kkt_groups = sorted({
                        piece * next_chunk.groups // (num_slots - 1)
                        for piece in range(num_slots)
                    })[1:]
                else:
                    pending_kkt_groups = []
            if pending_state_steps:
                pending_state_steps.pop(0)()
            if pending_kkt_groups:
                next_chunk.kkt_qk(pending_kkt_groups.pop(0))

        chunk.tinv(_fill)
        # Note (david): a chain without gaps never calls _fill, so the previous
        # chunk's state chain runs here in full.
        if pending_state_steps is not None:
            remaining_state_steps = pending_state_steps
        elif previous_chunk is not None:
            remaining_state_steps = _build_state_steps(previous_chunk,
                                                       NUM_STATE_PHASES)
        else:
            remaining_state_steps = []
        for step in remaining_state_steps:
            step()
        if next_chunk is not None:
            next_chunk.kkt_qk()
    for step in _build_state_steps(chunks[-1], NUM_STATE_PHASES):
        step()
    return chunk_outputs, state_carry


def chunked_kda(
    row_lo: jax.Array,
    row_hi: jax.Array,
    q_chunked: jax.Array,
    k_chunked: jax.Array,
    v_chunked: jax.Array,
    b_chunked: jax.Array,
    g_chunked: jax.Array,
    state_prev: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.KDAConfig,
) -> tuple[jax.Array, jax.Array]:
    """Chunked KDA over [seq, num_heads, chunk, head_dim] inputs.

    Returns out [seq, chunk, num_v_heads * v_head_dim] and the final state
    [seq, 1, num_v_heads, kq_head_dim, v_head_dim], laid out like one
    recurrent checkpoint. Only the final state is produced, so this path is
    never taken when a sequence needs more than one checkpoint.
    """
    q, k, v, g, beta = prepare_chunk(row_lo, row_hi, q_chunked, k_chunked,
                                     v_chunked, b_chunked, g_chunked, a_log,
                                     dt_bias, cfg)

    seq_outs = []
    seq_states = []
    for seq_idx in range(cfg.seq_tile_size):
        with profile_scope("kda"):
            chunk = ChunkKDA(q[seq_idx], k[seq_idx], v[seq_idx], g[seq_idx],
                             beta[seq_idx], cfg)
            chunk_outputs, state = run_pipelined([chunk], state_prev[seq_idx])
        seq_outs.append(pack_out(chunk_outputs[0], cfg))
        seq_states.append(state)
    return (jnp.stack(seq_outs, axis=0),
            jnp.stack(seq_states, axis=0)[:, jnp.newaxis])


def recurrent_kda_per_seq(
    q_token: jax.Array,
    k_token: jax.Array,
    k_token_t: jax.Array,
    v_token: jax.Array,
    decay: jax.Array,
    beta: jax.Array,
    state_prev: jax.Array,
    cfg: config.KDAConfig,
) -> tuple[jax.Array, jax.Array]:
    """Recurrent KDA over one sequence, one token at a time.

    q_token and k_token are [num_kq_heads, chunk, 1, kq_head_dim], k_token_t
    [num_kq_heads, chunk, kq_head_dim, 1], v_token [num_v_heads, chunk, 1,
    v_head_dim], decay (the exp of the per-key-channel log decay)
    [num_v_heads, chunk, kq_head_dim, 1], beta [num_v_heads, chunk, 1, 1], and
    state_prev [num_v_heads, kq_head_dim, v_head_dim]. Returns out [chunk,
    num_v_heads * v_head_dim] and the post-token states of the last
    cfg.window_size positions, [window_size, num_v_heads, kq_head_dim,
    v_head_dim].
    """
    token_outs = []
    window_states = []
    state = state_prev
    for token_idx in range(cfg.compute_chunk_size):
        q_curr = jnp.repeat(q_token[:, token_idx], cfg.v_per_kq_head, axis=0)
        k_curr = jnp.repeat(k_token[:, token_idx], cfg.v_per_kq_head, axis=0)
        v_curr = v_token[:, token_idx]
        k_curr_t = jnp.repeat(k_token_t[:, token_idx], cfg.v_per_kq_head,
                              axis=0)
        beta_curr = beta[:, token_idx]
        decay_curr = decay[:, token_idx]

        state_decayed = state * decay_curr
        v_pred = batched_matmul(k_curr,
                                state_decayed).astype(cfg.dtypes.compute)
        v_new = beta_curr * (v_curr - v_pred)
        # Note (david): the outer product with k_curr_t expands the operand by
        # kq_head_dim, so it comes as late as possible.
        state = state_decayed + k_curr_t * v_new
        out = batched_matmul(q_curr, state).astype(cfg.dtypes.compute)
        token_outs.append(pack_out(out, cfg))
        # Note (david): the caller masks rows past real_sizes, so the state
        # stops changing there and trailing checkpoints repeat the last real
        # token's state; the compiler drops the unkept positions' states.
        if token_idx >= cfg.compute_chunk_size - cfg.window_size:
            window_states.append(state)

    return jnp.concatenate(token_outs, axis=0), jnp.stack(window_states, axis=0)


def recurrent_kda(
    real_sizes: jax.Array,
    q_token: jax.Array,
    k_token: jax.Array,
    v_token: jax.Array,
    b_token: jax.Array,
    g_token: jax.Array,
    state_prev: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.KDAConfig,
) -> tuple[jax.Array, jax.Array]:
    """Recurrent KDA over [seq, num_heads, chunk, 1, head_dim] inputs.

    Returns out [seq, chunk, num_v_heads * v_head_dim] and one state
    checkpoint per window position, [seq, window_size, num_v_heads,
    kq_head_dim, v_head_dim]. Positions at or past real_sizes repeat the last
    real state and are never written back to HBM.
    """
    mask_dtype = get_mask_dtype(cfg.dtypes.compute)
    token_iota = jax.lax.broadcasted_iota(
        mask_dtype, (cfg.seq_tile_size, 1, cfg.compute_chunk_size, 1, 1), 2)
    is_real_token = token_iota < real_sizes.reshape(-1, 1, 1, 1,
                                                    1).astype(mask_dtype)

    q_token = jnp.where(is_real_token, q_token.astype(cfg.dtypes.compute), 0)
    k_token = jnp.where(is_real_token, k_token.astype(cfg.dtypes.compute), 0)
    v_token = jnp.where(is_real_token, v_token.astype(cfg.dtypes.compute), 0)

    b_token = b_token.astype(cfg.dtypes.compute)
    g_token = g_token.astype(cfg.dtypes.compute)

    a_log = a_log[None, :, None].astype(cfg.dtypes.compute)
    dt_bias = dt_bias[None, :, None].astype(cfg.dtypes.compute)

    q_token = l2_norm(q_token) * cfg.kq_head_dim**-0.5
    k_token = l2_norm(k_token)
    k_token_t = fused_transpose_broadcast(k_token, src_dim=4, dst_dim=3)

    beta = b_token if cfg.beta_is_activated else jax.nn.sigmoid(b_token)
    g = log_decay(g_token, a_log, dt_bias, cfg.lower_bound)

    beta = jnp.where(is_real_token, beta, 0)
    # Note (david): pad rows get log decay 0, a decay of exp(0) = 1, so they
    # carry the previous state through unchanged.
    g = jnp.where(is_real_token, g, 0)
    decay = jnp.exp(g)

    beta = fused_transpose_broadcast(beta, src_dim=4,
                                     dst_dim=1)[:, :cfg.num_v_heads]
    decay = fused_transpose_broadcast(decay, src_dim=4, dst_dim=3)

    seq_outs = []
    seq_states = []
    for seq_idx in range(cfg.seq_tile_size):
        out, window_states = recurrent_kda_per_seq(
            q_token[seq_idx],
            k_token[seq_idx],
            k_token_t[seq_idx],
            v_token[seq_idx],
            decay[seq_idx],
            beta[seq_idx],
            state_prev[seq_idx],
            cfg,
        )
        seq_outs.append(out)
        seq_states.append(window_states)

    return jnp.stack(seq_outs, axis=0), jnp.stack(seq_states, axis=0)
