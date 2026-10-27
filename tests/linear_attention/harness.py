"""Runs [B, T, H, D] q, k, v through the fused conv1d GDN and KDA kernels.

The kernels take the mixed qkv before the short convolution and silu. The
tests draw q, k, v as the convolution's input, Fused gives the reference the
q, k, v that come out of it, and its runners feed the kernels the same
inputs. The kernels take bf16 qkv (and a bf16 KDA gate), so every input is
rounded to bf16 first and the reference sees the rounded values in f32. A
sequence with an initial state also starts from a conv history.

Fused.chunk runs one prefill call; Fused.recurrent runs decode steps of one
token per sequence, through the fused kernel or its ReplaySSM variant.
"""

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flywheel_tpu.linear_attention.gdn import wrapper as gdn_wrapper
from flywheel_tpu.linear_attention.kda import wrapper as kda_wrapper
from tests.linear_attention.reference import (
    naive_gdn_gate,
    naive_kda_gate,
    naive_kda_lowerbound_gate,
    randn,
    uniform,
)

KERNEL_SIZE = 4
REPLAY_WINDOW = 16
# Note (david): softplus(-200) and sigmoid(-200) round to exactly 0 in f32 and
# sigmoid(200) to exactly 1, so these logits stand for the gates' ends.
SATURATED_LOGIT = 200.0
# Note (david): both packages size their tiles from pltpu.get_tpu_info() and
# have no interpret mode.
TPU_ONLY = pytest.mark.skipif(
    jax.default_backend() != "tpu", reason="the fused kernels run on TPU only"
)
# Note (david): the decode tests run through fused_conv1d_* and through
# fused_conv1d_*_replay with f32 or bf16 records.
DECODE = pytest.mark.parametrize(
    "record_dtype",
    [None, jnp.float32, jnp.bfloat16],
    ids=["fused", "replay_f32", "replay_bf16"],
)


@dataclasses.dataclass(frozen=True)
class Ops:
    """One kernel family: the fused kernel, its ReplaySSM variant and the
    shape of its gate."""

    fused: Callable[..., tuple]
    replay: Callable[..., tuple]
    replay_layout: Callable[
        [int, int, int, int],
        gdn_wrapper.ReplayLayout | kda_wrapper.ReplayLayout,
    ]
    split_checkpoint: Callable[[jax.Array], jax.Array]
    merge_checkpoint: Callable[[jax.Array], jax.Array]
    next_replay_lens: Callable[..., jax.Array]
    per_channel_gate: bool
    gate_dtype: jnp.dtype


GDN = Ops(
    gdn_wrapper.fused_conv1d_gdn,
    gdn_wrapper.fused_conv1d_gdn_replay,
    gdn_wrapper.replay_layout,
    gdn_wrapper.split_checkpoint,
    gdn_wrapper.merge_checkpoint,
    gdn_wrapper.next_replay_lens,
    per_channel_gate=False,
    gate_dtype=jnp.float32,
)
KDA = Ops(
    kda_wrapper.fused_conv1d_kda,
    kda_wrapper.fused_conv1d_kda_replay,
    kda_wrapper.replay_layout,
    kda_wrapper.split_checkpoint,
    kda_wrapper.merge_checkpoint,
    kda_wrapper.next_replay_lens,
    per_channel_gate=True,
    gate_dtype=jnp.bfloat16,
)


def param_grid(id_format: str, rows: list[tuple]) -> list:
    """pytest params of a grid, each row's id formatted from its values."""
    return [pytest.param(*row, id=id_format.format(*row)) for row in rows]


def bf16_values(x: jax.Array) -> jax.Array:
    """x rounded to bf16, kept in f32."""
    return x.astype(jnp.bfloat16).astype(jnp.float32)


def softplus_inverse(y: jax.Array) -> jax.Array:
    """x with softplus(x) == y for y >= 0; y == 0 maps to -SATURATED_LOGIT."""
    return jnp.maximum(y + jnp.log(-jnp.expm1(-y)), -SATURATED_LOGIT)


def logit(p: jax.Array) -> jax.Array:
    """x with sigmoid(x) == p for p in [0, 1], the ends saturated."""
    return jnp.clip(
        jnp.log(p) - jnp.log1p(-p), -SATURATED_LOGIT, SATURATED_LOGIT
    )


def encode_gate(
    ops: Ops, g: jax.Array, lower_bound: float | None = None
) -> tuple[jax.Array, jax.Array]:
    """The raw gate under which the kernel, with A_log = 0 and dt_bias = 0,
    computes the log decay g, and the log decay it computes from that raw
    gate in the kernel's gate dtype.

    The fused kernels always activate the gate themselves, as -softplus(raw),
    or as lower_bound * sigmoid(raw) with lower_bound set, so a test that
    sets the log decay itself hands them this raw gate.
    """
    if lower_bound is None:
        raw = softplus_inverse(-g).astype(ops.gate_dtype)
        num_heads = g.shape[-2] if ops.per_channel_gate else g.shape[-1]
        zeros = jnp.zeros((num_heads,))
        gate_fn = naive_kda_gate if ops.per_channel_gate else naive_gdn_gate
        return raw, gate_fn(raw, zeros)
    raw = logit(g / lower_bound).astype(ops.gate_dtype)
    return raw, naive_kda_lowerbound_gate(raw, lower_bound=lower_bound)


class Fused:
    """q, k, v run through a fused conv1d kernel.

    The inputs are [B, T, H, K] q and k and [B, T, HV, V] v, or B = 1 and a
    packed cu_seqlens. Sequence s owns state slot 1 + s, or
    state_indices[s] in pools the caller hands over; slot 0 is the null
    block. Attributes q, k, v hold the convolution's activated output in the
    input layout, for the reference.
    """

    def __init__(
        self,
        ops: Ops,
        rng: np.random.Generator,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        cu_seqlens: list[int] | None = None,
        initial_state: jax.Array | None = None,
        state_indices: np.ndarray | None = None,
        state_pool: np.ndarray | None = None,
        conv_pool: np.ndarray | None = None,
    ) -> None:
        self.ops = ops
        batch, seq_len, self.n_kq, self.d_k = q.shape
        self.n_v, self.d_v = v.shape[2:]
        self.batch_shape = (batch, seq_len)
        self.varlen = cu_seqlens is not None
        if cu_seqlens is None:
            cu_seqlens = [seq_len * i for i in range(batch + 1)]
        else:
            assert batch == 1
        self.cu_seqlens = np.asarray(cu_seqlens)
        self.lengths = np.diff(self.cu_seqlens)
        num_seqs = self.lengths.size
        self.has_initial_state = initial_state is not None

        num_tokens = batch * seq_len
        qkv = jnp.concatenate(
            [
                q.reshape(num_tokens, -1),
                k.reshape(num_tokens, -1),
                v.reshape(num_tokens, -1),
            ],
            axis=-1,
        )
        self.qkv = qkv.astype(jnp.bfloat16)
        dim = qkv.shape[-1]
        # Note (david): weight and bias are drawn from U(-1/sqrt(K),
        # 1/sqrt(K)), the default init of a depthwise conv; the history
        # holds earlier qkv rows.
        bound = KERNEL_SIZE**-0.5
        self.conv_weight = bf16_values(
            uniform(rng, (dim, 1, KERNEL_SIZE), -bound, bound)
        )
        self.conv_bias = bf16_values(uniform(rng, (dim,), -bound, bound))
        self.conv_history = bf16_values(
            randn(rng, (num_seqs, KERNEL_SIZE - 1, dim))
        )

        if state_indices is None:
            self.state_indices = np.arange(1, num_seqs + 1, dtype=np.int32)
            num_slots = num_seqs + 1
            self.state_pool = np.zeros(
                (num_slots, self.n_v, self.d_k, self.d_v), np.float32
            )
            self.conv_pool = np.zeros(
                (num_slots, KERNEL_SIZE - 1, dim), np.float32
            )
        else:
            self.state_indices = np.asarray(state_indices, np.int32)
            self.state_pool = np.array(state_pool, np.float32)
            self.conv_pool = np.array(conv_pool, np.float32)
        if self.has_initial_state:
            self.state_pool[self.state_indices] = np.asarray(initial_state)
            self.conv_pool[self.state_indices] = np.asarray(self.conv_history)

        acts = jnp.concatenate([
            self._activate(seq, start, end)
            for seq, (start, end) in enumerate(
                zip(self.cu_seqlens[:-1], self.cu_seqlens[1:])
            )
        ])
        key_dim = self.n_kq * self.d_k
        self.q = acts[:, :key_dim].reshape(q.shape)
        self.k = acts[:, key_dim : 2 * key_dim].reshape(k.shape)
        self.v = acts[:, 2 * key_dim :].reshape(v.shape)

    def _activate(self, seq: int, start: int, end: int) -> jax.Array:
        """silu(conv1d(qkv)) of one sequence in f32."""
        rows = self.qkv[start:end].astype(jnp.float32)
        history = self.conv_history[seq]
        if not self.has_initial_state:
            history = jnp.zeros_like(history)
        window = jnp.concatenate([history, rows])
        conv = self.conv_bias + sum(
            window[tap : tap + end - start] * self.conv_weight[:, 0, tap]
            for tap in range(KERNEL_SIZE)
        )
        return jax.nn.silu(conv)

    def _gate_rows(self, g: jax.Array) -> jax.Array:
        if self.ops.per_channel_gate:
            return g.reshape(-1, self.n_v * self.d_k)
        return g.reshape(-1, self.n_v)

    def _gate_params(
        self, A_log: jax.Array | None, dt_bias: jax.Array | None
    ) -> tuple[jax.Array, jax.Array]:
        """A_log and dt_bias, zero when absent: exp(0) = 1 and a zero bias
        leave the gate unchanged."""
        if A_log is None:
            A_log = jnp.zeros((self.n_v,))
        if dt_bias is None:
            width = (
                self.n_v * self.d_k if self.ops.per_channel_gate else (self.n_v)
            )
            dt_bias = jnp.zeros((width,))
        return A_log.reshape(-1), dt_bias.reshape(-1)

    def _seq_lens(self, query_lens: np.ndarray) -> jax.Array:
        """A sequence with an initial state continues one earlier token."""
        return jnp.asarray(query_lens + int(self.has_initial_state), jnp.int32)

    def _output(self, out: jax.Array) -> jax.Array:
        return out.astype(jnp.float32).reshape(
            *self.batch_shape, self.n_v, self.d_v
        )

    def chunk(
        self,
        g: jax.Array,
        beta: jax.Array,
        A_log: jax.Array | None = None,
        dt_bias: jax.Array | None = None,
        chunk_size: int = 64,
        **kernel_kwargs,
    ) -> tuple[jax.Array, jax.Array]:
        """One prefill call; returns o [B, T, HV, V] and the final states
        [N, HV, K, V]."""
        num_seqs = self.lengths.size
        slots = jnp.asarray(self.state_indices)
        (conv_pool, state_pool), out = self.ops.fused(
            self.qkv,
            beta.reshape(-1, self.n_v),
            self._gate_rows(g),
            jnp.asarray(self.conv_pool),
            jnp.asarray(self.state_pool),
            self.conv_weight,
            self.conv_bias,
            *self._gate_params(A_log, dt_bias),
            jnp.asarray(self.cu_seqlens, jnp.int32),
            slots,
            jnp.array([0, num_seqs, num_seqs], jnp.int32),
            self._seq_lens(self.lengths),
            slots,
            n_kq=self.n_kq,
            n_v=self.n_v,
            d_k=self.d_k,
            d_v=self.d_v,
            kernel_size=KERNEL_SIZE,
            compute_chunk_size=chunk_size,
            **kernel_kwargs,
        )
        self.final_conv_pool, self.final_state_pool = conv_pool, state_pool
        return self._output(out), state_pool[slots]

    def recurrent(
        self,
        g: jax.Array,
        beta: jax.Array,
        record_dtype: jnp.dtype | None,
        A_log: jax.Array | None = None,
        dt_bias: jax.Array | None = None,
        **kernel_kwargs,
    ) -> tuple[jax.Array, jax.Array]:
        """Decode steps of one token per unfinished sequence, through
        fused_conv1d_* or, with record_dtype, its ReplaySSM variant; returns
        o [B, T, HV, V] and the final states [N, HV, K, V]."""
        b_rows = beta.reshape(-1, self.n_v)
        g_rows = self._gate_rows(g)
        dims = dict(
            n_kq=self.n_kq,
            n_v=self.n_v,
            d_k=self.d_k,
            d_v=self.d_v,
            kernel_size=KERNEL_SIZE,
        )
        weights = (
            self.conv_weight,
            self.conv_bias,
            *self._gate_params(A_log, dt_bias),
        )
        conv_pool = jnp.asarray(self.conv_pool)
        state_pool = jnp.asarray(self.state_pool)
        num_seqs = self.lengths.size
        if record_dtype is not None:
            state_pool = self.ops.split_checkpoint(state_pool)
            # Note (david): NaN records and gates catch any read past the
            # committed ones.
            layout = self.ops.replay_layout(
                self.n_kq, self.n_v, self.d_k, self.d_v
            )
            replay_state = jnp.full(
                (state_pool.shape[0], REPLAY_WINDOW, layout.record_dim),
                jnp.nan,
                record_dtype,
            )
            replay_gates = jnp.full(
                (state_pool.shape[0], REPLAY_WINDOW, layout.gate_dim),
                jnp.nan,
                jnp.float32,
            )
            replay_lens = jnp.zeros((num_seqs,), jnp.int32)

        out_rows, outs = [], []
        for step in range(int(self.lengths.max())):
            active = np.nonzero(self.lengths > step)[0]
            rows = self.cu_seqlens[active] + step
            num_active = active.size
            query_start_loc = jnp.arange(num_active + 1, dtype=jnp.int32)
            slots = jnp.asarray(self.state_indices[active])
            distribution = jnp.full((3,), num_active, jnp.int32)
            seq_lens = self._seq_lens(np.full(num_active, step + 1))
            step_inputs = (self.qkv[rows], b_rows[rows], g_rows[rows])
            if record_dtype is None:
                (conv_pool, state_pool), out = self.ops.fused(
                    *step_inputs,
                    conv_pool,
                    state_pool,
                    *weights,
                    query_start_loc,
                    slots,
                    distribution,
                    seq_lens,
                    slots,
                    **dims,
                    **kernel_kwargs,
                )
            else:
                active_lens = replay_lens[active]
                (conv_pool, state_pool, replay_state, replay_gates), out = (
                    self.ops.replay(
                        *step_inputs,
                        conv_pool,
                        state_pool,
                        replay_state,
                        replay_gates,
                        *weights,
                        query_start_loc,
                        slots,
                        distribution,
                        seq_lens,
                        slots,
                        active_lens,
                        replay_window=REPLAY_WINDOW,
                        **dims,
                        **kernel_kwargs,
                    )
                )
                replay_lens = replay_lens.at[active].set(
                    self.ops.next_replay_lens(
                        active_lens,
                        query_start_loc,
                        seq_lens,
                        slots,
                        slots,
                        distribution,
                        jnp.ones((num_active,), jnp.int32),
                        REPLAY_WINDOW,
                    )
                )
            out_rows.append(rows)
            outs.append(out)

        out = (
            jnp.zeros((self.qkv.shape[0], self.n_v * self.d_v), jnp.float32)
            .at[np.concatenate(out_rows)]
            .set(jnp.concatenate(outs).astype(jnp.float32))
        )
        if record_dtype is not None:
            state_pool = self.ops.merge_checkpoint(state_pool)
        self.final_conv_pool, self.final_state_pool = conv_pool, state_pool
        if record_dtype is None:
            return self._output(out), state_pool[self.state_indices]
        final_states = [
            self._materialize(
                state_pool[slot],
                replay_state[slot],
                replay_gates[slot],
                int(replay_lens[seq]),
            )
            for seq, slot in enumerate(self.state_indices)
        ]
        return self._output(out), jnp.asarray(np.stack(final_states))

    def _materialize(
        self,
        checkpoint: jax.Array,
        records: jax.Array,
        gates: jax.Array,
        num_records: int,
    ) -> np.ndarray:
        """The state a ReplaySSM checkpoint and its first num_records
        records represent, folded in f64.

        A record holds u [HV * V] and the key k [H * K], and its gate row
        the log decay g, [HV] (lane padded) for GDN or [HV * K] for KDA;
        folding it maps S to S * exp(g) + k u^T.
        """
        k_offset = self.n_v * self.d_v
        state = np.asarray(checkpoint, np.float64)
        for row, gate_row in zip(
            np.asarray(records[:num_records], np.float64),
            np.asarray(gates[:num_records], np.float64),
        ):
            u = row[:k_offset].reshape(self.n_v, self.d_v)
            k = np.repeat(
                row[k_offset:].reshape(self.n_kq, self.d_k),
                self.n_v // self.n_kq,
                axis=0,
            )
            if self.ops.per_channel_gate:
                decay = np.exp(gate_row.reshape(self.n_v, self.d_k))
                state = state * decay[:, :, None]
            else:
                decay = np.exp(gate_row[: self.n_v])
                state = state * decay[:, None, None]
            state = state + k[:, :, None] * u[:, None, :]
        return state.astype(np.float32)
