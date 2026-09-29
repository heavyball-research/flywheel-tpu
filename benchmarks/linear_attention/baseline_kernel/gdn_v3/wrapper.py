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

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from gdn_v3 import (
    compute_conv1d,
    compute_gdn,
    config,
    memory_ref,
    metadata,
    vmem_ldst,
)

F32_BYTES = 4
BF16_BYTES = 2
# Note (david): a verify window keeps one state checkpoint per position in
# VMEM, so large-head models (e.g. Qwen3.5-397B: 64 local v-heads, 21M per
# buffered window even at tile_size 1) do not fit the default budget; windowed
# kernels get a higher limit and the decode tile is sized against it.
DEFAULT_VMEM_FRACTION = 0.7
WINDOWED_VMEM_FRACTION = 0.9
# Note (david): pallas_call operands are the flattened metadata leaves, then
# qkv, b, a, conv_state, recurrent_state and the aliased output.
CONV_STATE_OPERAND = 3
RECURRENT_STATE_OPERAND = 4
ALIASED_OUT_OPERAND = 5


def inner_kernel(
    qkv_slot_ref: jax.Array,
    b_slot_ref: jax.Array,
    a_slot_ref: jax.Array,
    conv_state_slot_ref: jax.Array,
    recurrent_slot_ref: jax.Array,
    out_slot_ref: jax.Array,
    metadata_ref: memory_ref.MetadataRef,
    weights_ref: memory_ref.WeightRefs,
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: jax.Array | None,
    *,
    cfg: config.GDNConfig,
) -> None:
    """Conv1D + GDN over one tile of VMEM slots.

    The slots are qkv [seq, chunk, 1, dim_size], b and a [seq, chunk, 1,
    aligned_num_v_heads], the conv state [seq, window_size, prev_kernel_size,
    1, dim_size], the recurrent state [seq, window_size, num_v_heads,
    kq_head_dim, v_head_dim] and out [seq, chunk, num_v_heads, v_head_dim].
    The state slots are overwritten with the new checkpoints.
    """
    p_id = pl.program_id(0)

    real_sizes, prev_conv_state, prev_recurrent_state = (
        vmem_ldst.load_and_select_states(
            metadata_ref=metadata_ref,
            p_id=p_id,
            conv_state_slot_ref=conv_state_slot_ref,
            recurrent_slot_ref=recurrent_slot_ref,
            carry_conv_scratch_ref=carry_conv_scratch_ref,
            carry_recurrent_scratch_ref=carry_recurrent_scratch_ref,
            cfg=cfg,
        ))

    # Note (david): Conv1D slides its window across rows. The 2D layout packs
    # several rows per register and would shuffle on every slide, while the
    # compact layout keeps one row per register.
    qkv_in_compact = qkv_slot_ref[...].astype(jnp.float32)
    conv_input = jnp.concat([prev_conv_state, qkv_in_compact], axis=1)

    conv_weight = weights_ref.conv.weight[...].astype(jnp.float32)
    if weights_ref.conv.bias is None:
        conv_bias = None
    else:
        conv_bias = weights_ref.conv.bias[...].astype(jnp.float32)

    qkv_out_compact, new_conv_state = compute_conv1d.causal_conv1d(
        real_sizes=real_sizes,
        conv_input=conv_input,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        cfg=cfg,
    )

    conv_state_slot_ref[...] = new_conv_state
    if carry_conv_scratch_ref is not None:
        # Note (david): the next tile resumes after this tile's last token,
        # which is the final checkpoint.
        carry_conv_scratch_ref[...] = new_conv_state[:, -1]

    qkv_out_compact = jax.nn.silu(qkv_out_compact)

    num_v_padding = cfg.aligned_num_v_heads - cfg.num_v_heads
    a_log = jnp.pad(weights_ref.gdn.a_log[...], (0, num_v_padding))
    dt_bias = jnp.pad(weights_ref.gdn.dt_bias[...], (0, num_v_padding))

    qkv_slot_ref[...] = qkv_out_compact
    if cfg.use_recurrent:
        q_compact, k_compact, v_compact = vmem_ldst.load_as_qkv_compact(
            qkv_slot_ref, cfg)
        b_compact = jnp.expand_dims(b_slot_ref[...], axis=1)
        a_compact = jnp.expand_dims(a_slot_ref[...], axis=1)

        out, new_recurrent_state = compute_gdn.recurrent_gdn(
            q_compact=q_compact,
            k_compact=k_compact,
            v_compact=v_compact,
            b_compact=b_compact,
            a_compact=a_compact,
            state_prev=prev_recurrent_state,
            a_log=a_log,
            dt_bias=dt_bias,
            cfg=cfg,
            real_sizes=real_sizes,
        )
    else:
        q_large, k_large, v_large, b_large, a_large = (
            vmem_ldst.load_activation_as_large(
                qkv_vmem_ref=qkv_slot_ref,
                b_vmem_ref=b_slot_ref,
                a_vmem_ref=a_slot_ref,
                cfg=cfg,
            ))

        out, new_recurrent_state = compute_gdn.chunked_gdn(
            q_large=q_large,
            k_large=k_large,
            v_large=v_large,
            b_large=b_large,
            a_large=a_large,
            state_prev=prev_recurrent_state,
            a_log=a_log,
            dt_bias=dt_bias,
            cfg=cfg,
            real_sizes=real_sizes,
        )

    out_slot_ref[...] = out.astype(out_slot_ref.dtype)
    recurrent_slot_ref[...] = new_recurrent_state.astype(
        recurrent_slot_ref.dtype)

    if carry_recurrent_scratch_ref is not None:
        carry_recurrent_scratch_ref[...] = new_recurrent_state[:, -1]


def outer_kernel(
    metadata_ref: memory_ref.MetadataRef,
    qkv_ref: jax.Array,
    b_ref: jax.Array,
    a_ref: jax.Array,
    conv_state_ref: jax.Array,
    recurrent_state_ref: jax.Array,
    aliased_out_ref: jax.Array,
    weights_ref: memory_ref.WeightRefs,
    out_ref: jax.Array,
    conv_state_out_ref: jax.Array,
    recurrent_state_out_ref: jax.Array,
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: jax.Array | None,
    *,
    cfg: config.GDNConfig,
) -> None:
    del aliased_out_ref, conv_state_out_ref, recurrent_state_out_ref

    qkv_alloc, b_alloc, a_alloc, conv_alloc, recurrent_alloc, out_alloc = (
        memory_ref.create_allocs(
            metadata_ref=metadata_ref,
            qkv_ref=qkv_ref,
            b_ref=b_ref,
            a_ref=a_ref,
            out_ref=out_ref,
            conv_state_ref=conv_state_ref,
            recurrent_state_ref=recurrent_state_ref,
            cfg=cfg,
        ))

    num_tiles = metadata_ref.num_tiles[...]

    pipeline = pltpu.emit_pipeline(
        body=functools.partial(
            inner_kernel,
            cfg=cfg,
        ),
        grid=(num_tiles, ),
        in_specs=(
            qkv_alloc.spec,
            b_alloc.spec,
            a_alloc.spec,
            conv_alloc.spec,
            recurrent_alloc.spec,
        ),
        out_specs=(out_alloc.spec, ),
    )

    @pl.with_scoped(allocations=(
        qkv_alloc,
        b_alloc,
        a_alloc,
        conv_alloc,
        recurrent_alloc,
        out_alloc,
    ))
    def _run_pipeline(allocations):
        pipeline(
            qkv_ref,
            b_ref,
            a_ref,
            conv_state_ref,
            recurrent_state_ref,
            out_ref,
            scratches=(
                metadata_ref,
                weights_ref,
                carry_conv_scratch_ref,
                carry_recurrent_scratch_ref,
            ),
            allocations=allocations,
        )

    _run_pipeline()


@jax.jit(
    donate_argnames=("conv_state", "recurrent_state"),
    static_argnames=(
        "n_kq",
        "n_v",
        "d_k",
        "d_v",
        "kernel_size",
        "num_spec_tokens",
        "decode_tile_size",
        "mixed_tile_size",
        "zero_initialize_out",
        "compute_precision",
    ),
)
def fused_conv1d_gdn(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_state: jax.Array,
    recurrent_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    a_log: jax.Array,
    dt_bias: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    read_state_indices: jax.Array,
    read_offsets: jax.Array | None = None,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    num_spec_tokens: int = 0,
    zero_initialize_out: bool = True,
    compute_precision: jnp.dtype = jnp.float32.dtype,
    decode_tile_size: int = 4,
    mixed_tile_size: int = 64,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Conv1D + silu + gated delta rule in one fused kernel.

    Args:
        qkv: Mixed query, key, value input [batch_size, dim_size], where
            dim_size = n_kq * d_k * 2 + n_v * d_v.
        b: Input to beta [batch_size, n_v].
        a: Input to the decay g [batch_size, n_v].
        conv_state: Convolution state cache [num_seqs + 1, kernel_size - 1,
            dim_size] holding the last kernel_size - 1 tokens of the previous
            invocation. Slot 0 is the null block for padded or invalid tokens.
            A slot may hold garbage on the first invocation of a sequence.
        recurrent_state: Recurrent state cache [num_seqs + 1, n_v, d_k, d_v],
            with the same null block and garbage caveat.
        conv_weight: Convolution weight [dim_size, 1, kernel_size].
        conv_bias: Optional convolution bias [dim_size].
        a_log: a_log [n_v].
        dt_bias: dt_bias [n_v].
        query_start_loc: Start locations of the sequences [num_seqs + 1].
        state_indices: State cache slot of each sequence [num_seqs]. With
            speculative decoding these are the base slots of per-request
            groups of num_spec_tokens + 1 consecutive slots.
        distribution: int32 [3], [decode_end, prefill_end, mixed_end]. With
            num_spec_tokens > 0 the first segment holds speculative verify
            windows of up to num_spec_tokens + 1 tokens instead of 1-token
            decodes.
        seq_lens: Length of each sequence [num_seqs].
        read_state_indices: Slot each sequence reads its initial state from
            [num_seqs]. Equals state_indices unless mamba prefix caching
            (align mode) resumes from the cached state block of the previous
            block boundary.
        read_offsets: Optional int32 per-sequence state read offset
            [num_seqs] (num_accepted - 1 from the last verify step). Windowed
            sequences read their initial state from read_state_indices[s] +
            read_offsets[s] and write one checkpoint per window position to
            state_indices[s] + t. Required when num_spec_tokens > 0.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Key/query head dimension.
        d_v: Value head dimension.
        kernel_size: Convolution kernel size.
        num_spec_tokens: Number of speculative draft tokens; 0 gives plain
            1-token-per-sequence decode.
        zero_initialize_out: Whether to zero the output buffer before the
            non-batched sequences run.
        compute_precision: Computation dtype.
        decode_tile_size: Sequences per tile for decode sequences.
        mixed_tile_size: Tokens per tile for prefill / mixed sequences.

    Returns:
        (new_conv_state, new_recurrent_state): The updated state caches.
        out: The output [batch_size, n_v * d_v].
    """
    act_out_dtype = qkv.dtype
    conv_out_dtype = conv_state.dtype
    recurrent_out_dtype = recurrent_state.dtype

    qkv = qkv.astype(jnp.float32)
    b = b.astype(jnp.float32)
    a = a.astype(jnp.float32)
    conv_state = conv_state.astype(jnp.float32)

    num_seqs = state_indices.size
    batch_size, dim = qkv.shape
    assert conv_weight.shape == (dim, 1, kernel_size)
    if conv_bias is not None:
        assert conv_bias.shape == (dim, )
    assert query_start_loc.shape == (num_seqs + 1, )
    assert state_indices.shape == (num_seqs, )
    assert distribution.shape == (3, )
    if num_spec_tokens > 0:
        assert read_offsets is not None, (
            "read_offsets is required when num_spec_tokens > 0")
    if read_offsets is None:
        read_offsets = jnp.zeros((num_seqs, ), dtype=jnp.int32)
    else:
        read_offsets = read_offsets.astype(jnp.int32)
    assert read_offsets.shape == (num_seqs, )
    assert read_state_indices.shape == (num_seqs, )
    read_state_indices = read_state_indices.astype(state_indices.dtype)
    act_in_dtype = qkv.dtype

    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes
    packing = F32_BYTES // act_in_dtype.itemsize
    padded_batch_size = pl.cdiv(batch_size, packing) * packing
    mixed_tile_size = min(mixed_tile_size, batch_size)
    aligned_num_v_heads = pl.cdiv(n_v, num_lanes) * num_lanes

    if num_spec_tokens > 0:
        # Note (david): a verify window holds one state checkpoint per window
        # position per sequence in VMEM, which multiplies the per-sequence
        # footprint by the window size. The tile shrinks so the multi-buffered
        # windows fit in about half the scoped VMEM budget; the rest goes to
        # weights, activation scratch and compiler temporaries.
        verify_window_size = num_spec_tokens + 1
        bytes_per_seq = verify_window_size * (
            n_v * d_k * d_v * F32_BYTES + (kernel_size - 1) * dim * F32_BYTES +
            dim * F32_BYTES + 2 * aligned_num_v_heads * F32_BYTES +
            n_v * d_v * BF16_BYTES)
        vmem_budget = int(WINDOWED_VMEM_FRACTION *
                          tpu_info.vmem_capacity_bytes)
        spec_tile_budget = (vmem_budget // 2) // config.NUM_BUFFERS
        decode_tile_size = max(
            1,
            min(decode_tile_size, batch_size,
                spec_tile_budget // bytes_per_seq))
    else:
        decode_tile_size = min(decode_tile_size, batch_size)

    batch_padding_size = padded_batch_size - batch_size
    num_v_padding_size = aligned_num_v_heads - n_v
    qkv = jnp.pad(qkv, ((0, batch_padding_size), (0, 0)))
    b = jnp.pad(b, ((0, batch_padding_size), (0, num_v_padding_size)))
    a = jnp.pad(a, ((0, batch_padding_size), (0, num_v_padding_size)))

    qkv = qkv.reshape(padded_batch_size, 1, -1)
    b = b.reshape(padded_batch_size, 1, -1)
    a = a.reshape(padded_batch_size, 1, -1)

    conv_state_shape = conv_state.shape
    conv_state = conv_state.reshape(-1, kernel_size - 1, 1, dim)
    conv_weight = conv_weight.swapaxes(0, 2).astype(jnp.float32)
    conv_bias = None if conv_bias is None else conv_bias.astype(jnp.float32)

    weights = memory_ref.WeightRefs(
        conv=memory_ref.ConvWeightsRef(weight=conv_weight, bias=conv_bias),
        gdn=memory_ref.GDNWeightsRef(a_log=a_log, dt_bias=dt_bias),
    )

    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    weights_spec = jax.tree.map(lambda _: vmem_spec, weights)

    def _call_kernel(
        in_conv_state: jax.Array,
        in_recurrent_state: jax.Array,
        in_act: jax.Array | None,
        mode: config.GDNMode,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if mode == config.GDNMode.PER_SEQ:
            tile_size = mixed_tile_size
            # Note (david): prefill and mixed sequences keep a single state
            # checkpoint.
            window_size = 1
        else:
            tile_size = decode_tile_size
            window_size = num_spec_tokens + 1

        cfg = config.GDNConfig(
            mode=mode,
            batch_size=padded_batch_size,
            kernel_size=kernel_size,
            tile_size=tile_size,
            window_size=window_size,
            dim_size=dim,
            num_kq_heads=n_kq,
            num_v_heads=n_v,
            kq_head_dim=d_k,
            v_head_dim=d_v,
            dtypes=config.Dtypes(
                act_in=act_in_dtype,
                act_out=act_out_dtype,
                compute=compute_precision,
                recurrent_state=in_recurrent_state.dtype,
                conv_state=in_conv_state.dtype,
            ),
        )

        # Note (david): the metadata is rebuilt for every layer; the compiler
        # CSEs the copies.
        if mode == config.GDNMode.PER_SEQ:
            seq_metadata = metadata.compute_per_seq_metadata(
                cfg=cfg,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                state_indices=state_indices,
                start_seq=distribution[0],
                end_seq=distribution[-1],
                read_indices=read_state_indices,
            )
            # Note (david): only PER_SEQ splits a sequence across tiles, so only
            # it carries state from one tile to the next.
            scratch_shapes = dict(
                carry_conv_scratch_ref=pltpu.VMEM(
                    (cfg.seq_tile_size, cfg.prev_kernel_size, 1,
                     cfg.dim_size), jnp.float32),
                carry_recurrent_scratch_ref=pltpu.VMEM(
                    (cfg.seq_tile_size, cfg.num_v_heads, cfg.kq_head_dim,
                     cfg.v_head_dim), jnp.float32),
            )
        else:
            seq_metadata = metadata.compute_batched_seq_metadata(
                cfg=cfg,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                state_indices=state_indices,
                read_offsets=read_offsets,
                end_seq=distribution[0],
                read_indices=read_state_indices,
            )
            scratch_shapes = dict(
                carry_conv_scratch_ref=None,
                carry_recurrent_scratch_ref=None,
            )

        metadata_spec = jax.tree.map(lambda _: smem_spec, seq_metadata)

        out_shape = jax.ShapeDtypeStruct(
            (cfg.batch_size, cfg.num_v_heads, cfg.v_head_dim),
            cfg.dtypes.act_out,
        )
        if in_act is None and zero_initialize_out:
            aliased_out = jnp.zeros_like(out_shape)
        else:
            aliased_out = in_act

        num_metadata_leaves = len(seq_metadata)
        state_aliases = {
            num_metadata_leaves + CONV_STATE_OPERAND: 1,
            num_metadata_leaves + RECURRENT_STATE_OPERAND: 2,
        }
        if aliased_out is None:
            kernel_out_shape = out_shape
            aliased_out_spec = None
            input_output_aliases = state_aliases
        else:
            kernel_out_shape = aliased_out
            aliased_out_spec = hbm_spec
            input_output_aliases = state_aliases | {
                num_metadata_leaves + ALIASED_OUT_OPERAND: 0
            }

        if window_size > 1:
            vmem_fraction = WINDOWED_VMEM_FRACTION
            # Note (david): windows of different sizes compile to different
            # kernels; the suffix keeps them apart in profiles.
            kernel_name_suffix = f"_w{window_size}"
        else:
            vmem_fraction = DEFAULT_VMEM_FRACTION
            kernel_name_suffix = ""

        kernel_metadata = {}
        for path, leaf in jax.tree_util.tree_leaves_with_path(
                dataclasses.asdict(cfg)):
            key = jax.tree_util.keystr(path, simple=True, separator=".")
            if isinstance(leaf, str | int | float):
                kernel_metadata[key] = leaf
            else:
                kernel_metadata[key] = str(leaf)

        return pl.pallas_call(
            functools.partial(outer_kernel, cfg=cfg),
            out_shape=(kernel_out_shape, in_conv_state, in_recurrent_state),
            in_specs=(
                metadata_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
                aliased_out_spec,
                weights_spec,
            ),
            out_specs=(hbm_spec, hbm_spec, hbm_spec),
            scratch_shapes=scratch_shapes,
            input_output_aliases=input_output_aliases,
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                vmem_limit_bytes=int(vmem_fraction *
                                     tpu_info.vmem_capacity_bytes),
            ),
            name=f"fused_conv1d_gdn_{mode.value}{kernel_name_suffix}",
            metadata=kernel_metadata,
        )(
            seq_metadata,
            qkv,
            b,
            a,
            in_conv_state,
            in_recurrent_state,
            aliased_out,
            weights,
        )

    out_act, out_conv_state, out_recurrent_state = _call_kernel(
        conv_state, recurrent_state, None, config.GDNMode.BATCHED)
    out_act, out_conv_state, out_recurrent_state = _call_kernel(
        out_conv_state, out_recurrent_state, out_act, config.GDNMode.PER_SEQ)

    out_act = out_act.reshape(padded_batch_size, -1)[:batch_size]
    out_conv_state = out_conv_state.astype(conv_out_dtype)
    out_conv_state = out_conv_state.reshape(conv_state_shape)
    out_recurrent_state = out_recurrent_state.astype(recurrent_out_dtype)

    return (out_conv_state, out_recurrent_state), out_act
