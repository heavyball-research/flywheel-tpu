"""vLLM TorchTPU's KDA behind flywheel's fused_conv1d_kda signature.

    bench_kda_fwd.py --pkgs vllm_torchtpu_kda
    bench_replay_decode.py --kernel kda --pkg vllm_torchtpu_kda

vLLM TorchTPU serves KDA on its GDN v3 kernel with
attention_mode=AttentionMode.KDA, and this wrapper makes the call its
kda_attention_op.py makes: the raw gate as `a`, dt_bias as [n_v, d_k] and the
optional lower bound as gate_lower_bound. Its gate is flywheel's:
-exp(a_log) * softplus(g + dt_bias), or
lower_bound * sigmoid(exp(a_log) * (g + dt_bias)) under a bound.

The kernel is the third_party/vllm-torchtpu pin, loaded through
benchmarks/common/vllm_torchtpu.py when this module is imported, so import it
before JAX starts its backend and time flywheel in another process.

vLLM TorchTPU's mixed path has one knob, its DMA tile, which is also its math
chunk. As bench_gdn_fwd.py does for such kernels, the tile is
compute_chunk_size (64 by default, also vLLM TorchTPU's serving default);
flywheel's mixed_tile_size has no counterpart. Decode tiles keep vLLM
TorchTPU's default of 4, which the kernel itself drops to 1 for KDA verify
windows.
"""

from __future__ import annotations

from benchmarks.common.vllm_torchtpu import (
    DEFAULT_VLLM_TORCHTPU_SOURCE,
    load_gdn_v3,
)

gdn_wrapper, gdn_config = load_gdn_v3(DEFAULT_VLLM_TORCHTPU_SOURCE)


def fused_conv1d_kda(qkv, b, g, conv_state, recurrent_state, conv_weight,
                     conv_bias, a_log, dt_bias, query_start_loc, state_indices,
                     distribution, seq_lens, read_state_indices,
                     read_offsets=None, *, n_kq, n_v, d_k, d_v, kernel_size,
                     num_spec_tokens=0, decode_tile_size=4,
                     mixed_tile_size=None, compute_chunk_size=64,
                     lower_bound=None):
    """((conv_state, recurrent_state), out), as flywheel's fused_conv1d_kda."""
    del mixed_tile_size
    # vLLM TorchTPU reads each sequence's state from its own slot.
    if read_state_indices is not state_indices:
        raise ValueError("read_state_indices must be state_indices.")
    return gdn_wrapper.fused_conv1d_gdn(
        qkv, b, g, conv_state, recurrent_state, conv_weight, conv_bias,
        a_log, dt_bias.reshape(n_v, d_k), query_start_loc, state_indices,
        distribution, seq_lens, read_offsets, n_kq=n_kq, n_v=n_v, d_k=d_k,
        d_v=d_v, kernel_size=kernel_size,
        attention_mode=gdn_config.AttentionMode.KDA,
        gate_lower_bound=lower_bound, num_spec_tokens=num_spec_tokens,
        decode_tile_size=decode_tile_size, mixed_tile_size=compute_chunk_size)
