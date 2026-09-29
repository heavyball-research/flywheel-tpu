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

import jax
import jax.numpy as jnp

from gdn_v3 import config


def causal_conv1d(
    real_sizes: jax.Array,
    conv_input: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Causal Conv1D over a tile in the compact layout.

    Takes real_sizes [seq], conv_input [seq, prev_kernel_size + chunk, 1,
    dim_size] (the previous conv state followed by the tile's tokens),
    conv_weight [kernel_size, 1, dim_size] and conv_bias [dim_size].

    Returns the output [seq, chunk, 1, dim_size] and the conv states [seq,
    window_size, prev_kernel_size, 1, dim_size]: checkpoint w holds the last
    prev_kernel_size inputs ending at token chunk_size - window_size + w,
    clamped to the last real token, i.e. the state a sequence resuming right
    after that token starts from.
    """
    assert conv_input.ndim == 4

    conv_out_rows = []
    for c_idx in range(cfg.chunk_size):
        tap_sum = jnp.zeros((cfg.seq_tile_size, 1, cfg.dim_size), jnp.float32)
        for tap in range(cfg.kernel_size):
            tap_sum += conv_input[:, c_idx + tap] * conv_weight[tap:tap + 1]
        if conv_bias is None:
            conv_out_rows.append(tap_sum)
        else:
            conv_out_rows.append(tap_sum + conv_bias.reshape(1, 1, -1))

    # Note (david): real_sizes may be smaller than chunk_size, so the last
    # prev_kernel_size rows are not always the final state; each checkpoint
    # instead selects the rows ending at the sequence's last real token. The
    # loops are static, so the compiler fuses the repeated selects.
    real_sizes = real_sizes.reshape(-1, 1, 1, 1)
    conv_states = []
    for w_idx in range(cfg.window_size):
        last_row = 1 + cfg.chunk_size - cfg.window_size + w_idx
        # Note (david): the default serves sequences whose last real token is
        # at or past this window position. c_idx 0 would select the previous
        # conv state of an empty sequence, which writes no checkpoint.
        conv_state = conv_input[:, last_row:last_row + cfg.prev_kernel_size]
        for c_idx in range(1, last_row):
            conv_state = jnp.where(
                c_idx == real_sizes,
                conv_input[:, c_idx:c_idx + cfg.prev_kernel_size],
                conv_state,
            )
        conv_states.append(conv_state)

    return jnp.stack(conv_out_rows, axis=1), jnp.stack(conv_states, axis=1)
