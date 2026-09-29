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
import enum

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

NUM_BUFFERS = 2


class GDNMode(enum.StrEnum):
    """Tiling of the sequences a kernel call covers.

    BATCHED packs several sequences per tile, each holding at most window_size
    tokens (a decode token or a speculative verify window). PER_SEQ tiles one
    sequence along its tokens.
    """

    BATCHED = enum.auto()
    PER_SEQ = enum.auto()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Dtypes:
    act_in: jnp.dtype
    act_out: jnp.dtype
    compute: jnp.dtype
    recurrent_state: jnp.dtype
    conv_state: jnp.dtype


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GDNConfig:
    mode: GDNMode
    dtypes: Dtypes
    batch_size: int
    dim_size: int
    kernel_size: int
    tile_size: int
    num_kq_heads: int
    num_v_heads: int
    kq_head_dim: int
    v_head_dim: int
    num_buffers: int = NUM_BUFFERS
    # Note (david): max tokens per speculative verify window
    # (num_speculative_tokens + 1), which is also the number of state
    # checkpoints kept per sequence. A sequence reads its initial state from
    # state_indices[s] + read_offset[s] and checkpoints window position t to
    # state_indices[s] + t, so rejected draft tokens roll back by checkpoint
    # selection. It is 1 without speculative decoding, so shapes and loops size
    # off it unconditionally and the extra axis folds away.
    window_size: int = 1

    @property
    def chunk_size(self) -> int:
        return self.tile_size if self.mode == GDNMode.PER_SEQ else self.window_size

    @property
    def seq_tile_size(self) -> int:
        return 1 if self.mode == GDNMode.PER_SEQ else self.tile_size

    @property
    def prev_kernel_size(self) -> int:
        return self.kernel_size - 1

    @property
    def use_recurrent(self) -> bool:
        # Note (david): more than one state checkpoint per sequence needs the
        # token-recurrent scan, since the chunked path only produces the final
        # state.
        return self.chunk_size == 1 or self.window_size > 1

    @property
    def v_per_kq_head(self) -> int:
        return self.num_v_heads // self.num_kq_heads

    @property
    def aligned_num_v_heads(self) -> int:
        num_lanes = pltpu.get_tpu_info().num_lanes
        return pl.cdiv(self.num_v_heads, num_lanes) * num_lanes
