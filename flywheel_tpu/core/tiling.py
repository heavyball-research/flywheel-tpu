# Copyright 2026 heavyball-research
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

"""Integer helpers for sizing tiles and blocks."""

import math

import jax.numpy as jnp

from flywheel_tpu.core.tpu import F32_BYTES, NUM_LANES, NUM_SUBLANES


def round_up(value: int, alignment: int) -> int:
    """Smallest multiple of alignment that is >= value."""
    return -(-value // alignment) * alignment


def vmem_buffer_bytes(shape: tuple[int, ...], dtype: jnp.dtype) -> int:
    """Bytes of one VMEM scratch buffer under Mosaic's v6 memref tiling.

    A second-minor axis shorter than one large tile pads only to a power of two,
    so a (2, head_dim) bf16 head pair costs no pad.
    """
    itemsize = jnp.dtype(dtype).itemsize
    *leading, rows, lanes = shape
    packing = F32_BYTES // itemsize
    large_tile_rows = NUM_SUBLANES * packing
    if rows >= large_tile_rows:
        tile_rows = large_tile_rows
    else:
        short_rows = min(rows, NUM_SUBLANES)
        tile_rows = max(packing, 1 << max(0, short_rows - 1).bit_length())
    return (
        math.prod(leading)
        * round_up(rows, tile_rows)
        * round_up(lanes, NUM_LANES)
        * itemsize
    )
