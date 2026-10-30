# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TPU facts kernels are built against: tile geometry, dtype sizes, the VMEM
limit and the chip's device name."""

import functools
import re

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

NUM_LANES = 128
NUM_SUBLANES = 8
# Note (david): Mosaic tiles a bf16 VMEM scratch whose row count is a multiple
# of 16 as (16, 128) (v7x does; v6e may keep (8, 128)), and a dynamic vector
# load or store must start on that tile. Dynamic row accesses into bf16 stages
# therefore step in whole 16-row tiles, which is legal under either tiling;
# HBM DMAs stay on the hbm_bf16_tile_rows() grid.
BF16_TILE_ROWS = 2 * NUM_SUBLANES
BF16_BYTES = 2
F32_BYTES = 4
# Kernels on v6e's 128 MiB cores are built against 100 MiB of VMEM; a core with
# less is built against all of it (64 MiB on v7x).
MAX_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


@functools.cache
def vmem_limit_bytes() -> int:
    """The scoped VMEM limit kernels are planned and compiled against.

    Asked on first use, not at import: get_tpu_info() starts the TPU backend,
    which jax.distributed.initialize() has to precede. Off TPU (the Pallas
    interpreter) there is no chip to ask, so builds keep the v6e limit.
    """
    if jax.default_backend() != "tpu":
        return MAX_VMEM_LIMIT_BYTES
    return min(MAX_VMEM_LIMIT_BYTES, pltpu.get_tpu_info().vmem_capacity_bytes)


@functools.cache
def hbm_bf16_tile_rows() -> int:
    """Rows a traced window into a bf16 HBM operand has to be aligned to.

    It is TpuInfo.get_sublane_tiling for bf16. Before v7x that is 8: XLA tiles
    a bf16 HBM buffer (8, 128), unless
    xla_tpu_enable_large_2nd_minor_layout_for_x16 is set. On v7x it is 16: There
    XLA lays out a bf16 HBM buffer whose row count is a multiple of 16 in one of
    two ways:
    - (16, 128) tiles when another op of the jitted program reads the buffer,
      as with a kernel's output inside a model;
    - (8, 128) tiles when the jitted program returns the buffer, as when a
      test calls the kernel alone.
    A window that starts on a multiple of 16 rows is aligned under both.
    """
    if jax.default_backend() != "tpu":
        return NUM_SUBLANES
    return pltpu.get_tpu_info().get_sublane_tiling(jnp.bfloat16)


def get_device_variant_name(device_kind: str | None = None) -> str:
    """'TPU v6 lite' / 'TPU7x' -> 'TPU v6e' / 'TPU v7' (generation plus e/p).

    device_kind defaults to the chip JAX runs on.
    """
    if device_kind is None:
        device_kind = jax.devices()[0].device_kind
    kind_match = re.fullmatch(r"TPU\s*v?(\d+)(?:(e|p)|\s+(lite))?", device_kind)
    if device_kind == "TPU7x":
        return "TPU v7"
    elif kind_match is None:
        return device_kind
    elif kind_match.group(2):
        return f"TPU v{kind_match.group(1)}{kind_match.group(2)}"
    elif kind_match.group(3):
        return f"TPU v{kind_match.group(1)}e"
    else:
        return f"TPU v{kind_match.group(1)}"


def get_device_name(device_kind: str | None = None) -> str:
    """'TPU v6 lite' / 'TPU v5e' / ... -> 'TPU v6' (generation key, as JAX
    does)."""
    return re.sub(
        r"^(TPU v\d+)[ep]$", r"\1", get_device_variant_name(device_kind)
    )
