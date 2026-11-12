# Copyright 2023 The JAX Authors.
# Modifications copyright 2026 heavyball-research
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

"""TPU facts kernels are built against: tile geometry and dtype sizes."""

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
