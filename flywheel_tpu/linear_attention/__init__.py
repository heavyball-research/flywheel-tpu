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

"""Fused conv1d + GDN / KDA linear-attention kernels for TPU."""

from flywheel_tpu.linear_attention.gdn.wrapper import (
    fused_conv1d_gdn,
    fused_conv1d_gdn_replay,
)
from flywheel_tpu.linear_attention.kda.wrapper import (
    fused_conv1d_kda,
    fused_conv1d_kda_replay,
)

__all__ = [
    "fused_conv1d_gdn",
    "fused_conv1d_gdn_replay",
    "fused_conv1d_kda",
    "fused_conv1d_kda_replay",
]
