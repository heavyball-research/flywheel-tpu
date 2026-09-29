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
"""KDA (Kimi Delta Attention) prefill + decode kernel, ported from
the gdn package with the same file layout so the two packages diff cleanly.

Source of the gdn package: https://github.com/vllm-project/tpu-inference @
4420cae36157, tpu_inference/kernels/gdn/v3/, package imports rewritten to
local. KDA decays per key channel: the gate is a third native-layout stream
g [batch, num_v_heads * kq_head_dim], activated as
-exp(a_log[h]) * softplus(g + dt_bias[h, d]), that decays the recurrent state
[K, V] row-wise, so the intra-chunk decayed Aqk no longer factors into
(q k^T) * exp(G_i - G_j).
"""
