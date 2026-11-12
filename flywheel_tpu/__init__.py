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

"""FlyWheel: attention kernels for TPU, implemented in JAX Pallas."""

import importlib
import warnings

from flywheel_tpu.softmax_attention import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)

__version__ = "0.4.1"

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
]

# Note (david): the internal names tpu-inference's flywheel backend imports
# (tpu_inference/layers/common/flywheel_attention.py, which pins
# flywheel-tpu==0.4.0 and reads them from the old flash_attn_interface and
# pallas/ modules). They are exported here for now, so its next bump needs no
# internal path, but they are not public API: each access warns.
INTERNAL_EXPORTS = {
    "PAGE_SIZE_MULTIPLE": "flywheel_tpu.softmax_attention.config",
    "FWD_BLOCKS": "flywheel_tpu.softmax_attention.config",
    "NUM_LANES": "flywheel_tpu.core.tpu",
    "merged_cache_heads": "flywheel_tpu.softmax_attention.kv_layout",
    "write_kv_cache_pages": "flywheel_tpu.softmax_attention.page",
}


# PEP 562 names the module hook __getattr__.
def __getattr__(name: str):  # pylint: disable=invalid-name
    module_name = INTERNAL_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'flywheel_tpu' has no attribute {name!r}")
    warnings.warn(
        f"flywheel_tpu.{name} is internal ({module_name}.{name}): it is"
        " exported for tpu-inference's flywheel backend for now, is not"
        " public API and may move or be removed.",
        FutureWarning,
        stacklevel=2,
    )
    return getattr(importlib.import_module(module_name), name)
