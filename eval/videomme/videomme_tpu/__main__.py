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
"""lmms-eval's CLI with the vllm_generate_tpu chat model registered.

lmms-eval's LMMS_EVAL_PLUGINS hook registers a plugin model as a simple one,
which its registry then rejects for a chat model such as this subclass of
vllm_generate, so the manifest is registered here before the CLI resolves
`--model`. Every argument is lmms-eval's own.
"""

import torch.distributed
from lmms_eval.__main__ import cli_evaluate
from lmms_eval.models import MODEL_REGISTRY_V2
from lmms_eval.models.registry_v2 import ModelManifest

_barrier = torch.distributed.barrier


def _single_process_barrier(*args, **kwargs):
    """A barrier for the one-process world an in-process vLLM engine sets up.

    lmms-eval synchronizes ranks whenever torch.distributed is initialized,
    and vLLM initializes a world of one. torch's barrier then asks for the
    default accelerator, which torchax registers as PrivateUse1 without the
    hooks that query needs. With one process there is nothing to wait for.
    """
    if torch.distributed.get_world_size() != 1:
        return _barrier(*args, **kwargs)
    return None


if __name__ == "__main__":
    torch.distributed.barrier = _single_process_barrier
    MODEL_REGISTRY_V2.register_manifest(
        ModelManifest(
            model_id="vllm_generate_tpu",
            chat_class_path=(
                "videomme_tpu.models.vllm_generate_tpu.VLLMGenerateTPU"),
        ))
    cli_evaluate()
