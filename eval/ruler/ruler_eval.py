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
"""RULER evaluation for one (backend, sequence length) point, with throughput.

RULER sizes its prompts from a tokenizer at run time, so one sequence length is
one engine and one process. This driver reuses the vLLM ownership helper in
the tpu-inference tree's `scripts/vllm/integration/lm_eval_accuracy.py` and
wraps lm-eval's vLLM backend so the generation phase reports its own token
counts: `_model_generate` is handed prompts that are already tokenized and
returns the vLLM outputs, so both halves of tokens/s are exact and cost nothing
extra to collect.
"""

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

GLUE = Path("scripts") / "vllm" / "integration" / "lm_eval_accuracy.py"


def _load_glue():
    """Import the vLLM-owning lm-eval helper, which lives outside any package.

    It is taken from the source tree of the tpu_inference the engine imports,
    so it always matches the patched backend.
    """
    package = importlib.util.find_spec("tpu_inference")
    if package is None:
        raise ModuleNotFoundError("tpu_inference is not installed")
    root = Path(package.origin).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("lm_eval_accuracy",
                                                  root / GLUE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GenerationMeter:
    """Totals of the generation phase, collected inside the vLLM backend.

    `before_generate`, when set, is handed each call's prompts before the
    call's clock starts.
    """

    def __init__(self):
        self.before_generate = None
        self.reset()

    def reset(self):
        self.calls = 0
        self.requests = 0
        self.prompt_tokens = 0
        self.generated_tokens = 0
        self.seconds = 0.0

    def install(self):
        from lm_eval.models.vllm_causallms import VLLM

        original = VLLM._model_generate

        def instrumented(inner_self, requests=None, *args, **kwargs):
            if self.before_generate is not None:
                self.before_generate(requests)
            start = time.perf_counter()
            outputs = original(inner_self, requests, *args, **kwargs)
            self.seconds += time.perf_counter() - start
            self.calls += 1
            self.requests += len(requests)
            self.prompt_tokens += sum(len(request) for request in requests)
            self.generated_tokens += sum(
                len(completion.token_ids) for output in outputs
                for completion in output.outputs)
            return outputs

        VLLM._model_generate = instrumented

    def report(self) -> dict:
        return {
            "generate_calls": self.calls,
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "generate_seconds": self.seconds,
            "prefill_tokens_per_second": self.prompt_tokens / self.seconds,
            "total_tokens_per_second":
            (self.prompt_tokens + self.generated_tokens) / self.seconds,
            "seconds_per_request": self.seconds / self.requests,
        }


BLOCK_SIZE = 128
KV_POOL_SLACK_BLOCKS = 8
# Matches VLLM_TPU_BUCKET_PADDING_GAP in run_ruler.sh and run_babilong.sh:
# past it, the runner pads a prefill to the next multiple of the gap.
BUCKET_GAP = 16384


def prefill_bucket(num_tokens: int) -> int:
    if num_tokens <= BUCKET_GAP:
        return 1 << (num_tokens - 1).bit_length()
    return -(-num_tokens // BUCKET_GAP) * BUCKET_GAP


def compile_signature(num_tokens: int, flywheel: bool, cp_size: int) -> tuple:
    """The static prefill shape one prompt of this length compiles.

    The token bucket, and on the FlyWheel backend without PCP also the kernel
    the runner picks for the step (tpu_runner.py): its cost model runs on the
    real length, so one bucket splits into a dense row near its top and a
    ragged kernel at some block span below that, each a backbone of its own.
    """
    bucket = prefill_bucket(num_tokens)
    if not flywheel or cp_size > 1:
        return (bucket, )
    from tpu_inference.layers.common.flywheel_tpu_attention import (
        dense_prefill_shape, varlen_block_span)
    dense_shape = dense_prefill_shape([[num_tokens]], bucket)
    span = (varlen_block_span([[num_tokens]], bucket)
            if dense_shape[0] < 0 else -1)
    return (bucket, dense_shape, span)


def signature_groups(lengths: list[int], flywheel: bool,
                     cp_size: int) -> dict[tuple, list[int]]:
    groups = {}
    for num_tokens in lengths:
        groups.setdefault(compile_signature(num_tokens, flywheel, cp_size),
                          []).append(num_tokens)
    return groups


def warmup(model, args, prompts: list[list[int]], warmed: set) -> None:
    """Compile every prefill shape these prompts reach, off the clock.

    RULER builds its prompts inside lm-eval, so their lengths are known only
    once the backend is handed them; this runs on each generate call's prompts
    before the call is timed. A prompt at the target length alone is not
    enough: on the FlyWheel backend a shorter prompt in the same bucket (the QA
    tasks') can pick another kernel and compile inside the timed call. Each
    compile_signature group is warmed with a prompt as long as its longest
    member, since any length in the group compiles the same program, and the
    full max_gen_toks decode covers the decode shapes.

    Precompiling the whole bucket ladder instead costs more than it saves at
    these lengths: the ladder runs past max_num_batched_tokens and compiles
    shapes -- including the prompt-logprobs graphs, which generate_until never
    needs -- that no RULER prompt reaches.
    """
    from vllm import SamplingParams, TokensPrompt

    flywheel = os.environ["USE_FLYWHEEL_TPU_KERNEL"] == "1"
    groups = signature_groups([len(prompt) for prompt in prompts], flywheel,
                              args.cp_size)
    unit = model.tokenizer(" the", add_special_tokens=False).input_ids
    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=args.max_gen_toks,
                                     ignore_eos=True)
    for signature, lengths in sorted(groups.items()):
        if signature in warmed:
            continue
        num_tokens = max(lengths)
        token_ids = (unit * (num_tokens // len(unit) + 1))[:num_tokens]
        start = time.perf_counter()
        model.model.generate([TokensPrompt(prompt_token_ids=token_ids)],
                             sampling_params=sampling_params,
                             use_tqdm=False)
        warmed.add(signature)
        print("warmup %s (%d tokens, %d prompts): %.1f s" %
              (signature, num_tokens, len(lengths),
               time.perf_counter() - start),
              flush=True)


def kv_block_budget(max_model_len: int, max_num_seqs: int) -> int:
    """KV blocks to pin so the pool holds exactly the sequences in flight.

    Left to size itself the pool fills HBM -- at 128k that was 10.65x what one
    sequence needs -- and since the KV buffers are donated arguments they count
    against the step's buffer assignment, so the prefill of a long prompt had
    no room left and the compile died with RESOURCE_EXHAUSTED.

    vLLM schedules and admits a request in BLOCK_SIZE-token blocks under
    prefill context parallelism too (a 5120-token request needs 40 blocks at
    pcp=2), so the budget never divides by the PCP size.
    """
    blocks_per_request = (-(-max_model_len // BLOCK_SIZE) +
                          KV_POOL_SLACK_BLOCKS)
    return max_num_seqs * blocks_per_request + 1


def yarn_rope_overrides(tokenizer_dir: str, factor: float) -> dict:
    """hf_overrides that switch the text model's RoPE to YaRN.

    The checkpoint's own rope_parameters (mRoPE sections, partial rotary
    factor, theta) are kept and YaRN is layered on top of them, with the
    native max_position_embeddings as the original context. vLLM replaces a
    dict-valued attribute wholesale, so the override carries every field.
    """
    config = json.loads((Path(tokenizer_dir) / "config.json").read_text())
    text_config = config["text_config"]
    rope = dict(text_config["rope_parameters"])
    if rope["rope_type"] != "default":
        raise ValueError(
            f"expected a default RoPE checkpoint, got {rope['rope_type']}")
    rope.update(rope_type="yarn",
                factor=factor,
                original_max_position_embeddings=text_config[
                    "max_position_embeddings"])
    return {"text_config": {"rope_parameters": rope}}


def build_model_args(args) -> dict:
    # Chunked prefill off means the engine has to admit a whole prompt in one
    # step, so the token budget has to cover max_model_len.
    max_model_len = args.seqlen + args.headroom
    max_num_batched_tokens = ((max_model_len + BLOCK_SIZE - 1) //
                              BLOCK_SIZE) * BLOCK_SIZE
    num_gpu_blocks_override = (args.num_gpu_blocks_override
                               or kv_block_budget(max_model_len,
                                                  args.max_num_seqs))
    model_args = {
        "num_gpu_blocks_override": num_gpu_blocks_override,
        "pretrained": args.model,
        "revision": args.revision,
        "tokenizer_revision": args.revision,
        "dtype": "bfloat16",
        "seed": args.seed,
        "tensor_parallel_size": args.tp_size,
        "prefill_context_parallel_size": args.cp_size,
        "max_model_len": max_model_len,
        "max_gen_toks": args.max_gen_toks,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "block_size": BLOCK_SIZE,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "async_scheduling": False,
        "language_model_only": True,
        "limit_mm_per_prompt": {
            "image": 0,
            "video": 0
        },
        "enable_thinking": False,
        "disable_log_stats": True,
    }
    if args.enable_sp:
        model_args["compilation_config"] = {
            "pass_config": {"enable_sp": True, "sp_min_token_num": 2048}
        }
    if args.yarn_factor is not None:
        model_args["hf_overrides"] = yarn_rope_overrides(
            args.tokenizer_dir, args.yarn_factor)
    return model_args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--tokenizer-dir",
        required=True,
        help="Snapshot directory RULER's own tokenizer lookup reads; it takes "
        "no revision, and the cache has no refs/main to resolve offline.")
    parser.add_argument("--seqlen", type=int, required=True)
    parser.add_argument("--tasks", default="ruler")
    parser.add_argument("--limit", type=float)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--enable-sp", action="store_true")
    parser.add_argument(
        "--yarn-factor",
        type=float,
        help="Serve with YaRN RoPE scaling of this factor over the "
        "checkpoint's native context.")
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-gen-toks", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-gpu-blocks-override",
        type=int,
        help="KV blocks to pin; defaults to exactly what max_num_seqs "
        "sequences of max_model_len need.")
    parser.add_argument(
        "--headroom",
        type=int,
        default=1024,
        help="Tokens reserved above the target length for the chat template "
        "wrapper and the generated answer.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cp_size < 1 or (args.cp_size > 1 and args.cp_size % 2):
        parser.error("--cp-size must be 1 or a positive even integer")

    from lm_eval.utils import make_table, setup_logging

    setup_logging()
    glue = _load_glue()

    meter = GenerationMeter()
    meter.install()

    tasks = [task.strip() for task in args.tasks.split(",")]
    metadata = {
        "max_seq_lengths": [args.seqlen],
        "tokenizer": args.tokenizer_dir,
    }
    model_args = build_model_args(args)

    import lm_eval

    model = glue._create_vllm_model(model_args, "auto", None, None)
    warmed = set()
    meter.before_generate = (
        lambda prompts: warmup(model, args, prompts, warmed))
    started = time.perf_counter()
    try:
        results = lm_eval.simple_evaluate(
            model=model,
            tasks=tasks,
            batch_size="auto",
            limit=args.limit,
            apply_chat_template=True,
            metadata=metadata,
        )
    finally:
        # vLLM's synchronous client stops EngineCore through garbage
        # collection, which reference cycles can defer indefinitely.
        model.model.llm_engine.engine_core.shutdown()
    wall_seconds = time.perf_counter() - started

    args.out.mkdir(parents=True, exist_ok=True)
    scores = {
        task: {
            metric: value
            for metric, value in task_results.items()
        }
        for task, task_results in results["results"].items()
    }
    # The kernel selection lives in the environment, so record it with the
    # numbers it produced.
    kernel_env = {
        name: os.environ.get(name, "")
        for name in ("USE_FLYWHEEL_TPU_KERNEL",
                     "USE_FLYWHEEL_TPU_LINEAR_KERNEL",
                     "RPA_V3_MIXED_BLOCK_SIZES", "RPA_V3_PREFILL_BLOCK_SIZES",
                     "RPA_V3_DECODE_BLOCK_SIZES", "MODEL_IMPL_TYPE",
                     "NEW_MODEL_DESIGN", "TPU_MULTIHOST_BACKEND")
    }
    record = {
        "model": args.model,
        "revision": args.revision,
        "kernel_env": kernel_env,
        "seqlen": args.seqlen,
        "tasks": tasks,
        "limit": args.limit,
        "max_num_seqs": args.max_num_seqs,
        "tp_size": args.tp_size,
        "cp_size": args.cp_size,
        "enable_sp": args.enable_sp,
        "hf_overrides": model_args.get("hf_overrides"),
        "warmup_signatures": sorted(warmed),
        "num_gpu_blocks_override": model_args["num_gpu_blocks_override"],
        "wall_seconds": wall_seconds,
        "throughput": meter.report(),
        "scores": scores,
    }
    (args.out / "ruler_run.json").write_text(json.dumps(record, indent=2))
    print(make_table(results))
    if "groups" in results:
        print(make_table(results, "groups"))
    print(json.dumps(record["throughput"], indent=2))


if __name__ == "__main__":
    main()
