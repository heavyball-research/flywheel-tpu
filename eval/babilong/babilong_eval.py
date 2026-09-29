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
"""BABILong evaluation for one (backend, length) point, with throughput.

Prompts and scoring come from the official repository
(https://github.com/booydar/babilong), checked out at a pinned commit and
passed in with --babilong-repo: `get_formatted_input` with the instruction,
examples and post-prompt of each task, the chat template with the system
prompt its Qwen scripts use, greedy decoding of 20 new tokens, and
`compare_answers`. The one departure is `enable_thinking=False`: the official
script predates Qwen3's thinking mode, which would spend the 20-token budget
inside <think>.

The engine runs one sequence at a time with chunked prefill and prefix caching
off, like the RULER runs, and every request is timed on its own. Results are
appended to predictions.jsonl as they land, so a killed run resumes where it
stopped.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ruler"))
from ruler_eval import (  # noqa: E402
    BLOCK_SIZE, kv_block_budget, signature_groups, yarn_rope_overrides)

SYSTEM_PROMPT = "You are a helpful assistant."


def load_samples(data_dir: Path, task: str, length: str) -> list[dict]:
    return json.loads((data_dir / task / f"{length}.json").read_text())


def build_prompts(args, tokenizer) -> list[dict]:
    """Every sample of every task as token ids, formatted the official way."""
    sys.path.insert(0, str(args.babilong_repo))
    from babilong.prompts import (DEFAULT_PROMPTS, DEFAULT_TEMPLATE,
                                  get_formatted_input)

    items = []
    for task in args.tasks:
        samples = load_samples(args.data_dir, task, args.length)
        if args.limit is not None:
            samples = samples[:args.limit]
        prompt_cfg = DEFAULT_PROMPTS[task]
        texts = []
        for sample in samples:
            user_text = get_formatted_input(sample["input"],
                                            sample["question"],
                                            prompt_cfg["examples"],
                                            prompt_cfg["instruction"],
                                            prompt_cfg["post_prompt"],
                                            template=DEFAULT_TEMPLATE)
            messages = [{
                "role": "system",
                "content": SYSTEM_PROMPT
            }, {
                "role": "user",
                "content": user_text
            }]
            texts.append(
                tokenizer.apply_chat_template(messages,
                                              tokenize=False,
                                              add_generation_prompt=True,
                                              enable_thinking=False))
        # The rendered template already carries every special token.
        token_ids = tokenizer(texts, add_special_tokens=False).input_ids
        for index, (sample, ids) in enumerate(zip(samples, token_ids)):
            items.append({
                "task": task,
                "index": index,
                "question": sample["question"],
                "target": sample["target"],
                "token_ids": ids,
            })
    return items


def length_stats(items: list[dict]) -> dict:
    by_task = {}
    for item in items:
        by_task.setdefault(item["task"], []).append(len(item["token_ids"]))
    stats = {
        task: {
            "samples": len(lengths),
            "min": min(lengths),
            "mean": statistics.fmean(lengths),
            "max": max(lengths),
        }
        for task, lengths in by_task.items()
    }
    all_lengths = [len(item["token_ids"]) for item in items]
    stats["all"] = {
        "samples": len(all_lengths),
        "min": min(all_lengths),
        "mean": statistics.fmean(all_lengths),
        "max": max(all_lengths),
    }
    return stats


def prompt_lengths(items: list[dict]) -> list[int]:
    return [len(item["token_ids"]) for item in items]


def native_context(snapshot: Path) -> int:
    config = json.loads((snapshot / "config.json").read_text())
    return config["text_config"]["max_position_embeddings"]


def build_engine_args(args, max_model_len: int) -> dict:
    # Chunked prefill off means one step admits a whole prompt, so the token
    # budget has to cover max_model_len.
    engine_args = {
        "model": str(args.snapshot),
        "tokenizer": str(args.snapshot),
        "dtype": "bfloat16",
        "seed": args.seed,
        "tensor_parallel_size": args.tp_size,
        "prefill_context_parallel_size": args.cp_size,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_model_len,
        "max_num_seqs": 1,
        "num_gpu_blocks_override": kv_block_budget(max_model_len, 1),
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
        "disable_log_stats": True,
    }
    if args.enable_sp:
        engine_args["compilation_config"] = {
            "pass_config": {
                "enable_sp": True,
                "sp_min_token_num": 2048
            }
        }
    if args.yarn_factor is not None:
        engine_args["hf_overrides"] = yarn_rope_overrides(
            str(args.snapshot), args.yarn_factor)
    return engine_args


def run_config(args, max_model_len: int) -> dict:
    """What a resumed run must share with the one it continues."""
    return {
        "revision": args.revision,
        "length": args.length,
        "tasks": args.tasks,
        "limit": args.limit,
        "tp_size": args.tp_size,
        "cp_size": args.cp_size,
        "enable_sp": args.enable_sp,
        "yarn_factor": args.yarn_factor,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": max_model_len,
        "system_prompt": SYSTEM_PROMPT,
        "babilong_commit": args.babilong_commit,
        "kernel_env": {
            name: os.environ.get(name, "")
            for name in ("USE_FLYWHEEL_TPU_KERNEL",
                         "USE_FLYWHEEL_TPU_LINEAR_KERNEL",
                         "RPA_V3_MIXED_BLOCK_SIZES", "MODEL_IMPL_TYPE",
                         "NEW_MODEL_DESIGN", "TPU_MULTIHOST_BACKEND")
        },
    }


def warmup(llm, items: list[dict], sampling_params, cp_size: int) -> None:
    """Compile every static prefill shape the dataset reaches, off the clock.

    Each compile_signature group is warmed with a prompt as long as its
    longest sample; any length in the group compiles the same program.
    """
    from vllm import TokensPrompt

    flywheel = os.environ["USE_FLYWHEEL_TPU_KERNEL"] == "1"
    groups = signature_groups(prompt_lengths(items), flywheel, cp_size)
    filler = items[0]["token_ids"]
    for signature, lengths in sorted(groups.items()):
        num_tokens = max(lengths)
        token_ids = (filler * (num_tokens // len(filler) + 1))[:num_tokens]
        start = time.perf_counter()
        llm.generate([TokensPrompt(prompt_token_ids=token_ids)],
                     sampling_params=sampling_params,
                     use_tqdm=False)
        print("warmup %s (%d tokens, %d samples): %.1f s" %
              (signature, num_tokens, len(lengths),
               time.perf_counter() - start),
              flush=True)


def summarize(records: list[dict], tasks: list[str]) -> dict:
    accuracy = {}
    for task in tasks:
        task_records = [r for r in records if r["task"] == task]
        accuracy[task] = sum(r["correct"]
                             for r in task_records) / len(task_records)
    seconds = [r["seconds"] for r in records]
    prompt_tokens = sum(r["prompt_tokens"] for r in records)
    generated_tokens = sum(r["generated_tokens"] for r in records)
    total_seconds = sum(seconds)
    return {
        "accuracy": accuracy,
        "accuracy_avg": statistics.fmean(accuracy.values()),
        "throughput": {
            "requests": len(records),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "generate_seconds": total_seconds,
            "prefill_tokens_per_second": prompt_tokens / total_seconds,
            "seconds_per_request": total_seconds / len(records),
            "seconds_per_request_median": statistics.median(seconds),
            "seconds_per_request_max": max(seconds),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--babilong-repo", type=Path, required=True)
    parser.add_argument("--babilong-commit", required=True)
    parser.add_argument("--length", required=True, help="e.g. 128k")
    parser.add_argument("--tasks", default="qa1,qa2,qa3,qa4,qa5")
    parser.add_argument("--limit",
                        type=int,
                        help="Samples per task (default: all 100).")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--enable-sp", action="store_true")
    parser.add_argument("--yarn-factor", type=float)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--measure-only",
        action="store_true",
        help="Tokenize the prompts, report their lengths and exit.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.tasks = [task.strip() for task in args.tasks.split(",")]
    if args.cp_size < 1 or (args.cp_size > 1 and args.cp_size % 2):
        parser.error("--cp-size must be 1 or a positive even integer")

    from transformers import AutoTokenizer

    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    started = time.perf_counter()
    items = build_prompts(args, tokenizer)
    stats = length_stats(items)
    print("tokenized %d prompts in %.1f s" %
          (len(items), time.perf_counter() - started),
          flush=True)
    print(json.dumps(stats, indent=2), flush=True)
    (args.out / "prompt_lengths.json").write_text(json.dumps(stats, indent=2))

    max_model_len = -(-(stats["all"]["max"] + args.max_new_tokens) //
                      BLOCK_SIZE) * BLOCK_SIZE
    position_limit = native_context(args.snapshot) * (args.yarn_factor or 1)
    print("max_model_len %d, position limit %d" %
          (max_model_len, position_limit),
          flush=True)
    if max_model_len > position_limit:
        raise SystemExit(
            f"the longest prompt needs max_model_len {max_model_len}, past "
            f"the {position_limit} positions this RoPE setup covers")
    if args.measure_only:
        for flywheel in (False, True):
            groups = signature_groups(prompt_lengths(items), flywheel,
                                      args.cp_size)
            print("compile signatures, flywheel=%s:" % flywheel)
            for signature, lengths in sorted(groups.items()):
                print("  %s: %d samples, %d-%d tokens" %
                      (signature, len(lengths), min(lengths), max(lengths)))
        return

    config = run_config(args, max_model_len)
    config_path = args.out / "run_config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved != config:
            raise SystemExit(
                f"{config_path} differs from this run; use a fresh --out:\n"
                f"saved={saved}\nnow={config}")
    else:
        config_path.write_text(json.dumps(config, indent=2))

    predictions_path = args.out / "predictions.jsonl"
    records = []
    if predictions_path.exists():
        records = [
            json.loads(line)
            for line in predictions_path.read_text().splitlines()
        ]
    done = {(r["task"], r["index"]) for r in records}
    pending = [
        item for item in items if (item["task"], item["index"]) not in done
    ]
    print("%d done, %d pending" % (len(done), len(pending)), flush=True)

    sys.path.insert(0, str(args.babilong_repo))
    from babilong.metrics import TASK_LABELS, compare_answers
    from vllm import LLM, SamplingParams, TokensPrompt

    engine_args = build_engine_args(args, max_model_len)
    print("engine args: %s" %
          json.dumps({
              k: v
              for k, v in engine_args.items() if k != "hf_overrides"
          }),
          flush=True)
    llm = LLM(**engine_args)
    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=args.max_new_tokens)
    try:
        if pending:
            warmup(llm, pending, sampling_params, args.cp_size)
        with predictions_path.open("a") as predictions:
            for position, item in enumerate(pending):
                start = time.perf_counter()
                outputs = llm.generate(
                    [TokensPrompt(prompt_token_ids=item["token_ids"])],
                    sampling_params=sampling_params,
                    use_tqdm=False)
                seconds = time.perf_counter() - start
                completion = outputs[0].outputs[0]
                output = completion.text.strip()
                record = {
                    "task": item["task"],
                    "index": item["index"],
                    "prompt_tokens": len(item["token_ids"]),
                    "generated_tokens": len(completion.token_ids),
                    "seconds": seconds,
                    "question": item["question"],
                    "target": item["target"],
                    "output": output,
                    "correct": compare_answers(item["target"], output,
                                               item["question"],
                                               TASK_LABELS[item["task"]]),
                }
                predictions.write(json.dumps(record) + "\n")
                predictions.flush()
                records.append(record)
                print("[%d/%d] %s #%d %d tok %.2f s correct=%s out=%r" %
                      (position + 1, len(pending), item["task"],
                       item["index"], record["prompt_tokens"], seconds,
                       record["correct"], output[:60]),
                      flush=True)
    finally:
        # vLLM's synchronous client stops EngineCore through garbage
        # collection, which reference cycles can defer indefinitely.
        llm.llm_engine.engine_core.shutdown()

    summary = summarize(records, args.tasks)
    result = {
        **config,
        "num_gpu_blocks_override": engine_args["num_gpu_blocks_override"],
        "prompt_lengths": stats,
        **summary,
    }
    (args.out / "babilong_run.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
