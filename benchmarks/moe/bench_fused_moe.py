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

"""Latency of fused_moe against vllm-torchtpu's and tpu-inference's MoE paths
and XLA, on TPU.

Every implementation starts from the router logits [T, E] (softmax, top-k,
renormalized) and returns the bf16 combined output; flywheel is
fused_moe_softmax_topk (on v7x the routing, permute and combine run on
SparseCore). See benchmarks/common/moe.py for the baselines. Each baseline's
output is checked against flywheel's before it is timed.

    PYTHONPATH=. uv run --no-sync python benchmarks/moe/bench_fused_moe.py \\
        --models qwen3-30b-a3b qwen3-next-80b-a3b \\
        --tokens 8 64 512 4096 16384 65536 --output results/moe.json

    # flywheel alone, experts larger than VMEM streamed in blocks.
    PYTHONPATH=. uv run --no-sync python benchmarks/moe/bench_fused_moe.py \\
        --shape 8192 2048 16 8 --tokens 16384 --impls flywheel \\
        --weight-block-columns 256 --tile-m 512 --gather-pipeline-rows 64
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp

from benchmarks.common import moe as baselines
from benchmarks.common.timing import time_call
from flywheel_tpu.moe import (
    PipelineDepths,
    fused_moe_softmax_topk,
    prepare_expert_weights,
)

# (hidden, intermediate, experts, top_k) of the routed experts.
MODELS = {
    "qwen3-30b-a3b": (2048, 768, 128, 8),
    "qwen3-next-80b-a3b": (2048, 512, 512, 10),
    "h2048-i1408-e256-k8": (2048, 1408, 256, 8),
}
IMPLS = ("flywheel", "gmm", "ragged_dot", "torchtpu")
# Note (david): torchtpu needs a vllm-torchtpu checkout (--torchtpu-source),
# so it runs only when asked.
DEFAULT_IMPLS = ("flywheel", "gmm", "ragged_dot")
# Note (david): a baseline that disagrees with flywheel by more than this,
# relative to the output's largest magnitude, is reported as a mismatch, not
# timed.
MAX_REL_ERR = 2e-2
# Note (david): a rejected baseline's error is cut to its first line and
# this many characters, to keep the table readable.
MAX_ERROR_CHARS = 120
# Note (david): each token's logits are a permutation of (k - E / 2) /
# LOGIT_RANK_DIVISOR: distinct, exact in bf16 and O(1).
LOGIT_RANK_DIVISOR = 64


def make_inputs(num_tokens, hidden, intermediate, num_experts):
    """bf16 activations, bf16 router logits and bf16 weights scaled by
    1 / sqrt(fan_in); w1 is the merged gate|up [E, hidden, 2 * inter].

    Each token's logits are a random permutation of distinct values, so
    every implementation picks the same experts with no top-k tie to break.
    """
    k_x, k_logits, k_w1, k_w2 = jax.random.split(jax.random.key(0), 4)
    x = jax.random.normal(k_x, (num_tokens, hidden), jnp.bfloat16)
    ranks = jnp.argsort(
        jax.random.uniform(k_logits, (num_tokens, num_experts)), axis=-1
    )
    logits = ((ranks - num_experts // 2) / LOGIT_RANK_DIVISOR).astype(
        jnp.bfloat16
    )
    w1 = (
        jax.random.normal(
            k_w1, (num_experts, hidden, 2 * intermediate), jnp.float32
        )
        * hidden**-0.5
    ).astype(jnp.bfloat16)
    w2 = (
        jax.random.normal(
            k_w2, (num_experts, intermediate, hidden), jnp.float32
        )
        * intermediate**-0.5
    ).astype(jnp.bfloat16)
    return x, logits, w1, w2


def make_impls(names, w1, w2, top_k, knobs, baseline_kernels, torchtpu):
    """name -> (jitted fn(x, logits, *weights), weights)."""
    impls = {}
    if "flywheel" in names:
        intermediate = w2.shape[1]
        impls["flywheel"] = (
            jax.jit(
                functools.partial(fused_moe_softmax_topk, top_k=top_k, **knobs)
            ),
            prepare_expert_weights(
                w1[..., :intermediate], w1[..., intermediate:], w2
            ),
        )
    if "gmm" in names:
        gmm_v2, dense_gather_reduce = baseline_kernels
        impls["gmm"] = (
            jax.jit(
                functools.partial(
                    baselines.gmm_moe,
                    top_k=top_k,
                    gmm_v2=gmm_v2,
                    dense_gather_reduce=dense_gather_reduce,
                )
            ),
            (w1, w2),
        )
    if "torchtpu" in names:
        fused_moe_func, router_topk, onehot_threshold = torchtpu
        impls["torchtpu"] = (
            jax.jit(
                functools.partial(
                    baselines.torchtpu_moe,
                    top_k=top_k,
                    fused_moe_func=fused_moe_func,
                    router_topk=router_topk,
                    onehot_threshold=onehot_threshold,
                )
            ),
            (w1, w2),
        )
    if "ragged_dot" in names:
        impls["ragged_dot"] = (
            jax.jit(functools.partial(baselines.ragged_dot_moe, top_k=top_k)),
            (w1, w2),
        )
    return impls


def run_baseline(fn: Callable, args: tuple, ref: jax.Array):
    """(out, None), or (None, why) when the baseline rejects the shape or its
    output disagrees with flywheel's."""
    try:
        out = jax.block_until_ready(fn(*args))
    except (
        ValueError,
        AssertionError,
        NotImplementedError,
        jax.errors.JaxRuntimeError,
    ) as e:
        return None, f"unsupported: {str(e).splitlines()[0][:MAX_ERROR_CHARS]}"
    ref32 = ref.astype(jnp.float32)
    rel_err = float(
        jnp.max(jnp.abs(out.astype(jnp.float32) - ref32))
        / jnp.max(jnp.abs(ref32))
    )
    if math.isnan(rel_err) or rel_err > MAX_REL_ERR:
        return None, f"mismatch: relative error {rel_err:.3g}"
    return out, None


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS))
    parser.add_argument(
        "--shape",
        type=int,
        nargs=4,
        action="append",
        metavar=("HIDDEN", "INTER", "EXPERTS", "TOPK"),
        help="a custom shape; repeatable",
    )
    parser.add_argument(
        "--impls", nargs="+", choices=IMPLS, default=DEFAULT_IMPLS
    )
    parser.add_argument(
        "--tail-bucket", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--tile-m", type=int)
    parser.add_argument("--weight-block-columns", type=int)
    parser.add_argument("--weight-block-segment-rows", type=int)
    parser.add_argument("--gather-pipeline-rows", type=int)
    parser.add_argument(
        "--pipeline-depths",
        type=int,
        nargs=7,
        metavar="D",
        help=(
            "PipelineDepths: lhs depth, ahead, w1 depth, ahead, w2 depth,"
            " ahead, out depth"
        ),
    )
    parser.add_argument(
        "--tpu-inference-source",
        type=Path,
        default=baselines.DEFAULT_TPU_INFERENCE_SOURCE,
    )
    parser.add_argument(
        "--tokamax-source", type=Path, default=baselines.DEFAULT_TOKAMAX_SOURCE
    )
    parser.add_argument(
        "--torchtpu-source",
        type=Path,
        help=(
            "torchtpu: a vllm-torchtpu checkout at"
            f" {baselines.TORCHTPU_GIT_SHA}"
        ),
    )
    parser.add_argument("--output", help="write the rows as JSON here")
    args = parser.parse_args()
    if "flywheel" not in args.impls:
        parser.error("--impls must include flywheel, the reference")
    if "torchtpu" in args.impls and args.torchtpu_source is None:
        parser.error("--impls torchtpu needs --torchtpu-source")
    if not args.models and not args.shape:
        parser.error("give --models or --shape")
    return args


def run_cell(
    shape_name, shape, num_tokens, *, build_impls, baseline_names, device
):
    """Times flywheel, then each baseline that matches its output, on one
    shape and token count; prints the table row and returns it."""
    hidden, intermediate, experts, top_k = shape
    x, logits, w1, w2 = make_inputs(num_tokens, hidden, intermediate, experts)
    impls = build_impls(w1, w2, top_k)
    fn, weights = impls.pop("flywheel")
    timing, ref = time_call(lambda s, fn=fn: (fn(*s), s), (x, logits, *weights))
    row = dict(
        device=device,
        shape=shape_name,
        hidden=hidden,
        intermediate=intermediate,
        experts=experts,
        top_k=top_k,
        tokens=num_tokens,
        flywheel_ms=timing["median_ms"],
    )
    cells = []
    # Note (david): the header's column order, not make_impls' insertion
    # order.
    for name in baseline_names:
        fn, weights = impls[name]
        _, why = run_baseline(fn, (x, logits, *weights), ref)
        if why is not None:
            row[f"{name}_error"] = why
            cells.append(why.split(":")[0])
            continue
        timing, _ = time_call(
            lambda s, fn=fn: (fn(*s), s), (x, logits, *weights)
        )
        row[f"{name}_ms"] = timing["median_ms"]
        speedup = timing["median_ms"] / row["flywheel_ms"]
        cells.append(f"{timing['median_ms']:.3f} ({speedup:.2f}x)")
    print(
        f"{shape_name:>20} {num_tokens:>6} | {row['flywheel_ms']:>11.3f}"
        + "".join(f" | {cell:>26}" for cell in cells),
        flush=True,
    )
    for name in baseline_names:
        if f"{name}_error" in row:
            print(f"    {name}: {row[f'{name}_error']}", flush=True)
    return row


def main():
    args = parse_args()
    if "torchtpu" in args.impls:
        baselines.set_torchtpu_libtpu_flags()
    if jax.default_backend() != "tpu":
        sys.exit(f"needs a TPU, the default backend is {jax.default_backend()}")
    if "gmm" in args.impls:
        baseline_kernels = baselines.load_baselines(
            args.tpu_inference_source, args.tokamax_source
        )
    else:
        baseline_kernels = None
    if "torchtpu" in args.impls:
        torchtpu = baselines.load_torchtpu_moe(args.torchtpu_source)
    else:
        torchtpu = None
    if args.pipeline_depths:
        pipeline_depths = PipelineDepths(*args.pipeline_depths)
    else:
        pipeline_depths = None
    knobs = dict(
        tail_bucket=args.tail_bucket,
        tile_m=args.tile_m,
        weight_block_columns=args.weight_block_columns,
        weight_block_segment_rows=args.weight_block_segment_rows,
        gather_pipeline_rows=args.gather_pipeline_rows,
        pipeline_depths=pipeline_depths,
    )
    shapes = [(name, MODELS[name]) for name in args.models or ()]
    shapes += [
        (f"h{h}-i{i}-e{e}-k{k}", (h, i, e, k))
        for h, i, e, k in args.shape or ()
    ]
    device = jax.devices()[0].device_kind
    print(f"{device}, fused_moe {knobs}")
    baseline_names = [name for name in args.impls if name != "flywheel"]
    print(
        f"{'shape':>20} {'tokens':>6} | {'flywheel ms':>11}"
        + "".join(f" | {name + ' ms (speedup)':>26}" for name in baseline_names)
    )

    build_impls = functools.partial(
        make_impls,
        args.impls,
        knobs=knobs,
        baseline_kernels=baseline_kernels,
        torchtpu=torchtpu,
    )
    rows = [
        run_cell(
            shape_name,
            shape,
            num_tokens,
            build_impls=build_impls,
            baseline_names=baseline_names,
            device=device,
        )
        for (shape_name, shape), num_tokens in itertools.product(
            shapes, args.tokens
        )
    ]
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(dict(knobs=str(knobs), rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
