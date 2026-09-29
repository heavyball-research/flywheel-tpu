#!/usr/bin/env bash
# Time the whole attention block, hidden states in and out, for every backend,
# with a profiler trace per cell, then print the kernel-level and block-level
# tables.
#
# Every cell appends to one JSONL; report.py keeps the last ok record of each
# cell, so rerunning into the same directory replaces them. A cell that fails
# is logged and the sweep goes on.
#
# Usage:
#     benchmarks/softmax_attention/attention_block.sh \
#         results/benchmark/softmax_attention
#
# Environment:
#     PYTHON        interpreter (default: .venv/bin/python)
#     IMPLS         default "rpa splash flywheel"
#     HEADS         default 32
#     HEADS_K       default "32 4"
#     HEAD_DIM      default 256
#     MASK          default causal
#     SEQLENS       default "1024 ... 131072"

set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.

PYTHON=${PYTHON:-.venv/bin/python}
OUT=${1:-results/benchmark/softmax_attention/$(date +%Y%m%d_%H%M%S)}
IMPLS=${IMPLS:-"rpa splash flywheel"}
HEADS=${HEADS:-32}
HEADS_K=${HEADS_K:-"32 4"}
HEAD_DIM=${HEAD_DIM:-256}
MASK=${MASK:-causal}
SEQLENS=${SEQLENS:-"1024 2048 4096 8192 16384 32768 65536 131072"}
mkdir -p "$OUT"

note() { echo "$*" | tee -a "$OUT/benchmark.log"; }

note "output:     $OUT"
note "python:     $PYTHON"
note "grid:       impls [$IMPLS] heads $HEADS heads_k [$HEADS_K]"
note "            head_dim $HEAD_DIM mask $MASK seqlens [$SEQLENS]"
note "start $(date -u +%FT%TZ)"

for heads_k in $HEADS_K; do
  for seq in $SEQLENS; do
    for impl in $IMPLS; do
      $PYTHON benchmarks/softmax_attention/attention_block.py \
          --impl "$impl" --heads "$HEADS" --heads-k "$heads_k" \
          --head-dim "$HEAD_DIM" --mask "$MASK" --seq "$seq" \
          --output "$OUT/benchmark.jsonl" --trace "$OUT/traces" \
          >> "$OUT/benchmark.log" 2>&1 \
        || note "failed: $impl h$HEADS-k$heads_k s$seq"
    done
  done
done

note "ALL DONE $(date -u +%FT%TZ)"
$PYTHON benchmarks/softmax_attention/report.py "$OUT/benchmark.jsonl" | tee -a "$OUT/benchmark.log"
