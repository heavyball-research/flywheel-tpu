#!/usr/bin/env bash
# Time attention over a paged KV cache in two serving scenarios, chunked
# prefill and decode, for flywheel and RPA v3, with a profiler trace per cell.
#
# Every cell appends to one JSONL. A cell that fails (out of VMEM, or a cache
# larger than HBM) is recorded as an error and the sweep goes on.
#
# Usage:
#     benchmarks/softmax_attention/kvcache_scenarios.sh \
#         results/benchmark/kvcache_scenarios
#
# Environment:
#     PYTHON          interpreter (default: .venv/bin/python)
#     IMPLS           default "flywheel rpa"
#     HEADS_K         default "4 32"
#     DECODE_BATCHES  default "256 128"
#     RPA_K32_BLOCKS  RPA blocks at 32 KV heads (default 128,256,128,256);
#                     empty runs RPA's own formula

set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.

PYTHON=${PYTHON:-.venv/bin/python}
OUT=${1:-results/benchmark/kvcache_scenarios/$(date +%Y%m%d_%H%M%S)}
IMPLS=${IMPLS:-"flywheel rpa"}
HEADS_K=${HEADS_K:-"4 32"}
DECODE_BATCHES=${DECODE_BATCHES:-"256 128"}
# At 32 KV heads of 256, RPA's default blocks run out of VMEM on v7x; this is
# the one config that fit its sweep (its v7x prefill entry). A decode call
# compiles RPA's mixed kernel as well, so it needs these blocks there too.
RPA_K32_BLOCKS=${RPA_K32_BLOCKS-"128,256,128,256"}
mkdir -p "$OUT"

note() { echo "$*" | tee -a "$OUT/benchmark.log"; }

run() {
  $PYTHON benchmarks/softmax_attention/kvcache_scenarios.py "$@" \
      --output "$OUT/benchmark.jsonl" --trace "$OUT/traces" \
      >> "$OUT/benchmark.log" 2>&1 \
    || note "failed: $*"
}

note "output:     $OUT"
note "python:     $PYTHON"
note "grid:       impls [$IMPLS] heads_k [$HEADS_K]"
note "            decode batches [$DECODE_BATCHES]"
note "start $(date -u +%FT%TZ)"

for heads_k in $HEADS_K; do
  chunk_flags=() decode_flags=()
  if [[ $heads_k == 32 && -n $RPA_K32_BLOCKS ]]; then
    chunk_flags=(--rpa-blocks "$RPA_K32_BLOCKS")
    decode_flags=(--rpa-mixed-blocks "$RPA_K32_BLOCKS")
  fi
  for impl in $IMPLS; do
    rpa_flags=()
    [[ $impl == rpa ]] && rpa_flags=("${chunk_flags[@]}")
    run --impl "$impl" --scenario chunked_prefill --heads-k "$heads_k" \
        "${rpa_flags[@]}"
  done
  for batch in $DECODE_BATCHES; do
    for impl in $IMPLS; do
      rpa_flags=()
      [[ $impl == rpa ]] && rpa_flags=("${decode_flags[@]}")
      run --impl "$impl" --scenario decode --heads-k "$heads_k" \
          --batch "$batch" "${rpa_flags[@]}"
    done
  done
done

note "ALL DONE $(date -u +%FT%TZ)"
