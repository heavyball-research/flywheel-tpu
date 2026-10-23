#!/usr/bin/env bash
# Time attention over a paged KV cache in serving scenarios, chunked prefill,
# decode and speculative verify, for flywheel, RPA v3 and vllm-torchtpu's
# batched RPA, with a profiler trace per cell.
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
#     IMPLS           default "flywheel rpa"; batched_rpa needs
#                     TORCHTPU_SOURCE
#     TORCHTPU_SOURCE a vllm-torchtpu checkout for batched_rpa
#     HEADS_K         default "4 32"
#     DECODE_BATCHES  default "256 128"; empty skips decode
#     RPA_K32_BLOCKS  RPA blocks at 32 KV heads (default 128,256,128,256);
#                     empty runs RPA's own formula
#     SCENARIOS       default "chunked_prefill decode"; add verify for the
#                     verify sweep below
#     VERIFY_HEADS    "heads heads_k head_dim" (default "32 8 128", Qwen3-4B)
#     VERIFY_BATCHES  default "32 128"
#     VERIFY_SEQS     kv tokens per sequence (default "2048 8192")
#     VERIFY_TOKENS   query tokens per sequence per step (default "1 4 8 16");
#                     1 runs each impl's decode path
#     VERIFY_STEPS    verify steps per timed call (default 8)

set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.

PYTHON=${PYTHON:-.venv/bin/python}
OUT=${1:-results/benchmark/kvcache_scenarios/$(date +%Y%m%d_%H%M%S)}
IMPLS=${IMPLS:-"flywheel rpa"}
TORCHTPU_SOURCE=${TORCHTPU_SOURCE:-}
if [[ " $IMPLS " == *" batched_rpa "* && -z $TORCHTPU_SOURCE ]]; then
  echo "IMPLS has batched_rpa: set TORCHTPU_SOURCE to a vllm-torchtpu" \
       "checkout." >&2
  exit 1
fi
HEADS_K=${HEADS_K:-"4 32"}
DECODE_BATCHES=${DECODE_BATCHES-"256 128"}
# At 32 KV heads of 256, RPA's default blocks run out of VMEM on v7x; this is
# the one config that fit its sweep (its v7x prefill entry). A decode call
# compiles RPA's mixed kernel as well, so it needs these blocks there too.
RPA_K32_BLOCKS=${RPA_K32_BLOCKS-"128,256,128,256"}
SCENARIOS=${SCENARIOS:-"chunked_prefill decode"}
VERIFY_HEADS=${VERIFY_HEADS:-"32 8 128"}
VERIFY_BATCHES=${VERIFY_BATCHES:-"32 128"}
VERIFY_SEQS=${VERIFY_SEQS:-"2048 8192"}
VERIFY_TOKENS=${VERIFY_TOKENS:-"1 4 8 16"}
VERIFY_STEPS=${VERIFY_STEPS:-8}
mkdir -p "$OUT"

has_scenario() { [[ " $SCENARIOS " == *" $1 "* ]]; }

note() { echo "$*" | tee -a "$OUT/benchmark.log"; }

run() {
  local source_flags=()
  [[ -n $TORCHTPU_SOURCE ]] \
    && source_flags=(--torchtpu-source "$TORCHTPU_SOURCE")
  $PYTHON benchmarks/softmax_attention/kvcache_scenarios.py "$@" \
      "${source_flags[@]}" \
      --output "$OUT/benchmark.jsonl" --trace "$OUT/traces" \
      >> "$OUT/benchmark.log" 2>&1 \
    || note "failed: $*"
}

note "output:     $OUT"
note "python:     $PYTHON"
note "grid:       scenarios [$SCENARIOS]"
note "            impls [$IMPLS] heads_k [$HEADS_K]"
note "            decode batches [$DECODE_BATCHES]"
if has_scenario verify; then
  note "            verify heads [$VERIFY_HEADS]"
  note "            verify batches [$VERIFY_BATCHES] seqs [$VERIFY_SEQS]"
  note "            verify tokens [$VERIFY_TOKENS] steps $VERIFY_STEPS"
fi
note "start $(date -u +%FT%TZ)"

for heads_k in $HEADS_K; do
  has_scenario chunked_prefill || has_scenario decode || break
  chunk_flags=() decode_flags=()
  if [[ $heads_k == 32 && -n $RPA_K32_BLOCKS ]]; then
    chunk_flags=(--rpa-blocks "$RPA_K32_BLOCKS")
    decode_flags=(--rpa-mixed-blocks "$RPA_K32_BLOCKS")
  fi
  for impl in $IMPLS; do
    has_scenario chunked_prefill || break
    rpa_flags=()
    [[ $impl == rpa ]] && rpa_flags=("${chunk_flags[@]}")
    run --impl "$impl" --scenario chunked_prefill --heads-k "$heads_k" \
        "${rpa_flags[@]}"
  done
  for batch in $DECODE_BATCHES; do
    has_scenario decode || break
    for impl in $IMPLS; do
      rpa_flags=()
      [[ $impl == rpa ]] && rpa_flags=("${decode_flags[@]}")
      run --impl "$impl" --scenario decode --heads-k "$heads_k" \
          --batch "$batch" "${rpa_flags[@]}"
    done
  done
done

if has_scenario verify; then
  read -r verify_heads verify_heads_k verify_head_dim <<< "$VERIFY_HEADS"
  for seq in $VERIFY_SEQS; do
    for batch in $VERIFY_BATCHES; do
      for tokens in $VERIFY_TOKENS; do
        for impl in $IMPLS; do
          run --impl "$impl" --scenario verify --heads "$verify_heads" \
              --heads-k "$verify_heads_k" --head-dim "$verify_head_dim" \
              --seq "$seq" --batch "$batch" --verify-tokens "$tokens" \
              --decode-tokens "$VERIFY_STEPS"
        done
      done
    done
  done
fi

note "ALL DONE $(date -u +%FT%TZ)"
