#!/usr/bin/env bash
# Time attention over a paged KV cache in serving scenarios, chunked prefill,
# decode and speculative verify, for flywheel, RPA v3 and vllm-torchtpu's
# batched RPA, and MLA chunked prefill, decode and verify for flywheel_mla and
# MLA v2, with a profiler trace per cell.
#
# MLA decode and verify: for each MLA_HEADS x MLA_KV_DTYPES, the core cell
# decodes MLA_CORE_BATCH sequences of MLA_CORE_SEQ tokens of equal length, and
# each sweep moves one setting of it:
#
#   batch    MLA_DECODE_BATCHES sequences
#   seqlen   MLA_DECODE_SEQS tokens per sequence
#   varlen   MLA_PATTERNS: the core's tokens split unevenly across its
#            sequences
#   verify   MLA_VERIFY_TOKENS query tokens per sequence per step (MTP's
#            K + 1); 1 runs the decode path under verify's accounting
#
# MLA_TUNED adds the fp8 shapes MLA v2's tuned table holds, where it runs its
# tuned blocks instead of its defaults (each record's block_source says which,
# for either impl).
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
#     IMPLS           default "flywheel rpa", empty for none; batched_rpa needs
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
#     MLA_IMPLS       MLA impls, e.g. "flywheel_mla mla_v2" (default: none),
#                     run on the chunked_prefill, decode and verify of
#                     SCENARIOS
#     MLA_HEADS       default "16 128"
#     MLA_KV_DTYPES   default "bfloat16 float8_e4m3fn"
#     MLA_SEQS        chunked prefill's tokens (default "16384 65536")
#     MLA_CORE_BATCH  default 64
#     MLA_CORE_SEQ    default 4096
#     MLA_DECODE_BATCHES  default "1 8 32 64 128 256"; empty skips the sweep,
#                     as for the four below
#     MLA_DECODE_SEQS default "1024 4096 16384 65536 131072"
#     MLA_PATTERNS    default "spiky longtail"
#     MLA_VERIFY_TOKENS  default "1 2 3 4"
#     MLA_TUNED       fp8 cells as heads:batch:seq:verify_tokens, decode at
#                     verify_tokens 1 (default "128:8:3072:1 64:64:9216:1
#                     64:64:9216:2")
#     MLA_DECODE_STEPS  decode steps per timed call (default 64); verify
#                     runs VERIFY_STEPS

set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.

PYTHON=${PYTHON:-.venv/bin/python}
OUT=${1:-results/benchmark/kvcache_scenarios/$(date +%Y%m%d_%H%M%S)}
IMPLS=${IMPLS-"flywheel rpa"}
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
MLA_IMPLS=${MLA_IMPLS:-}
MLA_HEADS=${MLA_HEADS:-"16 128"}
MLA_KV_DTYPES=${MLA_KV_DTYPES:-"bfloat16 float8_e4m3fn"}
MLA_SEQS=${MLA_SEQS:-"16384 65536"}
MLA_CORE_BATCH=${MLA_CORE_BATCH:-64}
MLA_CORE_SEQ=${MLA_CORE_SEQ:-4096}
MLA_DECODE_BATCHES=${MLA_DECODE_BATCHES-"1 8 32 64 128 256"}
MLA_DECODE_SEQS=${MLA_DECODE_SEQS-"1024 4096 16384 65536 131072"}
MLA_PATTERNS=${MLA_PATTERNS-"spiky longtail"}
MLA_VERIFY_TOKENS=${MLA_VERIFY_TOKENS-"1 2 3 4"}
MLA_TUNED=${MLA_TUNED-"128:8:3072:1 64:64:9216:1 64:64:9216:2"}
MLA_DECODE_STEPS=${MLA_DECODE_STEPS:-64}
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
if [[ -n $MLA_IMPLS ]]; then
  note "            mla impls [$MLA_IMPLS] heads [$MLA_HEADS]"
  note "            mla kv dtypes [$MLA_KV_DTYPES] seqs [$MLA_SEQS]"
  note "            mla core batch $MLA_CORE_BATCH seq $MLA_CORE_SEQ"
  note "            mla decode batches [$MLA_DECODE_BATCHES]" \
       "seqs [$MLA_DECODE_SEQS] patterns [$MLA_PATTERNS]"
  note "            mla verify tokens [$MLA_VERIFY_TOKENS] tuned [$MLA_TUNED]"
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

if has_scenario chunked_prefill; then
  for seq in $MLA_SEQS; do
    for heads in $MLA_HEADS; do
      for kv_dtype in $MLA_KV_DTYPES; do
        for impl in $MLA_IMPLS; do
          run --impl "$impl" --scenario chunked_prefill --heads "$heads" \
              --kv-dtype "$kv_dtype" --seq "$seq"
        done
      done
    done
  done
fi

# One MLA decode or verify cell on every MLA impl: the scenario, then the
# cell's flags. decode runs MLA_DECODE_STEPS steps; verify takes its query
# tokens per step first and runs VERIFY_STEPS steps.
run_mla() {
  local scenario=$1 impl
  shift
  local flags=(--scenario "$scenario" --decode-tokens "$MLA_DECODE_STEPS")
  if [[ $scenario == verify ]]; then
    flags=(--scenario verify --decode-tokens "$VERIFY_STEPS"
           --verify-tokens "$1")
    shift
  fi
  for impl in $MLA_IMPLS; do
    run --impl "$impl" "${flags[@]}" "$@"
  done
}

for heads in $MLA_HEADS; do
  for kv_dtype in $MLA_KV_DTYPES; do
    shape=(--heads "$heads" --kv-dtype "$kv_dtype")
    core=(--batch "$MLA_CORE_BATCH" --seq "$MLA_CORE_SEQ")
    if has_scenario decode; then
      for batch in $MLA_DECODE_BATCHES; do
        run_mla decode "${shape[@]}" --batch "$batch" --seq "$MLA_CORE_SEQ"
      done
      for seq in $MLA_DECODE_SEQS; do
        # The batch sweep has run the core cell.
        [[ $seq == "$MLA_CORE_SEQ" \
           && " $MLA_DECODE_BATCHES " == *" $MLA_CORE_BATCH "* ]] && continue
        run_mla decode "${shape[@]}" --batch "$MLA_CORE_BATCH" --seq "$seq"
      done
      for pattern in $MLA_PATTERNS; do
        run_mla decode "${shape[@]}" "${core[@]}" --pattern "$pattern"
      done
    fi
    if has_scenario verify; then
      for tokens in $MLA_VERIFY_TOKENS; do
        run_mla verify "$tokens" "${shape[@]}" "${core[@]}"
      done
    fi
  done
done
for spec in $MLA_TUNED; do
  IFS=: read -r heads batch seq tokens <<< "$spec"
  cell=(--heads "$heads" --kv-dtype float8_e4m3fn --batch "$batch"
        --seq "$seq")
  if [[ $tokens == 1 ]]; then
    has_scenario decode && run_mla decode "${cell[@]}"
  else
    has_scenario verify && run_mla verify "$tokens" "${cell[@]}"
  fi
done

note "ALL DONE $(date -u +%FT%TZ)"
