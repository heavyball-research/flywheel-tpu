#!/usr/bin/env bash
# Time attention over a paged KV cache in serving scenarios, chunked prefill,
# decode and speculative verify, for flywheel, RPA v3 and vllm-torchtpu's
# batched RPA, MLA chunked prefill for flywheel_mla and MLA v2, and DeepSeek
# sparse attention (DSA) decode and chunked prefill for flywheel_dsa and
# vllm-torchtpu's DSA, with a profiler trace per cell.
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
#     MLA_IMPLS       MLA chunked prefill impls, e.g. "flywheel_mla mla_v2"
#                     (default: none)
#     MLA_HEADS       default "16 128"
#     MLA_KV_DTYPES   default "bfloat16 float8_e4m3fn"
#     MLA_SEQS        default "16384 65536"
#     DSA_IMPLS       DSA layer impls, e.g. "flywheel_dsa torchtpu_dsa
#                     torchtpu_dsa_masked_dense" (default: none); the torchtpu
#                     ones need TORCHTPU_SOURCE, and torchtpu_dsa_masked_dense
#                     runs glm52 only
#     TORCHTPU_PYTHON interpreter of the torchtpu DSA impls (default: PYTHON)
#     DSA_GEOMETRIES  default "v32 glm52"
#     DSA_DECODE_SEQS default "8192 32768 131072"
#     DSA_DECODE_BATCHES default "1 32 128"
#     DSA_DECODE_STEPS decode steps per timed call (default 16)
#     DSA_PREFILL_SEQS default "32768 131072"
#     DSA_CHUNK       chunked prefill tokens per call (default 4096)

set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.

PYTHON=${PYTHON:-.venv/bin/python}
OUT=${1:-results/benchmark/kvcache_scenarios/$(date +%Y%m%d_%H%M%S)}
IMPLS=${IMPLS-"flywheel rpa"}
TORCHTPU_SOURCE=${TORCHTPU_SOURCE:-}
DSA_IMPLS=${DSA_IMPLS:-}
if [[ " $IMPLS " == *" batched_rpa "* && -z $TORCHTPU_SOURCE ]]; then
  echo "IMPLS has batched_rpa: set TORCHTPU_SOURCE to a vllm-torchtpu" \
       "checkout." >&2
  exit 1
elif [[ " $DSA_IMPLS " == *" torchtpu_dsa"* && -z $TORCHTPU_SOURCE ]]; then
  echo "DSA_IMPLS has a torchtpu impl: set TORCHTPU_SOURCE to a" \
       "vllm-torchtpu checkout." >&2
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
TORCHTPU_PYTHON=${TORCHTPU_PYTHON:-$PYTHON}
DSA_GEOMETRIES=${DSA_GEOMETRIES:-"v32 glm52"}
DSA_DECODE_SEQS=${DSA_DECODE_SEQS:-"8192 32768 131072"}
DSA_DECODE_BATCHES=${DSA_DECODE_BATCHES:-"1 32 128"}
DSA_DECODE_STEPS=${DSA_DECODE_STEPS:-16}
DSA_PREFILL_SEQS=${DSA_PREFILL_SEQS:-"32768 131072"}
DSA_CHUNK=${DSA_CHUNK:-4096}
mkdir -p "$OUT"

has_scenario() { [[ " $SCENARIOS " == *" $1 "* ]]; }

note() { echo "$*" | tee -a "$OUT/benchmark.log"; }

# Masked-dense routing admits glm52 only.
dsa_admits() { [[ $1 != torchtpu_dsa_masked_dense || $2 == glm52 ]]; }

run() {
  local source_flags=() python=$PYTHON
  [[ -n $TORCHTPU_SOURCE ]] \
    && source_flags=(--torchtpu-source "$TORCHTPU_SOURCE")
  [[ " $* " == *" --impl torchtpu_dsa"* ]] && python=$TORCHTPU_PYTHON
  $python benchmarks/softmax_attention/kvcache_scenarios.py "$@" \
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
fi
if [[ -n $DSA_IMPLS ]]; then
  note "            dsa impls [$DSA_IMPLS] geometries [$DSA_GEOMETRIES]"
  note "            dsa decode seqs [$DSA_DECODE_SEQS] batches" \
       "[$DSA_DECODE_BATCHES] steps $DSA_DECODE_STEPS"
  note "            dsa prefill seqs [$DSA_PREFILL_SEQS] chunk $DSA_CHUNK"
  note "            torchtpu python: $TORCHTPU_PYTHON"
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

for geometry in $DSA_GEOMETRIES; do
  has_scenario chunked_prefill || break
  for seq in $DSA_PREFILL_SEQS; do
    for impl in $DSA_IMPLS; do
      dsa_admits "$impl" "$geometry" || continue
      run --impl "$impl" --scenario chunked_prefill \
          --dsa-geometry "$geometry" --seq "$seq" --chunk "$DSA_CHUNK"
    done
  done
done

for geometry in $DSA_GEOMETRIES; do
  has_scenario decode || break
  for seq in $DSA_DECODE_SEQS; do
    for batch in $DSA_DECODE_BATCHES; do
      for impl in $DSA_IMPLS; do
        dsa_admits "$impl" "$geometry" || continue
        run --impl "$impl" --scenario decode --dsa-geometry "$geometry" \
            --seq "$seq" --batch "$batch" --decode-tokens "$DSA_DECODE_STEPS"
      done
    done
  done
done

note "ALL DONE $(date -u +%FT%TZ)"
