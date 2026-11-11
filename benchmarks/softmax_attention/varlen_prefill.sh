#!/usr/bin/env bash
# Time ragged prefill batches over a paged KV cache for flywheel, RPA v3 and
# vllm-torchtpu's batched RPA, MLA's for flywheel_mla and MLA v2, and DeepSeek
# sparse attention (DSA) layer steps for flywheel_dsa and vllm-torchtpu's DSA,
# with a profiler trace per cell.
#
# A token budget is split across a batch of sequences by a pattern's weights
# (uniform, spiky, longtail; see varlen_prefill.py), so every pattern at the
# same (budget, batch) packs the same tokens and differs only in shape. The
# impls of one cell run back to back.
#
# Every cell appends to one JSONL. A cell that fails (out of VMEM or HBM) is
# recorded as an error and the sweep goes on.
#
# Usage:
#     benchmarks/softmax_attention/varlen_prefill.sh \
#         results/benchmark/varlen_prefill
#
# Environment:
#     PYTHON          interpreter (default: .venv/bin/python)
#     IMPLS           default "flywheel rpa", empty for none; batched_rpa needs
#                     TORCHTPU_SOURCE
#     TORCHTPU_SOURCE a vllm-torchtpu checkout for batched_rpa
#     PATTERNS        default "uniform spiky longtail"
#     HEADS           default 32
#     HEADS_K         default 8
#     HEAD_DIMS       default "128 256"
#     BUDGETS         default "131072 65536 32768 16384"
#     BATCHES         default "8 16 32 64 128"
#     MLA_IMPLS       MLA impls, e.g. "flywheel_mla mla_v2" (default: none)
#     MLA_HEADS       default "16 128"
#     MLA_KV_DTYPES   default "bfloat16 float8_e4m3fn"
#     MLA_BUDGETS     default "32768 16384"
#     DSA_IMPLS       DSA layer impls, e.g. "flywheel_dsa torchtpu_dsa
#                     torchtpu_dsa_masked_dense" (default: none); the torchtpu
#                     ones need TORCHTPU_SOURCE, and torchtpu_dsa_masked_dense
#                     runs glm52 only
#     TORCHTPU_PYTHON interpreter of the torchtpu DSA impls (default: PYTHON)
#     DSA_GEOMETRIES  default "v32 glm52"
#     DSA_BUDGETS     default "32768 131072"
#     DSA_BATCHES     default 8

set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.

PYTHON=${PYTHON:-.venv/bin/python}
OUT=${1:-results/benchmark/varlen_prefill/$(date +%Y%m%d_%H%M%S)}
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
PATTERNS=${PATTERNS:-"uniform spiky longtail"}
HEADS=${HEADS:-32}
HEADS_K=${HEADS_K:-8}
HEAD_DIMS=${HEAD_DIMS:-"128 256"}
BUDGETS=${BUDGETS:-"131072 65536 32768 16384"}
BATCHES=${BATCHES:-"8 16 32 64 128"}
MLA_IMPLS=${MLA_IMPLS:-}
MLA_HEADS=${MLA_HEADS:-"16 128"}
MLA_KV_DTYPES=${MLA_KV_DTYPES:-"bfloat16 float8_e4m3fn"}
MLA_BUDGETS=${MLA_BUDGETS:-"32768 16384"}
TORCHTPU_PYTHON=${TORCHTPU_PYTHON:-$PYTHON}
DSA_GEOMETRIES=${DSA_GEOMETRIES:-"v32 glm52"}
DSA_BUDGETS=${DSA_BUDGETS:-"32768 131072"}
DSA_BATCHES=${DSA_BATCHES:-8}
mkdir -p "$OUT"

note() { echo "$*" | tee -a "$OUT/benchmark.log"; }

# Masked-dense routing admits glm52 only.
dsa_admits() { [[ $1 != torchtpu_dsa_masked_dense || $2 == glm52 ]]; }

run() {
  local source_flags=() python=$PYTHON
  [[ -n $TORCHTPU_SOURCE ]] \
    && source_flags=(--torchtpu-source "$TORCHTPU_SOURCE")
  [[ " $* " == *" --impl torchtpu_dsa"* ]] && python=$TORCHTPU_PYTHON
  $python benchmarks/softmax_attention/varlen_prefill.py "$@" \
      "${source_flags[@]}" \
      --output "$OUT/benchmark.jsonl" --trace "$OUT/traces" \
      >> "$OUT/benchmark.log" 2>&1 \
    || note "failed: $*"
}

note "output:     $OUT"
note "python:     $PYTHON"
note "grid:       impls [$IMPLS] patterns [$PATTERNS]"
note "            heads $HEADS heads_k $HEADS_K head_dims [$HEAD_DIMS]"
note "            budgets [$BUDGETS] batches [$BATCHES]"
if [[ -n $MLA_IMPLS ]]; then
  note "            mla impls [$MLA_IMPLS] heads [$MLA_HEADS]"
  note "            mla kv dtypes [$MLA_KV_DTYPES] budgets [$MLA_BUDGETS]"
fi
if [[ -n $DSA_IMPLS ]]; then
  note "            dsa impls [$DSA_IMPLS] geometries [$DSA_GEOMETRIES]"
  note "            dsa budgets [$DSA_BUDGETS] batches [$DSA_BATCHES]"
  note "            torchtpu python: $TORCHTPU_PYTHON"
fi
note "start $(date -u +%FT%TZ)"

for head_dim in $HEAD_DIMS; do
  for pattern in $PATTERNS; do
    for budget in $BUDGETS; do
      for batch in $BATCHES; do
        for impl in $IMPLS; do
          run --impl "$impl" --heads "$HEADS" --heads-k "$HEADS_K" \
              --head-dim "$head_dim" --pattern "$pattern" \
              --batch "$batch" --budget "$budget"
        done
      done
    done
  done
done

for heads in $MLA_HEADS; do
  for kv_dtype in $MLA_KV_DTYPES; do
    for pattern in $PATTERNS; do
      for budget in $MLA_BUDGETS; do
        for batch in $BATCHES; do
          for impl in $MLA_IMPLS; do
            run --impl "$impl" --heads "$heads" --kv-dtype "$kv_dtype" \
                --pattern "$pattern" --batch "$batch" --budget "$budget"
          done
        done
      done
    done
  done
done

for geometry in $DSA_GEOMETRIES; do
  for pattern in $PATTERNS; do
    for budget in $DSA_BUDGETS; do
      for batch in $DSA_BATCHES; do
        for impl in $DSA_IMPLS; do
          dsa_admits "$impl" "$geometry" || continue
          run --impl "$impl" --dsa-geometry "$geometry" --pattern "$pattern" \
              --batch "$batch" --budget "$budget"
        done
      done
    done
  done
done

note "ALL DONE $(date -u +%FT%TZ)"
