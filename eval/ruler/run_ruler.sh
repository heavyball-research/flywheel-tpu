#!/usr/bin/env bash
# RULER accuracy + throughput run for one (backend, sequence length) pair.
#
# RULER sizes its prompts at run time from a tokenizer, so one sequence length
# means one engine: max_model_len has to bound that length, and the length
# ladder is what the benchmark reports. Run this once per point.
#
# The engine runs one sequence at a time with chunked prefill and prefix
# caching off, so each prompt takes the dense forward path and the reported
# tokens/s is a single-sequence prefill rate rather than a batched aggregate.
#
# Usage:
#   eval/ruler/run_ruler.sh --backend flywheel --seqlen 4096 \
#     --tasks ruler --limit 500 --out ~/ruler-logs/flywheel_4096
#   eval/ruler/run_ruler.sh --backend baseline --seqlen 4096 \
#     --rpa-mixed-blocks eval/configs/qwen38_27b_tp8_rpa_mixed_blocks.json \
#     --tasks ruler --limit 500 --out ~/ruler-logs/baseline_4096
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

MODEL=Qwen/Qwen3.8-27B
REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
BACKEND=flywheel
SEQLEN=4096
TASKS=ruler
LIMIT=""
TP_SIZE=8
CP_SIZE=1
ENABLE_SP=0
MAX_NUM_SEQS=1
GPU_MEMORY_UTILIZATION=0.9
MAX_GEN_TOKS=128
RPA_MIXED_BLOCKS=""
RPA_MIXED_KEY=""
YARN_FACTOR=""
MULTIHOST=0
OUT=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --backend) BACKEND="$2"; shift ;;
        --seqlen) SEQLEN="$2"; shift ;;
        --tasks) TASKS="$2"; shift ;;
        --limit) LIMIT="$2"; shift ;;
        --tp-size) TP_SIZE="$2"; shift ;;
        --cp-size) CP_SIZE="$2"; shift ;;
        --enable-sp) ENABLE_SP=1 ;;
        --max-num-seqs) MAX_NUM_SEQS="$2"; shift ;;
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift ;;
        --max-gen-toks) MAX_GEN_TOKS="$2"; shift ;;
        --rpa-mixed-blocks) RPA_MIXED_BLOCKS="$2"; shift ;;
        --rpa-mixed-key) RPA_MIXED_KEY="$2"; shift ;;
        --yarn-factor) YARN_FACTOR="$2"; shift ;;
        --multihost) MULTIHOST=1 ;;
        --out) OUT="$2"; shift ;;
        *) echo "Unknown parameter: $1" >&2; exit 1 ;;
    esac
    shift
done

if [[ -z "$OUT" ]]; then
    echo "Error: --out is required" >&2
    exit 1
fi
case "$BACKEND" in
    baseline) KERNELS=0; LINEAR=0 ;;
    flywheel) KERNELS=1; LINEAR=1 ;;
    *) echo "Error: --backend must be baseline or flywheel" >&2; exit 1 ;;
esac
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "Error: activate the eval venv first (see eval/setup.sh)" >&2
    exit 1
fi

# The cache holds the model by revision and has no refs/main, so an offline
# `Qwen/Qwen3.8-27B` does not resolve. RULER's own tokenizer lookup takes no
# revision, so hand it the snapshot directory instead.
SNAPSHOT=$("$VIRTUAL_ENV/bin/python" -c "
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2], local_files_only=True))
" "$MODEL" "$REVISION")

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

export USE_FLYWHEEL_TPU_KERNEL="$KERNELS"
export USE_FLYWHEEL_TPU_LINEAR_KERNEL="$LINEAR"
export DISABLE_WEIGHT_REQUANTIZATION=1
export ENABLE_QUANTIZED_MATMUL_KERNEL=0
export USE_BATCHED_RPA_KERNEL=0
export USE_BATCHED_RPA_SEQ_ON_LANE=0
export MODEL_IMPL_TYPE=vllm
if (( CP_SIZE > 1 )); then
    export NEW_MODEL_DESIGN=1
fi
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# Lazy compilation: the precompiled ladder runs past max_num_batched_tokens
# and builds graphs no RULER prompt reaches, including the prompt-logprobs ones
# generate_until never calls. ruler_eval.py warms the shapes it does use before
# it starts timing.
export SKIP_JAX_PRECOMPILE=1
export VLLM_TPU_BUCKET_PADDING_GAP=16384
export VLLM_XLA_CACHE_PATH="$HOME/ruler-cache"
export JAX_COMPILATION_CACHE_DIR="$HOME/ruler-cache/jax"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_XLA_CHECK_RECOMPILATION=0

# The tuned RPA v3 MIXED block sizes are keyed by the (seqlen, batch) cell the
# tuner swept; the engine here runs one sequence at a time. --rpa-mixed-key
# names the cell instead, for a length the table does not reach.
if [[ -n "$RPA_MIXED_BLOCKS" ]]; then
    if [[ "$BACKEND" != "baseline" ]]; then
        echo "Error: --rpa-mixed-blocks only applies to --backend baseline" >&2
        exit 1
    fi
    RPA_MIXED_KEY="${RPA_MIXED_KEY:-s${SEQLEN}_b${MAX_NUM_SEQS}}"
    RPA_V3_MIXED_BLOCK_SIZES=$("$VIRTUAL_ENV/bin/python" -c "
import json, sys
table = json.load(open(sys.argv[1]))
key = sys.argv[2]
if key not in table:
    raise SystemExit('%s has no %s entry' % (sys.argv[1], key))
print(','.join(str(size) for size in table[key]))
" "$RPA_MIXED_BLOCKS" "$RPA_MIXED_KEY")
    export RPA_V3_MIXED_BLOCK_SIZES
elif [[ -n "$RPA_MIXED_KEY" ]]; then
    echo "Error: --rpa-mixed-key needs --rpa-mixed-blocks" >&2
    exit 1
fi

# Multi-host: the engine drives one Ray actor per TPU host of the slice, and
# the Ray cluster has to be up already. The Ray executor (v2) fills each
# actor's environment from this driver's with setdefault, so the kernel
# switches above reach the remote hosts only if their Ray nodes were started
# without them. The driver ships nothing to the actors, so every host needs
# the same venv and checkout.
if (( MULTIHOST )); then
    export TPU_MULTIHOST_BACKEND=ray
    # Every step is an RPC to the actors, and the first step at a new length
    # includes its compile: a 512k prefill outlived the 300 s default.
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
    "$VIRTUAL_ENV/bin/ray" status
fi

mkdir -p "$OUT"
exec > >(tee -a "$OUT/driver.log") 2>&1

# Killing this script alone would orphan the driver, which still holds the
# TPU while the flock it was serialised by is already released. Walk the tree
# instead of trusting a process group: whether `setsid` forks depends on the
# shell's job-control state, and if it does the pid here would be a
# short-lived middleman and `wait` would return early.
child_pid=""
kill_tree() {
    local pid=$1 kid
    for kid in $(pgrep -P "$pid" 2>/dev/null); do
        kill_tree "$kid"
    done
    kill "$pid" 2>/dev/null || true
}
cleanup() {
    status=$?
    if [[ -n "$child_pid" ]]; then
        kill_tree "$child_pid"
    fi
    printf "%s\n" "$status" > "$OUT/exit_code"
    exit "$status"
}
trap cleanup EXIT INT TERM

# One TPU job at a time on this host.
exec 9> "$HOME/tpu.lock"
flock -n 9

eval_args=(
    --model "$MODEL"
    --revision "$REVISION"
    --tokenizer-dir "$SNAPSHOT"
    --seqlen "$SEQLEN"
    --tasks "$TASKS"
    --tp-size "$TP_SIZE"
    --cp-size "$CP_SIZE"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-gen-toks "$MAX_GEN_TOKS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --out "$OUT"
)
if [[ -n "$LIMIT" ]]; then
    eval_args+=(--limit "$LIMIT")
fi
if (( ENABLE_SP )); then
    eval_args+=(--enable-sp)
fi
if [[ -n "$YARN_FACTOR" ]]; then
    eval_args+=(--yarn-factor "$YARN_FACTOR")
fi

printf 'RUN_START %s backend=%s seqlen=%d tasks=%s limit=%s rpa=%s tp=%s cp=%s sp=%s yarn=%s multihost=%s\n' \
    "$(date -u +%FT%TZ)" "$BACKEND" "$SEQLEN" "$TASKS" "${LIMIT:-all}" \
    "${RPA_V3_MIXED_BLOCK_SIZES:-default}${RPA_MIXED_KEY:+ ($RPA_MIXED_KEY)}" \
    "$TP_SIZE" "$CP_SIZE" "$ENABLE_SP" "${YARN_FACTOR:-none}" "$MULTIHOST"
"$VIRTUAL_ENV/bin/python" "$script_dir/ruler_eval.py" "${eval_args[@]}" &
child_pid=$!
wait "$child_pid"
printf 'RUN_DONE %s\n' "$(date -u +%FT%TZ)"
