#!/usr/bin/env bash
# BABILong accuracy + throughput run for one (backend, length) point.
#
# --multihost runs the engine across every host of the slice through a Ray
# cluster that is already up; run it on the Ray head.
#
# Usage:
#   eval/babilong/run_babilong.sh --backend flywheel --length 128k \
#     --out ~/babilong-logs/flywheel_128k
#   eval/babilong/run_babilong.sh --backend baseline --length 512k \
#     --cp-size 2 --enable-sp --yarn-factor 2 --multihost \
#     --rpa-mixed-blocks eval/configs/qwen38_27b_tp8_rpa_mixed_blocks.json \
#     --rpa-mixed-key s262144_b1 --out ~/babilong-logs/baseline_512k
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
BABILONG_COMMIT=7a6efee29f5cac03c3c410e6799c80fd2ffe3610
BACKEND=flywheel
LENGTH=""
TASKS=qa1,qa2,qa3,qa4,qa5
LIMIT=""
TP_SIZE=8
CP_SIZE=1
ENABLE_SP=0
YARN_FACTOR=""
MULTIHOST=0
RPA_MIXED_BLOCKS=""
RPA_MIXED_KEY=""
# The BABILong checkout (repo/) and dataset (dataset/) of eval/setup.sh.
BABILONG_DIR="$HOME/babilong"
OUT=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --backend) BACKEND="$2"; shift ;;
        --length) LENGTH="$2"; shift ;;
        --tasks) TASKS="$2"; shift ;;
        --limit) LIMIT="$2"; shift ;;
        --tp-size) TP_SIZE="$2"; shift ;;
        --cp-size) CP_SIZE="$2"; shift ;;
        --enable-sp) ENABLE_SP=1 ;;
        --yarn-factor) YARN_FACTOR="$2"; shift ;;
        --multihost) MULTIHOST=1 ;;
        --rpa-mixed-blocks) RPA_MIXED_BLOCKS="$2"; shift ;;
        --rpa-mixed-key) RPA_MIXED_KEY="$2"; shift ;;
        --babilong-dir) BABILONG_DIR="$2"; shift ;;
        --out) OUT="$2"; shift ;;
        *) echo "Unknown parameter: $1" >&2; exit 1 ;;
    esac
    shift
done

if [[ -z "$LENGTH" || -z "$OUT" ]]; then
    echo "Error: --length and --out are required" >&2
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

# The cache holds the model by revision and has no refs/main, so the engine
# is handed the snapshot directory.
SNAPSHOT=$("$VIRTUAL_ENV/bin/python" -c "
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2], local_files_only=True))
" Qwen/Qwen3.8-27B "$REVISION")
BABILONG_REPO="$BABILONG_DIR/repo"
DATA_DIR="$BABILONG_DIR/dataset/data"
if [[ "$(git -C "$BABILONG_REPO" rev-parse HEAD)" != "$BABILONG_COMMIT" ]]; then
    echo "Error: $BABILONG_REPO is not at $BABILONG_COMMIT" >&2
    exit 1
fi

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
# Lazy compilation; babilong_eval.py warms every prefill bucket the dataset
# reaches before it starts timing. The gap must match its BUCKET_GAP.
export SKIP_JAX_PRECOMPILE=1
export VLLM_TPU_BUCKET_PADDING_GAP=16384
export VLLM_XLA_CACHE_PATH="$HOME/babilong-cache"
export JAX_COMPILATION_CACHE_DIR="$HOME/babilong-cache/jax"
export VLLM_XLA_CHECK_RECOMPILATION=0

if [[ -n "$RPA_MIXED_BLOCKS" ]]; then
    if [[ "$BACKEND" != "baseline" || -z "$RPA_MIXED_KEY" ]]; then
        echo "Error: --rpa-mixed-blocks needs --backend baseline and --rpa-mixed-key" >&2
        exit 1
    fi
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
# engine and its Ray actors. Walk the tree instead.
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
    --snapshot "$SNAPSHOT"
    --revision "$REVISION"
    --data-dir "$DATA_DIR"
    --babilong-repo "$BABILONG_REPO"
    --babilong-commit "$BABILONG_COMMIT"
    --length "$LENGTH"
    --tasks "$TASKS"
    --tp-size "$TP_SIZE"
    --cp-size "$CP_SIZE"
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

printf 'RUN_START %s backend=%s length=%s tasks=%s limit=%s rpa=%s tp=%s cp=%s sp=%s yarn=%s multihost=%s\n' \
    "$(date -u +%FT%TZ)" "$BACKEND" "$LENGTH" "$TASKS" "${LIMIT:-all}" \
    "${RPA_V3_MIXED_BLOCK_SIZES:-default}${RPA_MIXED_KEY:+ ($RPA_MIXED_KEY)}" \
    "$TP_SIZE" "$CP_SIZE" "$ENABLE_SP" "${YARN_FACTOR:-none}" "$MULTIHOST"
"$VIRTUAL_ENV/bin/python" "$script_dir/babilong_eval.py" "${eval_args[@]}" &
child_pid=$!
wait "$child_pid"
printf 'RUN_DONE %s\n' "$(date -u +%FT%TZ)"
