#!/usr/bin/env bash
# Video-MME long-video reasoning accuracy + latency for one (backend, frames).
#
# lmms-eval runs the videomme_long_reasoning task through the vllm_generate_tpu
# backend: one request at a time, the whole prompt in one step (chunked prefill
# and prefix caching off), one output token. Each request is then exactly one
# engine step, and the runner's STEP_TIMING_LOG_PATH record of that step gives
# its vision encoder and LLM prefill times.
#
# Every frame is capped at 640x360, so the frame count sets the context length:
# 1024 frames are ~113k visual tokens and 2048 frames ~225k.
#
# Rerunning with the same --out resumes: requests already in requests.jsonl
# are not served again.
#
# Usage:
#   eval/videomme/run_videomme.sh --backend flywheel \
#     --frames 1024 --out ~/videomme-logs/flywheel_f1024
#   eval/videomme/run_videomme.sh \
#     --backend baseline --frames 2048 --limit 4 \
#     --out ~/videomme-logs/smoke_baseline_f2048
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Qwen/Qwen3.8-27B
REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
BACKEND=flywheel
FRAMES=1024
MAX_PIXELS=$((640 * 360))
LIMIT=""
TP_SIZE=8
GPU_MEMORY_UTILIZATION=""
MAX_MODEL_LEN=""
RPA_MIXED_BLOCKS=""
VIDEOMME_DIR=/dev/shm/videomme
CACHE_DIR=/dev/shm/videomme-cache
MULTIHOST=0
OUT=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --backend) BACKEND="$2"; shift ;;
        --frames) FRAMES="$2"; shift ;;
        --max-pixels) MAX_PIXELS="$2"; shift ;;
        --limit) LIMIT="$2"; shift ;;
        --tp-size) TP_SIZE="$2"; shift ;;
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift ;;
        --rpa-mixed-blocks) RPA_MIXED_BLOCKS="$2"; shift ;;
        --videomme-dir) VIDEOMME_DIR="$2"; shift ;;
        --cache-dir) CACHE_DIR="$2"; shift ;;
        # The whole slice through a Ray cluster that is already up; run this
        # on the Ray head.
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
# baseline runs the framework's legacy ViT kernel with one temporal patch per
# batch row, S * L^2 work, as upstream vLLM's per-sequence SDPA does, rather
# than masking each patch inside one flattened row of the whole clip.
case "$BACKEND" in
    baseline) KERNELS=0; LINEAR=0; VIT_SEGMENTS_AS_BATCH=1 ;;
    flywheel) KERNELS=1; LINEAR=1; VIT_SEGMENTS_AS_BATCH=0 ;;
    *) echo "Error: --backend must be baseline or flywheel" >&2; exit 1 ;;
esac

# Every compiled graph is keyed by max_model_len, so each frame count pins one
# length that holds its visual tokens, the timestamps and the question.
if [[ -z "$MAX_MODEL_LEN" ]]; then
    case "$FRAMES" in
        1024) MAX_MODEL_LEN=131072 ;;
        2048) MAX_MODEL_LEN=262144 ;;
        *) echo "Error: pass --max-model-len for --frames $FRAMES" >&2
           exit 1 ;;
    esac
fi

# The KV cache only ever holds one sequence, so it is kept near one
# max_model_len and the rest of HBM is left to the encoder and prefill
# activations: on v6e-8 these back 133k and 276k tokens, 1.02x and 1.05x.
if [[ -z "$GPU_MEMORY_UTILIZATION" ]]; then
    case "$MAX_MODEL_LEN" in
        131072) GPU_MEMORY_UTILIZATION=0.28 ;;
        262144) GPU_MEMORY_UTILIZATION=0.35 ;;
        *) echo "Error: pass --gpu-memory-utilization for" \
                "--max-model-len $MAX_MODEL_LEN" >&2
           exit 1 ;;
    esac
fi

PARQUET="$VIDEOMME_DIR/raw/videomme/test-00000-of-00001.parquet"
VIDEO_DIR="$VIDEOMME_DIR/videos"
if [[ ! -f "$PARQUET" || ! -d "$VIDEO_DIR" ]]; then
    echo "Error: Video-MME not found under $VIDEOMME_DIR" >&2
    exit 1
fi
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "Error: activate the eval venv first (see eval/setup.sh)" >&2
    exit 1
fi
# The cache holds the model by revision and has no refs/main, so the engine
# and the processor are both handed the snapshot directory.
SNAPSHOT=$("$VIRTUAL_ENV/bin/python" -c "
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2], local_files_only=True))
" Qwen/Qwen3.8-27B "$REVISION")

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

export USE_FLYWHEEL_TPU_KERNEL="$KERNELS"
export USE_FLYWHEEL_TPU_LINEAR_KERNEL="$LINEAR"
export VIT_SDPA_SEGMENTS_AS_BATCH="$VIT_SEGMENTS_AS_BATCH"
export DISABLE_WEIGHT_REQUANTIZATION=1
export ENABLE_QUANTIZED_MATMUL_KERNEL=0
export USE_BATCHED_RPA_KERNEL=0
export USE_BATCHED_RPA_SEQ_ON_LANE=0
export MODEL_IMPL_TYPE=vllm
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# Each request lands in one of a handful of shapes, so the warmup request and
# the backend's recompile-then-retime pass cover them without precompiling
# the whole bucket ladder.
export SKIP_JAX_PRECOMPILE=1
# A 118k-token prompt pads to 131072 on a 16384 gap and to 118784 on 4096,
# so the prefill runs on the length it was given rather than 11% more.
export VLLM_TPU_BUCKET_PADDING_GAP=4096
export VLLM_XLA_CACHE_PATH="$CACHE_DIR"
export JAX_COMPILATION_CACHE_DIR="$CACHE_DIR/jax"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_XLA_CHECK_RECOMPILATION=0

export FORCE_QWENVL_VIDEO_READER=decord
export VIDEOMME_VIDEO_DIR="$VIDEO_DIR"
export PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}"

# The tuned RPA v3 MIXED block sizes are keyed by the (seqlen, batch) cell the
# tuner swept, and the pinned engine length is one of the swept lengths.
if [[ -n "$RPA_MIXED_BLOCKS" ]]; then
    if [[ "$KERNELS" != "0" ]]; then
        echo "Error: --rpa-mixed-blocks only applies to --backend baseline" >&2
        exit 1
    fi
    RPA_V3_MIXED_BLOCK_SIZES=$("$VIRTUAL_ENV/bin/python" -c "
import json, sys
table, seqlen = json.load(open(sys.argv[1])), sys.argv[2]
key = 's%s_b1' % seqlen
if key not in table:
    raise SystemExit('%s has no %s entry' % (sys.argv[1], key))
print(','.join(str(size) for size in table[key]))
" "$RPA_MIXED_BLOCKS" "$MAX_MODEL_LEN")
    export RPA_V3_MIXED_BLOCK_SIZES
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
    # includes its compile.
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
    "$VIRTUAL_ENV/bin/ray" status
fi

mkdir -p "$OUT" "$CACHE_DIR"
OUT=$(cd "$OUT" && pwd)
exec > >(tee -a "$OUT/driver.log") 2>&1

# Killing this script alone would orphan the driver, which still holds the
# TPU while the flock it was serialised by is already released. Walk the tree
# instead of trusting a process group.
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

# lmms-eval resolves a task's `!function utils.*` next to its yaml, so the
# rendered task and its utils share one directory.
TASK_DIR="$OUT/tasks"
mkdir -p "$TASK_DIR"
template=$(<"$script_dir/tasks/videomme_long_reasoning.yaml.in")
printf '%s\n' "${template//__VIDEOMME_PARQUET__/$PARQUET}" \
    > "$TASK_DIR/videomme_long_reasoning.yaml"
ln -sfn "$script_dir/tasks/utils.py" "$TASK_DIR/utils.py"

export STEP_TIMING_LOG_PATH="$OUT/steps.jsonl"
touch "$STEP_TIMING_LOG_PATH"

model_args=(
    "model=$SNAPSHOT"
    "tensor_parallel_size=$TP_SIZE"
    "dtype=bfloat16"
    "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    "max_model_len=$MAX_MODEL_LEN"
    "max_num_batched_tokens=$MAX_MODEL_LEN"
    "max_num_seqs=1"
    "enable_chunked_prefill=False"
    "enable_prefix_caching=False"
    "async_scheduling=False"
    "block_size=128"
    "nframes=$FRAMES"
    "max_pixels=$MAX_PIXELS"
    "request_log=$OUT/requests.jsonl"
    # GDN makes this a hybrid mamba cache, whose page vLLM pads to the
    # attention page plus three mamba states. Sized from
    # gpu_memory_utilization, vLLM's engine-side check then prices one
    # max_model_len sequence at that padded page (163 GiB at 131072 against
    # 17.4 GiB) and refuses to start, although the worker's compact-mamba
    # split fits it. The split writes its own override into the worker's
    # cache config, which is the engine's only in a single-host process:
    # through Ray (--multihost) the engine never sees it, so flywheel fails there
    # too. Pin the pool for both backends to one sequence plus 8 blocks of
    # slack, as the RULER and BABILong runs do.
    "num_gpu_blocks_override=$(( (MAX_MODEL_LEN + 127) / 128 + 8 ))"
)
lmms_args=(
    --model vllm_generate_tpu
    --model_args "$(IFS=,; printf '%s' "${model_args[*]}")"
    --tasks videomme_long_reasoning
    --include_path "$TASK_DIR"
    --batch_size 1
    --log_samples
    --output_path "$OUT/lmms"
    # Below DEBUG, lmms-eval logs an evaluation error and still exits 0.
    --verbosity DEBUG
)
if [[ -n "$LIMIT" ]]; then
    lmms_args+=(--limit "$LIMIT")
fi

printf 'RUN_START %s backend=%s frames=%s max_pixels=%s max_model_len=%s gpu_mem=%s limit=%s rpa=%s multihost=%s\n' \
    "$(date -u +%FT%TZ)" "$BACKEND" "$FRAMES" "$MAX_PIXELS" "$MAX_MODEL_LEN" \
    "$GPU_MEMORY_UTILIZATION" "${LIMIT:-all}" "${RPA_V3_MIXED_BLOCK_SIZES:-default}" \
    "$MULTIHOST"
"$VIRTUAL_ENV/bin/python" -m videomme_tpu "${lmms_args[@]}" &
child_pid=$!
wait "$child_pid"
printf 'RUN_DONE %s\n' "$(date -u +%FT%TZ)"
