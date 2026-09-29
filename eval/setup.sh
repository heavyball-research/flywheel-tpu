#!/usr/bin/env bash
# Eval dependencies and data, on top of the vLLM venv of the README's "vLLM
# backend" section. Run every step with that venv activated; the pins are
# what the reported runs used.
#
# Usage:
#   eval/setup.sh deps        # lm_eval (RULER, BABILong), lmms-eval (Video-MME)
#   eval/setup.sh model       # Qwen3.8-27B into the Hugging Face cache
#   eval/setup.sh ruler       # RULER's haystack essays into the datasets cache
#   eval/setup.sh babilong --lengths "128k 256k" [--babilong-dir DIR]
#   eval/setup.sh videomme [--videomme-dir DIR]
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

MODEL=Qwen/Qwen3.8-27B
MODEL_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
LM_EVAL="lm_eval[ruler]==0.4.13"
LMMS_EVAL="lmms-eval @ git+https://github.com/EvolvingLMMs-Lab/lmms-eval@1cd474f858"
BABILONG_COMMIT=7a6efee29f5cac03c3c410e6799c80fd2ffe3610
BABILONG_DATA_REVISION=ee0d588794c7ac098062ee0d247c733d62e94fe2
VIDEOMME_DATA_REVISION=ead1408f75b618502df9a1d8e0950166bf0a2a0b

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "Error: activate the vLLM venv first" >&2
    exit 1
fi
python="$VIRTUAL_ENV/bin/python"
hf="$VIRTUAL_ENV/bin/hf"
export HF_XET_HIGH_PERFORMANCE=1

no_args() {
    if [[ "$#" -gt 0 ]]; then
        echo "Unknown parameter: $1" >&2
        exit 1
    fi
}

deps() {
    no_args "$@"
    "$python" -c "
import importlib.util
missing = [name for name in ('vllm', 'tpu_inference', 'flywheel_tpu')
           if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit('install the vLLM backend first; missing ' + ', '.join(missing))
"
    # Everything below installs under the engine's current versions as
    # constraints, so the evals cannot move vLLM, JAX or torch.
    local constraints
    constraints=$(mktemp)
    uv pip freeze --python "$python" | grep -v -e '^-e' -e ' @ ' > "$constraints"
    uv pip install --python "$python" "$LM_EVAL" -c "$constraints"
    # lmms-eval goes in without its declared dependencies; the runtime ones,
    # less wandb, are videomme/requirements.txt.
    uv pip install --python "$python" --no-deps "$LMMS_EVAL"
    uv pip install --python "$python" -r "$script_dir/videomme/requirements.txt" \
        -c "$constraints"
    rm -f "$constraints"
}

model() {
    no_args "$@"
    "$hf" download "$MODEL" --revision "$MODEL_REVISION"
}

ruler() {
    no_args "$@"
    # The runs are offline, and RULER's needle tasks hide the needle in these.
    "$python" -c "
from datasets import load_dataset
load_dataset('baber/paul_graham_essays')
"
}

babilong() {
    local dir="$HOME/babilong" lengths="" length task
    local files=()
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --lengths) lengths="$2"; shift ;;
            --babilong-dir) dir="$2"; shift ;;
            *) echo "Unknown parameter: $1" >&2; exit 1 ;;
        esac
        shift
    done
    if [[ -z "$lengths" ]]; then
        echo "Error: --lengths is required" >&2
        exit 1
    fi
    # The repo holds BABILong's prompts and answer check; the dataset holds
    # one file per (task, length).
    if [[ ! -d "$dir/repo" ]]; then
        git clone https://github.com/booydar/babilong "$dir/repo"
        git -C "$dir/repo" checkout -q "$BABILONG_COMMIT"
    fi
    if [[ "$(git -C "$dir/repo" rev-parse HEAD)" != "$BABILONG_COMMIT" ]]; then
        echo "Error: $dir/repo is not at $BABILONG_COMMIT" >&2
        exit 1
    fi
    for length in $lengths; do
        for task in 1 2 3 4 5; do
            files+=("data/qa$task/$length.json")
        done
    done
    "$hf" download RMT-team/babilong "${files[@]}" --repo-type dataset \
        --revision "$BABILONG_DATA_REVISION" --local-dir "$dir/dataset"
}

videomme() {
    local dir=/dev/shm/videomme
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --videomme-dir) dir="$2"; shift ;;
            *) echo "Unknown parameter: $1" >&2; exit 1 ;;
        esac
        shift
    done
    # The videos ship as zip chunks; only the long ones are kept.
    "$hf" download lmms-eval/Video-MME --repo-type dataset \
        --revision "$VIDEOMME_DATA_REVISION" --local-dir "$dir/raw"
    "$python" "$script_dir/videomme/extract_long_videos.py" "$dir"
    rm -f "$dir"/raw/videos_chunked_*.zip "$dir/raw/subtitle.zip"
}

step=${1:-}
if [[ "$#" -gt 0 ]]; then
    shift
fi
case "$step" in
    deps|model|ruler|babilong|videomme) "$step" "$@" ;;
    *) echo "Usage: eval/setup.sh deps|model|ruler|babilong|videomme [options]" >&2
       exit 1 ;;
esac
