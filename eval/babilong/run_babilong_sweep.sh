#!/usr/bin/env bash
# BABILong sweep: every (length, backend) point in series, flywheel before
# baseline at each length. A point that already finished cleanly is skipped,
# and an unfinished one resumes from its predictions.jsonl.
#
# The baseline takes the tuned RPA v3 MIXED block sizes of the cell matching
# the prompt length each rank prefills (length / cp_size); the table stops at
# 256k, which also serves the 512k PCP=2 point, as in the RULER 512k runs.
#
# Every point writes into its own directory under --out (default
# ~/eval-logs/babilong).
#
# Usage (--multihost on the Ray head, with the slice's Ray cluster up):
#   eval/babilong/run_babilong_sweep.sh --lengths "128k 256k"
#   eval/babilong/run_babilong_sweep.sh --lengths 512k \
#     --cp-size 2 --enable-sp --yarn-factor 2 --multihost \
#     --out ~/babilong-logs/sweep
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LENGTHS=""
BACKENDS="flywheel baseline"
CP_SIZE=1
EXTRA=()
OUT="$HOME/eval-logs/babilong"
RPA_MIXED_BLOCKS="$script_dir/../configs/qwen38_27b_tp8_rpa_mixed_blocks.json"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --lengths) LENGTHS="$2"; shift ;;
        --backends) BACKENDS="$2"; shift ;;
        --tp-size) EXTRA+=(--tp-size "$2"); shift ;;
        --multihost) EXTRA+=(--multihost) ;;
        --cp-size) CP_SIZE="$2"; EXTRA+=(--cp-size "$2"); shift ;;
        --enable-sp) EXTRA+=(--enable-sp) ;;
        --yarn-factor) EXTRA+=(--yarn-factor "$2"); shift ;;
        --limit) EXTRA+=(--limit "$2"); shift ;;
        --babilong-dir) EXTRA+=(--babilong-dir "$2"); shift ;;
        --out) OUT="$2"; shift ;;
        *) echo "Unknown parameter: $1" >&2; exit 1 ;;
    esac
    shift
done

if [[ -z "$LENGTHS" ]]; then
    echo "Error: --lengths is required" >&2
    exit 1
fi

mkdir -p "$OUT"
printf 'SWEEP_OUT %s\n' "$OUT"

for length in $LENGTHS; do
    local_tokens=$(( ${length%k} * 1024 / CP_SIZE ))
    rpa_key="s$(( local_tokens > 262144 ? 262144 : local_tokens ))_b1"
    for backend in $BACKENDS; do
        point="$OUT/${backend}_${length}"
        # The EXIT trap writes an exit_code even for a killed run; the result
        # file is what a completed point leaves behind.
        if [[ -f "$point/babilong_run.json" && -f "$point/exit_code" &&
              "$(cat "$point/exit_code")" == "0" ]]; then
            printf 'SWEEP_SKIP %s backend=%s length=%s (already finished)\n' \
                "$(date -u +%FT%TZ)" "$backend" "$length"
            continue
        fi
        printf 'SWEEP_POINT %s backend=%s length=%s\n' \
            "$(date -u +%FT%TZ)" "$backend" "$length"
        args=(--backend "$backend" --length "$length" --out "$point"
              "${EXTRA[@]}")
        if [[ "$backend" == "baseline" ]]; then
            args+=(--rpa-mixed-blocks "$RPA_MIXED_BLOCKS"
                   --rpa-mixed-key "$rpa_key")
        fi
        bash "$script_dir/run_babilong.sh" "${args[@]}"
    done
done
printf 'SWEEP_DONE %s\n' "$(date -u +%FT%TZ)"
