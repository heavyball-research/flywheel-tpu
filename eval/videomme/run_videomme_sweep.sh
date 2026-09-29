#!/usr/bin/env bash
# Video-MME sweep: every (backend, frames) point, one after another.
#
# Only one process can hold the TPU, so the points run in series, backend in
# the outer loop so that a backend's compilation cache stays warm across its
# frame budgets. A point that finished cleanly is skipped, and one that did
# not resumes from its request log, which makes the sweep safe to restart
# after the spot VM is reclaimed.
#
# Every point writes into its own directory under --out (default
# ~/eval-logs/videomme).
#
# Usage:
#   eval/videomme/run_videomme_sweep.sh
#   eval/videomme/run_videomme_sweep.sh \
#     --out ~/videomme-logs/smoke --frames 1024 --limit 4
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BACKENDS="flywheel baseline"
FRAMES="1024 2048"
LIMIT=""
OUT="$HOME/eval-logs/videomme"
RPA_MIXED_BLOCKS="$script_dir/../configs/qwen38_27b_tp8_rpa_mixed_blocks.json"
# Passed through to run_videomme.sh.
EXTRA=()

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --backends) BACKENDS="$2"; shift ;;
        --frames) FRAMES="$2"; shift ;;
        --limit) LIMIT="$2"; shift ;;
        --rpa-mixed-blocks) RPA_MIXED_BLOCKS="$2"; shift ;;
        --tp-size) EXTRA+=(--tp-size "$2"); shift ;;
        --multihost) EXTRA+=(--multihost) ;;
        --videomme-dir) EXTRA+=(--videomme-dir "$2"); shift ;;
        --out) OUT="$2"; shift ;;
        *) echo "Unknown parameter: $1" >&2; exit 1 ;;
    esac
    shift
done

mkdir -p "$OUT"
printf 'SWEEP_OUT %s\n' "$OUT"

for backend in $BACKENDS; do
    for frames in $FRAMES; do
        point="$OUT/${backend}_f${frames}"
        if [[ -f "$point/exit_code" && "$(cat "$point/exit_code")" == "0" ]]
        then
            printf 'SWEEP_SKIP %s backend=%s frames=%s (already finished)\n' \
                "$(date -u +%FT%TZ)" "$backend" "$frames"
            continue
        fi
        printf 'SWEEP_POINT %s backend=%s frames=%s\n' \
            "$(date -u +%FT%TZ)" "$backend" "$frames"
        args=(--backend "$backend" --frames "$frames" --out "$point"
              "${EXTRA[@]}")
        if [[ -n "$LIMIT" ]]; then
            args+=(--limit "$LIMIT")
        fi
        if [[ "$backend" == "baseline" && -n "$RPA_MIXED_BLOCKS" ]]; then
            args+=(--rpa-mixed-blocks "$RPA_MIXED_BLOCKS")
        fi
        bash "$script_dir/run_videomme.sh" "${args[@]}"
    done
done
printf 'SWEEP_DONE %s\n' "$(date -u +%FT%TZ)"
