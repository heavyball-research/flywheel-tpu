#!/usr/bin/env bash
# RULER sweep: every (backend, sequence length) point, one after another.
#
# Only one process can hold the TPU, so the points run in series. They are
# ordered cheapest first, so the short lengths land while the long ones are
# still ahead; a point that already finished cleanly is skipped, which makes
# the sweep resumable after the spot VM is reclaimed.
#
# Every point writes into its own directory under --out (default
# ~/eval-logs/ruler).
#
# The baseline takes the tuned RPA v3 MIXED block sizes of the cell matching
# the prompt length each rank prefills (seqlen / cp_size); the table stops at
# 256k, which also serves the 512k PCP=2 point, as in the BABILong sweep.
#
# Usage (--multihost on the Ray head, with the slice's Ray cluster up):
#   eval/ruler/run_ruler_sweep.sh --seqlens "4096 32768 131072" --limit 500
#   eval/ruler/run_ruler_sweep.sh --out ~/ruler-logs/sweep_20260920 \
#     --seqlens 4096 --tp-size 4
#   eval/ruler/run_ruler_sweep.sh --seqlens 524288 --cp-size 2 --enable-sp \
#     --yarn-factor 2 --multihost
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SEQLENS="4096 32768 131072"
BACKENDS="flywheel baseline"
LIMIT=500
TASKS=ruler
CP_SIZE=1
MAX_NUM_SEQS=1
OUT="$HOME/eval-logs/ruler"
RPA_MIXED_BLOCKS="$script_dir/../configs/qwen38_27b_tp8_rpa_mixed_blocks.json"
# Passed through to run_ruler.sh.
EXTRA=()

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --seqlens) SEQLENS="$2"; shift ;;
        --backends) BACKENDS="$2"; shift ;;
        --limit) LIMIT="$2"; shift ;;
        --tasks) TASKS="$2"; shift ;;
        --rpa-mixed-blocks) RPA_MIXED_BLOCKS="$2"; shift ;;
        --tp-size) EXTRA+=(--tp-size "$2"); shift ;;
        --cp-size) CP_SIZE="$2"; EXTRA+=(--cp-size "$2"); shift ;;
        --enable-sp) EXTRA+=(--enable-sp) ;;
        --yarn-factor) EXTRA+=(--yarn-factor "$2"); shift ;;
        --max-num-seqs) MAX_NUM_SEQS="$2"; EXTRA+=(--max-num-seqs "$2"); shift ;;
        --gpu-memory-utilization) EXTRA+=(--gpu-memory-utilization "$2"); shift ;;
        --max-gen-toks) EXTRA+=(--max-gen-toks "$2"); shift ;;
        --multihost) EXTRA+=(--multihost) ;;
        --out) OUT="$2"; shift ;;
        *) echo "Unknown parameter: $1" >&2; exit 1 ;;
    esac
    shift
done

mkdir -p "$OUT"
printf 'SWEEP_OUT %s\n' "$OUT"

for seqlen in $SEQLENS; do
    local_tokens=$(( seqlen / CP_SIZE ))
    rpa_key="s$(( local_tokens > 262144 ? 262144 : local_tokens ))_b${MAX_NUM_SEQS}"
    for backend in $BACKENDS; do
        point="$OUT/${backend}_${seqlen}"
        # A zero exit_code alone is not proof of a finished point: the EXIT
        # trap also writes one when the run is killed, and $? can be 0 there.
        # The result file is what a completed point leaves behind.
        if [[ -f "$point/ruler_run.json" && -f "$point/exit_code" &&
              "$(cat "$point/exit_code")" == "0" ]]; then
            printf 'SWEEP_SKIP %s backend=%s seqlen=%s (already finished)\n' \
                "$(date -u +%FT%TZ)" "$backend" "$seqlen"
            continue
        fi
        rm -rf "$point"
        printf 'SWEEP_POINT %s backend=%s seqlen=%s\n' \
            "$(date -u +%FT%TZ)" "$backend" "$seqlen"
        args=(--backend "$backend" --seqlen "$seqlen" --tasks "$TASKS"
              --limit "$LIMIT" --out "$point" "${EXTRA[@]}")
        if [[ "$backend" == "baseline" ]]; then
            args+=(--rpa-mixed-blocks "$RPA_MIXED_BLOCKS"
                   --rpa-mixed-key "$rpa_key")
        fi
        bash "$script_dir/run_ruler.sh" "${args[@]}"
    done
done
printf 'SWEEP_DONE %s\n' "$(date -u +%FT%TZ)"
