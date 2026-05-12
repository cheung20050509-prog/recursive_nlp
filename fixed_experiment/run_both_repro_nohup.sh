#!/usr/bin/env bash
# Start MOSI (refine2 trial 278) and MOSEI (trial 95) fixed repros in parallel via nohup.
# Uses absolute paths so this works from any cwd.
# Single ~32GB GPU: both fixed runs together are usually fine. Use MOSI_CUDA / MOSEI_CUDA to split across two cards.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Per-job GPU index (logical ID after any parent CUDA_VISIBLE_DEVICES). Defaults: same card as ${CUDA_VISIBLE_DEVICES:-0}.
MOSI_CUDA="${MOSI_CUDA:-${CUDA_VISIBLE_DEVICES:-0}}"
MOSEI_CUDA="${MOSEI_CUDA:-${CUDA_VISIBLE_DEVICES:-0}}"

nohup env CUDA_VISIBLE_DEVICES="$MOSI_CUDA" bash "$DIR/run_mosi_refine2_best.sh" > "$DIR/repro_mosi_refine2_t278.log" 2>&1 &
echo $! > "$DIR/repro_mosi_refine2_t278.pid"

nohup env CUDA_VISIBLE_DEVICES="$MOSEI_CUDA" bash "$DIR/run_mosei_best.sh" > "$DIR/repro_mosei_t95.log" 2>&1 &
echo $! > "$DIR/repro_mosei_t95.pid"

echo "MOSI  CUDA_VISIBLE_DEVICES=$MOSI_CUDA  pid=$(cat "$DIR/repro_mosi_refine2_t278.pid")  log=$DIR/repro_mosi_refine2_t278.log"
echo "MOSEI CUDA_VISIBLE_DEVICES=$MOSEI_CUDA pid=$(cat "$DIR/repro_mosei_t95.pid")       log=$DIR/repro_mosei_t95.log"
echo "Two-GPU split example: MOSI_CUDA=0 MOSEI_CUDA=1 bash \"$DIR/run_both_repro_nohup.sh\""
echo "Tail: tail -f \"$DIR/repro_mosi_refine2_t278.log\"   # grep '^TEST:' when done"
