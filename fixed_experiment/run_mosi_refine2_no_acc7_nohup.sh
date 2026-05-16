#!/usr/bin/env bash
# Full MOSI fixed repro (refine2 trial 278 hyperparams) with --acc7_loss_weight 0, under nohup.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="${PY:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python3}"
mkdir -p log
LOG="log/mosi_refine2_t278_no_acc7_$(date +%Y%m%d_%H%M%S).log"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
nohup "$PY" -u run_fixed.py mosi_refine2_t278 --acc7_loss_weight 0 >>"$LOG" 2>&1 &
echo $! >log/mosi_refine2_t278_no_acc7.pid
echo "$LOG" >log/mosi_refine2_t278_no_acc7_latest.log
echo "PID=$(cat log/mosi_refine2_t278_no_acc7.pid)"
echo "LOG=$HERE/$LOG"
echo "tail: tail -f $HERE/$LOG"
