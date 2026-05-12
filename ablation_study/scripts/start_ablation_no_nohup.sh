#!/usr/bin/env bash
# Full manifest ablation without nohup: all output goes to logs/ablation_no_nohup_<stamp>.log
# Default: optuna profile (MMSA 8/1, matches repro) + -j 1. Extra args go to run_manifest_parallel.py.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_ROOT="$(cd "$HERE/.." && pwd)"
cd "$ABLATION_ROOT"
mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ABLATION_ROOT/logs/ablation_no_nohup_${STAMP}.log"
echo "$LOG" >"$ABLATION_ROOT/logs/ablation_latest.log"
PY="${PY:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python3}"
export PY
exec >>"$LOG" 2>&1
echo "=== ablation start ${STAMP} (no nohup) PY=${PY} ==="
echo "LOG=${LOG}"
exec "$PY" -u "$HERE/run_manifest_parallel.py" --profile optuna -j 1 "$@"
