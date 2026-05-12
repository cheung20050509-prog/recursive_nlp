#!/usr/bin/env bash
# MOSEI: Optuna best from mosei / trial 95 (ithp_mosei_mae).
# Delegates to run_fixed.py → fixed_training (embedded copy of recursive_ITHP train stack).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
exec "$PY" -u "$HERE/run_fixed.py" mosei_t95 "$@"
