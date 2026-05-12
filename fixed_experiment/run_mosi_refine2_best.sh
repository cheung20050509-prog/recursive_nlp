#!/usr/bin/env bash
# MOSI: Optuna best from mosi_refine2 / trial 278 (ithp_mosi_mae_refine2).
# Delegates to run_fixed.py → fixed_training (embedded copy of recursive_ITHP train stack).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
exec "$PY" -u "$HERE/run_fixed.py" mosi_refine2_t278 "$@"
