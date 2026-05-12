#!/usr/bin/env bash
# Run MOSI/MOSEI ablation via fixed_experiment/run_fixed.py
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXED="${ROOT}/../fixed_experiment/run_fixed.py"
CFG="${1:?usage: $0 /path/to/config.json}"
PY="${PY:-python3}"
exec "$PY" -u "$FIXED" --config "$CFG" "${@:2}"
