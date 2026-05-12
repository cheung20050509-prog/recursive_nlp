#!/usr/bin/env bash
# Paper-comparable manifest: Optuna-aligned MMSA batch (8/1) + single job by default.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
exec "$PY" -u "$HERE/run_manifest_parallel.py" --profile optuna -j 1 "$@"
