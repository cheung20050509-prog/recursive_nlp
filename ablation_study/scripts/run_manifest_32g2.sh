#!/usr/bin/env bash
# Throughput / VRAM mode: two concurrent jobs with smaller per-process batch.
# NOT strictly comparable to Optuna MMSA or fixed_experiment repro (uses 4x2 for fixed).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
exec "$PY" -u "$HERE/run_manifest_parallel.py" --profile 32g2 -j 2 "$@"
