#!/usr/bin/env bash
# Sequentially run all manifest entries (MOSI/MOSEI fixed + UR-FUNNY HKT)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
exec "$PY" -u "$HERE/run_manifest.py" "$@"
