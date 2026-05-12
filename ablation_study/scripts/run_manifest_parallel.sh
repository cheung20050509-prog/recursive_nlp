#!/usr/bin/env bash
# Run manifest with concurrent jobs on the same GPU (use --fixed-extra / --hkt-extra to cut VRAM).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
exec "$PY" -u "$HERE/run_manifest_parallel.py" "$@"
