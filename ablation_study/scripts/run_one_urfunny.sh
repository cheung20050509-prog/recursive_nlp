#!/usr/bin/env bash
# Run UR-FUNNY ablation via train_hkt_binary.py (JSON argv launcher)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${1:?usage: $0 /path/to/urfunny_ablation.json}"
PY="${PY:-python3}"
exec "$PY" -u "$HERE/launch_hkt_json.py" "$CFG" "${@:2}"
