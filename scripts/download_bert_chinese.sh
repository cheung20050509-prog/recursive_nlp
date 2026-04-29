#!/usr/bin/env bash
# Download google-bert/bert-base-chinese into ITHP/recursive_ITHP/pretrained/bert-base-chinese
# (not committed; see .gitignore).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$ROOT/pretrained/bert-base-chinese}"
PY="${PY:-python}"

mkdir -p "$(dirname "$OUT")"
echo "Downloading to $OUT ..."
"$PY" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="google-bert/bert-base-chinese",
    local_dir="${OUT}",
    local_dir_use_symlinks=False,
)
print("done:", "${OUT}")
PY
