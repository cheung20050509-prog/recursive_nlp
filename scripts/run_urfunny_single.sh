#!/usr/bin/env bash
# Single-run UR-FUNNY binary trainer with the HKT-style pipeline.
# Writes log/4080_restart_hkt/urfunny/train.log and result.json.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
GPU="${GPU:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/log/4080_restart_hkt}"
N_EPOCHS="${N_EPOCHS:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
DEV_BATCH_SIZE="${DEV_BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-64}"
EARLY_STOP="${EARLY_STOP:-2}"
SELECTION_METRIC="${SELECTION_METRIC:-accuracy}"
SEED="${SEED:-5149}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false

RUN_DIR="$OUTPUT_DIR/urfunny"
mkdir -p "$RUN_DIR"
echo "=== $(date -Is) starting urfunny on GPU $GPU (output=$RUN_DIR) ==="
$PY -u train_hkt_binary.py \
  --dataset urfunny \
  --n_epochs "$N_EPOCHS" \
  --train_batch_size "$TRAIN_BATCH_SIZE" \
  --dev_batch_size "$DEV_BATCH_SIZE" \
  --test_batch_size "$TEST_BATCH_SIZE" \
  --early_stopping_patience "$EARLY_STOP" \
  --selection_metric "$SELECTION_METRIC" \
  --seed "$SEED" \
  --output_dir "$OUTPUT_DIR" \
  --run_name urfunny \
  $EXTRA_ARGS \
  > "$RUN_DIR/train.log" 2>&1
echo "=== $(date -Is) finished urfunny ==="
grep -E "^BEST_RESULT:" "$RUN_DIR/train.log" | tail -1 || true

$PY - <<PY
import json, os, sys
path = os.path.join("$RUN_DIR", "result.json")
if not os.path.exists(path):
    print(f"WARN: {path} missing", file=sys.stderr)
    sys.exit(0)
with open(path) as handle:
    record = json.load(handle)
summary = {
    "dataset": "urfunny",
    "best_epoch": record["best"].get("epoch"),
    "valid": record["best"].get("valid"),
    "test": record["best"].get("test"),
    "data": record.get("data"),
}
with open(os.path.join("$RUN_DIR", "summary.json"), "w") as handle:
    json.dump(summary, handle, indent=2, ensure_ascii=False)
print("SUMMARY:", os.path.join("$RUN_DIR", "summary.json"))
print("  test_acc:", summary["test"].get("accuracy"))
print("  test_f1:", summary["test"].get("f1"))
PY
