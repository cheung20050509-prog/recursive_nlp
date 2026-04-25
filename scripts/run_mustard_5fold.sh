#!/usr/bin/env bash
# Run MUStARD speaker-independent 5-fold CV end-to-end using the HKT-style
# binary trainer, then aggregate per-fold BEST_RESULT lines into summary.json.
#
# Do not pipe this script through `head` or `tail -f` on its stdout — a closed
# pipe will send SIGPIPE to `tee` and `train_hkt_binary.py` and silently kill
# the whole loop. Use `tail -f log/4080_restart_hkt/mustard/<fold>/train.log`
# against the on-disk file instead.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
GPU="${GPU:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/log/4080_restart_hkt}"
N_EPOCHS="${N_EPOCHS:-15}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
DEV_BATCH_SIZE="${DEV_BATCH_SIZE:-32}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-32}"
EARLY_STOP="${EARLY_STOP:-0}"
SELECTION_METRIC="${SELECTION_METRIC:-accuracy}"
SEED="${SEED:-5149}"
NUM_FOLDS="${NUM_FOLDS:-5}"
DATASET="${DATASET:-mustard}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_DIR/$DATASET"

for FOLD in $(seq 0 $((NUM_FOLDS - 1))); do
  FOLD_DIR="$OUTPUT_DIR/$DATASET/fold_$FOLD"
  mkdir -p "$FOLD_DIR"
  echo "=== $(date -Is) starting $DATASET fold $FOLD on GPU $GPU (output=$FOLD_DIR) ==="
  $PY -u train_hkt_binary.py \
    --dataset "$DATASET" \
    --fold "$FOLD" \
    --n_epochs "$N_EPOCHS" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --dev_batch_size "$DEV_BATCH_SIZE" \
    --test_batch_size "$TEST_BATCH_SIZE" \
    --early_stopping_patience "$EARLY_STOP" \
    --selection_metric "$SELECTION_METRIC" \
    --seed "$SEED" \
    --num_folds "$NUM_FOLDS" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$DATASET/fold_$FOLD" \
    $EXTRA_ARGS \
    > "$FOLD_DIR/train.log" 2>&1
  echo "=== $(date -Is) finished $DATASET fold $FOLD ==="
  grep -E "^BEST_RESULT:" "$FOLD_DIR/train.log" | tail -1 || true
done

# Aggregate per-fold result.json into summary.json.
$PY - <<PY
import json, os, statistics, sys
root = "$OUTPUT_DIR/$DATASET"
records = []
for fold in range($NUM_FOLDS):
    path = os.path.join(root, f"fold_{fold}", "result.json")
    if not os.path.exists(path):
        print(f"WARN: {path} missing", file=sys.stderr)
        continue
    with open(path) as handle:
        records.append(json.load(handle))

def pick(record, key):
    return record["best"]["test"].get(key)

def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.mean(vals)), float(statistics.pstdev(vals))

summary = {
    "dataset": "$DATASET",
    "folds": [
        {
            "fold": record.get("fold"),
            "best_epoch": record["best"].get("epoch"),
            "valid": record["best"].get("valid"),
            "test": record["best"].get("test"),
            "data": record.get("data"),
        }
        for record in records
    ],
    "aggregate": {
        "test_accuracy": dict(zip(("mean", "std"), mean_std([pick(r, "accuracy") for r in records]))),
        "test_f1": dict(zip(("mean", "std"), mean_std([pick(r, "f1") for r in records]))),
        "test_f1_weighted": dict(zip(("mean", "std"), mean_std([pick(r, "f1_weighted") for r in records]))),
    },
    "num_folds_complete": len(records),
}
summary_path = os.path.join(root, "summary.json")
with open(summary_path, "w") as handle:
    json.dump(summary, handle, indent=2, ensure_ascii=False)
print("SUMMARY:", summary_path)
print("  mean test_acc:", summary["aggregate"]["test_accuracy"])
print("  mean test_f1:", summary["aggregate"]["test_f1"])
PY
