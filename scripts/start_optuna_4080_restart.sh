#!/usr/bin/env bash
# Fresh Optuna search after switching to RTX 4080 (no resume).
# Outputs (sqlite + trial_logs + study_summary.json + parent stdout) all live in log/4080_restart/.
#
#   bash scripts/start_optuna_4080_restart.sh
#
# Tunables via env:  MOSI_RANDOM_TRIALS, MOSI_TPE_TRIALS, MOSEI_RANDOM_TRIALS, MOSEI_TPE_TRIALS, MOSI_GPU, MOSEI_GPU.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
OUT_DIR="$ROOT/log/4080_restart"
mkdir -p "$OUT_DIR"

MOSI_GPU="${MOSI_GPU:-0}"
MOSEI_GPU="${MOSEI_GPU:-0}"
MOSI_RANDOM_TRIALS="${MOSI_RANDOM_TRIALS:-50}"
MOSI_TPE_TRIALS="${MOSI_TPE_TRIALS:-150}"
MOSEI_RANDOM_TRIALS="${MOSEI_RANDOM_TRIALS:-50}"
MOSEI_TPE_TRIALS="${MOSEI_TPE_TRIALS:-150}"

cat >"$OUT_DIR/README.txt" <<EOF
Fresh Optuna runs on RTX 4080.
Started: $(date -Is)
MOSI  -> GPU $MOSI_GPU  | random=$MOSI_RANDOM_TRIALS  tpe=$MOSI_TPE_TRIALS
MOSEI -> GPU $MOSEI_GPU | random=$MOSEI_RANDOM_TRIALS tpe=$MOSEI_TPE_TRIALS
Layout:
  $OUT_DIR/mosi/optuna_study.sqlite3
  $OUT_DIR/mosi/trial_logs/...
  $OUT_DIR/mosi/study_summary.json
  $OUT_DIR/mosi_optuna.log    # parent stdout
  $OUT_DIR/mosei/optuna_study.sqlite3
  $OUT_DIR/mosei/trial_logs/...
  $OUT_DIR/mosei/study_summary.json
  $OUT_DIR/mosei_optuna.log
EOF

(
  export CUDA_VISIBLE_DEVICES="$MOSI_GPU"
  cd "$ROOT"
  exec $PY -u scripts/optuna_search.py \
    --dataset mosi --gpu "$MOSI_GPU" \
    --output_dir log/4080_restart \
    --random_trials "$MOSI_RANDOM_TRIALS" \
    --tpe_trials "$MOSI_TPE_TRIALS"
) >"$OUT_DIR/mosi_optuna.log" 2>&1 &
echo $! >"$OUT_DIR/mosi_optuna.pid"

(
  export CUDA_VISIBLE_DEVICES="$MOSEI_GPU"
  cd "$ROOT"
  exec $PY -u scripts/optuna_search.py \
    --dataset mosei --gpu "$MOSEI_GPU" \
    --output_dir log/4080_restart \
    --random_trials "$MOSEI_RANDOM_TRIALS" \
    --tpe_trials "$MOSEI_TPE_TRIALS"
) >"$OUT_DIR/mosei_optuna.log" 2>&1 &
echo $! >"$OUT_DIR/mosei_optuna.pid"

echo "OUT_DIR=$OUT_DIR"
echo "MOSI  pid=$(cat "$OUT_DIR/mosi_optuna.pid")  GPU=$MOSI_GPU"
echo "MOSEI pid=$(cat "$OUT_DIR/mosei_optuna.pid") GPU=$MOSEI_GPU"
