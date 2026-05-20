#!/usr/bin/env bash
# Fresh MOSI + MOSEI Optuna under log/4090D_restart/more (4090D_restart/more layout).
# - Optuna objective: min valid (dev) MAE (--primary-metric valid_mae)
# - Trial 0: paper hyperparameters (--bootstrap-config)
# - Checkpoint selection inside train.py: --selection-metric mae (valid MAE)
#
# Stops prior log/4080_restart_no_acc7 drivers if still running.
#
# Usage (from ITHP/recursive_ITHP):
#   bash scripts/start_optuna_4090D_dev_mae.sh
#
# Env:
#   OUT_ROOT   default log/4090D_restart/more
#   GPU_MOSI GPU_MOSEI  default 0
#   MOSI_RANDOM_TRIALS MOSI_TPE_TRIALS  default 30 / 120
#   MOSEI_RANDOM_TRIALS MOSEI_TPE_TRIALS default 30 / 200

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python3}"
OUT_REL="${OUT_ROOT:-log/4090D_restart/more}"
if [[ "$OUT_REL" = /* ]]; then
  OUT="$OUT_REL"
else
  OUT="$ROOT/$OUT_REL"
fi
mkdir -p "$OUT"

GPU_MOSI="${GPU_MOSI:-0}"
GPU_MOSEI="${GPU_MOSEI:-0}"
MOSI_R="${MOSI_RANDOM_TRIALS:-30}"
MOSI_T="${MOSI_TPE_TRIALS:-120}"
MOSEI_R="${MOSEI_RANDOM_TRIALS:-30}"
MOSEI_T="${MOSEI_TPE_TRIALS:-200}"

stop_old() {
  local pidf
  for pidf in \
    "$ROOT/log/4080_restart_no_acc7/mosi_refine2_optuna.pid" \
    "$ROOT/log/4080_restart_no_acc7/mosi_refine2_optuna_resume.pid" \
    "$ROOT/log/4080_restart_no_acc7/mosei_optuna.pid" \
    "$ROOT/log/4080_restart_no_acc7/urfunny_optuna.pid"; do
    if [[ -f "$pidf" ]]; then
      local pid
      pid="$(cat "$pidf" 2>/dev/null || true)"
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "[stop] SIGTERM pid=$pid from $pidf"
        kill -TERM "$pid" 2>/dev/null || true
      fi
    fi
  done
  sleep 3
  pkill -TERM -f 'optuna_mosi_refine.py.*4080_restart_no_acc7' 2>/dev/null || true
  pkill -TERM -f 'optuna_search.py.*4080_restart_no_acc7' 2>/dev/null || true
  pkill -TERM -f 'optuna_hkt_search.py.*4080_restart_no_acc7' 2>/dev/null || true
  sleep 2
  pkill -KILL -f 'optuna_mosi_refine.py.*4080_restart_no_acc7' 2>/dev/null || true
  pkill -KILL -f 'optuna_search.py.*4080_restart_no_acc7' 2>/dev/null || true
  pkill -KILL -f 'optuna_hkt_search.py.*4080_restart_no_acc7' 2>/dev/null || true
}

stop_old

cat >"$OUT/README_dev_mae.txt" <<EOF
4090D_restart/more — Optuna min valid_mae (dev MAE).

Started: $(date -Is)
  MOSI:  $OUT_REL/mosi_refine2  study=ithp_mosi_mae_refine2_devmae
  MOSEI: $OUT_REL/mosei         study=ithp_4090d_mosei_devmae

Trial 0: configs/optuna_paper_trial0_{mosi,mosei}.json (paper / baseline_table)
train.py: --selection_metric mae, --acc7_loss_weight 0.2
EOF

(
  exec "$PY" -u scripts/optuna_mosi_refine.py \
    --gpu "$GPU_MOSI" \
    --output_dir "$OUT_REL/mosi_refine2" \
    --study_name "ithp_mosi_mae_refine2_devmae" \
    --import_mode none \
    --primary_metric valid_mae \
    --selection_metric mae \
    --bootstrap-config configs/optuna_paper_trial0_mosi.json \
    --random_trials "$MOSI_R" \
    --tpe_trials "$MOSI_T" \
    --n_epochs 20 \
    --early_stopping_patience 0 \
    --acc7-loss-weight 0.2
) >"$OUT/mosi_refine2_optuna.log" 2>&1 &
echo $! >"$OUT/mosi_refine2_optuna.pid"

(
  exec "$PY" -u scripts/optuna_search.py \
    --dataset mosei \
    --gpu "$GPU_MOSEI" \
    --output_dir "$OUT_REL" \
    --study_prefix "ithp_4090d" \
    --import-mode none \
    --primary_metric valid_mae \
    --selection_metric mae \
    --bootstrap-config configs/optuna_paper_trial0_mosei.json \
    --random_trials "$MOSEI_R" \
    --tpe_trials "$MOSEI_T" \
    --n_epochs 10 \
    --early_stopping_patience 3 \
    --acc7-loss-weight 0.2
) >"$OUT/mosei_optuna.log" 2>&1 &
echo $! >"$OUT/mosei_optuna.pid"

echo "OUT=$OUT"
echo "MOSI  pid=$(cat "$OUT/mosi_refine2_optuna.pid") GPU=$GPU_MOSI random=$MOSI_R tpe=$MOSI_T"
echo "MOSEI pid=$(cat "$OUT/mosei_optuna.pid") GPU=$GPU_MOSEI random=$MOSEI_R tpe=$MOSEI_T"
echo "Logs: $OUT/mosi_refine2_optuna.log  $OUT/mosei_optuna.log"
