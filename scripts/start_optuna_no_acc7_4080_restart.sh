#!/usr/bin/env bash
# Optuna sweeps: MOSI/MOSEI with train.py --acc7_loss_weight 0 (Acc7 CE off).
# Phase-1 starts from **top-K configs of the prior 4080_restart studies** (enqueue), then random + TPE.
#
# UR-Funny (HKT) has no Acc7 loss; we still enqueue top-K from the old urfunny sqlite so the first
# trials reproduce strong configs before exploring.
#
# IMPORTANT: target sqlite must be **empty** for enqueue to run. If you already started a cold
# no-acc7 study, stop the drivers and remove e.g.:
#   rm -rf log/4080_restart_no_acc7/mosi_refine2 log/4080_restart_no_acc7/mosei log/4080_restart_no_acc7/urfunny
# then re-run this script (or set NO_ACC7_OUT to a fresh directory).
#
# Usage (from ITHP/recursive_ITHP):
#   bash scripts/start_optuna_no_acc7_4080_restart.sh
#
# Env (optional):
#   NO_ACC7_OUT          root log dir (default: log/4080_restart_no_acc7)
#   GPU_MOSI GPU_MOSEI GPU_URFUNNY
#   SEED_TOP_K          prior trials to enqueue first (default: 5)
#   MOSI_RANDOM_TRIALS  phase-1 budget >= SEED_TOP_K (default: SEED_TOP_K + 20)
#   MOSEI_RANDOM_TRIALS MOSEI_TPE_TRIALS  UR_RANDOM_TRIALS UR_TPE_TRIALS

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python3}"
OUT_REL="${NO_ACC7_OUT:-log/4080_restart_no_acc7}"
if [[ "$OUT_REL" = /* ]]; then
  OUT="$OUT_REL"
else
  OUT="$ROOT/$OUT_REL"
fi
mkdir -p "$OUT"

GPU_MOSI="${GPU_MOSI:-0}"
GPU_MOSEI="${GPU_MOSEI:-0}"
GPU_URFUNNY="${GPU_URFUNNY:-0}"

SEED_TOP_K="${SEED_TOP_K:-5}"
MOSI_R="${MOSI_RANDOM_TRIALS:-$((SEED_TOP_K + 20))}"
MOSI_T="${MOSI_TPE_TRIALS:-120}"
MOSEI_R="${MOSEI_RANDOM_TRIALS:-$((SEED_TOP_K + 25))}"
MOSEI_T="${MOSEI_TPE_TRIALS:-200}"
UR_R="${UR_RANDOM_TRIALS:-$((SEED_TOP_K + 15))}"
UR_T="${UR_TPE_TRIALS:-200}"

cat >"$OUT/README_no_acc7.txt" <<EOF
Acc7 CE auxiliary: OFF for MOSI/MOSEI (--acc7_loss_weight 0).

Phase-1: enqueue top ${SEED_TOP_K} configs from prior log/4080_restart/* studies, then random exploration.
UR-Funny: enqueue from prior urfunny sqlite (same HKT protocol).

Root: $OUT_REL (resolved: $OUT)

Started: $(date -Is)
  MOSI:   $OUT_REL/mosi_refine2  (enqueue from mosi_refine2 / ithp_mosi_mae_refine2)
  MOSEI:  $OUT_REL/mosei         (enqueue from mosei / ithp_mosei_mae)
  UR:     $OUT_REL/urfunny       (enqueue from urfunny / ithp_hkt_albert_20260507_225531_urfunny_valid_accuracy_threshold_tuned)

If enqueue was skipped, delete dataset subdirs under this root and restart.
EOF

(
  cd "$ROOT"
  exec "$PY" -u scripts/optuna_mosi_refine.py \
    --gpu "$GPU_MOSI" \
    --output_dir "$OUT_REL/mosi_refine2" \
    --study_name "ithp_mosi_mae_refine2_noacc7" \
    --import_mode enqueue \
    --seed_sqlite "log/4080_restart/mosi_refine2/optuna_study.sqlite3" \
    --seed_study_name "ithp_mosi_mae_refine2" \
    --seed_top_k "$SEED_TOP_K" \
    --random_trials "$MOSI_R" \
    --tpe_trials "$MOSI_T" \
    --acc7-loss-weight 0
) >"$OUT/mosi_refine2_optuna.log" 2>&1 &
echo $! >"$OUT/mosi_refine2_optuna.pid"

(
  cd "$ROOT"
  exec "$PY" -u scripts/optuna_search.py \
    --dataset mosei \
    --gpu "$GPU_MOSEI" \
    --output_dir "$OUT_REL" \
    --study_prefix "ithp_noacc7" \
    --import-mode enqueue \
    --seed-sqlite "log/4080_restart/mosei/optuna_study.sqlite3" \
    --seed-study-name "ithp_mosei_mae" \
    --seed-top-k "$SEED_TOP_K" \
    --random_trials "$MOSEI_R" \
    --tpe_trials "$MOSEI_T" \
    --acc7-loss-weight 0
) >"$OUT/mosei_optuna.log" 2>&1 &
echo $! >"$OUT/mosei_optuna.pid"

(
  cd "$ROOT"
  exec "$PY" -u scripts/optuna_hkt_search.py \
    --dataset urfunny \
    --gpu "$GPU_URFUNNY" \
    --output_dir "$OUT_REL" \
    --study_prefix "ithp_hkt_albert_noacc7" \
    --import-mode enqueue \
    --seed-sqlite "log/4080_restart/urfunny/optuna_study.sqlite3" \
    --seed-study-name "ithp_hkt_albert_20260507_225531_urfunny_valid_accuracy_threshold_tuned" \
    --seed-top-k "$SEED_TOP_K" \
    --primary_metric valid_accuracy_threshold_tuned \
    --decision-threshold-mode tune_on_valid \
    --threshold-tune-objective accuracy \
    --random_trials "$UR_R" \
    --tpe_trials "$UR_T" \
    --n_epochs 10 \
    --early_stopping_patience 2 \
    --seed 5149 \
    --backbone albert \
    --base_model "/root/autodl-tmp/recursive_nlp/albert-base-v2" \
    --syntax-loss-weight-high 0.5
) >"$OUT/urfunny_optuna.log" 2>&1 &
echo $! >"$OUT/urfunny_optuna.pid"

echo "OUT_DIR=$OUT (NO_ACC7_OUT=$OUT_REL)"
echo "MOSI refine (no acc7, seeded)  pid=$(cat "$OUT/mosi_refine2_optuna.pid")  GPU=$GPU_MOSI"
echo "MOSEI (no acc7, seeded)        pid=$(cat "$OUT/mosei_optuna.pid")       GPU=$GPU_MOSEI"
echo "UR-Funny (seeded)              pid=$(cat "$OUT/urfunny_optuna.pid")     GPU=$GPU_URFUNNY"
