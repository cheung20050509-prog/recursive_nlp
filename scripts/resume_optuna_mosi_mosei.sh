#!/usr/bin/env bash
# Resume both Optuna studies on the same GPU in parallel.
# - MOSI : ithp_local_refine_mosi_mae  (scripts/optuna_local_refine_search.py)
# - MOSEI: ithp_mosei_mae               (scripts/optuna_search.py)
#
# Logs and per-side run dirs are placed under runs/optuna_resume_<ts>/.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$ROOT/runs/optuna_resume_$TS}"
mkdir -p "$RUN_DIR"

MOSI_GPU="${MOSI_GPU:-0}"
MOSEI_GPU="${MOSEI_GPU:-0}"
MOSI_LOCAL_TRIALS="${MOSI_LOCAL_TRIALS:-60}"
MOSEI_RANDOM_TRIALS="${MOSEI_RANDOM_TRIALS:-50}"
MOSEI_TPE_TRIALS="${MOSEI_TPE_TRIALS:-200}"

cat >"$RUN_DIR/README.txt" <<EOF
MOSI  -> GPU $MOSI_GPU  : optuna_local_refine_search.py --local_trials $MOSI_LOCAL_TRIALS
MOSEI -> GPU $MOSEI_GPU : optuna_search.py --random_trials $MOSEI_RANDOM_TRIALS --tpe_trials $MOSEI_TPE_TRIALS
Started: $(date -Is)
EOF

# MOSI: continue local-refine TPE phase. Output dir must match the original sqlite location.
(
  export CUDA_VISIBLE_DEVICES="$MOSI_GPU"
  cd "$ROOT"
  exec $PY -u scripts/optuna_local_refine_search.py \
    --dataset mosi --gpu "$MOSI_GPU" \
    --output_dir optuna_results_local_refine_mae_boundary_expand_moretrials_20260419_213850 \
    --local_trials "$MOSI_LOCAL_TRIALS"
) >"$RUN_DIR/mosi_optuna.log" 2>&1 &
echo $! >"$RUN_DIR/mosi_optuna.pid"

# MOSEI: continue broad random+TPE search on the same study.
(
  export CUDA_VISIBLE_DEVICES="$MOSEI_GPU"
  cd "$ROOT"
  exec $PY -u scripts/optuna_search.py \
    --dataset mosei --gpu "$MOSEI_GPU" \
    --output_dir optuna_results_mae \
    --random_trials "$MOSEI_RANDOM_TRIALS" \
    --tpe_trials "$MOSEI_TPE_TRIALS"
) >"$RUN_DIR/mosei_optuna.log" 2>&1 &
echo $! >"$RUN_DIR/mosei_optuna.pid"

echo "RUN_DIR=$RUN_DIR"
echo "MOSI  pid=$(cat "$RUN_DIR/mosi_optuna.pid")  GPU=$MOSI_GPU"
echo "MOSEI pid=$(cat "$RUN_DIR/mosei_optuna.pid") GPU=$MOSEI_GPU"
