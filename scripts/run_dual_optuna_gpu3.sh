#!/bin/bash
set -e

cd /root/autodl-tmp/recursive_language/ITHP/recursive_ITHP
PYTHON=/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python

COMMON_ARGS=(
  --gpu 3
  --random_trials 50
  --tpe_trials 100
  --primary_metric mae
  --selection_metric mae
  --output_dir optuna_results_mae
)

MOSI_ARGS=(
  --n_epochs 20
  --early_stopping_patience 0
)

MOSEI_ARGS=(
  --n_epochs 10
  --early_stopping_patience 3
)

nohup "$PYTHON" -u scripts/optuna_search.py --dataset mosi "${COMMON_ARGS[@]}" "${MOSI_ARGS[@]}" > optuna_mosi_mae_gpu3.log 2>&1 &
echo "MOSI PID: $!"

nohup "$PYTHON" -u scripts/optuna_search.py --dataset mosei "${COMMON_ARGS[@]}" "${MOSEI_ARGS[@]}" > optuna_mosei_mae_gpu3.log 2>&1 &
echo "MOSEI PID: $!"