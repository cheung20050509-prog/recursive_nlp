#!/usr/bin/env bash
# Resume MUStARD HKT Optuna on SAME log/4080_restart + original study_prefix (more TPE).
# Keeps categorical syntax_loss_weight (compatible with existing sqlite).
#
# Env: MUSTARD_GPU, HKT_OUT, HKT_STUDY_PREFIX, MUSTARD_RANDOM_TRIALS, MUSTARD_TPE_TRIALS, PY

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"

MUSTARD_GPU="${MUSTARD_GPU:-0}"
HKT_OUT="${HKT_OUT:-log/4080_restart}"
HKT_STUDY_PREFIX="${HKT_STUDY_PREFIX:-ithp_hkt_silver_20260430_114654}"
MUSTARD_RANDOM_TRIALS="${MUSTARD_RANDOM_TRIALS:-20}"
MUSTARD_TPE_TRIALS="${MUSTARD_TPE_TRIALS:-200}"

export CUDA_VISIBLE_DEVICES="$MUSTARD_GPU"
exec "$PY" -u scripts/optuna_hkt_search.py \
  --dataset mustard \
  --gpu "$MUSTARD_GPU" \
  --output_dir "$HKT_OUT" \
  --study_prefix "$HKT_STUDY_PREFIX" \
  --random_trials "$MUSTARD_RANDOM_TRIALS" \
  --tpe_trials "$MUSTARD_TPE_TRIALS" \
  --syntax-loss-weight-sampling categorical
