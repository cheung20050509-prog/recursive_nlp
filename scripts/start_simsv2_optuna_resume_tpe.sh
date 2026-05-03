#!/usr/bin/env bash
# Resume SIMSv2 Optuna on the SAME sqlite as log/4080_restart (more TPE trials only
# once random phase is full). Does not change search space or sampling mode.
#
# Env: SIMSV2_GPU, SIMSV2_STUDY_PREFIX, SIMSV2_OUT, SIMSV2_RANDOM_TRIALS, SIMSV2_TPE_TRIALS, PY

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"

SIMSV2_GPU="${SIMSV2_GPU:-0}"
SIMSV2_OUT="${SIMSV2_OUT:-log/4080_restart}"
SIMSV2_STUDY_PREFIX="${SIMSV2_STUDY_PREFIX:-ithp_silver_20260430_114654}"
SIMSV2_RANDOM_TRIALS="${SIMSV2_RANDOM_TRIALS:-50}"
SIMSV2_TPE_TRIALS="${SIMSV2_TPE_TRIALS:-200}"

export CUDA_VISIBLE_DEVICES="$SIMSV2_GPU"
exec "$PY" -u scripts/optuna_search.py \
  --dataset simsv2 \
  --gpu "$SIMSV2_GPU" \
  --output_dir "$SIMSV2_OUT" \
  --study_prefix "$SIMSV2_STUDY_PREFIX" \
  --random_trials "$SIMSV2_RANDOM_TRIALS" \
  --tpe_trials "$SIMSV2_TPE_TRIALS" \
  --simsv2-search-space aligned \
  --silver-span-loss-weight-sampling categorical
