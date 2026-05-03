#!/usr/bin/env bash
# NEW Optuna study: SIMSv2 with uniform silver_span_loss_weight in [0, high].
# Requires new output dir + study_prefix (cannot mix with old categorical sqlite).
#
# Env: SIMSV2_GPU, SIMSV2_OUT, SIMSV2_STUDY_PREFIX, SIMSV2_RANDOM_TRIALS, SIMSV2_TPE_TRIALS,
#      SILVER_UNIFORM_HIGH, PY

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"

SIMSV2_GPU="${SIMSV2_GPU:-0}"
SIMSV2_OUT="${SIMSV2_OUT:-log/simsv2_uniform_silver}"
SIMSV2_STUDY_PREFIX="${SIMSV2_STUDY_PREFIX:-ithp_silver_uniform}"
SIMSV2_RANDOM_TRIALS="${SIMSV2_RANDOM_TRIALS:-30}"
SIMSV2_TPE_TRIALS="${SIMSV2_TPE_TRIALS:-50}"
SILVER_UNIFORM_HIGH="${SILVER_UNIFORM_HIGH:-0.25}"

mkdir -p "$SIMSV2_OUT"
export CUDA_VISIBLE_DEVICES="$SIMSV2_GPU"
exec "$PY" -u scripts/optuna_search.py \
  --dataset simsv2 \
  --gpu "$SIMSV2_GPU" \
  --output_dir "$SIMSV2_OUT" \
  --study_prefix "$SIMSV2_STUDY_PREFIX" \
  --random_trials "$SIMSV2_RANDOM_TRIALS" \
  --tpe_trials "$SIMSV2_TPE_TRIALS" \
  --simsv2-search-space aligned \
  --silver-span-loss-weight-sampling uniform \
  --silver-span-loss-weight-high "$SILVER_UNIFORM_HIGH"
