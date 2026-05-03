#!/usr/bin/env bash
# NEW MUStARD HKT study: uniform syntax_loss_weight in [0, high]. New output dir + prefix.
# Optional small-weight categorical study: use start_mustard_optuna_resume_tpe.sh instead.
#
# Env: MUSTARD_GPU, HKT_OUT, HKT_STUDY_PREFIX, MUSTARD_RANDOM_TRIALS, MUSTARD_TPE_TRIALS,
#      SYNTAX_UNIFORM_HIGH, PY

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"

MUSTARD_GPU="${MUSTARD_GPU:-0}"
HKT_OUT="${HKT_OUT:-log/mustard_uniform_syntax}"
HKT_STUDY_PREFIX="${HKT_STUDY_PREFIX:-ithp_hkt_syntax_uniform}"
MUSTARD_RANDOM_TRIALS="${MUSTARD_RANDOM_TRIALS:-20}"
MUSTARD_TPE_TRIALS="${MUSTARD_TPE_TRIALS:-60}"
SYNTAX_UNIFORM_HIGH="${SYNTAX_UNIFORM_HIGH:-0.12}"

mkdir -p "$HKT_OUT"
export CUDA_VISIBLE_DEVICES="$MUSTARD_GPU"
exec "$PY" -u scripts/optuna_hkt_search.py \
  --dataset mustard \
  --gpu "$MUSTARD_GPU" \
  --output_dir "$HKT_OUT" \
  --study_prefix "$HKT_STUDY_PREFIX" \
  --random_trials "$MUSTARD_RANDOM_TRIALS" \
  --tpe_trials "$MUSTARD_TPE_TRIALS" \
  --syntax-loss-weight-sampling uniform \
  --syntax-loss-weight-high "$SYNTAX_UNIFORM_HIGH"
