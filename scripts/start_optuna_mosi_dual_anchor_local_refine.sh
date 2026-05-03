#!/usr/bin/env bash
# MOSI: fresh local-refine study with dual anchors (see scripts/mosi_dual_anchor_configs.json
# and scripts/optuna_local_refine_search.py). Writes to a NEW tree — does not touch log/4080_restart.
#
# Env overrides:
#   MOSI_GPU, MOSI_RANDOM_TRIALS, MOSI_LOCAL_TRIALS, MOSI_SEED, OUT_DIR, STUDY_PREFIX

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"

MOSI_GPU="${MOSI_GPU:-0}"
MOSI_RANDOM_TRIALS="${MOSI_RANDOM_TRIALS:-50}"
MOSI_LOCAL_TRIALS="${MOSI_LOCAL_TRIALS:-60}"
MOSI_SEED="${MOSI_SEED:-128}"
OUT_DIR="${OUT_DIR:-log/mosi_dual_anchor_local_refine}"
STUDY_PREFIX="${STUDY_PREFIX:-ithp_mosi_dual_anchor}"

mkdir -p "$OUT_DIR"
export CUDA_VISIBLE_DEVICES="$MOSI_GPU"

exec $PY -u scripts/optuna_local_refine_search.py \
  --dataset mosi \
  --gpu "$MOSI_GPU" \
  --output_dir "$OUT_DIR" \
  --study_prefix "$STUDY_PREFIX" \
  --random_trials "$MOSI_RANDOM_TRIALS" \
  --local_trials "$MOSI_LOCAL_TRIALS" \
  --seed "$MOSI_SEED" \
  --force_random_phase1
