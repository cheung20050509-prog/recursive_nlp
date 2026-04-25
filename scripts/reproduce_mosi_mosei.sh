#!/usr/bin/env bash
# Reproduce MOSI (local-refine best, study trial 31) then MOSEI (full TPE best, trial 117).
# Run from anywhere; uses ITHP5090 env and project-local DeBERTa + datasets.
# Do not pipe this script to `head`/`tail -f` on stdout — that closes the pipe and kills training (SIGPIPE).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/recursive_nlp/deberta-v3-base}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/reproduce_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== ROOT=$ROOT MODEL=$MODEL RUN_DIR=$RUN_DIR ===" | tee "$RUN_DIR/summary.log"

echo "=== MOSI (n_epochs=20, early_stop=0, selection=mae) ===" | tee -a "$RUN_DIR/summary.log"
$PY -u train.py --dataset mosi --model "$MODEL" --n_epochs 20 \
  --silver_span_cache datasets/mosi_silver_spans.pkl --merge_trace_samples 0 \
  --selection_metric mae --early_stopping_patience 0 \
  --learning_rate 2e-5 --p_beta 4 --p_gamma 16 \
  --B0_dim 64 --B1_dim 128 --max_recursion_depth 4 --halting_threshold 0.02 \
  --dropout_prob 0.5 --silver_span_loss_weight 0.2 --syntax_temperature 0.5 \
  2>&1 | tee "$RUN_DIR/mosi_train.log"

echo "=== MOSEI (n_epochs=10, early_stop=3, selection=mae) ===" | tee -a "$RUN_DIR/summary.log"
$PY -u train.py --dataset mosei --model "$MODEL" --n_epochs 10 \
  --silver_span_cache datasets/mosei_silver_spans.pkl --merge_trace_samples 0 \
  --selection_metric mae --early_stopping_patience 3 \
  --learning_rate 5e-6 --p_beta 4 --p_gamma 16 \
  --B0_dim 64 --B1_dim 64 --max_recursion_depth 4 --halting_threshold 0.0285 \
  --dropout_prob 0.3 --silver_span_loss_weight 0.2 --syntax_temperature 0.5 \
  2>&1 | tee "$RUN_DIR/mosei_train.log"

echo "=== done ===" | tee -a "$RUN_DIR/summary.log"
