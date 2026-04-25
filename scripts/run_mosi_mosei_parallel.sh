#!/usr/bin/env bash
# Run MOSI and MOSEI training at the same time (same GPU by default).
# Single RTX 4080-class ~32GB: two jobs ~5–6GB each is usually OK; if OOM, set MOSI_GPU / MOSEI_GPU to different cards or lower --train_batch_size.
# Do not pipe this script to `head` — breaks child processes (SIGPIPE).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/recursive_nlp/deberta-v3-base}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/parallel_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

MOSI_GPU="${MOSI_GPU:-0}"
MOSEI_GPU="${MOSEI_GPU:-0}"

cat >"$RUN_DIR/README.txt" <<EOF
MOSI  -> GPU $MOSI_GPU  log: mosi_train.log
MOSEI -> GPU $MOSEI_GPU  log: mosei_train.log
Started: $(date -Is)
EOF

(
  export CUDA_VISIBLE_DEVICES="$MOSI_GPU"
  cd "$ROOT"
  exec $PY -u train.py --dataset mosi --model "$MODEL" --n_epochs 20 \
    --silver_span_cache datasets/mosi_silver_spans.pkl --merge_trace_samples 0 \
    --selection_metric mae --early_stopping_patience 0 \
    --learning_rate 2e-5 --p_beta 4 --p_gamma 16 \
    --B0_dim 64 --B1_dim 128 --max_recursion_depth 4 --halting_threshold 0.02 \
    --dropout_prob 0.5 --silver_span_loss_weight 0.2 --syntax_temperature 0.5
) >"$RUN_DIR/mosi_train.log" 2>&1 &
echo $! >"$RUN_DIR/mosi.pid"

(
  export CUDA_VISIBLE_DEVICES="$MOSEI_GPU"
  cd "$ROOT"
  exec $PY -u train.py --dataset mosei --model "$MODEL" --n_epochs 10 \
    --silver_span_cache datasets/mosei_silver_spans.pkl --merge_trace_samples 0 \
    --selection_metric mae --early_stopping_patience 3 \
    --learning_rate 5e-6 --p_beta 4 --p_gamma 16 \
    --B0_dim 64 --B1_dim 64 --max_recursion_depth 4 --halting_threshold 0.0285 \
    --dropout_prob 0.3 --silver_span_loss_weight 0.2 --syntax_temperature 0.5
) >"$RUN_DIR/mosei_train.log" 2>&1 &
echo $! >"$RUN_DIR/mosei.pid"

echo "RUN_DIR=$RUN_DIR"
echo "MOSI  pid=$(cat "$RUN_DIR/mosi.pid")  GPU=$MOSI_GPU"
echo "MOSEI pid=$(cat "$RUN_DIR/mosei.pid")  GPU=$MOSEI_GPU"
