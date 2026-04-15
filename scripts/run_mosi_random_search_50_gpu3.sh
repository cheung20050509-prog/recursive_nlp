#!/bin/bash
set -e

cd /root/autodl-tmp/recursive_language/ITHP/recursive_ITHP
PYTHON=/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python
MOSEI_PID=486254

printf '%s: Waiting for MOSEI PID %s on GPU3 to finish...\n' "$(date)" "$MOSEI_PID"
while kill -0 "$MOSEI_PID" 2>/dev/null; do
    sleep 60
done

printf '%s: Starting MOSI random search on GPU3...\n' "$(date)"
CUDA_VISIBLE_DEVICES=3 exec "$PYTHON" -u scripts/random_search.py \
    --dataset mosi \
    --gpu 3 \
    --n_trials 50 \
    --primary_metric acc2_no_zero \
    --output_dir search_results_acc2_no_zero_50trial
