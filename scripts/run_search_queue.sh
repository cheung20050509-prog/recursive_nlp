#!/bin/bash
# Wait for MOSEI to finish, then start MOSI random search
set -e
cd /root/autodl-tmp/recursive_language/ITHP/recursive_ITHP

PYTHON=/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python

echo "$(date): Waiting for MOSEI training (PID check) to finish..."

# Wait for any python train.py on mosei to finish
while pgrep -f "train.py.*mosei" > /dev/null 2>&1; do
    sleep 60
done

echo "$(date): MOSEI done. Starting MOSI random search on GPU 3..."
CUDA_VISIBLE_DEVICES=3 $PYTHON -u scripts/random_search.py \
    --dataset mosi \
    --gpu 3 \
    --n_trials 20 \
    --n_epochs 20 \
    --output_dir search_results
