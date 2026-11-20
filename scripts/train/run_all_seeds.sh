#!/bin/bash
set -euo pipefail
CONFIG=${1:-configs/semeval.yaml}
SEEDS=(0 1 2 3 4)
GPU=${GPU:-0}

for seed in "${SEEDS[@]}"; do
    echo "=== Training seed=$seed ==="
    CUDA_VISIBLE_DEVICES=$GPU python scripts/train/train.py \
        --config "$CONFIG" \
        --seed "$seed" \
        --output_dir checkpoints
done
echo "All seeds complete."
