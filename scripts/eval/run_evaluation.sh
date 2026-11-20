#!/bin/bash
set -euo pipefail
DATASET=${1:-semeval}
CKPT_DIR=${2:-checkpoints}
SEEDS=(0 1 2 3 4)

for seed in "${SEEDS[@]}"; do
    python scripts/eval/evaluate.py \
        --config "configs/${DATASET}.yaml" \
        --checkpoint "${CKPT_DIR}/${DATASET}_seed${seed}_best.pt"
done
