#!/bin/bash
set -euo pipefail
MODEL=${1:-gpt-4o-2024-05-13}

for ds in semeval isarcasm sarc ptacek; do
    echo "Caching $ds..."
    for split in train val test; do
        python scripts/cache/cache_llm_features.py \
            --data_path "data/${ds}/splits/${split}.tsv" \
            --output_dir "cached_features/${ds}/gpt4o" \
            --model "$MODEL"
    done
done
echo "Caching complete."
