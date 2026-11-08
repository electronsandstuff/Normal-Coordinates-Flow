#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"

MANIFEST="$DATA_DIR/chicane-1-pts-v8/manifest.json"
TRAIN_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/train_chicane.py"
RUN_NAME="chicane-sympnet-best"

for SEED in 1 2 3 4 5; do
    OUTPUT_DIR="$RUNS_DIR/${RUN_NAME}-seed${SEED}"
    mkdir -p "$OUTPUT_DIR"
    python "$TRAIN_SCRIPT" \
        --manifest "$MANIFEST" \
        --output-dir "$OUTPUT_DIR" \
        --model-type la_sympnet \
        --la-num-modules 10 \
        --num-nvp-layers 4 \
        --nvp-num-hidden 1 \
        --hidden-size 32 \
        --n-particles 4096 \
        --batch-size 16 \
        --epochs 3050 \
        --lr 0.0011731442697970715 \
        --num-restarts 0 \
        --train-fraction 0.9 \
        --seed $SEED \
        --optimizer adamw \
        --device cuda \
        --mlflow-uri "$MLFLOW_URI" \
        --experiment-name "${RUN_NAME}"
done
