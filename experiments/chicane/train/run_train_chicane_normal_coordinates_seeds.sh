#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"

MANIFEST="$DATA_DIR/chicane-1-pts-v8/manifest.json"
TRAIN_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/train_chicane.py"
RUN_NAME="chicane-normal-coordinates-best"

for SEED in 1 2 3 4 5; do
    OUTPUT_DIR="$RUNS_DIR/${RUN_NAME}-seed${SEED}"
    mkdir -p "$OUTPUT_DIR"
    python "$TRAIN_SCRIPT" \
        --manifest "$MANIFEST" \
        --output-dir "$OUTPUT_DIR" \
        --model-type normal_coordinates \
        --num-nvp-layers 4 \
        --num-normalizing-layers 1 \
        --num-normalizing-hidden 2 \
        --nvp-num-hidden 1 \
        --hidden-size 16 \
        --rot-z-dim 8 \
        --rot-num-hidden 3 \
        --rot-hidden-size 32 \
        --rot-cond-num-hidden 3 \
        --rot-cond-hidden-size 16 \
        --la-num-modules 8 \
        --n-particles 4096 \
        --batch-size 16 \
        --epochs 24 \
        --lr 0.009776009988883581 \
        --lr-min-cos 1.0597002062527034e-05 \
        --num-restarts 6 \
        --train-fraction 0.9 \
        --seed $SEED \
        --optimizer adamw \
        --device cuda \
        --mlflow-uri "$MLFLOW_URI" \
        --experiment-name "${RUN_NAME}"
done
