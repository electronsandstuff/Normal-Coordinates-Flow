#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"

MANIFEST="$DATA_DIR/chicane-1-pts-v8/manifest.json"
TRAIN_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/train_chicane.py"
RUN_NAME="chicane-nvp-best"

for SEED in 1 2 3 4 5; do
    OUTPUT_DIR="$RUNS_DIR/${RUN_NAME}-seed${SEED}"
    mkdir -p "$OUTPUT_DIR"
    python "$TRAIN_SCRIPT" \
        --manifest "$MANIFEST" \
        --output-dir "$OUTPUT_DIR" \
        --model-type nvp \
        --num-layers 10 \
        --nvp-num-hidden 2 \
        --hidden-size 32 \
        --n-particles 4096 \
        --batch-size 16 \
        --epochs 48 \
        --lr 0.002416532338835846 \
        --lr-min-cos 2.416532338835846e-05 \
        --num-restarts 5 \
        --train-fraction 0.9 \
        --seed $SEED \
        --optimizer adamw \
        --device cuda \
        --mlflow-uri "$MLFLOW_URI" \
        --experiment-name "${RUN_NAME}"
done
