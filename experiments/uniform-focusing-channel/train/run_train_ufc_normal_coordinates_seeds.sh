#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"

DB="$DATA_DIR/uniform-focusing-channel-1d-2M-parts-1-pts-v1.db"
TRAIN_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/train_ufc.py"
RUN_NAME="ufc-normal-coordinates-best"

for SEED in 1 2 3 4 5; do
    OUTPUT_DIR="$RUNS_DIR/${RUN_NAME}-seed${SEED}"
    mkdir -p "$OUTPUT_DIR"
    python "$TRAIN_SCRIPT" \
        --database "$DB" \
        --result-id 1 \
        --output-dir "$OUTPUT_DIR" \
        --model-type normal_coordinates \
        --num-nvp-layers 4 \
        --num-normalizing-layers 4 \
        --num-normalizing-hidden 3 \
        --nvp-num-hidden 3 \
        --hidden-size 32 \
        --rot-z-dim 8 \
        --rot-num-hidden 4 \
        --rot-hidden-size 32 \
        --rot-cond-num-hidden 4 \
        --rot-cond-hidden-size 64 \
        --la-num-modules 3 \
        --batch-size 16 \
        --n-samps 5000 \
        --epochs 50 \
        --lr 0.00025 \
        --lr-min-cos 1e-06 \
        --num-restarts 3 \
        --train-fraction 0.9 \
        --seed $SEED \
        --optimizer adamw \
        --device cuda \
        --mlflow-uri "$MLFLOW_URI" \
        --experiment-name "${RUN_NAME}"
done
