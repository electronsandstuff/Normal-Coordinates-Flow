#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"

DB="$DATA_DIR/uniform-focusing-channel-1d-2M-parts-1-pts-v1.db"
TRAIN_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/train_ufc.py"
RUN_NAME="ufc-nvp-best"

for SEED in 1 2 3 4 5; do
    OUTPUT_DIR="$RUNS_DIR/${RUN_NAME}-seed${SEED}"
    mkdir -p "$OUTPUT_DIR"
    python "$TRAIN_SCRIPT" \
        --database "$DB" \
        --result-id 1 \
        --output-dir "$OUTPUT_DIR" \
        --model-type nvp \
        --num-layers 12 \
        --nvp-num-hidden 4 \
        --hidden-size 64 \
        --batch-size 8 \
        --n-samps 4000 \
        --epochs 750 \
        --lr 0.001053136252927325 \
        --num-restarts 0 \
        --train-fraction 0.9 \
        --seed $SEED \
        --optimizer adamw \
        --device cuda \
        --mlflow-uri "$MLFLOW_URI" \
        --experiment-name "${RUN_NAME}"
done
