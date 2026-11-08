#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"

DATASET="$DATA_DIR/duffing-chain/duffing-chain-N8-5ksamp-0.1beta-0.01eps.h5"
TRAIN_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/train_duffing_chain.py"
RUN_NAME="duffing-chain-sympnet-best"

for SEED in 1 2 3 4 5 6; do
    OUTPUT_DIR="$RUNS_DIR/${RUN_NAME}-seed${SEED}"
    mkdir -p "$OUTPUT_DIR"
    python "$TRAIN_SCRIPT" \
        --dataset "$DATASET" \
        --output-dir "$OUTPUT_DIR" \
        --model-type la_sympnet \
        --num-nvp-layers 4 \
        --nvp-num-hidden 4 \
        --hidden-size 16 \
        --la-num-modules 10 \
        --normalizing-activation tanh \
        --batch-size 32 \
        --n-systems 4096 \
        --val-n-systems 5000 \
        --max-time 31.4159265 \
        --epochs 100 \
        --lr 0.0027712909714441487 \
        --lr-min-cos 2.7712909714441488e-05 \
        --num-restarts 3 \
        --train-fraction 0.9 \
        --warmup-steps 50 \
        --nan-patience 3 \
        --weight-decay 0.0 \
        --seed $SEED \
        --optimizer adamw \
        --device cuda \
        --mlflow-uri "$MLFLOW_URI" \
        --experiment-name "${RUN_NAME}"
done
