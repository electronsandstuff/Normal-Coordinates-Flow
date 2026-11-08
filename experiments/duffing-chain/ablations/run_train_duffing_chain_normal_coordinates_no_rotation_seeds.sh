#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"

DATASET="$DATA_DIR/duffing-chain/duffing-chain-N8-5ksamp-0.1beta-0.01eps.h5"
TRAIN_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/train_duffing_chain.py"
RUN_NAME="duffing-chain-normal-coordinates-ablation-no-rotation"

for SEED in 1 2 3 4; do
    OUTPUT_DIR="$RUNS_DIR/${RUN_NAME}-seed${SEED}"
    mkdir -p "$OUTPUT_DIR"
    python "$TRAIN_SCRIPT" \
        --dataset "$DATASET" \
        --output-dir "$OUTPUT_DIR" \
        --model-type normal_coordinates \
        --num-layers 8 \
        --num-nvp-layers 4 \
        --num-normalizing-layers 4 \
        --num-normalizing-hidden 2 \
        --nvp-num-hidden 1 \
        --hidden-size 64 \
        --rot-z-dim 8 \
        --rot-num-hidden 2 \
        --rot-hidden-size 64 \
        --rot-cond-num-hidden 2 \
        --rot-cond-hidden-size 64 \
        --la-num-modules 2 \
        --latent-aspect-ratio 3.770425930294602 \
        --rotation-ablation no-rotation \
        --batch-size 32 \
        --n-systems 4096 \
        --val-n-systems 5000 \
        --micro-batch-size 16384 \
        --max-time 31.4159265 \
        --epochs 214 \
        --lr 0.0024558062459728294 \
        --lr-min-cos 2.4558062459728295e-05 \
        --num-restarts 2 \
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
