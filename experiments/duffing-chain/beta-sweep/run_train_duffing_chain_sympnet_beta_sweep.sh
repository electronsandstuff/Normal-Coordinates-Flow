#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"

# Trains the tuned model on every dataset in the Duffing-chain beta sweep.
DATASET_DIR="$DATA_DIR/duffing-chain"
TRAIN_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/train_duffing_chain.py"
RUN_NAME="duffing-chain-sympnet-beta-sweep"
EXPERIMENT_NAME="normal-coords-flow-duffing-sweep"
PARENT_RUN_NAME="sympnet/seeds"

# Beta grid. These values reproduce the dataset filenames.
# Note %.6g renders 0.0 as "0", matching duffing-chain-N8-5ksamp-0beta-...
BETA_MIN=0.0
BETA_MAX=0.1
N_BETAS=8
mapfile -t BETAS < <(awk -v lo="$BETA_MIN" -v hi="$BETA_MAX" -v n="$N_BETAS" \
    'BEGIN { for (i = 0; i < n; ++i)
                 printf "%.6g\n", (n == 1) ? lo : lo + i * (hi - lo) / (n - 1) }')

dataset_for() {
    printf '%s/duffing-chain-N8-5ksamp-%sbeta-0.01eps.h5' "$DATASET_DIR" "$1"
}

# Check every dataset up front: discovering a missing file part way through a
# multi-run sweep would waste hours.
missing=0
for beta in "${BETAS[@]}"; do
    if [[ ! -f "$(dataset_for "$beta")" ]]; then
        echo "missing dataset: $(dataset_for "$beta")" >&2
        missing=1
    fi
done
if (( missing )); then
    echo "aborting before launching any runs" >&2
    exit 1
fi
echo "Sweeping beta over: ${BETAS[*]}"

# Create the parent run that groups the sweep. A parent left by an earlier sweep
# is reused, so reruns collect under the same parent instead of beside it.
PARENT_RUN_ID="$(python -m normal_coordinates_flow.parent_run start \
    "$MLFLOW_URI" "$EXPERIMENT_NAME" "$PARENT_RUN_NAME")"
echo "Parent run: $PARENT_RUN_NAME ($PARENT_RUN_ID)"

# Beta is the outer loop, so stopping the sweep early leaves whole datasets
# finished rather than many partial runs.
for BETA in "${BETAS[@]}"; do
    for SEED in {1..16}; do
        OUTPUT_DIR="$RUNS_DIR/${RUN_NAME}-beta${BETA}-seed${SEED}"
        mkdir -p "$OUTPUT_DIR"
        python "$TRAIN_SCRIPT" \
            --dataset "$(dataset_for "$BETA")" \
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
            --experiment-name "$EXPERIMENT_NAME" \
            --run-name "beta${BETA}-seed${SEED}" \
            --parent-run-id "$PARENT_RUN_ID"
    done
done

# Close the parent run.
python -m normal_coordinates_flow.parent_run finish "$MLFLOW_URI" "$PARENT_RUN_ID"
