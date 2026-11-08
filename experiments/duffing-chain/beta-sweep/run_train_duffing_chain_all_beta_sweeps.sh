#!/usr/bin/env bash

# Runs the beta sweep for every architecture back to back.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEPS=(
    run_train_duffing_chain_normal_coordinates_beta_sweep.sh
    run_train_duffing_chain_nvp_beta_sweep.sh
    run_train_duffing_chain_sympnet_beta_sweep.sh
)

for sweep in "${SWEEPS[@]}"; do
    echo "$sweep"
    "$SCRIPT_DIR/$sweep"
done
