# Normal Coordinates Flow

This repo contains the open-sourced version of the code and data used for the NeurIPS 2026 paper "Symplectic Normal Coordinates Flow for Generative Modeling of Hamiltonian Systems" (C. M. Pierce).

- `src/normal_coordinates_flow` - Flow models and dataset loaders
- `experiments` - Training scripts and the shell scripts that run them, one directory per system
- `data` - Datasets

## Setup

The datasets in `data/` are stored with [Git LFS](https://git-lfs.com). Install it and run `git lfs install` before cloning so the data downloads with the repo. If you cloned without it, run `git lfs pull` inside the repo.

Create a conda environment from the included YAML file.

```bash
conda env create -f environment.yml
conda activate normal-coordinates-flow
```

## Running the Experiments

These training scripts log to MLflow, so start the server first with the following.

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

Each script below trains its models with the hyperparameters from the paper. Trained models are written to `runs/`. The data, output and MLflow locations are set in `experiments/env.sh`.

| Paper result | Train with |
| --- | --- |
| Duffing chain NLL vs beta | `experiments/duffing-chain/beta-sweep/run_train_duffing_chain_all_beta_sweeps.sh` |
| Duffing chain phase space | `experiments/duffing-chain/train/*.sh` |
| Uniform focusing channel NLL and histograms | `experiments/uniform-focusing-channel/train/*.sh` |
| Chicane NLL and phase spaces | `experiments/chicane/train/*.sh` |
| Chicane beam statistics and cosine similarity table | `experiments/chicane/train/*.sh` |
| Duffing chain ablations | `experiments/duffing-chain/ablations/*.sh` |
| Duffing chain conditioning network width | `experiments/duffing-chain/cond-sensitivity/*.sh` |

Each run logs its validation NLL to MLflow as `val_nll`. The baseline row in the ablation and conditioning width tables is the `beta = 0.1` part of the beta sweep.

## Citation

If you use this code or data, please cite the following paper.

```bibtex
@inproceedings{pierce2026symplectic,
  title     = {Symplectic Normal Coordinates Flow for Generative Modeling of Hamiltonian Systems},
  author    = {Pierce, Christopher M.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
