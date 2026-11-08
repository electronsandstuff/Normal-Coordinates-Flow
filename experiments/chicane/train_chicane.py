import argparse
import json
import math
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.optim
from tqdm import tqdm

from normal_coordinates_flow.datasets.chicane import ChicaneDatasetGPU
from normal_coordinates_flow.models.normalizing_flow import (
    ConditionedLASympNetFlow,
    ConditionedRealNVPFlow,
    ConditionedNormalCoordinatesFlow,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MANIFEST_PATH = str(REPO_ROOT / "data" / "chicane-1-pts-v8" / "manifest.json")

N_COORD = 6  # (x, y, t, px, py, pz)
N_FEAT = 1  # (s_position)


def compute_scaling_stats(dataset, indices, batch_size, device, cond_transform=None):
    x_count = 0
    x_sum = torch.zeros(N_COORD, dtype=torch.float64, device=device)
    x_sq_sum = torch.zeros(N_COORD, dtype=torch.float64, device=device)
    c_count = 0
    c_sum = None
    c_sq_sum = None

    for particles, s_pos in tqdm(
        dataset.iter_batches(batch_size, shuffle=False, indices=indices),
        desc="Computing scaling stats",
    ):
        n_batch, n_samp, _ = particles.shape
        x_flat = particles.reshape(n_batch * n_samp, N_COORD).double()
        x_count += x_flat.shape[0]
        x_sum += x_flat.sum(0)
        x_sq_sum += (x_flat**2).sum(0)

        c = s_pos.double()  # (n_batch, 1)
        c = cond_transform(c) if cond_transform is not None else c
        if c_sum is None:
            c_sum = torch.zeros(c.shape[1], dtype=torch.float64, device=device)
            c_sq_sum = torch.zeros(c.shape[1], dtype=torch.float64, device=device)
        c_count += c.shape[0]
        c_sum += c.sum(0)
        c_sq_sum += (c**2).sum(0)

    x_mean = (x_sum / x_count).float().cpu().numpy()
    x_std = ((x_sq_sum / x_count - (x_sum / x_count) ** 2).sqrt()).float().cpu().numpy()
    cond_mean = (c_sum / c_count).float().cpu().numpy()
    cond_std = (
        ((c_sq_sum / c_count - (c_sum / c_count) ** 2).sqrt()).float().cpu().numpy()
    )

    x_std = np.clip(x_std, 1e-32, None)
    cond_std = np.clip(cond_std, 1e-32, None)

    return x_mean, x_std, cond_mean, cond_std


def _param_grad_norms(model) -> dict[str, float]:
    """Return L2 gradient norm for each named parameter that has a gradient."""
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.detach().norm(2).item()
    return norms


def _subsample_particles(particles: torch.Tensor, n_particles: int) -> torch.Tensor:
    """Randomly select up to n_particles per screen along dim 1."""
    n_samp = particles.shape[1]
    k = min(n_particles, n_samp)
    idx = torch.randperm(n_samp, device=particles.device)[:k]
    return particles[:, idx, :]


def train_epoch(
    model,
    dataset,
    train_idx,
    batch_size,
    optimizer,
    device,
    epoch,
    n_particles: int = 4096,
    log_grad_norm: bool = False,
    global_step: int = 0,
    warmup_steps: int = 0,
):
    model.train()
    total_nll = 0.0
    total_samples = 0
    n_batches = 0
    n_skipped = 0
    grad_norm_sums: dict[str, float] = {}

    # The scheduler sets the learning rate once per epoch; warmup scales it
    # within the epoch and becomes a no-op once global_step passes warmup_steps.
    sched_lr = optimizer.param_groups[0]["lr"]

    perm = train_idx[torch.randperm(len(train_idx), device=train_idx.device)]

    pbar = tqdm(range(0, len(perm), batch_size), desc=f"Train epoch {epoch}")
    for start in pbar:
        batch_idx = perm[start : start + batch_size]
        particles = dataset.particles[batch_idx]  # (n_batch, n_particles, 6)
        s_pos = dataset.s_positions[batch_idx].unsqueeze(1)  # (n_batch, 1)

        n_batch = particles.shape[0]
        particles = _subsample_particles(particles, n_particles)
        n_sub = particles.shape[1]
        x = particles.reshape(n_batch * n_sub, N_COORD)
        cond = (
            s_pos.unsqueeze(1)
            .expand(n_batch, n_sub, N_FEAT)
            .reshape(n_batch * n_sub, N_FEAT)
        )

        if global_step < warmup_steps:
            for group in optimizer.param_groups:
                group["lr"] = sched_lr * (global_step + 1) / warmup_steps

        optimizer.zero_grad()
        log_p = model.log_prob(x, cond)
        loss = -log_p.mean()
        loss.backward()

        postfix: dict[str, str] = {"nll": f"{loss.item():.4f}"}
        if log_grad_norm:
            batch_norms = _param_grad_norms(model)
            for name, v in batch_norms.items():
                grad_norm_sums[name] = grad_norm_sums.get(name, 0.0) + v
            total_grad_norm = sum(v**2 for v in batch_norms.values()) ** 0.5
            postfix["grad"] = f"{total_grad_norm:.3f}"

        # A non-finite total norm makes the clip coefficient non-finite, which
        # would propagate NaN into every parameter and destroy the model.
        clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        global_step += 1
        if not torch.isfinite(clipped_norm):
            optimizer.zero_grad(set_to_none=True)
            n_skipped += 1
            continue
        optimizer.step()

        total_nll += loss.item() * x.shape[0]
        total_samples += x.shape[0]
        n_batches += 1
        pbar.set_postfix(**postfix)

    if warmup_steps:
        for group in optimizer.param_groups:
            group["lr"] = sched_lr

    avg_grad_norms = (
        {name: v / n_batches for name, v in grad_norm_sums.items()}
        if log_grad_norm
        else {}
    )
    epoch_nll = total_nll / total_samples if total_samples else float("nan")
    return epoch_nll, avg_grad_norms, n_skipped, global_step


def _rms_stats(samples: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-screen RMS and mean for each of the 6 coordinates."""
    mean = samples.mean(dim=1)  # (n_batch, 6)
    rms = samples.std(dim=1)  # (n_batch, 6)
    return {"mean": mean, "rms": rms}


def validate(model, dataset, test_idx, batch_size, device, n_particles: int = 4096):
    model.eval()
    total_nll = 0.0
    total_samples = 0

    rms_true_list, rms_pred_list = [], []

    with torch.no_grad():
        for particles, s_pos in tqdm(
            dataset.iter_batches(batch_size, shuffle=False, indices=test_idx),
            desc="Validation",
        ):
            n_batch, n_samp, _ = particles.shape
            particles_sub = _subsample_particles(particles, n_particles)
            k = particles_sub.shape[1]
            x = particles_sub.reshape(n_batch * k, N_COORD)
            cond = (
                s_pos.unsqueeze(1)
                .expand(n_batch, k, N_FEAT)
                .reshape(n_batch * k, N_FEAT)
            )

            log_p = model.log_prob(x, cond)
            loss = -log_p.mean()
            total_nll += loss.item() * x.shape[0]
            total_samples += x.shape[0]

            pred_coords = model.sample(s_pos, n_per_cond=n_particles)  # (B, S, N_COORD)

            rms_true_list.append(_rms_stats(particles_sub)["rms"])
            rms_pred_list.append(_rms_stats(pred_coords)["rms"])

    rms_true = torch.cat(rms_true_list)  # (n_test, 6)
    rms_pred = torch.cat(rms_pred_list)

    coord_names = ["x", "y", "t", "px", "py", "pz"]
    rms_mape = {}
    for i, name in enumerate(coord_names):
        mape = (
            (rms_pred[:, i] - rms_true[:, i]).abs()
            / rms_true[:, i].abs().clamp(min=1e-20)
        ).mean().item() * 100
        rms_mape[f"rms_mape_{name}"] = mape

    return {"nll": total_nll / total_samples, **rms_mape}


def _stratified_split(
    c: np.ndarray, n_test: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/test split by quantile bins of conditioning variable c."""
    n = len(c)
    rng = np.random.default_rng(seed)
    sorted_idx = np.argsort(c, kind="stable")
    bin_edges = np.round(np.linspace(0, n, n_test + 1)).astype(int)
    test_sorted_pos = np.array(
        [rng.integers(bin_edges[i], bin_edges[i + 1]) for i in range(n_test)]
    )
    test_idx = sorted_idx[test_sorted_pos]
    mask = np.ones(n, dtype=bool)
    mask[test_idx] = False
    train_idx = np.where(mask)[0]
    return train_idx, test_idx


def train(
    manifest: str,
    output_dir: str | None = None,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 1e-3,
    lr_min_cos: float | None = None,
    num_restarts: int = 0,
    train_fraction: float = 0.8,
    seed: int = 42,
    model_type: str = "nvp",
    num_layers: int = 8,
    num_nvp_layers: int = 4,
    num_normalizing_layers: int = 1,
    num_normalizing_hidden: int = 3,
    nvp_num_hidden: int = 2,
    hidden_size: int = 128,
    n_particles: int = 4096,
    val_n_particles: int | None = None,
    rot_z_dim: int = 64,
    rot_num_hidden: int | None = None,
    rot_hidden_size: int | None = None,
    rot_cond_num_hidden: int = 1,
    rot_cond_hidden_size: int | None = None,
    la_num_modules: int = 3,
    latent_aspect_ratio: float = 1.0,
    normalizing_activation: str = "tanh",
    log_grad_norm: bool = False,
    warmup_steps: int = 50,
    nan_patience: int = 3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    on_epoch_end: Callable[[dict], None] | None = None,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
):
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="chicane_nvp_")
        print(f"Using temp output directory: {output_dir}")

    torch.manual_seed(seed)

    print(f"Manifest: {manifest}")
    print(f"Device:   {device}")

    _val_n_particles = val_n_particles if val_n_particles is not None else n_particles

    mlflow.log_params(
        {
            "manifest": manifest,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "lr_min_cos": lr_min_cos,
            "num_restarts": num_restarts,
            "train_fraction": train_fraction,
            "seed": seed,
            "model_type": model_type,
            "num_layers": num_layers,
            "num_nvp_layers": num_nvp_layers,
            "num_normalizing_layers": num_normalizing_layers,
            "num_normalizing_hidden": num_normalizing_hidden,
            "nvp_num_hidden": nvp_num_hidden,
            "hidden_size": hidden_size,
            "n_particles": n_particles,
            "val_n_particles": _val_n_particles,
            "rot_z_dim": rot_z_dim,
            "rot_num_hidden": rot_num_hidden,
            "rot_hidden_size": rot_hidden_size,
            "rot_cond_num_hidden": rot_cond_num_hidden,
            "rot_cond_hidden_size": rot_cond_hidden_size,
            "la_num_modules": la_num_modules,
            "latent_aspect_ratio": latent_aspect_ratio,
            "device": device,
            "optimizer": optimizer,
            "weight_decay": weight_decay,
        }
    )

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    dataset = ChicaneDatasetGPU.from_manifest(manifest, progressbar=tqdm, device=device)

    # Stratified train/test split by quantile bins of the conditioning variable
    n = len(dataset)
    c_np = dataset.s_positions.cpu().numpy()
    n_test = n - int(n * train_fraction)
    _train_np, _test_np = _stratified_split(c_np, n_test, seed)
    train_idx = torch.from_numpy(_train_np).long().to(device)
    test_idx = torch.from_numpy(_test_np).long().to(device)
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    split_path = os.path.join(output_dir, "split_indices.json")
    with open(split_path, "w") as f:
        json.dump(
            {
                "train": _train_np.tolist(),
                "test": _test_np.tolist(),
            },
            f,
        )

    print("Computing scaling statistics...")
    x_mean, x_std, cond_mean, cond_std = compute_scaling_stats(
        dataset, train_idx, batch_size, device
    )
    print(f"x_mean:    {x_mean}")
    print(f"x_std:     {x_std}")
    print(f"cond_mean: {cond_mean}")
    print(f"cond_std:  {cond_std}")

    stats_path = os.path.join(output_dir, "scaling_stats.json")
    with open(stats_path, "w") as f:
        json.dump(
            {
                "x_mean": x_mean.tolist(),
                "x_std": x_std.tolist(),
                "cond_mean": cond_mean.tolist(),
                "cond_std": cond_std.tolist(),
            },
            f,
            indent=2,
        )
    print(f"Scaling stats saved to {stats_path}")

    if model_type == "normal_coordinates":
        _rot_h = rot_hidden_size if rot_hidden_size is not None else hidden_size
        _rot_n = (
            rot_num_hidden if rot_num_hidden is not None else num_normalizing_hidden
        )
        _cond_h = (
            rot_cond_hidden_size if rot_cond_hidden_size is not None else hidden_size
        )
        _model_config = {
            "data_dim": N_COORD,
            "num_nvp_layers": num_nvp_layers,
            "num_normalizing_layers": num_normalizing_layers,
            "num_normalizing_hidden": num_normalizing_hidden,
            "nvp_num_hidden": nvp_num_hidden,
            "cond_dim": N_FEAT,
            "hidden_size": hidden_size,
            "rot_z_dim": rot_z_dim,
            "rot_num_hidden": _rot_n,
            "rot_hidden_size": _rot_h,
            "rot_cond_hidden_dims": [_cond_h] * rot_cond_num_hidden,
            "la_num_modules": la_num_modules,
            "latent_aspect_ratio": latent_aspect_ratio,
        }
        model = ConditionedNormalCoordinatesFlow(
            **_model_config,
            x_mean=x_mean,
            x_std=x_std,
            cond_mean=cond_mean,
            cond_std=cond_std,
        ).to(device)
    elif model_type == "nvp":
        _model_config = None
        model = ConditionedRealNVPFlow(
            data_dim=N_COORD,
            cond_dim=N_FEAT,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_hidden=nvp_num_hidden,
            x_mean=x_mean,
            x_std=x_std,
            cond_mean=cond_mean,
            cond_std=cond_std,
        ).to(device)
    elif model_type == "la_sympnet":
        _model_config = {
            "data_dim": N_COORD,
            "cond_dim": N_FEAT,
            "num_modules": la_num_modules,
            "hidden_size": hidden_size,
            "num_hidden": nvp_num_hidden,
            "activation": normalizing_activation,
            "num_nvp_layers": num_nvp_layers,
        }
        model = ConditionedLASympNetFlow(
            **_model_config,
            x_mean=x_mean,
            x_std=x_std,
            cond_mean=cond_mean,
            cond_std=cond_std,
        ).to(device)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from: nvp, normal_coordinates, la_sympnet"
        )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    print("\nBaseline validation (untrained model)")
    baseline_metrics = validate(
        model, dataset, test_idx, batch_size, device, n_particles=_val_n_particles
    )
    print(f"Baseline NLL: {baseline_metrics['nll']:.4f} nats/sample")

    if optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    # total epochs: T_0 initial cycle + num_restarts doublings, always ending at eta_min
    if lr_min_cos is not None:
        total_epochs = epochs * (2 ** (num_restarts + 1) - 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=epochs, T_mult=2, eta_min=lr_min_cos
        )
        print(
            f"Cosine annealing warm restarts: lr {lr} -> {lr_min_cos}, "
            f"T_0={epochs}, {num_restarts} restarts, total {total_epochs} epochs"
        )
    else:
        total_epochs = epochs
        scheduler = None

    best_val_nll = float("inf")
    global_step = 0
    n_nan_epochs = 0

    for epoch in range(1, total_epochs + 1):
        print(f"\nEpoch {epoch}/{total_epochs}")
        epoch_start_t = time.perf_counter()

        t0 = time.time()
        train_nll, avg_grad_norms, n_skipped, global_step = train_epoch(
            model,
            dataset,
            train_idx,
            batch_size,
            opt,
            device,
            epoch,
            n_particles=n_particles,
            log_grad_norm=log_grad_norm,
            global_step=global_step,
            warmup_steps=warmup_steps,
        )
        train_time = time.time() - t0

        t0 = time.time()
        val_metrics = validate(
            model, dataset, test_idx, batch_size, device, n_particles=_val_n_particles
        )
        val_time = time.time() - t0

        current_lr = opt.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        grad_norm_metrics = {f"grad_norm/{k}": v for k, v in avg_grad_norms.items()}
        log_metrics = {
            "train_nll": train_nll,
            "val_nll": val_metrics["nll"],
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "nll"},
            "epoch_time": time.perf_counter() - epoch_start_t,
            "train_time": train_time,
            "val_time": val_time,
            "lr": current_lr,
            "n_skipped_steps": n_skipped,
            **grad_norm_metrics,
        }

        print(f"Train NLL: {train_nll:.4f} nats/sample  [{train_time:.1f}s]")
        print(f"Val NLL:   {log_metrics['val_nll']:.4f} nats/sample  [{val_time:.1f}s]")
        if n_skipped:
            print(f"Skipped {n_skipped} steps with non-finite gradients")

        mlflow.log_metrics(log_metrics, step=epoch)

        diverged = not math.isfinite(train_nll) or not math.isfinite(
            log_metrics["val_nll"]
        )
        n_nan_epochs = n_nan_epochs + 1 if diverged else 0

        if on_epoch_end is not None:
            # Report inf rather than NaN so the search records a usable (very
            # bad) objective; Optuna treats a NaN result as a failed trial and
            # learns nothing from it.
            reported = dict(log_metrics)
            if diverged:
                reported["val_nll"] = float("inf")
            on_epoch_end({**reported, "epoch": epoch, "n_params": n_params})

        if n_nan_epochs >= nan_patience:
            print(
                f"\nDiverged: {n_nan_epochs} consecutive non-finite epochs. "
                f"Stopping at epoch {epoch}/{total_epochs}."
            )
            break

        if log_metrics["val_nll"] < best_val_nll:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "model_config": _model_config,
                **log_metrics,
            }

            best_val_nll = log_metrics["val_nll"]
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"New best model saved to {best_path}")

    print(f"\nTraining complete. Best val NLL: {best_val_nll:.4f} nats/sample")

    mlflow.log_metric("best_val_nll", best_val_nll)
    best_path = os.path.join(checkpoint_dir, "best_model.pt")
    if os.path.exists(best_path):
        mlflow.log_artifact(best_path)
    else:
        print("No epoch improved on inf; no checkpoint to log.")
    mlflow.log_artifact(stats_path)
    mlflow.log_artifact(split_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train a conditional Real-NVP flow on the chicane particle dataset"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Path to manifest.json (default: {DEFAULT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and logs (default: temp directory)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of screens per batch (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Epochs in the first cosine cycle T_0 (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--lr-min-cos",
        type=float,
        default=None,
        help="If set, enables cosine annealing warm restarts decaying to this value",
    )
    parser.add_argument(
        "--num-restarts",
        type=int,
        default=0,
        help="Number of warm restarts (period doubles each time); total epochs = T_0*(2^(N+1)-1) (default: 0)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of screens for training (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="nvp",
        choices=["nvp", "normal_coordinates", "la_sympnet"],
        help="Flow architecture: 'nvp', 'normal_coordinates', or 'la_sympnet' (default: nvp)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=8,
        help="Number of NVP coupling layers, used when --model-type=nvp (default: 8)",
    )
    parser.add_argument(
        "--num-nvp-layers",
        type=int,
        default=4,
        help="Number of unconditioned NVP coupling layers, used when --model-type=normal_coordinates (default: 4)",
    )
    parser.add_argument(
        "--num-normalizing-layers",
        type=int,
        default=1,
        help="Number of conditioned symplectic blocks, used when --model-type=normal_coordinates (default: 1)",
    )
    parser.add_argument(
        "--num-normalizing-hidden",
        type=int,
        default=3,
        help="MLP depth inside each symplectic block, used when --model-type=normal_coordinates (default: 3)",
    )
    parser.add_argument(
        "--nvp-num-hidden",
        type=int,
        default=2,
        help="Number of hidden layers inside each NVP coupling layer MLP (default: 2)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Width of each coupling-layer MLP (default: 128)",
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        default=4096,
        help="Number of particles per screen sampled each iteration (default: 4096)",
    )
    parser.add_argument(
        "--val-n-particles",
        type=int,
        default=None,
        help="Number of particles per screen sampled during validation (default: same as --n-particles)",
    )
    parser.add_argument(
        "--rot-z-dim",
        type=int,
        default=64,
        help="Latent dim of the rotation conditioning MLP (default: 64)",
    )
    parser.add_argument(
        "--rot-num-hidden",
        type=int,
        default=None,
        help="Number of hidden layers in each GradNet rotation MLP (default: num-normalizing-hidden)",
    )
    parser.add_argument(
        "--rot-hidden-size",
        type=int,
        default=None,
        help="Width of each GradNet rotation MLP layer (default: hidden-size)",
    )
    parser.add_argument(
        "--rot-cond-num-hidden",
        type=int,
        default=1,
        help="Number of hidden layers in the rotation conditioning MLP (default: 1)",
    )
    parser.add_argument(
        "--rot-cond-hidden-size",
        type=int,
        default=None,
        help="Width of the rotation conditioning MLP (default: hidden-size)",
    )
    parser.add_argument(
        "--latent-aspect-ratio",
        type=float,
        default=1.0,
        help="Latent sigma_p/sigma_q per phase plane, sigma_q*sigma_p=1, used when --model-type=normal_coordinates (default: 1.0)",
    )
    parser.add_argument(
        "--la-num-modules",
        type=int,
        default=3,
        help="Number of (linear + Q + P) blocks in each LASympNet (default: 3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device (default: cuda if available)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "adamw"],
        help="Optimizer (default: adam)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="L2 regularization coefficient for AdamW (default: 0.0)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
        help="Linear learning-rate warmup over this many optimizer steps; 0 disables (default: 50)",
    )
    parser.add_argument(
        "--nan-patience",
        type=int,
        default=3,
        help="Stop after this many consecutive non-finite epochs (default: 3)",
    )
    parser.add_argument(
        "--log-grad-norm",
        action="store_true",
        default=False,
        help="Compute and log per-parameter gradient norms to MLflow (default: off)",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="chicane-nvp",
        help="MLflow experiment name (default: chicane-nvp)",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():
        train(
            manifest=os.path.expandvars(args.manifest),
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            lr_min_cos=args.lr_min_cos,
            num_restarts=args.num_restarts,
            train_fraction=args.train_fraction,
            seed=args.seed,
            model_type=args.model_type,
            num_layers=args.num_layers,
            num_nvp_layers=args.num_nvp_layers,
            num_normalizing_layers=args.num_normalizing_layers,
            num_normalizing_hidden=args.num_normalizing_hidden,
            nvp_num_hidden=args.nvp_num_hidden,
            hidden_size=args.hidden_size,
            n_particles=args.n_particles,
            val_n_particles=args.val_n_particles,
            rot_z_dim=args.rot_z_dim,
            rot_num_hidden=args.rot_num_hidden,
            rot_hidden_size=args.rot_hidden_size,
            rot_cond_num_hidden=args.rot_cond_num_hidden,
            rot_cond_hidden_size=args.rot_cond_hidden_size,
            la_num_modules=args.la_num_modules,
            latent_aspect_ratio=args.latent_aspect_ratio,
            device=args.device,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            log_grad_norm=args.log_grad_norm,
            warmup_steps=args.warmup_steps,
            nan_patience=args.nan_patience,
        )


if __name__ == "__main__":
    main()
