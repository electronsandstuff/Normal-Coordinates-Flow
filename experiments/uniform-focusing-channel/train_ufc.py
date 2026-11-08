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

from normal_coordinates_flow.datasets.ufc_1d import PhaseSpaceDistGPUSampler
from normal_coordinates_flow.models.normalizing_flow import (
    ConditionedLASympNetFlow,
    ConditionedRealNVPFlow,
    ConditionedNormalCoordinatesFlow,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DB_PATH = str(
    REPO_ROOT / "data" / "uniform-focusing-channel-1d-2M-parts-1-pts-v1.db"
)

N_COORD = 2  # (x, xp)
N_FEAT = 1  # (s_position only)


def compute_scaling_stats(dataset, indices, batch_size, device, cond_transform=None):
    x_count = 0
    x_sum = torch.zeros(N_COORD, dtype=torch.float64, device=device)
    x_sq_sum = torch.zeros(N_COORD, dtype=torch.float64, device=device)
    c_count = 0
    c_sum = None
    c_sq_sum = None

    for coords, feat in tqdm(
        dataset.iter_batches(batch_size, shuffle=False, indices=indices),
        desc="Computing scaling stats",
    ):
        n_batch, n_samp, _ = coords.shape
        x_flat = coords.reshape(n_batch * n_samp, N_COORD).double()
        x_count += x_flat.shape[0]
        x_sum += x_flat.sum(0)
        x_sq_sum += (x_flat**2).sum(0)

        c = (cond_transform(feat) if cond_transform is not None else feat).double()
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

    x_std = np.clip(x_std, 1e-8, None)
    cond_std = np.clip(cond_std, 1e-8, None)

    return x_mean, x_std, cond_mean, cond_std


def _param_grad_norms(model) -> dict[str, float]:
    """Return L2 gradient norm for each named parameter that has a gradient."""
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.detach().norm(2).item()
    return norms


def train_epoch(
    model,
    dataset,
    train_idx,
    batch_size,
    optimizer,
    device,
    epoch,
    log_grad_norm: bool = False,
    global_step: int = 0,
    warmup_steps: int = 0,
):
    model.train()
    total_nll = 0.0
    total_samples = 0
    grad_norm_sums: dict[str, float] = {}
    n_batches = 0
    n_skipped = 0

    # The scheduler sets the learning rate once per epoch; warmup scales it
    # within the epoch and becomes a no-op once global_step passes warmup_steps.
    sched_lr = optimizer.param_groups[0]["lr"]

    pbar = tqdm(
        dataset.iter_batches(batch_size, indices=train_idx),
        desc=f"Train epoch {epoch}",
    )
    for coords, feat in pbar:
        n_batch, n_samp, _ = coords.shape
        x = coords.reshape(n_batch * n_samp, N_COORD)
        cond = (
            feat.unsqueeze(1)
            .expand(n_batch, n_samp, -1)
            .reshape(n_batch * n_samp, N_FEAT)
        )

        if global_step < warmup_steps:
            for group in optimizer.param_groups:
                group["lr"] = sched_lr * (global_step + 1) / warmup_steps

        optimizer.zero_grad()
        log_p = model.log_prob(x, cond)
        loss = -log_p.mean()
        loss.backward()

        postfix = {"nll": f"{loss.item():.4f}"}
        if log_grad_norm:
            batch_norms = _param_grad_norms(model)
            for k, v in batch_norms.items():
                grad_norm_sums[k] = grad_norm_sums.get(k, 0.0) + v
            n_batches += 1
            total_grad_norm = sum(v**2 for v in batch_norms.values()) ** 0.5
            postfix["grad"] = f"{total_grad_norm:.3f}"

        # Skip updates with non-finite gradients; a single bad batch would
        # otherwise put NaN into every parameter permanently.
        grad_norm = torch.nn.utils.get_total_norm(
            [p.grad for p in model.parameters() if p.grad is not None]
        )
        global_step += 1
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            n_skipped += 1
            continue
        optimizer.step()

        n = x.shape[0]
        total_nll += loss.item() * n
        total_samples += n
        pbar.set_postfix(**postfix)

    if warmup_steps:
        for group in optimizer.param_groups:
            group["lr"] = sched_lr

    avg_grad_norms = (
        {k: v / n_batches for k, v in grad_norm_sums.items()} if log_grad_norm else {}
    )
    epoch_nll = total_nll / total_samples if total_samples else float("nan")
    return epoch_nll, avg_grad_norms, n_skipped, global_step


def _batch_covariance_twiss(samples: torch.Tensor):
    mean = samples.mean(dim=1, keepdim=True)
    dx = samples - mean
    sig11 = (dx[:, :, 0] ** 2).mean(dim=1)
    sig12 = (dx[:, :, 0] * dx[:, :, 1]).mean(dim=1)
    sig22 = (dx[:, :, 1] ** 2).mean(dim=1)
    emit = (sig11 * sig22 - sig12**2).clamp(min=0).sqrt()
    safe_emit = emit.clamp(min=1e-20)
    beta = sig11 / safe_emit
    alpha = -sig12 / safe_emit
    gamma = sig22 / safe_emit
    return emit, beta, alpha, gamma


def _bmag_batch(beta, alpha, gamma, beta0, alpha0):
    gamma0 = (1 + alpha0**2) / beta0
    return (beta * gamma0 - 2 * alpha * alpha0 + gamma * beta0) / 2


def validate(model, dataset, test_idx, batch_size, device):
    model.eval()
    total_nll = 0.0
    total_samples = 0

    emit_true_list, emit_pred_list = [], []
    beta_true_list, beta_pred_list = [], []
    alpha_true_list, alpha_pred_list = [], []
    gamma_true_list, gamma_pred_list = [], []

    with torch.no_grad():
        for coords, feat in tqdm(
            dataset.iter_batches(batch_size, shuffle=False, indices=test_idx),
            desc="Validation",
        ):
            n_batch, n_samp, _ = coords.shape
            x = coords.reshape(n_batch * n_samp, N_COORD)
            cond = (
                feat.unsqueeze(1)
                .expand(n_batch, n_samp, -1)
                .reshape(n_batch * n_samp, N_FEAT)
            )

            log_p = model.log_prob(x, cond)
            loss = -log_p.mean()
            total_nll += loss.item() * x.shape[0]
            total_samples += x.shape[0]

            pred_coords = model.sample(feat, n_per_cond=n_samp)

            emit_t, beta_t, alpha_t, gamma_t = _batch_covariance_twiss(coords)
            emit_p, beta_p, alpha_p, gamma_p = _batch_covariance_twiss(pred_coords)

            emit_true_list.append(emit_t)
            emit_pred_list.append(emit_p)
            beta_true_list.append(beta_t)
            beta_pred_list.append(beta_p)
            alpha_true_list.append(alpha_t)
            alpha_pred_list.append(alpha_p)
            gamma_true_list.append(gamma_t)
            gamma_pred_list.append(gamma_p)

    emit_true = torch.cat(emit_true_list)
    emit_pred = torch.cat(emit_pred_list)
    beta_true = torch.cat(beta_true_list)
    beta_pred = torch.cat(beta_pred_list)
    alpha_true = torch.cat(alpha_true_list)
    alpha_pred = torch.cat(alpha_pred_list)
    gamma_true = torch.cat(gamma_true_list)
    gamma_pred = torch.cat(gamma_pred_list)

    def safe(t):
        return t.abs().clamp(min=1e-20)

    emit_mape = ((emit_pred - emit_true).abs() / safe(emit_true)).mean().item() * 100
    beta_mape = ((beta_pred - beta_true).abs() / safe(beta_true)).mean().item() * 100
    alpha_rmse = ((alpha_pred - alpha_true) ** 2).mean().sqrt().item()
    gamma_mape = (
        (gamma_pred - gamma_true).abs() / safe(gamma_true)
    ).mean().item() * 100

    bmag = _bmag_batch(beta_pred, alpha_pred, gamma_pred, beta_true, alpha_true)
    bmag_geomean = bmag.clamp(min=1e-20).log().mean().exp().item()

    return {
        "nll": total_nll / total_samples,
        "emit_mape": emit_mape,
        "beta_mape": beta_mape,
        "alpha_rmse": alpha_rmse,
        "gamma_mape": gamma_mape,
        "bmag_geomean": bmag_geomean,
    }


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
    database: str,
    result_id: int,
    output_dir: str | None = None,
    batch_size: int = 32,
    n_samps: int = 512,
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
        output_dir = tempfile.mkdtemp(prefix="nvp_single_")
        print(f"Using temp output directory: {output_dir}")

    torch.manual_seed(seed)

    db_url = f"sqlite:///{database}"
    print(f"Database: {db_url}")
    print(f"Device:   {device}")
    print(f"Loading result ID: {result_id}")

    mlflow.log_params(
        {
            "batch_size": batch_size,
            "n_samps": n_samps,
            "epochs": epochs,
            "lr": lr,
            "lr_min_cos": lr_min_cos,
            "num_restarts": num_restarts,
            "train_fraction": train_fraction,
            "seed": seed,
            "result_id": result_id,
            "model_type": model_type,
            "num_layers": num_layers,
            "num_nvp_layers": num_nvp_layers,
            "num_normalizing_layers": num_normalizing_layers,
            "num_normalizing_hidden": num_normalizing_hidden,
            "nvp_num_hidden": nvp_num_hidden,
            "hidden_size": hidden_size,
            "rot_z_dim": rot_z_dim,
            "rot_num_hidden": rot_num_hidden,
            "rot_hidden_size": rot_hidden_size,
            "rot_cond_num_hidden": rot_cond_num_hidden,
            "rot_cond_hidden_size": rot_cond_hidden_size,
            "la_num_modules": la_num_modules,
            "latent_aspect_ratio": latent_aspect_ratio,
            "normalizing_activation": normalizing_activation,
            "device": device,
            "optimizer": optimizer,
            "weight_decay": weight_decay,
        }
    )

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    dataset = PhaseSpaceDistGPUSampler.from_database(
        db_url,
        tqdm,
        n_samps=n_samps,
        device=device,
        result_ids=[result_id],
    )

    # Restrict features to s_position only (column 0)
    dataset.features = dataset.features[:, 0:1]

    n = len(dataset)
    print(f"Histograms loaded: {n}")

    # Stratified train/test split by quantile bins of the conditioning variable
    c_np = dataset.features[:, 0].cpu().numpy()
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
        model = ConditionedNormalCoordinatesFlow(
            data_dim=N_COORD,
            num_nvp_layers=num_nvp_layers,
            num_normalizing_layers=num_normalizing_layers,
            num_normalizing_hidden=num_normalizing_hidden,
            nvp_num_hidden=nvp_num_hidden,
            cond_dim=N_FEAT,
            hidden_size=hidden_size,
            rot_z_dim=rot_z_dim,
            rot_num_hidden=_rot_n,
            rot_hidden_size=_rot_h,
            rot_cond_hidden_dims=[_cond_h] * rot_cond_num_hidden,
            la_num_modules=la_num_modules,
            latent_aspect_ratio=latent_aspect_ratio,
            x_mean=x_mean,
            x_std=x_std,
            cond_mean=cond_mean,
            cond_std=cond_std,
        ).to(device)
    elif model_type == "nvp":
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
        model = ConditionedLASympNetFlow(
            data_dim=N_COORD,
            cond_dim=N_FEAT,
            num_modules=la_num_modules,
            hidden_size=hidden_size,
            num_hidden=nvp_num_hidden,
            activation=normalizing_activation,
            num_nvp_layers=num_nvp_layers,
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
    baseline_metrics = validate(model, dataset, test_idx, batch_size, device)
    print(f"Baseline NLL:        {baseline_metrics['nll']:.4f} nats/sample")
    print(f"  emit MAPE:         {baseline_metrics['emit_mape']:.2f}%")
    print(f"  beta MAPE:         {baseline_metrics['beta_mape']:.2f}%")
    print(f"  alpha RMSE:        {baseline_metrics['alpha_rmse']:.4f}")
    print(f"  gamma MAPE:        {baseline_metrics['gamma_mape']:.2f}%")
    print(f"  Bmag geomean:      {baseline_metrics['bmag_geomean']:.4f}")

    if optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

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
            log_grad_norm=log_grad_norm,
            global_step=global_step,
            warmup_steps=warmup_steps,
        )
        train_time = time.time() - t0

        t0 = time.time()
        val_metrics = validate(model, dataset, test_idx, batch_size, device)
        val_time = time.time() - t0

        current_lr = opt.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        grad_norm_metrics = {f"grad_norm/{k}": v for k, v in avg_grad_norms.items()}
        log_metrics = {
            "train_nll": train_nll,
            "val_nll": val_metrics["nll"],
            "val_emit_mape": val_metrics["emit_mape"],
            "val_beta_mape": val_metrics["beta_mape"],
            "val_alpha_rmse": val_metrics["alpha_rmse"],
            "val_gamma_mape": val_metrics["gamma_mape"],
            "val_bmag_geomean": val_metrics["bmag_geomean"],
            "epoch_time": time.perf_counter() - epoch_start_t,
            "train_time": train_time,
            "val_time": val_time,
            "lr": current_lr,
            "n_skipped_steps": n_skipped,
            **grad_norm_metrics,
        }

        print(f"Train NLL: {train_nll:.4f} nats/sample  [{train_time:.1f}s]")
        print(f"Val NLL:   {log_metrics['val_nll']:.4f} nats/sample  [{val_time:.1f}s]")
        print(
            f"  emit MAPE: {log_metrics['val_emit_mape']:.2f}%  "
            f"beta MAPE: {log_metrics['val_beta_mape']:.2f}%  "
            f"alpha RMSE: {log_metrics['val_alpha_rmse']:.4f}  "
            f"gamma MAPE: {log_metrics['val_gamma_mape']:.2f}%  "
            f"Bmag geomean: {log_metrics['val_bmag_geomean']:.4f}"
        )

        if n_skipped:
            print(f"  Skipped {n_skipped} steps with non-finite gradients")

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

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            **log_metrics,
        }
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)

        if log_metrics["val_nll"] < best_val_nll:
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
    last_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{total_epochs}.pt")
    if os.path.exists(last_path):
        mlflow.log_artifact(last_path)
    mlflow.log_artifact(stats_path)
    mlflow.log_artifact(split_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train a conditional Real-NVP flow on a single simulation result"
    )
    parser.add_argument(
        "--database",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--result-id",
        type=int,
        required=True,
        help="Result ID to load from the database",
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
        help="Number of histograms per batch (default: 32)",
    )
    parser.add_argument(
        "--n-samps",
        type=int,
        default=512,
        help="Particle samples drawn per histogram per batch (default: 512)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
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
        help="Fraction of histograms for training (default: 0.8)",
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
        "--rot-z-dim",
        type=int,
        default=64,
        help="Latent dim of the rotation conditioning MLP, used when --model-type=normal_coordinates (default: 64)",
    )
    parser.add_argument(
        "--rot-num-hidden",
        type=int,
        default=None,
        help="Number of hidden layers in each GradNet rotation MLP (default: num_normalizing_hidden)",
    )
    parser.add_argument(
        "--rot-hidden-size",
        type=int,
        default=None,
        help="Width of each GradNet rotation MLP layer (default: hidden_size)",
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
        help="Width of the rotation conditioning MLP (default: hidden_size)",
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
        default="nvp-single-result",
        help="MLflow experiment name (default: nvp-single-result)",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():
        train(
            database=os.path.expandvars(args.database),
            result_id=args.result_id,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            n_samps=args.n_samps,
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
