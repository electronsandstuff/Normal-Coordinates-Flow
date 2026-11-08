import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch

_INT_ATTRS = ("n_oscillators", "n_systems", "n_outputs", "output_every_steps")
_LONG_ATTRS = ("n_steps", "seed")
_FLOAT_ATTRS = ("beta", "epsilon", "dt", "total_time", "output_time")
_STR_ATTRS = ("coordinate_order", "creation_time")


def _coerce_attrs(attrs) -> dict:
    """Convert HDF5 root attributes to plain python scalars."""
    out = dict(attrs)
    for key in _INT_ATTRS + _LONG_ATTRS:
        if key in out:
            out[key] = int(out[key])
    for key in _FLOAT_ATTRS:
        if key in out:
            out[key] = float(out[key])
    for key in _STR_ATTRS:
        if key in out and isinstance(out[key], bytes):
            out[key] = out[key].decode()
    if "sigmas" in out:
        out["sigmas"] = np.asarray(out["sigmas"], dtype=np.float64)
    return out


def interleaved_to_blocked_indices(n_oscillators: int) -> np.ndarray:
    """Column permutation from ``q0,p0,q1,p1,...`` to ``q0,...,qN-1,p0,...,pN-1``.

    We use this to take data from simulation tool output and put it in order expected
    by flow models.

    Parameters
    ----------
    n_oscillators : int

    Returns
    -------
    ndarray, shape ``(2 * n_oscillators,)``
    """
    return np.concatenate(
        [
            np.arange(0, 2 * n_oscillators, 2),
            np.arange(1, 2 * n_oscillators, 2),
        ]
    )


class DuffingChainDatasetGPU:
    """Dataset for coupled Duffing chain simulations."""

    def __init__(
        self,
        states: torch.Tensor,
        times: torch.Tensor,
        device,
        beta: float = 0.0,
        epsilon: float = 0.0,
        n_oscillators: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        states : torch.Tensor
            Phase-space coordinates, float32, shape
            ``(n_snapshots, n_systems, 2 * n_oscillators)``.
            Column order: ``[q_0, ..., q_{N-1}, p_0, ..., p_{N-1}]``.
        times : torch.Tensor
            Snapshot times, float32, shape ``(n_snapshots,)``.
        device : str or torch.device
            Target device (e.g. ``"cuda"`` or ``"cpu"``).
        beta : float
            Quartic nonlinearity coefficient.
        epsilon : float
            Nearest-neighbor coupling coefficient.
        n_oscillators : int, optional
            Defaults to ``states.shape[-1] // 2``.
        """
        self.device = torch.device(device)
        self.states = states.to(self.device)
        self.times = times.to(self.device)
        self.beta = beta
        self.epsilon = epsilon
        self.n_oscillators = (
            n_oscillators if n_oscillators is not None else self.states.shape[-1] // 2
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @property
    def data_dim(self) -> int:
        """Phase-space dimension ``2 * n_oscillators``."""
        return self.states.shape[-1]

    @property
    def n_systems(self) -> int:
        """Number of independent systems in the ensemble."""
        return self.states.shape[1]

    def __len__(self) -> int:
        return self.states.shape[0]

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        indices: Optional[torch.Tensor] = None,
    ):
        """Yield ``(states, times)`` batches on `device`.

        Parameters
        ----------
        batch_size : int
        shuffle : bool
            Draw a fresh permutation at the start of every call.
        indices : Tensor, optional
            1-D long tensor of snapshot indices to draw from. Defaults to all snapshots.

        Yields
        ------
        states : Tensor, shape ``(B, n_systems, 2 * n_oscillators)``
        times : Tensor, shape ``(B, 1)``
        """
        src = (
            indices
            if indices is not None
            else torch.arange(len(self), device=self.device)
        )
        if shuffle:
            src = src[torch.randperm(len(src), device=self.device)]
        for start in range(0, len(src), batch_size):
            idx = src[start : start + batch_size]
            yield self.states[idx], self.times[idx].unsqueeze(1)

    def energy(self, states: torch.Tensor) -> torch.Tensor:
        """Per-system Hamiltonian of the generating model on a periodic ring.

        Can be used to check energy conservation. Not explored in paper, but
        enforced by the Normal Coordinates flow.

        Parameters
        ----------
        states : Tensor, shape ``(..., 2 * n_oscillators)``
            Phase-space coordinates in blocked order.

        Returns
        -------
        Tensor, shape ``(...,)``
        """
        n = self.n_oscillators
        q, p = states[..., :n], states[..., n:]
        q_next = torch.roll(q, -1, dims=-1)
        h = (
            0.5 * p**2
            + 0.5 * q**2
            + 0.25 * self.beta * q**4
            + 0.5 * self.epsilon * (q_next - q) ** 2
        )
        return h.sum(dim=-1)

    @classmethod
    def from_file(
        cls,
        path: "str | Path",
        progressbar=None,
        device="cuda",
    ) -> "DuffingChainDatasetGPU":
        """Load every snapshot of a duffing_chain HDF5 run.

        Parameters
        ----------
        path : str or Path
            Path to the HDF5 file.  Environment variables are expanded.
        progressbar : callable, optional
            Wraps the snapshot iterator (e.g. ``tqdm``).
        device : str or torch.device

        Returns
        -------
        DuffingChainDatasetGPU
        """
        path = Path(os.path.expandvars(str(path)))

        if progressbar is None:

            def progressbar(x):
                return x

        with h5py.File(path, "r") as f:
            attrs = _coerce_attrs(f.attrs)
            n_oscillators = attrs["n_oscillators"]
            n_systems = attrs["n_systems"]
            n_outputs = attrs["n_outputs"]
            data_dim = 2 * n_oscillators
            perm = interleaved_to_blocked_indices(n_oscillators)

            group = f["states"]
            names = sorted(group.keys(), key=lambda name: int(name.split("_")[1]))
            if len(names) != n_outputs:
                raise ValueError(
                    f"{path}: found {len(names)} snapshots but n_outputs={n_outputs}"
                )

            states_arr = np.empty((n_outputs, n_systems, data_dim), dtype=np.float32)
            for i, name in enumerate(progressbar(names)):
                snapshot = group[name][()]
                if snapshot.shape != (n_systems, data_dim):
                    raise ValueError(
                        f"{path}: snapshot {name} has shape {snapshot.shape}, "
                        f"expected {(n_systems, data_dim)}"
                    )
                states_arr[i] = snapshot[:, perm]

            times_arr = f["times"][()].astype(np.float32)

        return cls(
            states=torch.from_numpy(states_arr),
            times=torch.from_numpy(times_arr),
            device=device,
            beta=attrs["beta"],
            epsilon=attrs["epsilon"],
            n_oscillators=n_oscillators,
        )
