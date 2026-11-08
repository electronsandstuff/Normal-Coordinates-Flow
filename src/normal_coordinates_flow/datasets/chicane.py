from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from beamphysics import ParticleGroup
from pydantic import BaseModel


class ParticleOutput(BaseModel):
    path: Path
    s_position: float
    element_name: str


class Manifest(BaseModel):
    particles: list[ParticleOutput]
    start_time: datetime
    end_time: datetime
    comment: str = ""


class ChicaneDatasetGPU:
    """Dataset for chicane particle tracking simulations.

    Stores 6D particle coordinates as ``(x, y, t, px, py, pz)`` grouped by
    screen  The s-position of each screen is kept as a per-screen conditioning feature.
    """

    def __init__(
        self,
        particles: torch.Tensor,
        s_positions: torch.Tensor,
        device,
    ):
        """
        Parameters
        ----------
        particles : torch.Tensor
            Phase-space coordinates, float32, shape ``(n_screens, n_particles, 6)``.
            Column order: ``[x, y, t, px, py, pz]``.
        s_positions : torch.Tensor
            Longitudinal position of each screen, float32, shape ``(n_screens,)``.
        device : str or torch.device
            Target device (e.g. ``"cuda"`` or ``"cpu"``).
        """
        self.device = torch.device(device)
        self.particles = particles.to(self.device)
        self.s_positions = s_positions.to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def __len__(self) -> int:
        return self.particles.shape[0]

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        indices: Optional[torch.Tensor] = None,
    ):
        """Yield ``(particles, s_positions)`` batches on `device`.

        Parameters
        ----------
        batch_size : int
        shuffle : bool
            Draw a fresh permutation at the start of every call.
        indices : Tensor, optional
            1-D long tensor of screen indices to draw from. Defaults to all screens.

        Yields
        ------
        particles : Tensor, shape ``(B, n_particles, 6)``
        s_positions : Tensor, shape ``(B, 1)``
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
            yield self.particles[idx], self.s_positions[idx].unsqueeze(1)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: "str | Path",
        progressbar=None,
        device="cuda",
    ) -> "ChicaneDatasetGPU":
        """Load all particle groups listed in a manifest file.

        Parameters
        ----------
        manifest_path : str or Path
            Path to ``manifest.json``.
        progressbar : callable, optional
            Wraps the screen iterator (e.g. ``tqdm``).
        device : str or torch.device

        Returns
        -------
        ChicaneDatasetGPU
        """
        manifest_path = Path(manifest_path)
        manifest = Manifest.model_validate_json(manifest_path.read_text())
        base_dir = manifest_path.parent

        if progressbar is None:

            def progressbar(x):
                return x

        all_coords: list[np.ndarray] = []
        s_pos_list: list[float] = []

        for particle_output in progressbar(manifest.particles):
            pg = ParticleGroup(str(base_dir / particle_output.path))
            coords = np.stack([pg.x, pg.y, pg.t, pg.px, pg.py, pg.pz], axis=-1).astype(
                np.float32
            )
            all_coords.append(coords)
            s_pos_list.append(particle_output.s_position)

        # Truncate to the minimum live-particle count across all screens
        min_n = min(c.shape[0] for c in all_coords)
        particles_arr = np.stack(
            [c[:min_n] for c in all_coords]
        )  # (n_screens, min_n, 6)
        s_pos_arr = np.array(s_pos_list, dtype=np.float32)

        return cls(
            particles=torch.from_numpy(particles_arr),
            s_positions=torch.from_numpy(s_pos_arr),
            device=device,
        )
