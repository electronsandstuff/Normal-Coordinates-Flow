"""
The uniform focusing channel dataset comes from a legacy format stored as images in a sqlite
database. This dataloader handles the complexity of opening it and resampling the histograms
into data which is appropriate for training generative models on.

The pydantic models at the top replicate the schema for the histogram data being stored along
with transformations of phase space. Helper functions for dealing with images and the database
are below that.

See PhaseSpaceDistGPUSampler for the main dataset class.
"""

import json
from io import BytesIO
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel
from scipy import interpolate
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.orm import sessionmaker


class HistBounds(BaseModel):
    coord_min: list[float]
    coord_max: list[float]
    n_bins: list[int]

    def get_pixel_coords(self):
        """
        Get coordinates for points around bounding box
        Note: these are histograms, so we get bin centers of outermost pixels
        """
        dx = [
            (self.coord_max[idx] - self.coord_min[idx]) / self.n_bins[idx]
            for idx in range(2)
        ]
        upper_left = np.array(
            [self.coord_min[0] + dx[0] / 2, self.coord_max[1] - dx[1] / 2]
        )
        lower_right = np.array(
            [self.coord_max[0] - dx[0] / 2, self.coord_min[1] + dx[1] / 2]
        )
        lower_left = np.array(
            [self.coord_min[0] + dx[0] / 2, self.coord_min[1] + dx[1] / 2]
        )

        # Make grid from 0 to one (first index is coordinate index)
        r = np.mgrid[0 : 1 : self.n_bins[1] * 1j, 0 : 1 : self.n_bins[0] * 1j]

        # Construct transformation matrix from [0, 1], [0, 1] to pixel space w/ lower left pixel at origin
        mat = np.vstack((upper_left - lower_left, lower_right - lower_left)).T

        # Apply the transformation
        r_prime = np.einsum("ij,jkl->ikl", mat, r) + lower_left[:, None, None]
        return r_prime


class CorrelationTransform(BaseModel):
    intercept: float
    slope: float


class HistogramBytes(BaseModel):
    """Beam phase-space histogram stored as a PNG with its coordinate transforms."""

    bounds: HistBounds
    correlation_transform: CorrelationTransform
    max_density: float
    s_position: float
    pos_vel_symmetry: bool
    percentage_excluded: float
    image_data: bytes

    def get_density(self):
        rho = np.asarray(Image.open(BytesIO(self.image_data), formats=["png"]))
        rho = np.flip(rho, axis=0)
        return rho

    def get_pixel_coords(self):
        # Get coords from the bounds (transformed space)
        pix_x, pix_y = self.bounds.get_pixel_coords()

        # Apply symmetry transform
        if self.pos_vel_symmetry:
            # Construct coordinates for symmetric data
            pix_x = np.concat((-np.flip(np.flip(pix_x, axis=1), axis=0), pix_x), axis=1)
            pix_y = np.concat(
                (-(np.flip(np.flip(pix_y, axis=1), axis=0)), pix_y), axis=1
            )

        # Apply correlation transform
        pix_y = (
            pix_y
            + self.correlation_transform.slope * pix_x
            + self.correlation_transform.intercept
        )

        # Return them
        return pix_x, pix_y

    def get_density_and_pixel_coords(self):
        """Return pixel-center coordinates and the scaled density.

        Returns
        -------
        (coord1, coord2, rho) : tuple(np.ndarray, np.ndarray, np.ndarray)
        """
        # Load the image data itself
        rho = self.get_density()

        # Scale
        rho = rho / rho.max() * self.max_density

        # Get the pixel coordinates
        coord1, coord2 = self.get_pixel_coords()

        # Concat the image due to symmetry transform
        if self.pos_vel_symmetry:
            rho = np.concat((np.flip(np.flip(rho, axis=0), axis=1), rho), axis=1)

        return coord1, coord2, rho

    def interp_to_grid(self, bounds, res=512):
        """Interpolate the density onto an axis-aligned grid.

        Parameters
        ----------
        bounds : list[tuple[float, float]]
            ``(lower_bound, upper_bound)`` for each coordinate.
        res : int
            Number of pixels along each axis.

        Returns
        -------
        (coord1, coord2, rho_pred) : tuple(np.ndarray, np.ndarray, np.ndarray)
        """
        # Load the density
        coord1, coord2, rho = self.get_density_and_pixel_coords()

        # Interpolate onto an axis aligned grid
        coord1_interp, coord2_interp = np.mgrid[
            bounds[0][0] : bounds[0][1] : res * 1j,
            bounds[1][0] : bounds[1][1] : res * 1j,
        ]
        rho_pred = interpolate.griddata(
            np.vstack((coord1.ravel(), coord2.ravel())).T,
            rho.ravel(),
            np.vstack((coord1_interp.ravel(), coord2_interp.ravel())).T,
            method="linear",
            fill_value=0.0,
        )
        rho_pred = np.reshape(rho_pred, coord1_interp.shape)

        return coord1_interp, coord2_interp, rho_pred


def _load_histograms_from_database(
    database_url: str,
    progressbar=None,
    result_ids: list[int] | None = None,
) -> Tuple[list, np.ndarray]:
    """Load histograms and feature arrays from a database.

    Parameters
    ----------
    database_url : str
        SQLite database URL.
    progressbar : callable, optional
        Wraps the row iterator (e.g. tqdm).
    result_ids : list[int], optional
        Result IDs to load.  Defaults to all results.

    Returns
    -------
    histograms : list[HistogramBytes]
    features : np.ndarray, shape (n, 7), float32
        Columns: [s_position, twiss_beta, twiss_alpha, supergaussian_n,
        perveance, lattice_k, emittance]
    """
    engine = create_engine(database_url)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    if progressbar is None:

        def progressbar(x):
            return x

    try:
        result_filter = ""
        params = {}
        if result_ids is not None:
            result_filter = "WHERE ctr.id IN :result_ids"
            params["result_ids"] = list(result_ids)

        query = text(f"""
                SELECT
                    hb.coord_min,
                    hb.coord_max,
                    hb.n_bins,
                    hb.intercept,
                    hb.slope,
                    hb.max_density,
                    hb.s_position,
                    hb.pos_vel_symmetry,
                    hb.percentage_excluded,
                    hb.image_data,
                    sd.twiss_beta,
                    sd.twiss_alpha,
                    sd.n AS supergaussian_n,
                    b.perveance,
                    ufc.k AS lattice_k,
                    sd.emittance
                FROM histogram_bytes hb
                JOIN cmp_track_output cto ON hb.cmp_track_output_id = cto.id
                JOIN cmp_track_result ctr ON ctr.cmp_track_output_id = cto.id
                JOIN beam1d_x_config cfg ON ctr.cmp_track_input_id = cfg.id
                JOIN beam b ON cfg.beam_id = b.id
                JOIN supergaussian_distribution sd ON b.initial_distribution_id = sd.id
                JOIN uniform_focusing_channel_lattice ufc ON cfg.lattice_id = ufc.id
                {result_filter}
            """)
        if params:
            query = query.bindparams(bindparam("result_ids", expanding=True))
        rows = session.execute(query, params).fetchall()

        histograms = []
        features_list = []

        for row in progressbar(rows):
            coord_min = json.loads(row[0]) if isinstance(row[0], str) else row[0]
            coord_max = json.loads(row[1]) if isinstance(row[1], str) else row[1]
            n_bins = json.loads(row[2]) if isinstance(row[2], str) else row[2]

            hist = HistogramBytes(
                bounds=HistBounds(
                    coord_min=coord_min, coord_max=coord_max, n_bins=n_bins
                ),
                correlation_transform=CorrelationTransform(
                    intercept=row[3], slope=row[4]
                ),
                max_density=row[5],
                s_position=row[6],
                pos_vel_symmetry=bool(row[7]),
                percentage_excluded=row[8],
                image_data=row[9],
            )
            histograms.append(hist)
            features_list.append(
                [
                    row[6],  # s_position
                    row[10],  # twiss_beta
                    row[11],  # twiss_alpha
                    row[12],  # supergaussian_n
                    row[13],  # perveance
                    row[14],  # lattice_k
                    row[15],  # emittance
                ]
            )

        return histograms, np.array(features_list, dtype=np.float32)
    finally:
        session.close()


def _decode_histogram_data(
    histograms: list,
    features=None,
    progressbar=None,
) -> dict:
    """Decode a list of HistogramBytes into numpy arrays.

    Parameters
    ----------
    histograms : list[HistogramBytes]
    features : array-like, optional
        Per-histogram scalar features, shape (n, n_feat).
    progressbar : callable, optional

    Returns
    -------
    dict
        Keys: images, bounds, corr_m, corr_b, pos_vel_symmetry, features.
    """
    if not histograms:
        raise ValueError("Must provide at least one histogram")

    if progressbar is None:

        def progressbar(x):
            return x

    pos_vel_symmetry = histograms[0].pos_vel_symmetry

    decoded = []
    bounds_list = []
    corr_m_list = []
    corr_b_list = []

    for hist in progressbar(histograms):
        if hist.pos_vel_symmetry != pos_vel_symmetry:
            raise ValueError("All histograms must have the same pos_vel_symmetry value")

        decoded.append(hist.get_density())
        corr_b_list.append(hist.correlation_transform.intercept)
        corr_m_list.append(hist.correlation_transform.slope)
        bounds_list.append(
            [
                [hist.bounds.coord_min[0], hist.bounds.coord_min[1]],
                [hist.bounds.coord_max[0], hist.bounds.coord_max[1]],
            ]
        )

    if features is None:
        features = np.empty((len(decoded), 0), dtype=np.float32)

    return {
        "images": np.stack(decoded).astype(np.uint8),
        "bounds": np.array(bounds_list, dtype=np.float32),
        "corr_m": np.array(corr_m_list, dtype=np.float32),
        "corr_b": np.array(corr_b_list, dtype=np.float32),
        "pos_vel_symmetry": pos_vel_symmetry,
        "features": np.asarray(features, dtype=np.float32),
    }


def _interp_cdf(cdf: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Batch inverse-CDF via linear interpolation.

    Parameters
    ----------
    cdf : Tensor, shape ``(B, n_levels)``
        Normalized CDF values, increasing from 0 to 1 along dim 1.
    q : Tensor, shape ``(B, M)``
        Query quantile values in ``[0, 1]``.

    Returns
    -------
    Tensor, shape ``(B, M)``
        Floating-point indices into ``[0, n_levels-1]``.
    """
    n_levels = cdf.shape[1]
    idx = torch.searchsorted(cdf.contiguous(), q.contiguous())  # (B, M)
    idx = idx.clamp(1, n_levels - 1)
    x0 = cdf.gather(1, idx - 1)  # (B, M)
    x1 = cdf.gather(1, idx)  # (B, M)
    t = ((q - x0) / (x1 - x0).clamp(min=1e-10)).clamp(0.0, 1.0)
    return (idx - 1).float() + t


class PhaseSpaceDistGPUSampler:
    """Dataset of particle samples drawn from histogram distributions.

    Samples are produced on-device with inverse-CDF sampling.
    """

    def __init__(
        self,
        images: torch.Tensor,
        sample_mat: torch.Tensor,
        sample_trans: torch.Tensor,
        features: torch.Tensor,
        n_samps: int,
        device,
    ):
        """
        Parameters
        ----------
        images : torch.Tensor
            Full-size histogram images, uint8, shape ``(n, h, w_full)``.
        sample_mat : torch.Tensor
            Affine matrix from ``(row, col)`` pixel indices to ``(x, xp)``,
            float32, shape ``(n, 2, 2)``.
        sample_trans : torch.Tensor
            Translation of the affine map, float32, shape ``(n, 2)``.
        features : torch.Tensor
            Scalar features of each histogram, float32, shape ``(n, n_feat)``.
        n_samps : int
            Samples drawn per histogram per batch.
        device : str or torch.device
            Target device.
        """
        self.device = torch.device(device)
        self.images = images.to(self.device)
        self._sample_mat = sample_mat.to(self.device)
        self._sample_trans = sample_trans.to(self.device)
        self.features = features.to(self.device)
        self.n_samps = n_samps

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def __len__(self) -> int:
        return self.images.shape[0]

    def _compute_batch(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw ``n_samps`` particle samples from each histogram in ``idx``.

        Parameters
        ----------
        idx : torch.Tensor, shape ``(B,)``

        Returns
        -------
        coords : Tensor, shape ``(B, n_samps, 2)``
        feat : Tensor, shape ``(B, n_feat)``
        """
        n_batch = idx.shape[0]
        h, w = self.images.shape[1], self.images.shape[2]

        # Load images as float: (B, h, w)
        rho = self.images[idx].to(torch.float32)

        # Cumulative sum along rows for each column: conditional CDFs
        # rho_x[b, i, j] = sum_{k<=i} rho[b, k, j], shape (B, h, w)
        rho_x = rho.cumsum(dim=1)

        # Marginal CDF: total per column, then cumsum over columns
        rho_xy = rho_x[:, -1, :].cumsum(dim=1)  # (B, w)
        rho_xy_norm = rho_xy / rho_xy[:, -1:].clamp(min=1e-10)  # (B, w)

        # Sample column index by inverting the marginal CDF.
        v = torch.rand(n_batch, self.n_samps, device=self.device)  # (B, n_samps)
        col_idx = torch.searchsorted(rho_xy_norm.contiguous(), v.contiguous()).clamp(
            0, w - 1
        )  # (B, n_samps)
        n_float = _interp_cdf(rho_xy_norm, v)  # (B, n_samps)

        # Normalize conditional CDFs: (B, h, w)
        cond_cdf = rho_x / rho_x[:, -1:, :].clamp(min=1e-10)

        # Gather the conditional CDF for each sample's column: (B, h, n_samps)
        col_idx_exp = col_idx.unsqueeze(1).expand(n_batch, h, self.n_samps)
        cond_samps = torch.gather(cond_cdf, dim=2, index=col_idx_exp)

        # Invert the conditional CDFs: reshape to (B*n_samps, h) for _interp_cdf
        cond_flat = cond_samps.permute(0, 2, 1).reshape(n_batch * self.n_samps, h)
        u = torch.rand(n_batch * self.n_samps, 1, device=self.device)
        m_float = _interp_cdf(cond_flat, u).reshape(n_batch, self.n_samps)

        # Stack (row, col) indices: (B, n_samps, 2)
        mn = torch.stack([m_float, n_float], dim=2)

        # Apply per-histogram affine transform to physical (x, xp) coordinates
        mat = self._sample_mat[idx]  # (B, 2, 2)
        trans = self._sample_trans[idx]  # (B, 2)
        coords = torch.einsum("bij,bsj->bsi", mat, mn) + trans.unsqueeze(1)

        return coords, self.features[idx]

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        indices: torch.Tensor | None = None,
    ):
        """Yield ``(coords, feat)`` batches on :attr:`device`.

        Parameters
        ----------
        batch_size : int
        shuffle : bool
            Draw a fresh permutation at the start of every call.
        indices : Tensor, optional
            1-D long tensor of histogram indices to draw from.  Defaults to all
            histograms.

        Yields
        ------
        coords : Tensor, shape ``(B, n_samps, 2)``
        feat : Tensor, shape ``(B, n_feat)``
        """
        src = (
            indices
            if indices is not None
            else torch.arange(len(self), device=self.device)
        )
        if shuffle:
            src = src[torch.randperm(len(src), device=self.device)]
        for start in range(0, len(src), batch_size):
            yield self._compute_batch(src[start : start + batch_size])

    @classmethod
    def from_histograms(
        cls,
        histograms: list[HistogramBytes],
        features=None,
        progressbar=None,
        n_samps: int = 1000,
        device="cuda",
    ) -> "PhaseSpaceDistGPUSampler":
        """Create a sampler from a list of histograms.

        Parameters
        ----------
        histograms : list[HistogramBytes]
        features : array-like, optional
        progressbar : callable, optional
        n_samps : int
            Samples drawn per histogram per batch.
        device : str or torch.device
        """
        data = _decode_histogram_data(histograms, features, progressbar)
        images = data["images"]  # (n, h, w_orig), uint8

        pos_vel_symmetry = data["pos_vel_symmetry"]
        if pos_vel_symmetry:
            images = np.concatenate(
                [np.flip(np.flip(images, axis=1), axis=2), images], axis=2
            )

        n, h, w_full = images.shape
        bounds = data["bounds"]  # (n, 2, 2): [:,0,:]=coord_min, [:,1,:]=coord_max
        corr_m = data["corr_m"]  # (n,)
        corr_b = data["corr_b"]  # (n,)

        # Full-image x bounds (symmetry-aware)
        if pos_vel_symmetry:
            x_min_full = -bounds[:, 1, 0]  # -x_max_orig
            x_range_full = 2.0 * bounds[:, 1, 0]  # 2 * x_max_orig
        else:
            x_min_full = bounds[:, 0, 0]
            x_range_full = bounds[:, 1, 0] - bounds[:, 0, 0]

        xp_min = bounds[:, 0, 1]
        xp_range = bounds[:, 1, 1] - bounds[:, 0, 1]

        # Affine transform: (x, xp) = mat @ (row, col) + trans
        #   mat = [[0,            x_range_full/w_full],
        #          [xp_range/h,   corr_m*x_range_full/w_full]]
        #   trans = [x_min_full, corr_m*x_min_full + xp_min + corr_b]
        sample_mat = np.zeros((n, 2, 2), dtype=np.float32)
        sample_mat[:, 0, 1] = x_range_full / w_full
        sample_mat[:, 1, 0] = xp_range / h
        sample_mat[:, 1, 1] = corr_m * x_range_full / w_full

        sample_trans = np.zeros((n, 2), dtype=np.float32)
        sample_trans[:, 0] = x_min_full
        sample_trans[:, 1] = corr_m * x_min_full + xp_min + corr_b

        return cls(
            images=torch.from_numpy(np.ascontiguousarray(images)),
            sample_mat=torch.from_numpy(sample_mat),
            sample_trans=torch.from_numpy(sample_trans),
            features=torch.from_numpy(data["features"]),
            n_samps=n_samps,
            device=device,
        )

    @classmethod
    def from_database(
        cls,
        database_url: str,
        progressbar=None,
        n_samps: int = 1000,
        device="cuda",
        result_ids: list[int] | None = None,
    ) -> "PhaseSpaceDistGPUSampler":
        """Create a sampler from histograms stored in a database.

        Parameters
        ----------
        database_url : str
            SQLite database URL.
        progressbar : callable, optional
        n_samps : int
            Samples drawn per histogram per batch.
        device : str or torch.device
        result_ids : list[int], optional
            Result IDs to load.  Defaults to all results.
        """
        histograms, features = _load_histograms_from_database(
            database_url, progressbar, result_ids=result_ids
        )
        return cls.from_histograms(
            histograms,
            features=features,
            progressbar=progressbar,
            n_samps=n_samps,
            device=device,
        )
