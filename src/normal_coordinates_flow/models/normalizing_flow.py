import math

import numpy as np
import torch
import torch.nn as nn

from normal_coordinates_flow.models.gradient_network import CondGradNet, CondMLP
from normal_coordinates_flow.models.sympnet import LASympNet

# Rotation-layer ablation settings for ConditionedNormalCoordinatesFlow.
#   none         full amplitude- and cond-dependent GradNet rotation
#   no-amplitude angles depend on cond only, making the rotation linear in z
#   no-rotation  rotation layer omitted entirely
#   mlp          plain MLP angles: still volume preserving, no longer symplectic
ROTATION_ABLATIONS = ("none", "no-amplitude", "no-rotation", "mlp")


def _standard_normal_log_prob(z: torch.Tensor) -> torch.Tensor:
    """Log probability of z under N(0, I), shape (N,)."""
    return -0.5 * (z * z + math.log(2 * math.pi)).sum(dim=-1)


class ConditionedNormalCoordinatesLayer(nn.Module):
    """Conditioned flow layer for 2n-dimensional phase space.

    Forward pass: z -> L(0) -> R(cond) -> L_inv(cond) -> y, where L is an
    LASympNet and R is a SymplecticRotationLayer.  log_det = 0.

    Parameters
    ----------
    n : int
        Number of phase-space planes. Phase-space dimension is 2n.
    cond_dim : int
        Conditioning feature dimension.
    hidden_size : int
        Hidden width of the LASympNet MLPs.
    num_hidden : int
        Depth of the LASympNet MLPs.
    la_num_modules : int
        Number of modules in the LASympNet.
    rot_z_dim : int
        Latent dimension of the rotation conditioning MLP.
    rot_cond_hidden_dims : list of int, optional
        Hidden widths of the rotation conditioning MLP.
    rot_hidden_size : int
        Hidden width of the rotation network.
    rot_num_hidden : int
        Number of layers in the rotation network.
    use_rotation : bool
        If False, the rotation layer is omitted.
    rot_amplitude_dependent : bool
        If False, the angles depend on the conditioning features only.
    rot_net_type : {'gradnet', 'mlp'}
        Network producing the angles.
    """

    def __init__(
        self,
        n: int,
        cond_dim: int,
        hidden_size: int = 128,
        num_hidden: int = 2,
        la_num_modules: int = 3,
        rot_z_dim: int = 64,
        rot_cond_hidden_dims: list[int] | None = None,
        rot_hidden_size=32,
        rot_num_hidden=3,
        use_rotation: bool = True,
        rot_amplitude_dependent: bool = True,
        rot_net_type: str = "gradnet",
    ):
        super().__init__()
        self.n = n

        self.la_sympnet = LASympNet(
            n=n,
            cond_dim=cond_dim,
            num_modules=la_num_modules,
            hidden_size=hidden_size,
            num_hidden=num_hidden,
            activation="sigmoid",
        )
        self.rotation = (
            SymplecticRotationLayer(
                n=n,
                cond_dim=cond_dim,
                num_layers=rot_num_hidden,
                hidden_dim=rot_hidden_size,
                cond_latent_dim=rot_z_dim,
                cond_hidden_dims=rot_cond_hidden_dims
                if rot_cond_hidden_dims is not None
                else [hidden_size],
                activation="gelu",
                cond_activation="gelu",
                amplitude_dependent=rot_amplitude_dependent,
                net_type=rot_net_type,
            )
            if use_rotation
            else None
        )

    def forward(
        self, z: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """z -> LASympNet(0) -> rotation -> LASympNet_inv(cond) -> y."""
        zero_cond = torch.zeros_like(cond)
        z, _ = self.la_sympnet.forward(z, zero_cond)
        if self.rotation is not None:
            z, _ = self.rotation.forward(z, cond)
        z, _ = self.la_sympnet.inverse(z, cond)
        return z, torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

    def inverse(
        self, z: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """y -> LASympNet(cond) -> rotation_inv -> LASympNet_inv(0) -> z."""
        zero_cond = torch.zeros_like(cond)
        z, _ = self.la_sympnet.forward(z, cond)
        if self.rotation is not None:
            z, _ = self.rotation.inverse(z, cond)
        z, _ = self.la_sympnet.inverse(z, zero_cond)
        return z, torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)


class ConditionedCouplingLayer(nn.Module):
    """Real-NVP coupling layer conditioned on external features.

    ``swap`` controls which half of the data vector conditions the other.
    """

    def __init__(
        self,
        swap: bool,
        data_dim: int,
        cond_dim: int,
        hidden_size: int,
        num_hidden: int = 2,
    ):
        super().__init__()
        self.swap = swap
        self.n_a = (data_dim + 1) // 2  # first chunk
        self.n_b = data_dim // 2  # second chunk
        n_cond = self.n_b if swap else self.n_a
        n_transform = self.n_a if swap else self.n_b

        layers: list[nn.Module] = [
            nn.Linear(n_cond + cond_dim, hidden_size),
            nn.LeakyReLU(),
        ]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.LeakyReLU()]
        layers.append(nn.Linear(hidden_size, n_transform * 2))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.net[-1].weight)

    def _scale_shift(self, inp: torch.Tensor):
        s, t = self.net(inp).chunk(2, dim=-1)
        return s.tanh(), t

    def _split(self, x: torch.Tensor):
        """Split into (x_cond, x_change) respecting swap."""
        x_a, x_b = x.split([self.n_a, self.n_b], dim=-1)
        return (x_b, x_a) if self.swap else (x_a, x_b)

    def _join(self, x_cond: torch.Tensor, y_change: torch.Tensor):
        """Reassemble in original order."""
        return torch.cat(
            [y_change, x_cond] if self.swap else [x_cond, y_change], dim=-1
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """x -> y.  Returns (y, log_det_jac)."""
        x_cond, x_change = self._split(x)
        s, t = self._scale_shift(torch.cat([x_cond, cond], dim=-1))
        y_change = x_change * s.exp() + t
        return self._join(x_cond, y_change), s.sum(dim=-1)

    def inverse(self, y: torch.Tensor, cond: torch.Tensor):
        """y -> x.  Returns (x, log_det_jac_inv)."""
        x_cond, y_change = self._split(y)
        s, t = self._scale_shift(torch.cat([x_cond, cond], dim=-1))
        x_change = (y_change - t) * (-s).exp()
        return self._join(x_cond, x_change), (-s).sum(dim=-1)


class ConditionedRealNVPFlow(nn.Module):
    """Conditional Real-NVP normalizing flow.

    Coordinate and conditioning normalization is stored in buffers and applied in
    log_prob() and sample().
    """

    def __init__(
        self,
        data_dim: int = 2,
        cond_dim: int = 7,
        num_layers: int = 8,
        hidden_size: int = 128,
        num_hidden: int = 2,
        x_mean: np.ndarray | None = None,
        x_std: np.ndarray | None = None,
        cond_mean: np.ndarray | None = None,
        cond_std: np.ndarray | None = None,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.cond_dim = cond_dim

        def _to_buf(arr, default):
            return (
                torch.from_numpy(arr.astype(np.float32)) if arr is not None else default
            )

        self.register_buffer("x_mean_buf", _to_buf(x_mean, torch.zeros(data_dim)))
        self.register_buffer("x_std_buf", _to_buf(x_std, torch.ones(data_dim)))
        self.register_buffer("cond_mean_buf", _to_buf(cond_mean, torch.zeros(cond_dim)))
        self.register_buffer("cond_std_buf", _to_buf(cond_std, torch.ones(cond_dim)))

        self.layers = nn.ModuleList(
            [
                ConditionedCouplingLayer(
                    swap=(i % 2 == 1),
                    data_dim=data_dim,
                    cond_dim=cond_dim,
                    hidden_size=hidden_size,
                    num_hidden=num_hidden,
                )
                for i in range(num_layers)
            ]
        )

    def log_prob(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Log p(x | cond) in original units.

        Parameters
        ----------
        x : Tensor, shape (N, data_dim)
        cond : Tensor, shape (N, cond_dim)

        Returns
        -------
        Tensor, shape (N,)
        """
        x_n = (x - self.x_mean_buf) / self.x_std_buf
        cond_n = (cond - self.cond_mean_buf) / self.cond_std_buf

        log_det = torch.zeros(x_n.shape[0], device=x_n.device)
        z = x_n
        for layer in reversed(self.layers):
            z, ld = layer.inverse(z, cond_n)
            log_det += ld

        # Jacobian of the coordinate normalization: det diag(1/x_std) = prod(1/x_std)
        log_det -= torch.log(self.x_std_buf).sum()

        return _standard_normal_log_prob(z) + log_det

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n_per_cond: int = 1) -> torch.Tensor:
        """Sample from the flow conditioned on features.

        Parameters
        ----------
        cond : Tensor, shape (n_batch, cond_dim)
        n_per_cond : int
            Samples per conditioning vector.

        Returns
        -------
        Tensor, shape (n_batch, n_per_cond, data_dim)
        """
        n_batch = cond.shape[0]
        cond_n = (cond - self.cond_mean_buf) / self.cond_std_buf
        cond_rep = (
            cond_n.unsqueeze(1)
            .expand(n_batch, n_per_cond, -1)
            .reshape(n_batch * n_per_cond, -1)
        )

        z = torch.randn(n_batch * n_per_cond, self.data_dim, device=cond.device)
        x = z
        for layer in self.layers:
            x, _ = layer.forward(x, cond_rep)

        x = x * self.x_std_buf + self.x_mean_buf
        return x.reshape(n_batch, n_per_cond, self.data_dim)


class ConditionedNormalCoordinatesFlow(nn.Module):
    """Flow of unconditioned Real-NVP layers followed by normal coordinates layers.

    Parameters
    ----------
    data_dim : int
        Phase-space dimension (must be even).
    num_nvp_layers : int
        Number of unconditioned coupling layers.
    num_normalizing_layers : int
        Number of ConditionedNormalCoordinatesLayer blocks.
    num_normalizing_hidden : int
        Depth of the LASympNet MLPs in each block.
    nvp_num_hidden : int
        Depth of each coupling MLP.
    cond_dim : int
        Conditioning feature dimension.
    hidden_size : int
        Hidden width of the LASympNet and coupling MLPs.
    rot_hidden_size : int
        Hidden width of the rotation network.
    rot_num_hidden : int
        Number of layers in the rotation network.
    rot_z_dim : int
        Latent dimension of the rotation conditioning MLP.
    rot_cond_hidden_dims : list of int, optional
        Hidden widths of the rotation conditioning MLP.
    la_num_modules : int
        Number of modules in each LASympNet.
    latent_aspect_ratio : float
        Ratio sigma_p / sigma_q of the latent distribution in each phase plane,
        with sigma_q * sigma_p = 1.
    rotation_ablation : {'none', 'no-amplitude', 'no-rotation', 'mlp'}
        Ablation applied to every rotation layer. See ROTATION_ABLATIONS.
    x_mean, x_std : array, optional
        Coordinate normalization.
    cond_mean, cond_std : array, optional
        Conditioning normalization.
    """

    def __init__(
        self,
        data_dim: int = 2,
        num_nvp_layers: int = 4,
        num_normalizing_layers: int = 1,
        num_normalizing_hidden: int = 3,
        nvp_num_hidden: int = 2,
        cond_dim: int = 7,
        hidden_size: int = 128,
        rot_hidden_size: int = 128,
        rot_num_hidden: int = 2,
        rot_z_dim: int = 64,
        rot_cond_hidden_dims: list[int] | None = None,
        la_num_modules: int = 3,
        latent_aspect_ratio: float = 1.0,
        rotation_ablation: str = "none",
        x_mean: np.ndarray | None = None,
        x_std: np.ndarray | None = None,
        cond_mean: np.ndarray | None = None,
        cond_std: np.ndarray | None = None,
    ):
        super().__init__()
        if data_dim % 2 != 0:
            raise ValueError(f"data_dim must be even, got {data_dim}")
        if latent_aspect_ratio <= 0:
            raise ValueError(
                f"latent_aspect_ratio must be positive, got {latent_aspect_ratio}"
            )
        if rotation_ablation not in ROTATION_ABLATIONS:
            raise ValueError(
                f"Unknown rotation_ablation '{rotation_ablation}'. "
                f"Choose from: {list(ROTATION_ABLATIONS)}"
            )
        self.rotation_ablation = rotation_ablation
        self.data_dim = data_dim
        self.n = data_dim // 2
        self.cond_dim = cond_dim
        self.latent_aspect_ratio = latent_aspect_ratio

        def _to_buf(arr, default):
            return (
                torch.from_numpy(arr.astype(np.float32)) if arr is not None else default
            )

        self.register_buffer("x_mean_buf", _to_buf(x_mean, torch.zeros(data_dim)))
        self.register_buffer("x_std_buf", _to_buf(x_std, torch.ones(data_dim)))
        self.register_buffer("cond_mean_buf", _to_buf(cond_mean, torch.zeros(cond_dim)))
        self.register_buffer("cond_std_buf", _to_buf(cond_std, torch.ones(cond_dim)))

        # Unconditioned coupling layers (cond_dim=0 means they only see data)
        self.nvp_layers = nn.ModuleList(
            [
                ConditionedCouplingLayer(
                    swap=(i % 2 == 1),
                    data_dim=data_dim,
                    cond_dim=0,
                    hidden_size=hidden_size,
                    num_hidden=nvp_num_hidden,
                )
                for i in range(num_nvp_layers)
            ]
        )
        # Conditioned symplectic blocks applied last
        self.normal_coordinates_layers = nn.ModuleList(
            [
                ConditionedNormalCoordinatesLayer(
                    n=self.n,
                    cond_dim=cond_dim,
                    hidden_size=hidden_size,
                    num_hidden=num_normalizing_hidden,
                    la_num_modules=la_num_modules,
                    rot_num_hidden=rot_num_hidden,
                    rot_hidden_size=rot_hidden_size,
                    rot_z_dim=rot_z_dim,
                    rot_cond_hidden_dims=rot_cond_hidden_dims,
                    use_rotation=rotation_ablation != "no-rotation",
                    rot_amplitude_dependent=rotation_ablation != "no-amplitude",
                    rot_net_type="mlp" if rotation_ablation == "mlp" else "gradnet",
                )
                for _ in range(num_normalizing_layers)
            ]
        )

        # log-scale shaping the latent base distribution
        half_log_aspect = 0.5 * math.log(latent_aspect_ratio)
        self.register_buffer(
            "log_latent_scale",
            torch.cat(
                [
                    torch.full((self.n,), -half_log_aspect),
                    torch.full((self.n,), half_log_aspect),
                ]
            ),
        )

    def _prepare_cond(self, cond: torch.Tensor) -> torch.Tensor:
        return (cond - self.cond_mean_buf) / self.cond_std_buf

    def log_prob(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Log p(x | cond) in original units.

        Parameters
        ----------
        x : Tensor, shape (N, data_dim)
        cond : Tensor, shape (N, cond_dim)

        Returns
        -------
        Tensor, shape (N,)
        """
        x_n = (x - self.x_mean_buf) / self.x_std_buf
        cond_n = self._prepare_cond(cond)

        log_det = torch.zeros(x_n.shape[0], device=x_n.device)
        empty = torch.empty(x_n.shape[0], 0, device=x_n.device, dtype=x_n.dtype)
        z = x_n
        for layer in reversed(self.normal_coordinates_layers):
            z, ld = layer.inverse(z, cond_n)
            log_det += ld
        for layer in reversed(self.nvp_layers):
            z, ld = layer.inverse(z, empty)
            log_det += ld

        log_det -= torch.log(self.x_std_buf).sum()

        s = self.log_latent_scale  # (data_dim,)
        base_log_prob = -0.5 * ((z / s.exp()).pow(2) + math.log(2 * math.pi)).sum(
            dim=-1
        )
        base_log_prob -= s.sum()  # log det of diagonal covariance: sum(log s)
        return base_log_prob + log_det

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n_per_cond: int = 1) -> torch.Tensor:
        """Sample from the flow conditioned on features.

        Parameters
        ----------
        cond : Tensor, shape (n_batch, cond_dim)
        n_per_cond : int
            Samples per conditioning vector.

        Returns
        -------
        Tensor, shape (n_batch, n_per_cond, data_dim)
        """
        n_batch = cond.shape[0]
        cond_n = self._prepare_cond(cond)
        cond_rep = (
            cond_n.unsqueeze(1)
            .expand(n_batch, n_per_cond, -1)
            .reshape(n_batch * n_per_cond, -1)
        )

        z = torch.randn(n_batch * n_per_cond, self.data_dim, device=cond.device)
        z = z * self.log_latent_scale.exp()
        empty = torch.empty(z.shape[0], 0, device=z.device, dtype=z.dtype)
        x = z
        for layer in self.nvp_layers:
            x, _ = layer.forward(x, empty)
        for layer in self.normal_coordinates_layers:
            x, _ = layer.forward(x, cond_rep)

        x = x * self.x_std_buf + self.x_mean_buf
        return x.reshape(n_batch, n_per_cond, self.data_dim)


class ConditionedLASympNetFlow(nn.Module):
    """Flow of unconditioned Real-NVP layers followed by a conditioned LASympNet.

    Parameters
    ----------
    data_dim : int
        Phase-space dimension (must be even).
    cond_dim : int
        Conditioning feature dimension.
    num_modules : int
        Number of modules in the LASympNet.
    hidden_size : int
        Hidden width of all MLPs.
    num_hidden : int
        Depth of each MLP.
    activation : str
        Activation for the SympNet activation layers.
    num_nvp_layers : int
        Number of unconditioned coupling layers.
    x_mean, x_std : array, optional
        Coordinate normalization.
    cond_mean, cond_std : array, optional
        Conditioning normalization.
    """

    def __init__(
        self,
        data_dim: int = 2,
        cond_dim: int = 1,
        num_modules: int = 4,
        hidden_size: int = 64,
        num_hidden: int = 2,
        activation: str = "tanh",
        num_nvp_layers: int = 0,
        x_mean: np.ndarray | None = None,
        x_std: np.ndarray | None = None,
        cond_mean: np.ndarray | None = None,
        cond_std: np.ndarray | None = None,
    ):
        super().__init__()
        if data_dim % 2 != 0:
            raise ValueError(f"data_dim must be even, got {data_dim}")
        self.data_dim = data_dim

        def _to_buf(arr, default):
            return (
                torch.from_numpy(arr.astype(np.float32)) if arr is not None else default
            )

        self.register_buffer("x_mean_buf", _to_buf(x_mean, torch.zeros(data_dim)))
        self.register_buffer("x_std_buf", _to_buf(x_std, torch.ones(data_dim)))
        self.register_buffer("cond_mean_buf", _to_buf(cond_mean, torch.zeros(cond_dim)))
        self.register_buffer("cond_std_buf", _to_buf(cond_std, torch.ones(cond_dim)))

        self.nvp_layers = nn.ModuleList(
            [
                ConditionedCouplingLayer(
                    swap=(i % 2 == 1),
                    data_dim=data_dim,
                    cond_dim=0,
                    hidden_size=hidden_size,
                    num_hidden=num_hidden,
                )
                for i in range(num_nvp_layers)
            ]
        )
        self.sympnet = LASympNet(
            n=data_dim // 2,
            cond_dim=cond_dim,
            num_modules=num_modules,
            hidden_size=hidden_size,
            num_hidden=num_hidden,
            activation=activation,
        )

    def _prepare_cond(self, cond: torch.Tensor) -> torch.Tensor:
        return (cond - self.cond_mean_buf) / self.cond_std_buf

    def log_prob(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x_n = (x - self.x_mean_buf) / self.x_std_buf
        cond_n = self._prepare_cond(cond)
        log_det = -torch.log(self.x_std_buf).sum()
        z = x_n
        empty = torch.empty(x_n.shape[0], 0, device=x_n.device, dtype=x_n.dtype)
        for layer in reversed(self.nvp_layers):
            z, ld = layer.inverse(z, empty)
            log_det = log_det + ld
        z, _ = self.sympnet.inverse(z, cond_n)
        return _standard_normal_log_prob(z) + log_det

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n_per_cond: int = 1) -> torch.Tensor:
        n_batch = cond.shape[0]
        cond_n = self._prepare_cond(cond)
        cond_rep = (
            cond_n.unsqueeze(1)
            .expand(n_batch, n_per_cond, -1)
            .reshape(n_batch * n_per_cond, -1)
        )
        z = torch.randn(n_batch * n_per_cond, self.data_dim, device=cond.device)
        empty = torch.empty(z.shape[0], 0, device=z.device, dtype=z.dtype)
        x_n, _ = self.sympnet.forward(z, cond_rep)
        for layer in self.nvp_layers:
            x_n, _ = layer.forward(x_n, empty)
        x = x_n * self.x_std_buf + self.x_mean_buf
        return x.reshape(n_batch, n_per_cond, self.data_dim)


class SymplecticRotationLayer(nn.Module):
    """Per-plane rotation with angles computed from the actions.

    Phase space is ordered (q_1, ..., q_n, p_1, ..., p_n).  The actions are
    invariant under the rotation, so the inverse is exact and log_det = 0.

    Parameters
    ----------
    n : int
        Number of phase-space planes. Phase-space dimension is 2n.
    cond_dim : int
        Conditioning feature dimension.
    num_layers : int
        Number of layers in the angle network.
    hidden_dim : int
        Hidden width of the angle network.
    cond_latent_dim : int
        Output dimension of the conditioning MLP.
    cond_hidden_dims : list of int
        Hidden widths of the conditioning MLP.
    activation : str
        Activation for the angle network.
    cond_activation : str
        Activation for the conditioning MLP.
    amplitude_dependent : bool
        If False, the angles depend on the conditioning features only.
    net_type : {'gradnet', 'mlp'}
        Network producing the angles.  Only 'gradnet' is symplectic.
    """

    def __init__(
        self,
        n: int,
        cond_dim: int,
        num_layers,
        hidden_dim,
        cond_latent_dim: int,
        cond_hidden_dims: list[int],
        activation: str = "leaky_relu",
        cond_activation: str = "leaky_relu",
        amplitude_dependent: bool = True,
        net_type: str = "gradnet",
    ):
        super().__init__()
        self.n = n
        self.amplitude_dependent = amplitude_dependent

        if net_type == "gradnet":
            net_cls = CondGradNet
        elif net_type == "mlp":
            net_cls = CondMLP
        else:
            raise ValueError(
                f"Unknown net_type '{net_type}'. Choose from: gradnet, mlp"
            )

        self.net = net_cls(
            grad_dim=n,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            cond_dim=cond_dim,
            cond_latent_dim=cond_latent_dim,
            cond_hidden_dims=cond_hidden_dims,
            activation=activation,
            cond_activation=cond_activation,
        )
        self.log_scale = nn.Parameter(torch.full((n,), math.log(2 * math.pi)))

    def _actions(self, z: torch.Tensor) -> torch.Tensor:
        if not self.amplitude_dependent:
            return z.new_zeros(z.shape[0], self.n)
        q, p = z[:, : self.n], z[:, self.n :]
        return q * q + p * p  # (n_batch, n); invariant under per-plane rotation

    def _rotate(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        n = self.n
        q, p = z[:, :n], z[:, n:]
        c, s = theta.cos(), theta.sin()
        return torch.cat([c * q - s * p, s * q + c * p], dim=-1)

    def forward(
        self, z: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """z -> rotate by theta(actions, cond) -> y.  log_det = 0."""
        theta = self.net(self._actions(z), cond) * self.log_scale.exp()
        return self._rotate(z, theta), torch.zeros(
            z.shape[0], device=z.device, dtype=z.dtype
        )

    def inverse(
        self, z: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """y -> rotate by -theta(actions, cond) -> z.  log_det = 0."""
        theta = self.net(self._actions(z), cond) * self.log_scale.exp()
        return self._rotate(z, -theta), torch.zeros(
            z.shape[0], device=z.device, dtype=z.dtype
        )
