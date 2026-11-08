import torch
import torch.nn as nn

from normal_coordinates_flow.models.utils import ACTIVATIONS


class ConditionedLinearSymplecticLayer(nn.Module):
    """Conditioned linear symplectic layer for 2n-dimensional phase space.

    Applies M = U_D @ L_4 @ U_3 @ L_2 @ U_1 right-to-left, where
    U_k = [[I, Sk], [0, I]], L_k = [[I, 0], [Sk, I]] with Sk symmetric, and
    U_D = [[I, D], [0, I]] with D diagonal.

    As in: Jin, P., Lin, Z., & Xiao, B. (2022). Optimal unit triangular
    factorization of symplectic matrices (arXiv:2108.00223). arXiv.
    https://doi.org/10.48550/arXiv.2108.00223


    Parameters
    ----------
    n : int
        Number of phase-space planes. Phase-space dimension is 2n.
    cond_dim : int
        Conditioning feature dimension.
    hidden_size : int
        Hidden width of the conditioning MLP.
    num_hidden : int
        Depth of the conditioning MLP.
    """

    def __init__(
        self,
        n: int,
        cond_dim: int,
        hidden_size: int = 128,
        num_hidden: int = 2,
    ):
        super().__init__()
        self.n = n
        n_sq = n * n
        n_params = 4 * n_sq + n  # S1, S2, S3, S4 (n^2 each) + D (diagonal)

        layers: list[nn.Module] = [nn.Linear(cond_dim, hidden_size), nn.LeakyReLU()]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.LeakyReLU()]
        layers.append(nn.Linear(hidden_size, n_params))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _to_symmetric(self, flat: torch.Tensor) -> torch.Tensor:
        """(B, n^2) -> (B, n, n) symmetric matrix via M + M^T."""
        mat = flat.reshape(flat.shape[0], self.n, self.n)
        return mat + mat.transpose(1, 2)

    def _get_matrices(
        self, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(cond)  # (B, n_params)
        n_sq = self.n * self.n
        sym1 = self._to_symmetric(out[:, 0 * n_sq : 1 * n_sq])
        sym2 = self._to_symmetric(out[:, 1 * n_sq : 2 * n_sq])
        sym3 = self._to_symmetric(out[:, 2 * n_sq : 3 * n_sq])
        sym4 = self._to_symmetric(out[:, 3 * n_sq : 4 * n_sq])
        diag = out[:, 4 * n_sq :]  # (B, n)
        return sym1, sym2, sym3, sym4, diag

    def forward(
        self, z: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """z -> M(cond) -> y.  log_det = 0."""
        n = self.n
        q, p = z[:, :n], z[:, n:]
        sym1, sym2, sym3, sym4, diag = self._get_matrices(cond)

        q = q + torch.bmm(sym1, p.unsqueeze(-1)).squeeze(-1)  # U1
        p = p + torch.bmm(sym2, q.unsqueeze(-1)).squeeze(-1)  # L2
        q = q + torch.bmm(sym3, p.unsqueeze(-1)).squeeze(-1)  # U3
        p = p + torch.bmm(sym4, q.unsqueeze(-1)).squeeze(-1)  # L4
        q = q + diag * p  # U_D

        return torch.cat([q, p], dim=-1), torch.zeros(
            z.shape[0], device=z.device, dtype=z.dtype
        )

    def inverse(
        self, z: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """y -> M_inv(cond) -> z.  log_det = 0."""
        n = self.n
        q, p = z[:, :n], z[:, n:]
        sym1, sym2, sym3, sym4, diag = self._get_matrices(cond)

        q = q - diag * p  # U_D inv
        p = p - torch.bmm(sym4, q.unsqueeze(-1)).squeeze(-1)  # L4 inv
        q = q - torch.bmm(sym3, p.unsqueeze(-1)).squeeze(-1)  # U3 inv
        p = p - torch.bmm(sym2, q.unsqueeze(-1)).squeeze(-1)  # L2 inv
        q = q - torch.bmm(sym1, p.unsqueeze(-1)).squeeze(-1)  # U1 inv

        return torch.cat([q, p], dim=-1), torch.zeros(
            z.shape[0], device=z.device, dtype=z.dtype
        )


class SympNetActivationLayer(nn.Module):
    """SympNet activation layer (Q or P variant) from Jin et al. 2020.

    Q variant: (q, p) -> (q + a * sigma(p), p)
    P variant: (q, p) -> (q, p + a * sigma(q))

    Parameters
    ----------
    n : int
        Number of phase-space planes. Phase-space dimension is 2n.
    kind : str
        "Q" to update q, "P" to update p.
    activation : str
        Elementwise activation sigma.
    """

    def __init__(self, n: int, kind: str = "Q", activation: str = "tanh"):
        super().__init__()
        if kind not in ("Q", "P"):
            raise ValueError(f"kind must be 'Q' or 'P', got '{kind}'")
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. Choose from: {list(ACTIVATIONS)}"
            )
        self.n = n
        self.kind = kind
        self.a = nn.Parameter(torch.zeros(n))
        self.sigma = ACTIVATIONS[activation]()

    def _apply_layer(self, z: torch.Tensor, sign: float) -> torch.Tensor:
        q, p = z[:, : self.n], z[:, self.n :]
        if self.kind == "Q":
            q = q + sign * self.a * self.sigma(p)
        else:
            p = p + sign * self.a * self.sigma(q)
        return torch.cat([q, p], dim=-1)

    def forward(
        self, z: torch.Tensor, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(q, p) -> (q + a*sigma(p), p) or (q, p + a*sigma(q))."""
        return self._apply_layer(z, 1.0), torch.zeros(
            z.shape[0], device=z.device, dtype=z.dtype
        )

    def inverse(
        self, z: torch.Tensor, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact inverse via sign flip."""
        return self._apply_layer(z, -1.0), torch.zeros(
            z.shape[0], device=z.device, dtype=z.dtype
        )


class LASympNet(nn.Module):
    """LA-SympNet: alternating linear-activation blocks from Jin et al. 2020.

    Applies num_modules (linear, Q, P) blocks, then one final linear layer.

    Parameters
    ----------
    n : int
        Number of phase-space planes. Phase-space dimension is 2n.
    cond_dim : int
        Conditioning feature dimension.
    num_modules : int
        Number of (linear, Q, P) blocks before the final linear layer.
    hidden_size : int
        Hidden width of the conditioning MLP in each linear layer.
    num_hidden : int
        Depth of the conditioning MLP in each linear layer.
    activation : str
        Activation for the SympNet activation layers.
    """

    def __init__(
        self,
        n: int,
        cond_dim: int,
        num_modules: int = 3,
        hidden_size: int = 128,
        num_hidden: int = 2,
        activation: str = "tanh",
    ):
        super().__init__()
        self.n = n
        self.num_modules = num_modules

        self.linear_layers = nn.ModuleList(
            [
                ConditionedLinearSymplecticLayer(
                    n=n,
                    cond_dim=cond_dim,
                    hidden_size=hidden_size,
                    num_hidden=num_hidden,
                )
                for _ in range(num_modules + 1)
            ]
        )
        self.act_q = nn.ModuleList(
            [
                SympNetActivationLayer(n=n, kind="Q", activation=activation)
                for _ in range(num_modules)
            ]
        )
        self.act_p = nn.ModuleList(
            [
                SympNetActivationLayer(n=n, kind="P", activation=activation)
                for _ in range(num_modules)
            ]
        )

    def forward(
        self, z: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: num_modules blocks then final linear layer."""
        for i in range(self.num_modules):
            z, _ = self.linear_layers[i].forward(z, cond)
            z, _ = self.act_q[i].forward(z)
            z, _ = self.act_p[i].forward(z)
        z, _ = self.linear_layers[-1].forward(z, cond)
        return z, torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

    def inverse(
        self, z: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact inverse: reverse layer order, invert each."""
        z, _ = self.linear_layers[-1].inverse(z, cond)
        for i in reversed(range(self.num_modules)):
            z, _ = self.act_p[i].inverse(z)
            z, _ = self.act_q[i].inverse(z)
            z, _ = self.linear_layers[i].inverse(z, cond)
        return z, torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
