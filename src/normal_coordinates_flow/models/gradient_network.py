import torch
import torch.nn as nn

from normal_coordinates_flow.models.utils import ACTIVATIONS as _ACTIVATIONS


def _build_mlp(
    in_dim: int,
    hidden_dims: list[int],
    out_dim: int,
    activation: str = "leaky_relu",
) -> nn.Sequential:
    act_cls = _ACTIVATIONS[activation]
    dims = [in_dim] + hidden_dims
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]), act_cls()]
    layers.append(nn.Linear(dims[-1], out_dim))
    seq = nn.Sequential(*layers)
    for m in seq.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return seq


class CondGradNet(nn.Module):
    def __init__(
        self,
        grad_dim,
        cond_dim,
        hidden_dim=32,
        num_layers=4,
        activation="leaky_relu",
        cond_activation="leaky_relu",
        cond_latent_dim=32,
        cond_hidden_dims=None,
    ):
        super().__init__()
        if cond_hidden_dims is None:
            cond_hidden_dims = 4 * [32]
        self.w = nn.Parameter(torch.empty((grad_dim, hidden_dim)))
        self.b = nn.ModuleList(
            [nn.Linear(cond_latent_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.bl = nn.Linear(cond_latent_dim, grad_dim)

        self.cond_mlp = _build_mlp(
            cond_dim, cond_hidden_dims, cond_latent_dim, cond_activation
        )
        self.alpha = nn.ModuleList(
            [nn.Linear(cond_latent_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.beta = nn.ModuleList(
            [nn.Linear(cond_latent_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.sigma = _ACTIVATIONS[activation]()

        # Initialize model
        nn.init.kaiming_normal_(self.w)
        for b in self.b:
            nn.init.zeros_(b.bias)
            nn.init.zeros_(b.weight)
        for alpha in self.alpha:
            nn.init.uniform_(alpha.bias, a=-0.5, b=0.5)
            nn.init.zeros_(alpha.weight)
        for beta in self.beta:
            nn.init.uniform_(beta.bias, a=-0.5, b=0.5)
            nn.init.zeros_(beta.weight)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = self.cond_mlp(cond)
        z = self.beta[0](y) * (x @ self.w) + self.b[0](y)
        for beta, alpha, b in zip(self.beta[1:], self.alpha[:-1], self.b[1:]):
            z = beta(y) * (x @ self.w) + alpha(y) * self.sigma(z) + b(y)
        z = (self.alpha[-1](y) * self.sigma(z)) @ self.w.T + self.bl(y)
        return z


class CondMLP(nn.Module):
    """Conditioned MLP with the CondGradNet call signature.

    Parameters
    ----------
    grad_dim : int
        Input and output dimension.
    cond_dim : int
        Conditioning feature dimension.
    hidden_dim : int
        Hidden width of the main MLP.
    num_layers : int
        Number of hidden layers in the main MLP.
    activation : str
        Activation for the main MLP.
    cond_activation : str
        Activation for the conditioning MLP.
    cond_latent_dim : int
        Output dimension of the conditioning MLP.
    cond_hidden_dims : list of int, optional
        Hidden widths of the conditioning MLP.
    """

    def __init__(
        self,
        grad_dim,
        cond_dim,
        hidden_dim=32,
        num_layers=4,
        activation="leaky_relu",
        cond_activation="leaky_relu",
        cond_latent_dim=32,
        cond_hidden_dims=None,
    ):
        super().__init__()
        if cond_hidden_dims is None:
            cond_hidden_dims = 4 * [32]

        self.cond_mlp = _build_mlp(
            cond_dim, cond_hidden_dims, cond_latent_dim, cond_activation
        )
        self.net = _build_mlp(
            grad_dim + cond_latent_dim,
            num_layers * [hidden_dim],
            grad_dim,
            activation,
        )

        # Start as the identity map, matching the near-identity initialisation of
        # CondGradNet's zeroed output heads.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.cond_mlp(cond)], dim=-1))
