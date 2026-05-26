"""
models/vae/model.py — Variational Autoencoder for tabular health data.

Architecture:
  Encoder: FC layers -> (mu, logvar)
  Reparameterisation trick: z = mu + sigma * eps
  Decoder: FC layers -> reconstructed feature vector
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class Encoder(nn.Module):
    """Fully-connected encoder producing (mu, logvar) for reparameterisation."""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.net    = nn.Sequential(*layers)
        self.mu     = nn.Linear(prev, latent_dim)
        self.logvar = nn.Linear(prev, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    """Fully-connected decoder reconstructing the feature vector."""

    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, output_dim), nn.Sigmoid()]  # outputs in [0,1]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAE(nn.Module):
    """Variational Autoencoder combining Encoder and Decoder."""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int) -> None:
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)
        self.latent_dim = latent_dim

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = mu + sigma * eps; eps ~ N(0, I)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at inference

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z          = self.reparameterise(mu, logvar)
        recon      = self.decoder(z)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample n rows from the prior N(0, I)."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MSE reconstruction loss + KL divergence (with optional annealing weight)."""
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total      = recon_loss + kl_weight * kl_loss
    return total, recon_loss, kl_loss
