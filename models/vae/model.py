"""
models/vae/model.py — VAE with column-type-aware decoder output.

Architecture:
  Encoder: FC + BatchNorm1d + LeakyReLU -> (mu, logvar)
  Decoder: FC + BatchNorm1d + LeakyReLU -> per-column activation:
           - sigmoid for binary columns (only 0/1 after scaling)
           - clamp(x, 0, 1) for continuous columns (avoids sigmoid saturation)
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class Encoder(nn.Module):
    """Fully-connected encoder with BatchNorm, producing (mu, logvar)."""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2)]
            prev = h
        self.net    = nn.Sequential(*layers)
        self.mu     = nn.Linear(prev, latent_dim)
        self.logvar = nn.Linear(prev, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    """
    Fully-connected decoder with column-type-aware output activation.

    Binary columns receive sigmoid (output in [0,1] via saturation).
    Continuous columns receive clamp(x, 0, 1) — linear in the valid range,
    avoiding sigmoid's saturation artefacts on wide-range features.
    """

    def __init__(self, latent_dim: int, hidden_dims: List[int],
                 output_dim: int, binary_col_indices: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_layer = nn.Linear(prev, output_dim)
        self.binary_idx = set(binary_col_indices)
        self.output_dim = output_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.out_layer(self.net(z))
        # Per-column activation: sigmoid for binary, clamp for continuous
        parts = []
        for i in range(self.output_dim):
            if i in self.binary_idx:
                parts.append(torch.sigmoid(h[:, i : i + 1]))
            else:
                parts.append(torch.clamp(h[:, i : i + 1], 0.0, 1.0))
        return torch.cat(parts, dim=1)


class VAE(nn.Module):
    """Variational Autoencoder combining Encoder and Decoder."""

    def __init__(self, input_dim: int, binary_col_indices: List[int],
                 hidden_dims: List[int], latent_dim: int) -> None:
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim, binary_col_indices)
        self.latent_dim = latent_dim

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = mu + sigma * eps, eps ~ N(0, I)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z          = self.reparameterise(mu, logvar)
        recon      = self.decoder(z)
        return recon, mu, logvar

    @torch.no_grad()
    def generate(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample n rows from prior N(0, I)."""
        self.eval()
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MSE reconstruction loss + KL divergence with annealing weight."""
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total      = recon_loss + kl_weight * kl_loss
    return total, recon_loss, kl_loss
