"""
models/ctgan/model.py — Conditional Tabular GAN with WGAN-GP.

Architecture:
  Generator:     noise + one-hot(condition) -> synthetic row
  Discriminator: row + one-hot(condition)   -> scalar (no sigmoid; WGAN)
  Training:      WGAN-GP (gradient penalty, no weight clipping)
"""

import torch
import torch.nn as nn
from typing import List, Tuple


def _fc_block(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU())


class Generator(nn.Module):
    """Maps (noise, condition) -> synthetic tabular row."""

    def __init__(
        self,
        noise_dim: int,
        cond_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...],
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = noise_dim + cond_dim
        for h in hidden_dims:
            layers.append(_fc_block(prev, h))
            prev = h
        layers += [nn.Linear(prev, output_dim), nn.Sigmoid()]  # output in [0,1]
        self.net = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([noise, cond], dim=1))


class Discriminator(nn.Module):
    """Maps (row, condition) -> unbounded scalar (WGAN, no sigmoid)."""

    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        hidden_dims: Tuple[int, ...],
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim + cond_dim
        for h in hidden_dims:
            # No BatchNorm in discriminator for WGAN-GP stability
            layers += [nn.Linear(prev, h), nn.LeakyReLU(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, cond], dim=1))


def gradient_penalty(
    disc: Discriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
    cond: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """WGAN-GP gradient penalty on interpolated samples."""
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = disc(interp, cond)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()
