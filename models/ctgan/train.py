"""
models/ctgan/train.py — CTGAN training loop for NHANES data.

Usage:
  python models/ctgan/train.py              # train with default seed (42)
  python models/ctgan/train.py --seed 43    # train with specific seed
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config
from models.ctgan.model import Discriminator, Generator, gradient_penalty
from utils.seed_utils import set_all_seeds

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).float()


def train_ctgan(seed: int = config.RANDOM_SEED) -> Path:
    """
    Train CTGAN on NHANES training data and save a seed-specific checkpoint.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Path to the final checkpoint
    """
    set_all_seeds(seed)
    device = torch.device(config.DEVICE)
    log.info("CTGAN training | seed=%d | device=%s", seed, device)

    train_df = pd.read_csv(config.NHANES_TRAIN)
    feature_cols = [c for c in train_df.columns if c != config.TARGET_COL]
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[config.TARGET_COL].values.astype(np.int64)

    input_dim = X_train.shape[1]
    cond_dim  = config.NUM_CLASSES
    log.info("Input dim: %d  |  Cond dim: %d  |  Train: %d",
             input_dim, cond_dim, len(X_train))

    gen  = Generator(
        config.CTGAN_NOISE_DIM, cond_dim, input_dim, config.CTGAN_GEN_DIM
    ).to(device)
    disc = Discriminator(input_dim, cond_dim, config.CTGAN_DISC_DIM).to(device)

    opt_g = torch.optim.Adam(gen.parameters(),  lr=config.CTGAN_LR, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(), lr=config.CTGAN_LR, betas=(0.5, 0.9))

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=config.CTGAN_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    ckpt_path = config.MODELS_DIR / f"ctgan_seed{seed}.pt"
    log_rows: list[dict] = []

    for epoch in range(1, config.CTGAN_EPOCHS + 1):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        n_batches = 0

        for real_x, real_y in loader:
            real_x = real_x.to(device)
            real_y = real_y.to(device)
            cond   = _one_hot(real_y, cond_dim)
            bs     = real_x.size(0)

            # Discriminator — CTGAN_DISC_STEPS per generator step
            for _ in range(config.CTGAN_DISC_STEPS):
                noise  = torch.randn(bs, config.CTGAN_NOISE_DIM, device=device)
                fake_x = gen(noise, cond).detach()
                real_s = disc(real_x, cond)
                fake_s = disc(fake_x, cond)
                gp     = gradient_penalty(disc, real_x, fake_x, cond, device)
                d_loss = fake_s.mean() - real_s.mean() + config.CTGAN_GRADIENT_PENALTY * gp
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

            # Generator — 1 step
            noise  = torch.randn(bs, config.CTGAN_NOISE_DIM, device=device)
            fake_x = gen(noise, cond)
            g_loss = -disc(fake_x, cond).mean()
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            n_batches += 1

        epoch_d_loss /= n_batches
        epoch_g_loss /= n_batches
        log_rows.append({"epoch": epoch, "d_loss": epoch_d_loss, "g_loss": epoch_g_loss})

        if epoch % 50 == 0 or epoch == 1:
            log.info("Epoch %3d/%d  D=%.4f  G=%.4f",
                     epoch, config.CTGAN_EPOCHS, epoch_d_loss, epoch_g_loss)

        if epoch % 50 == 0 or epoch == config.CTGAN_EPOCHS:
            torch.save({
                "epoch":       epoch,
                "gen_state":   gen.state_dict(),
                "disc_state":  disc.state_dict(),
                "input_dim":   input_dim,
                "cond_dim":    cond_dim,
                "feature_cols": feature_cols,
                "seed":        seed,
            }, ckpt_path)

    # Save epoch log
    log_path = config.LOGS_DIR / f"ctgan_seed{seed}.csv"
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    log.info("Final checkpoint (epoch %d): %s", config.CTGAN_EPOCHS, ckpt_path)
    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()
    train_ctgan(seed=args.seed)
