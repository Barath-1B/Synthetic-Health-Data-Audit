"""
models/vae/train.py — VAE training loop for NHANES data.

Usage:
  python models/vae/train.py              # train with default seed (42)
  python models/vae/train.py --seed 43    # train with specific seed
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config
from models.vae.model import VAE, vae_loss
from utils.seed_utils import set_all_seeds

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def get_binary_col_indices(df: pd.DataFrame, target_col: str) -> list[int]:
    """
    Identify feature columns that are binary (only 0 and 1 after scaling).
    Returns their positional indices in the feature array.
    """
    feature_cols = [c for c in df.columns if c != target_col]
    binary_idx = []
    for i, col in enumerate(feature_cols):
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            binary_idx.append(i)
    return binary_idx


def _make_loader(X: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    t = torch.tensor(X, dtype=torch.float32)
    return DataLoader(TensorDataset(t), batch_size=batch_size, shuffle=shuffle,
                      drop_last=True)


def train_vae(seed: int = config.RANDOM_SEED) -> Path:
    """
    Train VAE on NHANES training data and save a seed-specific checkpoint.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Path to the best-val-loss checkpoint
    """
    set_all_seeds(seed)
    device = torch.device(config.DEVICE)
    log.info("VAE training | seed=%d | device=%s", seed, device)

    train_df = pd.read_csv(config.NHANES_TRAIN)
    val_df   = pd.read_csv(config.NHANES_VAL)

    feature_cols = [c for c in train_df.columns if c != config.TARGET_COL]
    X_train = train_df[feature_cols].values.astype(np.float32)
    X_val   = val_df[feature_cols].values.astype(np.float32)
    log.info("Input dim: %d  |  Train: %d  Val: %d", X_train.shape[1], len(X_train), len(X_val))

    binary_idx = get_binary_col_indices(train_df, config.TARGET_COL)
    log.info("Binary columns: %d / %d", len(binary_idx), len(feature_cols))

    model = VAE(
        input_dim=X_train.shape[1],
        binary_col_indices=binary_idx,
        hidden_dims=config.VAE_HIDDEN_DIMS,
        latent_dim=config.VAE_LATENT_DIM,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=config.VAE_LR)
    train_loader = _make_loader(X_train, config.VAE_BATCH_SIZE, shuffle=True)
    val_loader   = _make_loader(X_val,   config.VAE_BATCH_SIZE, shuffle=False)

    ckpt_path  = config.MODELS_DIR / f"vae_seed{seed}.pt"
    log_rows: list[dict] = []
    best_val_loss = float("inf")

    for epoch in range(1, config.VAE_EPOCHS + 1):
        beta = min(config.VAE_KL_MAX_WEIGHT,
                   epoch / config.VAE_KL_ANNEAL_EPOCHS * config.VAE_KL_MAX_WEIGHT)

        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss, _, _ = vae_loss(recon, batch, mu, logvar, kl_weight=beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(batch)
        train_loss /= len(X_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                loss, _, _ = vae_loss(recon, batch, mu, logvar, kl_weight=beta)
                val_loss += loss.item() * len(batch)
        val_loss /= len(X_val)

        log_rows.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "beta": beta})

        if epoch % 20 == 0 or epoch == 1:
            log.info("Epoch %3d/%d  train=%.4f  val=%.4f  beta=%.3f",
                     epoch, config.VAE_EPOCHS, train_loss, val_loss, beta)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "input_dim": X_train.shape[1],
                "binary_col_indices": binary_idx,
                "feature_cols": feature_cols,
                "seed": seed,
            }, ckpt_path)

    # Save epoch log
    log_path = config.LOGS_DIR / f"vae_seed{seed}.csv"
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    log.info("Best val loss: %.4f | Checkpoint: %s", best_val_loss, ckpt_path)
    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()
    train_vae(seed=args.seed)
