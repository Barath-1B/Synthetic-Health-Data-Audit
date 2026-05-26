"""
models/vae/train.py — VAE training loop for NHANES data.

Usage:
  python models/vae/train.py
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config
from models.vae.model import VAE, vae_loss

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)


def load_splits() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X_train, X_val, y_train, y_val as numpy arrays."""
    import pandas as pd
    df = pd.read_csv(config.NHANES_CLEAN)
    with open(config.SPLITS_PATH, "rb") as f:
        splits = pickle.load(f)
    feature_cols = [c for c in df.columns if c != config.TARGET_COL]
    X = df[feature_cols].values.astype(np.float32)
    y = df[config.TARGET_COL].values
    return (
        X[splits["train"]], X[splits["val"]],
        y[splits["train"]], y[splits["val"]],
    )


def make_loader(X: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    t = torch.tensor(X, dtype=torch.float32)
    return DataLoader(TensorDataset(t), batch_size=batch_size, shuffle=shuffle)


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    X_train, X_val, _, _ = load_splits()
    input_dim = X_train.shape[1]
    log.info("Input dim: %d  |  Train: %d  Val: %d", input_dim, len(X_train), len(X_val))

    model = VAE(input_dim, config.VAE_HIDDEN_DIMS, config.VAE_LATENT_DIM).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=config.VAE_LR)

    train_loader = make_loader(X_train, config.VAE_BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(X_val,   config.VAE_BATCH_SIZE, shuffle=False)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    ckpt_path     = config.MODELS_DIR / "vae_nhanes.pt"

    for epoch in range(1, config.VAE_EPOCHS + 1):
        kl_weight = min(1.0, epoch / config.VAE_KL_ANNEAL_EPOCHS)

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss, _, _ = vae_loss(recon, batch, mu, logvar, kl_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(batch)
        train_loss /= len(X_train)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                loss, _, _ = vae_loss(recon, batch, mu, logvar, kl_weight)
                val_loss += loss.item() * len(batch)
        val_loss /= len(X_val)

        if epoch % 10 == 0 or epoch == 1:
            log.info("Epoch %3d/%d  train=%.4f  val=%.4f  kl_w=%.2f",
                     epoch, config.VAE_EPOCHS, train_loss, val_loss, kl_weight)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "input_dim": input_dim}, ckpt_path)

    log.info("Best val loss: %.4f  |  Checkpoint: %s", best_val_loss, ckpt_path)


if __name__ == "__main__":
    train()
