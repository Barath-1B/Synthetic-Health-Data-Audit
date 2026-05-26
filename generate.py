"""
generate.py — Generate synthetic NHANES rows from a trained VAE or CTGAN.

Usage:
  python generate.py --model vae   --n 1000
  python generate.py --model ctgan --n 1000
  python generate.py --model vae   --n 1000 --out data/synthetic/my_output.csv
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config
from models.vae.model   import VAE
from models.ctgan.model import Generator

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _load_scaler() -> tuple[object, list[str]]:
    with open(config.SCALER_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["scaler"], bundle["continuous_cols"]


def _col_order() -> list[str]:
    """Return the exact column order of nhanes_clean.csv (excluding target)."""
    df = pd.read_csv(config.NHANES_CLEAN, nrows=0)
    return [c for c in df.columns if c != config.TARGET_COL]


def _inverse_transform(tensor: torch.Tensor, col_order: list[str],
                       scaler: object, cont_cols: list[str]) -> pd.DataFrame:
    """Convert raw model output (all [0,1]) back to a DataFrame."""
    arr = tensor.cpu().numpy()
    df  = pd.DataFrame(arr, columns=col_order)

    # Inverse-scale only continuous columns
    present = [c for c in cont_cols if c in df.columns]
    df[present] = scaler.inverse_transform(df[present])

    # Round integer-typed columns
    int_cols = [c for c in col_order if c not in cont_cols]
    for c in int_cols:
        df[c] = df[c].round().astype(int)

    return df


# ── VAE generation ────────────────────────────────────────────────────────────

def generate_vae(n: int, device: torch.device) -> pd.DataFrame:
    ckpt_path = config.MODELS_DIR / "vae_nhanes.pt"
    ckpt      = torch.load(ckpt_path, map_location=device)
    input_dim = ckpt["input_dim"]

    model = VAE(input_dim, config.VAE_HIDDEN_DIMS, config.VAE_LATENT_DIM).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scaler, cont_cols = _load_scaler()
    col_order         = _col_order()

    with torch.no_grad():
        synthetic = model.sample(n, device)

    df = _inverse_transform(synthetic, col_order, scaler, cont_cols)

    # Derive Depression_Severity from PHQ9_TOTAL (re-scale to 0-27, then categorise)
    if "PHQ9_TOTAL" in df.columns and config.TARGET_COL not in df.columns:
        raw = (df["PHQ9_TOTAL"] * 27).round().clip(0, 27).astype(int)
        df[config.TARGET_COL] = raw.apply(
            lambda s: 0 if s <= 4 else 1 if s <= 9 else 2 if s <= 14 else 3 if s <= 19 else 4
        )

    return df


# ── CTGAN generation ──────────────────────────────────────────────────────────

def generate_ctgan(n: int, device: torch.device) -> pd.DataFrame:
    ckpt_path = config.MODELS_DIR / "ctgan_nhanes.pt"
    ckpt      = torch.load(ckpt_path, map_location=device)
    input_dim = ckpt["input_dim"]
    cond_dim  = ckpt["cond_dim"]

    gen = Generator(config.CTGAN_NOISE_DIM, cond_dim, input_dim, config.CTGAN_GEN_DIM).to(device)
    gen.load_state_dict(ckpt["gen_state"])
    gen.eval()

    scaler, cont_cols = _load_scaler()
    col_order         = _col_order()

    # Sample conditions proportional to real class distribution
    real_df    = pd.read_csv(config.NHANES_CLEAN)
    class_dist = real_df[config.TARGET_COL].value_counts(normalize=True).sort_index()
    labels     = np.random.choice(class_dist.index, size=n, p=class_dist.values)

    with torch.no_grad():
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)
        cond     = F.one_hot(labels_t, num_classes=cond_dim).float()
        noise    = torch.randn(n, config.CTGAN_NOISE_DIM, device=device)
        synthetic = gen(noise, cond)

    df = _inverse_transform(synthetic, col_order, scaler, cont_cols)
    df[config.TARGET_COL] = labels

    return df


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic NHANES data.")
    parser.add_argument("--model", choices=["vae", "ctgan"], required=True)
    parser.add_argument("--n",    type=int, default=1000, help="Number of rows to generate")
    parser.add_argument("--out",  type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Generating %d rows with %s on %s ...", args.n, args.model, device)

    if args.model == "vae":
        df = generate_vae(args.n, device)
    else:
        df = generate_ctgan(args.n, device)

    out_path = Path(args.out) if args.out else (
        config.DATA_SYNTHETIC / f"{args.model}_nhanes_synthetic.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Saved %d rows -> %s", len(df), out_path)
    log.info("Shape: %s", df.shape)


if __name__ == "__main__":
    main()
