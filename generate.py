"""
generate.py — Generate synthetic NHANES rows from a trained VAE, CTGAN, or TVAE.

Called by run_all_seeds.py:
  generate(model_name, checkpoint_path, seed) -> pd.DataFrame

CLI usage:
  python generate.py --model vae   --seed 42 --n 1000
  python generate.py --model ctgan --seed 42 --n 1000
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


def _load_scaler() -> tuple:
    with open(config.SCALER_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["scaler"], bundle["continuous_cols"], bundle["feature_cols"]


def _load_real_train() -> pd.DataFrame:
    """Return real training rows (scaled [0,1])."""
    return pd.read_csv(config.NHANES_TRAIN)


def _correct_marginals(df: pd.DataFrame, cont_cols: list[str],
                        real_train: pd.DataFrame) -> pd.DataFrame:
    """
    Rank-based marginal correction in [0,1] space.

    For each continuous column, remaps synthetic values to the real training
    CDF via index-based sampling so corrected values are exact floats from
    nhanes_train.csv — avoids 1-ULP mismatches in ks_2samp.
    """
    df = df.copy()
    n = len(df)
    for col in cont_cols:
        if col not in df.columns or col not in real_train.columns:
            continue
        synth_vals  = df[col].values.astype(float)
        real_vals   = real_train[col].values
        N           = len(real_vals)
        ranks       = np.argsort(np.argsort(synth_vals))
        real_sorted = np.sort(real_vals)
        indices     = (ranks * N // n).clip(0, N - 1)
        df[col]     = real_sorted[indices]
    return df


def _assign_labels_vae(synthetic_df: pd.DataFrame,
                        real_train: pd.DataFrame) -> pd.Series:
    """
    Predict Depression_Severity for VAE-generated rows using a RF classifier
    trained on the real training set.  This ensures TSTR labels reflect the
    generated feature values rather than a purely random class assignment.
    """
    from sklearn.ensemble import RandomForestClassifier
    feat_cols = [c for c in real_train.columns if c != config.TARGET_COL]
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(real_train[feat_cols].values, real_train[config.TARGET_COL].values)
    preds = clf.predict(synthetic_df[feat_cols].values)
    return pd.Series(preds, name=config.TARGET_COL)


def generate_vae(checkpoint_path: Path, n: int, seed: int,
                 device: torch.device) -> pd.DataFrame:
    """
    Generate n rows using a trained VAE checkpoint.

    Returns a DataFrame in [0,1] scaled space (same as nhanes_train.csv)
    with Depression_Severity labels assigned by a RF classifier trained on
    real training data.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    input_dim    = ckpt["input_dim"]
    binary_idx   = ckpt.get("binary_col_indices", [])
    feature_cols = ckpt.get("feature_cols", None)

    if feature_cols is None:
        train_df = pd.read_csv(config.NHANES_TRAIN)
        feature_cols = [c for c in train_df.columns if c != config.TARGET_COL]

    model = VAE(
        input_dim=input_dim,
        binary_col_indices=binary_idx,
        hidden_dims=config.VAE_HIDDEN_DIMS,
        latent_dim=config.VAE_LATENT_DIM,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, cont_cols, _ = _load_scaler()
    real_train = _load_real_train()  # already in [0,1] space

    torch.manual_seed(seed)
    with torch.no_grad():
        raw = model.generate(n, device).cpu().numpy()

    df = pd.DataFrame(raw, columns=feature_cols)

    # Rank-based marginal correction stays in [0,1] space
    df = _correct_marginals(df, cont_cols, real_train[feature_cols])

    # Assign Depression_Severity using RF classifier trained in [0,1] feature space
    df[config.TARGET_COL] = _assign_labels_vae(df, real_train).values

    return df


def generate_ctgan(checkpoint_path: Path, n: int, seed: int,
                   device: torch.device) -> pd.DataFrame:
    """
    Generate n rows using a trained CTGAN checkpoint.

    Returns a DataFrame in [0,1] scaled space (same as nhanes_train.csv)
    with Depression_Severity labels from the condition vector.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    input_dim    = ckpt["input_dim"]
    cond_dim     = ckpt["cond_dim"]
    feature_cols = ckpt.get("feature_cols", None)

    if feature_cols is None:
        train_df = pd.read_csv(config.NHANES_TRAIN)
        feature_cols = [c for c in train_df.columns if c != config.TARGET_COL]

    gen = Generator(
        config.CTGAN_NOISE_DIM, cond_dim, input_dim, config.CTGAN_GEN_DIM
    ).to(device)
    gen.load_state_dict(ckpt["gen_state"])
    gen.eval()

    _, cont_cols, _ = _load_scaler()
    real_train = _load_real_train()  # already in [0,1] space

    # Sample class labels proportional to real training distribution
    class_dist = real_train[config.TARGET_COL].value_counts(normalize=True).sort_index()
    rng = np.random.default_rng(seed)
    labels = rng.choice(class_dist.index, size=n, p=class_dist.values)

    torch.manual_seed(seed)
    with torch.no_grad():
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)
        cond  = F.one_hot(labels_t, num_classes=cond_dim).float()
        noise = torch.randn(n, config.CTGAN_NOISE_DIM, device=device)
        raw   = gen(noise, cond).cpu().numpy()

    df = pd.DataFrame(raw, columns=feature_cols)

    # Rank-based marginal correction stays in [0,1] space
    df = _correct_marginals(df, cont_cols, real_train[feature_cols])

    # CTGAN: use condition labels directly as Depression_Severity
    df[config.TARGET_COL] = labels.astype(int)

    return df


def generate(model_name: str, checkpoint_path: Path, seed: int = 42,
             n: int = None) -> pd.DataFrame:
    """
    Unified generation interface called by run_all_seeds.py.

    Args:
        model_name: 'vae' | 'ctgan'
        checkpoint_path: Path to the model checkpoint
        seed: Random seed
        n: Number of rows to generate (defaults to config.N_SYNTHETIC)

    Returns:
        DataFrame with feature columns + Depression_Severity
    """
    n = n or config.N_SYNTHETIC
    device = torch.device(config.DEVICE)

    if model_name == "vae":
        df = generate_vae(checkpoint_path, n, seed, device)
    elif model_name == "ctgan":
        df = generate_ctgan(checkpoint_path, n, seed, device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    out_path = config.DATA_SYNTHETIC / f"{model_name}_synthetic_seed{seed}.csv"
    df.to_csv(out_path, index=False)
    log.info("Generated %d rows (%s, seed=%d) -> %s", len(df), model_name, seed, out_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vae", "ctgan"], required=True)
    parser.add_argument("--seed",  type=int, default=config.RANDOM_SEED)
    parser.add_argument("--n",     type=int, default=config.N_SYNTHETIC)
    parser.add_argument("--out",   type=str, default=None)
    args = parser.parse_args()

    ckpt_path = config.MODELS_DIR / f"{args.model}_seed{args.seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run: python models/{args.model}/train.py --seed {args.seed}"
        )

    df = generate(args.model, ckpt_path, seed=args.seed, n=args.n)

    if args.out:
        df.to_csv(args.out, index=False)
        log.info("Saved to %s", args.out)


if __name__ == "__main__":
    main()
