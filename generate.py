"""
generate.py — Generate synthetic NHANES rows from a trained VAE or CTGAN.

Called by run_all_seeds.py:
  generate(model_name, checkpoint_path, seed) -> pd.DataFrame

CLI usage:
  python generate.py --model vae   --seed 42 --n 1000
  python generate.py --model ctgan --seed 42 --n 1000
  python generate.py --model vae   --seed 42 --marginal-correction   # ablation only

Post-processing applied to every model:
  1. Categorical/binary columns are snapped to the nearest valid scaled level
     observed in the real training data (generators emit continuous values).
  2. Optionally (ablation only, --marginal-correction) continuous marginals are
     rank-remapped onto the real training CDF. This copies real training floats
     into the synthetic data, so it is DISABLED by default and reported as a
     metric-gaming ablation in the paper, never as the main configuration.

Label (Depression_Severity) mechanisms:
  - VAE:   generated jointly as an extra scaled column (trained with the target
           appended to the feature matrix), then snapped to {0..4}.
  - CTGAN: class labels sampled from the real training distribution, fed as the
           one-hot condition, and used directly as the label.
"""

import argparse
import logging
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


def _load_real_train() -> pd.DataFrame:
    """Return real training rows (scaled [0,1])."""
    return pd.read_csv(config.NHANES_TRAIN)


def _discretize_categoricals(df: pd.DataFrame, cat_cols: list[str],
                             real_train: pd.DataFrame) -> pd.DataFrame:
    """
    Snap each categorical/binary column to the nearest valid scaled level.

    Generators emit continuous values for every column; real categorical
    columns only take a small set of scaled levels (e.g. {0, 1} for binary,
    {0, 0.25, ...} for multi-level). Without this step synthetic rows are
    trivially distinguishable from real ones on every categorical column.
    """
    df = df.copy()
    for col in cat_cols:
        if col not in df.columns or col not in real_train.columns:
            continue
        levels = np.sort(real_train[col].unique()).astype(float)
        vals = df[col].values.astype(float)
        if len(levels) == 1:
            df[col] = levels[0]
            continue
        idx = np.searchsorted(levels, vals).clip(1, len(levels) - 1)
        left, right = levels[idx - 1], levels[idx]
        df[col] = np.where(np.abs(vals - left) <= np.abs(right - vals), left, right)
    return df


def apply_marginal_correction(df: pd.DataFrame,
                              real_train: pd.DataFrame) -> pd.DataFrame:
    """
    ABLATION ONLY — rank-based marginal correction in [0,1] space.

    For each continuous column, remaps synthetic values to the real training
    CDF via index-based sampling, so corrected values are exact floats from
    nhanes_train.csv. This guarantees marginal fidelity by construction
    (metric gaming) and injects verbatim real values into the synthetic data.
    Must be applied uniformly to all models when used, and disclosed.
    """
    df = df.copy()
    n = len(df)
    for col in config.CONTINUOUS_COLS:
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


def _postprocess(df: pd.DataFrame, real_train: pd.DataFrame,
                 marginal_correction: bool) -> pd.DataFrame:
    """Shared post-processing: discretize categoricals, optional MC ablation."""
    feature_cols = [c for c in real_train.columns if c != config.TARGET_COL]
    cat_cols = [c for c in feature_cols if c not in config.CONTINUOUS_COLS]
    df = _discretize_categoricals(df, cat_cols, real_train)
    if marginal_correction:
        log.warning("Marginal correction ENABLED (ablation mode) — synthetic "
                    "continuous values are copies of real training floats.")
        df = apply_marginal_correction(df, real_train)
    return df


def generate_vae(checkpoint_path: Path, n: int, seed: int,
                 device: torch.device,
                 marginal_correction: bool = False) -> pd.DataFrame:
    """
    Generate n rows using a trained VAE checkpoint.

    The VAE is trained jointly on features + scaled target, so the label is
    generated, not assigned post hoc. Returns a DataFrame in [0,1] scaled
    space with an integer Depression_Severity column.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    input_dim    = ckpt["input_dim"]
    binary_idx   = ckpt.get("binary_col_indices", [])
    feature_cols = ckpt.get("feature_cols", None)

    if not ckpt.get("joint_target", False):
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} predates joint-label training "
            f"(no 'joint_target' flag). Retrain: python models/vae/train.py "
            f"--seed {seed}"
        )

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

    real_train = _load_real_train()  # already in [0,1] space

    torch.manual_seed(seed)
    with torch.no_grad():
        raw = model.generate(n, device).cpu().numpy()

    # Last column is the jointly-generated scaled target
    df = pd.DataFrame(raw[:, :-1], columns=feature_cols)
    label_scaled = np.clip(raw[:, -1], 0.0, 1.0)
    labels = np.rint(label_scaled * (config.NUM_CLASSES - 1)).astype(int)

    df = _postprocess(df, real_train, marginal_correction)
    df[config.TARGET_COL] = labels
    return df


def generate_ctgan(checkpoint_path: Path, n: int, seed: int,
                   device: torch.device,
                   marginal_correction: bool = False) -> pd.DataFrame:
    """
    Generate n rows using a trained CTGAN checkpoint.

    Returns a DataFrame in [0,1] scaled space with Depression_Severity taken
    from the one-hot condition vector.
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
    df = _postprocess(df, real_train, marginal_correction)
    df[config.TARGET_COL] = labels.astype(int)
    return df


def generate(model_name: str, checkpoint_path: Path, seed: int = 42,
             n: int = None, marginal_correction: bool = False) -> pd.DataFrame:
    """
    Unified generation interface called by run_all_seeds.py.

    Args:
        model_name: 'vae' | 'ctgan'
        checkpoint_path: Path to the model checkpoint
        seed: Random seed
        n: Number of rows to generate (defaults to config.N_SYNTHETIC)
        marginal_correction: ablation flag — see apply_marginal_correction()

    Returns:
        DataFrame with feature columns + Depression_Severity
    """
    n = n or config.N_SYNTHETIC
    device = torch.device(config.DEVICE)

    if model_name == "vae":
        df = generate_vae(checkpoint_path, n, seed, device, marginal_correction)
    elif model_name == "ctgan":
        df = generate_ctgan(checkpoint_path, n, seed, device, marginal_correction)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    suffix = "_mc" if marginal_correction else ""
    out_path = config.DATA_SYNTHETIC / f"{model_name}_synthetic{suffix}_seed{seed}.csv"
    df.to_csv(out_path, index=False)
    log.info("Generated %d rows (%s, seed=%d, mc=%s) -> %s",
             len(df), model_name, seed, marginal_correction, out_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vae", "ctgan"], required=True)
    parser.add_argument("--seed",  type=int, default=config.RANDOM_SEED)
    parser.add_argument("--n",     type=int, default=config.N_SYNTHETIC)
    parser.add_argument("--out",   type=str, default=None)
    parser.add_argument("--marginal-correction", action="store_true",
                        help="Ablation only: rank-remap continuous marginals "
                             "onto the real training CDF (copies real values).")
    args = parser.parse_args()

    ckpt_path = config.MODELS_DIR / f"{args.model}_seed{args.seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run: python models/{args.model}/train.py --seed {args.seed}"
        )

    df = generate(args.model, ckpt_path, seed=args.seed, n=args.n,
                  marginal_correction=args.marginal_correction)

    if args.out:
        df.to_csv(args.out, index=False)
        log.info("Saved to %s", args.out)


if __name__ == "__main__":
    main()
