"""
evaluation/privacy.py — Axes 3 and 4: DCR and dual MIA (weak + shadow-model).

DCR (Axis 3): mean Euclidean distance from each synthetic row to the nearest
real training row. Higher = more private. Target mean > 0.10.

Weak MIA (Axis 4a): binary RF classifier distinguishing real training rows from
synthetic rows. Accuracy near 50% = good privacy. Target <= 55%.

Shadow MIA (Axis 4b): stronger attacker — trains on shadow-labelled members vs
non-members (real holdout + synthetic). This is the headline privacy metric.

Functional API (used by four_axis_audit.py):
  axis3_dcr(real_train, synthetic) -> float
  axis4_mia_weak(real_train, synthetic, seed) -> float
  axis4_mia_shadow(real_train, synthetic, seed) -> float

CLI usage (backwards compat):
  python evaluation/privacy.py --model vae
  python evaluation/privacy.py --model ctgan
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def axis3_dcr(real_train: pd.DataFrame, synthetic: pd.DataFrame) -> float:
    """
    Distance to Closest Record (DCR).

    For each synthetic row, find its minimum Euclidean distance to any real
    training row.  Returns the mean DCR across all synthetic rows.
    Computed in chunks (200 rows) to avoid OOM.
    """
    feature_cols = [c for c in real_train.columns if c != config.TARGET_COL]
    real_arr = real_train[feature_cols].values.astype(np.float32)
    syn_arr  = synthetic[feature_cols].values.astype(np.float32)

    chunk_size = 200
    min_dists: list[float] = []
    for i in range(0, len(syn_arr), chunk_size):
        chunk = syn_arr[i : i + chunk_size]
        dists = cdist(chunk, real_arr, metric="euclidean")
        min_dists.extend(dists.min(axis=1).tolist())

    mean_dcr = float(np.mean(min_dists))
    log.info("  DCR mean: %.4f | min: %.4f | max: %.4f  [target > %.2f]",
             mean_dcr, min(min_dists), max(min_dists), config.FAA_DCR_MEAN)
    return mean_dcr


def axis4_mia_weak(real_train: pd.DataFrame, synthetic: pd.DataFrame,
                   seed: int = 42) -> float:
    """
    Weak MIA: 5-fold RF classifier distinguishing real training rows from
    synthetic rows.  Label real=1, synthetic=0.
    """
    feature_cols = [c for c in real_train.columns if c != config.TARGET_COL]
    real_sample = real_train[feature_cols].sample(
        min(len(synthetic), len(real_train)), random_state=seed
    )
    real_arr = real_sample.values
    syn_arr  = synthetic[feature_cols].values[: len(real_arr)]

    X = np.vstack([real_arr, syn_arr])
    y = np.array([1] * len(real_arr) + [0] * len(syn_arr))

    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    mia_acc = float(scores.mean())
    log.info("  MIA (weak RF): %.4f ± %.4f  [target <= %.2f]",
             mia_acc, scores.std(), config.FAA_MIA_ACCURACY)
    return mia_acc


def axis4_mia_shadow(real_train: pd.DataFrame, synthetic: pd.DataFrame,
                     seed: int = 42) -> float:
    """
    Shadow-model MIA (stronger attacker).

    Method:
      1. Split real_train in half: shadow_train (members) and shadow_holdout.
      2. Build attack dataset:
         - MEMBERS  (label=1): shadow_train rows  — seen by generator
         - NON-MEMBERS (label=0): shadow_holdout + synthetic rows — unseen
      3. Train RF attack classifier with 5-fold CV.
    """
    feature_cols = [c for c in real_train.columns if c != config.TARGET_COL]
    real_arr = real_train[feature_cols].values
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(real_arr))
    half = len(idx) // 2

    shadow_train_arr   = real_arr[idx[:half]]    # members — seen by generator
    shadow_holdout_arr = real_arr[idx[half:]]    # non-members — real but unseen
    syn_arr            = synthetic[feature_cols].values

    n_use = min(len(shadow_train_arr), len(shadow_holdout_arr), len(syn_arr))
    half_use = max(n_use // 2, 1)

    X_attack = np.vstack([
        shadow_train_arr[:n_use],           # members
        shadow_holdout_arr[:half_use],      # real non-members
        syn_arr[:half_use],                 # synthetic non-members
    ])
    y_attack = np.array(
        [1] * n_use + [0] * half_use + [0] * half_use
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    scores = cross_val_score(clf, X_attack, y_attack, cv=5, scoring="accuracy")
    mia_acc = float(scores.mean())
    log.info("  MIA (shadow): %.4f ± %.4f  [target <= %.2f]",
             mia_acc, scores.std(), config.FAA_MIA_ACCURACY)
    return mia_acc


# ── Legacy CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vae", "ctgan"], required=True)
    args = parser.parse_args()

    real_df = pd.read_csv(config.NHANES_CLEAN)
    with open(config.SPLITS_PATH, "rb") as f:
        splits = pickle.load(f)
    real_train = real_df.iloc[splits["train"]].reset_index(drop=True)

    synth_path = config.DATA_SYNTHETIC / f"{args.model}_nhanes_synthetic.csv"
    if not synth_path.exists():
        raise FileNotFoundError(
            f"Synthetic file not found: {synth_path}\n"
            f"Run: python generate.py --model {args.model}"
        )
    synth_df = pd.read_csv(synth_path)
    with open(config.SCALER_PATH, "rb") as f:
        bundle = pickle.load(f)
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]
    shared = [c for c in feature_cols if c in synth_df.columns]
    synth_df[shared] = scaler.transform(synth_df[shared])

    dcr_mean  = axis3_dcr(real_train, synth_df)
    mia_weak  = axis4_mia_weak(real_train, synth_df, seed=config.RANDOM_SEED)
    mia_shadow = axis4_mia_shadow(real_train, synth_df, seed=config.RANDOM_SEED)

    results = {
        "model": args.model,
        "dcr_mean": round(dcr_mean, 4),
        "mia_weak_acc": round(mia_weak, 4),
        "mia_shadow_acc": round(mia_shadow, 4),
    }
    out = config.EVAL_PLOTS_DIR / f"privacy_{args.model}.csv"
    pd.DataFrame([results]).to_csv(out, index=False)
    log.info("Saved -> %s", out)


if __name__ == "__main__":
    main()
