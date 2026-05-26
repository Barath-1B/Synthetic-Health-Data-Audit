"""
evaluation/privacy.py — Privacy evaluation: DCR and MIA.

DCR (Distance to Closest Record):
  For each synthetic row, find its min Euclidean distance to any real training row.
  Low mean DCR -> memorisation risk.  Target mean > 0.1.

MIA (Membership Inference Attack):
  Binary classifier: real training rows = 1, synthetic rows = 0.
  Accuracy near 50% = good privacy.  Target <= 55%.

Usage:
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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def load_real_train() -> np.ndarray:
    real_df = pd.read_csv(config.NHANES_CLEAN)
    with open(config.SPLITS_PATH, "rb") as f:
        splits = pickle.load(f)
    feature_cols = [c for c in real_df.columns if c != config.TARGET_COL]
    return real_df[feature_cols].iloc[splits["train"]].values.astype(np.float32)


def load_synthetic(model: str, feature_cols: list[str]) -> np.ndarray:
    synth_path = config.DATA_SYNTHETIC / f"{model}_nhanes_synthetic.csv"
    if not synth_path.exists():
        raise FileNotFoundError(
            f"Synthetic file not found: {synth_path}\n"
            f"Run: python generate.py --model {model}"
        )
    df = pd.read_csv(synth_path)
    shared = [c for c in feature_cols if c in df.columns]
    return df[shared].values.astype(np.float32)


def dcr(X_real: np.ndarray, X_synth: np.ndarray, batch: int = 256) -> np.ndarray:
    """Compute min Euclidean distance from each synthetic row to any real row."""
    n    = len(X_synth)
    dists = np.empty(n, dtype=np.float32)
    for i in range(0, n, batch):
        chunk = X_synth[i : i + batch]                        # (b, d)
        diff  = chunk[:, None, :] - X_real[None, :, :]        # (b, N, d)
        d     = np.linalg.norm(diff, axis=2)                   # (b, N)
        dists[i : i + batch] = d.min(axis=1)
    return dists


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vae", "ctgan"], required=True)
    args = parser.parse_args()

    config.EVAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X_real  = load_real_train()
    real_df = pd.read_csv(config.NHANES_CLEAN)
    feat_cols = [c for c in real_df.columns if c != config.TARGET_COL]
    X_synth = load_synthetic(args.model, feat_cols)

    # ── DCR ──────────────────────────────────────────────────────────────────
    log.info("Computing DCR for %d synthetic rows ...", len(X_synth))
    distances = dcr(X_real, X_synth)
    mean_dcr  = distances.mean()
    log.info("DCR mean=%.4f  min=%.4f  max=%.4f  [target mean > 0.1]",
             mean_dcr, distances.min(), distances.max())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(distances, bins=50, color="steelblue", edgecolor="white")
    ax.axvline(mean_dcr, color="red", linestyle="--", label=f"mean={mean_dcr:.3f}")
    ax.set_xlabel("Distance to Closest Real Record")
    ax.set_ylabel("Count")
    ax.set_title(f"DCR Distribution — {args.model.upper()}")
    ax.legend()
    dcr_plot = config.EVAL_PLOTS_DIR / f"dcr_{args.model}.png"
    plt.tight_layout()
    plt.savefig(dcr_plot, dpi=150)
    plt.close()
    log.info("Saved DCR plot -> %s", dcr_plot)

    # ── MIA ──────────────────────────────────────────────────────────────────
    n_real  = min(len(X_real),  len(X_synth))
    n_synth = n_real
    idx_r   = np.random.choice(len(X_real),  n_real,  replace=False)
    idx_s   = np.random.choice(len(X_synth), n_synth, replace=False)

    X_mia = np.vstack([X_real[idx_r], X_synth[idx_s]])
    y_mia = np.array([1] * n_real + [0] * n_synth)

    clf   = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_SEED)
    scores = cross_val_score(clf, X_mia, y_mia, cv=5, scoring="accuracy")
    mia_acc = scores.mean()
    log.info("MIA 5-fold accuracy=%.4f +/- %.4f  [target <= 0.55]",
             mia_acc, scores.std())

    results = {
        "model":    args.model,
        "dcr_mean": round(float(mean_dcr), 4),
        "dcr_min":  round(float(distances.min()), 4),
        "dcr_max":  round(float(distances.max()), 4),
        "mia_acc":  round(float(mia_acc), 4),
    }
    out_csv = config.EVAL_PLOTS_DIR / f"privacy_{args.model}.csv"
    pd.DataFrame([results]).to_csv(out_csv, index=False)
    log.info("Saved privacy results -> %s", out_csv)


if __name__ == "__main__":
    main()
