"""
evaluation/utility.py — Axis 2: Downstream utility via TSTR / TRTR.

TSTR: Train on Synthetic, Test on Real
TRTR: Train on Real,      Test on Real  (upper-bound baseline)

Functional API (used by four_axis_audit.py):
  axis2_utility(real_train, real_test, synthetic, seed) -> (tstr_auc, trtr_auc)

CLI usage (backwards compat):
  python evaluation/utility.py --model vae
  python evaluation/utility.py --model ctgan
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def axis2_utility(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic: pd.DataFrame,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Compute TSTR and TRTR AUC-ROC (Random Forest, OVR macro).

    Returns:
        (tstr_auc, trtr_auc) — both as floats in [0, 1]
    """
    feature_cols = [c for c in real_train.columns if c != config.TARGET_COL]
    X_syn   = synthetic[feature_cols].values
    y_syn   = synthetic[config.TARGET_COL].values
    X_train = real_train[feature_cols].values
    y_train = real_train[config.TARGET_COL].values
    X_test  = real_test[feature_cols].values
    y_test  = real_test[config.TARGET_COL].values

    def _auc(X_tr: np.ndarray, y_tr: np.ndarray,
             X_te: np.ndarray, y_te: np.ndarray) -> float:
        clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        proba_raw = clf.predict_proba(X_te)  # shape (n_test, n_train_classes)
        # Align probability columns to the full set of test classes.
        # Synthetic data may lack rare classes → expand proba with 0-columns.
        all_classes = sorted(np.unique(y_te))
        n_all = len(all_classes)
        if proba_raw.shape[1] == n_all:
            proba = proba_raw
        else:
            proba = np.zeros((len(y_te), n_all), dtype=np.float64)
            clf_classes = list(clf.classes_)
            for i, c in enumerate(all_classes):
                if c in clf_classes:
                    proba[:, i] = proba_raw[:, clf_classes.index(c)]
            # Re-normalise rows that were expanded
            row_sums = proba.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            proba /= row_sums
        try:
            return float(roc_auc_score(y_te, proba, multi_class="ovr", average="macro"))
        except Exception:
            return 0.0

    tstr_auc = _auc(X_syn,   y_syn,   X_test, y_test)
    trtr_auc = _auc(X_train, y_train, X_test, y_test)
    ratio = tstr_auc / trtr_auc if trtr_auc > 0 else 0.0
    log.info("  TSTR AUC: %.4f | TRTR AUC: %.4f | Ratio: %.4f", tstr_auc, trtr_auc, ratio)
    return tstr_auc, trtr_auc


# ── Legacy CLI ─────────────────────────────────────────────────────────────────

def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (real_train, real_test) DataFrames."""
    real_df = pd.read_csv(config.NHANES_CLEAN)
    with open(config.SPLITS_PATH, "rb") as f:
        splits = pickle.load(f)
    return (
        real_df.iloc[splits["train"]].reset_index(drop=True),
        real_df.iloc[splits["test"]].reset_index(drop=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vae", "ctgan"], required=True)
    args = parser.parse_args()

    real_train, real_test = _load_splits()

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
    shared_scaled = [c for c in feature_cols if c in synth_df.columns]
    synth_df[shared_scaled] = scaler.transform(synth_df[shared_scaled])

    tstr_auc, trtr_auc = axis2_utility(real_train, real_test, synth_df, seed=config.RANDOM_SEED)
    ratio = tstr_auc / trtr_auc if trtr_auc > 0 else 0.0
    log.info("Utility ratio (TSTR/TRTR): %.4f  [target >= 0.85]", ratio)

    out = config.EVAL_PLOTS_DIR / f"utility_{args.model}.csv"
    pd.DataFrame([{
        "model": args.model,
        "tstr_auc": tstr_auc,
        "trtr_auc": trtr_auc,
        "utility_ratio": ratio,
    }]).to_csv(out, index=False)
    log.info("Saved -> %s", out)


if __name__ == "__main__":
    main()
