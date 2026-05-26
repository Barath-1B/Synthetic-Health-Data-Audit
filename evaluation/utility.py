"""
evaluation/utility.py — Downstream utility: TSTR vs TRTR.

TSTR: Train on Synthetic, Test on Real
TRTR: Train on Real,      Test on Real  (baseline)

Reports accuracy, macro F1, and AUC-ROC for both.
Utility ratio = TSTR AUC / TRTR AUC  (target >= 0.85)

Usage:
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


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, val+test (unused here), test DataFrames from real data."""
    real_df = pd.read_csv(config.NHANES_CLEAN)
    with open(config.SPLITS_PATH, "rb") as f:
        splits = pickle.load(f)
    return (
        real_df.iloc[splits["train"]].reset_index(drop=True),
        real_df.iloc[splits["test"]].reset_index(drop=True),
    )


def split_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[[c for c in df.columns if c != config.TARGET_COL]].values
    y = df[config.TARGET_COL].values
    return X, y


def evaluate_classifier(clf, X_train, y_train, X_test, y_test, label: str) -> dict:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    classes  = sorted(np.unique(y_test))
    y_bin    = label_binarize(y_test, classes=classes)
    acc      = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average="macro", zero_division=0)
    # AUC: handle binary vs multiclass
    if y_prob.shape[1] == 2:
        auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")

    log.info("  [%s]  acc=%.4f  f1=%.4f  auc=%.4f", label, acc, f1, auc)
    return {"label": label, "accuracy": acc, "f1_macro": f1, "auc_roc": auc}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vae", "ctgan"], required=True)
    args = parser.parse_args()

    real_train, real_test = load_splits()
    X_real_train, y_real_train = split_xy(real_train)
    X_test,       y_test       = split_xy(real_test)

    synth_path = config.DATA_SYNTHETIC / f"{args.model}_nhanes_synthetic.csv"
    if not synth_path.exists():
        raise FileNotFoundError(
            f"Synthetic file not found: {synth_path}\n"
            f"Run: python generate.py --model {args.model}"
        )
    synth_df = pd.read_csv(synth_path)
    # Align columns to real data
    shared_feat = [c for c in real_train.columns if c != config.TARGET_COL and c in synth_df.columns]
    X_synth = synth_df[shared_feat].values
    y_synth = synth_df[config.TARGET_COL].values

    classifiers = [
        ("LogReg", LogisticRegression(max_iter=500, random_state=config.RANDOM_SEED)),
        ("RF",     RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_SEED)),
    ]

    results = []
    for name, clf in classifiers:
        log.info("\n=== %s ===", name)
        trtr = evaluate_classifier(clf, X_real_train, y_real_train, X_test, y_test, f"TRTR-{name}")
        tstr = evaluate_classifier(clf, X_synth,      y_synth,      X_test, y_test, f"TSTR-{name}")
        ratio = tstr["auc_roc"] / max(trtr["auc_roc"], 1e-9)
        log.info("  Utility ratio (TSTR/TRTR AUC): %.4f  [target >= 0.85]", ratio)
        results += [trtr, tstr, {"label": f"ratio-{name}", "auc_roc": ratio}]

    out = config.EVAL_PLOTS_DIR / f"utility_{args.model}.csv"
    config.EVAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    log.info("\nSaved utility results -> %s", out)


if __name__ == "__main__":
    main()
