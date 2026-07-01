"""
evaluation/utility.py — Axis 2: Downstream utility via TSTR / TRTR.

TSTR: Train on Synthetic, Test on Real
TRTR: Train on Real,      Test on Real  (upper-bound baseline)

Gate: utility_ratio = TSTR/TRTR macro AUC (RandomForest) >= FAA_UTILITY_RATIO.
Also reported (non-gating): LogisticRegression evaluator (robustness to the
choice of downstream model) and per-class AUC/F1 (classes 3-4, moderately
severe / severe depression, are rare and clinically the most important).

Functional API (used by four_axis_audit.py):
  axis2_utility(real_train, real_test, synthetic, seed) -> (metrics, per_class_df)

CLI:
  python evaluation/utility.py --model vae --seed 42
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _aligned_proba(clf, X_te: np.ndarray, y_te: np.ndarray) -> np.ndarray:
    """predict_proba aligned to the full set of test classes (synthetic data
    may lack rare classes — missing classes get probability 0)."""
    proba_raw = clf.predict_proba(X_te)
    all_classes = sorted(np.unique(y_te))
    if proba_raw.shape[1] == len(all_classes) and list(clf.classes_) == all_classes:
        return proba_raw
    proba = np.zeros((len(y_te), len(all_classes)), dtype=np.float64)
    clf_classes = list(clf.classes_)
    for i, c in enumerate(all_classes):
        if c in clf_classes:
            proba[:, i] = proba_raw[:, clf_classes.index(c)]
    row_sums = proba.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return proba / row_sums


def _fit_eval(make_clf, X_tr: np.ndarray, y_tr: np.ndarray,
              X_te: np.ndarray, y_te: np.ndarray) -> tuple[float, dict]:
    """Fit a classifier; return (macro OVR AUC, per-class {auc, f1})."""
    clf = make_clf()
    clf.fit(X_tr, y_tr)
    proba = _aligned_proba(clf, X_te, y_te)
    preds = np.array(sorted(np.unique(y_te)))[proba.argmax(axis=1)]
    all_classes = sorted(np.unique(y_te))
    try:
        macro_auc = float(roc_auc_score(y_te, proba, multi_class="ovr", average="macro"))
    except Exception:
        macro_auc = 0.0
    per_class: dict[int, dict] = {}
    for i, c in enumerate(all_classes):
        y_bin = (y_te == c).astype(int)
        try:
            auc_c = float(roc_auc_score(y_bin, proba[:, i]))
        except Exception:
            auc_c = float("nan")
        f1_c = float(f1_score(y_bin, (preds == c).astype(int), zero_division=0))
        per_class[int(c)] = {"auc": auc_c, "f1": f1_c}
    return macro_auc, per_class


def axis2_utility(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic: pd.DataFrame,
    seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    """
    Compute TSTR and TRTR with two downstream evaluators.

    Returns:
        metrics: dict with tstr_auc, trtr_auc, utility_ratio (RF — gating),
                 tstr_auc_lr, trtr_auc_lr, utility_ratio_lr (LogReg)
        per_class_df: rows [evaluator, protocol, class, auc, f1]
    """
    feature_cols = [c for c in real_train.columns if c != config.TARGET_COL]
    X_syn   = synthetic[feature_cols].values
    y_syn   = synthetic[config.TARGET_COL].values
    X_train = real_train[feature_cols].values
    y_train = real_train[config.TARGET_COL].values
    X_test  = real_test[feature_cols].values
    y_test  = real_test[config.TARGET_COL].values

    evaluators = {
        "rf": lambda: RandomForestClassifier(n_estimators=100, random_state=seed,
                                             n_jobs=-1),
        "lr": lambda: LogisticRegression(max_iter=2000, random_state=seed),
    }

    metrics: dict[str, float] = {}
    rows: list[dict] = []
    for name, make_clf in evaluators.items():
        tstr_auc, tstr_pc = _fit_eval(make_clf, X_syn, y_syn, X_test, y_test)
        trtr_auc, trtr_pc = _fit_eval(make_clf, X_train, y_train, X_test, y_test)
        ratio = tstr_auc / trtr_auc if trtr_auc > 0 else 0.0
        suffix = "" if name == "rf" else f"_{name}"
        metrics[f"tstr_auc{suffix}"]      = tstr_auc
        metrics[f"trtr_auc{suffix}"]      = trtr_auc
        metrics[f"utility_ratio{suffix}"] = ratio
        for protocol, pc in [("tstr", tstr_pc), ("trtr", trtr_pc)]:
            for cls, vals in pc.items():
                rows.append({"evaluator": name, "protocol": protocol,
                             "class": cls, "auc": vals["auc"], "f1": vals["f1"]})
        log.info("  [%s] TSTR AUC: %.4f | TRTR AUC: %.4f | Ratio: %.4f",
                 name.upper(), tstr_auc, trtr_auc, ratio)

    return metrics, pd.DataFrame(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    real_train = pd.read_csv(config.NHANES_TRAIN)
    real_test  = pd.read_csv(config.NHANES_TEST)

    synth_path = config.DATA_SYNTHETIC / f"{args.model}_synthetic_seed{args.seed}.csv"
    if not synth_path.exists():
        raise FileNotFoundError(f"Synthetic file not found: {synth_path}")
    synth_df = pd.read_csv(synth_path)

    metrics, per_class = axis2_utility(real_train, real_test, synth_df,
                                       seed=args.seed)
    log.info("Utility ratio (RF TSTR/TRTR): %.4f  [target >= %.2f]",
             metrics["utility_ratio"], config.FAA_UTILITY_RATIO)

    out = config.EVAL_RESULTS / f"utility_{args.model}_seed{args.seed}.csv"
    pd.DataFrame([{"model": args.model, "seed": args.seed, **metrics}]).to_csv(
        out, index=False)
    per_class.to_csv(
        config.EVAL_RESULTS / f"utility_per_class_{args.model}_seed{args.seed}.csv",
        index=False)
    log.info("Saved -> %s", out)


if __name__ == "__main__":
    main()
