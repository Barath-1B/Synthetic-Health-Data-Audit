"""
evaluation/privacy.py — Axes 3 and 4: holdout-calibrated DCR and DOMIAS MIA.

Axis 3 (DCR, holdout-calibrated):
    DCR(synthetic -> train) compared against DCR(test -> train). Synthetic data
    should sit no closer to the training set than unseen real data does.
    Gate: dcr_ratio = mean_syn / mean_holdout >= FAA_DCR_RATIO (1.0).
    An absolute DCR threshold is NOT used — it conflates infidelity with
    privacy (a generator far from the data manifold gets a great DCR).

Axis 4 (membership inference, DOMIAS-style):
    Density-ratio attack following van Breugel et al. (AISTATS 2023):
    score(x) = log p_syn(x) - log p_ref(x), where p_syn is estimated from the
    synthetic data and p_ref from a reference set (validation split — never
    seen by the generator, never in the attack eval set). Evaluated as ROC AUC
    over a balanced set of training members vs test non-members.
    Gate: mia_auc_domias <= FAA_MIA_AUC (0.55).

Also reported (non-gating):
    - mia_auc_domias_clf: classifier-based density-ratio variant (sensitivity)
    - c2st_acc: classifier two-sample test, real-train vs synthetic. This is a
      DISTINGUISHABILITY measure, not membership inference (the quantity the
      old "weak MIA" actually computed — renamed for honesty).

Functional API (used by four_axis_audit.py):
  axis3_dcr(real_train, real_test, synthetic) -> dict
  axis4_mia_domias(real_train, real_test, synthetic, reference, seed) -> dict
  distinguishability_c2st(real_train, synthetic, seed) -> float

CLI (single model+seed, new split CSVs):
  python evaluation/privacy.py --model vae --seed 42
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KernelDensity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ── Axis 3: holdout-calibrated DCR ────────────────────────────────────────────

def _min_dists(queries: np.ndarray, refs: np.ndarray,
               chunk_size: int = 200) -> np.ndarray:
    """Min Euclidean distance from each query row to any reference row."""
    out: list[np.ndarray] = []
    for i in range(0, len(queries), chunk_size):
        d = cdist(queries[i: i + chunk_size], refs, metric="euclidean")
        out.append(d.min(axis=1))
    return np.concatenate(out)


def axis3_dcr(real_train: pd.DataFrame, real_test: pd.DataFrame,
              synthetic: pd.DataFrame) -> dict:
    """
    Holdout-calibrated Distance to Closest Record.

    Returns dict with:
      dcr_syn_mean      mean DCR(synthetic -> train)
      dcr_holdout_mean  mean DCR(test -> train), the calibration baseline
      dcr_ratio         dcr_syn_mean / dcr_holdout_mean (gate: >= 1.0)
      dcr_share_close   fraction of synthetic rows closer to train than the
                        5th percentile of holdout DCRs (memorisation tail)
      dcr_syn_raw       per-row synthetic DCR array (for figures)
      dcr_holdout_raw   per-row holdout DCR array (for figures)
    """
    feature_cols = [c for c in real_train.columns if c != config.TARGET_COL]
    train_arr   = real_train[feature_cols].values.astype(np.float32)
    test_arr    = real_test[feature_cols].values.astype(np.float32)
    syn_arr     = synthetic[feature_cols].values.astype(np.float32)

    dcr_syn     = _min_dists(syn_arr, train_arr)
    dcr_holdout = _min_dists(test_arr, train_arr)

    syn_mean     = float(dcr_syn.mean())
    holdout_mean = float(dcr_holdout.mean())
    ratio        = syn_mean / holdout_mean if holdout_mean > 0 else 0.0
    p5           = float(np.percentile(dcr_holdout, 5))
    share_close  = float((dcr_syn < p5).mean())

    log.info("  DCR syn: %.4f | holdout: %.4f | ratio: %.3f  [target >= %.2f] "
             "| share<holdout-p5: %.3f",
             syn_mean, holdout_mean, ratio, config.FAA_DCR_RATIO, share_close)
    return {
        "dcr_syn_mean":     syn_mean,
        "dcr_holdout_mean": holdout_mean,
        "dcr_ratio":        ratio,
        "dcr_share_close":  share_close,
        "dcr_syn_raw":      dcr_syn,
        "dcr_holdout_raw":  dcr_holdout,
    }


# ── Axis 4: DOMIAS membership inference ──────────────────────────────────────

def _standardize(fit_on: np.ndarray, *arrays: np.ndarray) -> list[np.ndarray]:
    """Z-score arrays using mean/std fitted on `fit_on` (avoids KDE scale issues)."""
    mu = fit_on.mean(axis=0)
    sd = fit_on.std(axis=0)
    sd[sd == 0] = 1.0
    return [(a - mu) / sd for a in arrays]


def _kde_log_density(fit: np.ndarray, score_on: np.ndarray) -> np.ndarray:
    """Gaussian KDE log-density with Scott's-rule bandwidth on standardized data."""
    n, d = fit.shape
    bandwidth = n ** (-1.0 / (d + 4))   # Scott's factor; data is standardized
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(fit)
    return kde.score_samples(score_on)


def axis4_mia_domias(real_train: pd.DataFrame, real_test: pd.DataFrame,
                     synthetic: pd.DataFrame, reference: pd.DataFrame,
                     seed: int = 42, n_eval_per_class: int = 500) -> dict:
    """
    DOMIAS density-ratio membership inference attack.

    Attack model: the adversary sees the synthetic data and a reference sample
    from the population (here: the validation split). For a target record x,
    the membership score is log p_syn(x) - log p_ref(x): records the generator
    has overfit to sit in regions where synthetic density exceeds population
    density. Evaluated as ROC AUC on a balanced set of train members vs test
    non-members. AUC ~ 0.5 means the attack learns nothing.

    Returns dict with:
      mia_auc_domias      KDE-based density-ratio attack AUC (the gating metric)
      mia_auc_domias_clf  classifier-based density-ratio AUC (sensitivity check)
    """
    feature_cols = [c for c in real_train.columns if c != config.TARGET_COL]
    rng = np.random.default_rng(seed)

    n_mem  = min(n_eval_per_class, len(real_train))
    n_non  = min(n_eval_per_class, len(real_test))
    mem_idx = rng.choice(len(real_train), size=n_mem, replace=False)
    non_idx = rng.choice(len(real_test),  size=n_non, replace=False)

    members     = real_train[feature_cols].values[mem_idx].astype(np.float64)
    non_members = real_test[feature_cols].values[non_idx].astype(np.float64)
    syn_arr     = synthetic[feature_cols].values.astype(np.float64)
    ref_arr     = reference[feature_cols].values.astype(np.float64)

    targets = np.vstack([members, non_members])
    y_true  = np.concatenate([np.ones(n_mem), np.zeros(n_non)])

    # KDE variant (primary, gating)
    syn_s, ref_s, tgt_s = _standardize(np.vstack([syn_arr, ref_arr]),
                                       syn_arr, ref_arr, targets)
    log_p_syn = _kde_log_density(syn_s, tgt_s)
    log_p_ref = _kde_log_density(ref_s, tgt_s)
    auc_kde = float(roc_auc_score(y_true, log_p_syn - log_p_ref))

    # Classifier-based density-ratio variant (sensitivity, non-gating):
    # P(synthetic | x) odds against the reference approximate p_syn/p_ref.
    X_disc = np.vstack([syn_arr, ref_arr])
    y_disc = np.concatenate([np.ones(len(syn_arr)), np.zeros(len(ref_arr))])
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    clf.fit(X_disc, y_disc)
    proba = clf.predict_proba(targets)[:, 1].clip(1e-6, 1 - 1e-6)
    auc_clf = float(roc_auc_score(y_true, np.log(proba / (1 - proba))))

    log.info("  MIA DOMIAS AUC (KDE): %.4f  [target <= %.2f] | clf variant: %.4f",
             auc_kde, config.FAA_MIA_AUC, auc_clf)
    return {"mia_auc_domias": auc_kde, "mia_auc_domias_clf": auc_clf}


# ── Distinguishability (reported, non-gating) ────────────────────────────────

def distinguishability_c2st(real_train: pd.DataFrame, synthetic: pd.DataFrame,
                            seed: int = 42) -> float:
    """
    Classifier two-sample test: 5-fold RF accuracy separating real training
    rows from synthetic rows (balanced). 0.5 = indistinguishable.

    NOTE: this measures sample quality, NOT membership inference. It was
    formerly mislabelled "weak MIA"; renamed to what it actually is.
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
    acc = float(scores.mean())
    log.info("  C2ST distinguishability: %.4f ± %.4f  (0.5 = ideal)",
             acc, scores.std())
    return acc


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    real_train = pd.read_csv(config.NHANES_TRAIN)
    real_val   = pd.read_csv(config.NHANES_VAL)
    real_test  = pd.read_csv(config.NHANES_TEST)

    synth_path = config.DATA_SYNTHETIC / f"{args.model}_synthetic_seed{args.seed}.csv"
    if not synth_path.exists():
        raise FileNotFoundError(f"Synthetic file not found: {synth_path}")
    synth_df = pd.read_csv(synth_path)

    dcr = axis3_dcr(real_train, real_test, synth_df)
    mia = axis4_mia_domias(real_train, real_test, synth_df, real_val,
                           seed=args.seed)
    c2st = distinguishability_c2st(real_train, synth_df, seed=args.seed)

    results = {
        "model": args.model,
        "seed": args.seed,
        "dcr_syn_mean": round(dcr["dcr_syn_mean"], 4),
        "dcr_holdout_mean": round(dcr["dcr_holdout_mean"], 4),
        "dcr_ratio": round(dcr["dcr_ratio"], 4),
        "dcr_share_close": round(dcr["dcr_share_close"], 4),
        "mia_auc_domias": round(mia["mia_auc_domias"], 4),
        "mia_auc_domias_clf": round(mia["mia_auc_domias_clf"], 4),
        "c2st_acc": round(c2st, 4),
    }
    out = config.EVAL_RESULTS / f"privacy_{args.model}_seed{args.seed}.csv"
    pd.DataFrame([results]).to_csv(out, index=False)
    log.info("Saved -> %s", out)


if __name__ == "__main__":
    main()
