"""
experiments/leakage_dial.py — a generator whose leakage is a single knob.

The audit has never been shown to CATCH anything: no generator in the main
matrix leaks under DOMIAS, so "DCR is a poor privacy proxy" rests on a rank
association with fidelity and nothing else. This supplies the missing arm.

A Gaussian KDE fitted on the FULL training split is a real generator whose
leakage is controlled by one scalar. At small bandwidth it emits a training row
plus N(0, bw^2) — total memorisation. At large bandwidth it is a smooth
population model. Fitting on the full split (not a subset) keeps every audit
member a row the generator actually saw, so the MIA is not diluted.

The question the sweep answers: at the bandwidth where dcr_ratio crosses its 1.0
gate, is the calibrated MIA still firing? If yes, DCR under-flags real leakage on
this data. If the two gates flip together, DCR is an adequate proxy here — a
bounded contradiction of the prior result, and equally reportable.

  python experiments/leakage_dial.py

Writes evaluation/results/leakage_dial.csv and faa_kdedial{i}_seed{seed}.csv.

CAVEAT, disclosed in the paper: DOMIAS estimates p_syn by KDE and this generator
IS a KDE, so the attack is favourably placed against it. mia_auc_domias_clf (a
random-forest density-ratio variant) is the cross-family check; the finding only
stands where both fire.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from evaluation.four_axis_audit import run_audit
from generate import _discretize_categoricals, _load_real_train
from utils.seed_utils import set_all_seeds

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Brackets the dcr_ratio == 1.0 crossing: mean DCR runs 0.015 (bw=0.005) to
# 0.583 (bw=0.20) against a holdout mean of ~0.433, so the gate flips near 0.15.
BANDWIDTHS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

KEEP = ["fidelity_score", "utility_ratio", "dcr_ratio", "dcr_share_close",
        "mia_auc_domias", "mia_auc_domias_clf", "mia_tpr_at_fpr01", "c2st_acc",
        "ax3_pass", "ax4_pass"]


def kde_generate(real_train: pd.DataFrame, bandwidth: float, n: int,
                 seed: int) -> pd.DataFrame:
    """
    Sample n rows from a Gaussian KDE fitted on the full training split.

    The target is min-max scaled into [0,1] before fitting so it cannot dominate
    the isotropic kernel (raw levels are 0..4 while every feature is in [0,1]),
    then rescaled and rounded back to {0..NUM_CLASSES-1}. Categorical columns get
    the same snapping every other generator in this repo receives.
    """
    feature_cols = [c for c in real_train.columns if c != config.TARGET_COL]
    cat_cols = [c for c in feature_cols if c not in config.CONTINUOUS_COLS]

    fit = real_train.copy()
    fit[config.TARGET_COL] = fit[config.TARGET_COL] / (config.NUM_CLASSES - 1)

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(fit.values)
    raw = np.clip(kde.sample(n, random_state=seed), 0.0, 1.0)

    df = pd.DataFrame(raw, columns=fit.columns)
    df = _discretize_categoricals(df, cat_cols, real_train)
    df[config.TARGET_COL] = np.rint(
        df[config.TARGET_COL] * (config.NUM_CLASSES - 1)).astype(int)
    return df


def main() -> int:
    real_train = _load_real_train()
    n = len(real_train)

    rows: list[dict] = []
    for i, bw in enumerate(BANDWIDTHS):
        name = f"kdedial{i}"
        for seed in config.SEEDS:
            set_all_seeds(seed)
            log.info("\n[leakage_dial] bandwidth=%.3f (%s) seed=%d", bw, name, seed)
            syn = kde_generate(real_train, bw, n, seed)
            syn.to_csv(config.DATA_SYNTHETIC / f"{name}_synthetic_seed{seed}.csv",
                       index=False)
            m = run_audit(syn, seed=seed, model_name=name)
            rows.append({"bandwidth": bw, "name": name, "seed": seed,
                         **{k: m[k] for k in KEEP}})

    df = pd.DataFrame(rows)
    out = config.EVAL_RESULTS / "leakage_dial.csv"
    df.to_csv(out, index=False)

    means = df.groupby("bandwidth")[
        ["dcr_ratio", "mia_auc_domias", "mia_auc_domias_clf",
         "mia_tpr_at_fpr01", "fidelity_score"]].mean()
    log.info("\nMean over %d seeds:\n%s", len(config.SEEDS), means.to_string())
    log.info("Saved -> %s", out)

    # If dcr_ratio is not monotone in bandwidth then it is not a dial and the
    # experiment says nothing — fail loudly rather than plot it.
    ratios = means["dcr_ratio"].values
    assert np.all(np.diff(ratios) > 0), \
        f"dcr_ratio not increasing in bandwidth: {ratios}"

    leaky = means[(means["dcr_ratio"] >= config.FAA_DCR_RATIO) &
                  (means["mia_auc_domias"] > config.FAA_MIA_AUC)]
    if len(leaky):
        log.info("\nDCR gate PASSES while MIA fires at bandwidth(s): %s",
                 list(leaky.index))
    else:
        log.info("\nNo bandwidth passes Axis 3 while failing Axis 4 — on this "
                 "data the two gates flip together.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
