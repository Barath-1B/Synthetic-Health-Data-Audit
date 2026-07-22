"""
The Four-Axis Audit (FAA)
=========================
A minimum-acceptable evaluation checklist for tabular synthetic mental-health
data, instantiated with holdout-calibrated metrics. (Framework lineage:
TAPAS, SynthEval; attack: DOMIAS, van Breugel et al. 2023.)

Four axes:
    Axis 1 — Statistical Fidelity:  per-column effect sizes (KSComplement / 1-TVD)
    Axis 2 — Downstream Utility:    TSTR vs TRTR AUC ratio (RandomForest)
    Axis 3 — NN Privacy (DCR):      DCR(syn->train) calibrated by DCR(test->train)
    Axis 4 — MIA Privacy:           DOMIAS density-ratio attack ROC AUC

A generator PASSES the FAA only if all four axes pass.
Reported alongside (non-gating): classifier two-sample test (distinguishability),
classifier-based DOMIAS variant, clinical-signal preservation flags.
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    DATA_PROCESSED, EVAL_RESULTS,
    FAA_FIDELITY_SCORE, FAA_UTILITY_RATIO,
    FAA_DCR_RATIO, FAA_MIA_AUC,
    TARGET_COL,
)

log = logging.getLogger(__name__)


def run_audit(synthetic_df: pd.DataFrame, seed: int,
              model_name: str = "unknown") -> dict:
    """
    Run all four FAA axes and return a metrics dict.

    Args:
        synthetic_df: Generated synthetic data (same schema as real data,
                      already scaled to [0, 1])
        seed:         The random seed used for this run
        model_name:   Model identifier used in output filenames

    Returns:
        dict with the per-axis metrics, pass flags, faa_pass, and dcr_mia_gap
        (True when Axis 3 passes but Axis 4 fails — the headline observation).

    Side effects (all under evaluation/results/):
        faa_{model}_seed{seed}.csv             one-row summary
        fidelity_cols_{model}_seed{seed}.csv   per-column fidelity detail
        utility_per_class_{model}_seed{seed}.csv  per-class AUC/F1 detail
        dcr_raw_{model}_seed{seed}.npy         per-row synthetic DCR (figures)
        dcr_raw_holdout.npy                    per-row holdout DCR (shared)
    """
    real_train = pd.read_csv(DATA_PROCESSED / "nhanes_train.csv")
    real_val   = pd.read_csv(DATA_PROCESSED / "nhanes_val.csv")
    real_test  = pd.read_csv(DATA_PROCESSED / "nhanes_test.csv")

    from evaluation.statistical import axis1_fidelity, clinical_signal_check
    from evaluation.utility     import axis2_utility
    from evaluation.privacy     import (axis3_dcr, axis4_mia_domias,
                                        distinguishability_c2st)

    # Axis 1 — fidelity (effect sizes)
    fidelity_score, per_col = axis1_fidelity(real_train, synthetic_df,
                                             seed=seed, model_name=model_name)
    per_col.to_csv(EVAL_RESULTS / f"fidelity_cols_{model_name}_seed{seed}.csv",
                   index=False)

    # Axis 2 — utility (RF gates; LR reported)
    utility, per_class = axis2_utility(real_train, real_test, synthetic_df,
                                       seed=seed)
    per_class.to_csv(
        EVAL_RESULTS / f"utility_per_class_{model_name}_seed{seed}.csv",
        index=False)

    # Axis 3 — holdout-calibrated DCR
    dcr = axis3_dcr(real_train, real_test, synthetic_df)
    np.save(EVAL_RESULTS / f"dcr_raw_{model_name}_seed{seed}.npy",
            dcr["dcr_syn_raw"])
    holdout_npy = EVAL_RESULTS / "dcr_raw_holdout.npy"
    if not holdout_npy.exists():
        np.save(holdout_npy, dcr["dcr_holdout_raw"])

    # Axis 4 — DOMIAS MIA (KDE gates; clf variant reported)
    mia = axis4_mia_domias(real_train, real_test, synthetic_df, real_val,
                           seed=seed)

    # Non-gating: distinguishability + clinical signals
    c2st_acc = distinguishability_c2st(real_train, synthetic_df, seed=seed)
    clinical = clinical_signal_check(real_train, synthetic_df)
    clinical_flags = int(sum(v["flagged"] for v in clinical.values()))

    ax1_pass = fidelity_score             >= FAA_FIDELITY_SCORE
    ax2_pass = utility["utility_ratio"]   >= FAA_UTILITY_RATIO
    ax3_pass = dcr["dcr_ratio"]           >= FAA_DCR_RATIO
    ax4_pass = mia["mia_auc_domias"]      <= FAA_MIA_AUC

    faa_pass    = ax1_pass and ax2_pass and ax3_pass and ax4_pass
    dcr_mia_gap = ax3_pass and not ax4_pass   # DCR passes but MIA fails

    result = {
        "model":               model_name,
        "seed":                seed,
        "fidelity_score":      fidelity_score,
        "tstr_auc":            utility["tstr_auc"],
        "trtr_auc":            utility["trtr_auc"],
        "utility_ratio":       utility["utility_ratio"],
        "tstr_auc_lr":         utility["tstr_auc_lr"],
        "trtr_auc_lr":         utility["trtr_auc_lr"],
        "utility_ratio_lr":    utility["utility_ratio_lr"],
        "dcr_syn_mean":        dcr["dcr_syn_mean"],
        "dcr_holdout_mean":    dcr["dcr_holdout_mean"],
        "dcr_ratio":           dcr["dcr_ratio"],
        "dcr_share_close":     dcr["dcr_share_close"],
        "mia_auc_domias":      mia["mia_auc_domias"],
        "mia_auc_domias_clf":  mia["mia_auc_domias_clf"],
        "mia_tpr_at_fpr01":    mia["mia_tpr_at_fpr01"],
        "c2st_acc":            c2st_acc,
        "clinical_flags":      clinical_flags,
        "ax1_pass":            ax1_pass,
        "ax2_pass":            ax2_pass,
        "ax3_pass":            ax3_pass,
        "ax4_pass":            ax4_pass,
        "faa_pass":            faa_pass,
        "dcr_mia_gap":         dcr_mia_gap,
    }

    out = EVAL_RESULTS / f"faa_{model_name}_seed{seed}.csv"
    pd.DataFrame([result]).to_csv(out, index=False)

    log.info("\n[FAA %s seed=%d]", model_name, seed)
    log.info("  Axis 1 (fidelity score):  %.3f  %s", fidelity_score,        "PASS" if ax1_pass else "FAIL")
    log.info("  Axis 2 (utility ratio):   %.3f  %s", utility["utility_ratio"], "PASS" if ax2_pass else "FAIL")
    log.info("  Axis 3 (DCR ratio):       %.3f  %s", dcr["dcr_ratio"],      "PASS" if ax3_pass else "FAIL")
    log.info("  Axis 4 (DOMIAS MIA AUC):  %.3f  %s", mia["mia_auc_domias"], "PASS" if ax4_pass else "FAIL")
    log.info("  FAA OVERALL: %s", "PASS" if faa_pass else "FAIL")
    if dcr_mia_gap:
        log.info("  *** DCR-vs-MIA GAP: DCR calibration passes but MIA fails ***")

    return result
