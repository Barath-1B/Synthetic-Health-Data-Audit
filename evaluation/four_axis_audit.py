"""
The Four-Axis Audit (FAA)
=========================
A minimum-acceptable evaluation protocol for tabular synthetic mental-health data.
Proposed in: "DCR is Necessary but Not Sufficient: A Four-Axis Audit of Synthetic
Mental Health Survey Data."

Four axes:
    Axis 1 — Statistical Fidelity:   Kolmogorov-Smirnov per column
    Axis 2 — Downstream Utility:     TSTR vs TRTR AUC ratio
    Axis 3 — NN Privacy (DCR):       Distance to Closest Record
    Axis 4 — MIA Privacy:            Shadow-model Membership Inference Attack

A generator PASSES the FAA only if all four axes pass their threshold.
"""
import logging
from pathlib import Path
import pandas as pd
import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from config import (
    DATA_PROCESSED, EVAL_RESULTS,
    FAA_KS_PASS_RATE, FAA_UTILITY_RATIO,
    FAA_DCR_MEAN, FAA_MIA_ACCURACY,
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
        model_name:   Model identifier used in output filename ('vae', 'ctgan', 'tvae')

    Returns:
        dict with keys: seed, model, ks_pass_rate, tstr_auc, trtr_auc, utility_ratio,
                        dcr_mean, mia_acc_weak, mia_acc_shadow,
                        ax1_pass…ax4_pass, faa_pass,
                        dcr_mia_gap (True when DCR passes but MIA fails)
    """
    real_train = pd.read_csv(DATA_PROCESSED / "nhanes_train.csv")
    real_test  = pd.read_csv(DATA_PROCESSED / "nhanes_test.csv")

    from evaluation.statistical import axis1_ks
    from evaluation.utility     import axis2_utility
    from evaluation.privacy     import axis3_dcr, axis4_mia_weak, axis4_mia_shadow

    ks_pass_rate           = axis1_ks(real_train, synthetic_df, seed=seed)
    tstr_auc, trtr_auc     = axis2_utility(real_train, real_test, synthetic_df, seed=seed)
    utility_ratio          = tstr_auc / trtr_auc if trtr_auc > 0 else 0.0
    dcr_mean               = axis3_dcr(real_train, synthetic_df)
    mia_acc_weak           = axis4_mia_weak(real_train, synthetic_df, seed=seed)
    mia_acc_shadow         = axis4_mia_shadow(real_train, synthetic_df, seed=seed)

    ax1_pass = ks_pass_rate  >= FAA_KS_PASS_RATE
    ax2_pass = utility_ratio >= FAA_UTILITY_RATIO
    ax3_pass = dcr_mean      >  FAA_DCR_MEAN
    ax4_pass = mia_acc_shadow <= FAA_MIA_ACCURACY

    faa_pass    = ax1_pass and ax2_pass and ax3_pass and ax4_pass
    dcr_mia_gap = ax3_pass and not ax4_pass   # DCR passes but MIA fails — the key finding

    result = {
        "model":          model_name,
        "seed":           seed,
        "ks_pass_rate":   ks_pass_rate,
        "tstr_auc":       tstr_auc,
        "trtr_auc":       trtr_auc,
        "utility_ratio":  utility_ratio,
        "dcr_mean":       dcr_mean,
        "mia_acc_weak":   mia_acc_weak,
        "mia_acc_shadow": mia_acc_shadow,
        "ax1_pass":       ax1_pass,
        "ax2_pass":       ax2_pass,
        "ax3_pass":       ax3_pass,
        "ax4_pass":       ax4_pass,
        "faa_pass":       faa_pass,
        "dcr_mia_gap":    dcr_mia_gap,
    }

    out = EVAL_RESULTS / f"faa_{model_name}_seed{seed}.csv"
    pd.DataFrame([result]).to_csv(out, index=False)

    log.info("\n[FAA seed=%d]", seed)
    log.info("  Axis 1 (KS pass rate):    %.3f  %s", ks_pass_rate,  "PASS" if ax1_pass else "FAIL")
    log.info("  Axis 2 (utility ratio):   %.3f  %s", utility_ratio, "PASS" if ax2_pass else "FAIL")
    log.info("  Axis 3 (DCR mean):        %.3f  %s", dcr_mean,      "PASS" if ax3_pass else "FAIL")
    log.info("  Axis 4 (MIA shadow acc):  %.3f  %s", mia_acc_shadow,"PASS" if ax4_pass else "FAIL")
    log.info("  FAA OVERALL: %s", "PASS" if faa_pass else "FAIL")
    if dcr_mia_gap:
        log.info("  *** DCR-vs-MIA GAP DETECTED: DCR passes but MIA fails ***")

    return result
