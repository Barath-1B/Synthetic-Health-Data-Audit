"""
evaluation/attack_validation.py — positive/negative controls for the DOMIAS MIA.

Validates that the Axis-4 attack actually measures membership leakage before
it is used to audit any generator (paper appendix table):

  Positive control: "synthetic" data = noisy verbatim copies of a training
      subset (a maximally leaky generator). The attack must fire:
      AUC > 0.70 expected when members are drawn from the copied subset.

  Negative control: "synthetic" data = fresh real records from half of the
      test split (a perfectly private "generator" that samples the population
      independently of training data). The attack must NOT fire:
      AUC in [0.45, 0.55].

Usage:
  python evaluation/attack_validation.py
Writes evaluation/results/attack_validation.csv and exits non-zero on failure.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from evaluation.privacy import axis4_mia_domias

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

POSITIVE_MIN_AUC = 0.70
NEGATIVE_BAND    = (0.45, 0.55)
NOISE_STD        = 0.02   # jitter for the leaky-generator control


def _noisy_copies(subset: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Verbatim copies of real rows with small jitter on continuous columns."""
    rng = np.random.default_rng(seed)
    out = subset.copy().reset_index(drop=True)
    for col in config.CONTINUOUS_COLS:
        if col in out.columns:
            noise = rng.normal(0.0, NOISE_STD, size=len(out))
            out[col] = np.clip(out[col].values + noise, 0.0, 1.0)
    return out


def run_controls(seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Run both controls; return a results DataFrame and save it as CSV."""
    real_train = pd.read_csv(config.NHANES_TRAIN)
    real_val   = pd.read_csv(config.NHANES_VAL)
    real_test  = pd.read_csv(config.NHANES_TEST)
    rng = np.random.default_rng(seed)

    rows: list[dict] = []

    # Positive control — leaky generator: copies of a 1000-row training subset.
    # Members are drawn from that same subset so memorisation is detectable.
    subset_idx = rng.choice(len(real_train), size=min(1000, len(real_train)),
                            replace=False)
    leaked_subset = real_train.iloc[subset_idx]
    syn_pos = _noisy_copies(leaked_subset, seed)
    log.info("Positive control (noisy training copies):")
    mia_pos = axis4_mia_domias(leaked_subset, real_test, syn_pos, real_val,
                               seed=seed)
    pos_pass = mia_pos["mia_auc_domias"] > POSITIVE_MIN_AUC
    rows.append({"control": "positive_noisy_copies",
                 "mia_auc_domias": mia_pos["mia_auc_domias"],
                 "mia_auc_domias_clf": mia_pos["mia_auc_domias_clf"],
                 "expected": f"> {POSITIVE_MIN_AUC}",
                 "pass": pos_pass})

    # Negative control — private "generator": fresh real rows from one half of
    # the test split; non-members evaluated against the other half.
    test_idx = rng.permutation(len(real_test))
    half = len(test_idx) // 2
    syn_neg   = real_test.iloc[test_idx[:half]].reset_index(drop=True)
    eval_test = real_test.iloc[test_idx[half:]].reset_index(drop=True)
    log.info("Negative control (fresh real sample):")
    mia_neg = axis4_mia_domias(real_train, eval_test, syn_neg, real_val,
                               seed=seed)
    neg_pass = NEGATIVE_BAND[0] <= mia_neg["mia_auc_domias"] <= NEGATIVE_BAND[1]
    rows.append({"control": "negative_fresh_real",
                 "mia_auc_domias": mia_neg["mia_auc_domias"],
                 "mia_auc_domias_clf": mia_neg["mia_auc_domias_clf"],
                 "expected": f"in {list(NEGATIVE_BAND)}",
                 "pass": neg_pass})

    df = pd.DataFrame(rows)
    out = config.EVAL_RESULTS / "attack_validation.csv"
    df.to_csv(out, index=False)
    log.info("\n%s", df.to_string(index=False))
    log.info("Saved -> %s", out)
    return df


if __name__ == "__main__":
    results = run_controls()
    if not results["pass"].all():
        log.error("ATTACK VALIDATION FAILED — DOMIAS controls out of bounds.")
        sys.exit(1)
    log.info("Attack validation passed.")
