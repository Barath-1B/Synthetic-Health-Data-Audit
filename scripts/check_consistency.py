"""
scripts/check_consistency.py — verify saved results match the committed data.

Guards against the stale-results failure mode (results CSVs recorded from an
older pipeline than the synthetic CSVs on disk). Run after every full run and
before freezing numbers for the paper:

  python scripts/check_consistency.py

Checks, for every evaluation/results/faa_{name}_seed{seed}.csv:
  1. fidelity_score recomputed from the synthetic CSV matches the saved value
  2. dcr_ratio recomputed from the synthetic CSV matches the saved value
  3. every categorical value in the synthetic CSV is a valid real-train level
  4. labels are integers in {0..NUM_CLASSES-1}
  5. (non-ablation files only) < 1% of synthetic rows are exact row-level
     copies of real training rows on the continuous columns
  6. evaluation/results/attack_validation.csv exists and both controls pass

Exits non-zero with a list of violations if anything fails.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from evaluation.privacy import _min_dists
from evaluation.statistical import _tvd

ATOL = 1e-8
MAX_ROW_COPY_FRACTION = 0.01


def _recompute_fidelity(real: pd.DataFrame, syn: pd.DataFrame) -> float:
    feature_cols = [c for c in real.columns if c != config.TARGET_COL]
    fids = []
    for col in feature_cols:
        r = np.round(real[col].dropna().values.astype(float), 10)
        s = np.round(syn[col].dropna().values.astype(float), 10)
        if col in config.CONTINUOUS_COLS:
            stat, _ = ks_2samp(r, s)
            fids.append(1.0 - float(stat))
        else:
            fids.append(1.0 - _tvd(r, s))
    return float(np.mean(fids))


def _recompute_dcr_ratio(real_train: pd.DataFrame, real_test: pd.DataFrame,
                         syn: pd.DataFrame) -> float:
    feature_cols = [c for c in real_train.columns if c != config.TARGET_COL]
    train_arr = real_train[feature_cols].values.astype(np.float32)
    syn_mean = _min_dists(syn[feature_cols].values.astype(np.float32), train_arr).mean()
    hold_mean = _min_dists(real_test[feature_cols].values.astype(np.float32), train_arr).mean()
    return float(syn_mean / hold_mean) if hold_mean > 0 else 0.0


def _synthetic_path(name: str, seed: int) -> Path:
    """Map an faa result name to its synthetic CSV ('vae_mc' -> ablation file)."""
    if name.endswith("_mc"):
        return config.DATA_SYNTHETIC / f"{name[:-3]}_synthetic_mc_seed{seed}.csv"
    return config.DATA_SYNTHETIC / f"{name}_synthetic_seed{seed}.csv"


def _row_copy_fraction(real_train: pd.DataFrame, syn: pd.DataFrame) -> float:
    """Fraction of synthetic rows exactly matching a real train row on all
    continuous columns (rounded to 10 dp)."""
    cols = [c for c in config.CONTINUOUS_COLS if c in syn.columns]
    if not cols:
        return 0.0
    real_keys = set(map(tuple, np.round(real_train[cols].values.astype(float), 10)))
    syn_keys = map(tuple, np.round(syn[cols].values.astype(float), 10))
    return sum(k in real_keys for k in syn_keys) / len(syn)


def main() -> int:
    real_train = pd.read_csv(config.NHANES_TRAIN)
    real_test  = pd.read_csv(config.NHANES_TEST)
    feature_cols = [c for c in real_train.columns if c != config.TARGET_COL]
    cat_cols = [c for c in feature_cols if c not in config.CONTINUOUS_COLS]
    valid_levels = {c: set(np.round(real_train[c].unique().astype(float), 10))
                    for c in cat_cols}

    violations: list[str] = []
    n_checked = 0

    for faa_path in sorted(config.EVAL_RESULTS.glob("faa_*_seed*.csv")):
        stem = faa_path.stem  # faa_{name}_seed{seed}
        name, seed_part = stem[len("faa_"):].rsplit("_seed", 1)
        seed = int(seed_part)
        syn_path = _synthetic_path(name, seed)
        if not syn_path.exists():
            violations.append(f"{faa_path.name}: synthetic file missing ({syn_path.name})")
            continue

        saved = pd.read_csv(faa_path).iloc[0]
        syn = pd.read_csv(syn_path)
        n_checked += 1

        fid = _recompute_fidelity(real_train, syn)
        if abs(fid - saved["fidelity_score"]) > ATOL:
            violations.append(
                f"{faa_path.name}: fidelity_score saved={saved['fidelity_score']:.6f} "
                f"recomputed={fid:.6f} (STALE RESULTS)")

        dcr_ratio = _recompute_dcr_ratio(real_train, real_test, syn)
        if abs(dcr_ratio - saved["dcr_ratio"]) > 1e-5:
            violations.append(
                f"{faa_path.name}: dcr_ratio saved={saved['dcr_ratio']:.6f} "
                f"recomputed={dcr_ratio:.6f} (STALE RESULTS)")

        for col in cat_cols:
            vals = set(np.round(syn[col].astype(float).values, 10))
            bad = vals - valid_levels[col]
            if bad:
                violations.append(
                    f"{syn_path.name}: column {col} has {len(bad)} invalid "
                    f"levels (e.g. {sorted(bad)[:3]})")
                break  # one categorical violation per file is enough signal

        labels = syn[config.TARGET_COL].values
        if not np.array_equal(labels, labels.astype(int)) or \
           labels.min() < 0 or labels.max() > config.NUM_CLASSES - 1:
            violations.append(f"{syn_path.name}: invalid labels")

        if not name.endswith("_mc"):
            frac = _row_copy_fraction(real_train, syn)
            if frac > MAX_ROW_COPY_FRACTION:
                violations.append(
                    f"{syn_path.name}: {frac:.1%} of rows are exact continuous-"
                    f"value copies of training rows (limit {MAX_ROW_COPY_FRACTION:.0%})")

    av_path = config.EVAL_RESULTS / "attack_validation.csv"
    if not av_path.exists():
        violations.append("attack_validation.csv missing — run "
                          "python evaluation/attack_validation.py")
    else:
        av = pd.read_csv(av_path)
        if not av["pass"].all():
            violations.append("attack validation controls FAILED — see "
                              f"{av_path}")

    print(f"Checked {n_checked} result file(s).")
    if violations:
        print("\nCONSISTENCY CHECK FAILED:")
        for v in violations:
            print(f"  - {v}")
        return 1
    print("All consistency checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
