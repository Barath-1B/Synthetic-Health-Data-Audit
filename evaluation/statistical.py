"""
evaluation/statistical.py — Axis 1: Statistical fidelity (effect sizes + clinical signals).

Axis 1 score = mean over feature columns of:
  - continuous columns:  KSComplement = 1 - KS statistic D   (SDMetrics convention)
  - categorical columns: 1 - Total Variation Distance (TVD)

Gate: fidelity_score >= FAA_FIDELITY_SCORE (0.90).

Why effect sizes, not p-values: a two-sample KS p-value gate at these sample
sizes (~6,275 real vs 1,000 synthetic) rejects on trivially small deviations —
its power grows with n, so the old "fraction of columns with p >= 0.05" gate
fails by construction for any imperfect generator. Raw p-values are still
recorded per column as supplementary output.

Functional API (used by four_axis_audit.py):
  axis1_fidelity(real, synthetic, seed, model_name) -> (score, per_col_df)
  clinical_signal_check(real, synthetic) -> dict

CLI:
  python evaluation/statistical.py --model vae --seed 42
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — avoids tkinter threading errors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _tvd(real_vals: np.ndarray, syn_vals: np.ndarray) -> float:
    """Total Variation Distance between two empirical categorical distributions."""
    levels = np.union1d(np.unique(real_vals), np.unique(syn_vals))
    p = np.array([(real_vals == lv).mean() for lv in levels])
    q = np.array([(syn_vals == lv).mean() for lv in levels])
    return 0.5 * float(np.abs(p - q).sum())


def axis1_fidelity(real: pd.DataFrame, synthetic: pd.DataFrame,
                   seed: int = 42, model_name: str = "unknown"
                   ) -> tuple[float, pd.DataFrame]:
    """
    Per-column marginal fidelity via effect sizes.

    Returns:
        (fidelity_score, per_col_df) — score is the mean per-column fidelity;
        per_col_df has columns [column, kind, fidelity, ks_stat, ks_pvalue].
    Saves lowest-fidelity histograms and correlation heatmaps to EVAL_PLOTS_DIR.
    """
    feature_cols = [c for c in real.columns if c != config.TARGET_COL]
    rows: list[dict] = []
    for col in feature_cols:
        r = np.round(real[col].dropna().values.astype(float), 10)
        s = np.round(synthetic[col].dropna().values.astype(float), 10)
        stat, p = ks_2samp(r, s)
        if col in config.CONTINUOUS_COLS:
            kind, fid = "continuous", 1.0 - float(stat)      # KSComplement
        else:
            kind, fid = "categorical", 1.0 - _tvd(r, s)      # 1 - TVD
        rows.append({"column": col, "kind": kind, "fidelity": fid,
                     "ks_stat": float(stat), "ks_pvalue": float(p)})

    per_col = pd.DataFrame(rows)
    score = float(per_col["fidelity"].mean())

    # Overlay histograms for the lowest-fidelity columns
    worst = per_col.nsmallest(min(6, len(per_col)), "fidelity")["column"].tolist()
    if worst:
        fig, axes = plt.subplots(len(worst), 1, figsize=(8, 3 * len(worst)))
        if len(worst) == 1:
            axes = [axes]
        for ax, col in zip(axes, worst):
            fid = per_col.loc[per_col["column"] == col, "fidelity"].iloc[0]
            ax.hist(real[col], alpha=0.5, bins=30, label="Real", density=True)
            ax.hist(synthetic[col], alpha=0.5, bins=30, label="Synthetic", density=True)
            ax.set_title(f"{col}  (fidelity={fid:.3f})")
            ax.legend()
        plt.tight_layout()
        plt.savefig(config.EVAL_PLOTS_DIR /
                    f"fidelity_worst_cols_{model_name}_seed{seed}.png", dpi=150)
        plt.close()

    # Correlation heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(real[feature_cols].corr(), ax=ax1, cmap="coolwarm", center=0, vmin=-1, vmax=1)
    ax1.set_title("Real data correlations")
    sns.heatmap(synthetic[feature_cols].corr(), ax=ax2, cmap="coolwarm", center=0, vmin=-1, vmax=1)
    ax2.set_title("Synthetic data correlations")
    plt.tight_layout()
    plt.savefig(config.EVAL_PLOTS_DIR /
                f"correlation_heatmap_{model_name}_seed{seed}.png", dpi=150)
    plt.close()

    log.info("  Fidelity score: %.3f (mean of %d columns)  [target >= %.2f]",
             score, len(per_col), config.FAA_FIDELITY_SCORE)
    return score, per_col


def clinical_signal_check(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
    """
    Verify that known clinical correlations from literature are preserved.

    Any Spearman correlation that diverges > 0.15 from the real data is flagged.
    Uses renamed column names (post-preprocessing).
    """
    warnings.filterwarnings("ignore")
    # (feature_col, target_col, expected_direction)
    PAIRS = [
        ("Age",                 config.TARGET_COL, "positive"),
        ("Sleep_Hours_Weekday", config.TARGET_COL, "negative"),
        ("Avg_Drinks_Per_Day",  config.TARGET_COL, "positive"),
    ]
    results: dict[str, dict] = {}
    for col_a, col_b, _ in PAIRS:
        if col_a not in real.columns or col_b not in real.columns:
            continue
        r_real, _ = spearmanr(real[col_a], real[col_b])
        r_syn, _  = spearmanr(synthetic[col_a], synthetic[col_b])
        divergence = abs(r_real - r_syn)
        flagged = divergence > 0.15
        key = f"{col_a}_vs_{col_b}"
        results[key] = {
            "real_spearman": float(r_real),
            "syn_spearman":  float(r_syn),
            "divergence":    float(divergence),
            "flagged":       flagged,
        }
        status = "FLAGGED" if flagged else "OK"
        log.info("  Clinical: %s vs %s: real=%.3f syn=%.3f gap=%.3f %s",
                 col_a, col_b, r_real, r_syn, divergence, status)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    real_train = pd.read_csv(config.NHANES_TRAIN)
    synth_path = config.DATA_SYNTHETIC / f"{args.model}_synthetic_seed{args.seed}.csv"
    if not synth_path.exists():
        raise FileNotFoundError(f"Synthetic file not found: {synth_path}")
    synth_df = pd.read_csv(synth_path)

    score, per_col = axis1_fidelity(real_train, synth_df,
                                    seed=args.seed, model_name=args.model)
    log.info("Fidelity score: %.3f", score)
    log.info("\nPer-column fidelity:\n%s",
             per_col.sort_values("fidelity").to_string(index=False))

    log.info("\nClinical signal check:")
    clinical_signal_check(real_train, synth_df)


if __name__ == "__main__":
    main()
