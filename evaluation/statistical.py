"""
evaluation/statistical.py — Axis 1: Statistical fidelity (KS tests + clinical signals).

Functional API (used by four_axis_audit.py):
  axis1_ks(real, synthetic, seed) -> float (pass rate)
  clinical_signal_check(real, synthetic) -> dict

CLI usage (backwards compat):
  python evaluation/statistical.py --model vae
  python evaluation/statistical.py --model ctgan
"""

import argparse
import logging
import pickle
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


def axis1_ks(real: pd.DataFrame, synthetic: pd.DataFrame, seed: int = 42) -> float:
    """
    Run two-sample KS test on every feature column.

    Returns fraction of columns where p >= 0.05 (KS pass rate).
    Saves failing-column histograms and correlation heatmaps to EVAL_PLOTS_DIR.
    """
    feature_cols = [c for c in real.columns if c != config.TARGET_COL]
    results: dict[str, dict] = {}
    for col in feature_cols:
        r = np.round(real[col].dropna().values.astype(float), 10)
        s = np.round(synthetic[col].dropna().values.astype(float), 10)
        stat, p = ks_2samp(r, s)
        results[col] = {"statistic": stat, "p_value": p, "pass": p >= 0.05}

    df_ks = pd.DataFrame(results).T
    pass_rate = float(df_ks["pass"].mean())

    # Overlay histograms for failing columns
    failing = df_ks[~df_ks["pass"]].index.tolist()
    if failing:
        n_show = min(len(failing), 6)
        fig, axes = plt.subplots(n_show, 1, figsize=(8, 3 * n_show))
        if n_show == 1:
            axes = [axes]
        for ax, col in zip(axes, failing[:n_show]):
            ax.hist(real[col], alpha=0.5, bins=30, label="Real", density=True)
            ax.hist(synthetic[col], alpha=0.5, bins=30, label="Synthetic", density=True)
            ax.set_title(f"{col}  (p={results[col]['p_value']:.4f})")
            ax.legend()
        plt.tight_layout()
        plt.savefig(config.EVAL_PLOTS_DIR / f"ks_failing_cols_seed{seed}.png", dpi=150)
        plt.close()

    # Correlation heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(real[feature_cols].corr(), ax=ax1, cmap="coolwarm", center=0, vmin=-1, vmax=1)
    ax1.set_title("Real data correlations")
    sns.heatmap(synthetic[feature_cols].corr(), ax=ax2, cmap="coolwarm", center=0, vmin=-1, vmax=1)
    ax2.set_title("Synthetic data correlations")
    plt.tight_layout()
    plt.savefig(config.EVAL_PLOTS_DIR / f"correlation_heatmap_seed{seed}.png", dpi=150)
    plt.close()

    log.info("  KS: %d/%d columns pass (p≥0.05). Pass rate: %.3f",
             int(df_ks["pass"].sum()), len(df_ks), pass_rate)
    return pass_rate


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


# ── Legacy CLI helpers ────────────────────────────────────────────────────────

def _load_data(model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load real test split and synthetic CSV; return both in [0,1] scale."""
    real_df = pd.read_csv(config.NHANES_CLEAN)
    with open(config.SPLITS_PATH, "rb") as f:
        splits = pickle.load(f)
    real_test = real_df.iloc[splits["test"]].reset_index(drop=True)

    synth_path = config.DATA_SYNTHETIC / f"{model}_nhanes_synthetic.csv"
    if not synth_path.exists():
        raise FileNotFoundError(
            f"Synthetic file not found: {synth_path}\n"
            f"Run: python generate.py --model {model} --n 1000"
        )
    synth_df = pd.read_csv(synth_path)

    with open(config.SCALER_PATH, "rb") as f:
        bundle = pickle.load(f)
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]
    shared = [c for c in feature_cols if c in synth_df.columns]
    synth_df[shared] = scaler.transform(synth_df[shared])
    return real_test, synth_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vae", "ctgan"], required=True)
    args = parser.parse_args()

    real, synth = _load_data(args.model)
    log.info("Real test: %s  |  Synthetic: %s", real.shape, synth.shape)

    pass_rate = axis1_ks(real, synth, seed=config.RANDOM_SEED)
    log.info("KS pass rate: %.3f", pass_rate)

    log.info("\nClinical signal check:")
    clinical_signal_check(real, synth)


if __name__ == "__main__":
    main()
