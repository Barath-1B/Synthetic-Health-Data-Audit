"""
evaluation/statistical.py — Statistical fidelity of synthetic vs real data.

Runs:
  - Per-column KS test (real vs synthetic)
  - Pearson correlation matrix comparison (side-by-side heatmap)
  - Overlaid histogram for each feature

Usage:
  python evaluation/statistical.py --model vae
  python evaluation/statistical.py --model ctgan
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def load_data(model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load real test split and synthetic CSV."""
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
    return real_test, synth_df


def ks_tests(real: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    """Run KS test on each shared column; flag failures (p < 0.05)."""
    shared = [c for c in real.columns if c in synth.columns]
    rows = []
    for col in shared:
        stat, p = ks_2samp(real[col].dropna(), synth[col].dropna())
        rows.append({"column": col, "ks_stat": round(stat, 4),
                     "p_value": round(p, 4), "pass": p >= 0.05})
    return pd.DataFrame(rows)


def correlation_heatmap(real: pd.DataFrame, synth: pd.DataFrame, plots_dir: Path) -> None:
    """Side-by-side Pearson correlation heatmaps."""
    shared = [c for c in real.columns if c in synth.columns]
    corr_r = real[shared].corr()
    corr_s = synth[shared].corr()

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    kw = dict(cmap="coolwarm", vmin=-1, vmax=1, square=True, linewidths=0.3, annot=False)
    sns.heatmap(corr_r, ax=axes[0], **kw)
    axes[0].set_title("Real data — correlation matrix")
    sns.heatmap(corr_s, ax=axes[1], **kw)
    axes[1].set_title("Synthetic data — correlation matrix")

    out = plots_dir / "correlation.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    log.info("Saved correlation heatmap -> %s", out)


def overlaid_histograms(real: pd.DataFrame, synth: pd.DataFrame, plots_dir: Path) -> None:
    """Overlaid histogram for each feature."""
    shared = [c for c in real.columns if c in synth.columns]
    for col in shared:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(real[col].dropna(),  bins=30, alpha=0.5, label="Real",      density=True)
        ax.hist(synth[col].dropna(), bins=30, alpha=0.5, label="Synthetic", density=True)
        ax.set_title(col)
        ax.legend()
        out = plots_dir / f"hist_{col}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=100)
        plt.close()
    log.info("Saved %d histograms -> %s", len(shared), plots_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vae", "ctgan"], required=True)
    args = parser.parse_args()

    config.EVAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    real, synth = load_data(args.model)
    log.info("Real test: %s  |  Synthetic: %s", real.shape, synth.shape)

    # KS tests
    ks_df = ks_tests(real, synth)
    n_pass = ks_df["pass"].sum()
    n_total = len(ks_df)
    log.info("\nKS test results (%d/%d columns pass p>=0.05):", n_pass, n_total)
    log.info(ks_df.to_string(index=False))

    out_csv = config.EVAL_PLOTS_DIR / f"ks_results_{args.model}.csv"
    ks_df.to_csv(out_csv, index=False)
    log.info("Saved KS results -> %s", out_csv)

    # Plots
    correlation_heatmap(real, synth, config.EVAL_PLOTS_DIR)
    overlaid_histograms(real, synth, config.EVAL_PLOTS_DIR)


if __name__ == "__main__":
    main()
