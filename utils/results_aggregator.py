"""Collect per-seed metric dicts and report mean ± std."""
import numpy as np
import pandas as pd
from config import EVAL_RESULTS


def aggregate(results: list[dict], model_name: str, save: bool = True) -> pd.DataFrame:
    """
    Aggregate per-seed results into mean ± std summary.

    Args:
        results: list of dicts, one per seed, e.g.
                 [{'ks_pass_rate': 0.82, 'tstr_auc': 0.71, ...}, ...]
        model_name: 'vae' | 'ctgan' | 'tvae'
        save: write CSV to evaluation/results/{model_name}_results.csv

    Returns:
        DataFrame with columns [metric, mean, std, min, max]
    """
    df = pd.DataFrame(results)
    numeric_df = df.select_dtypes(include="number")
    summary = pd.DataFrame({
        "metric": numeric_df.columns,
        "mean":   numeric_df.mean().values,
        "std":    numeric_df.std().values,
        "min":    numeric_df.min().values,
        "max":    numeric_df.max().values,
    })
    if save:
        path = EVAL_RESULTS / f"{model_name}_results.csv"
        summary.to_csv(path, index=False)
        print(f"Saved aggregated results to {path}")
    return summary


def print_table(summaries: dict[str, pd.DataFrame]) -> None:
    """Print a side-by-side comparison table across models."""
    metrics = list(summaries.values())[0]["metric"].tolist()
    header = f"{'Metric':<25}" + "".join(f"{m:>22}" for m in summaries.keys())
    print(header)
    print("-" * len(header))
    for metric in metrics:
        row = f"{metric:<25}"
        for model, df in summaries.items():
            r = df[df["metric"] == metric].iloc[0]
            row += f"  {r['mean']:.4f} ± {r['std']:.4f}  "
        print(row)


def print_significance(per_seed: dict[str, list[dict]],
                       metrics: list[str]) -> None:
    """
    Pairwise Wilcoxon signed-rank tests across seeds for the given metrics.

    With only len(SEEDS)=5 paired samples the test is low-powered (smallest
    achievable two-sided p is 0.0625); p-values are reported descriptively,
    not as a pass/fail gate.
    """
    from itertools import combinations
    from scipy.stats import wilcoxon

    models = list(per_seed.keys())
    print(f"\n{'Pairwise Wilcoxon (across seeds)':<40}")
    for metric in metrics:
        print(f"\n  {metric}:")
        for m1, m2 in combinations(models, 2):
            v1 = np.array([r[metric] for r in per_seed[m1]], dtype=float)
            v2 = np.array([r[metric] for r in per_seed[m2]], dtype=float)
            if len(v1) != len(v2) or len(v1) < 3 or np.allclose(v1, v2):
                continue
            try:
                stat, p = wilcoxon(v1, v2)
                print(f"    {m1} vs {m2}: diff={v1.mean() - v2.mean():+.4f}  "
                      f"p={p:.4f}")
            except ValueError:
                continue
