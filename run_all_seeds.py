"""run_all_seeds.py — generate synthetic data and run the FAA for all models × seeds.

Skips re-training if a checkpoint already exists for a given (model, seed).

Usage:
  python run_all_seeds.py                  # main matrix: 4 models × 5 seeds
  python run_all_seeds.py --ablation-mc    # marginal-correction ablation
                                           # (applied uniformly to ALL models,
                                           #  audited as '{model}_mc')
"""
import argparse
import logging
from pathlib import Path

from config import (SEEDS, MODELS_DIR, DATA_SYNTHETIC,
                    TARGET_COL, CONTINUOUS_COLS)
from utils.seed_utils import set_all_seeds
from utils.results_aggregator import aggregate, print_table, print_significance

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

MODELS = ["vae", "ctgan", "tvae", "gc"]

SIGNIFICANCE_METRICS = ["fidelity_score", "utility_ratio",
                        "dcr_ratio", "mia_auc_domias"]


def _ckpt_path(model_name: str, seed: int) -> Path:
    ext = ".pkl" if model_name in ("tvae", "gc") else ".pt"
    return MODELS_DIR / f"{model_name}_seed{seed}{ext}"


def run_model(model_name: str, seed: int, marginal_correction: bool = False) -> dict:
    """
    For a given (model, seed): load existing checkpoint or train if missing,
    generate synthetic data, run the Four-Axis Audit, and return metrics.

    Every model releases as many synthetic rows as there are training rows.
    This is not cosmetic: the DOMIAS attack estimates the synthetic density
    from exactly these rows, so the release size is a parameter of the attack.
    At the old N_SYNTHETIC=1000 an AUC of ~0.5 is indistinguishable from an
    underpowered attack — see experiments/mia_power.py.

    Args:
        model_name: 'vae' | 'ctgan' | 'tvae' | 'gc'
        seed: random seed integer
        marginal_correction: ablation — rank-remap continuous marginals onto
            the real training CDF (copies real floats; audited as '{model}_mc')

    Returns:
        FAA metrics dict
    """
    set_all_seeds(seed)
    ckpt = _ckpt_path(model_name, seed)
    from generate import _load_real_train
    n_synth = len(_load_real_train())

    if model_name in ("vae", "ctgan"):
        if not ckpt.exists():
            log.info("No checkpoint found for %s seed=%d — training now...",
                     model_name, seed)
            if model_name == "vae":
                from models.vae.train import train_vae
                ckpt = train_vae(seed=seed)
            else:
                from models.ctgan.train import train_ctgan
                ckpt = train_ctgan(seed=seed)
        from generate import generate
        synthetic = generate(model_name, ckpt, seed=seed, n=n_synth,
                             marginal_correction=marginal_correction)

    elif model_name in ("tvae", "gc"):
        from baselines.sdv_baselines import train_sdv, generate_sdv
        if not ckpt.exists():
            log.info("No checkpoint found for %s seed=%d — training now...",
                     model_name, seed)
            ckpt = train_sdv(model_name, seed=seed)
        synthetic = generate_sdv(model_name, ckpt, n=n_synth, seed=seed)
        # SDV emits continuous values for categorical columns — snap to valid
        # levels, same post-processing VAE/CTGAN get (fair comparison).
        from generate import _discretize_categoricals
        rt = _load_real_train()
        feats = [c for c in rt.columns if c != TARGET_COL]
        cats = [c for c in feats if c not in CONTINUOUS_COLS]
        synthetic = _discretize_categoricals(synthetic, cats, rt)
        synthetic.to_csv(DATA_SYNTHETIC / f"{model_name}_synthetic_seed{seed}.csv",
                         index=False)
        if marginal_correction:
            # SDV wrappers do not post-process; apply the ablation uniformly.
            from generate import apply_marginal_correction
            synthetic = apply_marginal_correction(synthetic, rt)
            out = DATA_SYNTHETIC / f"{model_name}_synthetic_mc_seed{seed}.csv"
            synthetic.to_csv(out, index=False)
            log.info("Marginal-corrected (ablation) -> %s", out)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    audit_name = f"{model_name}_mc" if marginal_correction else model_name
    from evaluation.four_axis_audit import run_audit
    metrics = run_audit(synthetic, seed=seed, model_name=audit_name)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-mc", action="store_true",
                        help="Run the marginal-correction ablation for all models.")
    args = parser.parse_args()

    all_summaries: dict = {}
    per_seed_results: dict[str, list[dict]] = {}
    for model_name in MODELS:
        label = f"{model_name}_mc" if args.ablation_mc else model_name
        print(f"\n{'='*50}")
        print(f"Running {label.upper()} across {len(SEEDS)} seeds")
        print("=" * 50)
        results = []
        for seed in SEEDS:
            print(f"  Seed {seed}...")
            metrics = run_model(model_name, seed,
                                marginal_correction=args.ablation_mc)
            results.append(metrics)
            print(
                f"    Fidelity={metrics['fidelity_score']:.3f} | "
                f"Utility={metrics['utility_ratio']:.3f} | "
                f"DCRratio={metrics['dcr_ratio']:.3f} | "
                f"MIA-AUC={metrics['mia_auc_domias']:.3f}"
            )
        all_summaries[label] = aggregate(results, label)
        per_seed_results[label] = results

    print("\n\n=== FINAL COMPARISON TABLE ===\n")
    print_table(all_summaries)
    print_significance(per_seed_results, SIGNIFICANCE_METRICS)
