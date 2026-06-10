"""run_all_seeds.py — generate synthetic data and run FAA for all models × all seeds.

Skips re-training if a checkpoint already exists for a given (model, seed).
"""
import logging
from pathlib import Path
from config import SEEDS, MODELS_DIR, N_SYNTHETIC
from utils.seed_utils import set_all_seeds
from utils.results_aggregator import aggregate, print_table

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _ckpt_path(model_name: str, seed: int) -> Path:
    ext = ".pkl" if model_name == "tvae" else ".pt"
    return MODELS_DIR / f"{model_name}_seed{seed}{ext}"


def run_model(model_name: str, seed: int) -> dict:
    """
    For a given (model, seed): load existing checkpoint or train if missing,
    generate synthetic data, run the Four-Axis Audit, and return metrics.

    Args:
        model_name: 'vae' | 'ctgan' | 'tvae'
        seed: random seed integer

    Returns:
        FAA metrics dict
    """
    set_all_seeds(seed)
    ckpt = _ckpt_path(model_name, seed)

    if model_name == "vae":
        if not ckpt.exists():
            log.info("No checkpoint found for vae seed=%d — training now...", seed)
            from models.vae.train import train_vae
            ckpt = train_vae(seed=seed)
        from generate import generate
        synthetic = generate(model_name, ckpt, seed=seed)

    elif model_name == "ctgan":
        if not ckpt.exists():
            log.info("No checkpoint found for ctgan seed=%d — training now...", seed)
            from models.ctgan.train import train_ctgan
            ckpt = train_ctgan(seed=seed)
        from generate import generate
        synthetic = generate(model_name, ckpt, seed=seed)

    elif model_name == "tvae":
        if not ckpt.exists():
            log.info("No checkpoint found for tvae seed=%d — training now...", seed)
            from baselines.tvae_baseline import train_tvae
            ckpt = train_tvae(seed=seed)
        from baselines.tvae_baseline import generate_tvae
        synthetic = generate_tvae(ckpt, n=N_SYNTHETIC, seed=seed)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    from evaluation.four_axis_audit import run_audit
    metrics = run_audit(synthetic, seed=seed, model_name=model_name)
    return metrics


if __name__ == "__main__":
    all_summaries = {}
    for model_name in ["vae", "ctgan", "tvae"]:
        print(f"\n{'='*50}")
        print(f"Running {model_name.upper()} across {len(SEEDS)} seeds")
        print("=" * 50)
        results = []
        for seed in SEEDS:
            print(f"  Seed {seed}...")
            metrics = run_model(model_name, seed)
            results.append(metrics)
            print(
                f"    KS={metrics['ks_pass_rate']:.3f} | "
                f"TSTR={metrics['tstr_auc']:.3f} | "
                f"DCR={metrics['dcr_mean']:.3f} | "
                f"MIA={metrics['mia_acc_shadow']:.3f}"
            )
        all_summaries[model_name] = aggregate(results, model_name)

    print("\n\n=== FINAL COMPARISON TABLE ===\n")
    print_table(all_summaries)
