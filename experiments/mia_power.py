"""
experiments/mia_power.py — is the DOMIAS null a real null, or an underpowered attack?

DOMIAS estimates p_syn by KDE over the *released* synthetic records. The number
of released records is therefore a parameter of the attack, not only of the
release. At the pipeline's old N_SYNTHETIC=1000 (against 6,275 training rows) an
AUC of ~0.5 cannot be distinguished from a density estimate too coarse to
resolve local overfitting.

This sweeps one generator over release size and audits each one, so the paper
can show the null holds at full sample size rather than assert it.

  python experiments/mia_power.py

Writes evaluation/results/mia_power.csv and one faa_pow{n}_seed{seed}.csv per
point. Run BEFORE run_all_seeds.py: the final sweep point is n = |train|, which
leaves the canonical {model}_synthetic_seed{seed}.csv in the state the main
matrix expects.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from evaluation.four_axis_audit import run_audit
from generate import generate, _load_real_train
from utils.seed_utils import set_all_seeds

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

MODEL = "ctgan"   # highest-fidelity from-scratch model — the hardest null to explain away
SEED  = config.RANDOM_SEED

KEEP = ["fidelity_score", "dcr_ratio", "mia_auc_domias",
        "mia_auc_domias_clf", "mia_tpr_at_fpr01", "c2st_acc"]


def main() -> int:
    n_train = len(_load_real_train())
    # Ascending, ending at n_train: the last generate() call rewrites the
    # canonical synthetic CSV at the size the main matrix uses.
    sizes = [500, 1000, 2500, n_train]

    ckpt = config.MODELS_DIR / f"{MODEL}_seed{SEED}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"{ckpt} missing — run: python models/{MODEL}/train.py --seed {SEED}")

    rows: list[dict] = []
    for n in sizes:
        set_all_seeds(SEED)
        log.info("\n[mia_power] %s seed=%d n_synthetic=%d", MODEL, SEED, n)
        syn = generate(MODEL, ckpt, seed=SEED, n=n)

        # check_consistency.py maps faa_{name}_seed{s}.csv -> {name}_synthetic_seed{s}.csv,
        # so each sweep point needs its own synthetic file under the audit name.
        name = f"pow{n}"
        syn.to_csv(config.DATA_SYNTHETIC / f"{name}_synthetic_seed{SEED}.csv",
                   index=False)
        m = run_audit(syn, seed=SEED, model_name=name)
        rows.append({"n_synthetic": n, **{k: m[k] for k in KEEP}})

    df = pd.DataFrame(rows)
    out = config.EVAL_RESULTS / "mia_power.csv"
    df.to_csv(out, index=False)
    log.info("\n%s", df.to_string(index=False))
    log.info("Saved -> %s", out)

    assert len(df) == len(sizes), "sweep did not complete"
    assert df["mia_auc_domias"].notna().all(), "attack produced NaN AUC"
    full = df.iloc[-1]
    log.info("\nAt full release size (n=%d): AUC=%.4f  TPR@FPR=0.01=%.4f  [gate <= %.2f]",
             int(full["n_synthetic"]), full["mia_auc_domias"],
             full["mia_tpr_at_fpr01"], config.FAA_MIA_AUC)
    return 0


if __name__ == "__main__":
    sys.exit(main())
