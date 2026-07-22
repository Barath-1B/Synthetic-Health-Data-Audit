"""
External SDV baselines: TVAE ('tvae') and Gaussian Copula ('gc').

TVAE is the deep-learning reference point the two from-scratch models are
compared against. Gaussian Copula is the near-zero-cost statistical floor —
marginals plus a Gaussian copula over rank correlations — included so the deep
generators are measured against something cheap, not only against each other.

Reference: SDV library — https://sdv.dev/

Both are driven through one pair of functions:
  train_sdv(kind, seed)                    -> models/saved/{kind}_seed{seed}.pkl
  generate_sdv(kind, checkpoint, n, seed)  -> pd.DataFrame

The caller (run_all_seeds.py) snaps categoricals and writes the synthetic CSV;
this module deliberately does not, so there is one writer per artefact.
"""
import logging
from pathlib import Path

import pandas as pd

from sdv.single_table import GaussianCopulaSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from config import DATA_PROCESSED, MODELS_DIR, TARGET_COL, N_SYNTHETIC
from utils.seed_utils import set_all_seeds

log = logging.getLogger(__name__)

# kind -> (synthesizer class, constructor kwargs). These hyperparameters are
# frozen: changing one retrains a different model and invalidates the saved
# checkpoints backing the paper's results table.
SYNTHESIZERS: dict[str, tuple[type, dict]] = {
    "tvae": (TVAESynthesizer, dict(
        epochs=config.TVAE_EPOCHS,
        batch_size=500,
        embedding_dim=config.TVAE_EMBEDDING_DIM,
        compress_dims=(256, 256),
        decompress_dims=(256, 256),
        l2scale=1e-5,
        loss_factor=2,
        enforce_min_max_values=True,
        enforce_rounding=False,
        verbose=True,
    )),
    "gc": (GaussianCopulaSynthesizer, dict(
        enforce_min_max_values=True,
        enforce_rounding=False,
    )),
}


def build_metadata(df: pd.DataFrame) -> SingleTableMetadata:
    """Auto-build SDV metadata from the processed dataframe."""
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    metadata.update_column(column_name=TARGET_COL, sdtype="categorical")
    return metadata


def train_sdv(kind: str, seed: int = 42) -> Path:
    """
    Fit an SDV synthesizer on the NHANES training split.

    Args:
        kind: 'tvae' | 'gc'
        seed: Random seed for reproducibility

    Returns:
        Path to the saved synthesizer checkpoint (.pkl)
    """
    cls, kwargs = SYNTHESIZERS[kind]
    set_all_seeds(seed)
    train_df = pd.read_csv(DATA_PROCESSED / "nhanes_train.csv")

    synthesizer = cls(build_metadata(train_df), **kwargs)
    synthesizer.fit(train_df)

    save_path = MODELS_DIR / f"{kind}_seed{seed}.pkl"
    synthesizer.save(str(save_path))
    log.info("%s checkpoint saved: %s", kind.upper(), save_path)
    return save_path


def generate_sdv(kind: str, checkpoint_path: Path, n: int = None,
                 seed: int = 42) -> pd.DataFrame:
    """
    Load a saved SDV synthesizer and sample n synthetic rows.

    Args:
        kind: 'tvae' | 'gc'
        checkpoint_path: Path to the .pkl saved by train_sdv()
        n: Number of rows to generate (defaults to config.N_SYNTHETIC)
        seed: Random seed

    Returns:
        DataFrame with the same schema as the training data. Categorical
        columns still hold raw SDV output — the caller snaps them.
    """
    cls, _ = SYNTHESIZERS[kind]
    set_all_seeds(seed)
    synthesizer = cls.load(str(checkpoint_path))
    # SDV samples under its own FIXED_RNG_SEED and ignores the global RNG, so
    # without this every seed returns byte-identical rows (GaussianCopula's fit
    # is deterministic, so its whole 5-seed spread collapsed to one sample).
    synthesizer._set_random_state(seed)
    synthetic = synthesizer.sample(num_rows=n or N_SYNTHETIC)
    log.info("Generated %d %s rows", len(synthetic), kind.upper())
    return synthetic
