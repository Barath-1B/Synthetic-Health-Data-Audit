"""
Gaussian Copula external baseline using the SDV library.

The classic shallow statistical baseline: fits marginal distributions plus a
Gaussian copula over the rank correlations. Included so the deep generators
are compared against a near-zero-cost method, not only against each other.

Reference: SDV library — https://sdv.dev/
"""
import logging
from pathlib import Path

import pandas as pd

from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from config import DATA_PROCESSED, MODELS_DIR, DATA_SYNTHETIC, TARGET_COL, N_SYNTHETIC
from utils.seed_utils import set_all_seeds

log = logging.getLogger(__name__)


def build_metadata(df: pd.DataFrame) -> SingleTableMetadata:
    """Auto-build SDV metadata from the processed dataframe."""
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    metadata.update_column(column_name=TARGET_COL, sdtype="categorical")
    return metadata


def train_gc(seed: int = 42) -> Path:
    """
    Train a Gaussian Copula synthesizer on NHANES processed training data.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Path to saved synthesizer checkpoint (.pkl)
    """
    set_all_seeds(seed)
    train_df = pd.read_csv(DATA_PROCESSED / "nhanes_train.csv")

    metadata = build_metadata(train_df)
    synthesizer = GaussianCopulaSynthesizer(
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=False,
    )
    synthesizer.fit(train_df)

    save_path = MODELS_DIR / f"gc_seed{seed}.pkl"
    synthesizer.save(str(save_path))
    log.info("Gaussian Copula checkpoint saved: %s", save_path)
    return save_path


def generate_gc(checkpoint_path: Path, n: int = None, seed: int = 42) -> pd.DataFrame:
    """
    Load a saved Gaussian Copula synthesizer and generate n synthetic rows.

    Args:
        checkpoint_path: Path to .pkl file saved by train_gc()
        n: Number of rows to generate (defaults to config.N_SYNTHETIC)
        seed: Random seed

    Returns:
        DataFrame of synthetic data with same schema as training data
    """
    set_all_seeds(seed)
    n = n or N_SYNTHETIC
    synthesizer = GaussianCopulaSynthesizer.load(str(checkpoint_path))
    synthetic = synthesizer.sample(num_rows=n)

    out_path = DATA_SYNTHETIC / f"gc_synthetic_seed{seed}.csv"
    synthetic.to_csv(out_path, index=False)
    log.info("Generated %d Gaussian Copula rows → %s", len(synthetic), out_path)
    return synthetic
