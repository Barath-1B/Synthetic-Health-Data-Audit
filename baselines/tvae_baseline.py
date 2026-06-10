"""
TVAE external baseline using the SDV library.
Used as the third column in FAA results tables.

Reference: SDV library — https://sdv.dev/
"""
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from sdv.single_table import TVAESynthesizer
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


def train_tvae(seed: int = 42) -> Path:
    """
    Train TVAE on NHANES processed training data.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Path to saved synthesizer checkpoint (.pkl)
    """
    set_all_seeds(seed)
    train_df = pd.read_csv(DATA_PROCESSED / "nhanes_train.csv")

    metadata = build_metadata(train_df)
    synthesizer = TVAESynthesizer(
        metadata,
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
    )
    synthesizer.fit(train_df)

    save_path = MODELS_DIR / f"tvae_seed{seed}.pkl"
    synthesizer.save(str(save_path))
    log.info("TVAE checkpoint saved: %s", save_path)
    return save_path


def generate_tvae(checkpoint_path: Path, n: int = None, seed: int = 42) -> pd.DataFrame:
    """
    Load a saved TVAE synthesizer and generate n synthetic rows.

    Args:
        checkpoint_path: Path to .pkl file saved by train_tvae()
        n: Number of rows to generate (defaults to config.N_SYNTHETIC)
        seed: Random seed

    Returns:
        DataFrame of synthetic data with same schema as training data
    """
    set_all_seeds(seed)
    n = n or N_SYNTHETIC
    synthesizer = TVAESynthesizer.load(str(checkpoint_path))
    synthetic = synthesizer.sample(num_rows=n)

    out_path = DATA_SYNTHETIC / f"tvae_synthetic_seed{seed}.csv"
    synthetic.to_csv(out_path, index=False)
    log.info("Generated %d TVAE rows → %s", len(synthetic), out_path)
    return synthetic
