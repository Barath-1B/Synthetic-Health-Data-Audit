"""
preprocess.py — NHANES preprocessing pipeline (leakage-fixed version).

Loads raw XPT files from data/raw/, fixes the 5 dataset bugs, derives
Depression_Severity from PHQ-9 items, then drops PHQ9_TOTAL and all
DPQ items before model training to prevent target leakage.

Writes:
  data/processed/nhanes_clean.csv   — full cleaned dataset (no leakage cols)
  data/processed/nhanes_train.csv   — 70% stratified split
  data/processed/nhanes_val.csv     — 15% stratified split
  data/processed/nhanes_test.csv    — 15% stratified split
  data/processed/scaler.pkl         — fitted MinMaxScaler + column lists
  data/processed/splits.pkl         — index arrays (legacy compatibility)

Usage:
  python preprocess.py
"""

import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import config

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

np.random.seed(config.RANDOM_SEED)

PROXY = 1e-70

DPQ_COLS = {
    "DPQ010": "Little_Interest",
    "DPQ020": "Feeling_Down",
    "DPQ030": "Trouble_Sleeping_DPQ",
    "DPQ040": "Feeling_Tired",
    "DPQ050": "Poor_Appetite",
    "DPQ060": "Feeling_Bad_About_Self",
    "DPQ070": "Trouble_Concentrating",
    "DPQ080": "Moving_Slowly",
    "DPQ090": "Suicidal_Thoughts",
}
DEMO_COLS = {
    "RIAGENDR": "Gender",
    "RIDAGEYR": "Age",
    "RIDRETH1": "Race",
    "RIDRETH3": "Race_Ext",
    "DMDEDUC2": "Education",
    "DMDMARTZ": "Marital_Status",
    "INDFMPIR": "Income_Ratio",
}
MCQ_COLS = {
    "MCQ160A": "Arthritis",
    "MCQ160B": "Congestive_Heart_Failure",
    "MCQ160C": "Coronary_Heart_Disease",
    "MCQ160E": "Heart_Attack",
    "MCQ160F": "Stroke",
    "MCQ220":  "Cancer",
}
SLQ_COLS = {
    "SLD012": "Sleep_Hours_Weekday",
    "SLD013": "Sleep_Hours_Weekend",
    "SLQ050": "Trouble_Sleeping_Doc",
}
ALQ_COLS = {
    "ALQ111": "Ever_Had_Drink",
    "ALQ121": "Drinking_Freq",
    "ALQ130": "Avg_Drinks_Per_Day",
    "ALQ142": "Binge_Drinking_Freq",
}
DR1TOT_COLS = {
    "DR1TKCAL": "Total_Calories",
    "DR1TSUGR": "Total_Sugar",
    "DR1TCAFF": "Total_Caffeine",
}

DPQ_ITEMS = list(DPQ_COLS.values())
MCQ_ITEMS = list(MCQ_COLS.values())

# Columns dropped before model training — leakage prevention.
# PHQ9_TOTAL is deterministically derived from DPQ items; Depression_Severity
# is derived from PHQ9_TOTAL via fixed thresholds. Including either makes
# utility evaluation meaningless (TRTR AUC → 0.9999).
LEAKAGE_COLS = ["PHQ9_TOTAL"] + DPQ_ITEMS


def _load_xpt(filename: str, col_map: dict) -> pd.DataFrame:
    """Load an XPT file and rename columns according to col_map."""
    path = config.DATA_RAW / filename
    df = pd.read_sas(str(path), format="xport", encoding="utf-8")
    keep = ["SEQN"] + [c for c in col_map if c in df.columns]
    return df[keep].rename(columns=col_map)


def _replace_proxy(series: pd.Series, to_zero: bool = False) -> pd.Series:
    """Replace SAS proxy value (0 < x < 1e-70) with NaN or 0.0."""
    mask = (series > 0) & (series < PROXY)
    series = series.copy()
    series[mask] = 0.0 if to_zero else np.nan
    return series


def _binary_remap(series: pd.Series) -> pd.Series:
    """NHANES yes/no coding: 1->1, 2->0, 7->0, 9->0, NaN->0."""
    return series.map({1: 1, 2: 0, 7: 0, 9: 0}).fillna(0).astype(int)


def _categorise_severity(raw: int) -> int:
    """Map PHQ-9 raw score to 5-class Depression_Severity (standard clinical thresholds)."""
    if raw <= 4:  return 0
    if raw <= 9:  return 1
    if raw <= 14: return 2
    if raw <= 19: return 3
    return 4


def build_dataset() -> pd.DataFrame:
    """Load, merge, clean, and derive features. Returns cleaned DataFrame without leakage cols."""
    log.info("Loading XPT files from %s ...", config.DATA_RAW)
    dpq    = _load_xpt("P_DPQ.xpt",    DPQ_COLS)
    demo   = _load_xpt("P_DEMO.xpt",   DEMO_COLS)
    mcq    = _load_xpt("P_MCQ.xpt",    MCQ_COLS)
    slq    = _load_xpt("P_SLQ.xpt",    SLQ_COLS)
    alq    = _load_xpt("P_ALQ.xpt",    ALQ_COLS)
    dr1tot = _load_xpt("P_DR1TOT.xpt", DR1TOT_COLS)

    # DPQ as base -> only PHQ-9 completers (~8,965 rows)
    log.info("Merging with DPQ as base ...")
    df = dpq.copy()
    for other in [demo, mcq, slq, alq, dr1tot]:
        df = df.merge(other, on="SEQN", how="left")
    df = df.drop_duplicates(subset=["SEQN"])
    log.info("  Rows after merge: %d", len(df))

    # Proxy replacement (SAS stores missing as values near 5.4e-79)
    if "Total_Caffeine" in df.columns:
        df["Total_Caffeine"] = _replace_proxy(df["Total_Caffeine"], to_zero=True)
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in ("SEQN", "Total_Caffeine"):
            df[col] = _replace_proxy(df[col], to_zero=False)

    # DPQ items: keep {0,1,2,3}, sentinel 7/9 → NaN → 0
    for col in DPQ_ITEMS:
        if col in df.columns:
            df.loc[df[col].isin([7.0, 9.0]), col] = np.nan
            df[col] = df[col].fillna(0).astype(int)

    # MCQ binary remap
    for col in MCQ_ITEMS:
        if col in df.columns:
            df[col] = _binary_remap(df[col])

    # SLQ binary
    if "Trouble_Sleeping_Doc" in df.columns:
        df["Trouble_Sleeping_Doc"] = _binary_remap(df["Trouble_Sleeping_Doc"])

    # ALQ
    if "Ever_Had_Drink" in df.columns:
        df["Ever_Had_Drink"] = _binary_remap(df["Ever_Had_Drink"])
    for col, fallback, sentinels in [
        ("Drinking_Freq",       "zero",   [77.0, 99.0]),
        ("Avg_Drinks_Per_Day",  "median", [777.0, 999.0]),
        ("Binge_Drinking_Freq", "zero",   [77.0, 99.0]),
    ]:
        if col in df.columns:
            df.loc[df[col].isin(sentinels), col] = np.nan
            df[col] = df[col].fillna(0 if fallback == "zero" else df[col].median())

    # DEMO
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({1: 0, 2: 1}).fillna(df["Gender"].mode()[0]).astype(int)
    for col, bad in [("Education", [7.0, 9.0]), ("Marital_Status", [77.0, 99.0])]:
        if col in df.columns:
            df.loc[df[col].isin(bad), col] = np.nan
            df[col] = df[col].fillna(df[col].mode()[0]).astype(int)
    for col in ("Race", "Race_Ext"):
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0]).astype(int)

    # Derive target from PHQ-9 items, then drop leakage columns
    df["PHQ9_RAW"] = df[DPQ_ITEMS].sum(axis=1).astype(int)
    df["Depression_Severity"] = df["PHQ9_RAW"].apply(_categorise_severity).astype(int)

    # Drop SEQN, PHQ9_RAW, PHQ9_TOTAL, and all DPQ items (leakage prevention)
    drop_cols = ["SEQN", "PHQ9_RAW"] + LEAKAGE_COLS
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Fill remaining missings with median
    present_cont = [c for c in config.CONTINUOUS_COLS if c in df.columns]
    for col in present_cont:
        df[col] = df[col].fillna(df[col].median())

    return df


def main() -> None:
    """Run the full preprocessing pipeline."""
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = build_dataset()

    # Scale all feature columns to [0,1]
    feature_cols = [c for c in df.columns if c != config.TARGET_COL]
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    log.info("Running assertions ...")
    assert "PHQ9_TOTAL" not in df.columns, "LEAKAGE: PHQ9_TOTAL present"
    assert "SEQN" not in df.columns, "ID column present"
    for item in DPQ_ITEMS:
        assert item not in df.columns, f"DPQ item {item} present (leakage)"
    assert df.isnull().sum().sum() == 0, "Nulls remain"
    assert len(df) > 5000, f"Too few rows: {len(df)}"
    assert df["Depression_Severity"].nunique() == 5, "Wrong number of severity classes"
    assert df[feature_cols].min().min() >= 0.0
    assert df[feature_cols].max().max() <= 1.0
    log.info("  All assertions passed.")

    # Save full cleaned dataset
    df.to_csv(config.NHANES_CLEAN, index=False)
    log.info("Saved -> %s", config.NHANES_CLEAN)

    # Save scaler
    present_cont = [c for c in config.CONTINUOUS_COLS if c in df.columns]
    with open(config.SCALER_PATH, "wb") as f:
        pickle.dump({"scaler": scaler, "continuous_cols": present_cont,
                     "feature_cols": feature_cols}, f)
    log.info("Saved -> %s", config.SCALER_PATH)

    # Stratified 70/15/15 split -> separate CSV files
    y = df[config.TARGET_COL].values
    train_df, temp_df = train_test_split(
        df, test_size=1 - config.TRAIN_RATIO,
        stratify=y, random_state=config.RANDOM_SEED,
    )
    val_frac = config.VAL_RATIO / (config.VAL_RATIO + (1 - config.TRAIN_RATIO - config.VAL_RATIO))
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5,
        stratify=temp_df[config.TARGET_COL].values, random_state=config.RANDOM_SEED,
    )
    train_df.to_csv(config.NHANES_TRAIN, index=False)
    val_df.to_csv(config.NHANES_VAL,   index=False)
    test_df.to_csv(config.NHANES_TEST,  index=False)
    log.info("Splits saved -> train=%d, val=%d, test=%d",
             len(train_df), len(val_df), len(test_df))

    # Legacy splits.pkl (index arrays into nhanes_clean.csv)
    idx = np.arange(len(df))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=1 - config.TRAIN_RATIO,
        stratify=y, random_state=config.RANDOM_SEED,
    )
    val_frac_legacy = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_frac_legacy,
        stratify=y[idx_trainval], random_state=config.RANDOM_SEED,
    )
    with open(config.SPLITS_PATH, "wb") as f:
        pickle.dump({"train": idx_train, "val": idx_val, "test": idx_test}, f)
    log.info("Saved -> %s", config.SPLITS_PATH)

    log.info("\nShape: %s", df.shape)
    log.info("Feature columns (%d): %s", len(feature_cols), feature_cols)
    log.info("Depression_Severity:\n%s",
             df["Depression_Severity"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
