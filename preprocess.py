"""
preprocess.py — NHANES preprocessing pipeline (root entry point).

Loads raw XPT files from data/raw/, fixes the 5 dataset bugs documented in
GAN/Data Prep..md, and writes:
  data/processed/nhanes_clean.csv
  data/processed/scaler.pkl        (fitted MinMaxScaler for inverse-transform)
  data/processed/splits.pkl        (train/val/test index arrays)

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


def _load_xpt(filename: str, col_map: dict) -> pd.DataFrame:
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
    if raw <= 4:  return 0
    if raw <= 9:  return 1
    if raw <= 14: return 2
    if raw <= 19: return 3
    return 4


def build_dataset() -> tuple[pd.DataFrame, list[str]]:
    log.info("Loading XPT files from %s ...", config.DATA_RAW)
    dpq    = _load_xpt("P_DPQ.xpt",    DPQ_COLS)
    demo   = _load_xpt("P_DEMO.xpt",   DEMO_COLS)
    mcq    = _load_xpt("P_MCQ.xpt",    MCQ_COLS)
    slq    = _load_xpt("P_SLQ.xpt",    SLQ_COLS)
    alq    = _load_xpt("P_ALQ.xpt",    ALQ_COLS)
    dr1tot = _load_xpt("P_DR1TOT.xpt", DR1TOT_COLS)

    # Bug 3 fix: DPQ as base -> only PHQ-9 completers (~8,965 rows)
    log.info("Merging with DPQ as base ...")
    df = dpq.copy()
    for other in [demo, mcq, slq, alq, dr1tot]:
        df = df.merge(other, on="SEQN", how="left")
    df = df.drop_duplicates(subset=["SEQN"])
    log.info("  Rows: %d", len(df))

    # Bug 4 fix: proxy replacement
    if "Total_Caffeine" in df.columns:
        df["Total_Caffeine"] = _replace_proxy(df["Total_Caffeine"], to_zero=True)
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in ("SEQN", "Total_Caffeine"):
            df[col] = _replace_proxy(df[col], to_zero=False)

    # Bug 1 fix: DPQ items — keep {0,1,2,3}, no normalisation
    for col in DPQ_ITEMS:
        if col in df.columns:
            df.loc[df[col].isin([7.0, 9.0]), col] = np.nan
            df[col] = df[col].fillna(0).astype(int)

    # Bug 2 fix: MCQ binary remap
    for col in MCQ_ITEMS:
        if col in df.columns:
            df[col] = _binary_remap(df[col])

    # SLQ binary
    if "Trouble_Sleeping_Doc" in df.columns:
        df["Trouble_Sleeping_Doc"] = _binary_remap(df["Trouble_Sleeping_Doc"])

    # ALQ
    if "Ever_Had_Drink" in df.columns:
        df["Ever_Had_Drink"] = _binary_remap(df["Ever_Had_Drink"])
    for col, fallback in [
        ("Drinking_Freq",      "zero"),
        ("Avg_Drinks_Per_Day", "median"),
        ("Binge_Drinking_Freq","zero"),
    ]:
        if col in df.columns:
            df.loc[df[col].isin([777.0, 999.0]), col] = np.nan
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

    # Derived features
    df["PHQ9_RAW"] = df[DPQ_ITEMS].sum(axis=1).astype(int)
    df["Depression_Severity"] = df["PHQ9_RAW"].apply(_categorise_severity).astype(int)
    df["PHQ9_TOTAL"] = df["PHQ9_RAW"] / 27.0

    # Bug 5 fix: drop SEQN and PHQ9_RAW
    df = df.drop(columns=["SEQN", "PHQ9_RAW"])

    present_cont = [c for c in config.CONTINUOUS_COLS if c in df.columns]
    for col in present_cont:
        df[col] = df[col].fillna(df[col].median())

    return df, present_cont


def main() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df, present_cont = build_dataset()

    scaler = MinMaxScaler()
    df[present_cont] = scaler.fit_transform(df[present_cont])

    log.info("Running assertions ...")
    assert df.isnull().sum().sum() == 0
    assert df.shape[0] > 8000
    assert df["Depression_Severity"].nunique() == 5
    assert df[DPQ_ITEMS].max().max() == 3
    assert set(df["Arthritis"].unique()).issubset({0, 1})
    assert all(df.dtypes != object)
    for c in present_cont:
        assert df[c].min() >= 0.0
        assert df[c].max() <= 1.0
    log.info("  All assertions passed.")

    df.to_csv(config.NHANES_CLEAN, index=False)
    log.info("Saved -> %s", config.NHANES_CLEAN)

    with open(config.SCALER_PATH, "wb") as f:
        pickle.dump({"scaler": scaler, "continuous_cols": present_cont}, f)
    log.info("Saved -> %s", config.SCALER_PATH)

    idx = np.arange(len(df))
    y   = df[config.TARGET_COL].values
    idx_trainval, idx_test = train_test_split(
        idx, test_size=1 - config.TRAIN_RATIO,
        stratify=y, random_state=config.RANDOM_SEED,
    )
    val_frac = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_frac,
        stratify=y[idx_trainval], random_state=config.RANDOM_SEED,
    )
    splits = {"train": idx_train, "val": idx_val, "test": idx_test}
    with open(config.SPLITS_PATH, "wb") as f:
        pickle.dump(splits, f)
    log.info("Saved -> %s", config.SPLITS_PATH)

    log.info("\nShape: %s", df.shape)
    log.info("Depression_Severity:\n%s",
             df["Depression_Severity"].value_counts().sort_index().to_string())
    log.info("Train/Val/Test: %d / %d / %d",
             len(idx_train), len(idx_val), len(idx_test))


if __name__ == "__main__":
    main()
