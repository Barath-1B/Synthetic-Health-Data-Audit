from pathlib import Path
import torch

ROOT = Path(__file__).parent

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SYNTHETIC = ROOT / "data" / "synthetic"
NHANES_CLEAN   = DATA_PROCESSED / "nhanes_clean.csv"   # legacy full dataset
NHANES_TRAIN   = DATA_PROCESSED / "nhanes_train.csv"
NHANES_VAL     = DATA_PROCESSED / "nhanes_val.csv"
NHANES_TEST    = DATA_PROCESSED / "nhanes_test.csv"
SCALER_PATH    = DATA_PROCESSED / "scaler.pkl"
SPLITS_PATH    = DATA_PROCESSED / "splits.pkl"
MODELS_DIR     = ROOT / "models" / "saved"
EVAL_PLOTS_DIR = ROOT / "evaluation" / "plots"
EVAL_RESULTS   = ROOT / "evaluation" / "results"
LOGS_DIR       = ROOT / "logs"

# Ensure output directories exist at import time
for _p in [DATA_SYNTHETIC, MODELS_DIR, EVAL_PLOTS_DIR, EVAL_RESULTS, LOGS_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Data ─────────────────────────────────────────────────────────────────────
TARGET_COL   = "Depression_Severity"
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# TEST_RATIO = 0.15 (implicit: 1 - TRAIN - VAL)
RANDOM_SEED  = 42
SEEDS        = [42, 43, 44, 45, 46]   # mandatory 5-seed variation
NUM_CLASSES  = 5   # Depression_Severity values {0,1,2,3,4}

# PHQ9_TOTAL and DPQ items are EXCLUDED from model features (leakage fix).
# They are computed during preprocessing only to derive Depression_Severity.
CONTINUOUS_COLS = [
    "Age", "Income_Ratio", "Sleep_Hours_Weekday", "Sleep_Hours_Weekend",
    "Drinking_Freq", "Avg_Drinks_Per_Day", "Binge_Drinking_Freq",
    "Total_Calories", "Total_Sugar", "Total_Caffeine",
]

# ── Four-Axis Audit thresholds ───────────────────────────────────────────────
FAA_KS_PASS_RATE    = 0.80   # axis 1: fraction of columns with KS p >= 0.05
FAA_UTILITY_RATIO   = 0.85   # axis 2: TSTR/TRTR AUC ratio
FAA_DCR_MEAN        = 0.10   # axis 3: mean distance-to-closest-record
FAA_MIA_ACCURACY    = 0.55   # axis 4: shadow-model MIA accuracy (must be <=)

# ── VAE ──────────────────────────────────────────────────────────────────────
VAE_LATENT_DIM       = 64
VAE_HIDDEN_DIMS      = [256, 128]
VAE_EPOCHS           = 200
VAE_LR               = 1e-3
VAE_BATCH_SIZE       = 256
VAE_KL_ANNEAL_EPOCHS = 50    # KL weight ramps 0→1 over this many epochs
VAE_KL_MAX_WEIGHT    = 1.0   # full KL after annealing

# ── CTGAN ────────────────────────────────────────────────────────────────────
CTGAN_NOISE_DIM        = 128
CTGAN_EPOCHS           = 300
CTGAN_BATCH_SIZE       = 500
CTGAN_GEN_DIM          = (256, 256)
CTGAN_DISC_DIM         = (256, 256)
CTGAN_LR               = 2e-4
CTGAN_GRADIENT_PENALTY = 10.0
CTGAN_DISC_STEPS       = 5   # discriminator updates per generator update

# ── TVAE baseline ────────────────────────────────────────────────────────────
TVAE_EPOCHS        = 300
TVAE_EMBEDDING_DIM = 128

# ── Generation ────────────────────────────────────────────────────────────────
N_SYNTHETIC = 1000   # rows generated per model per seed
