from pathlib import Path

ROOT = Path(__file__).parent

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SYNTHETIC = ROOT / "data" / "synthetic"
NHANES_CLEAN   = DATA_PROCESSED / "nhanes_clean.csv"
SCALER_PATH    = DATA_PROCESSED / "scaler.pkl"
SPLITS_PATH    = DATA_PROCESSED / "splits.pkl"
MODELS_DIR     = ROOT / "models" / "saved"
EVAL_PLOTS_DIR = ROOT / "evaluation" / "plots"

# ── Data ─────────────────────────────────────────────────────────────────────
TARGET_COL   = "Depression_Severity"
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# TEST_RATIO = 0.15 (implicit: 1 - TRAIN - VAL)
RANDOM_SEED  = 42

# Feature groups (used by models and evaluation)
DPQ_ITEMS = [
    "Little_Interest", "Feeling_Down", "Trouble_Sleeping_DPQ", "Feeling_Tired",
    "Poor_Appetite", "Feeling_Bad_About_Self", "Trouble_Concentrating",
    "Moving_Slowly", "Suicidal_Thoughts",
]
CONTINUOUS_COLS = [
    "Age", "Income_Ratio", "Sleep_Hours_Weekday", "Sleep_Hours_Weekend",
    "Drinking_Freq", "Avg_Drinks_Per_Day", "Binge_Drinking_Freq",
    "Total_Calories", "Total_Sugar", "Total_Caffeine", "PHQ9_TOTAL",
]

# ── VAE ──────────────────────────────────────────────────────────────────────
VAE_LATENT_DIM       = 16
VAE_HIDDEN_DIMS      = [128, 64]
VAE_EPOCHS           = 100
VAE_LR               = 1e-3
VAE_BATCH_SIZE       = 64
VAE_KL_ANNEAL_EPOCHS = 20   # KL weight ramps 0→1 over this many epochs

# ── CTGAN ────────────────────────────────────────────────────────────────────
CTGAN_NOISE_DIM        = 128
CTGAN_EPOCHS           = 300
CTGAN_BATCH_SIZE       = 500
CTGAN_GEN_DIM          = (256, 256)
CTGAN_DISC_DIM         = (256, 256)
CTGAN_LR               = 2e-4
CTGAN_GRADIENT_PENALTY = 10.0
CTGAN_DISC_STEPS       = 5   # discriminator updates per generator update
NUM_CLASSES            = 5   # Depression_Severity values {0,1,2,3,4}
