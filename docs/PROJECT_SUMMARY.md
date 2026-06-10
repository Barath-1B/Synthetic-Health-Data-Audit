# PROJECT SUMMARY

## What This Project Does

This project builds a pipeline that takes real mental health survey data (NHANES) and produces **privacy-safe synthetic tabular data** using two generative models: a Variational Autoencoder (VAE) and a Conditional Tabular GAN (CTGAN). The synthetic data is designed to statistically mirror the original while having no 1-to-1 correspondence with any real individual. The pipeline covers the full ML research cycle: raw data ingestion → preprocessing → model training → synthetic generation → three-dimensional evaluation (statistical fidelity, downstream utility, privacy resilience).

---

## End-to-End Pipeline

```
Raw NHANES XPT files (6 modules)
        ↓  preprocess.py
Cleaned, normalized CSV + splits + scaler
        ↓  models/vae/train.py  OR  models/ctgan/train.py
Trained model checkpoint (.pt)
        ↓  generate.py
Synthetic CSV (N rows)
        ↓  evaluation/*.py
KS test results, TSTR/TRTR utility ratio, DCR + MIA privacy scores
```

---

## Build Order (Logical Sequence)

1. **Configuration** — `config.py` written first to anchor all paths and hyperparameters.
2. **Preprocessing** — `preprocess.py` developed against raw NHANES XPT files; five critical bugs discovered and fixed (see `DATA_PREP.md`). Outputs `nhanes_clean.csv`, `scaler.pkl`, `splits.pkl`.
3. **VAE model + training** — `models/vae/model.py` and `train.py` implemented with KL annealing.
4. **CTGAN model + training** — `models/ctgan/model.py` and `train.py` implemented with WGAN-GP.
5. **Generation script** — `generate.py` written to load either checkpoint and produce synthetic CSV.
6. **Evaluation suite** — `evaluation/statistical.py`, `utility.py`, `privacy.py` implemented as standalone scripts.
7. **Documentation** — `CLAUDE.md`, `README.md`, `PROJECT_PLAN.md`, `DATA_PREP.md` written throughout.

---

## Every File

### Documentation

| File | What it does |
|------|--------------|
| `CLAUDE.md` | Master architecture reference: datasets, preprocessing steps, model designs, evaluation framework, hyperparameters, run commands, success metrics, known pitfalls. The authoritative spec for the whole project. |
| `README.md` | High-level overview and quick-start commands for a new reader. |
| `PROJECT_PLAN.md` | Phased task breakdown (Phase 1–6) used to track implementation progress. |
| `DATA_PREP.md` | Documents five critical bugs found during NHANES preprocessing and the exact fix applied for each. Essential reading before touching `preprocess.py`. |

### Configuration and Dependencies

| File | What it does |
|------|--------------|
| `config.py` | Single source of truth for every hyperparameter and file path. VAE settings (latent dim 16, hidden dims [128,64], 100 epochs, lr 1e-3), CTGAN settings (noise dim 128, gen/disc dims (256,256), 300 epochs, lr 2e-4, GP lambda 10.0), data split ratios (70/15/15), all directory paths. Nothing is hardcoded elsewhere. |
| `requirements.txt` | `torch>=2.0`, `numpy>=1.24`, `pandas>=2.0`, `scikit-learn>=1.3`, `matplotlib>=3.7`, `seaborn>=0.12`, `scipy>=1.11`. No experimental or custom packages. |

### Data Pipeline

| File | What it does |
|------|--------------|
| `preprocess.py` | Loads six NHANES XPT modules via `pd.read_sas()`, merges on SEQN with DPQ as the base (PHQ-9 completers only), applies all five bug fixes, imputes missing values (median/mode), min-max normalises continuous columns to [0,1], derives `Depression_Severity` (0–4) from raw PHQ-9 sum, performs stratified 70/15/15 split, validates with seven assertions, saves `nhanes_clean.csv` + `scaler.pkl` + `splits.pkl`. |

### Raw Data (`data/raw/`)

| File | Size | Content |
|------|------|---------|
| `P_DPQ.xpt` | 791 KB | PHQ-9 depression screener (9 items) |
| `P_DEMO.xpt` | 3.6 MB | Demographics (age, gender, race, education, income) |
| `P_MCQ.xpt` | 7.6 MB | Medical conditions (heart disease, stroke, cancer, etc.) |
| `P_DR1TOT.xpt` | 19.2 MB | Dietary intake (calories, sugar, caffeine) |
| `P_ALQ.xpt` | 719 KB | Alcohol use (frequency, quantity, binge) |
| `P_SLQ.xpt` | 777 KB | Sleep (weekday/weekend hours, trouble sleeping) |

All six files are NHANES Cycle P (2017–2020 pre-pandemic), publicly available from CDC, no IRB required.

### Processed Data (`data/processed/`)

| File | What it contains |
|------|-----------------|
| `nhanes_clean.csv` | 8,965 rows × 35 columns (34 features + `Depression_Severity` target). All values numeric, no NaNs, continuous columns in [0,1]. |
| `scaler.pkl` | `MinMaxScaler` fitted on the training split for continuous columns. Used in `generate.py` to inverse-transform model outputs back to original scale. |
| `splits.pkl` | Dictionary of row-index arrays: `{"train": [...], "val": [...], "test": [...]}`. Stratified by `Depression_Severity`. Sizes: 5,275 / 1,345 / 1,345. |

### VAE (`models/vae/`)

| File | What it does |
|------|--------------|
| `model.py` | Defines `Encoder` (FC → μ and log σ²), `Decoder` (FC → sigmoid output), `VAE` (ties them with reparameterisation trick z = μ + σ·ε), and `vae_loss` (MSE reconstruction + KL divergence with optional annealing weight). Latent dim 16, hidden dims [128, 64]. |
| `train.py` | Loads training split, runs 100 epochs with Adam (lr 1e-3), ramps KL weight from 0 to 1 over the first 20 epochs, tracks best validation loss, saves checkpoint to `models/saved/vae_nhanes.pt`. |

### CTGAN (`models/ctgan/`)

| File | What it does |
|------|--------------|
| `model.py` | Defines `Generator` (noise + one-hot condition → FC + BatchNorm + ReLU → sigmoid), `Discriminator` (data + condition → FC + LeakyReLU, no BatchNorm, no sigmoid — WGAN requirement), and `gradient_penalty` (interpolated sample penalty, λ=10.0). |
| `train.py` | Loads training split with class labels, runs 300 epochs with Adam (lr 2e-4, betas 0.5/0.9), executes 5 discriminator steps per 1 generator step (WGAN-GP standard), saves checkpoint to `models/saved/ctgan_nhanes.pt`. |

### Generation Script

| File | What it does |
|------|--------------|
| `generate.py` | CLI entry point. Loads a saved checkpoint (`--model vae` or `--model ctgan`), samples N rows (`--n`, default 1000) from the latent space (VAE) or from conditioned noise proportional to the real class distribution (CTGAN), inverse-transforms via `scaler.pkl`, rounds integer columns, saves to `data/synthetic/`. |

### Evaluation Suite (`evaluation/`)

| File | What it does |
|------|--------------|
| `statistical.py` | Per-column two-sample KS test (real vs synthetic, target p ≥ 0.05); Pearson correlation heatmaps side-by-side; overlaid density histograms per feature. Outputs CSV + PNG files to `evaluation/plots/`. |
| `utility.py` | TRTR (train real, test real) vs TSTR (train synthetic, test real) comparison using Logistic Regression and Random Forest. Reports accuracy, F1-macro, AUC-ROC, and the utility ratio TSTR_AUC / TRTR_AUC (target ≥ 0.85). |
| `privacy.py` | DCR (Distance to Closest Record): mean minimum Euclidean distance from each synthetic row to any real training row, target mean > 0.1. MIA (Membership Inference Attack): 5-fold cross-validated Random Forest distinguishing real training rows from synthetic rows, target accuracy ≤ 55%. |

---

## Architecture Decisions

**Fully connected networks only.** This is tabular data, not images. No convolutions anywhere.

**WGAN-GP for CTGAN.** Gradient penalty (λ=10) is more stable than Wasserstein weight clipping for mixed tabular data. The discriminator has no BatchNorm (required by WGAN-GP to preserve gradient flow through interpolated samples).

**Conditional generation.** CTGAN conditions on `Depression_Severity` (5-class one-hot) so the generator can reproduce the class imbalance present in the real data rather than collapsing to the majority class.

**KL annealing for VAE.** KL weight ramps 0→1 over 20 epochs. Without this, the KL term dominates early training and the encoder collapses to the prior before learning useful structure.

**DPQ-first merge base.** The merge in `preprocess.py` starts from the depression screener module (PHQ-9 completers only). Merging the other direction produced phantom zero scores for non-respondents — one of the five bugs found during development.

**Min-max normalisation over standardisation.** Chosen because both model decoders use sigmoid activations with output range [0,1], making [0,1]-normalised targets directly compatible.

**Separate models per dataset.** NHANES and DASS-42 use different schemas and class structures; the architecture mandates training them independently with no shared weights.

---

## Five Preprocessing Bugs Fixed

All five are documented in detail in [DATA_PREP.md](DATA_PREP.md) and fixed in `preprocess.py`.

1. **DPQ value 3 capped to 2** — raw PHQ-9 items have four response options (0–3); an early remap truncated 3→2, losing severity information. Fix: preserve 0–3 as-is.
2. **MCQ binary coding** — NHANES codes "Yes" as 1 and "No" as 2 (not 0). Fix: remap 2→0 before training.
3. **Phantom zero scores from wrong merge base** — merging demographics first brought in all ~10,000 participants; non-DPQ respondents got imputed PHQ-9 zeros. Fix: start merge from P_DPQ (the smallest, most restrictive module).
4. **Proxy / refusal values not replaced** — values like 7, 9, 77, 99 encode "refused" or "don't know"; leaving them in distorts distributions. Fix: replace with NaN before imputation.
5. **SEQN left in features** — participant ID included as a training feature, leaking identity information. Fix: drop SEQN after merging.

---

## Known Issues and Limitations

**DASS-42 not implemented.** The project plan and `CLAUDE.md` describe DASS-42 support; it was never built. Only NHANES is supported.

**DP-SGD not implemented.** Differential privacy via `opacus` is mentioned in the plan as an optional `--dp` flag for CTGAN training. It was not added.

**No model checkpoints in the repository.** `models/saved/` is empty — training must be run to produce `.pt` files before generation or evaluation can proceed.

**`data/synthetic/` is empty.** Generation must be run after training.

**`evaluation/plots/` is empty.** Evaluation must be run after generation.

**No unit tests.** The `preprocess.py` assertions act as integration checks at runtime, but there is no `tests/` directory or test runner.

**Single dataset only.** All code is wired for NHANES Cycle P. A different NHANES cycle or a different dataset would require changes to column name lists in `config.py` and `preprocess.py`.

---

## Commands to Run the Project

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess raw NHANES data
python preprocess.py
# Produces: data/processed/nhanes_clean.csv, scaler.pkl, splits.pkl

# 3. Train VAE (≈8–10 min on GPU, longer on CPU)
python models/vae/train.py
# Produces: models/saved/vae_nhanes.pt

# 4. Train CTGAN (≈50 min on GPU, longer on CPU)
python models/ctgan/train.py
# Produces: models/saved/ctgan_nhanes.pt

# 5. Generate 1000 synthetic rows
python generate.py --model vae --n 1000
python generate.py --model ctgan --n 1000
# Produces: data/synthetic/vae_nhanes_synthetic.csv
#           data/synthetic/ctgan_nhanes_synthetic.csv

# 6. Evaluate — run for each model
python evaluation/statistical.py --model vae
python evaluation/utility.py    --model vae
python evaluation/privacy.py    --model vae

python evaluation/statistical.py --model ctgan
python evaluation/utility.py    --model ctgan
python evaluation/privacy.py    --model ctgan
# Produces: evaluation/plots/ (CSV tables + PNG figures)
```

---

## Success Targets

| Metric | Target | Evaluated by |
|--------|--------|--------------|
| KS test p-value (per column) | > 0.05 | `evaluation/statistical.py` |
| TSTR / TRTR AUC ratio | ≥ 0.85 | `evaluation/utility.py` |
| DCR mean | > 0.1 | `evaluation/privacy.py` |
| MIA classifier accuracy | ≤ 55% | `evaluation/privacy.py` |
