# PROJECT_PLAN.md — Implementation Phases

> # ⚠️ SUPERSEDED — DO NOT QUOTE NUMBERS FROM THIS FILE
> Written against the pre-pivot pipeline. Any "DCR-MIA gap", "15/15 runs", shadow-MIA
> accuracy, KS pass-rate, or absolute-DCR figure below is a retracted artifact of the old
> Axis 4. Current numbers: the Known Results table in `/CLAUDE.md`. Claim status and reasons:
> `/reference.md` §1 and §5.

This file breaks the project into clear phases with specific tasks. Work through them in order. Each phase should be complete and tested before moving to the next.

---

## Phase 1 — Setup & Data

**Goal:** Working environment with clean, processed data ready for model training.

- [ ] Create `requirements.txt` with all dependencies (torch, pandas, numpy, scikit-learn, matplotlib, seaborn)
- [ ] Create `config.py` with all hyperparameters and paths
- [ ] Write `preprocess.py`:
  - Load NHANES CSV, select mental health columns
  - Drop high-missing columns (>30%)
  - Impute: median for continuous, mode for categorical
  - Label-encode categorical columns
  - Min-max normalise continuous columns
  - Stratified train/val/test split (70/15/15)
  - Save splits as CSV to `data/processed/`
  - Repeat same pipeline for DASS-42
- [ ] Verify output: print shape and class balance of each split

---

## Phase 2 — VAE

**Goal:** Trained VAE that can encode real data and decode synthetic samples.

- [ ] Write `models/vae/model.py`:
  - `Encoder` class: FC layers → outputs (μ, log σ²)
  - Reparameterisation method
  - `Decoder` class: FC layers → reconstructed features
  - `VAE` class: combines encoder + decoder, implements `forward()`
- [ ] Write `models/vae/train.py`:
  - DataLoader setup
  - Loss function: reconstruction (MSE) + KL divergence with annealing
  - Training loop with validation loss tracking
  - Save best model checkpoint to `models/saved/vae_nhanes.pt`
  - Log train/val loss per epoch
- [ ] Sanity check: reconstruct a batch of real rows and verify they look reasonable

---

## Phase 3 — CTGAN

**Goal:** Trained CTGAN that generates realistic tabular rows conditioned on target column.

- [ ] Write `models/ctgan/model.py`:
  - `Generator` class: noise + condition → synthetic row
  - `Discriminator` class: row + condition → real/fake score
  - WGAN-GP gradient penalty function
  - Mode-specific normalisation for continuous columns
- [ ] Write `models/ctgan/train.py`:
  - Conditional sampling logic (sample condition vector per batch)
  - Discriminator trains 5 steps per 1 generator step
  - WGAN-GP loss (no sigmoid on discriminator output)
  - Save best generator checkpoint to `models/saved/ctgan_nhanes.pt`
  - Log generator and discriminator loss per epoch
- [ ] Sanity check: generate 100 rows and verify column ranges are within expected bounds

---

## Phase 4 — Generation Script

**Goal:** Single entry point to produce synthetic datasets of any size.

- [ ] Write `generate.py`:
  - CLI args: `--model` (vae/ctgan), `--dataset` (nhanes/dass), `--n` (number of rows)
  - Load saved model checkpoint
  - Generate N rows
  - Inverse-transform normalisation and encoding back to original column types
  - Save output to `data/synthetic/{model}_{dataset}_synthetic.csv`

---

## Phase 5 — Evaluation

**Goal:** Quantified evidence that synthetic data is statistically faithful, useful, and private.

- [ ] Write `evaluation/statistical.py`:
  - Per-column KS test (print p-value, flag columns that fail)
  - Correlation matrix for real vs synthetic (side-by-side heatmap)
  - Overlaid histogram for each feature
  - Save plots to `evaluation/plots/`

- [ ] Write `evaluation/utility.py`:
  - Train logistic regression and random forest on synthetic train set
  - Evaluate on real test set → TSTR accuracy, F1, AUC-ROC
  - Train same models on real train set → TRTR baseline
  - Print utility ratio: TSTR AUC / TRTR AUC
  - Target ratio ≥ 0.85

- [ ] Write `evaluation/privacy.py`:
  - DCR: for each synthetic row, find min Euclidean distance to real training rows
  - Plot DCR distribution, report mean DCR
  - MIA: label real training rows as 1, synthetic rows as 0, train binary classifier, report accuracy
  - Target: MIA accuracy ≤ 55%

---

## Phase 6 — Reporting

**Goal:** Results compiled and ready for the research paper.

- [ ] Run full pipeline on NHANES with both VAE and CTGAN
- [ ] Run full pipeline on DASS-42 with both VAE and CTGAN
- [ ] Compile all evaluation outputs into a results table:

| Model | Dataset | KS pass rate | TSTR AUC | TRTR AUC | Utility Ratio | DCR mean | MIA Acc |
|---|---|---|---|---|---|---|---|
| VAE | NHANES | | | | | | |
| CTGAN | NHANES | | | | | | |
| VAE | DASS-42 | | | | | | |
| CTGAN | DASS-42 | | | | | | |

- [ ] Save all plots and the results table for inclusion in the paper

---

## Optional Extension — Differential Privacy

If time allows, add DP-SGD to the CTGAN training loop:

- [ ] Add `--dp` flag to `models/ctgan/train.py`
- [ ] Use `opacus` library (PyTorch-compatible DP-SGD)
- [ ] Report epsilon (privacy budget) alongside evaluation results
- [ ] Compare: standard CTGAN vs DP-CTGAN on utility and privacy metrics

---

## Dependency Order

```
Phase 1 → Phase 2 → Phase 4 (VAE) → Phase 5
Phase 1 → Phase 3 → Phase 4 (CTGAN) → Phase 5
Phase 5 → Phase 6
```

Phases 2 and 3 can be developed in parallel once Phase 1 is done.