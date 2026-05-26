# Synthetic Mental Health Data Generation

A graduate-level deep learning pipeline that generates privacy-safe synthetic tabular data from real mental health survey datasets using a Variational Autoencoder (VAE) and Conditional Tabular GAN (CTGAN).

---

## Overview

Sharing sensitive health data across institutions is legally and ethically constrained — even when the data is nominally anonymised. This project addresses that problem by learning the statistical distribution of real data and generating a synthetic twin: a dataset with no real individuals but identical statistical properties.

The pipeline covers end-to-end: data preprocessing → model training → synthetic generation → evaluation across three dimensions (statistical fidelity, downstream utility, privacy resilience).

---

## Datasets

| Dataset | Type | Samples | Target |
|---|---|---|---|
| NHANES (DPQ module) | Tabular survey | ~5,000 | Binary depression (PHQ-9 ≥ 10) |
| DASS-42 | Tabular survey | ~1,000+ | Severity category (5-class) |

---

## Models

| Model | Purpose | Key Feature |
|---|---|---|
| VAE | Continuous latent representation | KL divergence regularisation |
| CTGAN | Adversarial tabular generation | WGAN-GP + mode-specific normalisation |

---

## Evaluation

| Dimension | Method | Tool |
|---|---|---|
| Statistical fidelity | KS test + correlation matrix | `evaluation/statistical.py` |
| Downstream utility | TSTR vs TRTR (AUC-ROC) | `evaluation/utility.py` |
| Privacy | DCR + Membership Inference Attack | `evaluation/privacy.py` |

---

## Tech Stack

- Python 3.10+
- PyTorch
- scikit-learn
- pandas / numpy
- matplotlib / seaborn

---

## Quick Start

```bash
pip install -r requirements.txt
python preprocess.py --dataset nhanes
python models/vae/train.py --dataset nhanes
python generate.py --model vae --dataset nhanes --n 1000
```

See `CLAUDE.md` for full architecture details, hyperparameters, and implementation rules.