<div align="center">

# 🧬 Synthetic Mental-Health Data Generation

### Privacy-safe synthetic tabular data from real mental-health surveys — via VAE & CTGAN

*Learn the statistical distribution of sensitive survey data, then generate a synthetic twin: no real individuals, near-identical statistics.*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

</div>

---

## 📖 Overview

Sharing sensitive health data across institutions is legally and ethically constrained — even when the data is nominally anonymised. This project addresses that by learning the statistical distribution of real data and generating a **synthetic twin**: a dataset with no real individuals but matching statistical properties.

The pipeline runs end-to-end: **data preprocessing → model training → synthetic generation → evaluation** across three dimensions — statistical fidelity, downstream utility, and privacy resilience.

---

## 📊 Datasets

| Dataset | Type | Samples | Target |
|---|---|---|---|
| NHANES (DPQ module) | Tabular survey | ~5,000 | Binary depression (PHQ-9 ≥ 10) |
| DASS-42 | Tabular survey | ~1,000+ | Severity category (5-class) |

---

## 🤖 Models

| Model | Purpose | Key feature |
|---|---|---|
| **VAE** | Continuous latent representation | KL-divergence regularisation |
| **CTGAN** | Adversarial tabular generation | WGAN-GP + mode-specific normalisation |

---

## 🔬 Evaluation

The synthetic data is judged on three axes — it has to be statistically faithful, *useful* for downstream ML, **and** resistant to privacy attacks:

| Dimension | Method | Module |
|---|---|---|
| 📈 Statistical fidelity | KS test + correlation matrix | `evaluation/statistical.py` |
| 🎯 Downstream utility | TSTR vs TRTR (AUC-ROC) | `evaluation/utility.py` |
| 🔒 Privacy | DCR + Membership Inference Attack | `evaluation/privacy.py` |

> **TSTR** — *Train on Synthetic, Test on Real* — is the acid test: a model trained purely on the synthetic data should still perform on real data. **DCR** (Distance to Closest Record) and a Membership Inference Attack check the synthetic set isn't just memorising real rows.

---

## 🛠️ Tech stack

Python 3.10+ · PyTorch · scikit-learn · pandas / numpy · matplotlib / seaborn

---

## 🗂️ Project structure

```
.
├── preprocess.py            # dataset cleaning + encoding
├── generate.py              # sample synthetic rows from a trained model
├── models/
│   ├── vae/train.py         # Variational Autoencoder
│   └── ctgan/train.py       # Conditional Tabular GAN
├── evaluation/
│   ├── statistical.py       # KS test + correlation fidelity
│   ├── utility.py           # TSTR vs TRTR
│   └── privacy.py           # DCR + membership inference
└── requirements.txt
```

---

## ⚡ Quick start

```bash
pip install -r requirements.txt

python preprocess.py --dataset nhanes                 # clean + encode
python models/vae/train.py --dataset nhanes           # train the VAE
python generate.py --model vae --dataset nhanes --n 1000   # sample 1000 synthetic rows
```

Swap `--model vae` for `--model ctgan` to run the GAN path.

---

## ⚠️ Disclaimer

Research / coursework project. The synthetic data is for methodology study — not a substitute for real clinical data, and not for making decisions about individuals.

<div align="center">

**Built by [Barath M](https://github.com/Barath-1B)** · Chennai, India

</div>
