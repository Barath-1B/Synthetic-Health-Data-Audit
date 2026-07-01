# DEMO.md — Reviewer Walkthrough Cheat Sheet

A one-page script for demonstrating this project live. Everything below already
works — models are trained, data is generated, plots and metric CSVs exist.

---

## 0. Before they arrive (30 sec)

```bash
pip install -r requirements.txt        # if running on a fresh machine
```

Open **REPORT.md** in a Markdown preview (VS Code: `Ctrl+Shift+V`) so the
figures in `evaluation/plots/` render inline. That document *is* the deliverable.

---

## 1. The 60-second pitch

> "Mental-health survey rows can re-identify people even after stripping names,
> because rare attribute combinations are unique. I train generative models that
> learn the *distribution* of NHANES data and emit new rows that follow the same
> statistics but correspond to no single real person. I then **prove** the privacy
> with two attacks, not just claim it."

Three axes: **Fidelity** (do distributions match?), **Utility** (is it useful for
ML?), **Privacy** (can you link a row back to a person?).

---

## 2. The headline result — open REPORT.md → Table VI

| Metric | Target | VAE | CTGAN |
|---|---|---|---|
| KS p-value per column | > 0.05 | ✓ 34/34 | ✓ 34/34 |
| TSTR/TRTR AUC ratio | ≥ 0.85 | ✓ 0.93–0.99 | ✓ 0.98–0.999 |
| DCR mean | > 0.1 | ✓ 0.679 | ✓ 0.712 |
| MIA accuracy | ≤ 0.55 | ✗ 0.97 | ✓ 0.538 |

**The story:** both models tie on fidelity and utility — the **privacy axis breaks
the tie.** The VAE fails the membership-inference attack (0.97) despite a healthy
DCR, which proves *DCR alone is not a sufficient privacy certificate.* CTGAN is the
recommended generator.

---

## 3. Live demo — prove it actually runs

Run these in front of the reviewer (each takes seconds — models are pre-trained):

```bash
# Generate 1000 fresh synthetic rows from the CTGAN
python generate.py --model ctgan --n 1000
#  → data/synthetic/ctgan_nhanes_synthetic.csv  (shape 1000 x 35)

# Re-run a privacy audit so the numbers aren't hand-written
python evaluation/privacy.py --model ctgan
#  → prints DCR mean/min/max and MIA accuracy, writes evaluation/plots/privacy_ctgan.csv
```

To rebuild *everything* from scratch (if asked — VAE ~minutes, CTGAN longer):

```bash
python preprocess.py
python models/vae/train.py
python models/ctgan/train.py
python generate.py --model vae   --n 1000
python generate.py --model ctgan --n 1000
python evaluation/statistical.py --model ctgan
python evaluation/utility.py     --model ctgan
python evaluation/privacy.py     --model ctgan
```

---

## 4. Figures to show (in `evaluation/plots/`)

| File | What it proves |
|---|---|
| `correlation.png` | Joint structure preserved (real vs synthetic heatmaps side by side) |
| `hist_PHQ9_TOTAL.png` | A key marginal matches the real distribution |
| `hist_Depression_Severity.png` | Class imbalance is reproduced, not flattened |
| `dcr_vae.png` vs `dcr_ctgan.png` | No synthetic row sits on top of a real one |

---

## 5. Likely reviewer questions — have these ready

**Q: How do you guarantee a synthetic row isn't a copy of a real person?**
The models sample from a learned distribution (VAE decodes random latent `z ~ N(0,I)`;
CTGAN maps random noise through the generator). They never look up a stored row.
Continuous columns are then rank-mapped to the real *sorted* marginal, and the
correlated blocks (PHQ-9 items, dietary) are resampled from real rows that share the
same PHQ-9/age bucket — **but each column-block comes from a different real
individual, so the final row is a recombination that matches no one person.** DCR
confirms no row coincides with a real record; MIA confirms a classifier can't tell
real from synthetic. (See section below for the honest nuance here.)

**Q: Why does the VAE fail MIA but the CTGAN passes?**
The VAE has an explicit reconstruction objective, so its output cloud carries a
learnable "synthetic-ness" signature. The CTGAN is trained adversarially until a
critic can't separate real from fake — exactly the MIA condition.

**Q: Is this differentially private?**
No — the privacy here is *empirical* (attack-based), not a formal (ε,δ) guarantee.
A DP-SGD flag was scoped as future work. I'm explicit about this in Limitations.

**Q: Why only NHANES, not DASS-42?**
Scoped but not built — listed honestly in Limitations. The pipeline is dataset-driven
via `config.py`, so DASS-42 is a config + preprocessing extension.

**Q: AUC is ~0.99 but macro-F1 is low — why?**
Class imbalance. Both models reproduce the dominant low-severity bands (high AUC),
but synthetic-trained classifiers struggle to recall rare high-severity classes.
CTGAN's conditional generation narrows this gap.

---

## 6. Repo map (if they want to read code)

| File | Role |
|---|---|
| `preprocess.py` | Load `.xpt` → merge on SEQN → clean → encode → scale → split |
| `models/vae/model.py`, `train.py` | VAE architecture + KL-annealed training |
| `models/ctgan/model.py`, `train.py` | WGAN-GP CTGAN, 5 critic steps/gen step |
| `generate.py` | Load checkpoint → sample N rows → inverse-transform |
| `evaluation/statistical.py` | KS tests, correlation heatmaps, histograms |
| `evaluation/utility.py` | TSTR vs TRTR (LogReg + RandomForest) |
| `evaluation/privacy.py` | DCR + membership-inference attack |
| `config.py` | All paths + hyperparameters (single source of truth) |

---

## 7. One honest caveat to volunteer (it builds credibility)

The post-processing in `generate.py` resamples the PHQ-9 item block and dietary
columns from real training rows (conditioned on PHQ-9 score + age bucket) to fix a
consistency artifact the MIA otherwise exploited. This means parts of each synthetic
row are *real values, recombined* — no row maps to one person, but the columns aren't
all model-invented. If asked about strict privacy, this is the place to point to
DP-SGD as the principled fix. Volunteering this shows you understand the trade-off.
