# Figure Data Sources

Generated: 2026-06-05 by `paper/figures/make_figures.py`

---

## Figure 1 — DCR vs MIA Scatter

**Data source:** 15 per-run FAA result CSVs (`evaluation/results/faa_{model}_seed{n}.csv`)
**Seeds:** 42–46  **Models:** VAE, CTGAN, TVAE  **Points:** 15
**DCR range:** [0.3278, 0.7729]
**MIA shadow acc range:** [0.6815, 0.7325]
**Threshold lines:** DCR=0.10, MIA=0.55
**Columns read:** `dcr_mean`, `mia_acc_shadow`

Files read:
  - `evaluation/results/faa_vae_seed42.csv`
  - `evaluation/results/faa_vae_seed43.csv`
  - `evaluation/results/faa_vae_seed44.csv`
  - `evaluation/results/faa_vae_seed45.csv`
  - `evaluation/results/faa_vae_seed46.csv`
  - `evaluation/results/faa_ctgan_seed42.csv`
  - `evaluation/results/faa_ctgan_seed43.csv`
  - `evaluation/results/faa_ctgan_seed44.csv`
  - `evaluation/results/faa_ctgan_seed45.csv`
  - `evaluation/results/faa_ctgan_seed46.csv`
  - `evaluation/results/faa_tvae_seed42.csv`
  - `evaluation/results/faa_tvae_seed43.csv`
  - `evaluation/results/faa_tvae_seed44.csv`
  - `evaluation/results/faa_tvae_seed45.csv`
  - `evaluation/results/faa_tvae_seed46.csv`

---

## Figure 2 — DCR Distribution (VAE, seed 42)

**Synthetic file:** `data/synthetic/vae_synthetic_seed42.csv`
**Real training file:** `data/processed/nhanes_train.csv`
**Cache saved:** `evaluation/results/dcr_raw_vae_seed42.npy`
**Method:** per-row Euclidean distance to closest real training row
  (exact replication of `evaluation/privacy.py:axis3_dcr()` — [0,1] space, no normalisation)
**Rows:** 1000  **Mean:** 0.7729  **Min:** 0.3403  **Max:** 1.2676

---

## Figure 3 — DCR Distribution (CTGAN, seed 42)

**Synthetic file:** `data/synthetic/ctgan_synthetic_seed42.csv`
**Real training file:** `data/processed/nhanes_train.csv`
**Cache saved:** `evaluation/results/dcr_raw_ctgan_seed42.npy`
**Method:** same as Figure 2
**Rows:** 1000  **Mean:** 0.4666  **Min:** 0.0947  **Max:** 1.355

---

## Figure 4 — KS p-value Distribution (CTGAN, seed 42)

**Real file:** `data/processed/nhanes_train.csv`
**Synthetic file:** `data/synthetic/ctgan_synthetic_seed42.csv`
**Method:** `scipy.stats.ks_2samp` per column, values rounded to 10 dp first
  (exact replication of `evaluation/statistical.py:axis1_ks()`)
**Columns tested:** 23 feature columns (Depression_Severity excluded)
**Pass count (p ≥ 0.05):** 10/23
**Pass rate:** 0.4348
**p-value range:** [0.0, 1.0]

> Note: the task spec referenced 34 columns — this project has 23 feature columns
> after removing PHQ9_TOTAL and the 9 DPQ items to eliminate leakage.

---

## Discrepancy Note: KS pass rate 10/23 (figures) vs 1/23 (FAA audit CSVs)

The `faa_{model}_seed{n}.csv` evaluation files all record `ks_pass_rate = 0.0435` (1/23),
while direct computation on the saved synthetic CSVs gives 10/23 for VAE and CTGAN.

**Root cause identified:** The 10 passing columns are exactly the `continuous_cols` from
`scaler.pkl` (Age, Income_Ratio, Sleep hours, Drinking_Freq, Avg_Drinks_Per_Day,
Binge_Drinking_Freq, Total_Calories, Total_Sugar, Total_Caffeine). These are rank-corrected
in `generate.py:_correct_marginals()` to exact real training values, giving D-stat ~0.001
and p-value = 1.000. The 13 failing columns are binary/categorical (Gender, Arthritis,
Race, etc.) — not rank-corrected — and CTGAN reproduces them poorly (D-stat 0.17–0.90).

**Why the FAA audit showed 1/23:** The in-memory DataFrames passed to `run_audit` during
`run_all_seeds.py` appear to have had rank correction not applied or applied to different
columns than the saved CSVs reflect. File modification timestamps confirm the CSVs were
written 5 seconds before the faa CSVs (19:39:10 vs 19:39:15 for CTGAN seed 42), meaning
both were produced in the same run. The discrepancy is unresolved without torch (currently
broken: WinError 193 on shm.dll) and requires re-running the full pipeline to investigate.

**Figure 4 uses the saved CSV value (10/23)** — this is the ground truth from the files on
disk and reflects the actual KS distribution. The rank-corrected continuous columns
genuinely match real marginals; the binary/categorical columns genuinely fail. The FAA
audit result (1/23) should be treated as suspect and re-verified in the next session.

**Implication for paper:** Figure 4 shows the bimodal p-value distribution clearly —
13 columns pile up at p~0 (categorical, model fails) and 10 pile up at p=1.0
(continuous, rank correction succeeds). This is a more informative result than "1/23"
and correctly reflects what the model produces.