# DATA_PREP.md — NHANES Preprocessing Fix

Read this fully before writing any code. This file documents exactly what is wrong with the existing cleaned CSVs and how to fix them by rebuilding from the raw XPT files.

---

## Raw Files Location

All raw XPT files are in `data/raw/`:
```
P_ALQ.xpt       ← Alcohol use
P_DEMO.xpt      ← Demographics
P_DPQ.xpt       ← Depression screener (PHQ-9)
P_DRITTOT.xpt   ← Dietary intake
P_MCQ.xpt       ← Medical conditions
P_SLQ.xpt       ← Sleep
```

Load XPT files with:
```python
import pandas as pd
df = pd.read_sas("data/raw/P_DPQ.xpt", format="xport", encoding="utf-8")
```

---

## What Is Wrong With the Existing Cleaned CSVs

### Bug 1 — PHQ-9 responses capped at 2 (CRITICAL)
- PHQ-9 scale is 0–3: 0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day
- The existing DPQ_cleaned.csv and master_mental_health_dataset.csv show max value of 2 for all DPQ items
- The value 3 ("Nearly every day") exists in the raw XPT — hundreds of rows per column — but was silently lost
- This deflates all PHQ-9 scores and mislabels high-severity depression cases
- **Fix:** Preserve value 3. Do not remap or drop it.

### Bug 2 — Medical condition columns encoded backwards (CRITICAL)
- NHANES MCQ coding: 1=Yes, 2=No, 9=Don't know, 7=Refused
- The existing master dataset kept 1 and 2 as-is and filled NaN with 0
- This creates three categories: 0=missing/not-asked, 1=Yes, 2=No — which is wrong for a binary column
- **Fix:** Remap explicitly: 1→1, 2→0, 7→0, 9→0, NaN→0

### Bug 3 — 6,595 phantom rows with fake zero PHQ-9 scores (CRITICAL)
- The master dataset has 15,560 rows but only 8,965 people completed the PHQ-9
- The extra 6,595 rows are participants who never took the depression questionnaire
- Their PHQ9_TOTAL was set to 0 and Depression_Severity set to 0 — labelling them "not depressed"
- This is wrong. A blank questionnaire is not a score of zero
- **Fix:** Build the merged dataset starting from DPQ (PHQ-9 completers) as the base. Use left joins from DPQ outward. Final dataset should have ~8,965 rows.

### Bug 4 — Proxy value `5.397605e-79` used instead of NaN
- NHANES XPT files use this specific float as a null placeholder in several columns
- It appears in: Age, Income_Ratio, Sleep_Hours_Weekday, Sleep_Hours_Weekend, Total_Caffeine, PHQ9_TOTAL
- For Age, Income_Ratio, Sleep columns: proxy = truly missing → replace with NaN, impute with median
- For Total_Caffeine: proxy = valid zero (person drinks no caffeine) → replace with 0.0
- **Fix:** Detect with `value < 1e-70 AND value > 0` to distinguish from true zeros

### Bug 5 — SEQN (participant ID) included in model input
- SEQN is a row identifier, not a feature
- **Fix:** Drop before saving final dataset

---

## Correct NHANES Variable Mappings

### DPQ (Depression Screener — PHQ-9)
| Raw column | Rename to | Values |
|---|---|---|
| DPQ010 | Little_Interest | 0=Not at all, 1=Several days, 2=More than half, 3=Nearly every day, 7/9=NaN→0 |
| DPQ020 | Feeling_Down | same |
| DPQ030 | Trouble_Sleeping_DPQ | same |
| DPQ040 | Feeling_Tired | same |
| DPQ050 | Poor_Appetite | same |
| DPQ060 | Feeling_Bad_About_Self | same |
| DPQ070 | Trouble_Concentrating | same |
| DPQ080 | Moving_ Slowly | same |
| DPQ090 | Suicidal_Thoughts | same |

Compute after cleaning:
```python
PHQ9_RAW = sum of all 9 DPQ items  # range 0–27
PHQ9_TOTAL = PHQ9_RAW / 27.0       # normalised to [0, 1]
```

Depression severity from PHQ9_RAW:
```python
0 → None       (0–4)
1 → Mild       (5–9)
2 → Moderate   (10–14)
3 → Mod-Severe (15–19)
4 → Severe     (20–27)
```

### DEMO (Demographics)
| Raw column | Rename to | Notes |
|---|---|---|
| RIAGENDR | Gender | 1=Male→0, 2=Female→1 |
| RIDAGEYR | Age | continuous, normalise to [0,1] |
| RIDRETH1 | Race | 1–5, keep as ordinal int |
| RIDRETH3 | Race_Ext | 1–7, keep as ordinal int |
| DMDEDUC2 | Education | 1–5 ordinal; 7/9→NaN→mode |
| DMDMARTZ | Marital_Status | 1–5 ordinal; 77/99→NaN→mode |
| INDFMPIR | Income_Ratio | continuous 0–5, normalise; proxy→median |

### MCQ (Medical Conditions)
| Raw column | Rename to | Remap |
|---|---|---|
| MCQ160A | Arthritis | 1→1, 2→0, 7→0, 9→0, NaN→0 |
| MCQ160B | Congestive_Heart_Failure | same |
| MCQ160C | Coronary_Heart_Disease | same |
| MCQ160E | Heart_Attack | same |
| MCQ160F | Stroke | same |
| MCQ220  | Cancer | same |

### SLQ (Sleep)
| Raw column | Rename to | Notes |
|---|---|---|
| SLD012 | Sleep_Hours_Weekday | continuous; proxy→median; normalise |
| SLD013 | Sleep_Hours_Weekend | continuous; proxy→median; normalise |
| SLQ050 | Trouble_Sleeping_Doc | 1→1, 2→0, 7→0, 9→0, NaN→0 |

### ALQ (Alcohol)
| Raw column | Rename to | Notes |
|---|---|---|
| ALQ111 | Ever_Had_Drink | 1→1, 2→0, 7→0, 9→0, NaN→0 |
| ALQ121 | Drinking_Freq | ordinal 0–10; 777/999→NaN→0; normalise |
| ALQ130 | Avg_Drinks_Per_Day | continuous; 777/999→NaN→median; normalise |
| ALQ142 | Binge_Drinking_Freq | ordinal 0–6; 777/999→NaN→0; normalise |

### DR1TOT (Dietary Intake)
| Raw column | Rename to | Notes |
|---|---|---|
| DR1TKCAL | Total_Calories | continuous; normalise |
| DR1TSUGR | Total_Sugar | continuous; normalise |
| DR1TCAFF | Total_Caffeine | continuous; proxy→0.0 (valid zero); normalise |

---

## Output Requirements

Save to `data/processed/nhanes_clean.csv` with:
- ~8,965 rows (only PHQ-9 completers)
- 34 columns (no SEQN, no PHQ9_RAW)
- Zero NaN values
- All columns numeric
- All continuous columns normalised to [0, 1]
- DPQ items: int, values in {0, 1, 2, 3}
- Medical condition columns: int, values in {0, 1}
- Target column: `Depression_Severity` int, values in {0, 1, 2, 3, 4}

Verify with these assertions before saving:
```python
assert df.isnull().sum().sum() == 0
assert df.shape[0] > 8000
assert df['Depression_Severity'].nunique() == 5
assert df[dpq_items].max().max() == 3
assert set(df['Arthritis'].unique()).issubset({0, 1})
assert all(df.dtypes != object)
for c in continuous_cols:
    assert df[c].min() >= 0.0
    assert df[c].max() <= 1.0
```

---

## The Script

This logic lives in `preprocess.py` at the project root (run with
`python preprocess.py`). It:
1. Loads all 6 XPT files from `data/raw/`
2. Applies all column remappings above
3. Merges with DPQ as base (left join all others onto DPQ on SEQN)
4. Computes PHQ9_TOTAL and Depression_Severity
5. Drops SEQN and PHQ9_RAW
6. Runs all assertions
7. Saves to `data/processed/nhanes_clean.csv`
8. Prints a summary: shape, class distribution, NaN count, column ranges

Use Python 3.10+, pandas, numpy, scikit-learn MinMaxScaler. No Jupyter notebooks.