# Submission blockers

Branched from tag `audit-2026-07-22` — the verified full-pipeline audit checkpoint. **That tag never
moves.** Every item below changes the scientific or editorial state and therefore belongs on this
branch, not on the frozen one.

Ordered by what a reviewer would reject the paper for first. Items 1–5 correct existing claims;
6–7 are new experiments; 8–11 are release hygiene.

---

### 1. Human-written §II Related Work with verified DOIs — **HUMAN TASK**

§II is a stub. `reference.md` §3 holds a sourced citation base (26 net-new entries across six
clusters), but **none of it is DOI-verified** and no prose is drafted.

Do not let a model draft these citations. A project that retracted its own headline finding for being
a measurement artifact cannot afford a hallucinated reference; that would be a strictly worse problem
than the one already retracted. Every DOI gets checked by a human against the publisher record.

### 2. Report per-class utility in §VI, prominently classes 3 and 4

`evaluation/results/utility_per_class_*.csv` is written for every run and tabulated nowhere. §III-B
now states the finding inline, but the results section still gates and tabulates only the macro ratio.

State it with its control, or it overstates: for the severe class (class 4), RF/TSTR F1 was 0.000 in
all 20 generator-seed runs. The RF/TRTR control also produced class-4 F1 = 0.000 in each of the five
unique seed evaluations, showing that the zero-F1 outcome is not specific to synthetic training and is
consistent with the combination of class imbalance and the classifier's decision threshold. Class-4
AUC nevertheless separated the protocols: RF/TRTR achieved 0.775 ± 0.009, compared with 0.653 ± 0.065
for CTGAN RF/TSTR and 0.479 ± 0.059 for GaussianCopula RF/TSTR (mean ± sample SD across five seeds;
full per-model range below). The previously reported values 0.765, 0.566, and 0.391 were seed-42
values; the CTGAN and GaussianCopula values were also the minima across their five seeds. GaussianCopula
fell below 0.5 in only two of five seeds, so these results do not support a general below-chance claim.

| Protocol | Class-4 AUC (mean ± sample SD, n=5 seeds) | Range |
|---|---|---|
| RF/TRTR | 0.775 ± 0.009 | 0.765–0.788 |
| VAE RF/TSTR | 0.500 ± 0.000 | 0.500–0.500 |
| CTGAN RF/TSTR | 0.653 ± 0.065 | 0.566–0.729 |
| TVAE RF/TSTR | 0.575 ± 0.087 | 0.484–0.680 |
| GaussianCopula RF/TSTR | 0.479 ± 0.059 | 0.391–0.535 |

### 3. Say plainly that aggregate utility does not imply rare-class utility

CTGAN clears the Axis-2 gate at macro ratio 0.935 while contributing nothing on the two clinically
important classes. A clinical reviewer finds this immediately. It should be the paper's own
observation, not the reviewer's.

### 4. Correct the leakage guard in code

`build_dataset` never constructs `PHQ9_TOTAL` — it derives `PHQ9_RAW = sum(DPQ items)` directly. So:

- `LEAKAGE_COLS = ["PHQ9_TOTAL"] + DPQ_ITEMS` (`preprocess.py:88`) names a column that never exists.
- The assert at `:219` cannot fire — it is vacuous.
- `PHQ9_RAW` *is* the deterministic sum. It is dropped unconditionally at `:195` but **has no assert**.
- The nine per-item DPQ asserts at `:222` are the guard actually doing the work.

No leakage occurs. The assert net simply has a hole exactly where it matters most. Fix: drop the dead
name, add `assert "PHQ9_RAW" not in df.columns`. Documentation already says all of this; the code does
not. **Re-run the audit after this change** — it touches `preprocess.py`, which invalidates every
checkpoint (see item 10).

### 5. Resolve the unsupported `0.9999`

The pre-fix TRTR AUC has no regenerating code path anywhere in the repository. It is currently labelled
historical at all four sites. Either regenerate it from a committed path or drop it from the paper.
**Do not invent a replacement.**

### 6. Design and run a non-KDE leakage dial

The dial is the paper's strongest result and its weakest flank — see item 7. An overfit neural
generator with a capacity/early-stopping knob would give the same memorisation↔smoothing sweep without
a kernel density estimate anywhere in the generator.

### 7. Address the matched-KDE objection

DOMIAS estimates density by KDE and the dial *is* a KDE, so the gate-flip result could be an artifact
of matched model classes. The RF density-ratio variant was built as the cross-family check and
**failed**: it reads 0.51–0.54 across the entire sweep including bandwidth 0.005, where the generator
emits training rows plus N(0, 0.005²). An attack blind to near-verbatim memorisation validates
nothing. Needs a second attack family — Gen-LRA or ReMIA — or item 6.

### 8. Improve artifact acquisition and regeneration instructions

Partly done at the freeze: README now carries the verified NHANES 2017–March 2020 source for all six
XPT files and a regeneration order. Still open: expected wall-clock and hardware for a full re-run
(CTGAN at 5 critic steps is ~12 h on CPU), and what a reviewer can verify without retraining.

### 9. Verify a fresh clone reproduces the claimed scope

Measured at the freeze, `../project-audit-verify`:

| | |
|---|---|
| Source, paper, 184 result CSVs | present |
| `compileall`, `import config` | pass |
| `paper/figures/make_figures.py` | **runs; all three PNGs byte-identical to the committed ones** |
| `scripts/check_consistency.py` | **cannot run** — `FileNotFoundError` on `data/processed/nhanes_train.csv` |

Correct status: *source and compact-results complete; full recomputation requires the raw NHANES XPT
files, regenerated splits, and trained checkpoints.* Not "reproducible" without qualification. Closing
the gap needs either a checkpoint release or a documented reduced-scope verification path.

### 10. Re-run the full audit only after the scientific changes land

Items 4 and 6 change `preprocess.py` and add a generator. Checkpoints are reused silently, so after
either one: delete `models/saved/` and `data/synthetic/`, then `run_all_seeds.py`, then
`check_consistency.py`. Re-running before the changes are final just burns the CPU budget twice.

### 11. Tag a new submission candidate

New tag when the above lands. **`audit-2026-07-22` never moves** — it is the provenance record for
every number currently in the paper.
