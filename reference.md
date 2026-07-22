# reference.md — project decision ledger

**This file records *decisions and their provenance*. It does not restate conventions.**

[CLAUDE.md](CLAUDE.md) is authoritative for **what** the conventions, architecture, commands, and
current numbers are. This file is authoritative for **why** a choice was made, **when**, and **what it
invalidated**. If the two ever disagree about a fact, CLAUDE.md wins and this file gets a correcting
entry — never the reverse.

Read §1 before quoting any result from this repository. Several documents still on disk assert claims
that have been retracted (§5).

## Contents

- [§1 Status of claims](#1-status-of-claims)
- [§2 Decision log](#2-decision-log)
- [§3 External positioning](#3-external-positioning)
- [§4 Invariants](#4-invariants)
- [§5 Superseded documents](#5-superseded-documents)
- [How to add an entry](#how-to-add-an-entry)

---

## §1 Status of claims

| Claim | Status | Evidence / superseded by |
|---|---|---|
| **"DCR passes but MIA fails in 15/15 runs"** — the original headline | **RETRACTED** | Measurement artifact of a broken Axis 4: a real-vs-synthetic distinguishability classifier with a ~0.75 accuracy ceiling, mislabelled "shadow MIA". A calibrated DOMIAS attack puts every generator at AUC ≈ 0.50. See [§2 2026-07-01](#2026-07-01--axis-4-replaced-with-a-validated-domias-attack) and `docs/section 2/06_PAPER_REFRAME.md`. |
| **"DCR rewards infidelity"** — holdout-calibrated `dcr_ratio` is negatively associated with marginal fidelity | **ACTIVE, evidence currently weak** | 4 models × 5 seeds: GaussianCopula has the best `dcr_ratio` (1.676) with mediocre fidelity (0.834); CTGAN/TVAE (highest fidelity) sit at or below the 1.0 gate. Support is a rank association only — no run in the matrix has a firing MIA, so DCR has never been observed to *miss* real leakage here. Strengthening this is the purpose of the leakage dial ([§2 2026-07-21](#2026-07-21--power-fix-low-fpr-reporting-and-a-leakage-dial)). |
| **"DOMIAS AUC ≈ 0.50 for all four generators"** | **ACTIVE — power confound excluded 2026-07-21** | `evaluation/results/mia_power.csv`: CTGAN seed 42 audited at release sizes 500 / 1000 / 2500 / 6275 gives AUC 0.513 / 0.512 / 0.511 / **0.508** and TPR@FPR=0.01 of 0.007 / 0.011 / 0.016 / 0.013 — flat across a 12.5× range, so the null is not an artifact of too few released records. Over the same sweep C2ST climbs 0.954 → 0.992, which is independent support for the distinguishability-vs-membership claim below. |
| **"The audit catches a generator that does leak"** | **ACTIVE** | Leakage dial at bandwidth 0.005 (near-verbatim memorisation): `dcr_ratio` 0.034 → Axis 3 FAILS, DOMIAS AUC 0.755 with TPR@FPR=0.01 0.097 → Axis 4 FAILS. Both gates fire at the memorisation end, so the null in the main matrix is a property of those generators, not of a blind audit. |
| **"The holdout-calibrated DCR gate passes while a validated MIA detects membership"** | **ACTIVE — new 2026-07-21, the paper's strongest result** | `evaluation/results/leakage_dial.csv`, 7 bandwidths × 5 seeds. At bw = 0.15: `dcr_ratio` 1.054 (Axis 3 PASS) with DOMIAS AUC 0.601 and TPR@FPR=0.01 0.045 (Axis 4 FAIL). At bw = 0.20: `dcr_ratio` 1.348 (passes comfortably) with AUC 0.572. Holdout calibration is therefore **not sufficient** to rule out membership leakage on this data — the DCR Delusion reproduced in a clinical survey domain by our own audit. **Caveat below.** |
| **"The RF density-ratio variant confirms the dial result across model families"** | **NOT SUPPORTED — do not claim** | `mia_auc_domias_clf` reads 0.513–0.537 across the *entire* sweep, including bw = 0.005 where the generator emits training rows plus N(0, 0.005²). An attack that cannot detect near-verbatim memorisation has no sensitivity, so it fails as a cross-family validator rather than contradicting the KDE variant. The matched-model-class objection to the dial (DOMIAS estimates by KDE; the dial *is* a KDE) therefore stands unanswered. Resolving it needs a non-KDE dial or a second attack family — see §3, Gen-LRA / ReMIA. |
| **"The DCR/fidelity trade-off holds within a single generator family"** | **ACTIVE** | Along the dial, everything but bandwidth is held fixed: `dcr_ratio` rises 0.034 → 1.348 while `fidelity_score` falls 0.941 → 0.883, monotonically. This is stronger evidence for the mechanism than the cross-model rank association, which confounds architecture with fit quality. |
| **"Distinguishability ≠ membership inference"** — C2ST ≈ 0.97–0.99 while DOMIAS ≈ 0.50 on identical data | **ACTIVE** | Both computed per run in `evaluation/privacy.py`; attack controls at 0.93 (positive) / 0.49 (negative) in `evaluation/results/attack_validation.csv`. This is the cleanest surviving methodology finding. |
| **"PHQ-9 derived-target leakage inflates TRTR AUC"** | **ACTIVE — post-fix value only** | Post-fix TRTR 0.7125 traces to `evaluation/results/`. The pre-fix **0.9999 is historical: no code path in this repo regenerates it**, so quote it as history, never as a reproducible measurement. The guard that works is the per-item DPQ assert (`preprocess.py:222`); the `PHQ9_TOTAL` assert at `:219` is **vacuous** — `build_dataset` constructs `PHQ9_RAW`, never `PHQ9_TOTAL`. `PHQ9_RAW` is dropped at `:195` but **has no assert**. See §2 2026-07-22. |
| **"Rank-based marginal correction games Axis 1"** | **ACTIVE** | It remaps continuous columns onto exact real training floats. Kept only as the opt-in `--ablation-mc` path, audited as `{model}_mc`. |
| **"No generator passes the Four-Axis Audit"** | **ACTIVE** | All four fail the 0.90 fidelity gate. This is a statement about the gate on 23-column survey data as much as about the generators; do not report it as a generator ranking. |

---

## §2 Decision log

Newest first.

### 2026-07-22 — Audited state frozen in Git (tag `audit-2026-07-22`)

**Why.** Nothing was committed and `.gitignore` excluded `CLAUDE.md`, `paper/` and
`evaluation/results/`, so the paper, every number in it and the audit report existed only in one
working tree. A fresh clone reproduced nothing and contained no results — a submission blocker for any
venue with an artifact requirement, and a single-disk-failure risk for the whole project.

**Decision.** Five single-purpose commits on `calibrated-audit-pivot`, then the annotated tag
`audit-2026-07-22`. Un-ignored: `CLAUDE.md`, `paper/` (356 KB), `evaluation/results/*.csv` (184 files,
557 KB). Still ignored on size/regenerability grounds: `evaluation/results/*.npy` (2.9 MB),
`evaluation/plots/` (22 MB), `logs/`, `data/`, `models/saved/`. `requirements_lock.txt` (`pip freeze`)
replaces the stale hand-maintained `logs/environment.txt`, which claimed PyTorch 2.12.0+cpu against an
audited torch 2.13.0.

**Verification at freeze time.** `scripts/check_consistency.py` exit 0 over **59** result files;
`compileall` clean; attack controls 0.92916 / 0.50277, both passing.

**Two documentation corrections shipped with the freeze**, both found by the audit and neither
affecting any number:
1. The pre-fix TRTR **0.9999 has no regenerating code path** and is now labelled historical wherever it
   appears (§1, §2 above, paper §III-C and Table 7). No replacement value was invented.
2. `PHQ9_TOTAL` is never constructed — see the naming correction under the 2026-05-26 entry.

**Deliberately not done in the freeze.** No rerun of preprocessing, training, generation or any
experiment; no code change to `preprocess.py`'s dead `PHQ9_TOTAL` name; no per-class utility table in
§VI; no §II Related Work. All are tracked in `docs/SUBMISSION_BLOCKERS.md` on the
`submission-hardening` branch, which starts from this tag. **The tag never moves.**

### 2026-07-22 — Related Work citation base expanded (26 net-new, sourced not yet drafted)

**Why.** §II is a STUB with only 11 works, none DOI-verified — too thin to write real related-work prose
from, and thin enough that reviewers would ask about obvious omissions (foundational MIA, generation
architecture families beyond the 4 in our matrix, attribute inference literature backing the sanctioned
`experiments/attribute_inference_dpq.py` path). Sourced via Consensus (200M-paper academic search) across
six clusters: generation architectures, evaluation frameworks, foundational/low-FPR MIA, DCR/distance-metric
critique and lineage, attribute inference, and differential-privacy synthetic data — the last cluster filled
by direct web search since Consensus queries returned mostly domain-specific applications rather than the
canonical DP-GAN papers. One venue correction: the "Ganev et al. ESORICS'25" candidate named in the
superseded `docs/section 2/06_PAPER_REFRAME.md` outline is actually **IEEE S&P 2025**, confirmed via arXiv
2312.05114 — corrected here, not in that superseded file (§5).

**Decision.** Add a second, clearly-separated table to §3 ("Background & extended citation base") rather
than merging into the existing 11-row "prior work each claim must survive" table — the new entries are
broader supporting/lineage literature, not all of them carry a specific "our delta" the way the closest-prior-work
row does. Mirrored into `paper/paper.md` §II's drafting table; the STUB banner stays, since sourcing is not
drafting.

**Invalidates.** Nothing — additive only. Does not touch the 11 existing rows, §1 claim statuses, or any
number.

**Files.** `reference.md` §3, `paper/paper.md` §II.

---

### 2026-07-22 — SDV baselines were sampling under a fixed internal seed

**Why.** A full-pipeline verification re-run found that `gc_synthetic_seed4{2..6}.csv` were
**byte-identical** — one MD5 across all five seeds. `generate_sdv` called `set_all_seeds(seed)` and
then `cls.load(ckpt).sample(...)`, but SDV wraps sampling in its own `FIXED_RNG_SEED = 73251` and
ignores the global RNG. GaussianCopula's fit is deterministic, so identical model + identical sampler
seed = identical rows: its reported `1.661 ± 0.000` / `0.833 ± 0.000` was a **single sample**, and the
non-zero ±std on its utility and MIA columns was evaluator noise, not generator variance. TVAE was
affected too, less visibly — it varied across seeds only through its per-seed weights, never through
sampling, so both baselines understated their variance.

**Decision.** `generate_sdv` calls `synthesizer._set_random_state(seed)` before `sample()`.
`_set_random_state` is the available API on SDV 1.37.3 (`set_random_state` does not exist in this
version). The private name is deliberate and commented — there is no public equivalent here, and
without it the seed loop is decorative.

**Invalidates.** The `tvae` and `gc` rows of the frozen table (CLAUDE.md, paper Table 1) and the
Spearman pair in paper §VI-B. **Not** the `vae`/`ctgan` rows: both reproduce bit-exactly (maxdiff
0.000000 across all 10 runs), which is also the evidence that the fix's blast radius is exactly the
two SDV baselines. `mia_power.csv` (CTGAN) and `leakage_dial.csv` (KDE) reproduce unchanged, so §VII-C
and §VI-C are untouched. Axis pass counts (paper Table 2) are unchanged.

**Re-frozen.** `tvae` 0.891 / 0.844 / 0.798 / 0.523; `gc` 0.834 / 0.685 / **1.676 ± 0.005** / 0.490
(fidelity / utility / dcr_ratio / mia_auc). Spearman(fidelity, dcr_ratio) = −0.364, p = 0.115 — still
not significant, so the §VI-B honesty statement stands as written.

**Files.** `baselines/sdv_baselines.py`, `CLAUDE.md`, `paper/paper.md`, `reference.md`.

---

### 2026-07-21 — Results freeze at n = |train|

**Why.** All numbers regenerated after the power fix, so the whole table had to move together.

**What was run.** `attack_validation.py` (both controls pass: 0.929 / 0.503) → `mia_power.py` →
`run_all_seeds.py` (20 runs, checkpoints reused, no retraining) → `leakage_dial.py` (35 runs) →
`check_consistency.py` (**59 result files, all consistent**) → `make_figures.py` (figs 1–3).

**Outcome.** The main-matrix numbers barely moved from the n = 1000 table — fidelity, utility and DCR
ratios are within ~0.02 — so the power fix changed the *warrant* for the privacy null, not the null
itself. What did change: the DCR-vs-fidelity claim lost its cross-model statistical support
(Spearman p = 0.124) and gained much stronger within-family support from the dial.

**Invalidates.** The pre-2026-07-21 Known Results table; `docs/RESULTS.md`, `SESSION.md`,
`docs/dashboard_methodology.md`, `docs/PROJECT_PLAN.md`, `docs/PROJECT_SUMMARY.md`, `docs/REPORT.md`
— all now carry SUPERSEDED banners (§5).

**Files.** `CLAUDE.md` (Known Results), `paper/paper.md` (§VI, §VII-C, Abstract), all of
`evaluation/results/`, `paper/figures/fig{1,2,3}`.

**Not committed at time of writing** — record the SHA here on commit.

---

### 2026-07-21 — Power fix, low-FPR reporting, and a leakage dial

**Why.** A Consensus literature sweep put the project's money finding in direct collision with Yao et
al. 2025 (§3), which runs the same holdout-calibrated DCR test more broadly. Our differentiator has to
be evidence quality, and one confound undermines it: `N_SYNTHETIC = 1000` against a ~6.3k training
split, while DOMIAS estimates `p_syn` by KDE over exactly those 1000 rows. A flat AUC of 0.50 is not
distinguishable from an underpowered attack, and the existing positive control (noisy verbatim copies)
is too easy a target to exclude that. Separately, AUC-only MIA reporting is no longer the accepted
standard (Zarifzadeh et al. 2023; Ward et al. 2025), and no run in the matrix has ever exercised the
audit against a generator that demonstrably leaks.

**Decisions.**
1. Synthetic sample count is `len(nhanes_train)`, not `N_SYNTHETIC`. `config.N_SYNTHETIC` is demoted to
   a CLI default only — a row count in `config.py` goes stale the moment the splits change.
2. The MIA evaluation set uses the full holdout (`min(len(train), len(test))` per class) rather than a
   500-row cap.
3. `mia_tpr_at_fpr01` (TPR at FPR = 0.01) is reported in every `faa_*.csv`, **non-gating**. FPR = 0.001
   is deliberately *not* reported: with ~1345 non-members it resolves to ~1.3 negatives.
4. Leakage is dialled with a `KernelDensity` generator fit on the full training split, swept over
   bandwidth. Small bandwidth = verbatim memorisation, large = smooth population model. Chosen over an
   overfit VAE because it needs no training and no new dependency, and over a subset-fit KDE because
   fitting on the full split keeps every audit member a row the generator actually saw.

**Invalidates.** The frozen cycle-P results table in CLAUDE.md and every number in
`evaluation/results/` produced at n = 1000. Model checkpoints in `models/saved/` remain valid —
nothing here changes a model.

**Known weakness, disclosed not hidden.** DOMIAS estimates `p_syn` by KDE and the dial generator is a
KDE — matched model classes, so the attack is favourably placed against it. The random-forest
density-ratio variant was intended as the cross-family check; **it did not work.** It reads 0.513–0.537
at every bandwidth, including total memorisation, so it has no sensitivity to leakage and cannot
validate anything. The objection stands unanswered and is stated as a limitation rather than
mitigated. Closing it needs a non-KDE dial (an overfit neural generator) or a genuine second attack
family.

**Files.** `run_all_seeds.py`, `evaluation/privacy.py`, `evaluation/four_axis_audit.py`,
`experiments/mia_power.py`, `experiments/leakage_dial.py`, `paper/figures/make_figures.py`.

---

### 2026-07-01 — Axis 4 replaced with a validated DOMIAS attack

**Why.** The previous Axis 4 trained a classifier to separate real training rows from synthetic rows
and called the accuracy "shadow-model MIA". That measures **distinguishability**, not membership: it
saturates near its ceiling for any imperfect generator regardless of whether a single training record
leaked. It produced the retracted 15/15 headline in §1. Replaced with the DOMIAS density-ratio attack
(van Breugel et al. 2023): `score(x) = log p_syn(x) − log p_ref(x)`, with the validation split as the
population reference and ROC AUC over balanced train-members vs test-non-members.

**Also decided.** No privacy number is trustworthy without attack controls, so
`evaluation/attack_validation.py` gates the pipeline: a positive control (noisy training copies must
fire, AUC > 0.70) and a negative control (fresh real records must not, AUC ∈ [0.45, 0.55]). It exits
non-zero on failure. **Do not modify the attack without re-running it.** The old distinguishability
number is retained under its honest name, `c2st_acc`, as a non-gating metric — the contrast between
the two is now itself a finding.

**Invalidates.** All Axis-4 numbers in `docs/RESULTS.md`; the entire "DCR-MIA gap" framing.

**Files.** `evaluation/privacy.py`, `evaluation/attack_validation.py`.
**Source.** `docs/section 2/06_PAPER_REFRAME.md` (supersedes the original Task 06).

---

### 2026-07-01 — Axis 3 calibrated against a holdout instead of an absolute threshold

**Why.** The old gate was `mean DCR > 0.10`, an absolute distance. A generator that misses the data
manifold entirely maximises that number, so the metric rewarded being wrong. Replaced with
`dcr_ratio = DCR(syn→train) / DCR(test→train) ≥ 1.0`: the question is not "is synthetic close to
training data" but "is it closer than unseen real data is". This is Platzer & Reutterer's construction
(§3), not ours — the framing in the paper must say so.

**Invalidates.** Every Axis-3 number and pass/fail in `docs/RESULTS.md`.
**Files.** `evaluation/privacy.py::axis3_dcr`.

---

### 2026-07-01 — Axis 1 switched from KS p-values to effect sizes

**Why.** The old gate was "KS p ≥ 0.05 on ≥80% of columns". At n ≈ 6.3k real vs 1000 synthetic, a KS
test rejects on trivially small differences, so the pass rate was pinned at 0.043 for every model and
every seed — the metric carried no information and could not have passed by construction. Replaced
with mean per-column effect size (KSComplement for continuous, 1 − TVD for categorical, SDMetrics
convention), gate 0.90.

**Invalidates.** The "KS pass rate" row in `docs/RESULTS.md`.
**Files.** `evaluation/statistical.py::axis1_fidelity`, `config.py::FAA_FIDELITY_SCORE`.

---

### 2026-07-01 — Marginal correction demoted to an opt-in ablation

**Why.** Rank-remapping each continuous column onto the real training CDF makes Axis 1 pass almost by
construction — but the corrected values are *exact floats copied from `nhanes_train.csv`*, i.e. it buys
fidelity with verbatim disclosure. Leaving it on by default would have been metric gaming. It is now
`--marginal-correction` / `run_all_seeds.py --ablation-mc`, applied uniformly to all four models when
used, audited under the separate name `{model}_mc`, and reported in the paper as a worked example of
how a fidelity metric is gamed at the direct expense of privacy.

**Files.** `generate.py::apply_marginal_correction`, `run_all_seeds.py`.

---

### 2026-07-01 — VAE generates its label jointly; categoricals snapped for every model

**Why.** Two separate correctness bugs in generation. (a) VAE labels were previously assigned by a
random forest fit on real data, which is circular with the TSTR utility evaluation. The target is now
appended to the training matrix so the decoder emits it, and `generate_vae` **refuses any checkpoint
without a `joint_target` flag** rather than silently producing circular labels. (b) All generators
emit continuous values for categorical columns, which makes synthetic rows trivially separable on every
categorical column; they are now snapped to valid real training levels for all four models — VAE/CTGAN
in `generate.py`, the SDV baselines in `run_all_seeds.py`.

**Files.** `models/vae/train.py`, `generate.py::_discretize_categoricals`, `run_all_seeds.py`.

---

### 2026-07-01 — GaussianCopula added as a fourth model

**Why.** A cheap statistical floor was needed to make the DCR argument legible: GC has the best
`dcr_ratio` in the matrix despite mediocre fidelity, which is the clearest single illustration of the
active claim in §1. It sits alongside TVAE in `baselines/sdv_baselines.py`, keyed by `kind` in the
`SYNTHESIZERS` dict.

**Warning.** Editing the frozen constructor kwargs in `SYNTHESIZERS` retrains a different model and
invalidates the 10 saved `.pkl` checkpoints behind the results table.

**Source.** `docs/section 2/04_TVAE_BASELINE.md` (extended beyond TVAE).

---

### 2026-06-10 — VAE decoder made column-type-aware

**Why.** A single sigmoid on the decoder output saturates on continuous columns whose mass sits near
the tails after MinMax scaling, compressing the synthetic distribution and costing KS score. Binary
columns keep `sigmoid`; continuous columns use `clamp(x, 0, 1)`.

**Files.** `models/vae/model.py`. **Source.** `docs/section 2/05_VAE_ARCHITECTURE_FIX.md`,
`paper/vae_fix_note.md`.

---

### 2026-06-10 — Five seeds mandatory; splits are CSVs, not index arrays

**Why.** Single-run numbers are not reportable. Every experiment runs `SEEDS = [42, 43, 44, 45, 46]`
and every table is mean ± std, with `set_all_seeds()` (python/numpy/torch + cudnn-deterministic) at
the top of each entry point. The 70/15/15 stratified splits are written as three separate CSVs;
the old `splits.pkl` index arrays were write-only and were removed — **do not reintroduce them.**

**Files.** `config.py`, `utils/seed_utils.py`, `preprocess.py`.
**Source.** `docs/section 2/02_SEED_AND_REPRO.md`.

---

### 2026-05-26 — PHQ9_TOTAL and all 9 DPQ items dropped from the feature set

**Why.** `Depression_Severity` is derived from `PHQ9_RAW = sum(DPQ items)` by fixed clinical
thresholds, so both are deterministic functions of the target. With them present, TRTR AUC was 0.9999
(historical measurement — no code path in this repo regenerates it) — the classifier was recovering an
arithmetic identity, and every utility number computed before this fix is invalid. `preprocess.py`
asserts the nine DPQ items do not survive into the split CSVs.

**Naming correction, 2026-07-22.** The column this entry calls `PHQ9_TOTAL` is never constructed:
`build_dataset` derives `PHQ9_RAW = sum(DPQ items)` directly. So the `PHQ9_TOTAL` assert at
`preprocess.py:219` can never fire, and `PHQ9_RAW` — which *is* the deterministic sum — is dropped at
`:195` but has no assert of its own. The nine per-item DPQ asserts at `:222` are the guard actually
doing the work. The drop is unconditional, so no leakage occurs; the assert net simply has a hole
where it matters most.

**One sanctioned exception.** `build_dataset(drop_leakage=False)` keeps the 9 DPQ items so they can be
treated as *sensitive attributes* in the attribute-inference study
(`experiments/attribute_inference_dpq.py`). That path still drops `PHQ9_TOTAL`/`PHQ9_RAW`, never
writes the canonical split CSVs, and **must never feed the FAA pipeline.**

**Invalidates.** All pre-fix utility numbers. Any document describing generation as "conditioning on
PHQ9_TOTAL" or "re-deriving Depression_Severity from PHQ9_TOTAL" is describing code that no longer
exists.

**Files.** `preprocess.py` (`LEAKAGE_COLS`). **Source.** `docs/section 2/01_LEAKAGE_FIX.md`,
`paper/leakage_disclosure.md`.

---

## §3 External positioning

Prior work each claim must survive. The delta line is what we are allowed to claim over it — nothing
broader.

| Work | What it establishes | Our delta |
|---|---|---|
| **Yao et al. 2025, *The DCR Delusion*** | DCR and other distance proxies — including the holdout-calibrated binary test — fail to identify privacy leakage; datasets they call private are vulnerable to MIAs. Across multiple datasets, BayNet/CTGAN/diffusion. | Closest prior work; it scoops the bare claim. We may claim: the clinical survey domain, the *mechanism* (`dcr_ratio` measured as a function of fidelity, which they argue by design rather than measure), and the bandwidth at which the gate flips. |
| **Platzer & Reutterer 2021** (Frontiers in Big Data) | The holdout-calibrated assessment framework: show synthetic is no closer to train than a disjoint holdout is. | `dcr_ratio` **is** their metric with a threshold attached. Must be cited as lineage, never presented as ours. |
| **van Breugel et al. 2023** (DOMIAS) | Density-ratio MIA targeting local overfitting; the attack we use as Axis 4, unmodified. | Instrumentation only. Zero novelty claimed. |
| **Zarifzadeh et al. 2023** (RMIA) | Attacks that look like chance in AUC can succeed sharply at low FPR; AUC alone is insufficient reporting. | Why `mia_tpr_at_fpr01` is reported. Also why our AUC ≈ 0.50 null needs the low-FPR column before it can be called a null. |
| **Ward et al. 2025** (Gen-LRA) | No-box local-likelihood-ratio MIA; strongest gains at low FPR. | Candidate second attack family if the matched-KDE objection to our dial lands. Not implemented. |
| **Scassola et al. 2026** (ReMIA) | Practical MIA needing only two SDG training runs; notes DCR's limited sensitivity to MIA risk. | Alternative second attack. Not implemented. |
| **Kaabachi et al. 2025** (NPJ Digital Medicine) | Scoping review: no consensus on privacy/utility evaluation; most studies either omit privacy evaluation or significantly underestimate risk. | Supports the audit's framing, and pre-empts any claim that "attack calibration matters" is itself novel — it is not. |
| **Lautrup et al. 2024** (SynthEval), **Hernandez et al. 2025** (Frontiers Digital Health) | Existing fidelity/utility/privacy evaluation frameworks for tabular health data. | The Four-Axis Audit must be presented as a **holdout-calibrated instantiation of an existing checklist**, not a novel framework. Novelty is in the findings, not in the act of evaluating on four axes. |
| **Jiang et al. 2025** (KDD) | Utility/fidelity/privacy benchmark of Synthpop/CTGAN/TVAE/REaLTabFormer on national survey data. | Nearest domain neighbour. Distinguish on the calibrated privacy axes and the clinical instrument. |

### Background & extended citation base (sourced 2026-07-22, not yet drafted into prose)

Broader supporting literature for §II, found via Consensus + targeted web search (see §2 2026-07-22).
Unlike the table above, not every row is "prior work our claim must survive" — several are lineage,
foundational-method, or convergent-but-independent findings. DOI given where confirmed by direct lookup;
Consensus-sourced rows link to the Consensus paper page (stable, resolves to publisher/arXiv).

**Generation architectures** (completes the GAN/VAE/copula/diffusion/LLM family beyond our 4 models):

| Work | Establishes | Relevance |
|---|---|---|
| Patki, Wedge & Veeramachaneni 2016, *The Synthetic Data Vault* (IEEE DSAA, [DOI 10.1109/DSAA.2016.49](https://doi.org/10.1109/DSAA.2016.49)) | Origin of the Gaussian-copula synthesizer and the SDV project. | `gc` baseline's actual lineage — cite instead of only naming the library. |
| Kotelnikov, Baranchuk, Rubachev & Babenko 2023, *TabDDPM* (ICML, [arXiv:2209.15421](https://arxiv.org/abs/2209.15421)) | Diffusion model for heterogeneous tabular data, outperforms GAN/VAE baselines. | Names the architecture family (diffusion) our 4 models don't cover; standard "future work" citation. |
| Zhang et al. 2023, *Tabsyn* — [Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space](https://consensus.app/papers/details/6e1f2d39e7ec567b90983464ebbb30d6/?utm_source=claude_desktop) (ArXiv, 251 citations) | VAE-latent-space + diffusion hybrid for mixed-type tabular data. | Directly relevant to our VAE architecture discussion — a diffusion model built on the same latent-space idea. |
| Borisov et al. 2022, *GReaT* — [Language Models are Realistic Tabular Data Generators](https://consensus.app/papers/details/37c70e6d5be15b269d92208819d55feb/?utm_source=claude_desktop) (ArXiv, 410 citations) | Autoregressive LLM sampling for tabular synthesis. | Completes the architecture survey with the LLM family. |
| Solatorio & Dupriez 2023, *REaLTabFormer* ([arXiv:2302.02041](https://arxiv.org/abs/2302.02041), 123 citations) | Transformer-based relational/tabular synthesis with overfitting detection via a $Q_\delta$ statistic. | Namechecked only indirectly (via Jiang et al. 2025's benchmark) in the existing table — now cited directly; its overfitting-detection statistic is relevant to our leakage-dial discussion. |
| Shi et al. 2025, [A Comprehensive Survey of Synthetic Tabular Data Generation](https://consensus.app/papers/details/357de8a2dba352d78b952044efdc7f69/?utm_source=claude_desktop) (ArXiv, 35 citations) | Unifying survey spanning traditional/diffusion/LLM tabular generators. | Single Introduction-level citation for "the field has moved beyond GAN/VAE." |
| Hernandez et al. 2022, [Synthetic data generation for tabular health records: A systematic review](https://consensus.app/papers/details/0ff507302e57579b8d62773b8bf10dfd/?utm_source=claude_desktop) (Neurocomputing, 312 citations) | 2016–2021 review of GAN-based synthetic health-record generation; finds no universal evaluation method/metric. | Earlier, health-domain-specific companion to the already-listed Hernandez et al. 2025 (Frontiers Digital Health) — verify if same author group before citing both. |

**Evaluation frameworks / benchmarks:**

| Work | Establishes | Relevance |
|---|---|---|
| Figueira & Vaz 2022, [Survey on Synthetic Data Generation, Evaluation Methods and GANs](https://consensus.app/papers/details/170b622eaae75f71b33c4f571203d4d0/?utm_source=claude_desktop) (Mathematics, 402 citations) | Combined GAN + synthetic-data-evaluation survey across WoS/Scopus/IEEE/ACM. | Broad-coverage citation for "no consensus on evaluation," complements Kaabachi et al. |
| Endres, Mannarapotta Venugopal & Tran 2022, [Synthetic Data Generation: A Comparative Study](https://consensus.app/papers/details/9aedc4af9df458f49a2395dc901a0b18/?utm_source=claude_desktop) (IDEAS, 66 citations) | Empirical comparison of GAN/VAE/SMOTE/Gaussian-copula/CTGAN/SynthPop on utility metrics. | Independent empirical precedent for comparing our exact model family (GC vs. CTGAN vs. VAE-style). |

**Membership inference — foundational & low-FPR reporting:**

| Work | Establishes | Relevance |
|---|---|---|
| Shokri, Stronati, Song & Shmatikov 2017, [Membership Inference Attacks Against Machine Learning Models](https://consensus.app/papers/details/23b408d45e155fab98e4ecc65bd08492/?utm_source=claude_desktop) (IEEE S&P, 5324 citations) | The foundational MIA formulation (shadow models, black-box). | Root citation the entire Axis-4 lineage (DOMIAS, RMIA, Gen-LRA) descends from; currently missing. |
| Salem et al. 2018, [ML-Leaks](https://consensus.app/papers/details/6ab31115b59e543fbac33be6e6e54579/?utm_source=claude_desktop) (ArXiv, 1164 citations) | Relaxes Shokri et al.'s assumptions (no shadow models / target architecture needed). | Relevant given our own retracted Axis-4 used an undisclosed shadow-style classifier (§1, §2 2026-07-01). |
| Hu et al. 2022, [Membership Inference Attacks on Machine Learning: A Survey](https://consensus.app/papers/details/44937d6d1d1f5c3eb22860b3d09b5a6b/?utm_source=claude_desktop) (ACM Computing Surveys, 685 citations) | Taxonomy of MIA attacks and defenses. | Single survey citation covering the attack family tree. |
| Carlini, Chien, Nasr, Song, Terzis & Tramèr 2022, *LiRA* — [Membership Inference Attacks From First Principles](https://consensus.app/papers/details/1ff0d023386752189c898c53979d3b51/?utm_source=claude_desktop) (IEEE S&P, 1118 citations) | Argues MIA must be reported as TPR at low FPR, not average-case accuracy/AUC; introduces LiRA. | Co-founds (with Zarifzadeh et al., already listed) the justification for our `mia_tpr_at_fpr01` metric — arguably should be cited ahead of RMIA as the original source of the low-FPR argument. |
| Ye et al. 2022, [Enhanced Membership Inference Attacks against Machine Learning Models](https://consensus.app/papers/details/b03b34b37027531d847ba1373fcaf675/?utm_source=claude_desktop) (ACM CCS, 372 citations) | Hypothesis-testing framework unifying prior MIAs via reference models ("Privacy Meter"). | Alternative attack-family framing to DOMIAS/RMIA; useful contrast in a Related Work paragraph on attack design choices. |
| Jebreel, Domingo-Ferrer et al. 2026, [Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions](https://consensus.app/papers/details/3e9e0c5e58a55fa7950e81a7422dcbba/?utm_source=claude_desktop) (ArXiv) | Shows LiRA's reported power is inflated by unrealistic evaluation assumptions (overconfident models, target-calibrated thresholds). | Directly parallel to our own methodology point: an uncontrolled attack overstates/understates risk — this is why `attack_validation.py`'s controls exist. |
| Zhu, Salehi & Simeone 2024, [On the Impact of Uncertainty and Calibration on Likelihood-Ratio Membership Inference Attacks](https://consensus.app/papers/details/cd6957073ce853c589cefe1ac6999c91/?utm_source=claude_desktop) (IEEE TIFS) | Theoretical bounds on LiRA effectiveness under model calibration/uncertainty. | Secondary technical citation on why attack calibration matters, if the paragraph needs a theory anchor. |
| Rezaei & Liu 2021, [On the Difficulty of Membership Inference Attacks](https://consensus.app/papers/details/a6420fe928e256bcaf17eee68fb694b1/?utm_source=claude_desktop) (CVPR, 114 citations) | Shows reported MIA "success" is often an artifact of unreported false-positive/false-alarm rate. | Closest external parallel to our own cautionary tale (Contribution 3, paper.md §I) — an uncontrolled attack producing a confident, wrong privacy finding. |
| Cheng et al. 2025, [Membership Inference over Diffusion-models-based Synthetic Tabular Data](https://consensus.app/papers/details/d460152658e250b7a5ca4a617a1a2405/?utm_source=claude_desktop) (ArXiv) | TabDDPM more MIA-vulnerable than TabSyn — attack success is architecture-dependent. | Independent evidence that MIA risk varies by generator family, echoing our own cross-model matrix. |

**DCR / distance-based privacy metrics — critique and lineage (beyond Yao et al. and Platzer & Reutterer, already listed):**

| Work | Establishes | Relevance |
|---|---|---|
| Meeus, Guépin, Cretu & de Montjoye 2023, [Achilles' Heels: Vulnerable Record Identification in Synthetic Data Publishing](https://consensus.app/papers/details/576823557a985ed5a91b618498e2778a/?utm_source=claude_desktop) (ArXiv/ESORICS, 35 citations) | Nearest-neighbour-distance-based method to identify MIA-vulnerable records without running full shadow-model MIAs. | Distance-based reasoning aimed *at* predicting MIA risk (contrast with DCR, which is claimed to predict it and fails per Yao et al.) — useful counterpoint. |
| Ganev & De Cristofaro 2025, [The Inadequacy of Similarity-Based Privacy Metrics](https://arxiv.org/abs/2312.05114) (IEEE S&P, 23 citations) | Independent critique of similarity/distance-based privacy tests; introduces a reconstruction attack (Recon-Syn) beating datasets that pass the metrics. | **Venue-corrected** version of the paper flagged as a dropped candidate in the superseded `06_PAPER_REFRAME.md` (that doc guessed ESORICS'25; it's IEEE S&P 2025). Convergent, independent evidence for "distance-based privacy tests are insufficient" — strengthens our claim by being a separate research group reaching the same conclusion via a different attack. |
| Pathak et al. 2026, [Quantifying Membership Disclosure Risk for Tabular Synthetic Data Using Kernel Density Estimators](https://consensus.app/papers/details/b74decbccb3b55148064c5d8f2f81b27/?utm_source=claude_desktop) (ArXiv) | KDE-based membership-disclosure-risk quantification for tabular synthetic data via nearest-neighbour-distance distributions. | Methodologically close to our own leakage dial (a `KernelDensity` generator swept over bandwidth, §2 2026-07-21) — flag as closely related concurrent work, not prior art we build on. |

**Attribute inference** (backs the sanctioned `experiments/attribute_inference_dpq.py` path, §"Full feature set" exception in CLAUDE.md):

| Work | Establishes | Relevance |
|---|---|---|
| Jayaraman & Evans 2022, [Are Attribute Inference Attacks Just Imputation?](https://consensus.app/papers/details/279797ceaed75036ac3d5e7ae54a9c2f/?utm_source=claude_desktop) (ACM CCS, 87 citations) | Black-box attribute inference rarely reveals more than population-level imputation would; white-box can. | Methodological caution directly applicable if/when the DPQ attribute-inference study is written up — distinguishes real leakage from population-statistics recovery. |
| Annamalai, Gadotti & Rocher 2023, [A Linear Reconstruction Approach for Attribute Inference Attacks against Synthetic Data](https://consensus.app/papers/details/6e68f42af34f58d2ba8b474cee9a66b8/?utm_source=claude_desktop) (ArXiv, 40 citations) | Attribute-inference attack targeting all records (not only outliers) via aggregate-statistic reconstruction. | Prior-work anchor for the attribute-inference axis if it's ever promoted from experiment to a fifth audit axis. |
| Kwatra et al. 2024, [Empirical Evaluation of Synthetic Data Created by Generative Models via Attribute Inference Attack](https://consensus.app/papers/details/a9d92eb356b3543fb9f1987fc54e8c88/?utm_source=claude_desktop) | Empirical AIA across GAN/diffusion/DPGAN synthetic tabular data; categorical attributes resist AIA better than continuous. | Directly relevant to our binary/categorical vs. continuous feature split — predicts which of our 23 features would be more attribute-inference-vulnerable. |

**Differential privacy for synthetic data** (sourced via web search — Consensus returned mostly domain applications, not the canonical DP-GAN papers):

| Work | Establishes | Relevance |
|---|---|---|
| Jordon, Yoon & van der Schaar 2019, [PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees](https://openreview.net/forum?id=S1zk9iRqF7) (ICLR) | Canonical formally-private synthetic-data generator (PATE applied to GANs). | The alternative privacy paradigm we deliberately don't use — worth one sentence in Discussion/Limitations contrasting empirical (holdout+MIA) vs. formal (DP) guarantees, alongside Stadler et al. 2022 (already listed). |

---

## §4 Invariants

Each must never regress; each has an enforcing check.

| Invariant | Enforced by |
|---|---|
| No DPQ item reaches the feature set | per-item assert, `preprocess.py:222`; §2 2026-05-26 |
| No `PHQ9_RAW` reaches the feature set | unconditional drop at `preprocess.py:195` — **not asserted**; §2 2026-07-22 |
| Synthetic data stays in [0,1] scaled space — never re-scale before metrics | `scripts/check_consistency.py` |
| Categorical columns hold only valid real training levels | `generate.py::_discretize_categoricals`; asserted in `check_consistency.py` |
| Labels are integers in `{0..NUM_CLASSES-1}` | `check_consistency.py` |
| < 1% of non-ablation synthetic rows are exact copies of real training rows | `check_consistency.py` |
| Attack controls pass before any privacy number is quoted | `evaluation/attack_validation.py`, exits non-zero |
| `set_all_seeds(seed)` at the top of every training/generation run | convention; `utils/seed_utils.py` |
| `matplotlib.use("Agg")` before importing pyplot | convention — the tkinter backend crashes under threading |
| `paper/figures/make_figures.py` does not import `config.py` | avoids the torch dependency at figure time; it inlines its constants |
| Checkpoints are reused silently — delete `models/saved/` + `data/synthetic/` after changing a model, `preprocess.py`, or a hyperparameter | `check_consistency.py` is the backstop |

---

## §5 Superseded documents

Do not quote these. They are kept because they are the historical record of the artifact the paper
narrates, not because they are correct.

| File | Why superseded |
|---|---|
| `docs/RESULTS.md` | Every table uses the retracted thresholds (KS p-value pass rate ≥ 0.80, absolute DCR > 0.10, shadow-MIA ≤ 0.55) and asserts the retracted "15/15 runs" gap. Superseded by the Known Results table in CLAUDE.md. |
| `docs/section 2/CLAUDE.md` | An older master instruction file. Its FAA table, its MIA specification ("train a shadow model… label real rows 1, synthetic 0" — the distinguishability bug), and its "the DCR-vs-MIA gap is the paper's headline finding" line are all retracted. It also describes a repo layout (`GAN/preprocess_nhanes.py`, `baselines/tvae_baseline.py`) that no longer exists. |
| `docs/section 2/06_PAPER_REFRAME.md` §"original 06" references | The document itself is **current** and is the authoritative paper spec; only its description of the *original* Task 06 framing is historical. |
| `SESSION.md` | Session log from the pre-pivot run. States "DCR-MIA gap: 15/15 runs (100%)" and shadow-MIA 0.69–0.73 as results, and calls that gap "the DCR Delusion" — which inverts the actual meaning of the term (Yao et al. use it for DCR *under*-flagging real leakage, not for a distinguishability classifier firing). |
| `docs/dashboard_methodology.md` and `dashboard.html` | The dashboard reports the old axis definitions and "PASS — 100% of runs (15/15)". Not regenerated after the pivot. |
| `docs/PROJECT_SUMMARY.md`, `docs/REPORT.md`, `docs/PROJECT_PLAN.md` | Written against the pre-pivot pipeline. Verify any number against `evaluation/results/` before reuse. |

---

## How to add an entry

Append to the top of §2 in this format, then update §1 if the decision changes the status of a claim:

```markdown
### YYYY-MM-DD — <one-line decision>

**Why.** The problem that forced it. Name the alternative you rejected and why.

**Invalidates.** Artifacts, numbers, or documents this makes stale. Say "nothing" if nothing.

**Files.** Paths touched.
```

Record the decision here *before* implementing it, and cite the section in the implementation summary.
