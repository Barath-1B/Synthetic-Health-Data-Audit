# Auditing Synthetic Mental-Health Survey Data: When Distance Metrics Reward Infidelity

*Working draft. Section order and contribution list follow `docs/section 2/06_PAPER_REFRAME.md`,
which is the authoritative spec. Claim status is tracked in `reference.md` §1; do not promote a claim
here that is not `ACTIVE` there.*

**Drafting rules in force**
- Every number traces to a file under `evaluation/results/`, or it is an explicit `<!-- TBD-N -->`
  marker. Nothing is estimated, rounded from memory, or carried over from a superseded document.
- The retracted "DCR passes but MIA fails" claim appears **only** as the methodology cautionary tale
  in §VII, never as a result.
- The word "prove" is not used.

---

## Abstract

Synthetic tabular data is increasingly proposed as a route to sharing sensitive mental-health records,
and its privacy is usually certified by one of two empirical checks: a nearest-neighbour distance
proxy (Distance to Closest Record, DCR) or a membership inference attack (MIA). Calibrating DCR
against a real holdout — requiring that synthetic records sit no closer to the training set than
unseen real records do — is the standard remedy for the crudest failure of the raw distance. We show
on NHANES PHQ-9 depression-severity data that the calibrated version is still not sufficient. Auditing
four generators (a from-scratch VAE, a WGAN-GP conditional tabular GAN, and the SDV TVAE and
GaussianCopula baselines) over five seeds, we find no detectable membership leakage: a validated
DOMIAS attack reads AUC 0.495–0.519 with TPR at 1% FPR at or below the 0.01 chance rate, and this null
is stable across a 12.5-fold range in released-record count. We then sweep a kernel-density generator
across its bandwidth to vary leakage while holding everything else fixed. At bandwidth 0.15 the
generator **passes** the holdout-calibrated DCR gate (ratio 1.054) while the same attack reads AUC
0.601 at 4.5× chance TPR; at bandwidth 0.20 the DCR ratio reaches 1.348 and the attack still reads
0.572. Over the same sweep the DCR ratio rises monotonically from 0.034 to 1.348 while marginal
fidelity falls monotonically from 0.941 to 0.883 — a generator buys its DCR score with infidelity,
demonstrated within one model family rather than inferred across architectures. Separately, on
identical data a real-vs-synthetic distinguishability classifier reads 0.991–0.999 where the
calibrated attack reads ≈0.50; conflating the two, as applied work routinely does, manufactures
privacy alarms in proportion to how imperfect the generator is. We release the pipeline, the attack
controls that license every privacy number reported, and a consistency checker that recomputes each
metric from the synthetic data on disk.

---

## I. Introduction

Mental-health survey records are among the hardest health data to share. The instruments are short,
the populations are identifiable at small cell counts, and the outcome — depression severity — is
exactly the attribute a subject would least want disclosed. Synthetic data generation is the standard
proposed remedy: fit a generative model to the real records, release samples, and share the
statistical content without the individuals.

The remedy is only as good as the audit that certifies it. In applied practice two checks dominate.
The first is **Distance to Closest Record (DCR)**: measure how far each synthetic record sits from its
nearest real training record, and treat a large distance as evidence of privacy. The second is a
**membership inference attack (MIA)**: attempt to determine whether a specific real record was in the
generator's training set. DCR is cheap and appears in a large fraction of applied health-synthesis
papers; MIA is expensive and appears in far fewer.

Both are routinely misapplied, and this paper is about the two specific ways they fail on clinical
survey data.

**Calibrated DCR is still not sufficient, and it rewards infidelity.** A raw distance threshold —
"mean DCR > 0.10" and similar — is maximised by a generator that misses the data manifold entirely.
Calibrating against a holdout removes the crudest version of this failure by asking whether synthetic
data is closer to the training set than unseen real data is [Platzer & Reutterer 2021]. We construct a
generator that **passes** the calibrated gate while a validated membership attack detects its training
members at AUC 0.60, and we show that along the same one-parameter family the DCR ratio and marginal
fidelity move in opposite directions monotonically. DCR is measuring distance from the data manifold;
a poor generator maximises that quantity for free, and a generator can be simultaneously far from the
manifold and locally overfit to it.

**A distinguishability classifier is not a membership attack.** The most common cheap "MIA" in applied
work trains a classifier to separate real records from synthetic ones and reports its accuracy. That
quantity saturates near its ceiling for any imperfect generator, whether or not a single training
record leaked. We report this directly: on the identical data and generators, the distinguishability
classifier and a calibrated density-ratio attack (DOMIAS, [van Breugel et al. 2023]) return opposite
verdicts, and validation controls show the density-ratio result is the correct one.

**Relation to Yao et al. 2025.** The bare claim that DCR is uninformative about MIA risk is not ours;
*The DCR Delusion* establishes it across several datasets and generator families, including the
holdout-calibrated binary test we use. We claim three narrower things on top of it: the result holds in
a clinical survey domain with a derived clinical target; the *mechanism* is measured as a monotone
relationship between the DCR ratio and marginal fidelity within a single generator family, rather than
argued from construction across architectures; and a single-parameter sweep locates the operating
point at which the DCR gate flips while the attack is still firing, which turns a qualitative warning
into a specific region. We claim nothing broader, and §VI-C states plainly what our construction does
not establish.

**Contributions.**
1. A holdout-calibrated instantiation of a four-axis audit (fidelity, utility, nearest-neighbour
   privacy, membership privacy) applied to four generators × five seeds on NHANES PHQ-9 data, with the
   membership attack validated by positive and negative controls before use.
2. Evidence that the calibrated DCR ratio is negatively associated with marginal fidelity, and a
   single-parameter leakage dial locating where its gate flips.
3. A methodology cautionary tale: the same pipeline produced a confident, wrong privacy finding under
   a distinguishability-based Axis 4, and the controls that caught it.
4. A PHQ-9 derived-target leakage case study, and a metric-gaming ablation showing a fidelity metric
   being bought with verbatim disclosure.

---

## II. Related Work

> **STUB — HUMAN: write this section and verify every DOI.**
> Per `docs/section 2/06_PAPER_REFRAME.md` acceptance criteria, this section is not drafted
> automatically. The reference list and the required delta line for each entry are maintained in
> `reference.md` §3; they are reproduced here as drafting notes only. **No citation below has had its
> DOI verified. Do not submit this section as-is.**

| Work | Establishes | Delta we are permitted to claim |
|---|---|---|
| Yao et al. 2025, *The DCR Delusion* | Distance proxies, including the holdout-calibrated binary test, fail to identify leakage; data they call private is vulnerable to MIA. Multiple datasets, BayNet/CTGAN/diffusion. | Closest prior work. Ours: clinical survey domain; the fidelity mechanism measured rather than asserted; the gate-flip operating point. |
| Platzer & Reutterer 2021, *Holdout-Based Empirical Assessment of Mixed-Type Synthetic Data* | The holdout calibration itself. | `dcr_ratio` **is** their metric with a threshold attached. Cite as lineage; do not present as ours. |
| van Breugel et al. 2023, DOMIAS | Density-ratio MIA targeting local overfitting. | Used unmodified as Axis 4. No novelty claimed. |
| Zarifzadeh et al. 2023, RMIA | Attacks near chance in AUC can succeed sharply at low FPR; AUC alone is insufficient. | Why we report TPR at FPR = 0.01 alongside AUC. |
| Ward et al. 2025, Gen-LRA | No-box local likelihood-ratio MIA. | Candidate second attack family; not implemented (see §VIII). |
| Scassola et al. 2026, ReMIA | Practical MIA from two training runs; notes DCR's limited MIA sensitivity. | Alternative second attack; not implemented. |
| Kaabachi et al. 2025, NPJ Digital Medicine | Scoping review: no evaluation consensus; most studies omit or underestimate privacy risk. | Establishes the problem; also means "attack calibration matters" is **not** by itself a contribution. |
| Lautrup et al. 2024 (SynthEval); Hernandez et al. 2025 | Existing tabular fidelity/utility/privacy frameworks. | The Four-Axis Audit is a **calibrated instantiation of an existing checklist**, not a new framework. |
| Jiang et al. 2025 (KDD) | Utility/fidelity/privacy benchmark of Synthpop/CTGAN/TVAE/REaLTabFormer on national survey data. | Nearest domain neighbour; distinguish on the calibrated privacy axes and the clinical instrument. |
| Xu et al. 2019 | CTGAN / TVAE. | Generator architectures. |
| Stadler et al. 2022 (USENIX) | Synthetic data does not provide privacy for free. | Framing. |

**Extended citation base (sourced 2026-07-22, drafting notes only — same caveat as above; full detail
and links in `reference.md` §3 "Background & extended citation base").**

| Work | Establishes | Delta / relevance |
|---|---|---|
| Patki et al. 2016, *The Synthetic Data Vault* | Origin of the Gaussian-copula synthesizer. | `gc` baseline's actual lineage. |
| Kotelnikov et al. 2023, TabDDPM | Diffusion model for tabular data. | Names the architecture family (diffusion) our 4 models don't cover. |
| Zhang et al. 2023, Tabsyn | VAE-latent-space + diffusion hybrid. | Relevant to our VAE architecture discussion. |
| Borisov et al. 2022, GReaT | LLM-based tabular synthesis. | Completes the architecture survey with the LLM family. |
| Solatorio & Dupriez 2023, REaLTabFormer | Transformer tabular/relational synthesis, overfitting detection statistic. | Cited only indirectly via Jiang et al. 2025 today; now direct. |
| Shi et al. 2025 | Survey spanning traditional/diffusion/LLM tabular generators. | Introduction-level "field has moved past GAN/VAE" citation. |
| Hernandez et al. 2022 (Neurocomputing) | 2016–2021 review of GAN-based synthetic health-record generation. | Earlier health-domain companion to Hernandez et al. 2025 — verify same author group. |
| Figueira & Vaz 2022 | Combined GAN + synthetic-data-evaluation survey. | Complements Kaabachi et al. on "no evaluation consensus." |
| Endres et al. 2022 | Empirical benchmark of GC/CTGAN/VAE/SynthPop. | Independent precedent for comparing our exact model family. |
| Shokri et al. 2017 | Foundational MIA formulation. | Root citation the DOMIAS/RMIA/Gen-LRA lineage descends from — currently missing. |
| Salem et al. 2018, ML-Leaks | Relaxes shadow-model assumptions for MIA. | Relevant given our own retracted Axis-4 used an undisclosed shadow-style classifier. |
| Hu et al. 2022 | MIA attack/defense taxonomy survey. | Single survey citation for the attack family tree. |
| Carlini et al. 2022, LiRA | Argues MIA must report TPR at low FPR, not AUC. | Co-founds (with Zarifzadeh et al.) the justification for `mia_tpr_at_fpr01`. |
| Ye et al. 2022, Privacy Meter | Hypothesis-testing framework unifying MIAs via reference models. | Contrast framing to DOMIAS/RMIA attack design. |
| Jebreel et al. 2026 | LiRA's reported power inflated by unrealistic eval assumptions. | Parallels our own point: an uncontrolled attack misleads — why `attack_validation.py` exists. |
| Zhu et al. 2024 | Theoretical bounds on LiRA under calibration/uncertainty. | Secondary technical anchor if the attack-calibration paragraph needs theory. |
| Rezaei & Liu 2021 | Reported MIA "success" often an artifact of unreported FPR. | Closest external parallel to our retracted-Axis-4 cautionary tale. |
| Cheng et al. 2025 | TabDDPM more MIA-vulnerable than TabSyn. | Independent evidence MIA risk is architecture-dependent, echoing our cross-model matrix. |
| Meeus et al. 2023, Achilles' Heels | NN-distance method to identify MIA-vulnerable records. | Distance-based reasoning aimed *at* predicting MIA risk — counterpoint to DCR. |
| Ganev & De Cristofaro 2025 (IEEE S&P) | Independent critique of similarity/distance privacy metrics + Recon-Syn reconstruction attack. | Venue-corrected from the superseded reframe doc (guessed ESORICS'25; actually S&P'25). Convergent independent evidence for "distance-based privacy tests are insufficient." |
| Pathak et al. 2026 | KDE-based membership-disclosure-risk quantification for tabular synthetic data. | Methodologically close to our own leakage dial — flag as related concurrent work, not prior art we build on. |
| Jayaraman & Evans 2022 | Black-box attribute inference ≈ population imputation; white-box can exceed it. | Methodological caution for the sanctioned DPQ attribute-inference study. |
| Annamalai et al. 2023 | Attribute-inference attack targeting all records via aggregate-statistic reconstruction. | Prior-work anchor if attribute inference is ever promoted to a fifth audit axis. |
| Kwatra et al. 2024 | Empirical AIA across GAN/diffusion/DPGAN synthetic data; categorical resists better than continuous. | Predicts which of our 23 features are more attribute-inference-vulnerable. |
| Jordon et al. 2019, PATE-GAN | Canonical formally-private (DP) synthetic-data generator. | Alternative privacy paradigm we deliberately don't use — one sentence for Discussion/Limitations. |

---

## III. Dataset

### A. Source and construction

We use the U.S. National Health and Nutrition Examination Survey (NHANES), pre-pandemic cycle P
(2017–March 2020). Six component files are merged on the participant identifier `SEQN`, with the
PHQ-9 depression screener (`P_DPQ`) as the base table: `P_DEMO` (demographics), `P_MCQ` (medical
conditions), `P_SLQ` (sleep), `P_ALQ` (alcohol use), and `P_DR1TOT` (24-hour dietary recall). Using
DPQ as the base restricts the population to PHQ-9 completers, giving **8,965 records**; participants
who did not complete the screener are excluded rather than imputed.

NHANES stores several sentinel and proxy values that are not valid measurements — in particular SAS
numeric proxies in the interval $(0, 10^{-70})$. These are remapped before any other transform
(`preprocess.py::_replace_proxy`): `Total_Caffeine` proxies map to zero, since a proxy there denotes
"no caffeine recorded" rather than a missing measurement; the remainder map to `NaN` and are then
median-imputed. Categorical NHANES codes are remapped to contiguous levels, and `SEQN` is dropped.

All feature columns are MinMax-scaled to $[0,1]$. Synthetic data is produced and consumed in this same
scaled space throughout — no metric re-scales before computing, which keeps distances comparable
across columns of very different natural range (`Age` vs `Total_Calories`).

### B. Target

`Depression_Severity` $\in \{0,\dots,4\}$ is derived from the PHQ-9 raw sum using the standard clinical
thresholds ($\le 4$ none, $\le 9$ mild, $\le 14$ moderate, $\le 19$ moderately severe, $>19$ severe).
The resulting class distribution in the training split is severely imbalanced:

| Class | 0 (none) | 1 (mild) | 2 (moderate) | 3 (mod. severe) | 4 (severe) |
|---|---|---|---|---|---|
| Train count | 4,751 | 985 | 352 | 134 | 53 |
| Share | 75.7% | 15.7% | 5.6% | 2.1% | 0.8% |

Classes 3 and 4 together are under 3% of the data and are simultaneously the clinically important
ones. This matters for two axes. Utility is computed per class as well as macro-averaged — the
per-class AUC and F1 for every run are released in `evaluation/results/utility_per_class_*.csv`,
though §VI gates and tabulates only the macro ratio, so the aggregate number in Table 1 should not be
read as evidence of rare-class utility. Concretely: severe-class (class 4) TSTR F1 is 0.000 for every
generator — but so is **TRTR** F1, as it is for classes 2 and 3 under both protocols, so this is a
property of the class imbalance rather than of synthetic data. The measure that does separate them is
class-4 AUC: TRTR 0.765, CTGAN TSTR 0.566, GaussianCopula TSTR 0.391 — below chance. Separately, any
membership attack that succeeds only on rare records would be invisible in an aggregate AUC — which is
part of why we report TPR at low FPR (§IV-E).

### C. Feature set and the derived-target leakage fix

**This is the single most important construction decision in the pipeline, and it invalidated every
utility number the project produced before it.**

`Depression_Severity` is a deterministic function of `PHQ9_RAW`, which is the sum of the nine DPQ
items. With the raw sum and the DPQ items present as features, a downstream classifier does not model
depression — it recovers an arithmetic identity. Measured directly:

| | With the PHQ-9 sum + DPQ items | After removal |
|---|---|---|
| Feature count | 33 | 23 |
| TRTR macro AUC (RandomForest, OvR) | **0.9999**^ | **0.7125** |

^ Historical pre-fix measurement, recorded before the leakage columns were removed. **No code path in
the released artifact regenerates it**; it is reported as provenance, not as a reproducible result. The
post-fix 0.7125 traces to `evaluation/results/`.

Source: `paper/leakage_disclosure.md`. All ten leakage columns — `PHQ9_RAW` and the nine DPQ items —
are dropped before any split is written, and `preprocess.py` asserts the absence of each DPQ item. The retained feature set is 23 columns: 13
binary/categorical (`Gender`, `Race`, `Race_Ext`, `Education`, `Marital_Status`, `Arthritis`,
`Congestive_Heart_Failure`, `Coronary_Heart_Disease`, `Heart_Attack`, `Stroke`, `Cancer`,
`Trouble_Sleeping_Doc`, `Ever_Had_Drink`) and 10 continuous (`Age`, `Income_Ratio`,
`Sleep_Hours_Weekday`, `Sleep_Hours_Weekend`, `Drinking_Freq`, `Avg_Drinks_Per_Day`,
`Binge_Drinking_Freq`, `Total_Calories`, `Total_Sugar`, `Total_Caffeine`).

We report this as a case study rather than a footnote because the failure generalises to any
survey-instrument dataset whose target is a threshold on a summed scale: the leakage is invisible to a
schema review, produces a spectacular-looking result, and is only caught by noticing that the AUC is
too good.

### D. Splits

70/15/15 stratified on the target, written as three separate CSVs — **6,275 train / 1,345 validation /
1,345 test**. The three splits carry three distinct roles in the audit and are never interchanged:

- **train** — the only data any generator sees; the membership "in" set for Axis 4.
- **test** — the holdout that calibrates Axis 3, the real-data test set for utility, and the
  membership "out" set for Axis 4.
- **validation** — model selection during training, and the *population reference* for the DOMIAS
  attack. It is never in the attack's evaluation set, so the reference density is estimated from data
  disjoint from both the members and the non-members.

---

## IV. Methodology

### A. Generators

Four generators, two implemented from scratch and two external baselines. We do not claim to beat the
baselines; they situate the from-scratch models for the reader.

| | Type | Label mechanism |
|---|---|---|
| **VAE** | From scratch. BatchNorm FC encoder → $(\mu, \log\sigma^2)$, latent dim 64, hidden [256, 128]. Column-type-aware decoder: `sigmoid` on binary columns, `clamp(x,0,1)` on continuous. Loss MSE + KL with the KL weight annealed 0→1 over 50 epochs. | **Joint.** The scaled target is appended to the input matrix, so the decoder emits it; the generated value is clipped and rounded to $\{0..4\}$. |
| **CTGAN** | From scratch. WGAN-GP: generator takes noise + one-hot class condition, BatchNorm, sigmoid output; critic has **no** BatchNorm and no sigmoid (WGAN-GP requirement); gradient penalty $\lambda = 10$; 5 critic steps per generator step. | Conditional. Labels sampled from the real training distribution, fed as the condition, used directly. |
| **TVAE** | SDV `TVAESynthesizer`, embedding dim 128, 300 epochs. | Joint (SDV internal). |
| **GaussianCopula** | SDV `GaussianCopulaSynthesizer`. The cheap statistical floor. | Joint (SDV internal). |

A single decoder detail is worth stating because it cost real fidelity: using `sigmoid` on continuous
columns saturates on columns whose mass sits near the scaled range boundaries, compressing the
synthetic distribution. The column-type-aware decoder is the fix (`paper/vae_fix_note.md`).

**Post-processing, applied identically to all four generators.** Every generator emits continuous
values for categorical columns; without correction, synthetic rows are trivially separable on every
categorical column and both the fidelity and distinguishability numbers become meaningless. All
categorical columns are snapped to the nearest valid scaled level observed in the real training data.

**The VAE label mechanism is a correctness fix, not a design flourish.** An earlier version assigned
VAE labels with a random forest fit on real data, which is circular with the TSTR utility evaluation
in Axis 2. Generation now refuses any checkpoint that lacks the joint-target flag rather than silently
producing circular labels.

### B. The audit

Four axes; a generator passes only if all four pass. Every axis is calibrated against the holdout: the
question is never "is the synthetic data close to the training data" but "is it closer than a real
holdout sample is". Presented as a **calibrated instantiation of an existing evaluation checklist**
(lineage: TAPAS, SynthEval), not as a new framework.

| Axis | Gating metric | Threshold |
|---|---|---|
| 1 Fidelity | Mean per-column effect size | $\ge 0.90$ |
| 2 Utility | TSTR/TRTR macro AUC ratio (RandomForest) | $\ge 0.85$ |
| 3 NN privacy | `dcr_ratio` = DCR(syn→train) / DCR(test→train) | $\ge 1.00$ |
| 4 MIA privacy | DOMIAS density-ratio attack ROC AUC | $\le 0.55$ |

Reported but non-gating: a LogisticRegression utility variant, a classifier-based DOMIAS variant, the
classifier two-sample test (C2ST), the share of synthetic records closer to training data than the 5th
percentile of holdout distances, TPR at FPR = 0.01, and a clinical-signal check.

### C. Axis 1 — fidelity by effect size, not p-value

Per column: KSComplement $1 - D$ for continuous columns, $1 - \mathrm{TVD}$ for categorical columns
(SDMetrics convention); the axis score is the mean across the 23 feature columns.

The choice of effect sizes over p-values is load-bearing. The earlier gate was "KS $p \ge 0.05$ on
$\ge 80\%$ of columns". At 6,275 real against 1,000 synthetic records, KS power is high enough that the
test rejects on trivially small deviations — the pass rate was pinned at 0.043 for **every model and
every seed**, i.e. the metric could not have passed by construction and carried no information about
which generator was better. Raw p-values are still recorded per column as supplementary output.

### D. Axis 3 — holdout-calibrated DCR

For each synthetic record, the minimum Euclidean distance to any real training record; the same for
each holdout record. The gate is the ratio of means, $\ge 1.0$: synthetic data must sit no closer to
the training set than unseen real data does.

An absolute DCR threshold is not used, for the reason the paper is about — it conflates infidelity with
privacy. The calibrated ratio is a weaker version of the same objection, which §VI quantifies.

### E. Axis 4 — DOMIAS, and why it is validated before use

The attacker sees the synthetic data and a reference sample from the population; they do not see the
generator's parameters. For a target record $x$ the membership score is

$$\text{score}(x) = \log p_{\text{syn}}(x) - \log p_{\text{ref}}(x),$$

with $p_{\text{syn}}$ estimated by Gaussian KDE over the synthetic data and $p_{\text{ref}}$ over the
validation split. Records the generator has locally overfit sit where synthetic density exceeds
population density. Both densities are estimated on standardised features with a Scott's-rule
bandwidth. Performance is ROC AUC over a balanced set of training members against test non-members.

We report two things alongside the AUC. A **classifier-based density-ratio variant** (random forest
separating synthetic from reference, scored as log-odds) provides a check from a different model
family. And **TPR at FPR = 0.01**, because an attack that is near chance in aggregate AUC can still
succeed sharply at low false-positive rate on a small number of vulnerable records — which, given the
class imbalance in §III-B, is exactly the regime that would matter here. TPR at FPR = 0.001 is *not*
reported: with 1,345 non-members it resolves to roughly one negative example and would be noise
dressed as a number.

**No privacy number in this paper is quoted before the attack passes its controls.**
`evaluation/attack_validation.py` runs two, and exits non-zero on failure:

- **Positive control** — a maximally leaky "generator" emitting noisy verbatim copies of a
  1,000-record training subset ($\sigma = 0.02$ on continuous columns). The attack must fire:
  AUC > 0.70.
- **Negative control** — a perfectly private "generator" emitting fresh real records from half the
  test split, with non-members drawn from the other half. The attack must not fire: AUC $\in [0.45,
  0.55]$.

This is the section that the earlier version of this work did not have, and §VII describes what
happened without it.

### F. Distinguishability, reported separately

C2ST: 5-fold random-forest accuracy separating real training records from synthetic ones, balanced.
This measures sample quality, **not** membership. It is reported next to the MIA precisely so the two
cannot be conflated.

### G. Metric-gaming ablation

An opt-in rank-based marginal correction remaps each continuous column onto the real training CDF.
This raises Axis 1 sharply — and the corrected values are exact floating-point values copied out of
the training file. It is disabled by default, applied uniformly to all four generators when enabled,
audited under a separate name, and reported as a worked example of a fidelity metric bought with
verbatim disclosure rather than as a configuration we endorse.

---

## V. Experimental Setup

- **Matrix.** 4 generators × 5 seeds `[42, 43, 44, 45, 46]` = 20 runs. Every reported figure is
  mean ± standard deviation across seeds. Single-run numbers are not reported.
- **Synthetic sample size.** Each run generates as many synthetic records as there are training
  records (6,275). This is a change from an earlier configuration of 1,000, made because the DOMIAS
  density estimate is computed over exactly these records — see §VII-C and
  `evaluation/results/mia_power.csv`.
- **Determinism.** `set_all_seeds(seed)` fixes python, numpy and torch RNGs and sets cuDNN
  deterministic mode at the top of every training and generation entry point.
- **Compute.** CPU-feasible throughout; the from-scratch models use CUDA when available. No result
  depends on GPU availability.
- **Leakage dial.** A `KernelDensity` generator fit on the full training split, swept over bandwidth
  $\{0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20\}$ × 5 seeds and audited identically to the four
  generators (§VI-C). The grid was chosen to bracket the point at which `dcr_ratio` crosses 1.0:
  mean DCR runs from 0.015 at the smallest bandwidth to 0.583 at the largest, against a holdout mean
  of 0.433.
- **Reproducibility.** All source is released. Generated artifacts are not committed; the pipeline is
  reproduced by `preprocess.py` → `evaluation/attack_validation.py` → `run_all_seeds.py`, and
  `scripts/check_consistency.py` recomputes fidelity and the DCR ratio from the synthetic files on
  disk and fails if they disagree with the recorded results.

---

## VI. Results

All numbers are mean ± standard deviation over five seeds at $n = |{\rm train}| = 6{,}275$ released
records, from `evaluation/results/faa_{model}_seed{seed}.csv`, and are verified against the synthetic
files on disk by `scripts/check_consistency.py` (59 result files, all consistent).

### A. The audit across four generators

**Table 1 — the Four-Axis Audit.** Bold marks a passing gate.

| | fidelity ≥0.90 | utility ≥0.85 | `dcr_ratio` ≥1.00 | DOMIAS AUC ≤0.55 | TPR@FPR=0.01 | C2ST |
|---|---|---|---|---|---|---|
| VAE | 0.753 ± 0.005 | 0.798 ± 0.042 | **1.006 ± 0.012** | **0.502 ± 0.010** | 0.012 ± 0.001 | 0.999 ± 0.001 |
| CTGAN | 0.883 ± 0.003 | **0.935 ± 0.023** | **1.028 ± 0.009** | **0.519 ± 0.013** | 0.012 ± 0.004 | 0.993 ± 0.001 |
| TVAE | 0.891 ± 0.006 | 0.844 ± 0.050 | 0.798 ± 0.021 | **0.523 ± 0.008** | 0.010 ± 0.003 | 0.992 ± 0.001 |
| GaussianCopula | 0.834 ± 0.001 | 0.685 ± 0.026 | **1.676 ± 0.005** | **0.490 ± 0.010** | 0.008 ± 0.001 | 0.997 ± 0.000 |

**Table 2 — seeds passing each axis, out of 5.**

| | Axis 1 | Axis 2 | Axis 3 | Axis 4 | Overall |
|---|---|---|---|---|---|
| VAE | 0 | 0 | 4 | 5 | 0 |
| CTGAN | 0 | 5 | 5 | 5 | 0 |
| TVAE | 0 | 2 | 0 | 5 | 0 |
| GaussianCopula | 0 | 0 | 5 | 5 | 0 |

**No generator passes the audit**, and the binding constraint is fidelity in every case. We do not read
this as a generator ranking: a 0.90 mean-effect-size gate on 23 columns of survey data with a 76%
majority class is demanding, and the honest statement is that the gate and these generators are
mismatched. The utility column separates them usefully — CTGAN clears 0.85 in all five seeds while
GaussianCopula does not clear it in any.

Every generator passes Axis 4 in every seed, with TPR at 1% FPR sitting at or below the 0.01 rate
itself. §VII-C establishes that this is not an artifact of attack power.

The two SDV baselines carry a reporting caveat worth stating rather than burying. Until the run that
produced this table, `generate_sdv` loaded a per-seed checkpoint but let SDV sample under its own
fixed internal RNG seed, so the released rows did not depend on our seed at all. GaussianCopula's fit
is deterministic, so all five of its "seeds" were byte-identical and its spread was zero by
construction; TVAE varied only through its per-seed weights. Both baselines therefore understated
their sampling variance in every earlier version of this table. The from-scratch VAE and CTGAN were
never affected and reproduce bit-exactly across the fix.

### B. DCR rewards infidelity

Figure 1 plots `dcr_ratio` against `fidelity_score` for all 20 runs. The ordering is visible in Table
1: GaussianCopula has by far the best DCR ratio (1.676) on mediocre fidelity (0.834), while TVAE has
the best fidelity (0.891) and the *only* failing DCR ratio (0.798). CTGAN, the strongest overall
model, sits just above the gate at 1.028.

**We must state the statistics honestly.** Across the 20 runs the rank association is negative but not
significant: Spearman $\rho = -0.364$, $p = 0.115$. With four generators and near-zero within-model
variance, the effective sample size for this correlation is four points, not twenty — the seeds
replicate the model, not the relationship. **The cross-model correlation is suggestive and is not
evidence on its own.** The evidence for this claim is §VI-C, where bandwidth is the only thing that
varies and the relationship is monotone over seven points.

### C. The leakage dial

The four generators in §VI-A are four points. To separate the effect of leakage from the effect of
architecture we need a family in which leakage varies and everything else is held fixed. A Gaussian
KDE fitted on the full training split is that family: at small bandwidth it emits a training record
plus $\mathcal{N}(0,\text{bw}^2)$ — total memorisation — and at large bandwidth it is a smooth
population model. Seven bandwidths, five seeds, audited identically to the four generators.

**Table 5 — leakage dial (mean over 5 seeds).** Gates: `dcr_ratio` $\ge 1.0$, DOMIAS AUC $\le 0.55$.

| bandwidth | 0.005 | 0.010 | 0.020 | 0.050 | 0.100 | **0.150** | **0.200** |
|---|---|---|---|---|---|---|---|
| `dcr_ratio` | 0.034 | 0.067 | 0.133 | 0.345 | 0.736 | **1.054** | **1.348** |
| Axis 3 | FAIL | FAIL | FAIL | FAIL | FAIL | **PASS** | **PASS** |
| DOMIAS AUC | 0.755 | 0.753 | 0.746 | 0.709 | 0.644 | **0.601** | **0.572** |
| Axis 4 | FAIL | FAIL | FAIL | FAIL | FAIL | **FAIL** | **FAIL** |
| TPR @ FPR = 0.01 | 0.097 | 0.097 | 0.097 | 0.080 | 0.059 | 0.045 | 0.036 |
| DOMIAS AUC (clf variant) | 0.537 | 0.533 | 0.529 | 0.518 | 0.514 | 0.515 | 0.514 |
| `fidelity_score` | 0.941 | 0.941 | 0.940 | 0.932 | 0.913 | 0.897 | 0.883 |

Source: `evaluation/results/leakage_dial.csv`. Three things follow.

**(i) The DCR gate passes while membership is detectable.** At bandwidth 0.15 the synthetic data sits
*further* from the training set than the real holdout does — `dcr_ratio` 1.054, a clean pass — and the
calibrated attack reads AUC 0.601 with a TPR at 1% FPR of 0.045, four and a half times chance. At
bandwidth 0.20 the DCR ratio is 1.348, a comfortable pass by any margin a practitioner would use, and
the attack still reads 0.572. **Holdout calibration removes the crudest failure of DCR but is not
sufficient to rule out membership leakage.** This is the DCR Delusion, reproduced in a clinical survey
domain with a validated attack and a generator whose leakage is known by construction.

**(ii) The audit does catch leakage when leakage is present.** At the memorisation end both gates fire
— `dcr_ratio` 0.034 and AUC 0.755. The chance-level result for the four real generators in §VI-A is
therefore a property of those generators, not of an audit that cannot detect anything.

**(iii) The fidelity mechanism is visible within one family.** Along the dial, bandwidth is the only
thing that changes: `dcr_ratio` rises monotonically from 0.034 to 1.348 while `fidelity_score` falls
monotonically from 0.941 to 0.883. This is cleaner evidence for §VI-B than the cross-model
association, which confounds fit quality with architecture. Buying a better DCR score costs marginal
fidelity, within a single generator family, with everything else held constant.

**What this section does not establish.** DOMIAS estimates the synthetic density by KDE and this
generator *is* a KDE, so the attack and the target share a model class and the attack is favourably
placed. We intended the random-forest density-ratio variant as a cross-family check and it did not
serve: it reads 0.513–0.537 at *every* bandwidth, including 0.005, where the generator is emitting
training records with $\sigma = 0.005$ of noise. An attack that cannot detect near-verbatim
memorisation has no sensitivity to leakage, so its flat response neither confirms nor contradicts the
KDE result — it simply carries no information. The matched-model-class objection therefore stands, and
we state it as an open limitation (§VIII) rather than claim a confirmation we do not have. Closing it
requires a dial from a different model class — an overfit neural generator — or a genuine second
attack family such as Gen-LRA or ReMIA.

Figure 3 (`paper/figures/fig3_leakage_dial.pdf`) renders the first two rows of Table 5 on a twin axis
with both gates drawn and the passing-DCR / firing-MIA region shaded.

### C.1 Reconciling §VI-A and §VI-C

The two results are not in tension and the discussion must not let a reader take them as such. §VI-A
says the four generators under audit do not leak detectably. §VI-C says the DCR gate would not have
told us if they had. The audit's Axis 4 is doing the work; Axis 3 is passing alongside it without
contributing evidence. A practitioner who ran only Axis 3 — which is the common case in applied health
synthesis — would have reached the same verdict on this data for the wrong reason, and would have
reached the wrong verdict on the bandwidth-0.15 generator.

### D. Distinguishability against membership

Figure 2 places the two quantities side by side per generator. C2ST accuracy runs 0.991–0.999 — the
generators are almost perfectly distinguishable from real data — while DOMIAS AUC on the identical
data runs 0.495–0.519. Read as a privacy metric, the first would condemn every generator in the study;
the second clears all of them. They are not measuring the same thing, and the difference is not
marginal.

**Table 6 — attack validation controls.** These license every privacy number above and belong in the
main body, not an appendix. Source: `evaluation/results/attack_validation.csv`.

| Control | Construction | DOMIAS AUC | Expected | Verdict |
|---|---|---|---|---|
| Positive | Noisy verbatim copies of 1,000 training records ($\sigma=0.02$) | **0.929** | > 0.70 | fires as required |
| Negative | Fresh real records from half the test split | **0.503** | ∈ [0.45, 0.55] | correctly silent |

The negative control is the one that matters and the one the earlier version of this work lacked. A
"privacy metric" that fires against a generator with no training set to leak is measuring something
else.

### E. Leakage and ablation

**Table 7 — the derived-target leakage fix** (§III-C). Downstream RandomForest, macro OvR AUC on the
real test split.

| | Features | TRTR AUC |
|---|---|---|
| With `PHQ9_RAW` + 9 DPQ items | 33 | 0.9999 (historical) |
| After removal (this paper) | 23 | 0.7125 |

The pre-fix 0.9999 is a historical measurement with no regenerating code path in the released
artifact; only the post-fix row is reproducible from `evaluation/results/`.

<!-- TBD-optional: Table 8 — marginal-correction ablation (run_all_seeds.py --ablation-mc): Axis-1
     gain against the verbatim-copy rate. Not yet run at n = |train|; the mechanism is described in
     §IV-G and the ablation is optional for the submission. -->

---

## VII. Discussion

### A. Why the DCR ratio anti-correlates with fidelity

DCR measures distance from the training manifold. A generator that models the data well places its
samples where the real records are, which is by construction *near* training records — so it is
penalised. A generator that models the data poorly scatters samples into low-density regions and is
rewarded. Holdout calibration corrects the crudest form of this — it removes the incentive to be
uniformly far away by asking only that synthetic data be no closer than real holdout data — but it
does not remove the underlying direction of the effect, because the holdout distance is a fixed
constant while the numerator still rises with error.

The practical consequence is narrow and worth stating precisely: **a DCR-based gate cannot rank
generators, and clearing it is not evidence of privacy.** It can catch gross copying, which is a real
failure mode and worth catching — our dial confirms it fires at the memorisation end. It cannot be
used to choose between two generators that both clear it, and a generator that clears it by a wide
margin has not thereby been shown to be more private than one that barely clears it. The
bandwidth-0.20 generator clears it by 35% and leaks; GaussianCopula clears it by 66% and does not. The
margin carries no privacy information at all.

The second point is why the two properties can coexist. Nearest-neighbour distance is a *global*
average over the released records, while a density-ratio attack targets *local* excess density around
individual training records. A generator can place most of its mass away from the training manifold —
scoring well on the average — while still concentrating disproportionate mass around the particular
records it was fitted on. The two quantities are not in tension; they are answers to different
questions, and only the second is the one a privacy gate is being asked.

### B. Why a distinguishability classifier is not a membership attack

C2ST asks: given a record, was it drawn from the real distribution or the synthetic one? MIA asks:
given a real record, was this specific record in the training set? The first question is answerable
from any systematic difference between the two distributions — a slightly wrong marginal on one column
suffices — and has nothing to do with whether any individual was memorised. Reporting C2ST accuracy
under the name "MIA" therefore produces privacy alarms in exact proportion to how imperfect the
generator is, which is close to the inverse of the intended reading.

This is not a hypothetical. An earlier version of this pipeline used precisely such a classifier as its
Axis 4, reported accuracies of 0.69–0.73 against a 0.55 gate, and concluded — in every run of the
matrix — that DCR passed while membership privacy failed. That conclusion was published nowhere and
retracted internally, and it was caught by exactly one thing: adding a negative control and observing
that the "attack" fired at the same strength against a generator that emitted *fresh real records and
had no training set to leak*. An attack that cannot be turned off by the absence of leakage is not
measuring leakage. **The controls are the contribution here, not the attack.**

### C. Why the calibrated attack finds nothing, and why that is believable

A null privacy result is only as credible as the power of the attack that produced it, and there is a
specific confound to exclude. DOMIAS estimates the synthetic density from the released synthetic
records themselves; with too few of them the density estimate is too coarse to resolve local
overfitting, and the attack returns chance-level AUC regardless of whether the generator leaked. An
earlier configuration released 1,000 records against a 6,275-record training set, and the positive
control — noisy verbatim copies — is an easier target than any real generator, so it does not exclude
the confound on its own.

We therefore report attack performance as a function of released-record count (CTGAN, seed 42, all
other settings held fixed):

| Released records | 500 | 1,000 | 2,500 | 6,275 |
|---|---|---|---|---|
| DOMIAS AUC | 0.513 | 0.512 | 0.511 | **0.508** |
| DOMIAS AUC (classifier variant) | 0.504 | 0.511 | 0.499 | 0.498 |
| TPR @ FPR = 0.01 | 0.007 | 0.011 | 0.016 | 0.013 |
| C2ST accuracy | 0.954 | 0.962 | 0.981 | 0.992 |

Source: `evaluation/results/mia_power.csv`. The attack is flat at chance across a 12.5-fold range in
release size, and the low-FPR column sits at the 0.01 rate itself — a chance attack, not a weak one.
Releasing more records does not make the generator more attackable, so the null is a property of the
generator rather than of the density estimate. Every result in this paper is reported at
$n = |{\rm train}|$.

The final row is the same argument from the other direction. Distinguishability rises monotonically
with release size, because more samples make any systematic distributional error easier to detect.
Membership does not move at all. If the two quantities measured the same thing, they would move
together.

That leaves the mechanism, which we state as a hypothesis rather than a result: these generators have
low capacity relative to a 23-column dataset with a 76% majority class, and none shows the local
overfitting on rare records that density-ratio attacks target. The hypothesis makes a prediction — the
attack should succeed against a generator that *does* overfit locally — and §VI-C tests it directly.

### D. Guidance for practitioners

1. Do not gate on DCR alone. It catches copying and nothing else, and it improves as the generator
   gets worse.
2. Do not report a real-vs-synthetic classifier as a membership attack. Report it as
   distinguishability, next to a real attack.
3. Validate the attack with a negative control before quoting any privacy number. A positive control
   alone is insufficient — it tells you the attack can fire, not that it can stay silent.
4. Report TPR at a low, resolvable FPR alongside AUC, and state which FPRs your holdout size can
   actually resolve.
5. State the synthetic sample size next to every attack result. It is a parameter of the attack, not
   only of the release.

---

## VIII. Conclusion and Limitations

On NHANES mental-health survey data, four standard generators show no detectable membership leakage
under a validated density-ratio attack, and that null is stable across a 12.5-fold range in released
sample size. The same audit, applied to a one-parameter family whose leakage is known by construction,
finds a region where the holdout-calibrated DCR gate passes while the attack detects membership at
AUC 0.60 — so passing the calibrated distance check is not sufficient to conclude that a release is
private, even after the calibration that was introduced to fix the raw distance metric. Along that
family the DCR ratio and marginal fidelity move in opposite directions monotonically, which is the
mechanism: distance from the training manifold is what DCR measures and what a bad generator
maximises, while local overfitting is what an attack exploits, and the two are independent. For
practitioners the operational summary is short: report a real attack, validate it with a negative
control, and treat DCR as a check for gross copying rather than as a privacy gate.

**Limitations**, stated plainly:

- **Public data.** NHANES is already public, so the privacy stakes here are simulated. The metric
  behaviour transfers; the consequences of a breach do not.
- **No survey weights.** NHANES is a complex weighted survey and we ignore the weights throughout.
  Every distributional claim is about the unweighted sample, not the U.S. population.
- **Five seeds.** All significance testing across seeds is underpowered at $n=5$ per model and is
  reported with that caveat attached rather than as an inferential claim.
- **The dial and the attack share a model class, and our intended check for it failed.** The leakage
  dial is a kernel density estimator and DOMIAS estimates its synthetic density by KDE, so the attack
  is favourably placed against this particular generator. We reported the random-forest density-ratio
  variant as a cross-family check; it reads 0.513–0.537 at every bandwidth including total
  memorisation, which means it lacks the sensitivity to validate anything and its flat response is not
  evidence either way. **The §VI-C result should be read as: a generator of this family passes the DCR
  gate while a density-ratio attack of a matched family detects membership.** Whether the same holds
  for a mismatched attack is open. An overfit neural dial is the direct next experiment.
- **One attack family.** DOMIAS only, and its classifier variant is too weak to count as a second.
  Gen-LRA and ReMIA are cheaper than shadow-model attacks and would settle both the null in §VI-A and
  the matched-class objection in §VI-C; neither is implemented here.
- **Low-FPR resolution.** With 1,345 non-members, FPR = 0.01 is the lowest rate we can resolve
  meaningfully. Claims about very-low-FPR attack behaviour, which is where recent attacks show their
  largest gains, are out of reach at this holdout size.
- **No differentially private baseline.** There is no formal-guarantee comparator in the matrix; the
  bandwidth dial provides an empirical privacy/utility knob but not a guarantee.
- **No generator passes the audit.** All four fail the 0.90 fidelity gate. That is as much a statement
  about the gate on 23-column survey data with a dominant majority class as it is about the
  generators, and it should not be read as a generator ranking.

---

## Appendix A — Reproduction

```bash
python preprocess.py                       # NHANES XPT -> splits + scaler
python evaluation/attack_validation.py     # attack controls; gates everything downstream
python scripts/smoke_test.py               # one seed, all four generators
python experiments/mia_power.py            # attack power vs released-record count (§VII-C)
python run_all_seeds.py                    # the 4 x 5 matrix
python experiments/leakage_dial.py         # the bandwidth sweep (§VI-C)
python scripts/check_consistency.py        # recomputes metrics from disk; must exit 0
python paper/figures/make_figures.py       # figures 1-3
```

`check_consistency.py` is the guard against the failure mode that produced the retracted result: it
recomputes fidelity and the DCR ratio from the synthetic files on disk and fails if they disagree with
the recorded numbers, so a stale checkpoint cannot be silently audited.
