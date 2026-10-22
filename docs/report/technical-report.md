# ReJev Technical Report

**Lightweight decision-model post-training on MiniCPM5-2B: an independent reproduction of a Jev-style pipeline**

Version 1.2.1 · [中文版](technical-report.zh-CN.md)

## Version record

- **1.0.0** (2026-09-28) First release.
- **1.0.1** (2026-10-02) Correction.
- **1.0.2** (2026-10-04) Revision.
- **1.1.0** (2026-10-19) Added multi-seed retraining, an independent final evaluation, and probability calibration.
- **1.2.0** (2026-10-20) Reorganized around four research questions; corrected the upstream-reuse and aggregate-agreement wording.
- **1.2.1** (2026-10-21) Patch: included the plan/009 records as redacted copies in the public export set; removed account-status and credit figures from the report.

---

## Abstract

This report documents an **independent reproduction**: porting a Jev-style "structured decision" training pipeline onto MiniCPM5-2B (2B), organized around four questions — **can the pipeline be ported**, **does the classification gain survive retraining**, **are the candidate probabilities reliable and what does temperature scaling improve**, and **what boundaries do order sensitivity and out-of-distribution failure reveal**.

**tev1** here means Together's public Jev-style recipe — the repository accompanying *How to train your own Jev for \$17*. This report reproduces its **pipeline**, **not** TypeSafe's Jev product.

**Can the pipeline be ported?** Yes, and the measuring instrument is aligned with upstream. Our renderer reproduces 1,300 upstream prompts **verbatim**; our local readings of the **upstream published** Tev1-4B figures **agree with them on both sub-set aggregates (0.00pp)** — only those two aggregates were compared, **not per-item predictions**.
The in-distribution readings after porting: on a self-built, sealed holdout (1,892 items / 924 groups), the untuned base scores **51.11%** and the tuned model **80.50%** (+29.39pp, McNemar p=2.94e-96), with a **100% valid-output rate** under both constrained and unconstrained decoding. The port also left two costs: a **failed capacity probe** (LoRA rank 16 to 64 destroys the model, −23.31pp, below the base model) and the **license-remediation reading** (the cost of removing license-restricted corpora moves from −2.33pp to +0.63pp, 95% CI [−1.13, +2.37]).

**Does the gain survive retraining?** Yes, but its size varies with the seed. On a newly built and frozen test2 (3,800 items / 1,659 families; not used by any comparison arm for training, tuning, or temperature fitting), four LoRA arms differing only in training seed score **82.05 / 81.21 / 77.97 / 79.84%**, mean **80.27%**, sample SD **1.78pp**, and their same-item paired differences against base are all positive (**+29.74 / +28.89 / +25.66 / +27.53pp**, C1). The reproduction check between the historical arm and seed42 gives **−2.05pp** (interval [−3.27,−0.88]); the difference contains historical uncontrolled LoRA initialization plus environment/non-determinism and **cannot be decomposed** (C7).

**Are the candidate probabilities reliable, and what does temperature scaling improve?** The ranking information is usable, and calibration improves probability quality rather than classification. All five fine-tuned arms have lower raw NLL/Brier/ECE than base; per-arm temperature scaling lowers all three further, while the scalar temperature changes no prediction (**0/3,800**). Selective-prediction curves are reported only: for the five fine-tuned arms the calibrated − raw difference is within ±0.5pp; for base it reaches **−8.68pp** at 20% coverage (C2/C3).

**What boundaries do order sensitivity and out-of-distribution failure reveal?** Under semantic-key permutation, the fine-tuned arms' consistency is **93.71–96.16%** versus **65.08%** for base; seed42 (94.92%) and seed29 (93.71%) fall short of the pre-registered low-sensitivity bound and are labelled **uncertain**, not "low sensitivity" (C5). On a third-party exam (2,087 items), ReJev-2B's phishing recall collapses to **0** (all 1,000 positives classified as negatives) and **the untuned base does better** (57.05% vs 50.00%, p=2.58e-41): in-distribution numbers **must not be extrapolated**.

**What this report does not claim**: equivalence with the official Jev model; a reproduction of Tev1-4B (only its evaluation chain); or the extrapolation of in-distribution numbers out of distribution — on the contrary, §6.3 gives the measured out-of-distribution figures. It also does not claim that four seeds cover all training randomness, that calibration improves classification accuracy, or that permutation consistency is semantic-mechanism evidence.

A note on scope: our public release is an **explicitly enumerated** set (the authoritative whitelist is maintained in the release tooling, which is not distributed with this export; the outline below is not exhaustive). In outline: `src/`
(`align`/`data`/`train`; the weight-release toolchain under `publish/` is excluded, see below); `docs/plan/` `README` and 002–006, 009;
`docs/experiments/` `README` and exp001–004, 006–008; this report (both languages) and Figures 3–5;
the four scripts distributed with the release under `tools/` (three trigger scripts and `verify-repro-path.py`); and the build and environment files (`pyproject.toml`, `uv.lock`,
`sources.lock.json`, and the repository root files, including `LICENSE`/`NOTICE`/`CITATION.cff`/`REPRODUCING.md`).
**Three whole-item exclusions**: the `exp005` record (its third-party weight column contains readings measured here), `plan/007` and `008`
(weight-release plans), and the `src/publish/` weight-release toolchain (it publishes via HF/ModelScope and depends on files not distributed with
this export — the model card among them — and its unit-test fixtures contain **literal** malformed credential shapes that redaction would
break, so the whole directory is excluded). Data, model weights, raw text and full per-item result files are **not in this repository**; the per-item material is only the example case records described in Appendix B.5 (case id, predicted/gold key, and label). The S1 analysis record added in this revision is at `docs/experiments/exp008-s1-multiseed/`; its raw training/scoring JSONL,
adapters, and checkpoints live on a private Modal Volume and are not distributed with this repository. **The S1 analysis record, the plan/009 records, and Figures 3–5 are now in the public export whitelist** (maintained in the release tooling, not distributed with this export); the public set carrying this revision has been published with this version on GitHub.

---

## 1. Research Questions and Background

### 1.1 Research question: structured decisions with small models

Reproducing a published training pipeline is the only way to separate
what a *recipe* delivers from what its surrounding infrastructure
delivers. The task form studied here is **structured decision-making**,
not free-form generation: state + question + options → one option. The output is constrained to **one of the given options**, so "can the model talk"
never enters the evaluation — only "does it choose correctly". This form is a good
fit for studying small-model specialization under a fixed task, a fixed option set,
and a controlled budget: if the task boundary is clear, a small model with task SFT
and constrained decoding may be a better fit than a general-purpose generative model
in accuracy, latency, and cost. That motivation is **to be tested**, not a conclusion
of this report; the report gives readings for one 2B base model and one set of tasks.

The tev1 protocol (Together AI, 2026) compiles public datasets into a structured-decision
task and tunes a 4B base behind a hosted endpoint; its published figures travel with
the evaluation code, but the full chain — data build, training, scoring — has not, to
our knowledge, been independently re-run at a smaller scale. We therefore port the
pipeline onto MiniCPM5-2B (2B) and organize the evidence around four research questions:
**can the pipeline be ported**, **does the classification gain survive retraining**,
**are the candidate probabilities reliable and what does temperature scaling improve**,
and **what boundaries do order sensitivity and out-of-distribution failure reveal**.
Section 2 describes the pipeline end to end.

The reconstruction is faithful where it can be checked: our
renderer reproduces 1,300 upstream prompts verbatim, and our local readings of the
upstream-published Tev1-4B figures agree with them on both sub-set aggregates
(§3.1). What was obtained: on a sealed
holdout, the tuned model gains +29.39pp over its untuned base with zero invalid output
(§3.2). What was not: three negative results — out-of-distribution
collapse, a failed capacity probe, and a holdout-contamination correction
(§6.3, §3.4, §3.3) — which we report with the same rigor as the main result.

The 2026-10-19 extension adds a multi-seed, independent final evaluation and a
probability-calibration study. On a newly built test2 (3,800 items), four LoRA arms
differing only in training seed score 77.97–82.05% against a 52.32% base; all four
paired gains are positive (C1). Temperature scaling improves probability quality
without changing any argmax (C2/C3), and semantic-key permutation separates the
fine-tuned arms from base while leaving two seeds "uncertain" rather than "low
sensitivity" (C5). Chapters 4–6 report these results and their boundaries.

We do not claim equivalence with the official Jev model; we do not
claim to have reproduced Tev1-4B (only its evaluation chain); and we do not
extrapolate in-distribution numbers out of distribution. We also do not claim that
four seeds cover all training randomness, that calibration improves classification
accuracy, or that permutation consistency is semantic-mechanism evidence.

### 1.2 Related work

This section only positions the work against directly relevant lines; it is not a
survey, and each statement is limited to what the cited source supports.

The work reproduced here originates in the tev1 protocol (Together AI, 2026); this section
collects the upstream artifacts it depends on and the directly relevant literature.

**Task SFT and instruction tuning.** T5 casts text tasks into a text-to-text form and
fine-tunes on a multitask mixture, showing the feasibility of task SFT under a unified
interface (Raffel et al., 2020). FLAN and T0 further show that fine-tuning with natural
language instructions or multitask prompts can improve zero-shot generalization to
unseen tasks (Wei et al., 2022; Sanh et al., 2022). Our recipe belongs to this lineage:
it does not do general dialogue alignment, only supervised fine-tuning on fixed tasks.

**Small-model specialization.** The MiniCPM family studies the capability boundary of
small models through training strategy and data-model scaling, reporting that 2B-class
models can approach larger ones (Hu et al., 2024). MiniCPM5-2B is the base of this
report; we do not present "small models can approach large models" as our finding, only
as a traceable experimental condition. tev1 provides a public 4B-class Jev-style recipe
and upstream-published readings (Together AI, 2026); this report reproduces its
pipeline, not its weights or product.

**Constrained decoding and candidate scoring.** PICARD rejects inadmissible tokens at
each autoregressive step via incremental parsing, restricting output to a legal set
(Scholak et al., 2021); Outlines precomputes, per FSM state, the set of allowed tokens
and applies it as a per-token logit mask (Willard & Louf, 2023). Our primary evaluation
is not generation-then-parse but **candidate scoring**: mask the logits at the last
prompt token by the item's valid options and take a softmax. Both restrict the output
space to the option set; candidate scoring also exports probabilities for calibration.

**Distribution shift.** DomainBed shows that, under fair model selection on
distribution-shift benchmarks, empirical risk minimization remains a strong baseline
and many specialized algorithms do not reliably win (Gulrajani & Lopez-Paz, 2021).
Ovadia et al. report that accuracy and calibration degrade together as shift intensity
increases, and that standard post-hoc calibration is insufficient (Ovadia et al., 2019).
This is the background for §6.3: we report one
third-party exam, not a claim about "all out-of-distribution tasks".

**Probability calibration and selective prediction.** Guo et al. compare post-hoc
calibration methods and show that single-parameter temperature scaling improves
probability quality without changing the argmax (Guo et al., 2017); ECE and histogram
binning are standard calibration-error measures (Naeini et al., 2015; Guo et al., 2017).
Selective prediction trades coverage against error rate by ranking on confidence
(Geifman & El-Yaniv, 2017). TypeSafe's Jev defines outputs as typed probabilistic
decisions with probabilities and confidence scores, and claims that "higher confidence
means higher accuracy" (TypeSafe AI, 2026). This report does not test that product; it
borrows the "decision + calibration" problem framing and implements a reproducible
minimum with candidate softmax and a single scalar temperature. Candidate softmax is
**not** the official Jev calibration mechanism, and this report does not imply that it is.

Upstream tev1 compiles five public datasets into a `state / question / options /
answer` semantic layer using deterministic code (Together AI, 2026). We **reuse its builder**
— we pin the upstream commit, obtain its sources and run its scripts (full attribution in
§2.2.1) — and we **do not copy its builder code into this repository**;
rendering, splitting, sealing and auditing are ours.

- **tev1** (MIT, Together AI, 2026) — the task protocol spec, state
  normalisation and stratified splitting originate there; we **reuse its data builder**
  (including the vendored patch on the S0 chain, see §2.3.3), but we
  **do not copy its code into this repository**.
- **MiniCPM5-2B** (Apache-2.0) — base model
  and its official LoRA SFT recipe.
- **Tev1-4B-experimental** (Together AI, 2026) — upstream
  model whose published figures (via the hosted API) serve as the
  fidelity reference in §3.1;
  its weights are not redistributed by this work.

### 1.3 Scope and non-claims

- **Historical and new tables are separate.** The historical main result (1,892-item
  holdout; exp002 full r16) and the new test2 (3,800 items; four re-trained seeds +
  historical clean-r16 + base-C0) come from different items and different model roles;
  they are **not merged and not difference-attributed**.
- **Two historical anchors.** `clean-r16` reads **81.13%** on the `holdout1` single set
  (n=1,892) and **79.91%** on the 3,694 union `holdout1 ∪ holdout2`. The union includes
  `holdout2` (1,802 items; 1,663 de-duplicated states all hit `rejev-train`, i.e.
  **contaminated**), so the union column's cross-arm gaps are **not comparable**; the
  contamination attribute **must not be attached to 81.13%**.
- **Non-claims**: equivalence with official Jev; reproduction of Tev1-4B (only its
  evaluation chain); extrapolation of in-distribution numbers; that four seeds cover
  all training randomness; that calibration improves accuracy; that permutation
  consistency is semantic-mechanism evidence.
- **Not done**: no new Decision Head training; no Score/Noul; no base-model swap; no
  4B paired control (see #123); no retroactive confirmation of historical conclusions.

---

---

## 2. Method and Setup

### 2.1 Task formulation

The task is **structured decision-making**, not free-form generation: state + question + options → one option. The difference from a generative chat model is that the output is constrained to
**one of the given options**, so "can the model talk" never enters the evaluation —
only "did it choose correctly". We implement this with **state-machine constrained
decoding** (`prefix_allowed_tokens_fn`): the candidate set is the letter tokens of
**that item's actual options**, not a fixed A–X. This makes our constrained decoding
equivalent to upstream's "generate a regex per option list" protocol.

Every evaluation round runs both **constrained** and **unconstrained** decoding. That
the two agree bit-for-bit is itself evidence: the model is not relying on the
constraint to choose correctly.

The candidate-scoring protocol added in this revision (§2.4.2) keeps the same task
formulation and only changes the readout position from "the generated letter token"
to "the valid-option logits at the last prompt token"; both are checked against the
same option mask and label mapping.

### 2.2 Data

#### 2.2.1 Historical data chain

Upstream tev1 compiles five public datasets into a `state / question / options /
answer` semantic layer using deterministic code. We **reuse its builder** without copying
its code: we pin the upstream commit
`1dde7782382c9f49d627153759b8d1deab426ce0` (MIT), obtain its sources and run its
`fetch_sources.py` and `build_all.py` directly (the latter runs the four builders
`build_dataset`, `build_v2`, `build_v21` and `build_new_v1` in turn), producing the semantic
layer tabulated below. **The upstream builder code is not distributed with this repository**
(`resources/tev1` is an untracked read-only reference; the first build was produced by that
clone, while the 2026-10-11 isolation rebuild used a clean `git archive` of the fixed commit
as the builder input).

**Attribution, three columns**:

- **Reused (upstream, MIT, commit `1dde7782…`)** — `fetch_sources.py`, `build_all.py` and
  the four builders `build_dataset.py` / `build_v2.py` / `build_v21.py` /
  `build_new_v1.py`; the SYSTEM instruction and state-normalisation **semantics** in
  `src/data/render.py` and `src/align/v1a_check.py` likewise come from the corresponding
  upstream implementations (the string constants are verbatim, see `NOTICE` §1).
- **Modified by us (S0 chain only)** — `src/data/upstream_patches/0001-rejev-s0.patch`:
  the four builders' hard-coded SEEDs become environment variables; `build_v2.public_data`'s
  public-source barrier `forbidden` is extended to the union of all historical statehashes
  and **fails closed** when the list is missing; `build_v21`'s `group_id` incorporates the
  SEED; `build_new_v1`'s order seed becomes an environment variable. See §2.3.3.
- **Written by us** — the renderer `src/data/render.py` (base tokenizer and prompt rendering)
  and the training-only chat template `TRAIN_TEMPLATE` (`src/train/full_train.py`),
  the two stratified splits `src/data/split_holdout.py` and
  `src/train/split_holdout_v2.py`, the historical freeze lists `src/data/freeze_history.py`,
  the S0 driver `src/data/build_eval_v2.py` (quotas, sampling, split disjointness, audits
  and the RNG protocol), the verifiers (`src/data/verify_*.py`) and option permutation
  `src/data/permute_eval2.py`, plus all training, evaluation and analysis code under
  `src/train/` and `src/analysis/`, and the release tooling under `scripts/`.

| Item | Value |
|---|---|
| Built training set | 37,840 items (43,840 − 6,000 exact duplicates) / 16,929,529 tokens |
| Training set (exp002, full) | **35,948** items (37,840 − 1,892 holdout) |
| Dev set | 4,568 items |
| Sealed holdout | **1,892** items / **924** groups / 9 sources |
| Splitting | stratified by whole `group_id`, with zero-leak assertions |

**The holdout is sealed**: the splitting script writes content hashes and zero-leak
assertions over two keys (same-source detection key, group key); prompt hashes are covered
by the `cross_split_check` helper in the render module (exercised by its tests), and
identifiers by the rebuild verifier. After that the holdout may not be used for training or
tuning.

The license status of the five sources is covered in §7.2 — AG News is explicitly limited to
non-commercial use, and SST-5 has no verified redistribution authorization; we therefore
treat both conservatively and release no derived data. That handling directly caused the
source-removed retraining in §2.3.1 (exp006).

#### 2.2.2 New evaluation-side splits (S1)

This round uses **A2′**: the training side reuses historical `train-clean` (30,987
items, SHA `bce0ae24…5b664d`); only the evaluation side is rebuilt, with whole-family
(`group_id`) allocation. Data roles and sizes:

| Role | Use | Must not be used for | Size |
|---|---|---|---:|
| `train-clean` | training seeds 42/17/29/41 | — | 30,987 |
| dev2 | implementation diagnostics, throughput pilot | checkpoint selection | 500 |
| calibration2 | fitting a single scalar temperature `T_a` | final evaluation | 1,500 |
| test2 | frozen, one-shot final evaluation | training, development, tuning, temperature fitting | 3,800 (1,659 families) |
| test2-perm | option-permutation check | anything else | 3,800 |

The test2 source quota is: policy_v2 1,326, routing_v2 666, policy 165,
research_taxonomy_v21 424, mnli 554, boolq 332, banking77 333; all three splits use
the official held-out path; synthetic : public ≈ 68 : 32. The full per-source quota
and per-cell minimum family counts are in Appendix C.

**Isolation audit (S0, executed at generation time):**

1. Pairwise `statehash` and family intersections among the three splits are **0**;
   intersections with the **50,914** historical `statehash` values and **25,672**
   historical families are **0**.
2. Answer positions are constructed by within-family permutation; the surface-feature
   shortcut baseline's all-split aggregate Δ is at most 2.83pp, below the 10pp gate;
   `routing_v2`'s `longest_description` by-source Δ=13.3pp is a class prior caused by
   upstream's fixed option descriptions and is disclosed.
3. Template-residue and generator-batch-style checks have 0 failures; answer-position
   diagnostics are reported per source (boolq test2 A 57.2%, z=2.63, p=0.008).

**Disclosures (not rejection conditions):** about 32% of test2 comes from real public
corpora (mnli/boolq/banking77), and **base-model pretraining exposure cannot be ruled
out**; about 68% of synthetic sources reuse the same templates/vocabularies/rules, so
**template provenance is shared**; **near-duplicate checking was not done** — about 68%
of the new data is generated from the same templates, and an automatic near-duplicate
check would produce high false positives indistinguishable from "accidental
near-duplicates"; `research_taxonomy_v21`'s family identifier was changed to include
the SEED, so a new SEED cannot inherit old family identifiers. The full disclosure list
is in Appendix C and `009-s0-manifest.json`'s `disclosures`.

### 2.3 Training recipe

#### 2.3.1 Historical recipe

Base model `MiniCPM5-2B` (Apache-2.0, revision pinned). Training inputs are exported as
`sft` messages by `src/data/render.py`; each run then formats them with a training-specific
chat template (`TRAIN_TEMPLATE`, rebuilt from the official MiniCPM `docs/finetune/trl.md`
structure), not the model's own. We do not use the Tev1-4B tokenizer to prepare training data.
LoRA SFT, with the
hyperparameters **identical to the character** across runs **except for rank**
(exp004 uses r64/α128, see §3.4):

| Hyperparameter | Value |
|---|---|
| LoRA | r16 / α32 / dropout 0.05 / all-linear (7 target modules) |
| Learning rate | 2e-4 |
| Epochs / batch | 1 epoch · batch 2 × grad-accum 2 |
| Schedule | cosine · warmup 0.03 |
| Sequence length | 2048 |
| Seed | 42 |
| Hardware | L4 × 1 |

Differences across the three runs (**not a single-variable comparison** — the training
sets and eval sets differ too):

| Run | Training set | Steps | Duration | train_loss | Cost |
|---|---:|---:|---:|---:|---:|
| exp002 (full) | 35,948 | 8,987 | 15,111 s | 0.4569 | $5.31 |
| exp004 (rank probe) | 34,146 | 8,537 | 16,855.7 s | **0.8800** | $6.03 |
| exp006 (source-removed) | 30,987 | 7,747 | 13,774.2 s | 0.4486 | $4.97 |

**The cost figures are not the same caliber** and must not be compared or summed
directly: exp002's $5.31 is the `rejev` app's **cumulative** billing (including all of
that app's earlier exploratory spend), whereas exp004 (training $5.18 + eval $0.85)
and exp006 (training $3.85 + eval $1.12) are **per-app metered billing** covering only
that round's training and evaluation.

**exp006 is a retraining performed for license compliance**: removing AG News (1,354
items) and SST-5 (1,805 items), **3,159** items in total (9.25% of the 34,146-item
pre-removal training set under the same split),
yields an adapter with a clean license chain that can be released publicly. It doubles
as an **ablation** — source removal and re-splitting co-vary, so the difference cannot
be attributed to a single variable.

#### 2.3.2 Four-seed retraining and RNG protocol (S1)

Four arms (seeds 42/17/29/41) use the **same** `train-clean` and the **same** recipe,
changing only training randomness. The historical script's order is "LoRA random
initialization (`get_peft_model`) first, `SFTConfig(seed=...)` later"; changing only
`SFTConfig.seed` cannot control the initialization that already happened. This round
locks **RNG protocol ①**: call `torch.manual_seed(seed)` explicitly before
`get_peft_model`, so LoRA initialization also varies with seed; `SFTConfig(seed=seed)`
controls data order; no separate CUDA seed is set. This **departs** from the historical
discipline of "no extra torch/cuda seeding" and is declared in the report. It makes
the four arms' differences attributable to seed; the `hist` vs `seed42` difference
still contains historical uncontrolled initialization and environment terms and
**cannot be decomposed**.

All four arms completed 7,747 steps with health PASS and 0 failures; wall time
15,470–15,751 s, peak GPU 6.53–6.62 GB; last-50-step loss: seed42 0.268248,
seed17 0.287556, seed29 0.270453, seed41 0.215001. Adapter/config SHAs, W&B run
ids, and billing are in Appendix A and the exp008 record.

#### 2.3.3 Upstream patch and generator identity (S1)

The upstream generator is pinned to `togethercomputer/tev1 @
1dde7782382c9f49d627153759b8d1deab426ce0` (MIT). Its SEEDs are hard-coded source
constants; this round uses a **vendored patch** to change the three SEEDs that change
data instances (`build_dataset.py`, `build_v2.py`, `build_v21.py`) and one order-only
seed, and extends the public-source barrier `forbidden` to the **union of all historical
statehashes**. The patch file and SHA are stored with the S0 manifest; the read-only
`resources/tev1` reference is untouched. The new SEED is `rejev-eval2-20261017-1`;
frozen statehash/family lists and patch fingerprints are in Appendix C.

### 2.4 Evaluation protocol and statistics

#### 2.4.1 Historical evaluation protocol (preserved)

- **Primary metric**: accuracy under constrained decoding. "Valid rate" is the
  fraction of outputs landing on an option that actually exists for that item.
- **Interval estimation**: a **grouped bootstrap** resampling whole `group_id`
  clusters. The report's two sets of intervals use **different draw counts** (from
  two independent runs) — **do not conflate them**: §3.2 main table uses **2,000
  draws** (1,000 for the per-source tables emitted by
  `src/train/jev_compare.py`, `seed=20260925`), while §3.3 uses **10,000 draws**
  (see the exp006 record). The two sets also cover different evaluation sets and subsets; **defer
  to the experiment record of the section you are reading**.
  Variants within a group are highly correlated, so the
  per-item independence assumption fails; per-item Wilson intervals are reported for
  reference only.
- **Paired testing**: per-item McNemar, both arms on the same items.

#### 2.4.2 Primary protocol this round: candidate scoring

The primary classification protocol is **candidate scoring (A)**: take the LM logits at
the last valid prompt token, mask by **that item's actual valid options**, softmax to a
candidate distribution, and take the argmax as the prediction; raw logits and
probabilities are saved for calibration and selective prediction. The letter mapping is
A→54 … X→77. This protocol's argmax always lies inside the valid option set, so it
**cannot measure invalid-output rate**; coverage is reported only as the fraction of
successfully scored items, **not as a format-capability claim**. The historical
generative dual protocol (constrained/unconstrained) remains under its original
protocol; the two are not conflated.

#### 2.4.3 Metric definitions

- **Primary metric**: same-item paired accuracy difference (each arm − base-C0 on the
  same test2 items).
- **Task** = the `source` field of test2 records; **class set** = that task's fixed
  semantic label universe (`answer_key`), **not** the gold set observed in test2.
- **Semantic macro-F1**: computed only within a task's semantic class space, never
  over A/B/C… letters; a class missing from truth or prediction gets F1 **0**; classes
  are equally weighted within a task; tasks are equally weighted across tasks, with a
  sample-weighted sensitivity check.
- **Per-class recall**: same fixed class set; a class with no true samples is **NA**
  and excluded from the average.
- **Letter protocol**: diagnostic of option-position preference only, never primary
  evidence for a classification gain.

#### 2.4.4 Intervals and uncertainty

- Primary and per-task intervals use **family-clustered bootstrap by `group_id`**,
  10,000 draws, seed `20261018`, 2.5/97.5 percentiles; paired differences are
  recomputed within each resample.
- **Seed variance is computed only across the four arms seed 42/17/29/41** (4 arms =
  3 degrees of freedom); under a normal approximation and independent runs, the 95%
  interval for the true SD is about 0.57–3.73× the observed SD.
- **Training-run differences and sampling uncertainty are reported separately**, never
  merged into one interval.
- `hist` is not in the seed-variance set; its comparison with `seed42` is a
  reproduction check.

#### 2.4.5 Temperature calibration

For each arm we fit one scalar temperature `T_a` by minimizing hard-label NLL on
calibration2, grid [0.05, 20] (4,001 points, including T=1); test2 only applies the
frozen `T_a`, with no refitting or range change. We report raw/calibrated NLL, Brier,
and ECE: **top-label** ECE with 15 equal-width bins; **multi-class summed** Brier;
sample-averaged NLL. A positive scalar temperature does not change an item's argmax;
C3 asserts item-by-item that the argmax is unchanged, and any mismatch stops the
analysis. Selective prediction reports coverage–accuracy curves ranked by `p_max`;
**report only, no threshold tuning, no large-model fallback**.

#### 2.4.6 Option permutation

We generate a semantic-key permutation variant of test2 by the fixed rule
`shift = 1 + (int(sha256(id), 16) mod (n_options − 1))`, a left cyclic shift that
preserves semantic keys and descriptions and relabels A.. in the new order; after
restoring by semantic key we report consistency, flip rate, and family-clustered 95%
intervals. Effect-size bounds are locked in S0: high sensitivity = flip-rate point
estimate ≥ 0.20; low sensitivity = consistency ≥ 0.95 and interval lower bound ≥ 0.90;
otherwise **uncertain**. Permutation consistency supports only "sensitivity to option
order", **not a semantic-mechanism claim** — always picking a surface feature can also
preserve semantic-key consistency after reordering; a semantic claim needs content
interventions and surface-feature controls (not done here).

#### 2.4.7 Error and behavior analysis

The unit is the family, with a fixed single-label priority:
**state-parse failure → option-order sensitive → constant policy → label error → other**.
For each class, take the first 10 error families by ascending `group_id`, and the
smallest-id row in each family as the representative case; shared errors (all four
seeds wrong) and arm-divergent errors are counted separately; `base-C0`/`hist`-only
error families are included and attributed separately. Counts are descriptive, with no
significance claim; mechanistic statements are downgraded to candidate explanations.

### 2.5 Comparison arms and model identity

| Arm | Description | Artifact identity |
|---|---|---|
| `base-C0` | untuned MiniCPM5-2B | `openbmb/MiniCPM5-2B @ 12a3808a…d92e2` |
| `hist` | historical clean-r16 adapter, reproduction check only | adapter `294b07be…b9973e`; config `bf854ccc…` |
| `seed42` | retrained, control arm | adapter `87a07194…`; config `e82417bf…` |
| `seed17` | retrained | adapter `bbb1403e…`; config `484093d8…` |
| `seed29` | retrained | adapter `cb6ac86d…`; config `7906377c…` |
| `seed41` | retrained | adapter `8227f9ac…`; config `d8839a4d…` |

Scoring code: `eval_candidates.py a4986792…`, `candidate_math.py 0a714e17…`,
`s1_spec.py 734068aa…`; analysis code: `s1_analysis.py f93424a7…`,
`metrics.py 30d4b575…`. Appendices A and D list truncated SHAs and manifest pointers; full SHAs are in the S1/S0/analysis manifests or the private records cited in §7.3 (Audit Chain) (some not distributed with the repository).

---

---

## 3. Can the pipeline be ported?

The first question in a reproduction is whether the thing moved over is the same thing, and what the move cost. This chapter first checks the measuring instrument (rendering fidelity and aggregate agreement), then gives the in-distribution readings and the two historical anchors, and finally reports the three costs the port incurred — license remediation, a failed capacity probe, and a set of historical directions that must be read side by side rather than attributed.

### 3.1 Rendering fidelity and aggregate agreement

The first question in a reproduction is whether the thing reproduced is the same
thing. We answer with two independent pieces of evidence:

1. **Rendering fidelity**: rendering 1,300 items with Tev1-4B's own tokenizer
   reproduces the upstream `instruction/*.jsonl` prompts **verbatim (1300/1300)**;
   completions are 100% equal to `answer + <|im_end|>`.
2. **Behavioural fidelity**: against the **upstream published** Tev1-4B figures —
   **880/1000 (88.0%) + 300/300 (100.0%)**, published via the Together API — our local
   run of those weights on our reconstruction of upstream's exam showed **no deviation
   in either sub-set aggregate (0.00pp)**.
   Per NOTICE §4 we state only this **relation** ("our reading equals the published
   value"); the reading itself is **not presented as a result of ours**.

The local end-to-end run matched upstream's **two published sub-set figures** — 880/1000 for
`main` and 300/300 for `policy_transfer`; whether the per-item predictions are identical was
**not** checked. Still, every cross-model comparison that follows rests on a measuring
instrument **cross-checked against upstream's published figures**.

> Third-party weight-distribution licensing prevents us from publishing Tev1-4B's
> per-item results or other derived figures from our exam; we cite only the figures
> upstream has already published.

### 3.2 The post-port gain: historical main result and two anchors

| Arm | Accuracy | Grouped bootstrap 95% | Valid rate |
|---|---:|---|---:|
| MiniCPM5-2B base (untuned) | **51.11%** | [48.53, 53.93] | 100% |
| ReJev-2B (exp002 full r16) | **80.50%** | [78.44, 82.54] | 100% |

The gap is **+29.39pp**, McNemar χ²≈391.9, exact two-sided **p=2.94e-96**. The
constrained and unconstrained decodings agree **bit-for-bit**, with 0% invalid output.

The base model scores 51.11% on a mixed binary/multiple-choice exam, **above the
per-item random baseline of 27.66%** (the mean of 1/n over each item's actual option
count; option counts here range from 2 to 24). Three **stronger no-skill baselines**
fall far below it too — always emitting the most frequent letter (**27.91%**) or the
most frequent semantic key (**17.18%**), and per-option-count most frequent position
(**29.86%**). The exact definitions of all four, as runnable code, are in
[`src/train/noskill_baseline.py`](../../src/train/noskill_baseline.py) (reads the same
sealed holdout, with **no arguments**; reproduces all four numbers above **once both
holdouts exist on disk** — see `REPRODUCING.md` §2.4 and §2.5; the script prints an
explicit message if they do not).
⚠️ The last three baselines take the mode **on the same holdout they are scored on** —
they are post-hoc descriptive *upper bounds*, not independent no-skill baselines; only
the per-item random 27.66% involves no fitting. So **all four measured baselines
fall below the untuned base's 51.11%** (three of them being post-hoc upper bounds — even
mode-fitting on the scored labels does not reach it; we do not claim that *no*
guessing strategy could, since no pass threshold is defined and no bound over the
strategy space is proven). Tuning raises it to 80.50% while paying **no format
cost** — precisely the structural advantage of a decision model over a generative
one: under constrained decoding a format error is not "unlikely" but "impossible".

**The two historical anchors** (read **separately** from the table above; never merged):

| Basis | Model | Evaluation set | Reading | Note |
|---|---|---|---:|---|
| single set | `clean-r16` (released) | `holdout1`, n=1,892 | **81.13%** | intersection with exp002-r16 training set is 0; used as a dev set since exp004 |
| union | `clean-r16` (released) | `holdout1 ∪ holdout2`, n=3,694 | **79.91%** | union includes `holdout2` (1,802; 1,663 de-duplicated states all hit `rejev-train`, **contaminated**); cross-arm gaps not comparable |

**Do not attach the contamination attribute to 81.13%**; 80.50% is exp002 full r16 and
81.13% is the released clean-r16 — different models and training sets, not to be
conflated. The historical holdout has been consumed for development since exp004, and
that limitation is disclosed in §7.1.

### 3.3 The cost of license remediation: corrected figures

exp006's original headline comparison (source-removed vs full, 3,694 items combined)
gave **−2.33pp**. That number was later found to be a **contamination artifact**:

> The intersection of `rejev-train` (35,948) with the second holdout (1,802) is
> **1,663 de-duplicated states — every single one** (in item terms: **1,802/1,802**).
> The latter had been split out of the former. (1,663 counts *de-duplicated states*,
> 1,802 counts *items* — different units; writing "1,663/1,802" would understate the
> contamination.)
> The original comparison therefore contained 1,802 items **the model had seen during
> training**.

Recomputed on the **uncontaminated** sealed holdout (1,892 items):

| Subset | Δ (source-removed − full) | 95% CI |
|---|---:|---|
| **All (n=1,892)** | **+0.63pp** | [−1.13, +2.37] |
| Retained sources (n=1,717) | +1.69pp | [−0.06, +3.47] |
| Restricted sources (n=175) | −9.71pp | [−17.14, −2.29] |

Read this as: **on items neither model trained on, no difference was detected** (the
interval contains 0). The −9.71pp on restricted sources is consistent with
"the model never trained on these two corpora" rather than a loss of capability; but
as with the headline figure, source removal and re-splitting co-vary, so it likewise
cannot be attributed to a single variable. Weighted
consistency check: `(175 × −9.71 + 1,717 × 1.69) / 1,892 = +0.64pp`, 0.01pp away from
the **+0.63pp** headline — **both values fall inside the range allowed by the rounding
of the per-subset figures** (solving back: with the per-subset values anywhere within
their 2-dp rounding interval, the weighted mean lies in [+0.631, +0.641]), so this is
not two inconsistent computations.

**The limitation stands** (see §7.1): source removal and re-splitting co-vary, so
+0.63pp cannot be attributed to a single change.

### 3.4 The capacity probe: an informative failure

Raising LoRA rank from 16 to 64 in the hyperparameters (learning rate, schedule and
the rest unchanged, though **the data split and evaluation set also differ** — 34,146
training items and a newly cut 1,802-item eval set) was meant to answer whether the
weakness identified in the exp003 three-way comparison reflected insufficient training
or a 2B capacity ceiling.

The result: **total collapse**. On the second holdout the probe scores **27.91%**
against a base model at **51.22%** (−23.31pp, **below the untuned base**); its
`train_loss` is **0.8800** against 0.4569 at r16 — the larger
model has the *higher* loss, i.e. it **never even fit the training set**. Strictly,
"overfitting" denotes a gap between training and held-out loss, and **no run here
carves out a validation set** (only `train_loss` exists); what this rules out is the
single explanation "fit the training set better yet scored worse". That is **not
enough** to conclude "not overfitting". All 9 sources degrade or
flatline, without exception.

**Conclusion: this is an optimization failure, not a capacity limit.** The original
question **was not answered** — the probe's true capability was never measured, so it
cannot support a claim that "2B has hit its ceiling". The most plausible root cause is
that although the α/r ratio is equal in both arms (2 in each), r64 has **4×** the
trainable parameters while the learning rate was not scaled with rank; this is a
**hypothesis, not a conclusion**.

One supplementary observation: 27.91% is just **0.61pp** away from this exam's trivial
baseline of "always emit the most frequent letter" (27.30%, n=1,802), suggesting the
probe may have degenerated to a near-constant output — an **observation, not a
conclusion**; pinning down the failure mode requires the per-item predictions (on the
Volume, not analysed this round).
**Do not conflate this with the same-looking 27.91% in §3.2**: that one is a *baseline*
(n=1,892 on the sealed holdout — which also serves as the dev set from exp004 onward;
the "dev set, 4,568 items" row in §2.2.1 is exp002's, a different set), this one is r64's
*measured accuracy* (n=1,802) — same digits, different meaning.

A by-product is one data point on this base model: **this 16 → 64 attempt failed**. It
is **not** evidence that "the recipe is extremely sensitive to rank" — the training
split and eval set differ too, so the rank effect is not separated from them.

### 3.5 Directional consistency with history and boundaries (C4/C8)

- **C4 (direction)**: on test2, every fine-tuned arm is above base-C0, consistent in
  direction with the historical observation (clean-r16, 81.13% on `holdout1`). But the
  items and model roles differ, so this is **direction-only comparison**, with no
  numerical comparison or difference attribution; we do not subtract 81.13% from
  test2's 84.11% (`hist`) or 80.27% (four-seed mean).
- **C8 (precondition)**: test2 was not used for training any comparison arm; S0's
  exact statehash and registered-family intersection checks are both 0. This conclusion
  **covers only the exact statehashes and registered families actually checked**; it
  does not extend to semantic near-duplicates (not done) or pretraining exposure
  (about 32% public sources; cannot be ruled out).
- **Unanswered questions**: the 4B paired control is deferred (#123); test2 has no OOD
  arm; no Decision Head or Score/Noul training was done; no cross-task generalization
  claim. Negative/uncertain items keep their pre-written wording: seed42/29 permutation
  is uncertain; 30 single-class recall cells decline; the C7 difference cannot be
  decomposed; C1 only reaches "direction consistent, magnitude not established".

---

---

## 4. Does the classification gain survive retraining?

The historical main result came from a single training run. This chapter uses four retrainings that differ only in training seed, plus one reproduction check against the historical arm, to separate a stable property of the recipe from a single draw.

### 4.1 Multi-seed stability (C1)

The six arms' test2 main results are below. `hist` is a reproduction check only and
**not in the seed-variance set**; the four seeds' paired differences are item-by-item
differences against base-C0 on the same items, not against `hist`.

| Arm | accuracy | coverage | same-item paired diff vs base | 95% CI |
|---|---:|---:|---:|---|
| base-C0 | 52.32% | 1.0 | — | — |
| hist | **84.11%** | 1.0 | — | — |
| seed42 | 82.05% | 1.0 | **+29.74pp** | [+27.76, +31.69] |
| seed17 | 81.21% | 1.0 | **+28.89pp** | [+26.97, +30.83] |
| seed29 | 77.97% | 1.0 | **+25.66pp** | [+23.79, +27.48] |
| seed41 | 79.84% | 1.0 | **+27.53pp** | [+25.61, +29.42] |

The four seeds' mean is **80.27%**, sample SD **1.78pp**; the mean paired difference is
**+27.95pp**, SD **1.78pp**. **The positive-direction criterion holds** (all four
differences are positive; `positive_direction=True`, `zero_arm=False`). Per §2.4.4,
four arms give only 3 degrees of freedom; under a normal approximation and independent runs, the 95% interval for the true SD is about
1.01–6.64pp. This round **sets no SD threshold and no practical-improvement bound**,
and does not pick the best seed. The conclusion stops at "**direction consistent,
magnitude not established**": all four arms beat base, but the spread is not small
(seed29 is 4.08pp below seed42), so we do not claim "generally stable" or "a
practically important improvement".

![Figure 3 · test2 multi-seed stability](../paper/figures/fig4_seed_stability.png)

**Figure 3.** test2 multi-seed stability. Left: accuracy of base-C0, historical clean-r16 (`hist`; reproduction check only, not in the seed-variance set), and the four re-trained seeds; the four seed bars are the seed-variance set, and hist is not; the dashed line is base accuracy. Right: same-item paired accuracy difference against base-C0 with family-clustered 95% CI (10,000 bootstrap draws). All four differences are positive; the four-seed mean is +27.95pp with SD 1.78pp. The figure carries no practical-significance claim; see §4.1.

**Task-level consistency**: for every task, all four seeds' task-level accuracy,
macro-F1, and present-class macro recall are at least as high as base's corresponding
task-level metric; but at the **single-class recall** level, **30 cells**
(arm × task × class) fall below base (seed42/17/29/41: 7/8/5/10), e.g. seed41
`banking77/verify_my_identity` 1.00→0.25, seed42 `routing_v2/route_1` 0.91→0.64,
seed42 `research_taxonomy_v21/science-medicine` 1.00→0.75. **No per-class improvement
claim** is made; the full per-class recall is in `analysis.json` `by_source.*.*.per_class_recall`; `report_assets.py` rebuilds the table.

**Cost**: the four training runs total **\$17.11830176**; the monthly
`deployed_apps` total is **\$19.70799503** (it includes S1 training and evaluation;
the monthly breakdown is not split by app, so this total is used as an **upper bound**
for S1-specific cost); monthly metered is
**\$24.33265711** (including Volume \$4.62388736 and other non-S1 items); no Modal tasks were left running.

### 4.2 Reproduction check: `hist` vs `seed42` (C7)

On the same test2 items, `seed42 − hist` = **−2.05pp**, family-clustered 95% CI
**[−3.27, −0.88]**. The difference consists of **historical uncontrolled LoRA
initialization + environment/non-determinism**, which **cannot be decomposed**; it is
not seed variance and is not attributed to environment alone. plan/S0 contains no
pre-declared numerical range; the size of the difference is itself the reading. We
therefore **do not write "successful reproduction" or "weight-level reproduction"**;
we record the check and its non-decomposable limitation.

---

## 5. Are the candidate probabilities reliable, and what does temperature scaling improve?

A decision model is worth more than its choice: how confident it is matters too. This chapter reports the probability quality of the candidate softmax, the effect of a single scalar temperature, and the shape of the selective-prediction curves; the conclusions cover probability quality, not classification accuracy.

### 5.1 Probability quality and calibration (C2/C3)

Per-arm raw/calibrated metrics on test2 (`T_a` fitted on calibration2, only applied on test2):

| Arm | `T_a` | raw NLL | raw Brier | raw ECE | cal NLL | cal Brier | cal ECE | C3 mismatch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base-C0 | 4.9221 | 2.5433 | 0.8258 | 0.3854 | 1.1120 | 0.6098 | 0.1102 | 0 |
| hist | 1.4806 | 0.4446 | 0.2383 | 0.0586 | 0.4187 | 0.2291 | 0.0193 | 0 |
| seed42 | 1.4673 | 0.4814 | 0.2665 | 0.0678 | 0.4547 | 0.2549 | 0.0215 | 0 |
| seed17 | 1.5579 | 0.5121 | 0.2826 | 0.0817 | 0.4784 | 0.2660 | 0.0262 | 0 |
| seed29 | 1.4198 | 0.5295 | 0.2997 | 0.0596 | 0.5128 | 0.2915 | 0.0135 | 0 |
| seed41 | 1.4455 | 0.5275 | 0.2904 | 0.0695 | 0.5007 | 0.2799 | 0.0176 | 0 |

**C2**: all five fine-tuned arms have lower raw NLL/Brier/ECE than base-C0; per-arm
calibration lowers all three further (arm-level aggregate; not uniform across option-count strata — see Appendix B.2). For seed42, the raw differences vs base are
NLL −2.0619 [−2.1970, −1.9261], Brier −0.5593 [−0.5925, −0.5253], ECE −0.3176
[−0.3379, −0.2951]; after calibration, NLL −2.0887 [−2.2238, −1.9546], Brier −0.5710
[−0.6033, −0.5383], ECE −0.3639 [−0.3814, −0.3377]. The family-clustered intervals
for every arm exclude 0. All three metrics point the same way; **no single metric is
used alone**.

**Comparison basis**: the `calibrated` differences against base quoted above are `cal(arm) − raw(base)`, not `cal(arm) − cal(base)`. The within-arm `cal − raw` deltas are negative for all five fine-tuned arms (intervals exclude 0); for seed42 they are NLL −0.0267 [−0.0367, −0.0172], Brier −0.0116 [−0.0152, −0.0083], ECE −0.0463 [−0.0541, −0.0310]. These within-arm deltas are the evidence for "calibration lowers all three further"; the table's `cal` columns are level values, not differences.

**C3**: no arm's `T_a` touches the [0.05, 20] boundary; the per-arm argmax mismatch
before/after calibration is **0/3,800** for every arm. Temperature is a positive
scalar and does not change an item's argmax; the gain is in probability quality, **not accuracy**. For the five fine-tuned arms, the **calibrated − raw** selective-prediction difference is within ±0.5pp; for base it reaches **−8.68pp** at 20% coverage. The selective-prediction curves generally rise as coverage falls, though not monotonically in every interval (base calibrated drops 0.80pp from coverage 1.0 to 0.9). The table below is seed42's selective
prediction (other arms in `analysis.json` `calibration.*.selective`):

| Coverage | raw accuracy | cal accuracy |
|---:|---:|---:|
| 1.00 | 82.05% | 82.05% |
| 0.90 | 85.94% | 85.85% |
| 0.50 | 96.79% | 96.63% |

Selective curves are **reported only; no fallback threshold is tuned on test2**; the other arms are in `analysis.json` `calibration.*.selective`. Candidate softmax is not the official Jev calibration mechanism; this report does not claim to reproduce it.

![Figure 4 · test2 probability quality and temperature scaling](../paper/figures/fig5_calibration.png)

**Figure 4.** Probability quality before and after per-arm temperature scaling on test2. Top row: all six arms; bottom row: five fine-tuned arms with a zoomed x-axis (base-C0 is omitted from the bottom row so the fine-tuned differences stay visible). Open circles are raw values, filled circles are calibrated values; each pair is labelled `raw → calibrated`. The scalar temperature changes no argmax (0/3,800 per arm); the gain is in NLL/Brier/ECE, not accuracy. See §5.1.

---

## 6. What boundaries do order sensitivity and out-of-distribution failure reveal?

The first three chapters are all in-distribution. This chapter uses option permutation, error types, and one third-party exam to mark out what this recipe can and cannot support; the out-of-distribution reading is the heaviest negative result in this report.

### 6.1 Option permutation (C5)

| Arm | n | consistency | flip rate | 95% CI | label |
|---|---:|---:|---:|---|---|
| base-C0 | 3,800 | **65.08%** | 34.92% | [62.79, 67.37] | high_sensitivity |
| hist | 3,800 | **96.00%** | 4.00% | [95.32, 96.65] | low_sensitivity |
| seed42 | 3,800 | 94.92% | 5.08% | [94.15, 95.66] | **uncertain** |
| seed17 | 3,800 | 96.16% | 3.84% | [95.51, 96.79] | low_sensitivity |
| seed29 | 3,800 | 93.71% | 6.29% | [92.81, 94.59] | **uncertain** |
| seed41 | 3,800 | 95.42% | 4.58% | [94.73, 96.09] | low_sensitivity |

Read this (left column of Figure 5): the fine-tuned arms' consistency after semantic-key permutation is
clearly higher than base; but `low_sensitivity` only means consistency ≥ 0.95 with an
interval lower bound ≥ 0.90, **not** "the model reads semantic content". seed42 and
seed29 do not reach the pre-registered bound and remain **uncertain**, not "low
sensitivity". Position preference is diagnostic only: base's predicted letters skew
toward C (1,402/3,800), and the five fine-tuned arms skew toward A (1,084–1,153); the
letter distribution is itself affected by upstream option permutation and is not
evidence of a classification gain.

### 6.2 Error and behavior analysis (C6)

Family units, classified by the fixed priority in §2.4.7:

| Arm | error families / total | state-parse failure | order sensitive | constant policy | label error | other |
|---|---:|---:|---:|---:|---:|---:|
| base-C0 | 806/1,659 | 0 | 473 | 37 | 296 | 0 |
| hist | 390/1,659 | 0 | 105 | 7 | 278 | 0 |
| seed42 | 423/1,659 | 0 | 122 | 7 | 294 | 0 |
| seed17 | 428/1,659 | 0 | 102 | 8 | 318 | 0 |
| seed29 | 449/1,659 | 0 | 150 | 12 | 287 | 0 |
| seed41 | 479/1,659 | 0 | 134 | 10 | 335 | 0 |

The four seeds' **shared errors** cover **274** families and **divergent** errors
**365**; `base`-only **291**, `hist`-only **11**; `base` and `hist` both wrong while
all four seeds are right **14**; all four seeds wrong while `base` and `hist` are right
**1**. Errors concentrate in "label error" (within-sentence semantic/category
confusion) and "option-order sensitive" (the same semantic key changes prediction after
reordering); the state-parse failure and "other" categories are both 0; candidate
scoring cannot emit an invalid option, so this typology's state-parse failure count is 0. Counts are descriptive, with no significance claim; cases are drawn by
the fixed rule (first 10 per class by `group_id`, smallest id per family). The
representative case ids and predicted/gold keys are in `analysis.json` (`error_analysis.*.cases`); Appendix B.5 gives the extraction rule. **No raw text is
reproduced**.

![Figure 5 · option-permutation consistency and error types](../paper/figures/fig6_permutation_errors.png)

**Figure 5.** Left: semantic-key permutation consistency with family-clustered 95% CI; the x-axis starts at 55% and the dashed vertical line marks the 0.95 point-estimate bound; labels are the pre-registered categories (`high sensitivity` / `low sensitivity` / `uncertain`), and `uncertain` is not read as low sensitivity. Right: test2 error families by the fixed typology; state-parse failure and `other` are zero for every arm, and the number at each bar is the total error families. Counts are descriptive; no significance claim. See §§6.1–6.2.

### 6.3 Out-of-distribution: collapse

Everything above is in-distribution. To ask "how much of this recipe survives leaving
the training distribution", we use a **third-party exam of 2,087 items published
alongside its evaluation code** (`phishing` 2,000 binary / `tool_risk` 60 /
`ticket_routing` 27). Our corpora and this exam share **zero overlap** under three
keys: item identifier; content hash (NFKC + casefold + word-sequence sha256 of the
whole `state`); and a **structure-independent value bag** — every leaf value of
`state` normalised the same way, sorted and hashed under **two variants — a set
(ignores multiplicity; more aggressive) and a multiset (preserves it; more precise)**;
invariant to key renaming, and changed by any edit that survives normalisation. Both
variants return 0 hits (`src/train/verify_exam_overlap.py`,
0 hits across all six corpora). The content-based keys miss, so there is at least no
"same content, different identifier" renaming (the comparison covers `state`, not
`question` / `options`).

**Primary subset (phishing, 2,000 items, random baseline 50%)**:

| Arm | Accuracy | Recall | FPR | Confusion (TP/FN/FP/TN) |
|---|---:|---:|---:|---|
| MiniCPM5-2B base | **57.05%** | **14.20%** | 0.10% | 142/858/1/999 |
| ReJev-2B r16 (exp002) | 50.00% | **0.00%** | 0.00% | 0/1000/0/1000 |
| ReJev-2B source-removed r16 (exp006, **released**) | 50.00% | **0.00%** | 0.00% | 0/1000/0/1000 |
| ReJev-2B r64 probe (exp004) | 48.25% | 96.20% | 99.70% | 962/38/997/3 |
| sibling 4B ¹ | 50.65% | 1.30% | 0.00% | 13/987/0/1000 |

> ¹ These are the **upstream-published figures, not our measurement** (the Tev1-4B license
> is undecided, so we do not release our measured readings for the local Tev1-4B weights we
> ran — see `NOTICE` §4). We ran
> the weights locally and the three aggregate readings matched the published figures;
> the confusion matrix in
> that row is uniquely determined by the three published figures on a
> 1000/1000 balanced exam. The row evidences **scoring agreement** — not "both sides ran the
> same model" (see reading 3), and it is not a validation of our own evaluation path.

> Per-arm detail for the other two sub-tasks (`tool_risk`, 60 items; `ticket_routing`,
> 27 items) is in the per-suite table of the
> [exp007 record](../experiments/exp007-ood-exam/README.md). They are **not** a primary
> metric, and limitation (b) already states that this table cannot support
> "every out-of-distribution task collapses". **Every cell of the sibling 4B row of that
> table is an upstream-published figure or derived from published figures — none is our
> measurement**; the other arms are listed as usual.

Four readings:

1. **50.00% is not a capability score.** r16 and the released clean-r16 classify all
   1,000 positives as negatives — an entire column of the confusion matrix is zero.
   50.00% is simply what a constant prediction scores on a 1000/1000 balanced exam.
   **Accuracy must be read together with recall** in this table.
2. **The base model is better**: 57.05% vs 50.00%, McNemar **p=2.58e-41** (the base
   gets 142 items right that r16 misses; r16 gets 1 that the base misses).
   **Out of distribution, tuning made the model worse.**
   This is a **relative** reading, though: base recall 14.20% is itself far below the
   50% random baseline on this exam — "the base is better" is not "the base is good
   enough"; neither arm's positive-class recall on `phishing` is usable.
3. **The collapse is not specific to 2B capacity**: the sibling 4B (see caveat below)
   collapses the same way (recall 1.30%), and feeding upstream's own per-item artifacts
   into our analyzer **reproduces the upstream published figures bit-for-bit**
   (50.65% / 1.30% / 0.00%, confusion 13/987/0/1000).
   ⚠️ **Scope caveat**: the upstream arm ran against a **Together-hosted endpoint**
   (`hassan/Qwen3.5-4B-v1-new-…`), for which **no config or weight fingerprint is
   available**, so we call it the **"sibling 4B"**, not the same checkpoint — the
   wording `plan/005` pre-registered for the case where that comparison does not hold.
   The bit-for-bit check and the sibling 4B row of the table above are also **different
   sources**: the row cites the **upstream published** figures (we do not publish our
   own readings for that model), while the check feeds
   *upstream's* per-item artifacts into *our* analyzer. Neither is evidence for the
   other, and the check does not validate our model loading or generation path —
   it validates our scoring.
   What matches bit-for-bit are **three aggregate figures**; the exam is 1000/1000
   balanced, so recall 1.30% and FPR 0.00% uniquely determine that confusion matrix
   (it is not an independent 4-cell match). It shows our **scoring on this exam agrees
   with upstream's** — **not** that both sides ran the same model. See
   [exp007 implementation notes](../experiments/exp007-ood-exam/implementation-notes.md),
   deviation 7.
4. **Contrast: the official Jev model did not collapse** (upstream-published
   recall 42.70%).
   Same exam, same task — **the collapse is not the intrinsic difficulty of the task**;
   a non-collapsing solution exists. But the official Jev model differs from our recipe
   in both model and training, so more than the recipe varies here — this does not support
   attributing the collapse to the recipe alone.

**Two secondary observations**:

- **License remediation does not change out-of-distribution behaviour**: r16 and
  clean-r16 are **identical item-by-item** across all 2,000 (McNemar 0 vs 0, p=1).
- **r64 collapses in the opposite direction**: it classifies nearly everything as
  positive (recall 96.20% / FPR 99.70%). Its accuracy is indistinguishable from r16
  (p=0.442) but its error pattern is nearly complementary.

---

## 7. Limitations, Compliance, and Reproducibility

### 7.1 Validity and limitations

**a. In-distribution numbers must not be extrapolated.** 80.50% covers our self-built
sealed holdout and its kin. §6.3 gives the direct out-of-distribution reading: **collapse**. Any
claim that "the model can handle X", where X is outside this exam's distribution,
lacks evidence.

**b. Scope of the collapse reading.** It describes this recipe's behaviour on **one
binary sub-task of one exam**. The same model on another task within the same exam
(`tool_risk`) does not collapse (r16 scores 85.00% against the base's 60.00%). So
§6.3 **cannot** support "ReJev-2B is useless on any out-of-distribution task".

**c. The exam is self-built, and its composition is not neutral.** The primary exam
was constructed and balanced by this project, and contains a home-field component.
The third-party exam (§6.3) partially offsets this, but its primary subset is a single
binary task.

**d. The cost of having no group key.** Items on the third-party exam are independent,
with no `group_id`, so there is no grouped bootstrap and the per-item intervals are
**optimistic**. The primary exam has 924 groups and is better in this respect.

**e. Protocols are not strictly comparable.** The rows in §6.3 come from different
decoding implementations (upstream models use their own native interfaces and regexes;
we use state-machine constraints). Read the conclusions as "same exam, approximately
the same protocol".

**f. Pretraining exposure cannot be excluded.** The exam's phishing material is
publicly downloadable synthetic email; we cannot rule out that any model saw it during
pretraining. Likewise we cannot prove the base model never saw any content of our
primary exam — only that **our splitting** did not leak.

**g. Gold labels come from construction rules.** The exam's labels are determined by
its dataset construction rules, not independent human annotation. Upstream itself
notes that "a simple URL heuristic gets 91.6%" — that number is a **property of the
dataset**, not a deployment-grade security guarantee.

**h. Paired intervals and per-item independence.** Per-item McNemar assumes
independent items; variants within a group are highly correlated, so all per-item
p-values should be read as **descriptive**, with the grouped intervals deciding.

**i. Single runs.** Historical runs were each a single seed, single run; the four-seed retraining added in this revision
provides the first run-to-run variance reading; but four arms give only 3 degrees of freedom,
and under a normal approximation and independent runs, the 95% interval for the true SD is about 0.57–3.73× the observed SD — 1.78pp is
not a precise population SD. `hist` is not in the seed-variance set.

**j. No final evaluation set is untouched by the entire recipe-selection process.** The holdout is sealed, but the
pipeline that produced it (including hyperparameter selection) was developed with
reference to it; on the present material this cannot be ruled out. The new test2 added
in this revision was frozen before S1 and did not participate in training, tuning, or
temperature fitting, so it is untouched by this round's four arms and base-C0; however,
the historical recipe's hyperparameters were selected with reference to the old
holdout, so the new test2 supports only "same-condition comparison within this round's
recipe", not retroactive independent confirmation of the whole recipe-selection process.
The old holdout's consumption limit remains.

**k. "No difference detected" is not "equivalent".** §3.3 and any comparison whose
interval contains 0 show only that there is no evidence at this exam's scale — not
that the two are the same.

**l. No untuned-4B control arm.** In §6.3, the sibling 4B is described by upstream as a 4B base plus the tev1 recipe,
and we did not run that base's untuned version on the same exam. The contributions of
base and recipe therefore cannot be separated: the reading supports "the collapse is
not specific to 2B capacity" but **not** "the recipe caused the collapse".

**m. Task-level macro metrics vs single-class recall.** The four seeds agree in
direction at the task-level macro metrics, but **30 cells** (arm × task × class) have
lower single-class recall than base. Task-level macro metrics can hide minority-class
degradation; we report per-class readings and make no "every class improved" claim.

**n. Permutation measures order sensitivity only.** Semantic-key permutation preserves
content and changes only order; high consistency means the prediction is insensitive to
order perturbation, **not** that the model reads semantic content — a surface strategy
such as always picking the longest option can also preserve semantic-key consistency
after reordering. A semantic-mechanism claim needs content interventions and
surface-feature controls.

**o. Calibration boundaries.** This round fits **one global temperature per arm**, not
per task; temperature is a positive scalar and does not change the argmax, so the gain
is in NLL/Brier/ECE, **not** classification accuracy; selective-prediction curves are reported only; calibration does not consistently improve selective-prediction accuracy. Candidate
softmax is also not the official Jev calibration mechanism.

**p. Public-source pretraining exposure and near-duplicates.** About 32% of test2 comes
from real public corpora; base-model pretraining exposure cannot be ruled out. Near-
duplicate checking was not done (template provenance causes high false positives).
"Zero intersection" covers only the exact statehashes and registered families actually
checked.

**q. Two historical anchor bases.** 81.13% is the `holdout1` single set (n=1,892);
79.91% is the 3,694 union `holdout1 ∪ holdout2`, which includes contaminated
`holdout2`; cross-arm gaps are not comparable. Do not attach the contamination
attribute to 81.13%, and do not compare 81.13% numerically with test2 readings.

**r. 4B control deferred.** No 4B download, training, or evaluation this round; the
limitation remains (#123).

---

### 7.2 Licensing and compliance

Release decisions are made per **data source license status**, not per "is it publicly
available":

| Source | License status | Training | Redistributing derived data |
|---|---|---|---|
| MultiNLI | Mixed (predominantly OANC EULA, not CC) | ✅ | per terms |
| BoolQ | CC BY-SA 3.0 | ✅ | per terms (ShareAlike judged separately) |
| Banking77 | CC BY 4.0 | ✅ | per terms (attribution) |
| **AG News** | Card limits use to "research purposes … any other non-commercial activity" | ✅ | ❌ |
| **SST-5** | **Verified**: no license metadata — all four layers of the provenance chain (the Stanford page / the CS287-HW1 repository / the AcademiaSinicaNLPLab repository / the SetFit-sst5 dataset card) are silent. **Not verified**: the paper text and the work-level authorization chain are not resolved (the Stanford page was read and carries no terms). **Handling**: absence of a declaration is treated conservatively as absence of authorization, so nothing is released | ✅ | ❌ |

**Determination**: all five sources may be used for local training; **we release no
derived data containing the two restricted sources** (AG News is explicitly non-commercial;
SST-5 has no verified redistribution authorization). The publicly released adapter therefore
comes from a source-removed retraining (30,987 items, 3,159 removed).

**Synthetic portion**: upstream generates synthetic samples with purely deterministic
code (zero external data input, zero API calls, zero model generation) and **never
declared a license for them**. We are the generator, not a recipient.

**Tev1-4B weights**: their license status is undecided, so we **publish no copy of
those weights and no per-item results**, using them only for internal research
comparison; §3.1 and §6.3 cite only figures upstream has already published.

**S1 evaluation side**: synthetic sources (policy/policy_v2/routing_v2/research_taxonomy_v21) are deterministically generated by the vendored upstream
generator (the upstream builder with a patch applied to a temporary copy — attribution in §2.2.1); public sources (mnli/boolq/banking77) retain their original licenses, and
no **full** derived per-item data is released; the only per-item material shipped is the example case
records described in Appendix B.5 (case id, predicted/gold key and label). **Mixed training data stays local**, not
committed or uploaded; this report publishes only aggregate metrics, case ids, and
predicted/gold keys, not raw text. The upstream patch changes only SEEDs and the
`forbidden` barrier and is recorded as MIT-compatible; the read-only `resources/tev1`
reference is untouched.

**Unverified items (recorded as-is)**: one MultiNLI sub-item's license could not be
verified; SST-5's paper text and work-level authorization chain are not resolved. Our releases
claim only "handled according to the above determinations", not that verification is
exhaustive.

---

### 7.3 Reproducibility audit chain

Conclusions are tied to exact code, weights, data and environment versions, to the
extent those were recorded — the gaps are listed below.

**Code**: training commits are recorded in the experiment records for exp001
(`326ff04`) and exp002 (`9a40878`); they are **not recorded** for exp003 / exp004 /
exp006 / exp007 (see "Known gaps" below). The model-card build and push scripts live
in `src/publish/` (the weight-release toolchain, excluded from the public export — see
the scope note above).

**Models and data (content fingerprints)**:

| Artifact | Identifier |
|---|---|
| Base model | `openbmb/MiniCPM5-2B` @ `12a3808a956f869c767195e9266b59c4d21d92e2` |
| Released adapter (clean-r16) | SHA256 `294b07be…b9973e` (pure LoRA adapter, not merged weights, 7 target modules) |
| exp002 adapter (unreleased, comparison arm) | SHA256 `c82857d2…` |
| Training set (clean, 30,987) | SHA256 `bce0ae24…5b664d` |
| Evaluation set (3,694) | SHA256 `8423a838…720f96` (asserted per arm) |
| Sealed holdout (1,892) | SHA256 `2601d649…9063` |
| Upstream build artifacts | **bit-for-bit identical** to the upstream dataset manifest |

**Convention**: SHA256 values in this table are **truncated** (first 8 hex digits …
last 6) for human comparison. **Not all full values are in this repository**: the
training set, eval set and sealed holdout have their full hashes recoverable from shipped
code (`clean_train.py` / `jev_compare.py` and similar); **the released adapter's full
hash lives in the weight-release toolchain, which is not part of this export — only the
truncated form above is published**; the **exp002 adapter (an unreleased comparison arm)
is present only as a truncated prefix — its full hash lives in the Modal Volume manifest,
which is not released with this repository**. The base-model revision is given in full 40 hex digits
because it is an external identifier that must be quotable verbatim.

**S1 chain (added 2026-10-19)**:

| Item | Identifier |
|---|---|
| Upstream generator | `togethercomputer/tev1 @ 1dde7782382c9f49d627153759b8d1deab426ce0` |
| Upstream patch | `src/data/upstream_patches/0001-rejev-s0.patch`; fingerprint in S0 manifest |
| New SEED | `rejev-eval2-20261017-1` |
| Historical statehash list | 50,914 entries; `src/data/frozen/historical-statehashes.txt` |
| Historical family list | 25,672 entries; `src/data/frozen/historical-families.tsv` |
| S0 manifest | `docs/plan/009-s0-manifest.json` source-of-record SHA256 `5c33d48b…44f418`; public copy `504a7fcd…64b17` |
| S1 manifest | `docs/plan/009-s1-manifest.json` source-of-record SHA256 `41f1a53e…87f6e3`; public copy `3787d14f…db23b7` |
| Four-arm adapter/config | see §2.5 and the exp008 record |
| Scoring code | `eval_candidates.py a4986792…`, `candidate_math.py 0a714e17…`, `s1_spec.py 734068aa…` |
| Analysis code | `s1_analysis.py f93424a7…`, `metrics.py 30d4b575…` |
| Analysis artifacts | `analysis.json` `695e9054…a84ff`, `analysis.md` `97e4aba4…e28f6`, `analysis-manifest.json` source-of-record `f5dee8f5…8ef18`, public copy `18d724cc…b52e4` |
| 24 scoring-input SHAs | `analysis-manifest.json` `inputs`; `load_all_validated` passed over 57,600 records |

**The evaluator self-checks**: our out-of-distribution analyser feeds the per-item
outputs shipped with the upstream exam into itself and must **reproduce the upstream
published figures bit-for-bit**, refusing to produce output otherwise. This check
actually caught an implementation error during this round (the positive class was
being judged by letter label rather than semantic key — which letter carries the
positive class depends on option order, so a label-based judgement **silently**
misjudges items whose option order differs).

**Known gaps (recorded as-is)**:

- One training run's manifest records `code_commit` as a placeholder rather than a
  real commit hash; that run's code version **cannot be self-certified** from its
  manifest.
- **No code commit is recorded for exp003 / exp004 / exp006 / exp007** (exp001 and
  exp002 do record one). Those four rounds' readings therefore **cannot be reproduced
  from a specific commit**; what is reproducible is the scripts and configuration as
  listed in each record, not the code snapshot at the time.
- The end-to-end reproduction chain has **not been re-run in a clean environment**
  (it needs an HF token and GPU credit). Every step in the reproduction document
  carries a verification status — do not read "documented" as "verified".
- **S1's `analysis-manifest.json.created` is hard-coded by the analysis source to
  `2026-10-18`, while the actual run date is 2026-10-19**; the frozen analysis source
  was not modified, and the actual run time plus this directory's SHA are authoritative.
- S1's raw training/scoring JSONL, adapters, and checkpoints live on a private Modal
  Volume and are **not distributed with the repo**; W&B run ids and billing are
  transcriptions, not offline-verifiable.
- S1 manifest `notes[0]` is a pre-seed42 base-structure snapshot; `runs[]` already
  registers all four finished arms. Do not read that note as the current state (see
  the exp008 record §8).

---

### 7.4 Reproduction guide

Full steps, verification tiers and check commands are in
[REPRODUCING.md](https://github.com/Joe-rq/ReJev/blob/main/REPRODUCING.md) (shipped with this repository). Three essentials:

1. **`uv sync --group align`** — the data-chain and alignment-check dependencies are
   not in uv's default group; omitting this yields an empty environment.
2. **Use `deploy` + `spawn` for training and full evaluation, not a direct
   `modal run`.** The latter binds the local process to the remote job, so killing it
   locally sends a cancellation that **kills the running job**.
3. **Verify the sealed-set fingerprint for every arm** — a row-count assertion cannot
   catch "a different file with the same number of rows".

**S1 entry points** (per the manifest and exp008 record):

- S0 rebuild/audit: `src/data/build_eval_v2.py`, `src/data/verify_eval2.py`,
  `src/data/verify_patch_barrier.py`, `src/data/verify_rebuild_evidence.py`; read the
  frozen values in `docs/plan/009-s0-manifest.json` first.
- Training: `src/train/multiseed_train.py` (`modal deploy` + `.spawn()`); the four
  seeds and RNG protocol ① are in the S1 manifest.
- Scoring: `src/train/eval_candidates.py` (six arms × dev2/calibration2/test2/test2-perm).
- Analysis: `uv run python src/analysis/s1_analysis.py --scores-dir <scores> --out <out>`;
  report tables: `uv run python src/analysis/report_assets.py --analysis <analysis.json> --out <dir>`.
- Figures 3–5: `uv run --group figures python src/analysis/s1_figures.py --analysis <analysis.json> --out docs/paper/figures --force` (the `figures` dependency group is required; `--force` is required for the frozen delivery directory).
- Paper chain: the two paper check scripts (build and consistency guards; not distributed with this export).

**Not recomputable**: raw training/scoring JSONL, adapters, and checkpoints on the
Modal Volume; W&B runs; billing is a transcription of `modal billing summary --json`.
The report's aggregate numbers can be recomputed from `analysis.json`, but **full** per-item
records are not in the repository; the only per-item material shipped is the example case
records described in Appendix B.5.

---

### 7.5 Recorded as-is

This report and the work behind it **were not plain sailing**. The following is
recorded rather than smoothed over, because it is as informative as the results:

- **One data egress**: a comparison round sent an exam to a third-party API beyond the
  authorization in force at the time. It was dispositioned by the owner as
  "recorded, no further action", and constitutes **no** advance authorization for
  similar behaviour.
- **Three false claims**: review found three instances of "claimed changed, actually
  not changed". All are on record and fixed. They produced this project's practice
  that **criteria must not rest on human assertion but on mechanical checks** — for
  example, every redaction rule in the release exporter must match **at least once**,
  so claiming a change that was not made becomes a failure rather than a sentence.
- **One review finding that corrected a conclusion**: a review spotted an arithmetic
  anomaly (the model card said training used 35,948 items, while 30,987 + 3,159 =
  34,146 and 34,146 + 1,802 = 35,948) and dug out the holdout contamination in §3.3.
  The lesson: **keep every number next to its provenance and the anomaly surfaces on
  its own.**
- **One index gap, since fixed**: a plan index was missing one row (a pre-existing
  gap, explicitly recorded as "not to be fixed in passing"); it has since been filled in.
- **Two rounds of inconsistent figures**: earlier experiment records retain
  pre-correction numbers (as described in §3.3); this report uses the corrected figures throughout, and the original text in those records is left unchanged with
  the discrepancy recorded.
- **S1's seed41 and single-class declines**: seed41 has the lowest last-50-step loss
  (0.2150) but the third-highest test2 accuracy (79.84%); across the four seeds, 30
  single-class recall cells (7/8/5/10) fall below base. All four arms completed; per
  the pre-registered discipline they are **not retrained or dropped**, and are kept as
  research results.
- **S1 uncertainties**: seed42/seed29 permutation consistency does not reach the
  pre-registered low-sensitivity bound and is labelled **uncertain**; the C7
  `seed42−hist` difference contains uncontrolled initialization and environment terms
  and **cannot be decomposed**; no "successful reproduction" claim.
- **S1 provenance defect**: `analysis-manifest.json.created` is hard-coded to
  `2026-10-18`, while the actual run date is `2026-10-19`; the frozen analysis source
  was not modified, and the defect is recorded in §7.3 (Known gaps) and the exp008 record.

---

---

## 8. Conclusion

**What the evidence supports.** The pipeline ports onto a 2B local model, and the measuring instrument is aligned with upstream afterwards: rendering is reproduced verbatim and the two sub-set aggregate scores agree with the upstream published values. That proves **scoring agreement**, not "the same model" — per-item predictions were never compared. Under that calibrated instrument the port itself holds: the same data and recipe give a +29.39pp in-distribution gain on 2B with zero invalid output, and four independent seeds keep the gain positive (paired differences +25.66 to +29.74pp, SD 1.78pp).

**The gain is not lossless.** The best and worst seeds differ by 4.08pp, and the reproduction check against the historical arm gives −2.05pp with a difference that contains uncontrolled initialization and environment terms and cannot be decomposed. One can therefore say "the recipe is stably better than its base on this model", not "every run reproduces the same number". The capacity question is likewise **unanswered**: the r64 probe was an optimization failure, not a capacity limit.

**What the probabilities can and cannot support.** The ranking information in the candidate softmax is useful: temperature scaling improves NLL/Brier/ECE without changing any argmax, and the selective-prediction curves generally rise as coverage falls. But what improves is **probability quality**, not **classification accuracy**; reading calibrated confidence as "a more accurate model" is a misreading.

**Where the boundaries are.** Order sensitivity and the out-of-distribution reading are the heaviest negative results here. The fine-tuned arms are markedly less sensitive to option order than base, yet two seeds still miss the pre-registered low-sensitivity bound; and on the third-party exam, tuning drives phishing recall to 0 — worse than the untuned base. Every positive conclusion in this report therefore holds only **in-distribution, under a fixed task form and a fixed option set**; outside that condition the observed behaviour is collapse rather than mild degradation. Other sub-tasks on the same exam do not collapse, so this is not "the recipe fails on every out-of-distribution task".

**What was not obtained.** The capacity limit was not answered; "equivalent to the official Jev" was not tested; "permutation consistency equals a semantic mechanism" was not shown; four seeds do not cover all training randomness; and the out-of-distribution failure mode was observed on one exam and one binary sub-task only.

**Scope.** These conclusions correspond to MiniCPM5-2B, the pinned open and synthetic corpora, and the templates and evaluation protocol listed in this report. They do not transfer to other base models, other task forms, or other distributions, and they are not a general verdict on "small model + task SFT + constrained decoding" — the report gives readings for one 2B base model and one set of tasks.

---

## Citations and Acknowledgements

- **[tev1](https://github.com/togethercomputer/tev1)** (MIT) — the task protocol
  spec, state normalisation and stratified splitting originate there. We **reuse its
  data builder**, but we do not copy its code into this repository
  (the three-column attribution is in §2.2.1).
- **MiniCPM5-2B** (Apache-2.0) — base model and its official LoRA SFT recipe.
- **TypeSafe AI, "Introducing System One Models & Jev"** (2026-09-15,
  https://typesafe.ai/blog/introducing-system-one-models-and-jev) — the typed
  probabilistic decisions and calibrated-probabilities framing; this report does not
  test the product.
- **Together AI, "How to train your own Jev for \$17"** (2026,
  https://www.together.ai/blog/how-to-train-your-own-jev) and the tev1 repository —
  upstream recipe, data sizes, and published readings.
- Raffel, C. et al. (2020). *Exploring the Limits of Transfer Learning with a Unified
  Text-to-Text Transformer.* JMLR 21(140):1–67. arXiv:1910.10683.
- Wei, J. et al. (2022). *Finetuned Language Models Are Zero-Shot Learners.* ICLR 2022.
  arXiv:2109.01652.
- Sanh, V. et al. (2022). *Multitask Prompted Training Enables Zero-Shot Task
  Generalization.* ICLR 2022. arXiv:2110.08207.
- Hu, S. et al. (2024). *MiniCPM: Unveiling the Potential of Small Language Models
  with Scalable Training Strategies.* arXiv:2404.06395.
- Scholak, T., Schucher, N., & Bahdanau, D. (2021). *PICARD: Parsing Incrementally for
  Constrained Auto-Regressive Decoding from Language Models.* EMNLP 2021,
  pp. 9895–9901. DOI 10.18653/v1/2021.emnlp-main.779; arXiv:2109.05093.
- Willard, B. T., & Louf, R. (2023). *Efficient Guided Generation for Large Language
  Models.* arXiv:2307.09702.
- Gulrajani, I., & Lopez-Paz, D. (2021). *In Search of Lost Domain Generalization.*
  ICLR 2021. arXiv:2007.01434.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern
  Neural Networks.* ICML 2017, PMLR 70:1321–1330. arXiv:1706.04599.
- Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015). *Obtaining Well Calibrated
  Probabilities Using Bayesian Binning.* AAAI 2015, pp. 2901–2907.
  DOI 10.1609/aaai.v29i1.9602.
- Ovadia, Y. et al. (2019). *Can You Trust Your Model's Uncertainty? Evaluating
  Predictive Uncertainty Under Dataset Shift.* NeurIPS 2019. arXiv:1906.02530.
- Geifman, Y., & El-Yaniv, R. (2017). *Selective Classification for Deep Neural
  Networks.* NeurIPS 2017, pp. 4878–4887. arXiv:1705.08500.
- The authors and maintainers of the five public datasets; license status in §7.2.
- Per-item source locking is in `sources.lock.json`; third-party notices in
  [NOTICE](https://github.com/Joe-rq/ReJev/blob/main/NOTICE).

**This repository contains no data, model weights, raw text or complete per-item result files (only the example case records (case ids, predicted/gold keys and labels) described in Appendix B.5), and
distributes no Tev1-4B assets.**

---

---

## Appendix A · Full Configuration and Identity

### A.1 Training configuration

| Item | Value |
|---|---|
| Base model | `openbmb/MiniCPM5-2B @ 12a3808a956f869c767195e9266b59c4d21d92e2` |
| LoRA | r16 / α32 / dropout 0.05 / all-linear (7 target modules) |
| Learning rate | 2e-4 |
| Epochs / batch | 1 epoch · batch 2 × grad-accum 2 |
| Schedule | cosine · warmup 0.03 |
| Sequence length | 2048 |
| Precision | bf16; `assistant_only_loss`; `packing=False` |
| Hardware | L4 × 1; single training ≤ 6 hours |
| Platform | Modal `deploy` + `.spawn()` |
| RNG protocol | ①: `torch.manual_seed(seed)` before `get_peft_model`; `SFTConfig(seed=seed)`; no separate CUDA seed |
| Seeds | 42 / 17 / 29 / 41 |

### A.2 Four-arm run list

| seed | steps | wall | peak GPU | last-50 loss | cumulative | adapter SHA256 | config SHA256 | W&B | metered |
|---:|---:|---:|---:|---:|---:|---|---|---|---:|
| 17 | 7,747 | 15,740.6s | 6.53GB | 0.287556 | 0.463810 | `bbb1403e…` | `484093d8…` | `ecd5pw27` | $4.30145456 |
| 29 | 7,747 | 15,751.0s | 6.60GB | 0.270453 | 0.470156 | `cb6ac86d…` | `7906377c…` | `d8cku50z` | $4.31092469 |
| 41 | 7,747 | 15,470.1s | 6.61GB | 0.215001 | 0.490325 | `8227f9ac…` | `d8839a4d…` | `tcyn90l0` | $4.23096959 |
| 42 | 7,747 | 15,646.2s | 6.62GB | 0.268248 | 0.454274 | `87a07194…` | `e82417bf…` | `yfi4gjnp` | $4.27495292 |

### A.3 Scoring, calibration, and error protocols

- Scoring: `s1-candidate-v2`; logits at the last prompt token → valid-option mask →
  softmax; A→54…X→77; raw logits/probs saved.
- Calibration: per-arm `T_a`, fitted on calibration2, stable log-softmax over logits;
  grid [0.05,20], 4,001 points; applied to test2 only.
- Intervals: family-clustered bootstrap 10,000 draws, seed `20261018`, 2.5/97.5 percentiles.
- Permutation: `shift = 1 + (int(sha256(id),16) mod (n_options-1))`; effect-size bounds
  in §2.4.6.
- Errors: family unit; priority "state-parse failure → order sensitive → constant policy →
  label error → other"; first 10 by ascending `group_id` per class, smallest id per family.

## Appendix B · Per-task and Stratified Tables

### B.1 Six arms × seven tasks (test2)

The full table is in `analysis.json` `by_source` and exp008 `analysis.md`. Below are
the key accuracy readings for the four seeds and base/hist:

| Task | n | base-C0 | hist | seed42 | seed17 | seed29 | seed41 |
|---|---:|---:|---:|---:|---:|---:|---:|
| banking77 | 333 | 73.87% | 92.49% | 90.99% | 91.59% | 91.89% | 91.59% |
| boolq | 332 | 77.11% | 86.45% | 84.04% | 88.25% | 86.14% | 84.94% |
| mnli | 554 | 55.78% | 85.74% | 85.74% | 84.30% | 86.10% | 83.94% |
| policy | 165 | 50.91% | 100.00% | 100.00% | 100.00% | 99.39% | 100.00% |
| policy_v2 | 1,326 | 38.61% | 81.45% | 77.38% | 74.13% | 69.83% | 71.57% |
| research_taxonomy_v21 | 424 | 96.70% | 99.06% | 99.06% | 100.00% | 99.76% | 98.11% |
| routing_v2 | 666 | 25.68% | 69.22% | 67.57% | 67.42% | 57.21% | 67.87% |

Full CIs, macro-F1, macro recall, and confusion matrices are in `analysis.json` `by_source`;
per-source paired differences are in `seed_analysis.per_seed.<seed>.by_source_paired_diff.<source>`;
`report_assets.py` rebuilds the table structure.

The task-equal and sample-weighted macro-F1 aggregates are in `analysis.json` `by_source.<arm>._task_equal_macro_f1` and `by_source.<arm>._sample_weighted_macro_f1`; both put all four seeds above base-C0 (the two weightings swap seed42 and seed17).

### B.2 Option-count strata (seed42)

| Options | n | raw acc | raw ECE | cal acc | cal ECE |
|---:|---:|---:|---:|---:|---:|
| 2 | 332 | 84.04% | 0.1141 | 84.04% | 0.0744 |
| 3 | 2,045 | 81.47% | 0.0795 | 81.47% | 0.0312 |
| 4 | 122 | 93.44% | 0.0441 | 93.44% | 0.0644 |
| 5 | 666 | 67.57% | 0.0724 | 67.57% | 0.0412 |
| 6 | 116 | 87.93% | 0.0777 | 87.93% | 0.0723 |
| 8 | 201 | 95.52% | 0.0332 | 95.52% | 0.0311 |
| 12 | 106 | 100.00% | 0.0030 | 100.00% | 0.0087 |
| 24 | 212 | 98.58% | 0.0129 | 98.58% | 0.0106 |

The full six-arm option-count strata are in `analysis.json`
`calibration.*.by_option_count`.

This is an arm-level statement: within option-count strata the sign is not uniform. For seed42, the calibrated NLL and ECE are slightly higher than raw in the 4-option and 12-option strata (NLL 0.1997→0.2013 and 0.0034→0.0094; ECE 0.0441→0.0644 and 0.0030→0.0087).

### B.3 Selective prediction (seed42)

| Coverage | raw accuracy | cal accuracy |
|---:|---:|---:|
| 1.00 | 82.05% | 82.05% |
| 0.90 | 85.94% | 85.85% |
| 0.50 | 96.79% | 96.63% |

Other coverage levels (0.05/0.10/0.20/0.30/0.70) and the other five arms are in
`analysis.json` `calibration.*.selective`.

### B.4 Single-class recall declines (30 cells)

The full per-class table is rebuilt by `src/analysis/report_assets.py` from
`analysis.json` `by_source.*.*.per_class_recall`; the report gives the counts in §4.1.

### B.5 Case extraction

For each class, take the first 10 error families by ascending `group_id`, and the
smallest-id row in each family; the `group_id`, `id`, `pred_key`, `gold_key`,
`pred_label`, and `gold_label` of representative cases are in `analysis.json`
`error_analysis.*.cases`. No raw state/question text is reproduced.

## Appendix C · Data Construction, Isolation Audit, and Disclosures

### C.1 Upstream patch and frozen lists

- Upstream `togethercomputer/tev1 @ 1dde7782…` (MIT).
- Vendored patch: `src/data/upstream_patches/0001-rejev-s0.patch`; modifies the SEEDs
  in `build_dataset.py`, `build_v2.py`, `build_v21.py` and one order-only seed, and
  extends `forbidden` to the union of all historical statehashes.
- Frozen historical statehashes **50,914**, families **25,672**; SHAs in the S0 manifest.
- New SEED `rejev-eval2-20261017-1`; probe `rejev-eval2-probe-20261017`.

### C.2 Three-split quotas

| Source | dev2 | calibration2 | test2 | Total |
|---|---:|---:|---:|---:|
| policy_v2 | 174 | 522 | 1,326 | 2,022 |
| routing_v2 | 84 | 264 | 666 | 1,014 |
| policy | 24 | 66 | 165 | 255 |
| research_taxonomy_v21 | 56 | 168 | 424 | 648 |
| mnli | 74 | 218 | 554 | 846 |
| boolq | 45 | 131 | 332 | 508 |
| banking77 | 43 | 131 | 333 | 507 |
| **Total** | **500** | **1,500** | **3,800** | **5,800** |

Per-cell minimum families: dev2 5 / calibration2 15 / test2 40; actual minima 7/21/53.

### C.3 Audits 1–5 and disclosures

- Audits 1–5 all PASS (executed at generation time): pairwise statehash/family
  intersections among splits and with historical material are 0; answer-position
  construction invariants pass; surface-shortcut aggregate Δ at most 2.83pp; template
  structure 0 failures; per-cell minimum families met.
- Disclosures: public-source pretraining exposure (about 32%); shared
  templates/vocabularies/rules with history; near-duplicate checking not done; boolq
  position imbalance (test2 A 57.2%, z=2.63, p=0.008); routing_v2
  `longest_description` by-source Δ=13.3pp as an upstream fixed-option class prior;
  research_taxonomy_v21 constructed labels; banking77 `none` augmentation; variant 0
  canonical order.
- Full fields are in `009-s0-manifest.json` `disclosures` and exp008 README §9.

## Appendix D · Artifact Index and Traceable Entries

| Item | Path / identifier |
|---|---|
| This report, EN | `docs/report/technical-report.md` |
| This report, ZH | `docs/report/technical-report.zh-CN.md` |
| Previous version v1.0.2 | archive not distributed with this export |
| S1 record | `docs/experiments/exp008-s1-multiseed/README.md` |
| S1 analysis | `analysis.json` `695e9054…a84ff`; `analysis.md` `97e4aba4…e28f6`; `analysis-manifest.json` source-of-record `f5dee8f5…8ef18`; public copy `18d724cc…b52e4` |
| S0 manifest | `docs/plan/009-s0-manifest.json` source-of-record `5c33d48b…44f418`; public copy `504a7fcd…64b17` |
| S1 manifest | `docs/plan/009-s1-manifest.json` source-of-record `41f1a53e…87f6e3`; public copy `3787d14f…db23b7` |
| Plan | `docs/plan/009_report-enhancement-experiments.md` |
| Training entry | `src/train/multiseed_train.py`; `src/train/trigger_multiseed.py` |
| Scoring entry | `src/train/eval_candidates.py`; `src/train/candidate_math.py`; `src/train/s1_spec.py` |
| Analysis entry | `src/analysis/s1_analysis.py`; `src/analysis/metrics.py` |
| Report tables/figures | `src/analysis/report_assets.py`; `src/analysis/s1_figures.py` |
| Data build/audit | `src/data/build_eval_v2.py`; `src/data/verify_eval2.py`; `src/data/verify_patch_barrier.py`; `src/data/verify_rebuild_evidence.py` |
| Review archive | not distributed with this release set |

**Public release scope**: the public version of this report follows the release tooling's
whitelist (not distributed with this export); the review archive and local scratch directories
are not distributed with the public set; **the plan/009 records, exp008, and figures 3–5 are now in the whitelist;
the public set carrying this revision has been published with this version on GitHub**; S1's raw JSONL, adapters, and checkpoints are
on a private Modal Volume and not distributed with the repo. File existence or platform
inclusion is not model-effect acceptance.
