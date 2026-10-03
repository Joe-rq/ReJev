# ReJev Technical Report

**Lightweight decision-model post-training on MiniCPM5-2B: an independent reproduction of a Jev-style pipeline**

Version 1.0.1 · released 2026-09-28, corrected 2026-10-02 · [中文版](technical-report.zh-CN.md)

---

## Abstract

This report documents an **independent reproduction**: porting a Jev-style
"structured decision" training pipeline onto MiniCPM5-2B (2B), and answering two
falsifiable questions — **is the reconstruction faithful**, and **what was and was
not obtained**.

**tev1** here means Together's public Jev-style recipe — the repository
accompanying *How to train your own Jev for \$17*. This report reproduces its
**pipeline**, **not** TypeSafe's Jev product.

**Main result**: on a self-built, sealed holdout (1,892 items / 924 groups), the
untuned base model scores **51.11%** and the tuned model **80.50%**
(+29.39pp, McNemar p=2.94e-96), with a **100% valid-output rate** under both
constrained and unconstrained decoding.

**Reconstruction fidelity**: against the **upstream published** Tev1-4B figures
(**880/1000 + 300/300**, published via the Together API), our local readings are
identical **bit-for-bit (0.00pp)**; our renderer reproduces 1,300 upstream prompts
**verbatim**. The evaluation harness is thereby cross-checked against upstream's
published figures.

**This report also records three negative results**, which matter as much as the
main one:

1. **Out-of-distribution collapse** — on a third-party public exam (2,087 items),
   ReJev-2B's phishing recall collapses to **0** (all 1,000 positives classified as
   negatives) and accuracy degenerates to a constant prediction. **The untuned base
   model does better** (57.05% vs 50.00%, p=2.58e-41).
2. **The capacity probe was a training failure** — raising LoRA rank from 16 to 64
   destroys the model (−23.31pp, below the base model). This is an **optimization
   failure, not a capacity limit**; the original question was not answered.
3. **A contamination correction** — a holdout leak invalidated an earlier headline
   comparison; after correction, the cost of removing license-restricted corpora
   moves from −2.33pp to **+0.63pp** (95% CI [−1.13, +2.37]).

**What this report does not claim**: it does not claim equivalence with the official
Jev model; it does not claim to have reproduced Tev1-4B (only its evaluation chain);
and it does not extrapolate in-distribution numbers out of distribution — on the
contrary, §2.5 gives the measured out-of-distribution figures.

A note on scope: our public release is an **explicitly enumerated** set. In outline:
`src/` (`align`/`data`/`train`; the weight-release toolchain under `publish/` is
excluded, see below); the `docs/plan/` README and plans 002–006;
the `docs/experiments/` README and exp001–004, 006, 007; this report (both languages);
the three scripts under `tools/` distributed with the release; and the build/environment
files (`pyproject.toml`, `uv.lock`, `sources.lock.json` and the repository-root files,
including `LICENSE`/`NOTICE`/`CITATION.cff`/`REPRODUCING.md`). **Three groups are excluded
in full**: the exp005 record (whose third-party-weight column carries our own
measurements), plans 007/008 (the weight-release plans), and the weight-release
toolchain under `src/publish/` (it ships via the HF/ModelScope channel, depends on
files not part of the export — the model card among them — and its unit-test
fixtures contain **literal** malformed-credential patterns that redaction would
break — so the directory is excluded wholesale). **No data, model
weights or per-item results are included.**

---

## 1. Method and Setup

### 1.1 Task formulation

The task is **structured decision-making**, not free-form generation:

```
state + question + options → one option
```

The difference from a generative chat model is that the output is constrained to
**one of the given options**, so "can the model talk" never enters the evaluation —
only "did it choose correctly". We implement this with **state-machine constrained
decoding** (`prefix_allowed_tokens_fn`): the candidate set is the letter tokens of
**that item's actual options**, not a fixed A–X. This makes our constrained decoding
equivalent to upstream's "generate a regex per option list" protocol.

Every evaluation round runs both **constrained** and **unconstrained** decoding. That
the two agree bit-for-bit is itself evidence: the model is not relying on the
constraint to choose correctly.

### 1.2 Data

Upstream tev1 compiles five public datasets into a `state / question / options /
answer` semantic layer using deterministic code. We do **not** copy its build
pipeline; we pin the upstream commit and obtain and rebuild the sources ourselves.

| Item | Value |
|---|---|
| Built training set | 37,840 items (43,840 − 6,000 exact duplicates) / 16,929,529 tokens |
| Training set (exp002, full) | **35,948** items (37,840 − 1,892 holdout) |
| Dev set | 4,568 items |
| Sealed holdout | **1,892** items / **924** groups / 9 sources |
| Splitting | stratified by whole `group_id`, with zero-leak assertions |

**The holdout is sealed**: the splitting script writes content hashes and zero-leak
assertions over three keys (same-source detection key, group key, identifier), after
which it may not be used for training or tuning.

The license status of the five sources is covered in §5 — two of them (AG News,
SST-5) do not permit redistributing derived data, a fact that directly caused two of
the training runs in §1.3.

### 1.3 Training recipe

Base model `MiniCPM5-2B` (Apache-2.0, revision pinned). Training inputs are exported as
`sft` messages by `src/data/render.py`; each run then formats them with a training-specific
chat template (`TRAIN_TEMPLATE`, rebuilt from the official MiniCPM `docs/finetune/trl.md`
structure), not the model's own. We do not use the Tev1-4B tokenizer to prepare training data.
LoRA SFT, with the
hyperparameters **identical to the character** across runs **except for rank**
(exp004 uses r64/α128, see §2.4):

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

### 1.4 Evaluation protocol and statistics

- **Primary metric**: accuracy under constrained decoding. "Valid rate" is the
  fraction of outputs landing on an option that actually exists for that item.
- **Interval estimation**: a **grouped bootstrap** resampling whole `group_id`
  clusters. The report's two sets of intervals use **different draw counts** (from
  two independent runs) — **do not conflate them**: §2.1 main table uses **2,000
  draws** (1,000 for per-source tables, `seed=20260925`; see
  `src/train/jev_compare.py`), while §2.3 uses **10,000 draws** (see the exp006
  record). The two sets also cover different evaluation sets and subsets; **defer
  to the experiment record of the section you are reading**.
  Variants within a group are highly correlated, so the
  per-item independence assumption fails; per-item Wilson intervals are reported for
  reference only.
- **Paired testing**: per-item McNemar, both arms on the same items.

---

## 2. Results

### 2.1 Main result

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

### 2.2 Reconstruction fidelity: bit-for-bit

The first question in a reproduction is whether the thing reproduced is the same
thing. We answer with two independent pieces of evidence:

1. **Rendering fidelity**: rendering 1,300 items with Tev1-4B's own tokenizer
   reproduces the upstream `instruction/*.jsonl` prompts **verbatim (1300/1300)**;
   completions are 100% equal to `answer + <|im_end|>`.
2. **Behavioural fidelity**: against the **upstream published** Tev1-4B figures —
   **880/1000 (88.0%) + 300/300 (100.0%)**, published via the Together API — our local
   run of those weights on our reconstruction of upstream's exam showed **no deviation
   at all (0.00pp)**.
   Per NOTICE §4 we state only this **relation** ("our reading equals the published
   value"); the reading itself is **not presented as a result of ours**.

The local end-to-end run matched upstream's **two published sub-set figures** — 880/1000 for
`main` and 300/300 for `policy_transfer`; whether the per-item predictions are identical was
**not** checked. Still, every cross-model comparison that follows rests on a measuring
instrument **cross-checked against upstream's published figures**.

> Third-party weight-distribution licensing prevents us from publishing Tev1-4B's
> per-item results or other derived figures from our exam; we cite only the figures
> upstream has already published.

### 2.3 The cost of license remediation: corrected figures

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
interval contains 0). The −9.71pp on restricted sources is the direct expression of
"the model never trained on these two corpora", not a loss of capability. Weighted
consistency check: `(175 × −9.71 + 1,717 × 1.69) / 1,892 = +0.64pp`, 0.01pp away from
the **+0.63pp** headline — **both values fall inside the range allowed by the rounding
of the per-subset figures** (solving back: with the per-subset values anywhere within
their 2-dp rounding interval, the weighted mean lies in [+0.631, +0.641]), so this is
not two inconsistent computations.

**The limitation stands** (see §4): source removal and re-splitting co-vary, so
+0.63pp cannot be attributed to a single change.

### 2.4 The capacity probe: an informative failure

Raising LoRA rank from 16 to 64 in the hyperparameters (learning rate, schedule and
the rest unchanged, though **the data split and evaluation set also differ** — 34,146
training items and a newly cut 1,802-item eval set) was meant to answer whether the
weaknesses identified earlier reflected insufficient training or a 2B capacity ceiling.

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
**Do not conflate this with the same-looking 27.91% in §2.1**: that one is a *baseline*
(n=1,892 on the sealed holdout — which also serves as the dev set from exp004 onward;
the "dev set, 4,568 items" row in §1.2 is exp002's, a different set), this one is r64's
*measured accuracy* (n=1,802) — same digits, different meaning.

A by-product is one data point on this base model: **this 16 → 64 attempt failed**. It
is **not** evidence that "the recipe is extremely sensitive to rank" — the training
split and eval set differ too, so the rank effect is not separated from them.

### 2.5 Out-of-distribution: collapse

Everything above is in-distribution. To ask "how much of this recipe survives leaving
the training distribution", we use a **third-party exam of 2,087 items published
alongside its evaluation code** (`phishing` 2,000 binary / `tool_risk` 60 /
`ticket_routing` 27). Our corpora and this exam share **zero overlap** under three
keys: item identifier, content hash (NFKC + casefold + word-sequence sha256), and a
loose cross-key-structure content comparison. The latter two miss, so this is not
"the same items renamed".

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
> the weights locally and our readings matched upstream bit-for-bit; the confusion matrix in
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
4. **Contrast: the real Jev did not collapse** (upstream-published recall 42.70%).
   Same exam, same task — **the collapse is not the intrinsic difficulty of the task**;
   a non-collapsing solution exists. But the real Jev differs from our recipe in both
   model and training, so more than the recipe varies here — this does not support
   attributing the collapse to the recipe alone.

**Two secondary observations**:

- **License remediation does not change out-of-distribution behaviour**: r16 and
  clean-r16 are **identical item-by-item** across all 2,000 (McNemar 0 vs 0, p=1).
- **r64 collapses in the opposite direction**: it classifies nearly everything as
  positive (recall 96.20% / FPR 99.70%). Its accuracy is indistinguishable from r16
  (p=0.442) but its error pattern is nearly complementary.

---

## 3. Reproducibility Audit Chain

Every conclusion is tied to exact code, weights, data and environment versions.

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

---

## 4. Validity and Limitations

**a. In-distribution numbers must not be extrapolated.** 80.50% covers our self-built
sealed holdout and its kin. §2.5 gives the direct out-of-distribution reading: **collapse**. Any
claim that "the model can handle X", where X is outside this exam's distribution,
lacks evidence.

**b. Scope of the collapse reading.** It describes this recipe's behaviour on **one
binary sub-task of one exam**. The same model on another task within the same exam
(`tool_risk`) does not collapse (r16 scores 85.00% against the base's 60.00%). So
§2.5 **cannot** support "ReJev-2B is useless on any out-of-distribution task".

**c. The exam is self-built, and its composition is not neutral.** The primary exam
was constructed and balanced by this project, and contains a home-field component.
The third-party exam (§2.5) partially offsets this, but its primary subset is a single
binary task.

**d. The cost of having no group key.** Items on the third-party exam are independent,
with no `group_id`, so there is no grouped bootstrap and the per-item intervals are
**optimistic**. The primary exam has 924 groups and is better in this respect.

**e. Protocols are not strictly comparable.** The rows in §2.5 come from different
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

**i. Single runs.** Each training run is a single seed, single run; this report
contains no run-to-run variance.

**j. No final evaluation set untouched by tuning.** The holdout is sealed, but the
pipeline that produced it (including hyperparameter selection) was developed with
reference to it; on the present material this cannot be ruled out.

**k. "No difference detected" is not "equivalent".** §2.3 and any comparison whose
interval contains 0 show only that there is no evidence at this exam's scale — not
that the two are the same.

**l. No untuned-4B control arm.** In §2.5, the sibling 4B is described by upstream as a 4B base plus the tev1 recipe,
and we did not run that base's untuned version on the same exam. The contributions of
base and recipe therefore cannot be separated: the reading supports "the collapse is
not specific to 2B capacity" but **not** "the recipe caused the collapse".

---

## 5. Licensing and Compliance

Release decisions are made per **data source license status**, not per "is it publicly
available":

| Source | License status | Training | Redistributing derived data |
|---|---|---|---|
| MultiNLI | Mixed (predominantly OANC EULA, not CC) | ✅ | per terms |
| BoolQ | CC BY-SA 3.0 | ✅ | per terms (ShareAlike judged separately) |
| Banking77 | CC BY 4.0 | ✅ | per terms (attribution) |
| **AG News** | Card limits use to "research purposes … any other non-commercial activity" | ✅ | ❌ |
| **SST-5** | **Verified**: no license metadata — all four layers of the provenance chain (dataset card / HF metadata / repository / upstream build artifacts) are silent. **Not verified**: the original paper and source page were not read directly. **Handling**: absence of a declaration is treated conservatively as absence of authorization, so nothing is released | ✅ | ❌ |

**Determination**: all five sources may be used for local training; **the two
restricted sources may not be redistributed as derived data**. The publicly released
adapter therefore comes from a source-removed retraining (30,987 items, 3,159 removed).

**Synthetic portion**: upstream generates synthetic samples with purely deterministic
code (zero external data input, zero API calls, zero model generation) and **never
declared a license for them**. We are the generator, not a recipient.

**Tev1-4B weights**: their license status is undecided, so we **publish no copy of
those weights and no per-item results**, using them only for internal research
comparison; §2.2 and §2.5 cite only figures upstream has already published.

**Unverified items (recorded as-is)**: one MultiNLI sub-item's license could not be
verified; SST-5's original paper and source page were not read directly. Our releases
claim only "handled according to the above determinations", not that verification is
exhaustive.

---

## 6. Reproduction Guide

Full steps, verification tiers and check commands are in
[REPRODUCING.md](../../REPRODUCING.md) (shipped with this repository). Three essentials:

1. **`uv sync --group align`** — the data-chain and alignment-check dependencies are
   not in uv's default group; omitting this yields an empty environment.
2. **Use `deploy` + `spawn` for training and full evaluation, not a direct
   `modal run`.** The latter binds the local process to the remote job, so killing it
   locally sends a cancellation that **kills the running job**.
3. **Verify the sealed-set fingerprint for every arm** — a row-count assertion cannot
   catch "a different file with the same number of rows".

---

## 7. Recorded As-Is

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
  34,146 and 34,146 + 1,802 = 35,948) and dug out the holdout contamination in §2.3.
  The lesson: **keep every number next to its provenance and the anomaly surfaces on
  its own.**
- **One index gap, since fixed**: a plan index was missing one row (a pre-existing
  gap, explicitly recorded as "not to be fixed in passing"); it has since been filled in.
- **Two rounds of inconsistent figures**: earlier experiment records retain
  pre-correction numbers (as described in §2.3); this report uses the corrected
  figures throughout, and the original text in those records is left unchanged with
  the discrepancy recorded.

---

## Citations and Acknowledgements

- **[tev1](https://github.com/togethercomputer/tev1)** (MIT) — the task protocol
  spec, state normalisation and stratified splitting originate there. We do **not**
  copy its data-build pipeline.
- **MiniCPM5-2B** (Apache-2.0) — base model and its official LoRA SFT recipe.
- The authors and maintainers of the five public datasets; license status in §5.
- Per-item source locking is in `sources.lock.json`; third-party notices in
  [NOTICE](../../NOTICE).

**This repository contains no data, model weights or per-item results, and
distributes no Tev1-4B assets.**
