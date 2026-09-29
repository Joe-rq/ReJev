# ReJev

**Reproducible post-training of a lightweight decision model**

[中文版](README.zh-CN.md)

An independent reproduction of a Jev-style decision-model training pipeline on top of
MiniCPM5-2B, exploring whether a small model can learn stable structured decision
behavior *without* autoregressive free-form generation.

## Experiment

`state + question + options → one decision`

- MiniCPM5-2B + LoRA post-training
- sealed holdout with leakage checks
- constrained vs. unconstrained decoding evaluation
- statistical significance and grouped bootstrap analysis
- experiment records, cost tracking and reproducibility boundaries

## Result

**51.11% baseline → 80.50% ReJev-2B** on a sealed holdout (1,892 items / 924 groups)

`+29.39pp · 0% invalid output · $5.31 cumulative app billing`

> On cost: the `$5.31` figure is the **cumulative** billing of the `rejev` Modal app
> (it includes all earlier mis-run overhead for that app); it is *not* the cost of this
> training run alone. Cost conventions differ across rounds — they are not comparable
> or additive (technical report §1.3).

The goal is not to claim equivalence with Jev, but to understand and reproduce the
mechanics of lightweight decision models through controlled experiments.

**Focus**: Decision Models · Post-training · Evaluation · Reproducibility

## What this repository actually demonstrates

1. **Model understanding** — what separates a *decision model* from a generative LLM.
2. **A complete post-training loop** — data → LoRA → evaluation, not an API call.
3. **Experiment design** — sealed holdout, leakage control, baselines, significance
   testing, and a capacity probe that failed informatively.
4. **Engineering discipline** — cost tracking, checkpoints, source locking,
   reproducibility boundaries, and limitations stated rather than smoothed over.

## What we do *not* claim

- We do not claim equivalence with the official Jev model, nor that we reproduced Tev1-4B.
- We do not extrapolate the numbers above to out-of-distribution capability — the
  current evidence covers this exam and its kin, nothing wider.
- We do not claim the cost of removing restricted sources is attributable to a single
  variable (two factors co-vary in that run). ⚠️ The `−2.33pp` figure this line used to
  cite is **void (corrected 2026-09-28)** — it was measured on a contaminated split; the
  corrected reading on the clean holdout is **+0.63pp** (CI [−1.13, +2.37]). See the
  technical report §2.3.

Every experiment record carries a *validity boundaries* section stating what was
measured and what cannot be read from it.

> ⚠️ **Scope**: this repository ships the **bulk** of the experiment records, not all
> of them. **Three paths are excluded in full** — ① `exp005` (the four-way comparison
> against the real Jev): its conclusions rest on measured numbers for a third-party
> checkpoint (Tev1-4B) whose license is undecided; ② `plan/007` and `008` (the
> weight-release plans): their text *is* "which internal files were checked for what",
> so redacting them leaves a hollow shell; ③ `src/publish/` (the weight-release
> toolchain): it ships via the HF/ModelScope channel, depends on files that are not
> part of this export (the model card among them), and its unit-test fixtures contain
> **literal** malformed-credential patterns that redaction would break.
> Separately, the **upstream data-build pipeline is not included**: it is tev1's MIT
> code and must be obtained separately (see [REPRODUCING.md](REPRODUCING.md) §2).

## Layout

```
src/align/    pre-training alignment checks (tokenizer / template / loss mask / parsing), zero GPU
src/data/     renderer and sealed splitting (records → instruction; stratified split + zero-leak assertions)
src/train/    training and evaluation (LoRA SFT on Modal, constrained-decoding eval, aggregation and stats)
tools/        remote-job trigger scripts (the deploy + spawn pattern) and the repro-path checker
docs/plan/    plans and decision records (who decided what, and on what evidence)
docs/experiments/  per-round records (config, results, validity boundaries, repro path, billing)
```

## Quick start

```bash
uv sync --group align          # local env: data pipeline + alignment checks (tokenizer-only, zero GPU)
uv run python src/align/v1a_check.py --offline   # 7 alignment checks
```

Training and evaluation run on Modal (account and GPU credit required). Use
`deploy` + `spawn` — **not** `modal run` — for the training script: the latter binds
the local process to the remote job, and killing it locally will cancel the remote run.

```bash
modal deploy src/train/clean_train.py
modal run tools/trigger_clean.py        # returns in seconds; training is fully decoupled
```

Full steps, including a **verification tier table**, are in **[REPRODUCING.md](REPRODUCING.md)**.

## Upstream & acknowledgements

- **[tev1](https://github.com/togethercomputer/tev1)** (MIT, commit `1dde778`) — task
  protocol spec (system instruction, option-labelling convention, inference parameters),
  state normalisation and stratified splitting all originate there. This repository does
  **not** copy its data-build pipeline; obtain it separately to reproduce.
- **MiniCPM5-2B** (Apache-2.0) — base model and its official TRL LoRA SFT recipe.
- Data sources and their license status: see [NOTICE](NOTICE) and `sources.lock.json`.

This repository contains **no** data, model weights or per-item results, and
distributes **no** Tev1-4B assets.

## License

Code is released under **MIT** — see [LICENSE](LICENSE). Third-party notices in [NOTICE](NOTICE).

## Citation

See [CITATION.cff](CITATION.cff).
