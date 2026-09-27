# ReJev

**Reproducible post-training of a lightweight decision model**

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

**51.11% baseline → 80.50% ReJev-2B** on a sealed holdout（1,892 items / 924 groups）

`+29.39pp · 0% invalid output · $5.31 training cost`

The goal is not to claim equivalence with Jev, but to understand and reproduce the
mechanics of lightweight decision models through controlled experiments.

**Focus**: Decision Models · Post-training · Evaluation · Reproducibility

---

## 中文摘要

以 MiniCPM5-2B 为底座，对一个 Jev 风格的**选项决策模型**训练流程做**独立、可复现的
post-training 实验**：给定 state、问题与一组带标签的选项，模型输出**一个字母**——
不生成自由文本。

在一个**封存留出集**（1,892 题 / 924 组，与训练集三口径零泄漏）上，底座 51.11% →
LoRA 微调后 **80.50%**（+29.39pp）；无效输出率 0%；训练成本 **$5.31**（Modal L4）。

**本仓要证明的不是「我们复刻了 Jev」**，而是这套受控实验的四个面：
① **模型理解**——决策模型与生成式 LLM 的差别在哪里；
② **post-training 全链**——数据 → LoRA → 评测，而不是只调 API；
③ **实验设计**——封存留出集、泄漏控制、基线、统计检验、容量探针；
④ **工程纪律**——成本记录、checkpoint、来源锁定、可复现性与限制的如实登记。

> ⚠️ **发布范围**：本仓发布的是实验记录的**主体**，不是全部。两处缺口的性质不同——
> ① `exp005`（真 Jev 四方对照）**未随本仓发布**：它的结论建立在对第三方权重（Tev1-4B，
> 许可未定）的实测数字上；② **上游数据构建管线不在本仓**：它是 tev1 的 MIT 代码，
> 复现时需自行获取（见 [REPRODUCING.md](REPRODUCING.md) 第 2 节）。

## 目录结构

```
src/align/    训练前对齐检查（tokenizer / 模板 / loss mask / 解析对抗），零 GPU
src/data/     数据渲染器与切分封存（records → instruction；分层切分 + 零泄漏断言）
src/train/    训练与评测（Modal 端 LoRA SFT、约束解码评测、汇总与统计）
tools/        远程任务触发脚本（deploy + spawn 模式，见下）与复现路径校验
docs/plan/    方案与拍板记录（每处决策「谁定的、依据什么」）
docs/experiments/  各轮实验记录（配置、结果、效度边界、复现路径、账单）
```

## 快速开始

```bash
uv sync --group align          # 本地环境：数据链与对齐检查（tokenizer-only，零 GPU）
uv run python src/align/v1a_check.py --offline   # 7 项对齐检查
```

训练与评测在 Modal 上跑（需要账号与 GPU 额度），**必须**用 `deploy` + `spawn`
而非 `modal run` 直接跑——后者会把本地进程绑在远程任务上，本地终止会连带取消远程训练：

```bash
modal deploy src/train/clean_train.py
modal run tools/trigger_clean.py        # 秒级返回，训练与本地完全解耦
```

完整步骤（含验证分级）见 **[REPRODUCING.md](REPRODUCING.md)**。

## 我们不宣称的事

- 不声称与 Jev 官方模型能力等同，也不声称复现了 Tev1-4B；
- 不把上述成绩外推为分布外能力（现有证据只覆盖本考卷与其同类）；
- 不声称 −2.33pp 的剔源代价可归因到单一变量（该轮有两项共变）。

各实验记录里都有「效度边界」节，逐条写明读到什么、不能读到什么。

## 上游与致谢

- **[tev1](https://github.com/togethercomputer/tev1)**（MIT, commit `1dde778`）——
  任务协议规格（SYSTEM 指令、选项标签口径、推理参数）、状态归一化与分层切分口径
  均源自该项目。本仓**不复制**其数据构建管线，复现时需自行获取。
- **MiniCPM5-2B**（Apache-2.0）——底座模型与其官方 TRL LoRA SFT 配方。
- 各训练数据来源及其许可状态见 [NOTICE](NOTICE) 与 `sources.lock.json`。

本仓**不含**任何数据、模型权重或逐题结果；**不发布** Tev1-4B 相关资产。

## 许可

本仓代码以 **MIT** 发布，见 [LICENSE](LICENSE)。第三方来源见 [NOTICE](NOTICE)。

## 引用

见 [CITATION.cff](CITATION.cff)。
