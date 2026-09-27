# ReJev

> 以 **MiniCPM5-2B** 为底座，独立复现 Jev 风格的**选项决策模型**训练流程，
> 并检验它能否在保留选择正确率的同时，形成**可复现、可审计**的轻量模型。

**English abstract.** ReJev is an independent, reproducible reproduction of the
Jev-style decision-model training pipeline on top of MiniCPM5-2B. Stage 1 covers
**single-letter Choice classification**: given a state, a question and a set of
labelled options, the model emits one letter. The repository ships the data
pipeline, LoRA training and constrained-decoding evaluation code, the sealed
holdout discipline, and the experiment records. We report accuracy on a
locked, never-trained-on holdout, together with invalid-output rate, baseline
comparison and actual training cost. *Not affiliated with TypeSafe or with the
tev1 project — see [NOTICE](NOTICE).*

---

## 这是什么

一个**实验记录 + 可复现代码**仓库，回答一个问题：把 Jev 风格的任务语义、
提示协议与评测口径搬到一个小参数模型上，能否训练出行为稳定的决策模型，且全过程
可追溯到精确的代码、数据、配置与账单。

> ⚠️ **发布范围**：本仓发布的是实验记录的**主体**，不是全部。两处缺口的性质不同——
> ① `exp005`（真 Jev 四方对照）**未随本仓发布**：它的结论建立在对第三方权重（Tev1-4B，
> 许可未定）的实测数字上；② **上游数据构建管线不在本仓**：它是 tev1 的 MIT 代码，
> 复现时需自行获取（见 [REPRODUCING.md](REPRODUCING.md) 第 2 节）。

- **任务**：`state + question + options → 单个字母`（2–24 选项，标签 A–X 连续）
- **底座**：`openbmb/MiniCPM5-2B` @ `12a3808a`（Apache-2.0），LoRA 微调
- **评测**：锁定且未参与训练的封存集；约束解码与无约束两口径并报

## 主要结果

第一阶段（Choice 单字母分类）在一份 **1,892 题的封存留出集**（924 组，按
`source × group_id` 分层切分，与训练集双口径零泄漏）上：

| 模型 | 准确率 | 说明 |
|---|---:|---|
| MiniCPM5-2B 底座（未微调） | 51.11% | 同集锚定 |
| **ReJev-2B（本仓产物）** | **80.50%** | LoRA r16/α32，1 epoch |
| 提升 | **+29.39pp** | McNemar p≈3e-96 |

- 约束解码与无约束两口径**逐位相同**；无效输出率**均为 0%**（模型天然输出干净单字母，
  约束器从未触发）。
- 训练成本 **$5.31**（Modal L4，4.2 小时），全程有账单核对与 checkpoint 归档。
- 在 tev1 官方开发集（1,300 题）上，本仓 2B 得 **86.46%**，官方公布的 4B 成绩为
  **90.77%** —— 即达到官方 4B 的 **95.3%**。⚠️ 这只是**分布内**读数：考卷源自
  tev1 配方数据，对同谱系模型有利。

**我们不宣称的事**（见各实验记录里的「效度边界」节）：

- 不声称与 Jev 官方模型能力等同，也不声称复现了 Tev1-4B；
- 不把上述成绩外推为分布外能力（现有证据只覆盖本考卷与其同类）；
- 不声称 −2.33pp 的剔源代价可归因到单一变量（该轮有两项共变）。

## 目录结构

```
src/align/    训练前对齐检查（tokenizer / 模板 / loss mask / 解析对抗），零 GPU
src/data/     数据渲染器与切分封存（records → instruction；分层切分 + 零泄漏断言）
src/train/    训练与评测（Modal 端 LoRA SFT、约束解码评测、汇总与统计）
tools/        远程任务触发脚本（deploy + spawn 模式，见下）
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

完整步骤见 **[REPRODUCING.md](REPRODUCING.md)**。

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
