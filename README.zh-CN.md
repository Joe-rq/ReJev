# ReJev

**轻量决策模型的可复现 post-training**

[English](README.md) · [技术报告](docs/report/technical-report.zh-CN.md)

以 MiniCPM5-2B 为底座，对一个 Jev 风格的**选项决策模型**训练流程做独立复现，
探索小模型能否在**不生成自由文本**的前提下学会稳定的结构化决策行为。

## 实验

`state + question + options → 一个决策`

- MiniCPM5-2B + LoRA post-training
- 封存留出集，含泄漏检查
- 约束解码 vs. 无约束解码的对照评测
- 统计显著性检验与分组聚类 bootstrap 分析
- 实验记录、成本追踪与可复现性边界的如实登记

## 结果

**51.11% 底座 → 80.50% ReJev-2B**（封存留出集，1,892 题 / 924 组）

`+29.39pp · 无效输出 0% · app 累计计费 $5.31`

> 成本口径：$5.31 是 `rejev` 这个 Modal app 的**累计**计费，含该 app 此前全部探错开销，
> **不等于**本轮训练本身的费用；各轮口径不一，不可直接比较或相加（[技术报告](docs/report/technical-report.zh-CN.md) §1.3）。

目的不是宣称与 Jev 等价，而是通过受控实验去理解并复现轻量决策模型的工作机理。

**方向**：Decision Models · Post-training · Evaluation · Reproducibility

## 这个仓库真正证明的四件事

1. **模型理解**——决策模型与生成式 LLM 的差别在哪里。
2. **完整的 post-training 链路**——数据 → LoRA → 评测，而不是只调 API。
3. **实验设计**——封存留出集、泄漏控制、基线、显著性检验，以及一次**失败得很有信息量**的
   容量探针。
4. **工程纪律**——成本记录、checkpoint、来源锁定、可复现性边界，以及**如实写出限制**而非
   把它抹平。

## 我们不宣称的事

- 不声称与 Jev 官方模型能力等同，也不声称复现了 Tev1-4B。
- 不把上面的数字外推为分布外能力——现有证据只覆盖本考卷与其同类，没有更宽的范围。
- 不声称剔源代价可归因到单一变量（那一轮有两项共变）。⚠️ 此行原引的 `−2.33pp`
  **已作废（2026-09-28 更正）**——它测自被污染的切分；在干净 holdout 上更正后为 **+0.63pp**
  （CI [−1.13, +2.37]）。见[技术报告](docs/report/technical-report.zh-CN.md) §2.3。

每份实验记录都带「效度边界」节，逐条写明读到什么、不能读到什么。

> ⚠️ **发布范围**：本仓发布的是实验记录的**主体**，不是全部。**三处整篇排除**——
> ① `exp005`（真 Jev 四方对照）：结论建立在对第三方权重（Tev1-4B，许可未定）的实测
> 数字上；② `plan/007` 与 `008`（权重发布方案）：其正文就是「按哪些内部文件核验了
> 什么」，脱敏后只剩空壳；③ `src/publish/`（权重发布工具链）：它走 HF/魔搭渠道发布、
> 依赖不随本导出分发的文件（模型卡在内），且其单元测试样例含**字面**的坏凭据样式，
> 脱敏会把测试改坏。此外，**上游数据构建管线不在本仓**：它是 tev1 的 MIT 代码，
> 复现时需自行获取（见 [REPRODUCING.md](REPRODUCING.md) 第 2 节）。

## 目录结构

```
src/align/    训练前对齐检查（tokenizer / 模板 / loss mask / 解析对抗），零 GPU
src/data/     数据渲染器与切分封存（records → instruction；分层切分 + 零泄漏断言）
src/train/    训练与评测（Modal 端 LoRA SFT、约束解码评测、汇总与统计）
tools/        远程任务触发脚本（deploy + spawn 模式）与复现路径校验器
docs/plan/    方案与拍板记录（每处决策「谁定的、依据什么」）
docs/experiments/  各轮实验记录（配置、结果、效度边界、复现路径、账单）
```

## 快速开始

```bash
uv sync --group align          # 本地环境：数据链与对齐检查（tokenizer-only，零 GPU）
uv run python src/align/v1a_check.py --offline   # 7 项对齐检查
```

训练与评测在 Modal 上跑（需要账号与 GPU 额度）。训练脚本**必须**用 `deploy` + `spawn`
而非 `modal run` 直接跑——后者会把本地进程绑在远程任务上，本地终止会连带取消远程训练。

```bash
modal deploy src/train/clean_train.py
modal run tools/trigger_clean.py        # 秒级返回，训练与本地完全解耦
```

完整步骤（含**验证分级表**）见 **[REPRODUCING.md](REPRODUCING.md)**。

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
