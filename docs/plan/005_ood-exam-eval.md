---
type: plan
number: "005"
date: 2026-09-25
title: OOD 客卷评测：第三方客卷 2,087 题的分布外对照
tags: [evaluation, ood, reproduction, exp005]
status: draft
related: [plan/003_full-training-eval, plan/004_weakness-probe, experiments/exp004-jev-comparison]
---

# 005 · OOD 客卷评测方案

追踪 Issue：#5（内部追踪项）（P1，open）

## Context

现有全部对照评测都发生在**同一分布内**：

| 轮次 | 考卷 | 分布 | 主场 |
|---|---|---|---|
| exp002 | 自建 holdout 1,892 | tev1 分布内 | ReJev |
| exp003 | tev1 官方 dev 集 1,300 | tev1 分布内 | Tev1-4B |
| exp004 | 自建 holdout 1,892 | tev1 分布内 | ReJev（Jev 客场） |

因此 `95.3%`（ReJev-2B vs 官方 4B 公开成绩）与「ReJev-2B ≈ 真 Jev」两个读数**都无法区分「学会了决策」与「学会了这个数据分布的规则」**。exp004 已把这条写进效度边界，但没补上。

上游 tev1 仓库内保存着一份**第三方**构造的客卷（anisselbd / themsquared / WallerChen 三家），连同双模型重跑结果。**在这张卷子上三方都是客场**——这是本仓现有材料里唯一能回答「出了分布还剩多少」的东西。

## 现状（代码已确认）

### 考卷本体已在仓里

`resources/tev1/evaluation/public-third-party/inputs.json`（gitignored，只读）——2,087 条：

| suite | n | 选项数 | label |
|---|---:|---:|---|
| phishing | 2,000 | 2 | A–B（1000 正 / 1000 负 平衡） |
| tool_risk | 60 | 4 | A–D |
| ticket_routing | 27 | 5 | A–E |

逐题 schema：`{suite, id, task: {state, question, options: [{label, key, description}]}, gold: <option.key>}`

同目录另有三家自己的产物：`qwen.jsonl`（2,087）、`jev.jsonl`（2,088，含 1 次重试）、`report.json`、`analyze.py`、`README.md`。

### 上游已公布的参照

| | phishing 准确率 | recall | FPR |
|---|---:|---:|---:|
| tev1 官方 4B 复刻 | 50.6%（1013/2000） | 1.3% | 0.0% |
| 真 Jev | 62.9%（1259/2000） | 42.7% | 16.8% |
| 简单 URL 启发式（数据集自带） | 91.6% | — | — |

`tool_risk` 60 题双方均 93.3%、`ticket_routing` 27 题 96.3% / 100.0%——样本小且有天花板效应，**不作主口径**（与上游 README 的告诫一致）。

### 唯一的硬性 schema 断裂

`gold` 是**选项 key**（`"phishing"` / `"billing"`），而本仓渲染链要求 `answer` 是**字母 label**：

- `src/data/render.py:55-57` 缺 `answer` 抛 `ValueError`
- `src/data/render.py:64-65` `answer ∉ labels` 抛 `ValueError`

必须做 `key → label` 映射。实测 gold 100% 落在 key 集内，label 全部 A 起连续。

### 长度与 token（钉死 revision 的 tokenizer 实测）

| suite | p50 | p99 | max |
|---|---:|---:|---:|
| phishing | 331 | 458 | 868 |
| tool_risk | — | — | 247 |
| ticket_routing | — | — | 212 |

门槛是 prep 的 1900 与 `eval_cross.py:279` 的 2048——**全部安全**。A–E 均为单 token（`[54]`–`[58]`）。

### 评测路径：走 `eval_cross.py`，不能走 `eval_holdout.py`

| | `eval_cross.py` | `eval_holdout.py` |
|---|---|---|
| 考卷注册 | `PAPERS` 字典，加一行即可（`:63-68`） | `--split` **闭集**（`:49`）＋路径 ternary（`:52`）＋条数常量（`:54`） |
| 约束候选集 | **按题**（`_n_options(messages)` → `letter_ids[:n_opt]`，`:270-272,287-291`） | **全 24 字母**（`:77,124-125`）→ 2 选项题会放过 B 以外的字母，**无效率算错** |
| 逐题 `n_options` | 有（meta 含 `constraint=per-item-options`） | **无** → `cross_report.py:61` 有效率退化为 A–X 口径 |
| 加新模型 | `MODELS` 已含 base / adapter / **adapter_r64** / tev1 | 仅 base / adapter |

**结论**：`eval_cross.py:45-56` 的 `MODELS` 已经注册好本项目要跑的全部四个模型，**一行都不用改**。

### `src/` 下没有 recall / FPR / 混淆实现

全仓唯一的实现是上游的 `resources/tev1/evaluation/public-third-party/analyze.py:24-27`，但它：① 分母 `/1000` 写死；② 只认第三方 schema；③ 断言 `len(rr)==2087`。**不可直接复用，须自写分析器**（可抄其逻辑 + issue-2 worktree `jev_compare.py:52-59` 的 `wilson()`、`:69-78` 的 `mcnemar_exact()`）。

### 无 `group_id` 的连带效应

本卷没有分组键，因此：
- **聚类 bootstrap 不可用**（实现只在 `.claude/worktrees/issue-2/src/train/jev_compare.py`，且三重锁死只能吃 holdout）→ 主口径退回 **Wilson 区间 + 逐题 McNemar**，并在报告中显式声明假设
- `render.py:112` 的 `groups` 统计会塌成 1，写进 manifest 时会被误读——须在 prep 里显式记 `"groups": null`
- 泄漏校验**去掉 `(source, group_id)` 口径**（会退化成单键、等于没查），只留 `statehash` + `id`

## 目标

回答：**tev1 这套配方训练出来的模型，出了训练分布还剩多少？**

可证伪的具体形态：**ReJev-2B 在 phishing 上的塌陷方向是否与官方 4B 复刻一致——即 recall 是否也塌到个位数。**

## 执行步骤

| # | 步骤 | 产物 | verify |
|---|---|---|---|
| 1 | **checkpoint 一致性核查**：比对上游 README 的 `hassan/Qwen3.5-4B-v1-new-69617472-bdc3c2fc` 与本仓 `togethercomputer/Tev1-4B-experimental` @ `0b7becf` | 一句结论 | config/权重指纹相符或不符；不符则报告措辞改为「同门 4B」 |
| 2 | **零泄漏复核**：本卷 2,087 条 × `data/rejev/records/rejev-{train,holdout,dev}.jsonl` ＋ `data/paper/tev1-paper.jsonl` | 复核记录 | `statehash` / `id` 双口径全 0 命中 |
| 3 | **新写 `src/train/prep_exam.py`**（以 `prep_paper.py` 为骨架） | `data/exam/exam.jsonl` + `exam-manifest.json` | ① `answer = {o.key: o.label}[gold]` 映射后 `render.build_messages` 零抛错；② `source := suite`；③ 带 `n_options`；④ 2,087 条齐全；⑤ 长度断言通过 |
| 4 | **改 `eval_cross.py:63-68`**：`PAPERS` 加 `"exam": {"path": "/vol/data/exam.jsonl", "n": 2087}` | 一行 diff | `--n 5` sanity 跑通 |
| 5 | **上传**：`modal volume put rejev data/exam/exam.jsonl /data/exam.jsonl` | Volume 文件 | `modal volume ls rejev /data` 可见 |
| 6 | **触发四次评测**（`modal deploy` + `.spawn()`，理由见 [REPRODUCING.md](../../REPRODUCING.md) 第 4 节）：`base` / `adapter` / `adapter_r64` / `tev1` | `/vol/eval/cross-{model}-exam.jsonl` + meta + summary | 四个 jsonl 各 2,087 条；meta 的 `paper_sha256` 与本卷一致 |
| 7 | **新写 `src/train/exam_report.py`**：recall / FPR / 混淆矩阵 / 分 suite 表 / Wilson / McNemar | `exam-report.json` + md | 与上游 `analyze.py` 在 qwen/jev 两个现成产物上**对得上**（用它们的输出做自检） |
| 8 | **改 `cross_report.py:36-42`**：注册 `exam` 卷与模型集 | 二到三行 diff | `cross-report.json` 含 exam 卷（不注册会**静默跳过**） |
| 9 | **exp005 归档** ＋ 更新状态板 / changelog | `docs/experiments/exp005-ood-exam/README.md` | 含协议差异表、执行偏离、复现路径 |

**第 7 步的自检很关键**：拿上游随卷附带的 `qwen.jsonl` / `jev.jsonl` 喂进自写分析器，**必须复现出 50.6% / 62.9% / recall 1.3% / 42.7% / FPR 0.0% / 16.8%**。对不上说明分析器错了，不是上游错了。

## 验收（预注册判据，训前锁定）

**主口径 = phishing（2,000 题）**。`tool_risk` / `ticket_routing` 只报不分判。

### 判据 A：ReJev-2B 的塌陷方向

| recall | 读作 |
|---|---|
| **< 10%** | 与官方 4B 复刻同向塌陷 → **配方层面的分布外缺陷**，且本仓可复现该缺陷 |
| 10% – 42.7% | 部分保留，介于 4B 复刻与真 Jev 之间 |
| > 42.7% | 优于真 Jev → **须先排除底座偏置**再看判据 B |

### 判据 B：归因（必须与 base 同看）

| base-2B 的 recall | 读作 |
|---|---|
| 也 < 10% | 塌陷来自**底座**，不是微调引入 |
| 明显更高 | **微调引入了塌陷**——这是比准确率更值得写的发现 |

### 判据 C：同配方对照

ReJev-2B 与 Tev1-4B 的 recall 若**同向塌陷**，结论从「我们的 2B 塌了」升级为「**tev1 配方整体出了分布就塌**」——归因强度完全不同，也是本轮的最终产出形态。

### 判据 D：强度

|ReJev-2B 准确率 − 50.6%| ≤ 10pp ⇒ 2B 与 4B 复刻在分布外同档。

### 无论结果如何都要写入报告

① 与上游数字的对照表；② 四个模型的 recall / FPR / 混淆矩阵；③ 协议差异表（见下）。

## 风险与权衡

1. **协议不可比**：上游 Qwen 用「原生字母 wrapper + per-task regex」、Jev 用 native choice API、本仓用状态机约束。**只做「同考卷、近似同协议」的声称**，不做「完全同等条件」——沿用 exp003/exp004 的既有做法（`eval_cross.py:11-14`）。
2. **预训练接触不可避免**：phishing 为 PhishNChips 合成邮件、公开可下载。无法排除任一模型在预训练阶段见过。**须显式声明**（与 exp004 效度边界第 5 条同）。
3. **样本量不对称**：主口径 2,000 可用；`tool_risk` 60 条手标且类别主观、`ticket_routing` 27 条有明显天花板效应——**不合并成单一总分**。
4. **gold 来源**：phishing 的标签来自数据集构造规则，非独立人工标注；上游自述「简单 URL 启发式即 91.6%」——**91.6% 是数据集属性，不是部署级安全保证**。这句话必须原样进报告。
5. **无分组键** → 无聚类 bootstrap，区间偏乐观，须声明。
6. **续跑身份闸门**：`eval_cross.py:230-233` 拒绝「有 jsonl 无 meta」，`:246-248` 拒绝任何 meta 差异。**换卷内容必须同时删掉 `/vol/eval/cross-*-exam.jsonl` 与 `.meta.json`**，否则被 `paper_sha256` 挡住。
7. **本卷一次性使用**：考卷与标签均公开。**本轮之后不得据其结果调参**；若将来调参，须在报告里声明该卷已不再是干净留出集。（与「留出集不得用于调参」同源。）
8. **`eval_cross.py` 无 wandb**（镜像未装）。若需曲线须改镜像，**会变更环境指纹**，注意 meta 的可复现性说明——默认**不改**。

## 成本与停止

| 项 | 估算 |
|---|---|
| 4 模型 × 2,087 题，L4 | 每模型 15–30 分钟（4B 更慢） |
| 合计 | **≈ $1.5**（按 exp002 实测 0.26 s/题外推） |
| 硬上限 | **$5** |
| 本地部分（prep / 分析 / 自检） | **$0** |
| 并行 | 与 004 探针训练不冲突（不同 app）；启动前核当时 credit 余额 |

## 范围外

- **不重跑真 Jev**：上游已有 2,087 条真实响应（`jev.jsonl`）与公布数字，重跑要花钱且会把「数据不出境」这个优势丢掉。**直接用其公开数字并标注来源。**
- 不改本仓评测协议（约束机制、双口径、状态机）以迁就上游。
- 不并入发布决策——本轮的产出是 exp005 与报告素材，发布另行决策（见 Issue #6）。
- 不做 `--native-kind` 一类的措辞探索（那是 exp004 的议题，且本卷无对应信息）。

## 相关

- 追踪 Issue：#5（内部追踪项）；发布侧：#6（内部追踪项）
- 上游材料：`resources/tev1/evaluation/public-third-party/`（只读）
- 前序：[003 全量训练与验收](003_full-training-eval.md) · [004 弱项探针](004_weakness-probe.md) · `docs/experiments/exp003-three-way/` · `docs/experiments/exp004-jev-comparison/`
- 复用的现成实现：`src/train/eval_cross.py`（模型与考卷注册、按题约束、看门狗、断点续跑）· `src/train/prep_paper.py`（prep 骨架）· `src/train/compare.py`（四象限 + McNemar，可原样吃新卷）
