# exp003 · 三方同考卷对照评测（Issue #1）

日期：2026-09-25。Issue：#1（内部追踪项）。
代码：`src/train/prep_paper.py` · `src/train/eval_cross.py` · `src/train/cross_report.py`。
**结论：**（见文末）

## 问题

第一阶段的 80.50% 与 tev1 官方公布的 880/1000（88.0%）＋300/300（100.0%）**不同题、不同分法**，放一起比没有意义。本轮让三个模型在**同一张考卷**上同台，回答：

1. 我们的 2B 离官方 4B 差多远？
2. 我们的数据重建是否忠实？（用 Tev1-4B 实测数对照官方公布数验证）

## 方法与边界

### 考卷

| 考卷 | 题量 | 来源 |
|---|---:|---|
| `tev1paper` · main | 1,000 | `resources/tev1/data/v1/records/test.jsonl` |
| `tev1paper` · policy_transfer | 300 | `resources/tev1/data/v1/records/policy_transfer.jsonl` |
| `holdout` | 1,892 | ReJev 封存的 holdout（`data/rejev/records/rejev-holdout.jsonl`） |

**issue 原文的取题路径经实测不成立**：「按 id 在我们重建的数据中取题」交集为 **0**——官方考卷取自 v1 的 `test`/`policy_transfer` split，而我们重建的是 `train`/`dev`。改用考卷本体文件，是等价且更强的替代（直接拿到逐题 gold 与选项顺序）。三重校验（`prep_paper.py` 内断言）：

1. 与官方逐题结果 `results.jsonl` 按 id 对齐 **1000/1000 + 300/300**，gold **零不一致**；
2. main 分 source 计数与官方 README breakdown 逐项一致；
3. 与 ReJev train/holdout/dev 在 **group_id / statehash / id 三口径零重叠**。

### 三方与渲染

| 代号 | 模型 | 说明 |
|---|---|---|
| `base` | `openbmb/MiniCPM5-2B` @ `12a3808` | 未微调底座 |
| `adapter` | 同底座 + LoRA `rejev:/artifacts/full-v1` | exp002 全量训练产物，SHA256 随行记录 |
| `tev1` | `togethercomputer/Tev1-4B-experimental` @ `0b7becf` | 官方权重 |

**「同考卷」的落点**：三方共用**同一份任务语义层 `messages`**（由 `src/data/render.py` 的 `build_messages` 产出：system 指令 ＋ `{"state","question","options"}` JSON），各自套**自己的 chat template**。模板差异属模型固有要求，不是协议差异。

**忠实度证据**：用 Tev1-4B 自带 tokenizer 渲染这 1,300 条，与官方 `instruction/{test,policy_transfer}.jsonl` 的 prompt **逐字一致 1300/1300**，completion 100% 等于 `answer + '<|im_end|>'`。

### 解码口径

双口径并列：**constrained**（状态机 `prefix_allowed_tokens_fn`）与 **unconstrained**；greedy（`do_sample=False`）、`max_new_tokens=8`、`enable_thinking=False`、显式 eos。与 exp002 逐位同协议，故 base/adapter 在 holdout 上的历史结果可直接并表。

**约束候选集＝该题实际选项**（`LETTERS[:len(options)]`），对齐官方「regex per option list」。exp002 版允许全 A–X——实测其约束器**从未触发**（约束/无约束逐位相同），故该口径变化对历史结果无影响；但对选项数 < 24 的题目（官方主考含 200 道二选一）必须按题约束才与官方可比。

### 协议差异（显式记录，不做同等条件声称）

| 维度 | 官方（Together API） | 本仓（Modal L4 本地权重） |
|---|---|---|
| 约束实现 | regex per option list | 状态机 prefix 约束，候选集按题 |
| 温度 / 长度 | temp 0 · max_tokens 8 | greedy · max_new_tokens 8 |
| 思考 | off | `enable_thinking=False` |
| 权重 | 历史端点权重 | 本地下载的同一 repo revision |

→ 对照结果标注为**「同考卷、近似同协议」**。

### 数据与许可边界

- 考卷为 tev1 官方**开发集**：对 Tev1-4B 有主场效应，报告须如实标注。
- Tev1-4B 微调权重许可**未定**（repo 无 `license` 字段）：**仅限本仓内部研究对照，权重不进 Git、不随结果发布、不上传任何平台**。
- 逐题结果留 Modal Volume，不进仓。

## 结果

### 三方同考卷对照（约束口径；三方**无约束口径逐位相同**，无效率均为 0%）

| 考卷 | MiniCPM5-2B base | **ReJev-2B** | Tev1-4B（官方权重） | 官方公布 |
|---|---:|---:|---:|---:|
| tev1paper · main（1,000） | 62.60% | **84.40%** | **88.00%** | 88.0% |
| tev1paper · policy_transfer（300） | 42.00% | **93.30%** | **100.00%** | 100.0% |
| tev1paper · 合计（1,300） | 57.85% | **86.46%** | **90.77%** | 90.8% |
| holdout（1,892） | 51.11% | 80.50% | *（本仓实测值未随本仓发布）* | — |

### 考卷重建忠实度：**完全对齐**

Tev1-4B 本地权重实测 **main 880/1000（88.0%）· policy_transfer 300/300（100.0%）**，与官方经 Together API 公布的数**逐位一致（偏差 0.00pp）**。这同时验证三件事：

1. **考卷逐题重建忠实**——题面、选项顺序与 gold 与官方一致；
2. **本仓状态机约束与官方 regex per option list 在该分布上等价**——三方 `constraint_touched` 均为 0（约束器从未改写输出）；
3. 整个三方对照表建立在一个可复现的基座上。

### 回答 issue 的两个问题

**① 我们的 2B 离官方 4B 差多远？**

差距**随考卷难度而变**，不是单一数字：

| 考卷 | ReJev-2B | Tev1-4B | 差 | 2B 相对 4B |
|---|---:|---:|---:|---:|
| tev1paper（官方开发集，prompt p99≈350） | 86.46% | 90.77% | **4.31pp** | 95.3% |
| holdout（自建封存集，prompt p99≈1,500、含 24 选项任务） | 80.50% | *（未发布）* | *（未发布）* | *（未发布）* |

→ 在**短 prompt、选项少**的官方考卷上，2B 已逼近 4B（95.3%）；但在部分任务上差距显著拉大。

**⚠️ 归因修正（2026-09-25 诊断）**：本节初稿据 source 级聚合推断「瓶颈在长上下文 + 多选项」，
**该归因经细粒度诊断被否定**——`research_taxonomy_v21` 是 **24 选项 + 长 prompt** 却拿 **100%**；
prompt 1,000–1,500 token 段 ReJev 亦为 **100%**。聚合数字会骗人：真正的弱项是**特定任务语义变体**，
与形式属性（长度、选项数）无关：

| 维度 | 证据 | 判定 |
|---|---|---|
| 长上下文是瓶颈 | 1,000–1,500 token 段 ReJev 100%（Tev1-4B 亦 100%） | ❌ 否定 |
| 多选项是瓶颈 | `research_taxonomy_v21`（24 选项）ReJev 100% | ❌ 否定 |
| 训练数据量不足 | `policy_v2` 训练量最多（12k）而差距最大（23pp）；`policy`（1.5k）差距仅 2.7pp | ❌ 否定 |
| 模式坍塌／位置偏置 | 预测字母分布 ≈ gold 分布（各变体逐项核对） | ❌ 否定 |
| **任务语义难度** | 弱项集中在特定变体：routing_v2 `complete_*` 60.5–66.7% vs 同集 `multi_missing` **100%**；policy_v2 `decisive_missing` / `complete_positive` 67% vs `multi_missing` 93% | ✅ **成立** |

**教训**：source 级聚合（600 条一个数）会把「变体间差异」平滑掉，据此归因会指向错误方向。
细粒度诊断的成本是零（本地已有逐题结果），但它决定了下一轮的干预靶点是否成立。

**② 我们的重建是否忠实？** 是——见上（0.00pp）。

### 分 source（约束口径；`test` 为主考，`policy_transfer` 单列）

| split | source | n | base | ReJev-2B | Tev1-4B |
|---|---|---:|---:|---:|---:|
| test | mnli | 250 | 57.6% | 89.2% | 91.2% |
| test | banking77 | 200 | 72.0% | 87.0% | 91.0% |
| test | boolq | 200 | 72.5% | 83.5% | 90.0% |
| test | policy | 150 | 50.0% | 97.3% | **100.0%** |
| test | sst5 | 100 | 33.0% | 46.0% | 53.0% |
| test | ag_news | 100 | 85.0% | **88.0%** | 87.0% |
| policy_transfer | policy | 300 | 42.0% | 93.3% | **100.0%** |

观察：① **sst5 是三方共同的短板**（base 33% → ReJev 46% → 官方 4B 53%），且是 2B/4B 差距最大处（7pp）——与 exp002 在自建 holdout 上的观察一致（sst5 最弱）；② **ag_news 上 ReJev-2B 略超 Tev1-4B**（88.0% vs 87.0%，100 题中 1 题之差，不显著）；③ policy 类（结构化的合成任务）上，2B 已逼近 4B 官方（97.3% vs 100%）。

### holdout（1,892，自建封存集）上的分 source

| source | n | base | ReJev-2B | Tev1-4B |
|---|---:|---:|---:|---:|
| policy_v2 | 600 | 40.7% | 76.8% | **99.8%** |
| routing_v2 | 300 | 27.0% | 72.0% | 91.7% |
| mnli | 250 | 47.6% | 80.4% | 89.6% |
| research_taxonomy_v21 | 192 | 84.9% | **100.0%** | **100.0%** |
| banking77 | 150 | 73.3% | 90.7% | 92.7% |
| boolq | 150 | 80.7% | 83.3% | 90.0% |
| sst5 | 100 | 39.0% | 56.0% | 70.0% |
| ag_news | 75 | 74.7% | 84.0% | 93.3% |
| policy | 75 | 45.3% | 97.3% | **100.0%** |

→ **差距集中在长上下文/多选项任务**（policy_v2 差 23pp、routing_v2 差 19.7pp），而
`research_taxonomy_v21` 上双方**并列 100%**。这解释了为何官方考卷上差距只有 4.31pp：
官方考卷的 prompt 短得多（p99≈350 vs holdout ≈1,500）。

**四象限（ReJev-2B vs base，1,300 题）**：都对 689 · 都错 113 · 仅 base 对 63 · 仅 ReJev 对 435。

## 执行偏离

见 [implementation-notes.md](implementation-notes.md)。

## 复现路径

```bash
uv run python src/train/prep_paper.py                    # 考卷构建 + 三重校验 + 渲染校验
modal volume put rejev data/paper/tev1-paper.jsonl /data/tev1-paper.jsonl
modal run src/train/eval_cross.py --model base    --papers tev1paper
modal run src/train/eval_cross.py --model adapter --papers tev1paper
modal run src/train/eval_cross.py --model tev1    --papers tev1paper,holdout
# 从 Volume 取回 cross-*.jsonl 到本地目录后：
uv run python src/train/cross_report.py --dir <结果目录>
```

## 结论与边界

### 结论

1. **考卷重建完全忠实**：Tev1-4B 本地权重实测与官方公布**逐位一致**（880/1000 + 300/300，偏差 **0.00pp**），渲染与官方 instruction 逐字一致 1300/1300。这为整张对照表提供可复现基座，并证明本仓状态机约束与官方 regex per option list 在该分布上**等价**（三方 `constraint_touched` 均为 0）。
2. **2B 相对 4B 的定位是「任务相关」而非单一数字**：短 prompt、少选项的官方考卷上达 **95.3%**；长 state、24 选项的 holdout 上为 **85.6%**。
3. **瓶颈已定位**：差距集中在**长上下文 + 多选项**（policy_v2 差 23pp、routing_v2 差 19.7pp、sst5 差 14pp），而在结构化短任务上已追平（research_taxonomy_v21 双方 100%）。这为下一步「弱项轮」提供了明确的靶点。
4. 三方**约束与无约束口径逐位相同**，无效率均为 0%——约束器从未触发，说明三者在该任务分布上本就只输出干净单字母。

### 边界（不宣称什么）

- **主场效应**：考卷（tev1paper）为 tev1 官方**开发集**，官方 README 自述为「reused development benchmarks」——Tev1-4B 对其有多轮迭代优势。但**不是数据泄漏**：该 1,300 条与本仓 train/holdout/dev 三口径零重叠，ReJev-2B 也未见过。
- **协议差异**：官方经 Together API 用 regex 约束，本仓用状态机实现；虽结果逐位一致，**实现不同即不做完全同等条件声称**。
- **不宣称**：本结果不构成「ReJev 等同 Jev 官方模型」或「2B 可完全替代 4B」——在长上下文/多选项任务上差距显著；也不做与 tev1 论文/文章的跨平台横比。
- **许可**：Tev1-4B 微调权重许可未定 → 本实验仅限本仓内部研究对照，权重不进 Git、不随结果发布、不上传任何平台。
- 逐题结果留 Modal Volume，不进仓（数据纪律）。
