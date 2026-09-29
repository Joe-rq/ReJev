# exp002 · rejev-full-v1：全量训练与 holdout 验收（003 v2 执行）

日期：2026-09-24。方案：[003 v2](../../plan/003_full-training-eval.md)（双谱系评审修订，已批准）。
**验收结论：通过（三项预注册判据全过）。**

## 问题与判据（预注册，训前锁定）

> 全量 35,948 条训练后，微调模型在封存 holdout（1,892 条）上与 base 同协议对照。

| 判据 | 要求 | 实测 | 结果 |
|---|---|---|---|
| 微调（约束口径）准确率 | ≥70% | **80.50%**（1523/1892） | ✅ |
| 高于 base | ≥15pp | base 51.11% → **+29.39pp** | ✅ |
| 微调（无约束）有效率 | ≥85% | **100%** | ✅ |
| 训练末段 loss | 滑均 <0.6 | 末段日志 0.22–0.31，最终 train_loss **0.4569** | ✅ |
| 账单 | ≤\$12 | **\$5.31**（rejev 累计，含全部探错） | ✅ |

分段规则：base 实测 51.11% 落「30–60%」段 → 判据为该段规则；段界训前写死。

## 运行身份

| 项 | 值 |
|---|---|
| 代码 | 训练 commit `9a40878`；评测脚本随行记录于 summary |
| 模型 | `openbmb/MiniCPM5-2B` @ `12a3808`（同 v1.a/渲染器锁）；adapter SHA256 `c82857d2…`（manifest 全量） |
| 数据 | `train-full.jsonl` 35,948 条（tev1 重建，与官方口径逐位一致）；holdout 封存 sha `2601d649…` 训前验证通过 |
| 训练 | L4×1 · 15,111s（4.2h）· 峰值 6.6GB · 8,987 步 · 1.69–1.95s/步 · 配方与 exp001 逐字相同 |
| 运行标识 | Modal `ap-V1dzqwr55VLt8spfC4vAQ6` · wandb run `p6feej8j` · 训练/评测 ad-hoc 运行全部 stopped |
| 产物 | `rejev:/artifacts/full-v1/`（adapter＋manifest＋SHA256）；逐题结果 `rejev:/eval/*.jsonl`（按数据纪律留 Volume 不进仓） |

## 结果（约束口径；无约束逐位相同）

| 模型 | 准确率 | 无效率 |
|---|---|---|
| base | 51.11% | 0% |
| adapter | **80.50%** | 0% |

**四象限**（1,892）：都对 852 · 都错 254（13.4%，低于 25% 系统性难关标记线）· 仅 base 对 115 · 仅 adapter 对 671。
**McNemar**：χ²≈391.9，精确双侧 **p=2.94e-96**（显著）。

**分 source**（全部提升；按增益）：

| source | n | base | adapter | Δ |
|---|---:|---:|---:|---:|
| policy | 75 | 0.453 | **0.973** | +52.0pp |
| routing_v2 | 300 | 0.270 | 0.720 | +45.0pp |
| policy_v2 | 600 | 0.407 | 0.768 | +36.2pp |
| mnli | 250 | 0.476 | 0.804 | +32.8pp |
| banking77 | 150 | 0.733 | 0.907 | +17.3pp |
| sst5 | 100 | 0.390 | 0.560 | +17.0pp |
| research_taxonomy_v21 | 192 | 0.849 | **1.000** | +15.1pp |
| ag_news | 75 | 0.747 | 0.840 | +9.3pp |
| boolq | 150 | 0.807 | 0.833 | +2.7pp |

观察：① 约束/无约束逐位相同——base 与微调的无约束输出有效率均 100%（干净单字母），**约束器从未触发**，是纯保险；② 最弱残留：sst5 56.0%（5 分类情感、训练样本仅 2,000）、routing_v2 72.0%（24 选项、base 仅 27%）；③ 已近饱和：boolq、ag_news、research_taxonomy（100%）。

### 报告 §2.1 引用的补充读数（口径登记）

技术报告 §2.1 除上表两行外还引了两组数字，均在**本记录之外**计算，登记于此以备追溯：

| 读数 | 值 | 产出 |
|---|---|---|
| 聚类 bootstrap 95%（base） | [48.53, 53.93] | `src/train/jev_compare.py` |
| 聚类 bootstrap 95%（adapter） | [78.44, 82.54] | 同上 |
| 逐题随机猜（mean 1/n） | 27.66% | `src/train/noskill_baseline.py` |
| 恒定输出最常见字母 | 27.91% | 同上 |
| 恒定输出最常见语义 key | 17.18% | 同上 |
| 按选项数取该组最常见位置 | 29.86% | 同上 |

区间口径：**按 `group_id` 整组重抽**（本卷 924 组），主表 2,000 次、分表 1,000 次、
`seed=20260925`——参数写死在 `jev_compare.py` 的 `bootstrap` 记录里。后三个基线是
**在同一批标签上取众数**得出的**事后描述上界**（非无技能基线），报告 §2.1 已带此限定。

⚠️ **两者的输入都不随本仓发布**（bootstrap 读逐题结果，基线读封存留出集），故第三方
**无法直接复算**这两个块，只能按脚本内写死的口径自行重建输入。

## 执行偏离（按 board 纪律记录）

1. **p99 断言线 1024→1900**：holdout prompt 实测 p99=1480（任务分布本身，与训练 max 1526 同源），非异常；评测预算按实测重估。
2. **静默 CPU 事故**：首两次 base 评测因 `.to(model.device)` no-op 在 CPU 上等同挂死（白烧 ~$0.5）；根因是负例闸误报逼出的改写。修复＋闸适用范围修正，见 changelog 与 `src/align/negatives.md`。
3. base 评测与训练拆为两阶段（方案即如此设计）；dev sanity 8,192→200 条抽样（方案原文即 200）。

## 复现路径

```bash
uv run python src/train/prep_full.py                      # 数据准备＋封存校验
modal run src/train/full_train.py                         # 全量训练（幂等：manifest 在即跳过）
modal run src/train/eval_holdout.py --model-kind base    --split holdout
modal run src/train/eval_holdout.py --model-kind adapter --split dev    # sanity
modal run src/train/eval_holdout.py --model-kind adapter --split holdout
uv run python src/train/compare.py <base.jsonl> <adapter.jsonl> --mode constrained
```

⚠️ 上面的 `modal run` 是**本轮当时的跑法**，如实保留。但本仓此后的纪律是
**`modal deploy` + `.spawn()`**：`modal run` 会把本地进程绑在远程任务上，本地被终止时
Modal 会向远程发 cancellation 把任务杀掉（实测发生过）。全量训练与评测属长任务，请照
[REPRODUCING.md](../../../REPRODUCING.md) 的触发脚本方式跑。

## 结论与下一步

**第一阶段「Choice 单字母分类」验收通过**：MiniCPM5-2B + tev1 数据配方 + 官方 TRL 路线在 2B/LoRA/1epoch 下达成 80.5%，显著且全面地高于底座。未决事项按意向书边界不在本轮：Score/Noul/概率校准、与 tev1 数字横比、发布。

候选下一步（待项目所有者定）：① 效率轮（packing 探针→全量提速 1.5–2.5×，已有讨论与探针设计）；② 弱项轮（sst5/routing_v2 定向增强）；③ 进入第二阶段议题（Score/校准是否立项）。
