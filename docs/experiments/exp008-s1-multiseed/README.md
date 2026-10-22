# exp008 · S1 多种子、独立终评与概率校准

> 状态：**S1 完成**（四臂训练、六臂候选评分、温度校准、置换、错误与行为分析）。
> S2 报告增强与 S3 双谱系评审由 #122 承接；4B 配对对照由 #123 暂缓。
> 本目录归档 S1 的**分析产物**；原始训练/评分 JSONL、adapter 与 checkpoint 在私有 Modal Volume，不在 Git。

## 1. 要回答的问题

同配方、仅训练 seed 不同的四条 LoRA 臂（seed 42/17/29/41），相对未微调底座 base-C0 的**分类收益是否跨训练随机性保持**；分类收益是否伴随**概率质量**改善；决策在多大程度上受**选项顺序**扰动；错误能否按预定义类型学归类。

## 2. 协议与身份

| 项 | 值 |
|---|---|
| 方案 | `docs/plan/009_report-enhancement-experiments.md`（2026-10-17 批准；S1 协议未改）。**该方案与 S0/S1 manifest 随本发布集分发（脱敏公开副本）**；下列 SHA256 分两栏：source-of-record 指冻结时的源件，public copy 指本发布集里的脱敏件。manifest 内部的 SHA 字段（如 S1 manifest 记录的 plan/S0 指纹）一律指 source-of-record，核对公开副本请用 public copy 值 |
| S0 manifest | `009-s0-manifest.json`；source-of-record SHA256 `5c33d48ba9d03fac4bbfef71317a054642467ef8432381f4069104d4c444f418`；public copy `504a7fcda857fc4e21d257a45100857f6348cc9485a6ab382d80434f0fc64b17` |
| S1 manifest | `009-s1-manifest.json`；source-of-record SHA256 `41f1a53eea126acbcef6cbc5936443018ba8daf60b1feb89808bf5817d87f6e3`；public copy `3787d14fc666f3a696ddf4d919d1b5c6cf94ebd3b15df436377113df2bdb23b7` |
| 底座 | MiniCPM5-2B，revision `12a3808a956f869c767195e9266b59c4d21d92e2` |
| 训练集 | `train-clean` 30,987 行，SHA256 `bce0ae242975f23854a34341a11275de4cec2f7ec5d889cc3f726c78fe5b664d`（复用历史；S0 冻结） |
| 评测切分 | dev2 500 / calibration2 1,500 / test2 3,800（1,659 families）/ test2-perm 3,800；四文件 SHA 见 S1 manifest `eval2_files` |
| 配方 | LoRA r16 / α32 / dropout 0.05 / all-linear；1 epoch；bs 2×2；lr 2e-4；cosine；warmup 0.03；seq 2048；assistant_only_loss；bf16；packing=False |
| RNG | 协议 ①：`get_peft_model` 前显式 `torch.manual_seed(seed)`；`SFTConfig(seed=seed)` 控制数据顺序；不另设 cuda 种子 |
| 评分 | `s1-candidate-v2`：最后一个 prompt token 的 logits → 有效选项 mask → softmax；A→54…X→77；raw logits/probs 落盘 |
| 校准 | 逐臂 `T_a` 在 calibration2 拟合（logits 稳定 log-softmax；网格 [0.05,20]，4,001 点）；test2 只应用，不重拟合 |
| 区间 | 按 `group_id` 的 family-clustered bootstrap 10,000 次，seed `20261018` |
| 置换 | `shift = 1 + (int(sha256(id),16) mod (n_options-1))`，左循环移位；保留语义 key，按 key 还原后统计 |
| 错误分析 | family 单位；test2 分母 1,659；类型优先级：状态解析失败 → 选项顺序敏感 → 恒常策略 → 标签错误 → 其他 |

数据隔离前提（C8）：S0 审计 PASS——dev2/calibration2/test2 两两 statehash 与 family 交集均为 0；与历史 50,914 个 statehash、25,672 个 family 的交集均为 0。近重复**未做**（披露项）。

## 3. 运行清单

| seed | steps | wall | 峰值 GPU | 末 50 步 loss | cumulative | adapter SHA256 | config SHA256 | W&B | metered |
|---:|---:|---:|---:|---:|---:|---|---|---|---:|
| 17 | 7,747 | 15,740.6s | 6.53GB | 0.287556 | 0.463810 | `bbb1403e…` | `484093d8…` | `ecd5pw27` | $4.30145456 |
| 29 | 7,747 | 15,751.0s | 6.60GB | 0.270453 | 0.470156 | `cb6ac86d…` | `7906377c…` | `d8cku50z` | $4.31092469 |
| 41 | 7,747 | 15,470.1s | 6.61GB | 0.215001 | 0.490325 | `8227f9ac…` | `d8839a4d…` | `tcyn90l0` | $4.23096959 |
| 42 | 7,747 | 15,646.2s | 6.62GB | 0.268248 | 0.454274 | `87a07194…` | `e82417bf…` | `yfi4gjnp` | $4.27495292 |

四臂均 health PASS、0 failure；完整 adapter/config SHA、尝试记录、账单来源见 S1 manifest `runs`。
`hist` 臂是历史 clean-r16 adapter（`294b07beac96e42e4a69af41c1121bc2aa0a18cf4634ef2ea9e17b8032b9973e`，config `bf854ccc…`），只作复现检查，**不进 seed 方差集合**。

## 4. 主结果（test2，candidate scoring）

| 臂 | accuracy | 覆盖 |
|---|---:|---:|
| base-C0 | 0.5232 | 1.0 |
| hist | **0.8411** | 1.0 |
| seed42 | 0.8205 | 1.0 |
| seed17 | 0.8121 | 1.0 |
| seed29 | 0.7797 | 1.0 |
| seed41 | 0.7984 | 1.0 |

四 seed 均值 **0.8027**、样本 SD **1.78pp**；四臂对 base 的同题配对差均为正（按 seed 42/17/29/41）：
seed42 **+29.74** [+27.76, +31.69]、seed17 **+28.89** [+26.97, +30.83]、seed29 **+25.66** [+23.79, +27.48]、seed41 **+27.53** [+25.61, +29.42]（pp）。
训练末 50 步 loss 与 test2 accuracy 不单调（seed41 loss 最低、accuracy 第三）——仅观察，不作因果。

## 5. C1–C8

| 主张 | 读数 | 判读 |
|---|---|---|
| **C1** 多种子收益保持 | 四 seed 配对差均正，区间均不含 0；均值 +27.95pp、SD 1.78pp | 正向性成立（`positive_direction=True`、`zero_arm=False`）；四 seed 各任务**任务级宏指标**（accuracy / macro-F1 / macro recall）方向一致；单类 recall 层面有 **30 处**（臂×任务×类）低于 base（如 seed41 banking77/verify_my_identity 1.00→0.25、seed42 routing_v2/route_1 0.91→0.64、seed42 research_taxonomy_v21/science-medicine 1.00→0.75），**不作逐类一致主张**，完整逐类 recall 见 `analysis.json`；3 自由度，SD 区间宽；未挑最佳 seed |
| **C2** 概率质量 | 五条微调臂 raw NLL/Brier/ECE 均低于 base；逐臂校准后三项进一步下降；family-clustered 区间均不含 0 | 三指标同向；不主张校准提升准确率 |
| **C3** 温度不改变 argmax | T：base 4.922、hist 1.481、seed42 1.467、seed17 1.558、seed29 1.420、seed41 1.446；均未触 [0.05,20]；逐臂 mismatch **0/3,800** | 通过 |
| **C4** 方向与历史一致 | test2：hist 0.8411、四 seed 0.7797–0.8205、base 0.5232；历史 81.13% 为 holdout1 单集（n=1,892） | 只作方向并列；不作数值比较/差值归因 |
| **C5** 选项顺序敏感性 | base 一致率 0.6508（high，flip 0.3492）；hist 0.9600、seed17 0.9616、seed41 0.9542（low）；seed42 0.9492、seed29 0.9371（uncertain） | 按预锁定效应量界；`uncertain` 不写成低敏感；不主张语义机制 |
| **C6** 错误类型 | base 806 错（顺序敏感 473／恒常 37／标签 296）；hist 390（105/7/278）；四 seed 423–479（102–150／7–12／287–335）；状态解析失败与其他均 0（冻结的 `analysis.md` 沿用简称“解析失败”，与正文同指） | 计数描述，不作显著性；四 seed 共同错误 274、分歧 365；base-only 291、hist-only 11 |
| **C7** 复现检查 | seed42−hist = **−2.05pp** [−3.27,−0.88] | 差异读数；构成＝历史未受控 LoRA 初始化＋环境/非确定性，**不可分解**；不写「复现成功」，不单独归因环境；plan/S0 未见预先声明的数值范围 |
| **C8** 数据隔离前提 | S0 审计 PASS；历史与三切分交集均为 0 | S0 已完成；本轮分析未重跑；近重复未做；**公开源预训练接触**：test2 约 32% 来自真实公开语料（mnli/boolq/banking77），无法排除底座预训练接触；零交集只覆盖实际检查的精确 statehash 与登记 family，不外延到语义近重复或预训练接触 |

## 6. 校准、置换与错误分析（摘要）

| 臂 | T | raw NLL | cal NLL | raw ECE | cal ECE |
|---|---:|---:|---:|---:|---:|
| base-C0 | 4.9221 | 2.5433 | 1.1120 | 0.3854 | 0.1102 |
| hist | 1.4806 | 0.4446 | 0.4187 | 0.0586 | 0.0193 |
| seed42 | 1.4673 | 0.4814 | 0.4547 | 0.0678 | 0.0215 |
| seed17 | 1.5579 | 0.5121 | 0.4784 | 0.0817 | 0.0262 |
| seed29 | 1.4198 | 0.5295 | 0.5128 | 0.0596 | 0.0135 |
| seed41 | 1.4455 | 0.5275 | 0.5007 | 0.0695 | 0.0176 |

置换一致率（test2 原题 vs 语义 key 置换变体）：base 0.6508（flip 0.3492）、hist 0.9600、seed42 0.9492、seed17 0.9616、seed29 0.9371、seed41 0.9542。位置偏好（预测字母计数）：base 偏 C（1,402/3,800），五条微调臂偏 A（1,084–1,153）；只作诊断。

完整的分任务 accuracy / macro-F1 / recall / 混淆矩阵、选项数分层、选择性预测曲线、per-source 配对差与案例见 [`analysis.md`](analysis.md) 与 [`analysis.json`](analysis.json)。

## 7. 成本

- 四臂训练合计 **$17.11830176**（逐臂见上表）；本月 `deployed_apps` 合计 **$19.70799503**（含 S1 训练与评测；月度 breakdown 未按 app 拆分，S1 专属成本以该合计为上界），在方案 $30 credit 上限内。
- 本月 metered **\$24.33265711**（含 Volume **\$4.62388736** 等非 S1 直接项）；按「仅用实际可用 credit、零自付」的既定边界运行（账户状态不随发布集分发）；无遗留 Modal 任务。
- 未发生 4B 下载/训练/评测；#123 未动。

## 8. 产物与校验

| 文件 | SHA256 |
|---|---|
| `analysis.json` | `695e9054744ea7b21a6e011e07e082be0cf86c957956a0b3a665f50e6f8a84ff` |
| `analysis.md` | `97e4aba4634cb80679a0eff387a37ead1947d079864a968722c407a8b6ee28f6` |
| `analysis-manifest.json` | source-of-record `f5dee8f59346c32f5533a34eaf91d8aa0c32109b7f5e33e64afda1ab6ca8ef18`；public copy `18d724cc85608dd6d65ce85a3e4413a1a5a440c6d9617d00d7dfa75db49b52e4` |

- 分析运行：**2026-10-19（本地 10:05–10:08）**，命令
  `uv run python src/analysis/s1_analysis.py --scores-dir <scores-dir> --out <out-dir>`，CPU 3m04s。
- 24 个**评分输出文件** SHA 与 `analysis-manifest.json` 的 `inputs` 一致；四个评测数据文件 SHA、per-arm adapter/config、评分三源码（`eval_candidates.py` / `candidate_math.py` / `s1_spec.py`）SHA 与 S1 manifest 一致；2 个分析输出 SHA、S1/S0 manifest SHA、分析源码 SHA 见 `analysis-manifest.json`；`load_all_validated` 实跑 57,600 条记录通过。
- S1 manifest 的 `notes[0]`「S1 仍未产生任何远程结果；本 manifest 在 seed42 启动前冻结」是**启动前快照**；最终 `runs` 已登记四条 finished 运行，勿以该句判断当前状态。
- `analysis.md` 的 C2 首表列口径为「cal(臂) − raw(base)」；base 行的 cal−base 即 base 自身 cal−raw，单独读该行时注意。
- ⚠️ `analysis-manifest.json` 的 `created` 字段为分析源码硬编码 `2026-10-18`，**实际运行日为 2026-10-19**；未改冻结分析源码（`s1_analysis.py f93424a7…`、`metrics.py 30d4b575…`），以上述实际运行时间与本目录 SHA 为准。

## 9. 限制与未做

- 主指标是候选评分的同题 accuracy；温度是正标量，不改变 argmax，不主张校准提升准确率。
- 置换一致率只支持「对选项顺序扰动的敏感性」，不主张语义机制；高一致率不等于读取语义内容。
- test2 无 OOD 臂；历史 81.13% 与 test2 是不同题集，只作方向并列，不作差值归因。
- **公开源预训练接触**：test2 约 32% 来自真实公开语料（mnli/boolq/banking77），无法排除底座预训练接触；合成源约 68% 不受此限。零交集只覆盖本项目实际检查的精确 statehash 与登记 family，不外延到语义近重复或预训练接触。S0 的其余披露（模板/词表同源、research_taxonomy_v21 构造标签、banking77 `none` 增广、boolq 位置不平衡、routing_v2 长描述先验、variant 0 规范顺序）见 `009-s0-manifest.json` 的 `disclosures`。
- C7 差异含历史未受控 LoRA 初始化＋环境/非确定性，不可分解；不是 seed 方差。
- 四 seed 只有 3 个自由度；训练间差异与样本抽样不确定性分开报，未挑最佳 seed。
- 覆盖率为构造结果；选择性曲线只报告，不据 test2 选择阈值。
- 未做：S2 报告增强（#122）、S3 双谱系评审/图件实看、4B 配对对照（#123）、近重复检查。

## 10. 链接

- #111 S1 实验与分析
- #122 S2/S3 报告与评审（blocked by #111 S1 分析）
- #123 4B 配对对照（暂缓）
- S1 训练入口评审（评审归档不随本发布集分发）
- S1 评测/分析代码评审（评审归档不随本发布集分发）
