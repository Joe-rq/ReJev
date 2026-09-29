# 002 · 首轮小批训练（smoke）方案

> **发布说明（2026-09-29 追加，正文一字未改）**：本文件是**训前锁定**的预注册记录，
> 保留原文是刻意的。紧接着的状态行**已被后续执行覆盖**——本方案**已批准并执行**，
> 结果见 [exp001 记录](../experiments/exp001-smoke-v1/README.md)。

日期：2026-09-24。状态：**待项目所有者审批——批准前不启动任何远程 GPU 任务。**

## 可证伪问题（本轮唯一要回答的）

> MiniCPM5-2B 经官方 TRL `SFTTrainer` 路线 ＋ ReJev 渲染数据，训练管线能否**端到端跑通且协议零错误**（单 BOS、loss 只落 assistant span、监督＝字母＋`<|im_end|>`）？

不回答「模型好不好」——准确率与全量训练是下一轮的事。

## 配方（每项标出处）

| 项 | 值 | 出处 |
|---|---|---|
| 路线 | TRL `SFTTrainer` ＋ PEFT LoRA；**messages 格式 ＋ `assistant_only_loss`** | MiniCPM 官方 `docs/finetune/trl.md`（2026-09-24 读原文） |
| LoRA | r=16 · α=32 · dropout=0.05 · all-linear（q/k/v/o/gate/up/down） | 同上 |
| 训练参数 | **1 epoch**（官方 2，smoke 改 1）· bs 2 × grad-acc 2 · **lr 2e-4（官方值，smoke 不调参）** · cosine · warmup 0.03 · bf16 · seq 2048 · packing=False · seed 42 | 同上 |
| 数据 | `rejev-train` 随机抽 **500 条**（seed 42），用 `sft/messages` 导出 | `data/rejev/`（渲染 manifest 在档） |
| 模型 | `openbmb/MiniCPM5-2B` @ `12a3808`（同 v1.a/渲染器同锁）；bf16、sdpa、梯度检查点 | 真相源 R5 |
| GPU | **L4 ×1**（$0.80/hr；2B bf16 LoRA 峰值估 ~10 GB < 24 GB） | Modal 价格；旧项目实测 L4 单价 |
| 预算 | **单轮 ≤ $1**（首次大头是冷启动：模型 ~4.5 GB 下载＋镜像构建）；总闸＝spend limit $0（已设并复核） | 内部账户核验记录（未随本仓发布） |
| 墙钟/停止 | function timeout 1800 s；手动停止 `modal app stop <app-id>`；无早期停止 | 旧项目结构 |

## 协议防线（全部有 v1.a 实测背书）

1. **数据侧**：prompt 文本已含 post_processor 前置的单个 `<s>`——训练加载必须 `add_special_tokens=False`，**防双 BOS**（v1.a C4 修正后事实）。
2. **监督侧**：`assistant_only_loss` ＋ 官方训练模板把 `content + '<|im_end|>'` 包进 `{% generation %}`——与 v1.a C5「监督到 `<|im_end|>`、轮间 `\n` 不入监督」一致（官方路线逐字对上）。
3. **训练前断言**（进脚本，不过即 fail-fast）：抽 3 条 tokenize 后核对——首 token id=0（单 BOS）、被监督 token 的 decode 恰为「字母＋`<|im_end|>`」。
4. **pad**：MiniCPM 自带 pad=`</s>`（id 1，与 eos[0] 同 id）——只用于 padding，不参与 loss，不覆盖。

## 验收（一翻两瞪眼）

- [ ] train_loss 有限且下降（字母 24 类初始理论量级 ≈ ln24≈3.2 附近起步）
- [ ] 训练中抽 1 个 batch 打印：labels≠-100 的 token 恰为「字母＋`<|im_end|>`」
- [ ] 训练后用 `rejev-holdout` 抽 5 条生成（`temperature=0`、关思考、`max_tokens=8`）：有效率 >0 且为单字母
- [ ] 显存峰值、耗时、**按 app 核对账单 ≤$1**，全部记录归档
- [ ] adapter 存 Volume，附 manifest（代码 commit、模型 revision、数据清单 hash）

## 明确不做

不做全量、不调超参、不看准确率排名、不动旧 Volume、不发布任何产物。

## 启动条件

项目所有者批准本方案 → 写 `src/train/smoke_train.py`（Modal 脚本，借 LLM01-kaizhi `sft_train.py` 的 smoke 分档/显存记录/Volume 保存结构，模型与数据部分按本方案重建）→ 先本地 dry 断言 → 远程跑 → 按既有实验记录的格式归档（见 [docs/experiments/](../experiments/)）。
