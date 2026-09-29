# exp001 · rejev-smoke-v1：首轮小批训练（管线可证伪）

日期：2026-09-24。方案：[002](../../plan/002_first-smoke-training.md)（已批准）。可证伪问题：**训练管线端到端能否跑通且协议零错误**——不评测模型好坏。

## 运行身份

| 项 | 值 |
|---|---|
| 代码 | commit `326ff04`（修复后）；首次运行为 `32e1746` |
| 模型 / tokenizer | `openbmb/MiniCPM5-2B` @ `12a3808`（Apache-2.0） |
| 数据 | `smoke-500.jsonl`＝`rejev-train` 随机抽 500（seed 42）；`holdout-5.jsonl`＝`rejev-holdout` 抽 5（seed 42）；来源与许可见 [NOTICE](../../../NOTICE) |
| 模板 | 官方 TRL 训练专用模板（`{% generation %}` 包 `content+'<\|im_end\|>'`）；推理用原模板 `enable_thinking=False` |
| 环境 | Modal `nvidia/cuda:12.6.0-devel-ubuntu22.04` + Python 3.12 · torch 2.7.0 · transformers 5.6.2 · trl 1.13.0 · peft 0.21.0 |
| GPU | L4 ×1（峰值显存 **6.52 GB** / 24 GB） |
| 配置 | LoRA r16/α32/dropout 0.05/all-linear · 1 epoch · bs 2×ga 2 · lr 2e-4 · cosine · warmup 0.03 · seq 2048 · `assistant_only_loss=True` · seed 42 · packing off |
| 运行标识 | Modal app `ap-25pY7GZtDDU1wDkPUXrSga` · wandb run `omhn1att` |
| 产物 | adapter＋manifest → Modal Volume `rejev:/artifacts/smoke-v1`（几十 MB；模型缓存走容器临时盘，零存储残留） |

## 结果

- **协议验收全过**：远程 token 断言 3/3（单 BOS、监督＝字母＋`<|im_end|>`）；SFTTrainer 实弹 batch 监督 span decode＝`'B<|im_end|>'`（002 验收第 2 条）。
- **训练正常**：train_loss **1.086**（24 类均匀初始理论 ≈ln24≈3.18，单 epoch 明显下降）· 239.3 s · 峰值 6.52 GB。
- **生成冒烟**：holdout 5 条 greedy（`temperature=0` 口径、显式 `eos=[1,130073]`、关思考）——raw 输出依次为 `A<|im_end|>`、`C<|im_end|>`、`B<|im_end|>`、`C<|im_end|>`、`C<|im_end|>`：**格式 5/5 符合「单字母＋结束标记」协议**。与 gold 的对照未记录（smoke 不评准确率，留下一轮评测）。
- **账单**：第一次 \$0.085（收尾崩溃）＋第二次 \$0.090 ＝ **\$0.175**（≤\$1 预算 ✓）。运行后 `modal app list` 无残留任务、app stopped。
- 指标曲线：wandb run `omhn1att`（loss/lr 按步记录；仅指标，无样本正文）。

## 偏离与教训

1. **第一次运行收尾崩溃**（`32e1746`）：容器无 `git`，`code_commit` 现取导致 `FileNotFoundError`；异常退出未 commit Volume，**adapter 未持久化，$0.085 只买到日志**。修复：`code_commit` 改由调用方传入。
2. **生成解析假阴性**：`skip_special_tokens=False` 留下 `<|im_end|>` 字面量，`'A<|im_end|>'` 被判 invalid——manifest 记 `gen_valid: 0/5`，人工判读 raw 为 **5/5 协议正确**。解析已修（`skip_special_tokens=True`），未重跑（事实由 raw 记录充分）。**教训：判 invalid 前先看 raw。**
3. 首次运行日志被 `tail -60` 截断——远程运行必须全量落盘（本次已改）。

## 结论与下一步

**002 的可证伪问题回答：管线通，协议零错误。** 训练→生成→归档→账单全链路在 L4 上 4 分钟、\$0.09/次内完成；外推全量 35,948 条 ≈ 4.8 小时 ≈ **\$3.9**（在当月 credit 额度内，留有余量）。

下一步（待项目所有者决策）：003 全量训练方案——同配方 1 epoch 全量，训练后按 `rejev-holdout` 全集（1,892 条）评底座 vs 微调，含双解码口径（约束/无约束）与完整实验记录。
