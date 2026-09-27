# exp003 实现记录

## Deviations（偏离原 plan 的决策）

1. **考卷取题路径变更（等价更强替代）**
   issue 原文：「从 `results.jsonl` 提取 1,300 条逐题 id，**在我们重建的数据中按 id 精确取题**」。实测该反查交集为 **0**（官方考卷取 v1 `test`/`policy_transfer` split，我们重建的是 `train`/`dev`）。改为直接用考卷本体 `resources/tev1/data/v1/records/{test,policy_transfer}.jsonl`，并与官方 results.jsonl 逐条对齐验证（1000/1000 + 300/300）。**意图（拿到逐题一致的考卷）不变，方案更强**：直接得到题面、gold 与选项顺序。按项目约定「偏离不停等、记入本文件」处理。

2. **约束候选集：全 A–X → 按题实际选项**
   首轮实现沿用了 exp002 的「首步允许全 A–X」。评审指出官方协议是 **regex per option list**（主考含 200 道二选一）。评估后确认：exp002 约束器从未触发（约束/无约束逐位相同），故此改动**不改变任何历史结果**；但为与官方可比，必须按题。已改为 `LETTERS[:len(options)]`。

3. **主动停止首轮评测并重跑**
   首轮评测跑到 base 完成、adapter 进行中时，codex 评审返回 3 个 P0（含上述约束集问题）。**主动停止**（确认远程容器已终止）→ 修复 → 重新生成考卷 → 重跑。已耗成本约 $0.2 计入本轮账单。

4. **base/adapter 在 holdout 上沿用 exp002 已验收结果，不重跑**
   两者协议逐位相同（同 `apply_chat_template` 参数、同 eos/pad、同约束逻辑），且新脚本在 tev1paper 上重跑 base 得到与旧代码**逐位一致**的结果（0 条预测不同），佐证口径未漂移。汇总脚本对此有显式 fallback 并在 `sources_used` 中标注来源文件。

5. **sanity 产物独立命名**
   `--n>0` 时输出 `cross-{model}-{paper}-dev{n}.jsonl`，杜绝「sanity 子集被全量运行当已完成跳过」这类混入。

6. **未接 wandb**
   与 exp002 不同，本轮评测未接 wandb（减少无人值守下的失败点）。逐题结果与 summary 全部落 Volume，主指标不依赖外部服务。

7. **成本闸：整次调用墙钟上限 3h**
   `MAX_RUN_SEC = 10800` 自函数入口起算，由独立看门狗线程兜底（覆盖模型加载、单次 generate、收尾汇总三段）；另设 15 分钟零写入自杀。Modal spend limit $0 与 credit 额度为最终外部闸。

## 已知限制（记录不修，报告已标注）

1. **exp002 历史产物无 `n_options`**，其「无效率」为 A–X 口径而非按题。**已用只读核对证伪其影响**：那批 1,892×2 条预测**无一超出实际选项**，故数值不受影响。
2. **`_load_model` 阶段 STUCK 监控不生效**（仅总时限生效）——属成本边界，Modal 6h function timeout 兜底。
3. **`n_new=0`（纯续跑）时 `per_item_sec` 无意义**（分母为 0 时取 1）。
4. **`table3_skipped` 仅入 JSON**，未在 Markdown 表展示。
5. **`prompt_sha256` 不能跨模型比对**（两族 tokenizer 渲染不同），故三方一致性只核对 tokenizer 无关字段（id / gold / source / paper_split / n_options）。

## 评审

双谱系（codex/GPT 系 ＋ opencode/MiniMax 系）共 **5 轮**（原始报告未随本仓发布）。收敛：P0 由 3→0 并稳定为 0；P1 由 5→2 且降为审计/成本边界类。

**重要教训**：第 3 轮为满足「核对题面」而加入的跨模型 `prompt_tokens` 相等断言，本身制造了一个「报告永远无法出表」的缺陷，由第 4 轮评审拦下——**评审的价值不只在发现原缺陷，也在拦住修复过程引入的新缺陷**。
