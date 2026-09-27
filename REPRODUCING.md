# 复现路径

从零到复现本仓实验结论的完整步骤。

## ⚠️ 验证分级（先读这一节）

本文档的每一步都标注**验证状态**，请不要把「文档写了」当作「已被验证」：

| 标记 | 含义 |
|---|---|
| **[已验]** | 在本仓导出集上实际执行过 |
| **[未验·需 token]** | 需要 HuggingFace token 下载数 GB 数据；本仓发布前**未**在干净环境重跑 |
| **[未验·需额度]** | 需要 Modal 账号与 GPU 额度；本仓发布前**未**重跑（GPU 计费） |

**当前状态**：本文档的**结构完整性**（相对链接有效、文档引用的路径确实存在、Python/Shell 语法可解析）
可由你自己复跑核验，命令见文末「复跑校验」。但**端到端复现链未在干净环境跑过**——
它需要 HF token 与 Modal GPU 额度。这一点如实记录，不写成「已验证」。

---

## 0. 前置条件

- Python ≥ 3.12、[uv](https://docs.astral.sh/uv/)
- （步骤 2 起）HuggingFace 账号与 token
- （步骤 4 起）Modal 账号与 GPU 额度，且已配置零自付边界（见步骤 7）

## 1. 本地环境 **[已验]**

```bash
git clone https://github.com/Joe-rq/ReJev && cd ReJev
uv sync --group align
```

> **为什么必须带 `--group align`**：数据链与对齐检查所需的 `transformers` / `jinja2` 声明在
> `[dependency-groups] align` 中，它**不是** uv 的默认组（默认组是 `dev`）。不加这个参数
> 会得到一个空环境，随后在 `import transformers` 处失败。

验证安装：

```bash
uv run python -c "import transformers, jinja2; print(transformers.__version__)"
```

## 2. 数据链 **[未验·需 token]**

本仓**不分发数据**。数据链分四步，其中**前两步依赖 tev1 上游仓库**（MIT）——
本仓不复制它的构建管线，需自行获取。

### 2.1 获取上游构建器

```bash
git clone https://github.com/togethercomputer/tev1
cd tev1 && git checkout 1dde778     # 本项目锁定的 commit
```

### 2.2 下载数据源

按 `sources.lock.json` 中锁定的 repo 与 revision 下载五个数据集
（mnli / boolq / banking77 / ag_news / sst5）与两个 tokenizer。
用上游的 `fetch_sources.py`，或自行按 lock 文件下载——
**务必按 revision 拉取**，不要用默认分支。

> ⚠️ 许可提示：AG News 的上游卡片标注 research/non-commercial，SST-5 未标注许可。
> 用于商用或发布衍生权重前请自行确认，见 [NOTICE](NOTICE) 第 3 节。

### 2.3 构建 records（上游脚本）

用 tev1 的构建管线把原始数据变成 `state / question / options / answer` 语义层。
本仓锁定其 commit `1dde778`，产物为 `new-v1` 的 `train.jsonl` / `dev.jsonl`。

### 2.4 渲染与切分（本仓脚本）**[已验]**

```bash
# 分层切分：训练集 / 封存留出集 / 开发集，并写出零泄漏断言
uv run python src/data/split_holdout.py \
    --records-dir <上游 new-v1 records 目录> \
    --out-dir data/rejev

# 渲染为 MiniCPM5-2B 的 instruction / messages 格式
uv run python src/data/render.py \
    --records data/rejev/records/train.jsonl \
    --out data/rejev-render/train
```

`split_holdout.py` 会输出 `holdout-manifest.json`，内含内容哈希与「三口径零泄漏」断言
（`statehash` / `group_id` / `id`）。**留出集自此封存：不得用于训练或调参。**

## 3. 训练前对齐检查 **[已验]**

零 GPU（只需 tokenizer）：

```bash
uv run python src/align/v1a_check.py --offline
```

七项检查应全 PASS：来源锁定、特殊 token 映射、模板渲染、字母 tokenization、
loss mask、结束标记、解析对抗。若 `--offline` 用本地缓存；去掉它会在线比对 revision。

> 这一步的意义：把「模型/模板/标签/解析」四个层面的错误**在花 GPU 钱之前**挡掉。
> 本仓的教训是——一个 `.to(model.device)` 的 no-op 曾让评测在 CPU 上空转 20 分钟。

## 4. 训练（Modal） **[未验·需额度]**

⚠️ **必须用 `deploy` + `spawn`，不要用 `modal run` 直接跑训练脚本**。
`modal run` 会把本地进程绑在远程任务上——本地被终止时，Modal 会把 cancellation
送到远程**杀掉正在跑的训练**（`--detach` 亦然）。机制详见 `tools/trigger_clean.py` 的文档串。

```bash
# ① 数据准备（本地，零 GPU）：剔源 + 切分校验 + 渲染
uv run python src/train/prep_clean.py          # → data/clean/

# ② 上传数据到 Modal Volume
modal volume create rejev          # 已存在同名 Volume 会失败——那就跳过这条
modal volume put rejev data/clean/train-clean.jsonl /data/
modal volume put rejev data/clean/eval-set.jsonl /data/

# ③ 部署持久 app + 秒级触发
modal deploy src/train/clean_train.py
modal run tools/trigger_clean.py

# ④ 监控
modal app list                        # State 应为 deployed，Tasks 应 ≥1
modal app logs <app-id>
modal volume ls rejev /checkpoints-clean-r16   # 见到 checkpoint 才算「有保护」
```

**预期**：4 臂本地耗时约 4 小时（L4），训练集 30,987 条，1 epoch，LoRA r16/α32。
`save_steps=1000`，约首小时内出现第一个 checkpoint。

> 本项目实测的坑：**r64 配置是失败的**（holdout 准确率反而低于底座，train_loss 0.880 vs
> r16 的 0.457）——推测根因是 LoRA 参数量 4 倍而学习率未随 rank 缩放。
> 故 `trigger_clean.py` 默认锁 r16/α32。详见 `docs/experiments/exp004-capacity-probe/`。

## 5. 评测（Modal） **[未验·需额度]**

```bash
modal deploy src/train/eval_holdout.py
modal run tools/trigger_eval.py            # 或 --arms 006-clean-r16 只跑一臂
```

三臂同集：`base`（未微调底座）· `exp002-r16`（含源配方参考）· `clean-r16`（剔源产物）。
⚠️ 后两臂的 adapter 来自**更早的实验轮次**（`rejev:/artifacts/full-v1` 是 exp002 的产物）。
只跑本流程产出的 `clean-r16` 也是合法的用法——用 `--arms 006-clean-r16` 即可。

**每臂必须给不同 `tag`**，且**必须核对封存集指纹**——行数断言挡不住「同样行数的另一份文件」，
若各臂读到的不是同一份封存集，数字不可比且事后无从察觉。指纹是 `tools/trigger_eval.py`
顶部的 `EXPECT_SHA` 常量，**换了数据就要同步更新**：

```bash
sha256sum data/clean/eval-set.jsonl        # 先算出你自己的指纹
EXPECT_SHA=<你的指纹> modal run tools/trigger_eval.py
```

评测在约束解码与无约束两个口径下各跑一遍。

## 6. 结果分析 **[已验]**

```bash
# 逐题结果 → 汇总（分 source 分层、四象限、McNemar）
uv run python src/train/compare.py --help
```

统计口径注意：**同一 group 的多个变体题目高度相关**，逐题独立性假设不成立。
本仓的主口径是**按 `group_id` 整组重抽的聚类 bootstrap**，逐题 Wilson 区间仅作参考。

## 7. 成本与账单

每轮实验的实际成本记录在各 `docs/experiments/*/README.md` 的「账单」节。
训练前应确认：Modal spend limit 已设（本项目设为 **$0** 以保证零自付）、
credit 余量足以覆盖预计成本的 1.5 倍、且无遗留运行中的 app。

```bash
modal billing summary --for "this month"
modal app list        # 确认无残留任务（零残留是每轮收尾的必查项）
```

---

## 复跑校验

上面的「结构完整性」不是自述——你可以自己跑：

```bash
uv run python tools/verify-repro-path.py .
```

它核对：markdown 的**行内链接**是否有效（引用式链接与 HTML 锚点不在覆盖范围）、
文档里引用的 `src/`·`tools/`·`docs/` 路径是否存在且未逃出发布根、
所有 `.py`/`.sh` 是否语法可解析。**零依赖、不联网、不需要 token。**
输出里会单列「已登记豁免」项（那些是**看起来**像本仓路径、实则不是的引用，如指向上游
MiniCPM 官方仓库的 `docs/finetune/trl.md`，以及本仓计划文档里的待建产物）。

## 附：本仓未做的事（避免误解）

- **无端到端冷启动验证**：上述步骤源自原始实验的真实执行路径，但**未**在一个全新的
  干净环境里从零跑通。若你发现某步走不通，那是本仓的缺口，欢迎指出。
- **无 GPU 复现**：发布前未重跑训练与评测（需要额度）。
- **不含数据与权重**：两者均需自行获取/训练，见第 2 节与 [NOTICE](NOTICE)。
