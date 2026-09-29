# 复现路径

从零到复现本仓实验结论的完整步骤。

## ⚠️ 验证分级（先读这一节）

本文档的每一步都标注**验证状态**，请不要把「文档写了」当作「已被验证」：

| 标记 | 含义 |
|---|---|
| **[已验]** | 在本仓导出集上实际执行过 |
| **[未验·需 token]** | 需要 HuggingFace token 下载数 GB 数据；本仓发布前**未**在干净环境重跑 |
| **[未验·需额度]** | 需要 Modal 账号与 GPU 额度；本仓发布前**未**重跑（GPU 计费） |

**当前状态**：本文档的**结构完整性**可由你自己复跑核验，命令见文末「复跑校验」。
该命令核对的是：markdown **行内链接**有效、`src/`·`tools/`·`docs/`·`scripts/` **前缀**的路径引用（含以 `/` 结尾的**目录形式**）
确实存在、文本里不出现内部工作区路径、markdown **表格列数一致**、`.py`/`.sh` 语法可解析。
⚠️ **它不检查 `data/` 一类路径**——那些是你要自己构建的（见第 2 节），本仓不发布数据，
它们在发布集里**一个都不存在**；「路径引用」一栏全绿**不等于**「文档里提到的路径都在」。
**端到端复现链未在干净环境跑过**——它需要 HF token 与 Modal GPU 额度。这一点如实记录，
不写成「已验证」。

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
> `[dependency-groups] align` 中，而本仓 `[project]` 的 `dependencies` 是空的、`[dependency-groups]`
> 也只声明了 `align` 这一组（uv 的默认组 `dev` 本仓并不存在）。`uv sync` 不带参数
> 只会装出一个空环境，随后在 `import transformers` 处失败。

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

### 2.5 第二张封存集（exp004 / exp006 用）**[已验]**

报告 §2.4 与 §2.3 涉及的第二张留出集（1,802 题）由**另一个脚本**从 train 再切一次：

```bash
# 从 rejev-train 再切 5%（按 group 分层），产出 data/rejev2/
uv run python src/train/split_holdout_v2.py --frac 0.05 --seed 20260925
```

切分规则与 `split_holdout.py` 逐字相同，理由见 `docs/plan/004_weakness-probe.md`：原 holdout
已被 exp002/exp003 用过，按「留出集不得用于调参」的纪律不能再指导下一轮。切完之后原 holdout
**降级为开发集**。

> ⚠️ 报告 §2.3 的污染更正正是关于这张集合：它与 `rejev-train` 的 1,663 个去重 state 全部命中
> （题目口径 1,802/1,802）——**先切 train、再用 train 评测**就会得到这个结果。此处如实保留，
> 因为它正是报告里那条更正证据的来源。

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

**报告 §2.1 的四个无技能基线**由 `src/train/noskill_baseline.py` 复现（零 GPU、纯本地）：

```bash
# 默认读两张封存留出集：data/rejev/records/rejev-holdout.jsonl 与
# data/rejev2/records/rejev2-holdout.jsonl——**先完成 §2.4 与 §2.5**，否则会提示缺文件
uv run python src/train/noskill_baseline.py
```

输出**两张留出集各自的一块**（每块四个数：逐题随机 / 恒定最常见字母 / 恒定最常见语义 key /
该组最常见位置）。**报告 §2.1 表下那四个数（27.66% / 27.91% / 17.18% / 29.86%）取自
`rejev-holdout.jsonl`（n=1,892）那一块**；报告 §2.4 若引用 `rejev2-holdout.jsonl`（n=1,802）
那一块的同类数（如 27.30%），是另一块输出。注意后三个是在**同一张留出集上取众数**得出的
**事后描述上界**（拟合与评分用同一批标签），只有逐题随机是无拟合的基线——报告已如此标注。

### 6.1 分布外客卷分析（报告 §2.5 的读数）

报告第 2.5 节的分布外读数由 `src/train/exam_report.py` 产出。

⚠️ **该客卷不随本仓发布**：它是第三方**随其评测代码一并公开**的 2,087 题（来源见报告
「引用与致谢」的上游项目）。请自备一份副本，目录内需含 `inputs.json`、上游附带的
**`manifest.json`** 与逐题产物（`qwen.jsonl` / `jev.jsonl`）。

```bash
# ① 先只验副本真伪：把上游逐题产物喂进本分析器，必须逐位复现上游公布数字
uv run python src/train/exam_report.py --exam-dir <你的客卷目录> --selfcheck-only

# ② 自检通过后，再跑本仓臂的判读（--dir 指向含 cross-*-exam.jsonl 的目录）
uv run python src/train/exam_report.py --dir <臂产物目录> --exam-dir <你的客卷目录>
```

**自检不过则拒绝产出**——那说明副本不对（或分析器有 bug），**不是**「上游错了」。
这条自检在本项目实际拦下过一次实现错误。

**完整链路的第 0 步**（报告 §2.5 的读数由 `inputs.json` 起算）：

```bash
# 0a. 客卷 → 本项目评测格式（同样要 --exam-dir）
uv run python src/train/prep_exam.py --exam-dir <你的客卷目录>

# 0b. tev1 官方考卷 → 本项目评测格式（报告 §2.1/§2.3 的 1,300 题）
#     需要上游 tev1 检出（见第 2.1 节）：
uv run python src/train/prep_paper.py --tev1-dir <tev1 检出目录>/resources/tev1
```

`prep_exam.py` 的**校验 0** 会把 `inputs.json` 的 sha256 与同一目录下上游
`manifest.json` 记录的 `inputs_sha256` 比对，不符即中止——**自备副本必须与上游逐字节
相同**，改动过一题就会在此被拒。这也是第 ① 步自检之外的第二道副本校验：两者用不同的
依据（前者比对上游逐题产物，后者比对上游 manifest 的哈希）。

**跑评测臂（Modal，需额度）**——exp007 的五臂（`base` / `adapter` / `clean_r16` /
`adapter_r64` / `tev1`）由 `tools/trigger_exam.py` **秒级触发**（deploy + spawn，
不要用 `modal run` 直接跑全量，理由见 §4）：

```bash
modal volume put rejev data/exam/exam.jsonl /vol/data/exam.jsonl
modal deploy src/train/eval_cross.py
modal run tools/trigger_exam.py                  # 或 --models base,adapter 指定臂
```

逐题产物落在 Volume 的 `/vol/eval/cross-{model}-exam.jsonl`，取回后交给上一步判读。
**每臂产物自带模型名、互不覆盖**；续跑有 `.meta.json` 身份闸门（考卷 sha256 / 模型
revision / adapter sha 不符即拒绝续写）。

> ⚠️ **这两个脚本的「零泄漏校验」在本仓之外跑不全**（2026-09-28 第四轮评审 A/B 线 P1）：
> 它要把客卷与本项目**全部数据集**（`data/rejev/`、`data/rejev2/`、`data/paper/`）
> 逐题比对，而**本仓不发布数据**。输入不在本机时，脚本会把该口径记成**未检查**、
> 逐项打印并写进产物 manifest 的 `leak_check_missing`——**「未检查」不等于「通过」**，
> 引用读数时请一并说明。要跑全，先按第 2.4 节造 `data/rejev/`，再造 `data/rejev2/`
> （第 2.5 节）与 `data/paper/`（0b 步）。
>
> ⚠️ **顺序也要紧**：`data/paper/` 由 0b 生成，而 0a 的泄漏校验检查的是**当时磁盘上
> 已有的**数据集。按 0a → 0b 的顺序跑，客卷就**永远查不到** paper 那一份。要跑全，
> 全部数据造完之后**再重跑一次 0a**。

分析器还会校验各臂的完整性（题目覆盖是否齐全、**读到的行数与唯一 id 数是否一致**——
重复 id 会被 dict 静默覆盖、`model` 字段与臂名是否相符、**同一 tokenizer 族内**各臂的
prompt 指纹是否逐题一致），任何一项不过即**中止**而非告警：不完整的臂会让配对检验
**静默地**在子集上算出「无差异」——读起来像结论，实际只是数据不全。
⚠️ 指纹比对**只在同族臂之间**做：`tev1` 走的是另一个 tokenizer 族，渲染文本本就不同，
跨族比对只会产生假阳性（这一点本仓的 exp003 记录里已登记）。

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

它核对六件事：markdown 的**行内链接**是否有效（引用式链接与 HTML 锚点不在覆盖范围）、
文档里引用的 `src/`·`tools/`·`docs/`·`scripts/` 路径是否存在且未逃出发布根、
文本里**不出现内部工作区路径**、markdown **表格列数一致**（表头／分隔行／数据行三者格数
相同）、所有 `.py` 能否被解析、所有 `.sh` 能否通过 `bash -n`。
**零依赖、不联网、不需要 token。**

发布集另附**单元测试 6 个文件、96 个用例**（`src/data/test_render.py` 1 个 ＋ `src/train/test_*.py`
5 个），按目录分别复跑：

```bash
uv run --group align python -m unittest discover -s src/data    # 12 用例（需 transformers）
uv run --group align python -m unittest discover -s src/train   # 84 用例（零依赖）
```

它们覆盖数据链的渲染/切分与训练侧产物处理；**不保证**在任意环境通过（本仓未做端到端
冷启动验证，见文末），但这是发布时逐条实跑通过的入口。

⚠️ 四处边界请一并知道：
1. **路径引用只覆盖上述四个前缀**（文件与目录两种形式）——`data/…`、`resources/…` 不在检查范围内（它们本就不随
   发布集分发）。
2. **「内部工作区路径」查的是点开头的目录**（本仓的工作区元数据与工具配置目录都属这一类）。
   **下划线开头的过程目录不在**这条检查里：发布工具自己会把产物写进这类目录，一律封禁
   会把正常工具打成阻断项。
3. **表格检查只看格数**，不看内容是否被改坏；且若某张表被一行非表格文本**从中间截断**
   （常见于单元格跨行续写），它会报「表格被中断」并停止比对那张表的剩余行——报出来的是
   「这里需要人看」，不是「后面每一行都验过了」。另有一类**已知误报**（方向是拦住发布、
   不是放行）：**四空格缩进代码块**里画出的伪表格会被当成真表格报出来。刻意不做缩进启发式
   ——「跳过缩进行」的判据一旦写歪就变成**漏检**（真表格被整块跳过），那比多报一条噪音
   严重得多。围栏判定按 Markdown 规则限制在**三格缩进以内**：四格起是缩进代码块，不是围栏。
4. **裸 `.env` 不作为路径引用拦截**：`src/train/jev_api_eval.py` 里 `env = REPO / ".env"`
   是「从 .env 读 API key」的正常写法，且该文件明写「永不打印、永不入库」。被封禁的是
   凭据**内容**（另有两道专门规则），不是这个文件名。

输出里会单列「已登记豁免」项（那些是**看起来**像本仓路径、实则不是的引用，如指向上游
MiniCPM 官方仓库的 `docs/finetune/trl.md`，以及本仓计划文档里的待建产物）。

### 发布侧的对应核对

**读者看到的那份文件树，必须与「通过上面这些检查的那份导出集」逐文件一致**——否则
「校验通过」说的是另一个东西。发布完成后按**文件清单与哈希**核对一次：

```bash
# 在每棵树的根目录下各跑一次（相对路径一致，聚合值才有可比性）
find . -type f -not -path './.git/*' -not -path './.venv/*' -not -path '*/__pycache__/*' -not -name '*.pyc' | LC_ALL=C sort | xargs shasum -a 256 | shasum -a 256
```

两个聚合值必须**逐字符相等**（Linux 上 `shasum` 换 `sha256sum`）。

> 以下一段面向**发布方**的同步流程，普通读者可跳过。

这一步不是形式主义：
`tools/` 是发布时才生成的目录、在本仓源码树里**并不存在**，任何「按源码树的路径清单同步」
或「删掉目标里源端没有的文件」的做法都会**静默地**把它抹掉——而 `tools/verify-repro-path.py`
正是本文档上面叫读者去跑的那个入口。清单层面的差异，只有这道核对能发现。

## 附：本仓未做的事（避免误解）

- **无端到端冷启动验证**：上述步骤源自原始实验的真实执行路径，但**未**在一个全新的
  干净环境里从零跑通。若你发现某步走不通，那是本仓的缺口，欢迎指出。
- **无 GPU 复现**：发布前未重跑训练与评测（需要额度）。
- **不含数据与权重**：两者均需自行获取/训练，见第 2 节与 [NOTICE](NOTICE)。
