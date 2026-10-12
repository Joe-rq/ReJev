# 历史评测数据重建

本仓发布构建方法、来源锁定、文件级哈希与聚合计数；**不分发题干、答案、逐题结果、含样本的日志或数据副本**。本文重建的是第一阶段实际使用过的数据，不新建测试集。

两次历史留出已被开发消费，holdout2 又是 exp002 训练集的子集；重建副本不能恢复未触碰终评身份。污染更正与限制见[技术报告](../report/technical-report.zh-CN.md) §4。本页的哈希通过证明 records 身份与切分忠实，不证明模型效果，也不保证读者能在当前条件下复算报告的全部成绩。

## 1. 来源与身份

上游构建器：[togethercomputer/tev1 @ 1dde7782382c9f49d627153759b8d1deab426ce0](https://github.com/togethercomputer/tev1/tree/1dde7782382c9f49d627153759b8d1deab426ce0)，MIT 代码。原始数据与生成器产物各有自己的许可边界，不把代码许可等同于全部数据许可。

五个公开源的 revision 来自该提交的 [sources.lock.json](https://github.com/togethercomputer/tev1/blob/1dde7782382c9f49d627153759b8d1deab426ce0/sources.lock.json)，公开导出集根的 [sources.lock.json](../../sources.lock.json) 来自已有导出模板；五源的 repo/revision/config 与该上游 lock 相同，AG News/SST-5 另加许可解释字段。开发源码树中请用上游链接；根级文件由导出模板提供。以下许可说明是本仓 2026-09-24/09-28 回源记录的摘要，不是新增授权。

| source | 锁定来源与 revision | 历史许可核验摘要 |
|---|---|---|
| `mnli` | [nyu-mll/multi_nli @ `da70db2af9d09693783c3320c4249840212ee221`](https://huggingface.co/datasets/nyu-mll/multi_nli/tree/da70db2af9d09693783c3320c4249840212ee221) | 镜像列出混合许可；主体 OANC EULA，fiction 子项另有 CC/公有领域；不能写成全部 CC。 |
| `boolq` | [google/boolq @ `35b264d03638db9f4ce671b711558bf7ff0f80d5`](https://huggingface.co/datasets/google/boolq/tree/35b264d03638db9f4ce671b711558bf7ff0f80d5) | 镜像 CC-BY-SA-3.0；保留来源和许可说明。 |
| `banking77` | [legacy-datasets/banking77 @ `f54121560de48f2852f90be299010d1d6dc612ec`](https://huggingface.co/datasets/legacy-datasets/banking77/tree/f54121560de48f2852f90be299010d1d6dc612ec) | 镜像 CC-BY-4.0；保留来源和许可说明。 |
| `ag_news` | [fancyzhx/ag_news @ `eb185aade064a813bc0b7f42de02595523103ca4`](https://huggingface.co/datasets/fancyzhx/ag_news/tree/eb185aade064a813bc0b7f42de02595523103ca4) | 镜像 metadata=unknown，但卡片另有 research/non-commercial 表述。 |
| `sst5` | [SetFit/sst5 @ `e51bdcd8cd3a30da231967c1a249ba59361279a3`](https://huggingface.co/datasets/SetFit/sst5/tree/e51bdcd8cd3a30da231967c1a249ba59361279a3) | 镜像与已查溯源链缺许可声明；本仓不将其解释为公开数据授权。 |

原始许可出处：[OANC EULA](http://www.anc.org/OANC/license.txt)、[MultiNLI 0.9 论文附录](https://cims.nyu.edu/~sbowman/multinli/multinli_0.9.pdf)、[SST 原始页面](https://nlp.stanford.edu/sentiment/treebank.html)。其余三个源见上表固定版本卡片。MultiNLI 的 `mit` 元数据项没有完成子项来源对应；SST 作品授权链未核清。

另外四个 source（policy、policy_v2、routing_v2、research_taxonomy_v21）由上游 MIT 代码确定性生成，零外部 API/LLM 生成；上游未对合成产物另作数据许可声明。本仓是运行生成器的一方，不将忠实重建称为本仓原创数据。这些限制也是本仓选择只发方法与清单的原因。

各阶段的文件条数、source 分布与 group 数均见[固定哈希清单](history-hashes.json)，不另抄一份数表。上游 dev 原样保留。清单只含文件级哈希、聚合计数、上游 commit、切分参数及 holdout1 的状态集合聚合摘要 `statehashes_sha256`；该摘要是排序后去重 statehash 集合的整体 SHA256，不暴露单项。没有逐题标识或逐题哈希。

渲染输出目录须为空；复用旧目录会撤销旧成功标记并拒绝继续，防止旧 split 混入新的成功产物。失败目录内的残留不能用于后续步骤，重新执行请换新目录。

## 2. 隔离构建环境

建议全新检出，并使用一个原本不存在、位于仓库外的工作目录 `../rebuild-work`，防止误用旧中间产物或覆盖旧数据。上游与本仓各用自己的 `uv.lock`，**不要把本仓 MiniCPM tokenizer 写进上游 lock**：上游 records 的 token_count 和过滤已依赖 Qwen tokenizer。

以下从 ReJev 仓库根执行；本轮验证使用 CPython 3.14.3；以下环境安装固定该版本。项目代码要求 Python ≥3.12，但未逐个版本验证历史字节一致性。需要 uv、网络及足够本地存储，无 GPU、模型权重、训练账号或付费 API。访问数据源所需认证由来源平台决定，不打印或写入凭据。第三方应自行遵循各源使用条件。

```bash
mkdir ../rebuild-work
# 与上游 checkout 分别安装，避免 transformers/datasets 版本混用
uv sync --locked --group align --python 3.14.3

git clone https://github.com/togethercomputer/tev1 ../rebuild-work/tev1
git -C ../rebuild-work/tev1 checkout 1dde7782382c9f49d627153759b8d1deab426ce0
# 应打印上面完整 commit
git -C ../rebuild-work/tev1 rev-parse HEAD
uv sync --locked --project ../rebuild-work/tev1 --python 3.14.3
```

锁定源码 `sources.lock.json` 的文件 SHA256 为 `ee57fc1250aa4fd82349462c4b6370f344c87230d7de0a233fa671776a8f665a`。下载前后都应核对，不接受自动解析最新版本后的新 lock：

```bash
shasum -a 256 ../rebuild-work/tev1/sources.lock.json
```

Linux 可将 `shasum -a 256` 换成 `sha256sum`。下面指定新的 HF 缓存；fetch_sources 不下载模型权重。上游 lock 缺条目时会解析最新 revision，因此不得删除或修改已有 lock。

```bash
HF_HOME="$PWD/../rebuild-work/hf-upstream" HF_HUB_DISABLE_XET=1 \
  uv run --locked --project ../rebuild-work/tev1 python ../rebuild-work/tev1/fetch_sources.py
shasum -a 256 ../rebuild-work/tev1/sources.lock.json
HF_HOME="$PWD/../rebuild-work/hf-upstream" HF_HUB_DISABLE_XET=1 \
  uv run --locked --project ../rebuild-work/tev1 python ../rebuild-work/tev1/build_all.py
```

build_all 依次运行 build_dataset（v1）、build_v2（v2）、build_v21（v2.1）、build_new_v1（new-v1）；已有输出目录会使它停止。上游 new-v1 用 seed 42 shuffle 合并去重后的列表；早期各生成器有自己的固定 seed，均由该锁定提交决定。不能只运行最后一步并拿旧 v1/v2.1 充当新构建。

## 3. 两次历史切分

先锁定上游 records，再按原脚本两次切分。不修改历史记录的字段、行序、JSONL 空白或编码，否则文件哈希会变。

```bash
uv run --locked --group align python src/data/split_holdout.py \
  --records-dir ../rebuild-work/tev1/data/new-v1/records \
  --out-dir ../rebuild-work/rejev --holdout-groups-frac 0.05 --seed 20260924
uv run --locked --group align python src/train/split_holdout_v2.py \
  --src ../rebuild-work/rejev/records/rejev-train.jsonl \
  --out-dir ../rebuild-work/rejev2 --frac 0.05 --seed 20260925
```

两次都以 `(source, group_id)` 为最小单元，在每个 source 中按排序后的组列表用固定 RNG 抽取 `max(1, round(组数 × 0.05))` 组；同组变体不跨侧，保留输入行序。第二次只从第一次 train 再切，**不是与第一次平行重切**。两次脚本都检查 statehash/group 双口径不相交。statehash 使用 NFKC、casefold 与词序列归一化；dict state 先稳定序列化。第一次 JSONL 用紧凑 separators，第二次用默认空格；保持区别以重建历史 bytes。

```bash
uv run --locked --group align python src/data/verify_rebuild.py \
  --upstream-records ../rebuild-work/tev1/data/new-v1/records \
  --rejev-records ../rebuild-work/rejev/records \
  --rejev2-records ../rebuild-work/rejev2/records
```

只有 7 个 records 文件的固定 SHA256、行数、source/group 计数、holdout1 的 state 集摘要及两次切分的集合守恒/跨侧不相交都通过才输出 `REBUILD-CHECK: PASS`。缺输入、错文件、同条数换内容、重复 ID 或超出资源上限都失败。哈希不符时先核对本页的解释器版本、来源锁与完整构建顺序，不能改锚点来消除失败。校验只读，不输出题干或逐题标识；文件上限 256 MiB、单行 1 MiB、每文件 50,000 行；清单限 64 KiB，原始 JSON 的重复键直接拒绝。它不检验当前仓库自身来源真实性，使用前应选择可信发布版本，勿修改清单后再宣称通过历史锚点。

## 4. 本仓渲染与比较范围

上游构建锁 `Qwen/Qwen3.5-2B` @ `15852e8c16360a2fea060d615a32b45270f8a8fc`；本仓渲染锁 `openbmb/MiniCPM5-2B` @ `12a3808a956f869c767195e9266b59c4d21d92e2`。保留上游 records 语义，换成 MiniCPM 的模板与 token 计数；completion 仍为选项字母加 `<|im_end|>`，不使用 MiniCPM 的 eos_token `</s>`。

```bash
HF_HOME="$PWD/../rebuild-work/hf-minicpm" \
  uv run --locked --group align python src/data/render.py \
  --records ../rebuild-work/rejev/records --out ../rebuild-work/render-rejev
HF_HOME="$PWD/../rebuild-work/hf-minicpm" \
  uv run --locked --group align python src/data/render.py \
  --records ../rebuild-work/rejev2/records --out ../rebuild-work/render-rejev2
```

输入为文件时，输出仍按其 stem 建子目录，如 `rejev-train/instruction/rejev-train.jsonl`。每条检查答案单 token 与 ≤2048 tokens。

**哈希契约针对 records**：应与历史清单逐字节一致。Qwen 与 MiniCPM 的 prompt 文本/token_count 不要求相同；渲染 manifest 的运行时间/路径也不作历史字节比较。本文没有发布历史 MiniCPM 渲染文件哈希，不把 records 校验绿写成渲染与全部历史产物同 bytes。

## 5. 验证记录

本轮完整重建结果见[验证记录](verification.md)。其中必须分别说明源码取得方式、环境安装、HF 缓存是否为空、构建链、两次切分和渲染是否实际执行。训练、GPU 评测与模型成绩复算不是本数据重建验证的组成部分。

更多实验流程见[复现路径](../../REPRODUCING.md)。公开交付件与本机重建副本分开：数据副本始终留在读者本机，不随源码上传。
