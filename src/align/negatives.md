# 负例护栏 · MiniCPM5-2B 训练前对齐检查（chat 模板 · 结束标记 · loss mask · 答案解析）

> 2026-09-24。**正例给人看；能被机器判的负例，交给脚本查。**
> 判不了「像不像好东西」，判得了「有没有出现这几十条」——**把判好坏降级成判对错。**

## 规矩

1. **至少三倍**：负例不少于正例的三倍，并且生成的时候就保持这个比例。
2. **每一条都要落成三种形态之一**，让反例成为系统的一部分，不是评审时的一堆意见：
   **数据**（改前/改后对，在 `standard.md` 第二节）· **Skill**（checklist——机器查不了、
   人每次要核对的，进 CHECKLIST 段）· **脚本**（闸——能被机器找到的，进 PATTERNS 段）。
3. PATTERNS 段会被 `check.sh` 直接编译成检查；CHECKLIST 段会被它打印成人工核对清单。

---

## 一、事后：从废稿里抽

> 本模块尚无自己的废稿；下表从研究阶段被否掉的路线（内部研究取舍卡的「暂不选的路线」一节，未随本仓发布）抽取。

| 从哪一版废稿抽的 | 毛病是什么 | 抽成的规矩 |
|---|---|---|
| 暂不选路线①：`train_together.py` 只改模型名 | 平台、容器、存储全没换——改字符串不等于迁移 | 闸 `api\.together\.com`；Modal 训练器必须自建 |
| 暂不选路线②：直接跑全量 37,840 样本 | 许可、留出集、协议未闭环就上量 | 硬约束 6：训练数据不进对齐模块；CHECKLIST 第 6 条 |
| 暂不选路线③：以约 $17 设预算 | 平台口径不可比，且无历史账单佐证 | CHECKLIST 不写死 `$17` 字面（报告可能合法引用它作反例说明），靠人工核对 |

## 二、事前：逆向想象（红军测试）

> 「我想怎样**快速地毁掉**这个对齐检查？」——列完它们就成了负例。

给红军的提示词，照抄这一句（防它顺着你的框架把你夸一圈）：

> 自己读、自己跑，不要相信我的叙述。欢迎推翻我。

1. 让检查永远通过：`assert True`、空断言 → 闸
2. 把异常吞掉再报 PASS：裸 `except:` → 闸
3. 用错误 tokenizer 渲染还自洽：Qwen tokenizer 偷渡进来 → 闸；revision 中途漂移 → CHECKLIST
4. loss mask 声称正确但从没贴证据 → 硬约束 1 的四元组挡住（缺证据字段即不合格）
5. 报告把无关成绩当结论：官方名义、榜单词 → 闸
6. 检查脚本本身要 GPU / 远程才能跑 → 闸；破坏「零 GPU、离线可跑」
7. 硬编码本机路径，换台机器就废 → 闸
8. revision 钉在移动分支上（`main`/`latest`）＝没钉 → 闸
9. 密钥、token 进仓库 → 闸
10. 答案抽取退化成「先出现即取」的宽松抽取（`LLM01-kaizhi` 的老毛病）→ CHECKLIST；多字母/小写/空输出没被记 invalid → 硬约束 5

---

## 三、脚本（闸）：编译成检查

格式：一行一条，`模式<TAB>说明`（中间必须是 TAB）。模式按扩展正则（`grep -E`）解释。
字符类用 POSIX 写法（`[[:space:]]`），兼容 macOS 自带 grep。

```PATTERNS
官方复现|官方模型|Jev 官方|official reproduction	身份越界：本实验是 Jev-inspired，不是官方
Qwen[0-9A-Za-z_]*[Tt]okenizer|from_pretrained\([^)]*[Qq]wen	tokenizer 越权迁移：对齐对象是 MiniCPM5-2B，tev1 的 Qwen 预渲染文本不能喂它
labels[[:space:]]*=[[:space:]]*input_ids	loss mask 未做（全序列监督）——本模块里出现即错，掩码必须另行构造并断言
/Users/[A-Za-z0-9_.-]+	硬编码本机绝对路径（任何用户名），复现不了
sk-[A-Za-z0-9]{16,}	疑似密钥泄漏
pip[[:space:]]+install	依赖一律走 uv（文档里也用「依赖走 uv」的说法，别写这个短语）
api\.together\.com	本项目不用 Together API；其 $17/25 分钟报价无预算预测力
\.cuda\(|["']cuda["']	对齐检查须无 GPU 可跑；**适用范围：src/align、src/data**——src/train/ 是标准明示的 Modal GPU 豁免域，对本闸误报（2026-09-24 事故：为迁就此误报把 `.to("cuda")` 改成 `.to(model.device)` 造成 CPU no-op 挂死，白烧 ~$0.5。该目录只跑人工清单）
except[[:space:]]*:	裸 except 吞异常——检查器不许静默失败
assert[[:space:]]+True	永真断言＝没查
revision[[:space:]]*=[[:space:]]*["'](main|master|latest)["']	revision 钉在移动分支上＝没钉，要钉 commit hash
SOTA|state-of-the-art	无可比基线时不许用榜单词
```

## 四、Skill（checklist）：机器查不了的

```CHECKLIST
GPU 脚本的设备选择是否显式且非 no-op（from_pretrained 后 model.device 仍是 cpu；须真搬移或断言 device.type=="cuda"——2026-09-24 静默 CPU 推理事故）
模型与 tokenizer 是否钉死在同一 commit revision，检查全程未漂移
结束标记是否三处分别核对（训练模板 assistant 结束 / 推理停止配置 / 约束字母集），而不是机械拼接 tokenizer.eos_token
答案抽取是否只认完整单字母（锚定匹配）；有没有把 LLM01-kaizhi 的宽松「先出现即取」迁移过来
有约束与无约束两种解码口径是否都跑了，无效率是否分开记录
loss mask 首尾 token 的原文是否肉眼对照过（±0 token）
本轮是否未下载训练数据、未启动 Modal 任务（许可与零自付硬停止未核清前）
报告里每个数字能否回指到：代码版本 + 模型 revision + 样本清单 +（将来）Modal 运行账目
```

清单共七条，到上限；再加之前先问哪几条能降级成闸。

## 五、盘点

| | 条数 |
|---|---|
| 正例（`standard.md` 里的硬约束） | 6 |
| 负例·脚本（PATTERNS） | 12 |
| 负例·Skill（CHECKLIST） | 7 |
| 负例·数据（`standard.md` 第二节的亲改格子） | 0（2026-09-24 项目所有者决定不亲改，恒为 0） |
| **负例合计 ÷ 正例** | **19 ÷ 6 ≈ 3.17 ≥ 3 ✓** |

**数据形态恒空是有意的**：项目所有者于 2026-09-24 明确放弃亲改环节；
v1.a 结论表以既成参照生效，本表如实记账，不事后补造。
