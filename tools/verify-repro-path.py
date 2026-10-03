#!/usr/bin/env python3
"""复现路径静态校验：验证发布物里「文档说的」与「仓库里有的」对得上。

**为什么需要它**：`REPRODUCING.md` 可以声称「引用完整、无悬空路径」，但除非有人
真的逐条查过，那只是一句自述。评审（Issue #9 双谱系）正是抓住这一点：
当时该声明**没有任何脚本支撑**。这个脚本就是那个支撑——把「已校验」变成可复跑的命令。

它做七件事（前六件零依赖、不需要 GPU / token / 网络；第七件需要 pandoc，缺席时报跳过）：
  1. **行内链接**：markdown 的 `[文本](目标)` 形式所指向的仓内文件确实存在，
     且目标**未逃出发布根**（逃逸或软链目标即便在本机存在，对第三方也是死链）。
     ⚠️ 只解析行内链接——引用式 `[文本][ref]` 与 HTML `<a href>` **不在覆盖范围内**。
  2. **文档引用**：文档里以 `src/…` `tools/…` `docs/…` 形式引用的路径确实存在
     （**含以 `/` 结尾的目录形式**，第九轮 B 线 P1-2 补）。⚠️ 覆盖不到的形式：不带
     `src|tools|docs|scripts` 前缀的裸相对引用、frontmatter 里的无前缀条目、引用式
     `[文本][ref]` 与 HTML `<a href>`
  3. **隐藏工作区路径**：文本里不得出现点目录（`.claude`、`.42cog`）形式的引用
     （第 2 项的正则从构造上就看不见它们，见 `HIDDEN_WORKSPACE` 的注释）
  4. **表格列数一致性**：同一张表内每行的格数须与表头相同
     （内容审计只管文本、不管 markdown 形状，见 `check_tables` 的注释）
  5. **Python 语法**：所有 .py 能被 `ast.parse` 解析
  6. **Shell 语法**：所有 .sh 能被 `bash -n` 解析
  7. **金额 `$` 渲染存活**：`.md` 里成对的货币 `$` 会被 pandoc 当成行内数学吃掉
     （金额失去货币单位、强调被撕成字面 `*`）——本项按多重集比对把它们报出来
     （第十八轮 B 线 P2-1；无 pandoc 时报「跳过」，不计入判定）

用法：
    uv run python tools/verify-repro-path.py <目录>       # 默认当前仓库根
退出码：0 = 全过；1 = 有问题。
"""

from __future__ import annotations

import argparse
import ast
import collections
import html
import re
import subprocess
import sys
from pathlib import Path

# 文档里形如 src/... tools/... docs/... 的路径引用（须以已知顶层目录开头）
PATH_REF = re.compile(
    r"\b((?:src|tools|docs|scripts)/[A-Za-z0-9_./-]+\.(?:py|md|sh|json|cff))\b"
)
MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
# 目录形式（以 `/` 结尾）的仓内路径引用：第九轮 B 线 P1-2——plan 文档把真 Jev 对照实验
# 的目录名写错（该目录从未存在），而那种写法对 `PATH_REF` 从构造上不可见（它要求以
# 扩展名结尾）。
# ⚠️ 末尾的 `(?![A-Za-z0-9_.-])`：不吃文件引用的前缀——带扩展名的完整路径已由
# `PATH_REF` 整段认领，本正则若再匹配出它的目录前缀，同一处引用会报两遍。
_PATH_REF_DIR = re.compile(
    r"\b((?:src|tools|docs|scripts)/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*/)(?![A-Za-z0-9_.-])")

# ── 隐藏工作区前缀 ─────────────────────────────────────────────────────────
# 第五轮评审（A 线 P2 / B 线 P3 **各自独立**报出）：上面的 `PATH_REF` 只认
# `src|tools|docs|scripts` 开头的引用，于是 `docs/plan/005` 里那句指向 issue-2 工作树
# 实现文件的引用 **既不在扫描范围内、也不可能被报出**——「校验通过」并不等于
# 「没有内部路径」。这条按**前缀**判：发布文本里出现点开头的目录即为阻断项，
# 不论该文件在本机是否存在。
#
# 为什么只圈点目录、不圈过程目录与只读材料目录（`_tmp`、`vault` 等）：后者的裸路径名
# 在**复现指引里是必要的**（「资源目录不随本仓发布，请用 `--tev1-dir` 指向自备副本」
# 这类句子必须有它），一律阻断会把正常指引也打死。而内容层面的那批悬空引用由
# 内容审计器（不随本发布集分发）的 `dangling_ref` 规则管，职责不重叠。
#
# `.github/` 是例外：它既是点目录、又是公开发布物的正常组成部分。
#
# ⚠️ 2026-09-29 第六轮评审 A 线 P2-4：前边界原为 `(?<![A-Za-z0-9_./\\-])`，**含 `/`**
# ——于是「点目录挂在可见前缀之下」的写法整体逃过检查（那个点目录前面是 `/`，被前边界
# 否定）。这恰恰是本检查最该拦的一类。现改为只排除字母/数字/下划线/点/反斜杠，
# **不排除 `/`**：URL 里的 `.com/…` 仍因前面是字母而不误报，而嵌在路径中段的点目录
# 现在能命中。本文件**不写具体目录名**——名字本身就是规则要防的结构指纹。
#
# ⚠️ 刻意**不**纳入裸 `.env`（B 线 P2-7 建议过）：`src/train/jev_api_eval.py` 里
# `env = REPO / ".env"` 是「从 .env 读 API key」的**正常写法**，且该文件明写「永不打印、
# 永不入库」。凭据**内容**由 `api_key` / `account_identity` 两条规则兜，而封禁文件名会
# 把一条合法且被推荐的用法打成阻断项——这正是「误报逼出豁免、豁免一多真问题就淹」。
HIDDEN_WORKSPACE = re.compile(
    r"(?<![A-Za-z0-9_.\\-])(\.(?!github(?:/|$))[A-Za-z0-9_-]+/[A-Za-z0-9_./-]+)")


# ── 已登记豁免 ─────────────────────────────────────────────────────────────
# (文件相对路径, 引用/链接文本, 理由)。
#
# 豁免只用于「**看起来**像本仓路径、实则不是」与「plan 文档里合法的待建产物」两类。
# 真悬空（指向未发布的已有文件）必须改掉，不许登记豁免——那是拿豁免掩盖问题。

ALLOW: list[tuple[str, str, str]] = [
    (
        "src/align/negatives.md",
        "main|master|latest",
        "负例闸的正则片段，被 markdown 链接正则误匹配",
    ),
    (
        "docs/plan/002_first-smoke-training.md",
        "docs/finetune/trl.md",
        "上游 MiniCPM 官方仓库的文档路径，不是本仓路径",
    ),
    (
        "REPRODUCING.md",
        "docs/finetune/trl.md",
        "同上——本文件在说明「哪些豁免是上游路径」时又提到了它一次",
    ),
    (
        "docs/plan/005_ood-exam-eval.md",
        "docs/experiments/exp005-ood-exam/README.md",
        "plan 文档的**预期**归档路径——实际归档为 exp007（编号被 exp005-jev-comparison 占用，"
        "见 docs/experiments/exp007-ood-exam/implementation-notes.md 偏离 1）。plan 是训前"
        "锁定记录、不改写，故此处保留旧路径名",
    ),
    # ── 第十轮：检查面扩到 py/sh 文件与目录形式后的既有合法引用 ──
    (
        "src/train/precheck.py",
        "docs/finetune/trl.md",
        "上游 MiniCPM 官方仓库的文档路径，不是本仓路径（同 plan/002 那条的类别）",
    ),
    # 2026-10-03（#65 批次 ④ 顺手登记，#57 引入时漏登记）：报告 §1.3「训练输入 → SFT
    # 输入」一步写明的模板出处——同「上游路径」类别（非本仓路径）
    (
        "docs/report/technical-report.md",
        "docs/finetune/trl.md",
        "上游 MiniCPM 官方仓库的文档路径，不是本仓路径（§1.3 训练模板的出处）",
    ),
    (
        "docs/report/technical-report.zh-CN.md",
        "docs/finetune/trl.md",
        "同上（中文版）",
    ),
    # 范围注里**指名排除项**：路径不在发布集属有意为之（README/报告两版的 scope 注）
    ("README.md", "src/publish/", "范围注：说明该目录被整目录排除"),
    ("README.zh-CN.md", "src/publish/", "范围注：说明该目录被整目录排除"),
    ("docs/report/technical-report.md", "src/publish/", "范围注：说明该目录被整目录排除"),
    ("docs/report/technical-report.zh-CN.md", "src/publish/", "范围注：说明该目录被整目录排除"),
    ("docs/experiments/exp003-three-way/implementation-notes.md", "src/publish/",
     "计数订正说明里的提及（解释旧数从何而来），非可点击引用"),
    # 检查器自述文档：docstring 里的形态示例与 ALLOW 表自身的常量（对第三方是文档，
    # 不是可点击的路径）
    ("tools/verify-repro-path.py", "目标", "docstring 里的行内链接形态示例"),
    ("tools/verify-repro-path.py", "src/publish/", "ALLOW 表自身的豁免常量（范围注条目）"),
    ("tools/verify-repro-path.py", "docs/finetune/trl.md", "ALLOW 表自身的豁免常量"),
    ("tools/verify-repro-path.py", "docs/experiments/exp005-ood-exam/README.md", "同上"),
    ("tools/verify-repro-path.py", "docs/experiments/exp005-ood-exam/", "同上"),
    (
        "docs/experiments/exp007-ood-exam/implementation-notes.md",
        "docs/experiments/exp005-ood-exam/",
        "同一「预期 vs 实际」说明的**目录形式**（偏离 1 本体，引用的是 plan 当年写的"
        "预期目录，实际归档为本文所在的 exp007）",
    ),
]


# 遍历时跳过的目录：这些是**环境**而非发布物的一部分。不跳过的话，跑过一次
# `uv sync` 之后再校验，就会看到 site-packages 里第三方包自带的死链——那是噪音，
# 会把真正的失败淹掉。
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".cache",
    # 过程材料目录（见项目开工手册的目录约定）：不作为正式源文件。它们常常整份拷着
    # 发布集的旧快照，不跳过会让同一个问题被重复报上千次，把真失败淹掉。
    "_build", "_tmp", "_archive",
}


def _iter(root: Path, pattern: str):
    """递归找文件，但跳过 SKIP_DIRS 下的内容。"""
    for p in sorted(root.rglob(pattern)):
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        yield p


def _iter_patterns(root: Path, patterns: tuple[str, ...]):
    """同 `_iter`，但接受多种文件名模式（去重保序）。"""
    seen: set[Path] = set()
    for pat in patterns:
        for p in _iter(root, pat):
            if p not in seen:
                seen.add(p)
                yield p


def _allowed(rel_md: str, token: str) -> bool:
    return any(f == rel_md and t == token for f, t, _ in ALLOW)


def check_links(root: Path) -> tuple[list[str], list[str]]:
    """markdown 相对链接的有效性。返回 (问题, 已豁免)。

    注意：会把「写成 markdown 链接样式的正则文本」认作链接（如负例闸里的
    `(main|master|latest)`）——那类按 ALLOW 登记，而不是放宽正则。
    """
    problems: list[str] = []
    exempted: list[str] = []
    # 第十轮 B 线 P1-1（建议③）：路径引用不只在文档里——发布测试脚本在类体里读未随
    # 导出分发的模型卡，`check_doc_refs` 只扫 .md 时对这类坏件完全无感。与
    # `check_hidden_paths` 同一覆盖面（那处早就扫 py 与 sh 文件，本处是遗漏而非设计）。
    for md in _iter_patterns(root, ("*.md", "*.py", "*.sh")):
        rel_md = md.relative_to(root).as_posix()
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for _, target in MD_LINK.findall(line):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                path_part = target.split("#")[0]
                if not path_part:
                    continue
                resolved = (md.parent / path_part).resolve()
                # 逃出发布集的链接（`../../..` 到本机别处的文件）即便在这台机器上
                # 碰巧存在，第三方拿到的仍是死链——必须按失效处理。
                if not resolved.is_relative_to(root):
                    entry = f"{rel_md}:{lineno} → {target}（逃出发布集）"
                    if _allowed(rel_md, target):
                        exempted.append(entry)
                    else:
                        problems.append(f"[链接逃出发布集] {entry}")
                    continue
                if resolved.exists():
                    continue
                entry = f"{rel_md}:{lineno} → {target}"
                if _allowed(rel_md, target):
                    exempted.append(entry)
                else:
                    problems.append(f"[链接失效] {entry}")
    return problems, exempted


def check_doc_refs(root: Path) -> tuple[list[str], list[str]]:
    """文档里引用的仓内路径是否存在。"""
    problems: list[str] = []
    exempted: list[str] = []
    for md in _iter_patterns(root, ("*.md", "*.py", "*.sh")):
        rel_md = md.relative_to(root).as_posix()
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            refs = sorted(set(PATH_REF.findall(line)) | set(_PATH_REF_DIR.findall(line)))
            for ref in refs:
                if _allowed(rel_md, ref):
                    exempted.append(f"{rel_md}:{lineno} → {ref}")
                    continue
                target = root / ref
                if target.is_symlink() or not target.resolve().is_relative_to(root):
                    # 软链与逃逸路径都可能在**这台**机器上存在，但对第三方无效
                    problems.append(f"[引用逃出发布集或为软链] {rel_md}:{lineno} → {ref}")
                    continue
                if not target.exists():
                    problems.append(f"[引用不存在] {rel_md}:{lineno} → {ref}")
    return problems, exempted


def check_hidden_paths(root: Path) -> tuple[list[str], list[str]]:
    """发布文本里的**隐藏工作区前缀**引用（如 `.claude`、`.42cog` 下的文件）。

    与 `check_doc_refs` 的关键差别：它按「文件是否存在」判，本检查按「前缀是否该出现」
    判——`PATH_REF` 的正则从一开始就不认点开头的目录，所以这类引用**永远**不会进入
    前者的视野。**该检查没有豁免机制**：真需要引用就说明发布文本写错了。
    """
    problems: list[str] = []
    for pattern in ("*.md", "*.py", "*.sh"):
        for f in _iter(root, pattern):
            rel = f.relative_to(root).as_posix()
            for lineno, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                for ref in sorted(set(HIDDEN_WORKSPACE.findall(line))):
                    problems.append(f"[隐藏工作区路径] {rel}:{lineno} → {ref}")
    return problems, []


def _cells(line: str) -> int:
    """markdown 表格行的格数。转义竖线（`\\|`）不计为分隔符。

    必须处理转义：评测协议里的特殊 token 写成 `<\\|im_end\\|>` 才不会被表格解析器
    当成列分隔——**这正是 B 线 P2 那条「导出后有一行列数不符」的成因之一**。
    """
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return len(re.split(r"(?<!\\)\|", s))


# 围栏（GFM 规则）。**开启行** = 最多三格缩进 ＋ 三个及以上同一字符（``` 或 ~~~）；
# **闭合行** = 同一种字符、长度**不短于**开启行、且其后只允许空白。
# ⚠️ 反引号围栏的 info string **不得含反引号**——`` ```lang` `` 因此**不是**开启围栏
# （它是一条普通段落），这四个字符的差别关系到一处 fail-open，见 `check_tables`。
#
# ⚠️ 两条边界是**实测**结论，不是形式主义：
#   ① 四格及以上缩进是**缩进代码块**、不是围栏（第七轮 A 线 P2-2）；
#   ② 旧版只用「前缀匹配 + 翻转布尔」的 toggle，**既不校验开启行合法性、也不记录标记
#      字符与长度**：`` ```lang` `` 这种非法起始行会把状态翻成「在围栏里」，该文件其余
#      真表格**全部跳过**——第八轮评审 A 线 P1-6 用内存样本实测，破表返回「无问题」。
#      反向的 `~~~` 开、``` 关则多报一条（fail-closed，A 线 P2-1）。两者一并由状态机修掉。
# 第九轮 A 线 P2-1：列表项内的围栏是合法 GFM（`- \`\`\`text` 后跟缩进内容）。开启行
# 允许一个可选的列表标记前缀（`- ` / `* ` / `+ ` / `1. ` / `1) `）＋ 最多三格缩进。
# 已知限制（不实现、如实登记）：嵌套列表（缩进 >3）、引用块（`> `）内的围栏不认——
# 那需要完整容器解析；当前导出集没有这类构造，遇到时本检查会**多报**（fail-closed）。
_FENCE_OPEN = re.compile(
    r"^(?:[-*+] |\d{1,9}[.)] )? {0,3}(`{3,}|~{3,})(.*)$")
_FENCE_CLOSE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*$")

# 第十轮 A-P2-2 / B-P2-3：分隔行**必须含至少一个 `|`**——GFM 的无边框表格也必须
# 有列分隔符；裸 `---` 是 YAML frontmatter 或水平分隔线，旧正则会把「frontmatter 首尾
# 夹住的一行含 `|` 文本」误判成破表。
# 第十一轮 A 线 P2-2 补**逐格校验**：每个分隔单元格须形如 `:?-+:?`（冒号可选、
# 至少一连字符）。纯 `:` 之类的无效格此前被当有效分隔行——判定结果虽然相同
# （GFM 里无效分隔行使整表不成立，两行都成普通文本、无错位可报），但「把无效
# 分隔行当有效」是语义歧义，钉死。
_SEP_ROW = re.compile(r"^(?=[\s:|-]*\|)(?=[\s:|-]*-)[\s:|-]*$")
_SEP_CELL = re.compile(r"^\s*:?-+:?\s*$")


def _is_sep_row(line: str) -> bool:
    """分隔行判定 = 整行形态 ＋ 每个分隔单元格都合法（第十二轮 A 线 P2 收紧）。

    ⚠️ 只剥**一对**可选的边框竖线（`line.strip().strip("|")` 会剥掉任意多根，把
    `|| --- |` 里的空格当边框放行）；且**空分隔格不合法**（GFM 要求每格至少一个
    连字符——`| --- || --- |` 的空格使整行无效，旧判定的 `or not c.strip()` 把它
    当成了有效分隔行）。
    """
    if not _SEP_ROW.match(line):
        return False
    core = line.strip()
    if core.startswith("|"):
        core = core[1:]
    if core.endswith("|") and not core.endswith("\\|"):
        core = core[:-1]
    return all(_SEP_CELL.match(c) for c in re.split(r"(?<!\\)\|", core))


def check_tables(root: Path) -> list[str]:
    """markdown 表格的**列数一致性**：同一表内每行（含**分隔行**）列数须与表头相同。

    **为什么需要它**（第四/五轮评审）：内容审计只管**文本**，不管 markdown 的**形状**。
    上一轮整块脱敏时把表头从 5 列改成 4 列，某数据行却多留了一个 `— |`——表格在
    GitHub 上渲染错位，而所有内容闸全绿。本轮又发现三处表格内的 `<|im_end|>` 未转义，
    同样会把一格劈成三格。这两类都只能靠**形状**检查抓。

    围栏代码块（``` 与 ~~~）内的内容跳过（那里的 `|` 不是表格）。状态机按 GFM 校验
    开启行（含「反引号围栏的 info string 不得含反引号」）并记录标记字符与长度——**toggle
    式的实现有一处 fail-open、一处误报**，见 `_FENCE_OPEN` 上方的注释。

    ⚠️ 2026-09-29 第六轮评审 B 线 P2-2 ／ A 线 P2-4：**本函数此前有两处 fail-open**，
    都已修：
      ① 列数 `ncol` 只从**表头**算，分隔行自身从不参与比对——而 GFM 的渲染恰恰以分隔行
         为准，所以「表头 4 列、分隔行 5 列」这种真破版**一声不响**地过了。
      ② 内层循环遇到首个非 `|` 开头的行就 `i = j`，把游标交给那一行，**该表剩余各行
         再无表头可比**——单元格跨行续写会让检查静默失效（B 线用内存态合成表格实测）。
    另有一类**误报**（fail-closed，已知且接受）：4 空格缩进代码块里的伪表格会被当成真
    表格报出来。**刻意不做缩进启发式**——跳过缩进行的判据一旦写歪会变成 fail-open
    （真表格被整体跳过），那比多报一条噪音严重得多。此权衡由测试钉住。
    """
    problems: list[str] = []
    for md in _iter(root, "*.md"):
        rel = md.relative_to(root).as_posix()
        lines = md.read_text(encoding="utf-8").splitlines()
        fence: tuple[str, int] | None = None      # (围栏字符, 开启长度)
        i = 0
        while i < len(lines):
            line = lines[i]
            if fence is not None:
                closed = _FENCE_CLOSE.match(line)
                if (closed and closed.group(1)[0] == fence[0]
                        and len(closed.group(1)) >= fence[1]):
                    fence = None
                i += 1
                continue
            opened = _FENCE_OPEN.match(line)
            # 反引号围栏的 info string 不得含反引号——`` ```lang` `` **不是**围栏。
            # 少了这一句，状态就会被翻成「在围栏里」且再也不复位（fail-open）。
            if opened and not (opened.group(1)[0] == "`" and "`" in opened.group(2)):
                fence = (opened.group(1)[0], len(opened.group(1)))
                i += 1
                continue
            # 第九轮 B 线 P2-2：GFM 不要求行首竖线——无边框表格（`a | b` / `--- | ---`）
            # 同样是表格。表头判定靠「本行含未转义 `|` 且下一行是分隔行」这一强约束，
            # 散文里的 `a | b` 因缺少分隔行跟随而不会误判。
            if "|" not in line:
                i += 1
                continue
            # 表头 = 本行 + 下一行是分隔行
            if i + 1 >= len(lines) or not _is_sep_row(lines[i + 1]):
                i += 1
                continue
            ncol = _cells(line)
            # ① 分隔行必须与表头同列数——GFM 以分隔行定列数，不同即是真破版。
            nsep = _cells(lines[i + 1])
            if nsep != ncol:
                problems.append(
                    f"[表格分隔行列数不符] {rel}:{i + 2} 分隔行 {nsep} 格，表头 {ncol} 格")
            # 数据行同样不要求行首竖线（同一轮 B-P2-2）：掉行首 `|` 的行此前既不被比对、
            # 也不触发「表格被中断」——两头的检查都看不见它。
            j = i + 2
            while j < len(lines) and "|" in lines[j] and lines[j].strip():
                n = _cells(lines[j])
                if n != ncol:
                    problems.append(
                        f"[表格列数不符] {rel}:{j + 1} 该行 {n} 格，表头 {ncol} 格")
                j += 1
            # ② 表格被一行**非空、非表格**的行截断，而紧随其后又是表格行 → 几乎必然是
            #    某个单元格跨行续写（或漏了一个 `|`）。此时上表剩余各行未被比对，必须报。
            #    判据特意收得很窄：空白行结尾是 GFM 表格的**正常**终止方式，不报；
            #    围栏行（``` / ~~~）也不算中断——那后面是代码块里的伪表格，不是本表续行。
            tail = lines[j].strip() if j < len(lines) else ""
            if (tail and "|" not in tail and not tail.startswith(("```", "~~~"))
                    and j + 1 < len(lines) and "|" in lines[j + 1]):
                problems.append(
                    f"[表格被中断] {rel}:{j + 1} 该表在此被非表格行截断，"
                    f"其后各表格行未与表头比对（疑似单元格跨行续写）")
            i = j
    return problems


# ── 金额 `$` 的渲染存活（第十八轮 B 线 P2-1）──────────────────────────────────
# 同一条 pandoc 语义的另一面：一格里**两个及以上的货币 `$`** 会被当成行内数学定界符，
# 两个 `$` 连同中间的文字一起消失、强调被撕成字面 `*`（实测
# `第一次 $0.085…＋第二次 $0.090 ＝ **$0.175**…` → `第一次 $0.085…＋第二次 0.090 ＝
# * *0.175**…`）——发布物的账单金额没了货币单位，而内容审计、表格列数、链接三项检查
# 全绿（审计只查文本敏感面、不管渲染形状；表格检查只看格数）。
# 判据取**多重集比对**：排除代码上下文（围栏与行内代码 span——那里的 `$` 是字面）后的
# 原文里的每个 `$金额`，都必须出现在 pandoc 的可见文本里，个数一致。
# ⚠️ 表格按**格**成对解析（同一格里才有配对），故表格行只要一格里不足两个 `$` 就不报。
_RE_DOLLAR_AMOUNT = re.compile(r"(?<!\\)\$\s?[0-9][0-9.,]*")
_RE_INLINE_CODE = re.compile(r"(`+)(?:(?!\1).)*\1")


def _strip_code_contexts(text: str) -> str:
    """去掉围栏代码块与行内代码 span——那些上下文里的 `$` 是字面，不参与配对。"""
    out: list[str] = []
    lines = text.splitlines()
    fence: tuple[str, int] | None = None
    for line in lines:
        if fence is not None:
            closed = _FENCE_CLOSE.match(line)
            if closed and closed.group(1)[0] == fence[0] and len(closed.group(1)) >= fence[1]:
                fence = None
            continue
        opened = _FENCE_OPEN.match(line)
        if opened and not (opened.group(1)[0] == "`" and "`" in opened.group(2)):
            fence = (opened.group(1)[0], len(opened.group(1)))
            continue
        out.append(_RE_INLINE_CODE.sub("", line))
    return "\n".join(out)


def _visible_text(path: Path) -> str | None:
    """pandoc `-f gfm -t html` 去标签 ＋ 实体解码 = 读者可见文本。

    pandoc 缺席或渲染失败返回 None——**必须与「检查通过」可区分**（调用方报跳过）。
    """
    try:
        r = subprocess.run(("pandoc", "-f", "gfm", "-t", "html", str(path)),
                           capture_output=True, text=True)
    except OSError:
        return None
    if r.returncode != 0:
        return None
    return html.unescape(re.sub(r"<[^>]+>", "", r.stdout))


def _norm_amount(tok: str) -> str:
    return "$" + tok[1:].lstrip()


def check_money_render(root: Path) -> tuple[list[str], list[str]]:
    """金额 `$` 是否在渲染后存活。返回（问题列表, 跳过说明列表）。"""
    mds = list(_iter(root, "*.md"))
    if mds and _visible_text(mds[0]) is None:
        return [], ["金额渲染存活：未找到可用的 pandoc（`pandoc -f gfm`）——"
                    "本项**跳过**，不计入判定；有 pandoc 的环境请复跑"]
    problems: list[str] = []
    for md in mds:
        rel = md.relative_to(root).as_posix()
        src_lines = md.read_text(encoding="utf-8", errors="replace").splitlines()
        raw = _strip_code_contexts("\n".join(src_lines))
        vis = _visible_text(md)
        if vis is None:
            problems.append(f"[金额渲染] {rel}: pandoc 渲染失败，无法校验")
            continue
        want = collections.Counter(_norm_amount(m.group(0))
                                   for m in _RE_DOLLAR_AMOUNT.finditer(raw))
        got = collections.Counter(_norm_amount(m.group(0))
                                  for m in _RE_DOLLAR_AMOUNT.finditer(vis))
        for tok, n in (want - got).items():
            where = [i for i, l in enumerate(src_lines, 1) if tok.lstrip("$") in l]
            loc = f"（源码行 {', '.join(map(str, where[:3]))}）" if where else ""
            problems.append(f"[金额渲染] {rel}: 渲染后丢失 {n} 处 `{tok}`{loc}"
                            f"——该行有成对的 `$`，金额会被当成数学吃进 span（写成 `\\$`）")
    return problems, []


def check_python(root: Path) -> list[str]:
    problems: list[str] = []
    for py in _iter(root, "*.py"):
        rel = py.relative_to(root).as_posix()
        try:
            ast.parse(py.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError as exc:
            problems.append(f"[语法错误] {rel}:{exc.lineno} {exc.msg}")
    return problems


def check_shell(root: Path) -> list[str]:
    problems: list[str] = []
    for sh in _iter(root, "*.sh"):
        rel = sh.relative_to(root).as_posix()
        proc = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
        if proc.returncode != 0:
            problems.append(f"[语法错误] {rel}: {proc.stderr.strip().splitlines()[:1]}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="复现路径静态校验")
    ap.add_argument("target", nargs="?", default=".", help="待校验目录（默认当前目录）")
    args = ap.parse_args()

    root = Path(args.target).resolve()
    if not root.is_dir():
        print(f"错误：{root} 不是目录", file=sys.stderr)
        return 2

    n_py = sum(1 for _ in _iter(root, "*.py"))
    n_md = sum(1 for _ in _iter(root, "*.md"))
    n_sh = sum(1 for _ in _iter(root, "*.sh"))
    print(f"校验目录：{root}")
    print(f"文件：{n_md} 个 .md · {n_py} 个 .py · {n_sh} 个 .sh\n")

    link_problems, link_exempt = check_links(root)
    ref_problems, ref_exempt = check_doc_refs(root)
    hidden_problems = check_hidden_paths(root)[0]  # 本检查无豁免机制

    money_problems, money_skip = check_money_render(root)

    results = [
        ("相对链接有效性", link_problems),
        ("文档路径引用", ref_problems),
        ("隐藏工作区路径", hidden_problems),
        ("表格列数一致性", check_tables(root)),
        ("Python 语法", check_python(root)),
        ("Shell 语法", check_shell(root)),
        ("金额 `$` 渲染存活", money_problems),
    ]

    total = 0
    for name, problems in results:
        if problems:
            print(f"✗ {name}：{len(problems)} 个问题")
            for p in problems:
                print(f"    {p}")
            total += len(problems)
        else:
            print(f"✓ {name}")

    for s in money_skip:
        print(f"○ {s}")

    exempt = link_exempt + ref_exempt
    if exempt:
        print(f"\n○ 已登记豁免 {len(exempt)} 项（不计入判定）")
        for e in exempt:
            print(f"    {e}")

    print()
    if total:
        print(f"判定：FAIL（{total} 个问题）")
        return 1
    print("判定：PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
