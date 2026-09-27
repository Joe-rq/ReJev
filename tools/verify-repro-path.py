#!/usr/bin/env python3
"""复现路径静态校验：验证发布物里「文档说的」与「仓库里有的」对得上。

**为什么需要它**：`REPRODUCING.md` 可以声称「引用完整、无悬空路径」，但除非有人
真的逐条查过，那只是一句自述。评审（Issue #9 双谱系）正是抓住这一点：
当时该声明**没有任何脚本支撑**。这个脚本就是那个支撑——把「已校验」变成可复跑的命令。

它做四件事（全部零依赖、不需要 GPU / token / 网络）：
  1. **行内链接**：markdown 的 `[文本](目标)` 形式所指向的仓内文件确实存在，
     且目标**未逃出发布根**（逃逸或软链目标即便在本机存在，对第三方也是死链）。
     ⚠️ 只解析行内链接——引用式 `[文本][ref]` 与 HTML `<a href>` **不在覆盖范围内**。
  2. **文档引用**：文档里以 `src/…` `tools/…` `docs/…` 形式引用的路径确实存在
  3. **Python 语法**：所有 .py 能被 `ast.parse` 解析
  4. **Shell 语法**：所有 .sh 能被 `bash -n` 解析

用法：
    uv run python scripts/verify-repro-path.py <目录>       # 默认当前仓库根
退出码：0 = 全过；1 = 有问题。
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

# 文档里形如 src/... tools/... docs/... 的路径引用（须以已知顶层目录开头）
PATH_REF = re.compile(
    r"\b((?:src|tools|docs|scripts)/[A-Za-z0-9_./-]+\.(?:py|md|sh|json|cff))\b"
)
MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

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
        "src/train/prep_exam.py",
        "plan 文档的**待建**产物（交付物表第 3 步）",
    ),
    (
        "docs/plan/005_ood-exam-eval.md",
        "src/train/exam_report.py",
        "plan 文档的**待建**产物（交付物表第 7 步）",
    ),
    (
        "docs/plan/005_ood-exam-eval.md",
        "docs/experiments/exp005-ood-exam/README.md",
        "plan 文档的**预期**归档路径（交付物表第 9 步）",
    ),
]


# 遍历时跳过的目录：这些是**环境**而非发布物的一部分。不跳过的话，跑过一次
# `uv sync` 之后再校验，就会看到 site-packages 里第三方包自带的死链——那是噪音，
# 会把真正的失败淹掉。
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".cache",
}


def _iter(root: Path, pattern: str):
    """递归找文件，但跳过 SKIP_DIRS 下的内容。"""
    for p in sorted(root.rglob(pattern)):
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
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
    for md in _iter(root, "*.md"):
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
    for md in _iter(root, "*.md"):
        rel_md = md.relative_to(root).as_posix()
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for ref in sorted(set(PATH_REF.findall(line))):
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

    results = [
        ("相对链接有效性", link_problems),
        ("文档路径引用", ref_problems),
        ("Python 语法", check_python(root)),
        ("Shell 语法", check_shell(root)),
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
