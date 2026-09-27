"""一次性迁移：给早期结果产物逐行补 mode / model_requested / exam_sha256。

## 为什么需要

「逐行身份字段」是第 1 轮评审后才加进 writer 的，所以**在那之前写下的产物逐行都没有它们**：
`jev-holdout.jsonl`(1892) 与 `jev-native.jsonl`(250)。后果不是「少几个字段」这么轻——
`check_identity` 的三个集合恒空、三项断言**全部空转**，于是 `--out` 跨模式复用会静默通过。
其中一条路径尤其重：把 choice 结果读进 native 文件后，`jev_compare.py` 的 `native_arm`
会变成全 source 0.00pp，**伪造出「原生措辞无增益」的结论**。

那条 ✅ 在归档表里从第 1 轮挂到第 6 轮、五轮无人复核，因为**它只在代码里生效、对已有产物从未生效**。

## 取值依据（逐行可核验，不靠「同目录 summary」单独作证）

- `mode`：由**每一行的 `ans_type`** 推导（choice→choice；score/noul→native-kind），
  再与同目录 summary 的顶层 `mode` **交叉断言**——两者必须一致，任一缺失/矛盾即拒绝。
- `model_requested`：本次钉死的 `MODEL`。
- `exam_sha256`：封存考卷的 sha256（两份产物都是对封存考卷跑出来的）。

## 安全性（第八、九轮加固）

- 指纹覆盖**整行内容减去三个回填键**——曾经只哈希 5 个统计字段，改 `source`/`probs`/
  `confidence` 都检不出来（两谱系各自指出）。
- **两份文件全部校验并在内存构造完毕后，才统一写盘**（第九轮）——曾经是「逐文件先验
  后写」，第二份被拒时第一份已落盘，整次运行不原子。
- 写盘后**断言**读回的全行指纹等于写前指纹、且三字段齐备（第九轮）——曾经只 `print`
  一个「（须相同）」，重复 id 触发去重导致指纹变化时脚本仍成功退出。
- 写盘前断言 **id 唯一**：`_rewrite` 按 id 去重，有重复 id 会静默丢行。
- 显式 `null`/空串与缺键**同样会被修复**（`setdefault` 修不了显式空值）。
- 已有非空值必须等于期望值，否则拒绝（不做静默覆盖）。
- `ans_type` 逐行核验；**无效答案行**（`invalid_answer`：`correct=False` 但不写
  `ans_type`）改从 `raw_answer.type` 取类型核验（第九轮）——否则这类正常行会被误拒。

用法：
  uv run python src/train/backfill_identity_fields.py --check   # 只报告
  uv run python src/train/backfill_identity_fields.py           # 执行回填
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jev_api_eval import MODEL, SEALED_EXAM, _rewrite  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO / "data/exp005"
STEMS = ("jev-holdout", "jev-native")
FIELDS = ("mode", "model_requested", "exam_sha256")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]


def fingerprint(rows: list[dict]) -> str:
    """整行内容（剔除三个回填键）的哈希——回填前后必须相同。

    剔除键的副本用 dict(...) 重建，避免就地修改原行。
    """
    stripped = [{k: v for k, v in r.items() if k not in FIELDS} for r in rows]
    payload = json.dumps(stripped, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def expected_mode(rows: list[dict]) -> str:
    """由逐行 ans_type 推导模式；逐行核验，不接受「整集合模糊一致」。

    - 已判定行（correct 非 None）必须有 ans_type 且与所属模式的合法集一致；
    - 故障行（correct 为 None）可能没有 ans_type，跳过内容核验；
    - 全部行都没有 ans_type（如整文件皆故障行）→ 无法判定，拒绝。
    """
    types = {r.get("ans_type") for r in rows if r.get("ans_type")}
    if not types:
        raise SystemExit("所有行都无 ans_type——无法判定模式，拒绝回填")
    if types <= {"choice"}:
        want = "choice"
    elif types <= {"score", "noul"}:
        want = "native-kind"
    else:
        raise SystemExit(f"ans_type 混杂（{sorted(types)}）——无法判定模式，拒绝回填")
    legal = {"choice"} if want == "choice" else {"score", "noul"}
    for n, r in enumerate(rows, 1):
        t = r.get("ans_type")
        if t is None:
            if r.get("correct") is None:
                continue  # 故障行：传输/协议错误，可能没有 ans_type
            # 无效答案行（invalid_answer）：run_one 记 correct=False 但**不写 ans_type**，
            # 类型只留在原始响应里。第九轮指出：只看 ans_type 会把这类正常行误拒。
            t = ((r.get("raw_answer") or {}).get("type"))
            if t is None:
                raise SystemExit(
                    f"第 {n} 行已判定但既无 ans_type 也无 raw_answer.type——无法核验，拒绝回填")
        if t not in legal:
            raise SystemExit(f"第 {n} 行类型 {t!r} 不属 {want} 的合法集——拒绝回填")
    return want


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--check", action="store_true", help="只报告，不写盘")
    args = ap.parse_args()

    data = Path(args.dir)
    exam_sha = hashlib.sha256(SEALED_EXAM.read_bytes()).hexdigest()
    expected = {"mode": None, "model_requested": MODEL, "exam_sha256": exam_sha}

    # ── 阶段一：两份文件全部校验并在内存中构造结果（**不写盘**） ──
    # 第九轮指出：原实现是「逐文件先验后写」，第二份被拒时第一份已落盘——整次运行不原子。
    plans = []
    for stem in STEMS:
        out = data / f"{stem}.jsonl"
        rows = read_rows(out)
        before_fp = fingerprint(rows)
        have = {f: sum(1 for r in rows if r.get(f)) for f in FIELDS}
        content_mode = expected_mode(rows)

        summary_path = data / f"{stem}-summary.json"
        summary_mode = (json.loads(summary_path.read_text(encoding="utf-8")).get("mode")
                        if summary_path.exists() else None)
        assert summary_mode == content_mode, (
            f"{stem}：summary 记录模式 {summary_mode!r} 与逐行内容推断的 {content_mode!r} 不符"
            f"——拒绝回填（这正是「佐证文件与结果行不绑定」那个坑）")

        want = {**expected, "mode": content_mode}
        filled = []
        for r in rows:
            r2 = dict(r)
            for f, w in want.items():
                if not r2.get(f):          # 缺键 / null / "" 一律补
                    r2[f] = w
                elif r2[f] != w:           # 已有非空值必须相符，不静默覆盖
                    raise SystemExit(f"{stem}：某行 {f}={r2[f]!r} ≠ 期望 {w!r}——拒绝回填")
            filled.append(r2)

        assert fingerprint(filled) == before_fp, f"{stem}：回填改动了统计内容！"
        assert all(all(r.get(f) for f in FIELDS) for r in filled)
        assert len({r["id"] for r in filled}) == len(filled), (
            f"{stem}：有重复 id——写盘会按 id 去重、静默丢行，拒绝回填")
        print(f"{stem:14s} n={len(rows):5d} 内容模式={content_mode:12s} 已有字段={have}")
        plans.append((out, before_fp, filled))

    if args.check:
        return 0

    # ── 阶段二：全部写盘，并**断言**读回结果与内存构造一致 ──
    for out, before_fp, filled in plans:
        _rewrite(out, {r["id"]: r for r in filled})
        after_rows = read_rows(out)
        after_fp = fingerprint(after_rows)
        assert after_fp == before_fp, (
            f"{out.name}：写回后全行指纹不符 {before_fp} → {after_fp}——写盘被改动，须排查")
        ok = sum(1 for r in after_rows if all(r.get(f) for f in FIELDS))
        assert ok == len(after_rows), f"{out.name}：写回后仍有行缺字段 {ok}/{len(after_rows)}"
        print(f"{'':14s} 回填后 三字段齐备={ok}/{len(after_rows)}  "
              f"全行指纹 {before_fp} → {after_fp}（已断言相等）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
