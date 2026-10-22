"""S0：冻结历史 statehash / family 双清单（只读历史材料，生成本地清单）。

用途（plan/009 §3.3 审计 2、§8 S0）：
- 把本项目与上游全部历史 records 的 statehash、`(source, group_id)` family 汇总成
  两份冻结清单，供 S0 生成器作隔离屏障，并供交付版审计 2 逐条比对。
- 清单随仓入库；数据本身不入库。生成时记录每个输入文件的 sha256、行数与来源。
- statehash 口径与上游 `build_v2.statehash` 逐字一致（NFKC + casefold + `\\w+`，
  dict 态用 `json.dumps(sort_keys=True)` 再归一化）；脚本在样本上与上游函数交叉核对。

用法：
    uv run python src/data/freeze_history.py
    uv run python src/data/freeze_history.py --check   # 只校验现有清单未漂移
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import unicodedata

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO / "resources/tev1"
OUT = REPO / "src/data/frozen"
STATEHASHES = OUT / "historical-statehashes.txt"
FAMILIES = OUT / "historical-families.tsv"
MANIFEST = OUT / "freeze-manifest.json"
UPSTREAM_COMMIT = "1dde7782382c9f49d627153759b8d1deab426ce0"

# 历史材料：本项目 records + 上游各版本 records。`rejev2-train` 显式列入，
# 即使 train-clean 是它的子集——屏障宁可多挡，不可漏挡。
MATERIALS = (
    ("train-clean", REPO / "data/clean/records/clean-train.jsonl", "records"),
    ("rejev2-train", REPO / "data/rejev2/records/rejev2-train.jsonl", "records"),
    ("rejev-train", REPO / "data/rejev/records/rejev-train.jsonl", "records"),
    ("holdout1", REPO / "data/rejev/records/rejev-holdout.jsonl", "records"),
    ("holdout2", REPO / "data/rejev2/records/rejev2-holdout.jsonl", "records"),
    ("dev", REPO / "data/rejev/records/rejev-dev.jsonl", "records"),
    ("exam", REPO / "data/exam/exam.jsonl", "messages"),
)


def normalized(text: str) -> str:
    """与 tev1 build_dataset.py:33-34 逐字一致。"""
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def statehash(state) -> str:
    """与 tev1 build_v2.py:48 逐字一致：dict 态先 sort_keys 序列化（默认 ensure_ascii）。"""
    text = state if isinstance(state, str) else json.dumps(state, sort_keys=True)
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_state(row: dict, kind: str):
    if kind == "records":
        return row["state"]
    # messages / exam：user 轮是 {"state","question","options"} 的 JSON。
    return json.loads(row["messages"][1]["content"])["state"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derived_state(row: dict):
    """从派生材料提取 state；无 state 的渲染/预测文件返回 None。"""
    if isinstance(row, dict) and "state" in row:
        return row["state"]
    if isinstance(row, dict) and "messages" in row:
        try:
            payload = json.loads(row["messages"][1]["content"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            return None
        return payload.get("state")
    return None


def collect_derived(covered: set[str]) -> tuple[list[dict], int]:
    """枚举 data/ 下未列入 MATERIALS 的派生 jsonl，证明其 state ⊆ 冻结并集。

    这是「枚举」而不是「列举」：新增派生文件若含未覆盖 state，会以非零
    `derived_minus_covered` 暴露出来。
    """
    main_paths = {path.resolve() for _, path, _ in MATERIALS}
    entries, missing_total = [], 0
    for path in sorted(REPO.glob("data/**/*.jsonl")):
        rel = str(path.relative_to(REPO))
        if "/eval2-" in rel or path.resolve() in main_paths:
            continue
        rows = load_jsonl(path)
        states, no_state = [], 0
        for row in rows:
            state = derived_state(row)
            if state is None:
                no_state += 1
            else:
                states.append(state)
        hashes = {statehash(state) for state in states}
        missing = hashes - covered
        missing_total += len(missing)
        entries.append({
            "path": rel,
            "rows": len(rows),
            "state_rows": len(states),
            "no_state_rows": no_state,
            "unique_statehashes": len(hashes),
            "missing_from_frozen": len(missing),
        })
    return entries, missing_total


def crosscheck_upstream(samples: list[dict]) -> None:
    """样本上对照上游 build_v2.statehash，防止本脚本口径漂移。"""
    spec = importlib.util.spec_from_file_location("tev1_build_v2", UPSTREAM / "build_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for row in samples:
        assert statehash(row["state"]) == module.statehash(row), row.get("id")


def collect() -> tuple[set[str], set[tuple[str, str]], list[dict], dict]:
    hashes: set[str] = set()
    families: set[tuple[str, str]] = set()
    files = []
    per_source_hashes: dict[str, set[str]] = {}
    samples = []
    for name, path, kind in MATERIALS:
        rows = load_jsonl(path)
        local = {statehash(row_state(row, kind)) for row in rows}
        local_families = {(row["source"], row["group_id"]) for row in rows if "group_id" in row}
        hashes |= local
        families |= local_families
        per_source_hashes[name] = local
        if rows:
            samples.append({"id": rows[0].get("id"), "state": row_state(rows[0], kind)})
        files.append({
            "name": name,
            "path": str(path.relative_to(REPO)),
            "kind": kind,
            "rows": len(rows),
            "unique_statehashes": len(local),
            "families": len(local_families),
            "sha256": sha256_file(path),
            "sources": dict(sorted(Counter(row.get("source", "?") for row in rows).items())),
        })
    upstream_files = sorted((UPSTREAM / "data").glob("*/records/*.jsonl"))
    upstream_hashes: set[str] = set()
    for path in upstream_files:
        rows = load_jsonl(path)
        local = {statehash(row_state(row, "records")) for row in rows}
        local_families = {(row["source"], row["group_id"]) for row in rows if "group_id" in row}
        hashes |= local
        families |= local_families
        upstream_hashes |= local
        files.append({
            "name": f"upstream/{path.parent.parent.name}/{path.name}",
            "path": str(path.relative_to(REPO)),
            "kind": "records",
            "rows": len(rows),
            "unique_statehashes": len(local),
            "families": len(local_families),
            "sha256": sha256_file(path),
            "sources": dict(sorted(Counter(row.get("source", "?") for row in rows).items())),
        })
    # 上游函数交叉核对（只取少量样本，避免把整个 build_v2 的导入成本带进来）。
    crosscheck_upstream(samples)
    derived, derived_missing = collect_derived(hashes)
    coverage = {
        # §3.3 修法要求的闭合证明：rejev-train 100% 落在上游 records 内。
        "rejev_train_minus_upstream_records": len(per_source_hashes["rejev-train"] - upstream_hashes),
        # exam 由第三方构造，与上游 records 应零交集（不共享池）。
        "exam_intersect_upstream_records": len(per_source_hashes["exam"] & upstream_hashes),
        # train-clean 应是 rejev2-train 的子集（剔源不引入新实例）。
        "train_clean_minus_rejev2_train": len(per_source_hashes["train-clean"] - per_source_hashes["rejev2-train"]),
        # 派生材料（data/ 下未列入 MATERIALS 的 jsonl）应全部 ⊆ 冻结并集。
        "derived_minus_covered": derived_missing,
    }
    return hashes, families, files, coverage, derived


def write_outputs(hashes: set[str], families: set[tuple[str, str]], files: list[dict],
                  coverage: dict, derived: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STATEHASHES.write_text("\n".join(sorted(hashes)) + "\n", encoding="utf-8")
    FAMILIES.write_text("\n".join(f"{s}\t{g}" for s, g in sorted(families)) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created": "2026-10-17",
        "updated": "2026-10-18",
        "purpose": "plan/009 S0：历史 statehash / family 冻结清单（隔离屏障与审计 2 的共同基准）",
        "upstream": {"repo": "togethercomputer/tev1", "commit": UPSTREAM_COMMIT},
        "materials": files,
        "derived": derived,
        "totals": {
            "unique_statehashes": len(hashes),
            "families": len(families),
            "input_files": len(files),
            "derived_files": len(derived),
        },
        "coverage": coverage,
        "files": {
            STATEHASHES.name: {"sha256": sha256_file(STATEHASHES), "lines": len(hashes)},
            FAMILIES.name: {"sha256": sha256_file(FAMILIES), "lines": len(families)},
        },
        "notes": [
            "exam 无 group_id：family 检查对 exam 为 vacuous，由 statehash 检查覆盖并披露。",
            "清单只含哈希与 family 标识，不含题干、样本内容或标签。",
            "statehash 口径与 tev1 build_v2.statehash 逐字一致（样本交叉核对）。",
            "derived 段枚举 data/ 下未列入 materials 的派生 jsonl。derived_minus_covered=0 只覆盖 state_rows>0 的文件；no_state_rows>0 的 instruction/prompt 导出与 exp005 预测文件没有 state 可查，其覆盖由同源 sft/records 文件承担，不应读成「全部 24 个文件都经过 statehash 检查」。",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def check() -> int:
    hashes, families, _, coverage, _ = collect()
    old_hashes = set(STATEHASHES.read_text(encoding="utf-8").split())
    old_families = {tuple(line.split("\t", 1)) for line in
                    FAMILIES.read_text(encoding="utf-8").splitlines() if line}
    ok = (hashes == old_hashes and families == old_families
          and coverage["derived_minus_covered"] == 0)
    print(json.dumps({
        "statehashes": len(hashes),
        "families": len(families),
        "statehash_drift": len(hashes ^ old_hashes),
        "family_drift": len(families ^ old_families),
        "derived_minus_covered": coverage["derived_minus_covered"],
        "ok": ok,
    }, ensure_ascii=False))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只校验现有清单是否漂移")
    args = parser.parse_args()
    if args.check:
        return check()
    hashes, families, files, coverage, derived = collect()
    assert coverage["rejev_train_minus_upstream_records"] == 0, coverage
    assert coverage["exam_intersect_upstream_records"] == 0, coverage
    assert coverage["train_clean_minus_rejev2_train"] == 0, coverage
    assert coverage["derived_minus_covered"] == 0, coverage
    write_outputs(hashes, families, files, coverage, derived)
    print(json.dumps({
        "statehashes": len(hashes),
        "families": len(families),
        "coverage": coverage,
        "derived_files": len(derived),
        "out": str(OUT.relative_to(REPO)),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
