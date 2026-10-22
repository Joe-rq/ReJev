"""只读复核 S0 交付版评测三切分：SHA、配额、隔离交集、记录结构。

不导入生成器、不信任 manifest 里的审计结论；从交付文件与冻结清单重算。
用法：
    uv run python src/data/verify_eval2.py \\
        --manifest docs/plan/009-s0-manifest.json \\
        --data data/eval2-final-20261017
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from freeze_history import statehash  # noqa: E402

SPLITS = ("dev2", "calibration2", "test2")
# 独立于被审 manifest 的锁定配额（plan/009 §3.3，S0 2026-10-17 锁定）。
LOCKED_QUOTAS = {
    "dev2": {"policy_v2": 174, "routing_v2": 84, "policy": 24,
             "research_taxonomy_v21": 56, "mnli": 74, "boolq": 45, "banking77": 43},
    "calibration2": {"policy_v2": 522, "routing_v2": 264, "policy": 66,
                     "research_taxonomy_v21": 168, "mnli": 218, "boolq": 131, "banking77": 131},
    "test2": {"policy_v2": 1326, "routing_v2": 666, "policy": 165,
              "research_taxonomy_v21": 424, "mnli": 554, "boolq": 332, "banking77": 333},
}
LOCKED_COUNTS = {"dev2": 500, "calibration2": 1500, "test2": 3800}
# 冻结清单自身的 SHA（S0 2026-10-17 冻结；清单漂移/截断即报错）。
LOCKED_FROZEN_SHA = {
    "statehashes_sha256": "432eb8ef7183d1ed23b0c7401a98cf4de570e2f344b1ead6633a4630d88cb622",
    "families_sha256": "956d5b78aa29ba13899f642095698873f168de138052566fc8ec983cb8b45164",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, default=REPO / "src/data/frozen")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors: list[str] = []

    files = {}
    for split in SPLITS:
        path = args.data / "records" / f"{split}.jsonl"
        record = manifest["files"][f"records/{split}.jsonl"]
        rows = load_jsonl(path)
        files[split] = rows
        if len(rows) != LOCKED_COUNTS[split]:
            errors.append(f"{split}: 行数 {len(rows)} != 锁定值 {LOCKED_COUNTS[split]}")
        if len(rows) != record["rows"]:
            errors.append(f"{split}: 行数 {len(rows)} != manifest {record['rows']}")
        if sha256_file(path) != record["sha256"]:
            errors.append(f"{split}: SHA256 与 manifest 不符")
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            errors.append(f"{split}: 重复 id")
        for row in rows:
            if row["split"] != split:
                errors.append(f"{split}: {row['id']} split 字段异常")
                break
            labels = [o["label"] for o in row["options"]]
            if labels != list("ABCDEFGHIJKLMNOPQRSTUVWX"[:len(labels)]):
                errors.append(f"{split}: {row['id']} 选项 label 不连续")
                break
            matching = [o for o in row["options"] if o["label"] == row["answer"]]
            if len(matching) != 1 or matching[0]["key"] != row["answer_key"]:
                errors.append(f"{split}: {row['id']} answer/answer_key 不一致")
                break

    frozen_hashes = set((args.frozen / "historical-statehashes.txt")
                        .read_text(encoding="utf-8").split())
    frozen_families = {tuple(line.split("\t", 1)) for line in
                       (args.frozen / "historical-families.tsv")
                       .read_text(encoding="utf-8").splitlines() if line}
    for filename, key in (("historical-statehashes.txt", "statehashes_sha256"),
                          ("historical-families.tsv", "families_sha256")):
        actual = sha256_file(args.frozen / filename)
        if actual != LOCKED_FROZEN_SHA[key]:
            errors.append(f"{filename}: SHA256 {actual} != 锁定值 {LOCKED_FROZEN_SHA[key]}")
        if actual != manifest["frozen"][key]:
            errors.append(f"{filename}: SHA256 {actual} != manifest {manifest['frozen'][key]}")
    freeze_manifest_actual = sha256_file(args.frozen / "freeze-manifest.json")
    if freeze_manifest_actual != manifest["frozen"]["manifest_sha256"]:
        errors.append("freeze-manifest.json: SHA256 与 manifest 不符")
    if len(frozen_hashes) != manifest["frozen"]["statehashes"]:
        errors.append(f"statehash 行数 {len(frozen_hashes)} != manifest {manifest['frozen']['statehashes']}")
    if len(frozen_families) != manifest["frozen"]["families"]:
        errors.append(f"family 行数 {len(frozen_families)} != manifest {manifest['frozen']['families']}")
    state_sets = {s: {statehash(row["state"]) for row in rows} for s, rows in files.items()}
    family_sets = {s: {(row["source"], row["group_id"]) for row in rows}
                   for s, rows in files.items()}
    pairwise = {}
    for i, a in enumerate(SPLITS):
        for c in SPLITS[i + 1:]:
            pairwise[f"{a}_x_{c}"] = {
                "statehash": len(state_sets[a] & state_sets[c]),
                "family": len(family_sets[a] & family_sets[c]),
            }
            if any(pairwise[f"{a}_x_{c}"].values()):
                errors.append(f"{a} x {c}: 交集非 0")
    history = {s: {"statehash": len(state_sets[s] & frozen_hashes),
                   "family": len(family_sets[s] & frozen_families)} for s in SPLITS}
    for s, counts in history.items():
        if any(counts.values()):
            errors.append(f"{s}: 与历史交集非 0 {counts}")

    quotas = {}
    for split, rows in files.items():
        got = Counter(row["source"] for row in rows)
        want = Counter(manifest["quotas"][split])
        locked = Counter(LOCKED_QUOTAS[split])
        quotas[split] = {"rows": len(rows),
                         "per_source_match_manifest": got == want,
                         "per_source_match_locked": got == locked,
                         "got": dict(sorted(got.items()))}
        if got != want:
            errors.append(f"{split}: 逐源配额与 manifest 不符")
        if got != locked:
            errors.append(f"{split}: 逐源配额与锁定值不符（manifest 可能被同步篡改）")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "counts": {s: len(rows) for s, rows in files.items()},
        "pairwise_overlap": pairwise,
        "history_overlap": history,
        "quotas": quotas,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
