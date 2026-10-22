"""核验 patched 上游 `public_data` 的冻结屏障：端到端 + fail-closed。

用法（需要 datasets；用上游 venv，解释器路径按本机布置）：
    python src/data/verify_patch_barrier.py \\
        --patched <patched-dir> \\
        --out docs/plan/009-patch-barrier.json

检查：
1. `build_v2.public_data({'train': []}, lock)` 生成 v2 公开切分（9,000 行）；
2. 输出行的 statehash 与冻结清单交集为 0；
3. 缺 `REJEV_FROZEN_STATEHASHES` 时 `frozen_statehashes()` 抛 RuntimeError（fail-closed）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patched", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, default=REPO / "src/data/frozen")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    os.environ["OPEN_JEV_ROOT"] = str(args.patched)
    os.environ.setdefault("HF_HOME", str(REPO / "resources" / "tev1" / ".cache" / "huggingface"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["REJEV_SEED_V1"] = "patched-barrier-check"
    os.environ["REJEV_SEED_V2"] = "patched-barrier-check"
    os.environ["REJEV_SEED_V21"] = "patched-barrier-check"
    os.environ["REJEV_FROZEN_STATEHASHES"] = str(args.frozen / "historical-statehashes.txt")
    sys.path.insert(0, str(args.patched))
    import build_v2 as b  # noqa: E402

    lock = json.loads((args.patched / "sources.lock.json").read_text(encoding="utf-8"))
    parts, _ = b.public_data({"train": []}, lock)
    frozen = b.frozen_statehashes()
    hits = {split: sum(1 for row in rows if b.statehash(row) in frozen)
            for split, rows in parts.items()}
    counts = {split: len(rows) for split, rows in parts.items()}
    expected = {"train": sum(b.FRESH.values()), "dev": sum(b.EVAL.values()),
                "calibration": sum(b.EVAL.values()), "test": sum(b.EVAL.values())}
    ok = counts == expected and all(value == 0 for value in hits.values())

    # 负控：从无屏障输出里挑一条公开源 test 行，把它的 state 放进 old，
    # 再跑一次；该 statehash 应被 forbidden 排除，不出现在 dev/cal/test。
    target = next(row for row in parts["test"]
                  if row["source"] in ("mnli", "boolq", "banking77"))
    target_hash = b.statehash(target)
    parts_ctrl, _ = b.public_data(
        {"train": [{"source": target["source"], "state": target["state"],
                    "provenance": {}}]}, lock)
    control_hits = {split: sum(1 for row in parts_ctrl[split]
                               if b.statehash(row) == target_hash)
                    for split in ("dev", "calibration", "test")}
    negative_ok = all(value == 0 for value in control_hits.values())

    # 冻结清单侧阳性对照：把 target_hash 写进临时冻结清单，再跑一次；
    # 若 frozen_statehashes 真的被使用，该 state 应被排除。
    original = os.environ["REJEV_FROZEN_STATEHASHES"]
    with tempfile.TemporaryDirectory() as tmp:
        temp_frozen = Path(tmp) / "one-statehash.txt"
        temp_frozen.write_text(target_hash + "\n", encoding="utf-8")
        os.environ["REJEV_FROZEN_STATEHASHES"] = str(temp_frozen)
        try:
            parts_pos, _ = b.public_data({"train": []}, lock)
        finally:
            os.environ["REJEV_FROZEN_STATEHASHES"] = original
    positive_hits = {split: sum(1 for row in parts_pos[split]
                                if b.statehash(row) == target_hash)
                     for split in ("dev", "calibration", "test")}
    positive_ok = all(value == 0 for value in positive_hits.values())

    # fail-closed：子进程里去掉 env，调用 frozen_statehashes 应抛 RuntimeError。
    code = (
        "import os,sys\n"
        f"sys.path.insert(0, {str(args.patched)!r})\n"
        "import build_v2 as b\n"
        "try:\n"
        " b.frozen_statehashes()\n"
        " print('NO_ERROR')\n"
        "except RuntimeError:\n"
        " print('RUNTIME_ERROR')\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "REJEV_FROZEN_STATEHASHES"}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    fail_closed = proc.stdout.strip().endswith("RUNTIME_ERROR")

    result = {
        "status": "PASS" if ok and fail_closed and negative_ok and positive_ok else "FAIL",
        "patched_dir": str(args.patched),
        "patch_sha256": sha256_file(REPO / "src/data/upstream_patches/0001-rejev-s0.patch"),
        "frozen_statehashes_sha256": sha256_file(args.frozen / "historical-statehashes.txt"),
        "counts": counts,
        "expected_counts": expected,
        "frozen_hits": hits,
        "fail_closed": fail_closed,
        "negative_control": {
            "status": "PASS" if negative_ok else "FAIL",
            "target_source": target["source"],
            "target_statehash": target_hash,
            "target_hits_in_dev_cal_test": control_hits,
            "note": "把一个已知公开 test state 放进 old，再跑 public_data；该 statehash 被 forbidden 排除。",
        },
        "positive_control_frozen_list": {
            "status": "PASS" if positive_ok else "FAIL",
            "target_statehash": target_hash,
            "target_hits_in_dev_cal_test": positive_hits,
            "note": "把同一 statehash 写进临时冻结清单，再跑 public_data；证明 frozen_statehashes 的清单侧路径被使用。",
        },
        "note": "public_data 生成的 9,000 行全部不命中冻结清单；缺 env 时明确报错；负控（old 侧）与阳性对照（冻结清单侧）均证明屏障在排除已知 state。",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": counts, "hits": hits,
                      "fail_closed": fail_closed}, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
