"""实测 S0 交付版与 final-check2 的逐字节一致性；写 rebuild evidence JSON。

只读 data/ 下的 records，不读题干内容；输出 SHA 与布尔结论。
用法：
    uv run python src/data/verify_rebuild_evidence.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
FINAL = REPO / "data/eval2-final-20261017/records"
CHECK2 = REPO / "data/eval2-final-check2/records"
OUT = REPO / "docs/plan/009-rebuild-evidence.json"
SPLITS = ("dev2", "calibration2", "test2")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    final = {split: sha256_file(FINAL / f"{split}.jsonl") for split in SPLITS}
    check2 = {split: sha256_file(CHECK2 / f"{split}.jsonl") for split in SPLITS}
    identical = final == check2
    result = {
        "status": "PASS" if identical else "FAIL",
        "final_dir": str(FINAL.parent),
        "check2_dir": str(CHECK2.parent),
        "final": final,
        "check2": check2,
        "byte_identical": identical,
        "note": "逐文件 SHA256 实测；同一脚本版本 final ↔ final-check2。",
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "byte_identical": identical},
                     ensure_ascii=False))
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
