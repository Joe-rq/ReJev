"""JSONL 续跑修复：末尾半行/缺换行/截断 UTF-8 与中间坏行分开处理。

S1 评测结果可能因容器被杀而留下：
- 最后一条 JSON 完整但缺换行 → 续写会形成 `}{`；
- 最后一条被截断（含 UTF-8 多字节）→ 解析失败；
- 中间坏行 → 不能静默跳过，必须归档重跑。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import uuid


def read_repair_jsonl(path: str | Path) -> tuple[list[dict], str | None, str | None]:
    """返回 (records, repair_note, error)。error 非空表示中间坏行，调用方应归档。"""
    p = Path(path)
    try:
        raw = p.read_bytes()
    except FileNotFoundError:
        return [], None, None
    if not raw.strip():
        return [], None, None

    lines = raw.split(b"\n")
    records: list[dict] = []
    tail_bad = False
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if any(l.strip() for l in lines[i + 1:]):
                return [], None, "bad_middle"
            tail_bad = True
            break

    missing_newline = not raw.endswith(b"\n")
    if tail_bad or missing_newline:
        tmp = f"{p}.tmp-{uuid.uuid4().hex[:8]}"
        with open(tmp, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, p)
        return records, ("tail_truncated" if tail_bad else "missing_newline"), None
    return records, None, None
