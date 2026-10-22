"""S1 评测前本地渲染预扫：四个 eval2 文件逐条长度/标签/答案检查。

跑法：uv run --group align python src/data/check_eval2_render.py
（不启动 GPU；用 MiniCPM5-2B tokenizer 与 S1Trainer 同模板）
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/train"))
import s1_spec as S  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO / "data/eval2-final-20261017/records"))
    ap.add_argument("--s1", default=str(REPO / "docs/plan/009-s1-manifest.json"))
    args = ap.parse_args()
    s1 = json.loads(Path(args.s1).read_text(encoding="utf-8"))
    files = s1["eval2_files"]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S.MODEL, revision=S.REV)
    report = {}
    for key in ("dev2", "calibration2", "test2", "test2_permuted"):
        name = "test2-permuted" if key == "test2_permuted" else key
        p = Path(args.data_dir) / f"{name}.jsonl"
        raw = p.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        assert sha == files[key]["sha256"], f"{key} SHA 不符"
        rows = [json.loads(l) for l in raw.decode("utf-8").split("\n") if l.strip()]
        assert len(rows) == files[key]["rows"]
        lengths = []
        for r in rows:
            assert "messages" not in r, f"{r['id']} eval2 record 带 messages 字段"
            labels = [o["label"] for o in r["options"]]
            assert labels == list(S.LETTERS[:len(labels)]), f"{r['id']} label 非连续"
            assert r["answer"] in labels and r.get("answer_key") is not None
            enc = tok.apply_chat_template(
                S.build_messages(r)[:-1], tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_dict=True)
            ids = enc["input_ids"]
            assert len(ids) < S.MAX_TOKENS, f"{r['id']} prompt {len(ids)} ≥ {S.MAX_TOKENS}"
            lengths.append(len(ids))
        report[key] = {
            "path": str(p), "sha256": sha, "rows": len(rows),
            "max_len": max(lengths), "p99": sorted(lengths)[int(len(lengths) * 0.99)],
            "over_max": sum(1 for x in lengths if x >= S.MAX_TOKENS),
        }
        print(key, report[key])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
