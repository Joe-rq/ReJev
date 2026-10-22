"""S1 选项置换变体生成器（Issue #111 §3.6）。

输入冻结的 test2 records；按锁定的循环移位规则生成变体：
`shift = 1 + (int(sha256(id.encode('utf-8')),16) mod (n_options-1))`，
保留语义 key 与 description，按新顺序重标 A..。原记录与变体同题配对。

输出：变体 jsonl（本地、gitignored）＋ manifest（仓内可追踪的 SHA/计数）。
用法：
    uv run python src/data/permute_eval2.py \
      --input data/eval2-final-20261017/records/test2.jsonl \
      --out data/eval2-final-20261017/records/test2-permuted.jsonl \
      --manifest docs/plan/009-test2-permuted.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/train"))

import candidate_math as cm  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="生成 test2 的确定性选项置换变体")
    ap.add_argument("--input", required=True, help="test2 records jsonl")
    ap.add_argument("--out", required=True, help="输出变体 jsonl（不得已存在）")
    ap.add_argument("--manifest", required=True, help="输出 manifest json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.out)
    if dst.exists():
        raise SystemExit(f"输出已存在，拒绝覆盖：{dst}")
    if Path(args.manifest).exists():
        raise SystemExit(f"manifest 已存在，拒绝覆盖：{args.manifest}")
    raw = src.read_bytes()
    src_sha = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(l) for l in raw.decode("utf-8").split("\n") if l.strip()]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "输入 id 重复"

    out_rows = []
    shift_counts = Counter()
    n_opt_counts = Counter()
    for r in rows:
        p = cm.permute_record(r)
        # 不变式：id/state/question/语义键多重集/answer_key 不变；answer 映射正确。
        assert p["id"] == r["id"]
        assert p["state"] == r["state"] and p["question"] == r["question"]
        assert set(o["key"] for o in p["options"]) == set(o["key"] for o in r["options"])
        assert p["answer_key"] == r["answer_key"]
        assert p["options"][cm.LETTERS.index(p["answer"])]["key"] == r["answer_key"]
        assert [o["label"] for o in p["options"]] == list(
            cm.LETTERS[:len(p["options"])])
        out_rows.append(p)
        shift_counts[p["permutation"]["shift"]] += 1
        n_opt_counts[len(p["options"])] += 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    out_sha = hashlib.sha256(dst.read_bytes()).hexdigest()

    manifest = {
        "source": str(src), "source_sha256": src_sha, "source_rows": len(rows),
        "output": str(dst), "output_sha256": out_sha, "output_rows": len(out_rows),
        "rule": "shift = 1 + (int(sha256(id.encode('utf-8')),16) mod (n_options-1)); "
                "left rotation options[shift:]+options[:shift]; relabel A..; "
                "semantic key & description preserved",
        "frozen_at": "2026-10-18",
        "shift_counts": {str(k): v for k, v in sorted(shift_counts.items())},
        "option_count_counts": {str(k): v for k, v in sorted(n_opt_counts.items())},
        "invariants": {
            "ids_unique": True,
            "keyset_preserved": True,
            "answer_key_preserved": True,
            "answer_label_maps_to_answer_key": True,
            "labels_contiguous_A": True,
        },
    }
    mpath = Path(args.manifest)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                     encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in
                      ("source_rows", "output_rows", "output_sha256")},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
