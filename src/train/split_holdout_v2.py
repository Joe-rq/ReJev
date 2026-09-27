"""004 弱项探针的新封存集切分（零 GPU）。

**为什么另切**：原 holdout 已被 exp002 用于验收、被 exp003 用于三方对照，按项目纪律
「留出评测集不得用于训练或调参」，它不能再指导下一轮（review 003 的 P0 即为此预警）。
本脚本从**当前 train** 再切一个封存集，原 holdout 自此降级为**开发集**。

切分规则与 `src/data/split_holdout.py` 逐字相同：`group_id` 为最小切分单元
（同源变体永不跨侧，照 tev1 语义），按 `(source, group_id)` 分层，`statehash` /
`group` 双口径零泄漏断言。

用法：uv run python src/train/split_holdout_v2.py [--frac 0.05] [--seed 20260925]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/data"))
from render import statehash  # noqa: E402

SEAL_NOTE = ("holdout2 封存于 2026-09-25（004 弱项探针前）；不参与训练/调参；"
             "评测只在训练结束后一次。原 holdout 自此降级为开发集。")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.05, help="按 group 切的比例")
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--src", default="data/rejev/records/rejev-train.jsonl")
    ap.add_argument("--out-dir", default="data/rejev2")
    args = ap.parse_args()

    src = REPO / args.src
    rows = load_jsonl(src)
    print(f"源：{src.relative_to(REPO)} · {len(rows)} 条")
    assert len(rows) == 35948, f"源条数异常 {len(rows)}（应为 exp002 的 train 35,948）"

    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(r["source"], r["group_id"])].append(r)

    rng = random.Random(args.seed)
    ho_groups: set[tuple] = set()
    for source in sorted({s for s, _ in groups}):
        gids = sorted(g for (s, g) in groups if s == source)
        k = max(1, round(len(gids) * args.frac))
        ho_groups.update((source, g) for g in rng.sample(gids, k))

    train = [r for r in rows if (r["source"], r["group_id"]) not in ho_groups]
    holdout = [r for r in rows if (r["source"], r["group_id"]) in ho_groups]

    # 零泄漏断言（双口径，照 split_holdout.py）
    tr_states = {statehash(r["state"]) for r in train}
    ho_states = {statehash(r["state"]) for r in holdout}
    assert not tr_states & ho_states, "statehash 泄漏"
    tr_g = {(r["source"], r["group_id"]) for r in train}
    ho_g = {(r["source"], r["group_id"]) for r in holdout}
    assert not tr_g & ho_g, "group 泄漏"
    assert len(train) + len(holdout) == len(rows), "条数不守恒"

    out = REPO / args.out_dir / "records"
    write_jsonl(out / "rejev2-train.jsonl", train)
    write_jsonl(out / "rejev2-holdout.jsonl", holdout)

    def filesha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    manifest = {
        "created": "2026-09-25",
        "purpose": "004 弱项探针的新封存集；原 holdout 降级为开发集",
        "source": str(src.relative_to(REPO)),
        "source_sha256": filesha(src),
        "frac_groups": args.frac, "seed": args.seed,
        "counts": {"train": len(train), "holdout": len(holdout)},
        "groups": {"train": len(tr_g), "holdout": len(ho_g)},
        "sources": {"train": dict(Counter(r["source"] for r in train)),
                    "holdout": dict(Counter(r["source"] for r in holdout))},
        "files": {
            "rejev2-train.jsonl": {"sha256": filesha(out / "rejev2-train.jsonl"),
                                   "rows": len(train)},
            "rejev2-holdout.jsonl": {"sha256": filesha(out / "rejev2-holdout.jsonl"),
                                     "rows": len(holdout)},
        },
        "seal": {"cross_split_state_overlap": 0, "cross_split_group_overlap": 0,
                 "note": SEAL_NOTE},
    }
    mp = REPO / args.out_dir / "holdout2-manifest.json"
    mp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n→ {out}/rejev2-{{train,holdout}}.jsonl · {mp.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
