"""ReJev 切分与 holdout 封存（决策 2：自建新切分层 holdout）。

输入 tev1 new-v1 的 records（train/dev），输出 ReJev 三分：
- rejev-train   ← tev1 train 减去 holdout 部分
- rejev-holdout ← 从 tev1 train 按 (source × group_id) 分层新切，封存后不得参与训练/调参
- rejev-dev     ← tev1 dev 原样（训练监控用，不封存）

切分规则：group_id 为最小切分单元（同源变体永不跨侧，照 tev1 语义）；
按 source 分层、组数等比抽取，seed 固定可复现。封存时写 manifest：
records 行哈希、statehash 集摘要、source 分布、cross-split 断言结果。

用法：uv run python src/data/split_holdout.py \
        --records-dir resources/tev1/data/new-v1/records \
        --out-dir data/rejev [--holdout-groups-frac 0.05] [--seed 20260924]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from render import sha256_text, statehash


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def split_holdout(train_records: list[dict], frac: float, seed: int):
    """按 (source, group_id) 分层切 holdout；返回 (train, holdout, report)。"""
    groups = defaultdict(list)
    for r in train_records:
        groups[(r["source"], r["group_id"])].append(r)

    by_source = defaultdict(list)
    for (source, gid) in sorted(groups):
        by_source[source].append((source, gid))

    rng = random.Random(seed)
    ho_groups = set()
    for source in sorted(by_source):
        gids = by_source[source]
        k = max(1, round(len(gids) * frac))
        ho_groups.update(rng.sample(gids, k))

    train = [r for r in train_records if (r["source"], r["group_id"]) not in ho_groups]
    holdout = [r for r in train_records if (r["source"], r["group_id"]) in ho_groups]

    # 防泄漏断言：statehash / group 双口径不得相交（render.py 的 cross_split_check 用渲染层，
    # 这里在 records 层先挡一道）。
    tr_states = {statehash(r["state"]) for r in train}
    ho_states = {statehash(r["state"]) for r in holdout}
    assert not tr_states & ho_states, "statehash 泄漏"
    tr_groups = {(r["source"], r["group_id"]) for r in train}
    ho_groups_actual = {(r["source"], r["group_id"]) for r in holdout}
    assert not tr_groups & ho_groups_actual, "group 泄漏"

    report = {
        "train_n": len(train), "holdout_n": len(holdout),
        "train_groups": len(tr_groups), "holdout_groups": len(ho_groups_actual),
        "holdout_frac_groups": frac, "seed": seed,
        "train_sources": dict(Counter(r["source"] for r in train)),
        "holdout_sources": dict(Counter(r["source"] for r in holdout)),
        "cross_split_state_overlap": 0, "cross_split_group_overlap": 0,
    }
    return train, holdout, report


def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records-dir", required=True, help="tev1 new-v1 records 目录（含 train.jsonl/dev.jsonl）")
    ap.add_argument("--out-dir", required=True, help="输出目录（如 data/rejev）")
    ap.add_argument("--holdout-groups-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260924)
    args = ap.parse_args()

    rec_dir = Path(args.records_dir)
    out_dir = Path(args.out_dir)
    train_in = rec_dir / "train.jsonl"
    dev_in = rec_dir / "dev.jsonl"
    for p in (train_in, dev_in):
        if not p.is_file():
            print(f"缺输入：{p}", file=sys.stderr)
            return 2

    train_records = load_jsonl(train_in)
    dev_records = load_jsonl(dev_in)
    train, holdout, report = split_holdout(train_records, args.holdout_groups_frac, args.seed)

    write_jsonl(out_dir / "records" / "rejev-train.jsonl", train)
    write_jsonl(out_dir / "records" / "rejev-holdout.jsonl", holdout)
    write_jsonl(out_dir / "records" / "rejev-dev.jsonl", dev_records)

    # 封存 manifest：内容哈希固定，之后任何变动都可发现。
    manifest = {
        "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "policy": "rejev-holdout 封存后不得参与训练、调参或数据选择；评测只在训练全部结束后进行一次",
        "inputs": {"train": train_in.name, "dev": dev_in.name,
                   "train_sha256": sha256_text(train_in.read_text(encoding="utf-8")),
                   "dev_sha256": sha256_text(dev_in.read_text(encoding="utf-8"))},
        "split": report,
        "files": {
            "rejev-train.jsonl": {"sha256": None, "n": len(train)},
            "rejev-holdout.jsonl": {"sha256": None, "n": len(holdout)},
            "rejev-dev.jsonl": {"sha256": None, "n": len(dev_records)},
        },
        "holdout_statehashes_sha256": sha256_text(
            "\n".join(sorted(ho_states := {statehash(r["state"]) for r in holdout}))),
    }
    for name in list(manifest["files"]):
        p = out_dir / "records" / name
        manifest["files"][name]["sha256"] = sha256_text(p.read_text(encoding="utf-8"))
    (out_dir / "holdout-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"封存 → {out_dir / 'holdout-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
