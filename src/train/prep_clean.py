"""006 干净权重轮的数据准备（零 GPU）：剔源 → 切分校验 → 渲染 → 训练/评测 jsonl。

**为什么训练集取自 `rejev2-train` 而不是 `rejev-train`**：后者含 holdout2（它本身是从
`rejev-train` 里切出来的），按 issue 字面口径剔源得 32,623 会与 holdout2 **重叠 1,497 条**
（实测）。正确口径是本文件的做法——见 docs/plan/006_clean-weights.md「更正 1」。

**评测集由存量构成**（holdout1 开发集 1,892 ＋ holdout2 封存集 1,802 = 3,694），与训练集
双口径零泄漏；**保留 341 条 ag_news/sst5 题目**——评测集不训练、不随权重发布，且这是
「剔源代价」唯一可量化的地方。

产出（data/clean/ 下）：
- records/clean-train.jsonl         剔源训练 records（30,987 条）
- records/eval-set.jsonl            评测 records（3,694 条）
- train-clean.jsonl                 训练 sft/messages（30,987 条）
- eval-set.jsonl                    评测集（messages 前两轮 + answer + source）
- clean-split-manifest.json         封存 manifest（sha256 ＋ 零泄漏断言）
- prep-report.json                  长度分布与统计

用法：uv run python src/train/prep_clean.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/data"))
from render import MODEL, REV, build_messages, render_records, statehash  # noqa: E402

DROP = ("ag_news", "sst5")
N_TRAIN_SRC = 34146      # rejev2-train
N_HO1, N_HO2 = 1892, 1802
N_DROP_BY_SOURCE = {"ag_news": 1354, "sst5": 1805}
N_DROP = sum(N_DROP_BY_SOURCE.values())   # 3,159
N_CLEAN = N_TRAIN_SRC - N_DROP            # 30,987
N_EVAL = N_HO1 + N_HO2                    # 3,694
N_EVAL_RESTRICTED = 341                   # ho1(175) + ho2(166) 合计——**不是** ho2 单独

SEAL_NOTE = (
    "clean-split 封存于 2026-09-25（006 干净权重轮前）；训练集 = rejev2-train 剔 ag_news/sst5；"
    "评测集 = holdout1（开发集，已由 exp002/003/005 消费）+ holdout2（004 封存集）合并；"
    "配方训前锁定（与 004 探针逐字相同），未依据任何 holdout 结果调整。"
    "评测只在训练结束后一次；逐题错误信息不得用于设计下一轮训练/调参。"
)


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    # ── 1. 读源（条数硬断言）──
    src_tr = REPO / "data/rejev2/records/rejev2-train.jsonl"
    src_ho1 = REPO / "data/rejev/records/rejev-holdout.jsonl"
    src_ho2 = REPO / "data/rejev2/records/rejev2-holdout.jsonl"
    src_dev = REPO / "data/rejev/records/rejev-dev.jsonl"

    tr = load_jsonl(src_tr)
    ho1 = load_jsonl(src_ho1)
    ho2 = load_jsonl(src_ho2)
    dev = load_jsonl(src_dev)
    assert len(tr) == N_TRAIN_SRC, f"rejev2-train 条数异常：{len(tr)}"
    assert len(ho1) == N_HO1, f"holdout1 条数异常：{len(ho1)}"
    assert len(ho2) == N_HO2, f"holdout2 条数异常：{len(ho2)}"

    # ── 2. 封存校验：两个 holdout 与各自 manifest 的 sha 比对（封存被破坏即停）──
    m1 = json.loads((REPO / "data/rejev/holdout-manifest.json").read_text())
    m2 = json.loads((REPO / "data/rejev2/holdout2-manifest.json").read_text())
    sha_ho1, sha_ho2 = sha256_file(src_ho1), sha256_file(src_ho2)
    assert sha_ho1 == m1["files"]["rejev-holdout.jsonl"]["sha256"], "holdout1 封存被破坏"
    assert sha_ho2 == m2["files"]["rejev2-holdout.jsonl"]["sha256"], "holdout2 封存被破坏"

    # ── 3. 剔源 + 合并评测集 ──
    clean = [r for r in tr if r["source"] not in DROP]
    dropped = [r for r in tr if r["source"] in DROP]
    evalset = ho1 + ho2
    assert len(clean) == N_CLEAN, f"剔源后条数 {len(clean)} ≠ {N_CLEAN}"
    assert len(dropped) == N_DROP, f"剔除条数 {len(dropped)} ≠ {N_DROP}"
    per_source = {s: sum(1 for r in dropped if r["source"] == s) for s in DROP}
    # 对**实际出现在 dropped 里的源集**断言——不是对 per_source 的键集（那由 DROP 构造，恒真）
    seen = {r["source"] for r in dropped}
    assert seen == set(DROP), f"剔除集里出现的源是 {sorted(seen)}，与 DROP={sorted(DROP)} 不符"
    assert set(N_DROP_BY_SOURCE) == set(DROP), "DROP 与 N_DROP_BY_SOURCE 键集不同步（常量已漂移）"
    assert per_source == N_DROP_BY_SOURCE, (
        f"逐源剔除数不符：{per_source} ≠ {N_DROP_BY_SOURCE}——总数对得上不代表拆分对得上")
    assert len(evalset) == N_EVAL, f"评测集条数 {len(evalset)} ≠ {N_EVAL}"
    n_res = sum(1 for r in evalset if r["source"] in DROP)
    assert n_res == N_EVAL_RESTRICTED, f"评测集受限源题目 {n_res} ≠ {N_EVAL_RESTRICTED}"

    # ── 3b. 两个 holdout 必须互斥 + 合并集 ID 唯一 ──
    # 只查「clean × eval」不够：两份 holdout 若彼此有重复题，合并后行数仍是 3,694，
    # 但评测会把同一组重复计权（且 by_source 分层失真）。
    ho1_s, ho2_s = {statehash(r["state"]) for r in ho1}, {statehash(r["state"]) for r in ho2}
    assert not ho1_s & ho2_s, f"holdout1 × holdout2 有 {len(ho1_s & ho2_s)} 条 statehash 重叠"
    ho1_g, ho2_g = ({(r["source"], r["group_id"]) for r in ho1},
                    {(r["source"], r["group_id"]) for r in ho2})
    assert not ho1_g & ho2_g, f"holdout1 × holdout2 有 {len(ho1_g & ho2_g)} 组重叠"
    ev_ids = [r["id"] for r in evalset]
    assert len(set(ev_ids)) == len(ev_ids), \
        f"合并评测集 ID 重复 {len(ev_ids) - len(set(ev_ids))} 条（续跑会静默跳过）"

    # ── 4. 零泄漏断言（statehash / group 双口径，三对组合）──
    def keys(rows):
        return ({statehash(r["state"]) for r in rows},
                {(r["source"], r["group_id"]) for r in rows})

    c_s, c_g = keys(clean)
    e_s, e_g = keys(evalset)
    d_s, d_g = keys(dev)
    leak = {"clean_x_eval": (len(c_s & e_s), len(c_g & e_g)),
            "clean_x_dev": (len(c_s & d_s), len(c_g & d_g)),
            "eval_x_dev": (len(e_s & d_s), len(e_g & d_g))}
    for pair, (ns, ng) in leak.items():
        assert ns == 0 and ng == 0, f"切分泄漏 {pair}: state={ns} group={ng}"

    # ── 5. 落 records ──
    out = REPO / "data/clean"
    rec_dir = out / "records"
    write_jsonl(rec_dir / "clean-train.jsonl", clean)
    write_jsonl(rec_dir / "eval-set.jsonl", evalset)

    # ── 6. 渲染训练集（MiniCPM tokenizer；render_one 失败即拒收）──
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    items, stats = render_records(clean, tok)
    assert stats["n"] == N_CLEAN, f"渲染条数 {stats['n']} ≠ {N_CLEAN}"

    # ── 6b. 消融有效性：clean 臂与 probe 臂的保留题目必须逐条同文 ──
    # clean 重新渲染、probe 直接复制既有 sft 导出——若渲染实现发生漂移，
    # 「唯一变量是剔源」就不成立（会混入渲染差异）。按 id 对齐，逐条断言 messages 全等。
    probe_list = load_jsonl(REPO / "data/probe/train-probe.jsonl")
    probe_ids = [r["id"] for r in probe_list]
    clean_ids = [i["id"] for i in items]
    assert len(set(probe_ids)) == len(probe_ids), \
        f"probe 训练集 ID 重复 {len(probe_ids) - len(set(probe_ids))} 条——压成字典比对会漏检"
    assert len(set(clean_ids)) == len(clean_ids), \
        f"clean 训练集 ID 重复 {len(clean_ids) - len(set(clean_ids))} 条——压成字典比对会漏检"
    probe_rows = {r["id"]: r["messages"] for r in probe_list}
    clean_msgs = {i["id"]: i["sft"]["messages"] for i in items}
    missing = set(clean_msgs) - set(probe_rows)
    assert not missing, (
        f"clean 训练集有 {len(missing)} 条不在 probe 训练集中——probe 应是 clean 的超集（含受限源），"
        f"超出即口径异常，剔源消融不成立")
    drift = [i for i, m in clean_msgs.items() if probe_rows[i] != m]
    assert not drift, (
        f"{len(drift)} 条保留题目的 messages 与 probe 臂不一致（例：{drift[:3]}）——"
        f"重新渲染漂移会污染消融，须先查渲染层再继续")

    write_jsonl(out / "train-clean.jsonl", [{"id": i["id"], "messages": i["sft"]["messages"]}
                                            for i in items])

    # ── 7. 评测集 jsonl（messages 前两轮 + answer）──
    eval_rows = [{"id": r["id"], "source": r["source"], "answer": r["answer"],
                  "messages": build_messages(r)[:-1]} for r in evalset]
    write_jsonl(out / "eval-set.jsonl", eval_rows)

    # ── 7b. 落盘完整性：重开并**逐行解析**（只数行数抓不到「末行截断但仍占一行」）──
    for p, expect in ((rec_dir / "clean-train.jsonl", len(clean)),
                      (rec_dir / "eval-set.jsonl", len(evalset)),
                      (out / "train-clean.jsonl", len(clean)),
                      (out / "eval-set.jsonl", len(evalset))):
        n_ok = 0
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    json.loads(line)   # 解析失败即抛
                    n_ok += 1
        assert n_ok == expect, f"{p} 落盘不完整：{n_ok} 行可解析 ≠ {expect}"

    # ── 8. prompt 长度分布（抽 150；沿用 1900 上限）──
    sample = random.Random(7).sample(eval_rows, min(150, len(eval_rows)))
    lens = sorted(len(tok.apply_chat_template(r["messages"], tokenize=True,
                                              add_generation_prompt=True,
                                              enable_thinking=False,
                                              return_dict=True)["input_ids"])
                  for r in sample)
    q = lambda x: lens[min(len(lens) - 1, int(len(lens) * x))]
    plen = {"p50": q(0.5), "p95": q(0.95), "p99": q(0.99), "max": lens[-1],
            "n_sampled": len(lens)}
    assert plen["p99"] < 1900, f"评测集 prompt p99 超限：{plen}"

    # ── 9. 封存 manifest ──
    manifest = {
        "created": "2026-09-25",
        "purpose": "006 干净权重轮：剔源切分 + 评测集（评测集为存量构成，非新切）",
        "dropped_sources": {s: sum(1 for r in dropped if r["source"] == s) for s in DROP},
        "sources": {
            "train_src": {"path": str(src_tr.relative_to(REPO)), "sha256": sha256_file(src_tr),
                          "n": len(tr)},
            "holdout1_src": {"path": str(src_ho1.relative_to(REPO)), "sha256": sha_ho1,
                             "n": len(ho1), "role": "开发集（已消费）"},
            "holdout2_src": {"path": str(src_ho2.relative_to(REPO)), "sha256": sha_ho2,
                             "n": len(ho2), "role": "004 封存集"},
        },
        "counts": {"train": len(clean), "eval": len(evalset),
                   "dropped": len(dropped), "eval_restricted_items": n_res},
        "sources_train": dict(Counter(r["source"] for r in clean)),
        "sources_eval": dict(Counter(r["source"] for r in evalset)),
        "render": {k: stats[k] for k in ("n", "tokens", "max_tokens", "groups")},
        "files": {
            "records/clean-train.jsonl": {"sha256": sha256_file(rec_dir / "clean-train.jsonl"),
                                          "rows": len(clean)},
            "records/eval-set.jsonl": {"sha256": sha256_file(rec_dir / "eval-set.jsonl"),
                                       "rows": len(evalset)},
            "train-clean.jsonl": {"sha256": sha256_file(out / "train-clean.jsonl"),
                                  "rows": len(clean)},
            "eval-set.jsonl": {"sha256": sha256_file(out / "eval-set.jsonl"),
                               "rows": len(evalset)},
        },
        "seal": {"leak_check": leak, "note": SEAL_NOTE},
    }
    (out / "clean-split-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = {"dropped": len(dropped), "train": len(clean), "eval": len(evalset),
              "eval_restricted_items": n_res, "prompt_len": plen,
              "render_tokens": stats["tokens"], "leak_check": leak}
    (out / "prep-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n→ {out}/ · 封存 manifest: clean-split-manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
