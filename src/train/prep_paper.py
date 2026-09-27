"""exp003 考卷准备（零 GPU）：tev1 官方考卷 → 本项目评测格式。

考卷本体取自 `resources/tev1/data/v1/records/{test,policy_transfer}.jsonl`
（v1 test 1,000 ＋ policy_transfer 300），**不是**按 id 在重建数据中反查——
实测该反查交集为 0（官方考卷取 test split，我们重建的是 train/dev）。

三重校验（任一失败即中止）：
1. 与官方逐题结果 `evaluation/new-v1-4b/results.jsonl` 按 id 对齐，gold 逐条一致
2. 分 source 计数与官方 README 的 main breakdown 一致
3. 与 ReJev train/holdout/dev 三口径（group / statehash / id）零重叠

产出 `data/paper/tev1-paper.jsonl`：{id, source, answer, messages, paper_split}
（messages 只含 system+user，与 holdout-eval.jsonl 同构；评测时按各自 tokenizer 渲染）。

用法：uv run python src/train/prep_paper.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/data"))
from render import build_messages, statehash  # noqa: E402

OFFICIAL = REPO / "resources/tev1/evaluation/new-v1-4b/results.jsonl"
# 官方 README 的 main test breakdown（1,000 条的 source 计数）
OFFICIAL_MAIN_BREAKDOWN = {"ag_news": 100, "banking77": 200, "boolq": 200,
                           "mnli": 250, "policy": 150, "sst5": 100}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def main() -> int:
    import hashlib

    src = {"test": REPO / "resources/tev1/data/v1/records/test.jsonl",
           "policy_transfer": REPO / "resources/tev1/data/v1/records/policy_transfer.jsonl"}
    paper: list[dict] = []
    for split, path in src.items():
        rows = load_jsonl(path)
        for r in rows:
            r["paper_split"] = split
        paper.extend(rows)
    print(f"考卷载入：{len(paper)} 条 "
          f"({Counter(r['paper_split'] for r in paper)})")

    # ── 校验 1：与官方逐题结果对齐 ──
    official = {json.loads(l)["id"]: json.loads(l) for l in OFFICIAL.open(encoding="utf-8")}
    assert len(official) == 1300, f"官方 results 条数异常 {len(official)}"
    ours = {r["id"]: r for r in paper}
    assert set(ours) == set(official), (
        f"id 集不一致：缺 {len(set(official) - set(ours))}，多 {len(set(ours) - set(official))}")
    gold_bad = [i for i in official if official[i]["gold"] != ours[i]["answer"]]
    assert not gold_bad, f"gold 不一致 {len(gold_bad)} 条：{gold_bad[:5]}"
    # paper_split 与官方 split 字段一致
    split_bad = [i for i in official if official[i]["split"] != ours[i]["paper_split"]]
    assert not split_bad, f"split 标注不一致 {len(split_bad)} 条：{split_bad[:5]}"
    print(f"校验 1 ✅ id/gold 与官方逐条一致（{len(official)}/1300）")

    # ── 校验 2：main breakdown ──
    main_src = Counter(r["source"] for r in paper if r["paper_split"] == "test")
    assert dict(main_src) == OFFICIAL_MAIN_BREAKDOWN, (
        f"main source 分布不符：{dict(main_src)}")
    print(f"校验 2 ✅ main 分 source 与官方 README 一致：{dict(main_src)}")

    # ── 校验 3：与 ReJev 三个切分零重叠 ──
    leak = {}
    for name in ("train", "holdout", "dev"):
        rows = load_jsonl(REPO / f"data/rejev/records/rejev-{name}.jsonl")
        g = {(r["source"], r["group_id"]) for r in rows} & {(r["source"], r["group_id"]) for r in paper}
        s = {statehash(r["state"]) for r in rows} & {statehash(r["state"]) for r in paper}
        i_ = {r["id"] for r in rows} & set(ours)
        leak[name] = {"group": len(g), "state": len(s), "id": len(i_)}
        assert not (g or s or i_), f"{name} 与考卷重叠：{leak[name]}"
    print(f"校验 3 ✅ 三口径零泄漏：{leak}")

    # ── 构造 records（messages 只含 system+user）──
    # **先不落盘**：渲染一致性校验未过之前，不得在磁盘上留下「正式考卷」，
    # 否则后续上传流程会拿到没通过校验的文件
    out_dir = REPO / "data/paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tev1-paper.jsonl"
    records = [{"id": r["id"], "source": r["source"], "answer": r["answer"],
                "paper_split": r["paper_split"], "messages": build_messages(r)[:-1]}
               for r in paper]

    # ── prompt 长度分布 ＋ 渲染一致性校验（两套 tokenizer）──
    # 渲染校验补的是「id/gold 相同但正文被改」这个盲区：用 Tev1-4B 自带 tokenizer
    # 渲染我们的 messages，须与官方 instruction 层的 prompt 逐字一致——这样被验证的
    # 就不只是 id 与答案，而是真正送进模型的输入。
    lengths = {}
    render_check = {}
    for label, repo_id, rev, hf_home in (
            ("minicpm", "openbmb/MiniCPM5-2B", "12a3808a956f869c767195e9266b59c4d21d92e2", None),
            ("tev1", "togethercomputer/Tev1-4B-experimental",
             "0b7becf017daa0e5eb222f8ce7483c8c8259c52f", str(REPO / "resources/tev1/.cache"))):
        try:
            import os
            if hf_home:
                os.environ.setdefault("HF_HOME", hf_home)
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(repo_id, revision=rev)
            ls = sorted(len(tok.apply_chat_template(
                r["messages"], tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_dict=True)["input_ids"]) for r in records)
            q = lambda x: ls[min(len(ls) - 1, int(len(ls) * x))]
            lengths[label] = {"p50": q(0.5), "p95": q(0.95), "p99": q(0.99), "max": ls[-1]}
            if label == "tev1":
                # 行序与 records 一致（官方 exports 与 records 同序，已逐条核过）
                bad = []
                for split, path in src.items():
                    off_rows = load_jsonl(
                        REPO / f"resources/tev1/data/v1/instruction/{split}.jsonl")
                    mine = [r for r in records if r["paper_split"] == split]
                    assert len(off_rows) == len(mine), (
                        f"{split}: 官方 instruction {len(off_rows)} 条 ≠ 我方 {len(mine)} 条"
                        f"——zip 会静默截断，拒绝继续")
                    for r, o in zip(mine, off_rows):
                        ours = tok.apply_chat_template(
                            r["messages"], tokenize=False, add_generation_prompt=True,
                            enable_thinking=False)
                        if ours != o["prompt"]:
                            bad.append(r["id"])
                render_check = {"n_checked": len(records), "prompt_mismatch": len(bad),
                                "sample": bad[:3]}
                assert not bad, f"渲染与官方 instruction 不一致：{len(bad)} 条，{bad[:5]}"
        except AssertionError:
            raise
        except Exception as e:  # noqa: BLE001
            if label == "tev1":
                # tev1 的逐字渲染校验是「同考卷」忠实度的核心证据，不可静默跳过：
                # tokenizer 拉不到就当作没有证据，中止产出（而非写出无证据的考卷）
                raise RuntimeError(
                    f"Tev1-4B 渲染校验未能执行（{type(e).__name__}: {str(e)[:200]}）——"
                    f"考卷忠实度无证据，中止") from e
            lengths[label] = f"跳过：{type(e).__name__}: {str(e)[:120]}"
    print(f"prompt 长度：{json.dumps(lengths, ensure_ascii=False)}")
    print(f"渲染一致性：{json.dumps(render_check, ensure_ascii=False)}")
    assert render_check.get("prompt_mismatch") == 0 and render_check.get("n_checked"), (
        f"渲染校验未通过或未执行：{render_check}——不产出考卷")

    # ── 全部校验通过，落盘 ──
    with out_path.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"→ {out_path}（{len(records)} 条）")

    manifest = {
        "paper": "tev1-official",
        "n": len(records),
        "splits": dict(Counter(r["paper_split"] for r in records)),
        "sources": dict(Counter(r["source"] for r in records)),
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "official_source": "resources/tev1/data/v1/records/{test,policy_transfer}.jsonl",
        "official_results": str(OFFICIAL.relative_to(REPO)),
        "checks": {"id_gold_alignment": "1300/1300", "breakdown": "match",
                   "leak": leak, "render_vs_official": render_check or "未执行"},
        "prompt_len": lengths,
        "tokenizers": {  # 渲染/校验所用 tokenizer 的钉定 revision（审计用）
            "minicpm": "openbmb/MiniCPM5-2B@12a3808a956f869c767195e9266b59c4d21d92e2",
            "tev1": "togethercomputer/Tev1-4B-experimental@0b7becf017daa0e5eb222f8ce7483c8c8259c52f",
        },
        "note": "考卷为 tev1 官方开发集；对本项目 train/holdout/dev 三口径零泄漏，"
                "对 Tev1-4B 是其自身开发集（有主场效应，报告须标注）",
    }
    (out_dir / "paper-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
