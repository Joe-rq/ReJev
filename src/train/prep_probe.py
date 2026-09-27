"""004 弱项探针的数据准备（零 GPU）：新封存集 → 训练 sft / 评测 jsonl。

产出（data/probe/ 下）：
- train-probe.jsonl      rank-64 探针的训练集（sft/messages，34,146 条）
- holdout2-eval.jsonl    新封存集的评测集（messages 前两轮 + answer + source）
- prep-report.json       封存校验 + 长度分布

用法：uv run python src/train/prep_probe.py
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/data"))
from render import build_messages  # noqa: E402


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]


def main() -> int:
    out = REPO / "data/probe"
    out.mkdir(parents=True, exist_ok=True)

    # 1) 训练集：sft 导出（已渲染）直接复制并校验
    tr_src = REPO / "data/rejev2/rejev2-train/rejev2-train/sft/rejev2-train.jsonl"
    rows = load_jsonl(tr_src)
    assert len(rows) == 34146, f"训练集条数异常：{len(rows)}"
    (out / "train-probe.jsonl").write_text(tr_src.read_text(encoding="utf-8"), encoding="utf-8")

    # 2) 评测集（records → messages 前两轮 + answer）
    ho_records = REPO / "data/rejev2/records/rejev2-holdout.jsonl"
    ho = load_jsonl(ho_records)
    assert len(ho) == 1802, f"holdout2 条数异常：{len(ho)}"
    ho_eval = [{"id": r["id"], "source": r["source"], "answer": r["answer"],
                "messages": build_messages(r)[:-1]} for r in ho]
    with (out / "holdout2-eval.jsonl").open("w", encoding="utf-8") as fh:
        for r in ho_eval:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 3) 封存校验：与 holdout2-manifest 的 sha256 比对
    manifest = json.loads((REPO / "data/rejev2/holdout2-manifest.json").read_text())
    recorded = manifest["files"]["rejev2-holdout.jsonl"]["sha256"]
    actual = hashlib.sha256(ho_records.read_bytes()).hexdigest()
    assert recorded == actual, "holdout2 与封存 manifest 不符——封存被破坏，全流程停止"

    # 4) prompt 长度（MiniCPM tokenizer）
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-2B",
                                        revision="12a3808a956f869c767195e9266b59c4d21d92e2")
    sample = random.Random(7).sample(ho_eval, min(150, len(ho_eval)))
    lens = sorted(len(tok.apply_chat_template(r["messages"], tokenize=True,
                                              add_generation_prompt=True,
                                              enable_thinking=False,
                                              return_dict=True)["input_ids"]) for r in sample)
    q = lambda x: lens[min(len(lens) - 1, int(len(lens) * x))]
    stats = {"p50": q(0.5), "p95": q(0.95), "p99": q(0.99), "max": lens[-1], "n_sampled": len(lens)}
    assert stats["p99"] < 1900, f"prompt p99 超限：{stats}"

    report = {"seal": {"holdout2_sha256": actual, "matches_sealed_manifest": True},
              "files": {"train-probe.jsonl": len(rows), "holdout2-eval.jsonl": len(ho)},
              "holdout2_prompt_len": stats,
              "source": "data/rejev2/（004 新封存集）"}
    (out / "prep-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
