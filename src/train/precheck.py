"""ReJev smoke 训练预检：抽样 + 本地协议断言（零 GPU，tokenizer-only）。

产出 data/smoke/smoke-500.jsonl（sft/messages 格式，seed 42 从 rejev-train 抽）
与 data/smoke/holdout-5.jsonl（seed 42 从 rejev-holdout 抽），并对样本做
token 级断言——训练模板渲染后：单 BOS、监督 span 恰为「字母＋<|im_end|>」。
预检不过，远程训练不启动（002 方案的 fail-fast 防线）。

用法：uv run python src/train/precheck.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODEL = "openbmb/MiniCPM5-2B"
REV = "12a3808a956f869c767195e9266b59c4d21d92e2"

# 官方 TRL 路线的训练专用模板（MiniCPM docs/finetune/trl.md 的结构重建）：
# bos 开头、各轮 im_start/im_end、assistant 的 content+<|im_end|> 进 generation 块。
# 训练专用、不落盘为推理模板；assistant 轮不带 think（官方称与完整推理模板兼容）。
TRAIN_TEMPLATE = (
    "{{- bos_token }}\n"
    "{%- for message in messages %}\n"
    "{%- if message['role'] == 'assistant' %}\n"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- generation %}"
    "{{- message['content'] + '<|im_end|>' }}"
    "{%- endgeneration %}\n"
    "{{- '\\n' }}\n"
    "{%- else %}\n"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}\n"
    "{%- endif %}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "{{- '<|im_start|>assistant\\n' }}\n"
    "{%- endif %}"
)

IM_END = "<|im_end|>"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def main() -> int:
    from transformers import AutoTokenizer

    train_sft = REPO / "data/rejev/rejev-train/sft/rejev-train.jsonl"
    holdout_records = REPO / "data/rejev/records/rejev-holdout.jsonl"
    out_dir = REPO / "data/smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 抽样：train 500（messages 已含 assistant=字母）、holdout 5（records 转 messages）
    rows = load_jsonl(train_sft)
    smoke = random.Random(42).sample(rows, 500)
    (out_dir / "smoke-500.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in smoke) + "\n", encoding="utf-8")

    ho = load_jsonl(holdout_records)
    ho5 = random.Random(42).sample(ho, 5)
    sys.path.insert(0, str(REPO / "src/data"))
    from render import build_messages
    (out_dir / "holdout-5.jsonl").write_text(
        "\n".join(json.dumps({"id": r["id"], "answer": r["answer"],
                              "messages": build_messages(r)[:-1]}, ensure_ascii=False)
                  for r in ho5) + "\n", encoding="utf-8")

    # 2) token 级断言（前 3 条 + 全 500 条快扫）
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    failures = []
    for r in smoke[:3] + random.Random(7).sample(smoke, 20):
        enc = tok.apply_chat_template(
            r["messages"], chat_template=TRAIN_TEMPLATE, tokenize=True,
            return_assistant_tokens_mask=True, return_dict=True)
        ids, amask = enc["input_ids"], enc["assistant_masks"]
        sup = [i for i, m in enumerate(amask) if m]
        text = tok.decode(ids)
        checks = {
            "单BOS": ids[0] == 0 and ids[1] != 0,
            "监督恰2token": len(sup) == 2,
            "监督=字母+im_end": (tok.decode(ids[sup[0]]) == r["messages"][-1]["content"]
                                 if sup else False) and (tok.decode([ids[sup[-1]]]) == IM_END if sup else False),
            "渲染含bos字面量": text.startswith("<s><|im_start|>"),
        }
        bad = [k for k, v in checks.items() if not v]
        if bad:
            failures.append({"id": r.get("id"), "bad": bad,
                             "ids_head": ids[:5], "sup": sup})

    if failures:
        print(json.dumps(failures[:3], indent=2, ensure_ascii=False))
        print(f"预检失败：{len(failures)}/{23} 条断言不过——远程训练不启动。")
        return 1

    letters = {r["messages"][-1]["content"] for r in smoke}
    print(f"预检通过：23/23 条断言全过；500 条答案字母分布 {sorted(letters)}")
    print(f"数据 → {out_dir}/smoke-500.jsonl（500 条）+ holdout-5.jsonl（5 条）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
