"""实测并登记 ReJev 渲染契约（tokenizer-only，零 GPU）。

用 `openbmb/MiniCPM5-2B@12a3808…` 的 tokenizer 计算 A–X 的 token id、特殊 token
与 chat template 事实，写 `docs/plan/009-render-contract.json`；S0 manifest 的
`render_contract` 从这里取值，避免把外推当测量。

用法（项目 .venv 有 transformers 5.6.2；tokenizer 走本地缓存）：
    python src/data/verify_render_contract.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import transformers

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs/plan/009-render-contract.json"
MODEL = "openbmb/MiniCPM5-2B"
REV = "12a3808a956f869c767195e9266b59c4d21d92e2"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWX"


def main() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, local_files_only=True)
    ids = {letter: tok.encode(letter, add_special_tokens=False) for letter in LETTERS}
    assert all(len(value) == 1 for value in ids.values()), ids
    eos = tok.eos_token
    eos_id = tok.eos_token_id
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    assert eos == "</s>" and eos_id == 1, (eos, eos_id)
    assert im_end_id == 130073, im_end_id
    result = {
        "model": MODEL,
        "revision": REV,
        "measured_at": "2026-10-18",
        "command": "python src/data/verify_render_contract.py",
        "label_token_ids": ids,
        "single_token": True,
        "eos_token": eos,
        "eos_token_id": eos_id,
        "im_end_token_id": im_end_id,
        "generation_config_eos": [1, 130073],
        "sampling": "推理须显式 temperature=0 / do_sample=False",
        "measured_fields": ["label_token_ids", "eos_token", "eos_token_id",
                            "im_end_token_id", "single_token"],
        "from_report": {
            "generation_config_eos": "src/align/v1a-report.md C6",
            "sampling": "src/align/v1a-report.md C6",
            "assistant_suffix": "src/align/v1a-report.md C3",
            "max_tokens": "src/data/render.py MAX_TOKENS",
        },
        "measured_env": {"python": sys.version.split()[0],
                         "transformers": transformers.__version__},
        "assistant_suffix": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "max_tokens": 2048,
        "sources": ["src/align/v1a-report.md", "src/data/render.py"],
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "out": str(OUT.relative_to(REPO)),
                      "label_ids": {k: v[0] for k, v in ids.items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
