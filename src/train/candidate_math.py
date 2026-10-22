"""S1 候选评分与选项置换的纯函数（不 import torch / modal）。

本地单测直接调用；Modal 评测入口只把这里当作可独立核对的规则层。
置换规则锁定于 `docs/plan/009-s0-manifest.json` 的 `permutation_protocol`：
`shift = 1 + (int(sha256(id.encode('utf-8')),16) mod (n_options-1))`，保留语义 key
与 description，按新顺序重标 A..。
"""

from __future__ import annotations

import hashlib
import json
import math

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWX"


def validate_options(options) -> list[str]:
    """校验选项结构：list[dict]、label 为 A.. 连续、key 唯一。返回 label 列表。"""
    if not isinstance(options, list) or not 2 <= len(options) <= len(LETTERS):
        raise ValueError(f"选项数须在 2–{len(LETTERS)} 之间")
    if any(not isinstance(o, dict) for o in options):
        raise ValueError("选项须为对象")
    for field in ("label", "key", "description"):
        if any(field not in o for o in options):
            raise ValueError(f"选项缺字段 {field}")
    labels = [o["label"] for o in options]
    if labels != list(LETTERS[:len(options)]):
        raise ValueError("label 须为从 A 开始的连续字母")
    keys = [o["key"] for o in options]
    if len(keys) != len(set(keys)):
        raise ValueError("同一题的选项 key 必须唯一（语义还原需要）")
    return labels


def shift_for(record_id: str, n_options: int) -> int:
    """锁定的循环移位；n_options==2 时 mod 1 恒为 0，shift 恒为 1。"""
    if n_options < 2:
        raise ValueError("至少 2 个选项")
    return 1 + (int(hashlib.sha256(record_id.encode("utf-8")).hexdigest(), 16)
                % (n_options - 1))


def permute_record(record: dict, shift: int | None = None) -> dict:
    """返回带置换选项的新记录；不改原记录。answer 按答案语义 key 重算。"""
    options = record["options"]
    validate_options(options)
    n = len(options)
    if shift is None:
        shift = shift_for(record["id"], n)
    if not 1 <= shift <= n - 1:
        raise ValueError(f"shift 须在 1–{n-1}，收到 {shift}")
    rotated = options[shift:] + options[:shift]
    new_options = [{"label": LETTERS[i], "key": o["key"],
                    "description": o["description"]}
                   for i, o in enumerate(rotated)]
    answer_key = record.get("answer_key")
    if answer_key is not None:
        if answer_key not in [o["key"] for o in new_options]:
            raise ValueError("answer_key 不在选项 key 集")
        answer = new_options[[o["key"] for o in new_options].index(answer_key)]["label"]
    else:
        # 没有 answer_key 时只能按原答案字母的语义项迁移；调用方应优先给 answer_key。
        old_key = options[LETTERS.index(record["answer"])]["key"]
        answer = new_options[[o["key"] for o in new_options].index(old_key)]["label"]

    out = dict(record)
    out["options"] = new_options
    out["answer"] = answer
    out["permutation"] = {
        "shift": shift,
        "original_order": [o["key"] for o in options],
        "permuted_order": [o["key"] for o in new_options],
        "original_answer": record["answer"],
        "rule": "sha256(id) mod (n-1) + 1; left rotation; relabel A..",
    }
    return out


def softmax(xs: list[float]) -> list[float]:
    """数值稳定的 softmax（纯 Python，供单测/离线核对）。"""
    if not xs:
        raise ValueError("空输入")
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex)
    return [x / s for x in ex]


def argmax(xs: list[float]) -> int:
    return max(range(len(xs)), key=lambda i: xs[i])


def fingerprint(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def semantic_scores(probs_by_label: dict[str, float], options) -> dict[str, float]:
    """字母概率 → 语义 key 概率（同题选项置换后仍可配对）。"""
    validate_options(options)
    return {o["key"]: probs_by_label[o["label"]] for o in options}
