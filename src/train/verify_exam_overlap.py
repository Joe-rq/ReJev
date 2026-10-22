"""第三口径：客卷与本仓语料的**跨 key 结构**字段值比对（只读核查）。

背景：`prep_exam.py` 的校验 3 与 `docs/plan/005_ood-exam-eval.md` 只做
`statehash` + `id` 两口径。报告 §3.5 曾声称三个口径，第三个
（「跨 key 结构的字段值比对」）此前没有实现；本脚本补上它并留档。

定义（与 key 结构无关的内容袋）：

1. 取记录 `state` 的**全部叶子值**（字符串/数字/布尔；dict／list 下钻）；
2. 每个值按 `src/data/render.py:statehash` 同一套归一化处理
   （NFKC + casefold + `\\w+` 词序列）；
3. 归一化后的值排序、以 `\\x1f` 连接，取 sha256。同时算两种口径：

   - **集合**（去重）：忽略重数。命中集是多重集口径的**超集**——对「零重叠」
     这种否定性主张更保守（只会多报、不会漏报）。
   - **多重集**（Counter 保留重数）：对应「同一内容、只换标识」的原意，更精确。

内容袋哈希对**换 key** 不变（结构无关）；对字段值的改动，只要**归一化后
仍不同**就会变（按口径：词序列或重数）。大小写、标点、空白经归一化后
不可见，对这类改动**不敏感**。

判据：六个语料在**两个口径下都是 0 命中**。`--selftest` 跑七类用例
（换 key / 嵌套换 key / 大小写 / 加字符 / 列表重排 / 重数双口径 /
messages 回退），全部必须通过。

用法：
  uv run python -I src/train/verify_exam_overlap.py
  uv run python -I src/train/verify_exam_overlap.py --selftest
"""

import argparse
import collections
import hashlib
import json
import pathlib
import re
import sys
import unicodedata

REPO = pathlib.Path(__file__).resolve().parents[2]
EXAM = REPO / "data" / "exam" / "exam.jsonl"
CORPORA = (
    ("rejev-train", "data/rejev/records/rejev-train.jsonl"),
    ("rejev-holdout", "data/rejev/records/rejev-holdout.jsonl"),
    ("rejev-dev", "data/rejev/records/rejev-dev.jsonl"),
    ("rejev2-train", "data/rejev2/records/rejev2-train.jsonl"),
    ("rejev2-holdout", "data/rejev2/records/rejev2-holdout.jsonl"),
    ("tev1-paper", "data/paper/tev1-paper.jsonl"),
)

WORD = re.compile(r"\w+")

# 与 prep_exam.py 同一条纪律：`python -O` 会剥掉 assert，把校验门变成假通过。
if not __debug__:  # pragma: no cover
    raise RuntimeError("本脚本用 assert 做校验，禁止以 python -O 运行（会静默剥离校验）")
SEP = "\x1f"
# 强口径：不同叶子值 ≥3 且归一化总长 ≥40，排除短记录与常量字段的偶然碰撞
MIN_VALUES = 3
MIN_CHARS = 40


def norm_value(v) -> str:
    """与 render.statehash 同口径：NFKC + casefold + 词序列。"""
    s = unicodedata.normalize("NFKC", str(v)).casefold()
    return " ".join(WORD.findall(s))


def leaves(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from leaves(v)
    elif obj is not None:
        yield obj


def bag_hashes(state) -> tuple[str, str, int, int]:
    """返回 (集合口径 sha256, 多重集口径 sha256, 不同叶子值数, 归一化后总字符数)。

    - 集合口径忽略重数：`["a","a"]` 与 `["a","a","a"]` **同哈希**。
      它把「只差重数」的内容**误判为同一**（多报），因此是更激进的探测器；
      对「零重叠」这类否定性主张，它的 0 命中是更强的证据。
    - 多重集口径保留重数（Counter）：二者哈希不同，对应「同一内容、只换标识」
      的原意，但命中集是集合口径的子集。
    两个都报，避免用单一口径替结论辩护。
    """
    vals = [n for n in (norm_value(x) for x in leaves(state)) if n]
    h_set = hashlib.sha256(SEP.join(sorted(set(vals))).encode("utf-8")).hexdigest()
    cnt = collections.Counter(vals)
    parts = [v if cnt[v] == 1 else f"{v}{SEP}{cnt[v]}" for v in sorted(cnt)]
    joined = SEP.join(parts)
    h_multi = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return h_set, h_multi, len(cnt), len(joined)


def load(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def state_of(rec):
    """顶层 `state`；否则从 `messages` 里 user 角色的 JSON 题面取。"""
    if "state" in rec:
        return rec["state"]
    for m in rec.get("messages", []):
        if m.get("role") == "user":
            try:
                return json.loads(m["content"]).get("state")
            except (json.JSONDecodeError, AttributeError):
                return None
    return None


def bags(path) -> tuple[dict, dict]:
    """返回 (集合口径袋 → meta, 多重集口径袋 → meta)。"""
    by_set: dict[str, tuple[int, int, str]] = {}
    by_multi: dict[str, tuple[int, int, str]] = {}
    for rec in load(path):
        st = state_of(rec)
        assert st is not None, f"取不到 state：{rec.get('id')}"
        hs, hm, nvals, nchars = bag_hashes(st)
        meta = (nvals, nchars, rec.get("id", "?"))
        by_set.setdefault(hs, meta)
        by_multi.setdefault(hm, meta)
    assert by_set, f"{path}: 内容袋为空"
    return by_set, by_multi


def selftest() -> int:
    sample = next(load(REPO / CORPORA[0][1]))["state"]
    assert isinstance(sample, dict) and sample, "selftest 取样异常"
    k0 = next(iter(sample))
    base_set, base_multi = bag_hashes(sample)[:2]

    # 大小写用例必须真的产生一个可大写化的字符串，否则是空转
    upper = {k: (v.upper() if i == 0 else v) for i, (k, v) in enumerate(sample.items())}
    assert isinstance(sample[k0], str) and sample[k0] != sample[k0].upper(), \
        "selftest 取样首值不是可大写化的字符串，大小写用例会空转"

    cases = [
        ("换顶层 key", {"k%d" % i: v for i, v in enumerate(sample.values())}, True),
        ("嵌套 dict 换 key", {"outer": {"k%d" % i: v for i, v in enumerate(sample.values())}}, True),
        ("纯大小写改动", upper, True),
        ("加一个字符", {**sample, k0: str(sample[k0]) + "x"}, False),
    ]
    for name, obj, expect_same in cases:
        hs, hm = bag_hashes(obj)[:2]
        assert (hs == base_set) == expect_same, f"selftest[{name}] 集合口径：期望 same={expect_same}"
        assert (hm == base_multi) == expect_same, f"selftest[{name}] 多重集口径：期望 same={expect_same}"

    # 列表顺序无关
    assert bag_hashes({"l": ["a", "b"]})[0] == bag_hashes({"l": ["b", "a"]})[0]
    assert bag_hashes({"l": ["a", "b"]})[1] == bag_hashes({"l": ["b", "a"]})[1]
    # 重数：集合口径相同（更激进），多重集口径不同（更精确）——两个方向都锁定
    hs2, hm2 = bag_hashes({"a": "x", "b": "x"})[:2]
    hs3, hm3 = bag_hashes({"a": "x", "b": "x", "c": "x"})[:2]
    assert hs2 == hs3, "集合口径应忽略重数（同哈希）"
    assert hm2 != hm3, "多重集口径应保留重数（不同哈希）"
    # messages 回退路径
    rec = {"id": "t", "messages": [{"role": "user",
           "content": json.dumps({"state": {"z": "hello world"}})}]}
    assert state_of(rec) == {"z": "hello world"}, "messages 回退取 state 失败"
    print("SELFTEST: PASS（换 key / 嵌套换 key / 大小写 / 加字符 / 列表重排 / 重数双口径 / messages 回退）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exam", type=pathlib.Path, default=EXAM)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.exam.exists():
        print(f"缺客卷：{args.exam}（本仓不发布数据，需自备）", file=sys.stderr)
        return 2

    exam_set, exam_multi = bags(args.exam)
    print(f"客卷 {len(exam_set)} 条 → 集合口径 {len(exam_set)} 袋 / "
          f"多重集口径 {len(exam_multi)} 袋")

    report, raw_total, strong_total = {}, 0, 0
    missing = []
    for name, rel in CORPORA:
        path = REPO / rel
        if not path.exists():
            missing.append(name)
            print(f"  {name}: ⚠️ 未检查（{rel} 不在本机）——未检查 ≠ 通过")
            continue
        cs, cm = bags(path)
        hits_set = set(exam_set) & set(cs)
        hits_multi = set(exam_multi) & set(cm)
        # 强口径按集合口径（更激进）过滤
        strong = [h for h in hits_set
                  if exam_set[h][0] >= MIN_VALUES and exam_set[h][1] >= MIN_CHARS]
        raw_total += len(hits_set)
        strong_total += len(strong)
        report[name] = {"bags_set": len(cs), "bags_multi": len(cm),
                        "hits_set": len(hits_set), "hits_multi": len(hits_multi),
                        "strong_hits": len(strong)}
        print(f"  {name}: 集合 {len(cs)} 袋 / 多重集 {len(cm)} 袋 → "
              f"命中 集合 {len(hits_set)} / 多重集 {len(hits_multi)}"
              f"（强口径 {len(strong)}）")
        for h in strong:
            print(f"     HIT exam={exam_set[h][2]} corpus={cs[h][2]}")

    print()
    print(f"合计：集合口径命中 {raw_total}，多重集口径命中 "
          f"{sum(v['hits_multi'] for v in report.values())}，强口径命中 {strong_total}"
          f"（强口径 = 不同叶子值 ≥{MIN_VALUES} 且归一化总长 ≥{MIN_CHARS}）")
    if missing:
        print(f"⚠️ 未检查的语料：{missing}")
    if args.json:
        inputs = {"exam": str(args.exam),
                  "corpora": {n: str(REPO / rel) for n, rel in CORPORA}}
        payload = {"status": "incomplete" if missing else "complete",
                   "exam_bags_set": len(exam_set), "exam_bags_multi": len(exam_multi), "corpora": report,
                   "raw_hits": raw_total, "strong_hits": strong_total,
                   "missing": missing, "inputs": inputs,
                   "data_not_published": True,
                   "note": "输入语料不随本仓发布；本文件不能单独作为可复现证据"}
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        print(f"结果写入 {args.json}")
    # 集合/多重集命中与强口径任一非 0、或有语料未检查，都算未通过（exit 1）
    return 0 if raw_total == 0 and strong_total == 0 and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
