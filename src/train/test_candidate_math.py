"""S1 候选评分/置换规则层单测（不加载 torch / modal）。

跑法：uv run python src/train/test_candidate_math.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import candidate_math as cm  # noqa: E402


def rec(n=3, answer="B"):
    opts = [{"label": cm.LETTERS[i], "key": k, "description": f"d{i}"}
            for i, k in enumerate(["k1", "k2", "k3", "k4"][:n])]
    return {"id": "policy_v2:test2:1:x", "state": "s", "question": "q?",
            "options": opts, "answer": answer, "answer_key": opts[cm.LETTERS.index(answer)]["key"]}


class ValidateTest(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(cm.validate_options(rec()["options"]), ["A", "B", "C"])

    def test_rejects_non_prefix_labels(self):
        r = rec()
        r["options"][0]["label"] = "B"
        with self.assertRaises(ValueError):
            cm.validate_options(r["options"])

    def test_rejects_duplicate_keys(self):
        r = rec()
        r["options"][1]["key"] = r["options"][0]["key"]
        with self.assertRaises(ValueError):
            cm.validate_options(r["options"])

    def test_rejects_too_few(self):
        with self.assertRaises(ValueError):
            cm.validate_options([{"label": "A", "key": "x", "description": "x"}])


class ShiftTest(unittest.TestCase):
    def test_range_and_determinism(self):
        for n in (2, 3, 5, 24):
            s = cm.shift_for("abc", n)
            self.assertEqual(s, cm.shift_for("abc", n))
            self.assertTrue(1 <= s <= n - 1)

    def test_two_options_always_one(self):
        for rid in ("a", "b", "policy_v2:test2:1:x"):
            self.assertEqual(cm.shift_for(rid, 2), 1)


class PermuteTest(unittest.TestCase):
    def test_semantics_preserved(self):
        r = rec()
        p = cm.permute_record(r)
        self.assertEqual(r["answer"], "B")  # 原记录不变
        self.assertEqual(p["id"], r["id"])
        self.assertEqual(p["state"], r["state"])
        self.assertEqual(p["question"], r["question"])
        self.assertEqual([o["key"] for o in p["options"]],
                         p["permutation"]["permuted_order"])
        self.assertEqual(set(o["key"] for o in p["options"]),
                         set(o["key"] for o in r["options"]))
        self.assertEqual({o["key"]: o["description"] for o in p["options"]},
                         {o["key"]: o["description"] for o in r["options"]})
        # answer_key 的语义项必须落在新的 answer 字母上
        self.assertEqual(p["options"][cm.LETTERS.index(p["answer"])]["key"], r["answer_key"])
        self.assertEqual([o["label"] for o in p["options"]], ["A", "B", "C"])

    def test_explicit_shift(self):
        r = rec()
        p = cm.permute_record(r, shift=2)
        self.assertEqual([o["key"] for o in p["options"]], ["k3", "k1", "k2"])
        self.assertEqual(p["answer"], "C")  # k2 移到 C

    def test_invalid_shift(self):
        r = rec()
        for s in (0, len(r["options"])):
            with self.assertRaises(ValueError):
                cm.permute_record(r, shift=s)

    def test_real_sample(self):
        path = REPO / "data/eval2-final-20261017/records/test2.jsonl"
        if not path.exists():
            self.skipTest("本地 eval2 数据不存在（gitignored）")
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= 100:
                    break
                r = json.loads(line)
                p = cm.permute_record(r)
                self.assertEqual(p["answer_key"], r["answer_key"])
                self.assertEqual(p["options"][cm.LETTERS.index(p["answer"])]["key"],
                                 r["answer_key"])
                self.assertEqual([o["key"] for o in p["options"]],
                                 p["permutation"]["permuted_order"])
                self.assertNotEqual(p["permutation"]["shift"], 0)


class ProbTest(unittest.TestCase):
    def test_softmax(self):
        p = cm.softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(p), 1.0, places=12)
        self.assertEqual(cm.argmax(p), 2)
        self.assertAlmostEqual(p[2], 0.6652409557748219, places=12)

    def test_semantic_scores(self):
        r = rec()
        s = cm.semantic_scores({"A": 0.1, "B": 0.2, "C": 0.7}, r["options"])
        self.assertEqual(s, {"k1": 0.1, "k2": 0.2, "k3": 0.7})


if __name__ == "__main__":
    unittest.main(verbosity=2)
