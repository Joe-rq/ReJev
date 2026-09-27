"""ReJev 渲染器单测：合成样例驱动，不使用任何真实训练数据。

跑法：uv run python src/data/test_render.py
（首次需网络拉 tokenizer 至 HF 缓存；之后可离线。）
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("HF_HUB_OFFLINE", "0")

from render import (ASSISTANT_END, LETTERS, MAX_TOKENS, MODEL, REV, SYSTEM,
                    build_messages, cross_split_check, normalized, render_one,
                    render_records, statehash)


def sample(state="Card charged twice on 2026-09-20.", question="What to do first?",
           n_options=4, answer="A", rid="t-001"):
    options = [{"label": LETTERS[min(i, len(LETTERS) - 1)], "key": f"k{i}",
                "description": f"d{i}"} for i in range(n_options)]  # 越界时造重复 label，交由校验拒绝
    return {"id": rid, "source": "synthetic-test", "group_id": f"g-{rid}",
            "state": state, "question": question, "options": options, "answer": answer}


class TestBuild(unittest.TestCase):

    def test_messages_shape(self):
        msgs = build_messages(sample())
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant"])
        self.assertEqual(msgs[0]["content"], SYSTEM)
        self.assertIn('"state"', msgs[1]["content"])  # JSON 渲染
        self.assertEqual(msgs[2]["content"], "A")

    def test_option_bounds(self):
        build_messages(sample(n_options=2))
        build_messages(sample(n_options=24))
        for bad in (1, 25):
            with self.assertRaises(ValueError):
                build_messages(sample(n_options=bad))

    def test_answer_must_be_label(self):
        with self.assertRaises(ValueError):
            build_messages(sample(answer="Z"))
        with self.assertRaises(ValueError):
            build_messages(sample(answer="a"))


class TestNormalized(unittest.TestCase):

    def test_variants_same_hash(self):
        a = statehash("Return window is 30 days! Purchase was 12 days ago.")
        b = statehash("return   WINDOW is 30 days… purchase WAS 12 days ago")
        self.assertEqual(a, b)

    def test_different_state_different_hash(self):
        self.assertNotEqual(statehash("state one"), statehash("state two"))

    def test_dict_state_stable(self):
        s = {"k": [1, 2], "z": True}
        self.assertEqual(statehash(s), statehash({"z": True, "k": [1, 2]}))


@unittest.skipUnless(os.environ.get("RENDER_SKIP_TOK") is None, "跳过需 tokenizer 的用例")
class TestRender(unittest.TestCase):
    """需加载 MiniCPM tokenizer（HF 缓存命中则离线可跑）。"""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)

    def test_prompt_tail_and_completion(self):
        item = render_one(sample(), self.tok)
        self.assertTrue(item["instruction"]["prompt"].endswith(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertIn(SYSTEM, item["instruction"]["prompt"])
        self.assertEqual(item["instruction"]["completion"], "A" + ASSISTANT_END)

    def test_letter_single_token_all_24(self):
        for letter in LETTERS:
            self.assertEqual(len(self.tok.encode(letter, add_special_tokens=False)), 1)

    def test_token_count_consistent(self):
        item = render_one(sample(n_options=24), self.tok)
        recomputed = len(self.tok.encode(item["instruction"]["prompt"]
                                         + item["instruction"]["completion"],
                                         add_special_tokens=False))
        self.assertEqual(item["token_count"], recomputed)
        self.assertLessEqual(item["token_count"], MAX_TOKENS)

    def test_overlong_rejected(self):
        huge = sample(state="word " * 2000)  # 远超 2048 token
        with self.assertRaises(ValueError):
            render_one(huge, self.tok)

    def test_stats_and_cross_split(self):
        train = [sample(rid=f"tr-{i}", state=f"train state {i}") for i in range(3)]
        heldout = [sample(rid=f"ho-{i}", state=f"holdout state {i}") for i in range(2)]
        items, stats = render_records(train + heldout, self.tok)
        self.assertEqual(stats["n"], 5)
        self.assertEqual(stats["assistant_end"], ASSISTANT_END)
        check = cross_split_check(items[:3], items[3:])
        self.assertEqual(check["cross_split_state_overlap"], 0)

    def test_cross_split_leak_detected(self):
        items, _ = render_records(
            [sample(rid="a", state="same state"), sample(rid="b", state="other")], self.tok)
        with self.assertRaises(ValueError):
            cross_split_check(items, items[:1])  # 同 state 必须被逮住


if __name__ == "__main__":
    unittest.main(verbosity=2)
