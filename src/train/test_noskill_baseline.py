"""noskill_baseline 的单测：纯本地合成样本，不读真实留出集、不联网。

跑法：uv run python src/train/test_noskill_baseline.py

**为什么必须有这个文件**（2026-09-28 第四轮双谱系评审 B 线 P1）：技术报告 §2.1/§2.4
引用的五个无技能基线，此前全仓只有报告自己出现过——第三轮评审要求给它们一个可复算的
来源，于是新增了 `noskill_baseline.py` + 报告点名引用 + 一句「不带参数跑一次即可复现」。
**但那句话当时没有任何测试或执行记录支撑**，而脚本读的字段名 `answer_key`
在源码树里**没有任何生产者**（跑真实数据时会 KeyError；本仓数据里恰好有该字段，
所以只在本机跑一次「看起来没问题」）。这正是本仓反复出现的形态：**一个新的「假 ✅」**。
单侧验证（只跑真实数据、不看口径）读起来像「已覆盖」，实际把两个回归放了进去：
① 字段名依赖；② 四个基线的**口径**没被任何独立样本钉住。
"""

from __future__ import annotations

import unittest

from noskill_baseline import baselines, key_of


def rec(answer: str, n_options: int, key: str) -> dict:
    """一条 records 曲线：选项 label 是 A 起的连续字母，`key` 挂在选项上。

    ⚠️ 刻意**不**产出 record 级的 `answer_key` 字段——那是本轮要修的形态：文档化的
    records 口径只有 `state/question/options/answer`，第三方按文档重建的数据里没有它。
    """
    labels = "ABCDEFGHIJKLMNOPQRSTUVWX"[:n_options]
    return {"answer": answer,
            "options": [{"label": lb, "key": key if lb == answer else f"other-{lb}"}
                        for lb in labels]}


# 7 条，逐条可手算（分母 7）。四个期望值**两两不同**——这是 2026-09-29 第五轮评审
# B 线 P3 的要求：上一版 7 条样本里「逐题随机」与「恒定语义 key」都等于 3/7，于是
# 把这两个口径的实现互换、或让其中一个错误地复用另一个的结果，四条断言仍然全绿。
# 现在随机的分母是 3×½ + 4×⅓ = 2.8333，与 key 众数的 3 不再撞车。
#   字母：A×4 / B×3                → 众数 A → 4/7          ≈ 0.571429
#   语义 key：k1×2 / k2×2 / k3×3   → 众数 k3 → 3/7         ≈ 0.428571
#   逐题随机：3 条 2 选项 + 4 条 3 选项 → (3×½ + 4×⅓)/7    ≈ 0.404762
#   按选项数取位置：2 选项组众数位置 0（答 A，中 3 条）；
#                   3 选项组众数位置 1（答 B，中 3 条）→ 6/7  ≈ 0.857143
ROWS = [
    rec("A", 2, "k1"), rec("A", 2, "k1"), rec("A", 2, "k2"),
    rec("A", 3, "k2"), rec("B", 3, "k3"), rec("B", 3, "k3"), rec("B", 3, "k3"),
]


class TestKeyOf(unittest.TestCase):
    """`key_of` 从 options 反查语义 key（第四轮评审 B 线 P1 的正面修法）。"""

    def test_reverse_lookup_finds_the_gold_key(self) -> None:
        self.assertEqual(key_of(ROWS[0]), "k1")
        self.assertEqual(key_of(ROWS[4]), "k3")

    def test_does_not_depend_on_record_level_answer_key(self) -> None:
        """**回归钉**：输入里没有 `answer_key` 字段时必须照常工作。
        旧实现读 `r["answer_key"]`，在按文档重建的数据上会 KeyError——
        「跑不起来的数字来源比没有来源更糟」。"""
        self.assertNotIn("answer_key", ROWS[0])
        self.assertEqual(key_of(ROWS[0]), "k1")

    def test_missing_gold_label_yields_none(self) -> None:
        """gold 不在选项里时返回 None（而非抛错）——该题会被计入分母但不算命中。"""
        bad = {"answer": "Z", "options": [{"label": "A", "key": "k"}]}
        self.assertIsNone(key_of(bad))


class TestBaselines(unittest.TestCase):
    """四个基线的**口径**：每个数都由上表手算而来，不是「跑一遍看看」。"""

    def setUp(self) -> None:
        self.b = baselines(ROWS)

    def test_per_item_random(self) -> None:
        """3 条 2 选项 + 4 条 3 选项 → (3×½ + 4×⅓)/7。"""
        self.assertAlmostEqual(self.b["random_mean_1_over_n"], (3 / 2 + 4 / 3) / 7, places=6)

    def test_constant_most_frequent_letter(self) -> None:
        self.assertAlmostEqual(self.b["constant_most_frequent_letter"], 4 / 7, places=6)

    def test_constant_most_frequent_key(self) -> None:
        """key 众数与字母众数**不同**（3/7 vs 4/7）——两者算错成同一个数的形态会被这条抓住。"""
        self.assertAlmostEqual(self.b["constant_most_frequent_key"], 3 / 7, places=6)

    def test_per_option_count_most_frequent_position(self) -> None:
        self.assertAlmostEqual(
            self.b["per_option_count_most_frequent_position"], 6 / 7, places=6)

    def test_random_and_key_baselines_are_distinguishable(self) -> None:
        """★ 第五轮评审 B 线 P3：这两个口径的期望值必须**不同**。

        上一版 7 条样本让两者都等于 3/7，于是「把 `random_mean_1_over_n` 误写成 key 基线」
        （或反之）这种改动**四条断言全绿**——测试看着钉住了口径，实际钉不住。
        """
        self.assertNotAlmostEqual(self.b["random_mean_1_over_n"],
                                  self.b["constant_most_frequent_key"], places=6)

    def test_detail_echoes_the_winning_values(self) -> None:
        """产物里的 `detail` 是给人核对口径用的，必须与算出来的众数一致。"""
        self.assertEqual(self.b["detail"],
                         {"letter": "A", "key": "k3", "key_unresolved": 0, "n_options": 2})

    def test_unresolvable_gold_is_not_counted_as_a_key(self) -> None:
        """★ 第五轮评审 B 线 P3：`key_of` 返回 `None` 时，`None` **不得**作为众数参与统计。

        上一版的 `Counter(key_of(r) for r in rows)` 会把 `None` 当成一个正常取值——若这类题
        占多数，口径 3 会报出一个「以 None 为众数」的百分比，读起来像「最常见语义 key 的
        基线」，实际什么 key 都不是。这里用两条好题 ＋ 一条 gold 落空题验证：
        众数仍是 k1（2/3），落空的那条**未参与**统计、但计入分母。
        """
        rows = [rec("A", 2, "k1"), rec("A", 2, "k1"),
                {"answer": "Z", "options": [{"label": "A", "key": "kx"}]}]
        b = baselines(rows)
        self.assertEqual(b["detail"]["key"], "k1")
        self.assertEqual(b["detail"]["key_unresolved"], 1)
        self.assertAlmostEqual(b["constant_most_frequent_key"], 2 / 3, places=6)

    def test_all_gold_unresolvable_reports_unjudgeable(self) -> None:
        """一条 key 都解析不出时输出 `None`（**不可判读**），而不是一个以 None 为众数的假数。"""
        rows = [{"answer": "Z", "options": [{"label": "A", "key": "kx"}]}]
        b = baselines(rows)
        self.assertIsNone(b["constant_most_frequent_key"])
        self.assertIsNone(b["detail"]["key"])
        self.assertEqual(b["detail"]["key_unresolved"], 1)

    def test_baselines_never_exceed_one(self) -> None:
        """四口径都是「正确数 / 总数」，越界即口径写错（例如漏了分母）。"""
        self.assertEqual(self.b["n"], len(ROWS))
        for k, v in self.b.items():
            if k not in ("detail", "n"):
                self.assertIsNotNone(v, f"{k} 不该在这份样本上不可判读")
                self.assertLessEqual(v, 1.0, k)

    def test_no_duplicate_gold_letter_in_a_row(self) -> None:
        """选项 label 必须唯一，否则 `labels.index(answer)` 取到的位置有歧义。"""
        for r in ROWS:
            labels = [o["label"] for o in r["options"]]
            self.assertEqual(len(labels), len(set(labels)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
