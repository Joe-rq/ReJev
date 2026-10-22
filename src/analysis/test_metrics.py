"""S1 分析口径的规则层单测（合成数据，不读真实样本）。

跑法：uv run python src/analysis/test_metrics.py
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as M  # noqa: E402


def rec(probs, gold_label, labels=("A", "B", "C"), pred_key=None, gold_key=None):
    return {"option_labels": list(labels),
            "probs_by_label": dict(zip(labels, probs)),
            "logits_by_label": {lab: math.log(max(p, 1e-300))
                                for lab, p in zip(labels, probs)},
            "gold_label": gold_label,
            "pred_label": labels[int(np.argmax(probs))],
            "pred_key": pred_key if pred_key is not None else labels[int(np.argmax(probs))],
            "gold_key": gold_key if gold_key is not None else gold_label}


class BasicMetricTest(unittest.TestCase):
    def setUp(self):
        self.records = [rec([0.7, 0.2, 0.1], "A"),
                        rec([0.1, 0.8, 0.1], "B"),
                        rec([0.2, 0.3, 0.5], "C")]
        self.probs, self.labels, self.gold = M._probs(self.records)

    def test_nll(self):
        want = -np.mean(np.log([0.7, 0.8, 0.5]))
        self.assertAlmostEqual(M.nll(self.probs, self.gold), want, places=12)

    def test_brier(self):
        # 第一题：0.3²+0.2²+0.1² = 0.14；第二题：0.1²+0.2²+0.1² = 0.06；
        # 第三题：0.2²+0.3²+0.5² = 0.38；平均 0.193333...
        self.assertAlmostEqual(M.brier(self.probs, self.gold), (0.14 + 0.06 + 0.38) / 3,
                               places=12)

    def test_ece(self):
        records = [rec([0.90, 0.05, 0.05], "A"),   # bin [0.5,1]，correct
                   rec([0.10, 0.80, 0.10], "C"),   # bin [0.5,1]，wrong（pred B）
                   rec([0.45, 0.45, 0.10], "C")]   # bin [0,0.5)，wrong（pred A）
        probs, _, gold = M._probs(records)
        # bin0: n=1 acc=0 conf=.45 gap=.45，权重 1/3
        # bin1: n=2 acc=.5 conf=.85 gap=.35，权重 2/3
        want = (1 / 3) * 0.45 + (2 / 3) * 0.35
        self.assertAlmostEqual(M.ece_top_label(probs, gold, n_bins=2), want, places=12)

    def test_selective(self):
        records = [rec([0.70, 0.20, 0.10], "A"),   # correct
                   rec([0.10, 0.80, 0.10], "B"),   # correct
                   rec([0.20, 0.30, 0.50], "A")]   # wrong（pred C）
        probs, _, gold = M._probs(records)
        curve = M.selective_curve(probs, gold, coverages=(0.34, 1.0))
        self.assertEqual(curve[0]["n"], 2)
        self.assertAlmostEqual(curve[0]["accuracy"], 1.0)
        self.assertAlmostEqual(curve[-1]["accuracy"], 2 / 3, places=5)

    def test_mismatched_labels_rejected(self):
        with self.assertRaises(ValueError):
            M._probs([rec([0.5, 0.5], "A", labels=("A", "B")),
                      rec([0.5, 0.5, 0.0], "A")])


class TemperatureTest(unittest.TestCase):
    def test_recover_temperature(self):
        rng = np.random.default_rng(7)
        logits = rng.normal(size=(2000, 4)) * 2.0
        true_t = 2.5
        z_true = np.exp(logits / true_t)
        p_true = z_true / z_true.sum(axis=1, keepdims=True)
        # 金标从真实（已校准）分布抽样；输入给拟合器的是未校准的 T=1 概率。
        gold = np.array([rng.choice(len(p), p=p) for p in p_true])
        z_uncal = np.exp(logits)
        p_uncal = z_uncal / z_uncal.sum(axis=1, keepdims=True)
        fit = M.fit_temperature(p_uncal, gold, n_grid=801)
        self.assertFalse(fit["at_boundary"])
        self.assertLess(abs(fit["T"] - true_t), 0.5)

    def test_temperature_preserves_argmax(self):
        probs = np.array([[0.2, 0.5, 0.3]])
        for T in (0.05, 1.0, 20.0):
            self.assertEqual(M.temperature_scale(probs, T).argmax(), probs.argmax())

    def test_t1_in_grid(self):
        probs = np.array([[0.5, 0.5]])
        fit = M.fit_temperature(probs, np.array([0]), n_grid=10)
        self.assertTrue(math.isfinite(fit["T"]))

    def test_extreme_logits_no_clipping(self):
        # logits [0,-50], T=20：正确第二类概率应为 0.075858，不是裁剪后的 0.200760
        r = {"option_labels": ["A", "B"],
             "logits_by_label": {"A": 0.0, "B": -50.0},
             "gold_label": "B"}
        p = M.apply_temperature(r, 20.0)
        self.assertAlmostEqual(p["B"], 0.07585818, places=6)
        self.assertAlmostEqual(M.record_items([r], 20.0)["nll"][0], 2.57889, places=4)
        self.assertAlmostEqual(M.record_items([r], 1.0)["nll"][0], 50.0, places=4)


class RecordMetricTest(unittest.TestCase):
    def test_summarize_matches_matrix(self):
        records = [rec([0.7, 0.2, 0.1], "A"),
                   rec([0.1, 0.8, 0.1], "B"),
                   rec([0.2, 0.3, 0.5], "C")]
        probs, labels, gold = M._probs(records)
        s = M.summarize_records(records)
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["nll"], M.nll(probs, gold), places=12)
        self.assertAlmostEqual(s["brier"], M.brier(probs, gold), places=12)
        self.assertAlmostEqual(s["ece_top_label"],
                               M.ece_top_label(probs, gold), places=12)

    def test_fit_temperature_records_mixed_k(self):
        rng = np.random.default_rng(11)
        true_t = 2.0
        records = []
        for i in range(1200):
            k = 2 if i % 2 == 0 else 5
            logits = rng.normal(size=k) * 2.0
            zt = np.exp(logits / true_t)
            pt = zt / zt.sum()
            gold_i = int(rng.choice(k, p=pt))
            zu = np.exp(logits)
            pu = zu / zu.sum()
            labels = list("ABCDEFGHIJKLMNOPQRSTUVWX"[:k])
            records.append({"option_labels": labels,
                            "probs_by_label": dict(zip(labels, map(float, pu))),
                            "gold_label": labels[gold_i]})
        fit = M.fit_temperature_records(records, n_grid=601)
        self.assertFalse(fit["at_boundary"])
        self.assertLess(abs(fit["T"] - true_t), 0.4)
        self.assertEqual(fit["n_records"], 1200)


class PermutationLabelTest(unittest.TestCase):
    def test_exact_20_percent_is_high(self):
        # 5 题 1 翻转：0.2 浮点下旧实现会落到 uncertain
        self.assertEqual(M.permutation_label(5, 1, 0.8), "high_sensitivity")
        self.assertEqual(M.permutation_label(3800, 760, 0.2), "high_sensitivity")

    def test_low_and_uncertain(self):
        self.assertEqual(M.permutation_label(100, 5, 0.90), "low_sensitivity")
        self.assertEqual(M.permutation_label(100, 6, 0.90), "uncertain")
        self.assertEqual(M.permutation_label(100, 2, 0.89), "uncertain")


class F1Test(unittest.TestCase):
    def test_missing_class_f1_zero_and_recall_na(self):
        # 固定类别 A/B/C；C 从未作为 gold 出现
        records = [
            rec([0.8, 0.1, 0.1], "A", gold_key="a", pred_key="a"),
            rec([0.1, 0.8, 0.1], "B", gold_key="b", pred_key="b"),
            rec([0.7, 0.2, 0.1], "B", gold_key="b", pred_key="a"),
        ]
        f1 = M.macro_f1(records, ["a", "b", "c"])
        # a: tp1 fp1 fn0 -> 2/3; b: tp1 fp0 fn1 -> 2/3; c: 0
        self.assertAlmostEqual(f1, (2 / 3 + 2 / 3 + 0) / 3, places=12)
        r = M.per_class_recall(records, ["a", "b", "c"])
        self.assertAlmostEqual(r["a"], 1.0)
        self.assertAlmostEqual(r["b"], 0.5)
        self.assertIsNone(r["c"])
        self.assertEqual(r["n_defined"], 2)
        self.assertAlmostEqual(r["macro_recall_defined"], 0.75)


class BootstrapTest(unittest.TestCase):
    def test_deterministic_and_ci(self):
        values = np.array([1, 1, 0, 1, 0, 0, 1, 1])
        groups = ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"]
        a = M.cluster_bootstrap_mean(values, groups, n_boot=200, seed=1)
        b = M.cluster_bootstrap_mean(values, groups, n_boot=200, seed=1)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a["point"], 0.625)
        self.assertLessEqual(a["ci95"][0], a["point"])
        self.assertGreaterEqual(a["ci95"][1], a["point"])

    def test_paired_diff(self):
        base = np.array([1, 0, 1, 0, 1, 0])
        arm = np.array([1, 1, 1, 0, 0, 1])
        groups = ["a", "a", "b", "b", "c", "c"]
        d = M.cluster_bootstrap_paired_diff(arm, base, groups, n_boot=200, seed=2)
        self.assertAlmostEqual(d["point"], 1 / 6)
        self.assertEqual(d["n_groups"], 3)

    def test_degenerate_all_correct(self):
        values = np.ones(6)
        groups = ["a", "a", "b", "b", "c", "c"]
        d = M.cluster_bootstrap_mean(values, groups, n_boot=100, seed=3)
        self.assertEqual(d["ci95"], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
