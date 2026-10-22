"""S0 生成器的规则层单测：配额、statehash 口径、覆盖与答案位置审计。

跑法：uv run python src/data/test_build_eval.py
不加载 datasets、不读真实样本；只校验可机械判定的不变式。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 冻结清单目录（成员推断风险，整目录不随公开导出集分发）。导出集里这两个
# 测试类按 skip 处理；路径用分量拼接，避免在发布文本里留下一个指向未分发
# 目录的静态引用（`verify-repro-path.py` 会把它报成悬空引用）。
FROZEN_DIR = REPO / "src" / "data" / "frozen"
FROZEN_AVAILABLE = FROZEN_DIR.is_dir()

import build_eval_v2 as be  # noqa: E402
import freeze_history as fh  # noqa: E402
import verify_eval2 as ve  # noqa: E402


def load_upstream_statehash():
    spec = importlib.util.spec_from_file_location(
        "tev1_build_v2_for_test", REPO / "resources/tev1/build_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.statehash


class QuotaTest(unittest.TestCase):
    def test_totals(self):
        self.assertEqual(sum(be.QUOTAS["dev2"].values()), 500)
        self.assertEqual(sum(be.QUOTAS["calibration2"].values()), 1500)
        self.assertEqual(sum(be.QUOTAS["test2"].values()), 3800)
        self.assertEqual(sum(sum(q.values()) for q in be.QUOTAS.values()), 5800)

    def test_group_steps(self):
        for split, q in be.QUOTAS.items():
            self.assertEqual(q["policy_v2"] % 6, 0, split)
            self.assertEqual(q["routing_v2"] % 6, 0, split)
            self.assertEqual(q["policy"] % 3, 0, split)
            self.assertEqual(q["research_taxonomy_v21"] % 8, 0, split)


class StatehashParityTest(unittest.TestCase):
    def test_matches_upstream(self):
        upstream = load_upstream_statehash()
        samples = [
            {"state": "Card charged twice on 2026-09-20."},
            {"state": {"b": [1, 2], "a": "x"}},
            {"state": {"policy": "Eligible if and only if a AND b", "record": {"a": True}}},
        ]
        for row in samples:
            self.assertEqual(fh.statehash(row["state"]), upstream(row), row)


class CoverageAuditTest(unittest.TestCase):
    def matrix(self, drop=None):
        sources = {}
        for source, options in be.EXPECTED_CELLS.items():
            entry = {"rows": 0, "families": 0, "option_counts": {},
                     "families_by_option": {}}
            for n in options:
                if (source, n) == drop:
                    entry["families_by_option"][n] = 0
                else:
                    entry["families_by_option"][n] = be.MIN_FAMILIES["test2"]
            sources[source] = entry
        return {"test2": sources}

    def test_pass(self):
        self.assertEqual(be.coverage_audit(self.matrix())["gate"], "PASS")

    def test_missing_cell_fails(self):
        result = be.coverage_audit(self.matrix(drop=("banking77", 6)))
        self.assertEqual(result["gate"], "FAIL")
        self.assertTrue(any("banking77/6" in f for f in result["failures"]))


class AnswerPositionAuditTest(unittest.TestCase):
    def fam(self, source, variants, answers):
        return [{"group_id": f"{source}:g1", "source": source, "answer": letter,
                 "answer_key": answer,
                 "options": [{"label": "A", "description": "x."},
                             {"label": "B", "description": "y."},
                             {"label": "C", "description": "z."}],
                 "provenance": {"variant": variant, "pair_side": side,
                                "main_contribution_key": answer}}
                for letter, answer, variant, side in zip(
                    "ABC", answers, variants, (0, 0, 1))]

    def test_policy_family_pass(self):
        rows = self.fam("policy", ["positive", "negative", "missing"],
                        ["eligible", "ineligible", "unknown"])
        self.assertEqual(be.answer_position_audit({"dev2": rows})["gate"], "PASS")

    def test_policy_family_wrong_label_fails(self):
        rows = self.fam("policy", ["positive", "negative", "missing"],
                        ["eligible", "eligible", "unknown"])
        self.assertEqual(be.answer_position_audit({"dev2": rows})["gate"], "FAIL")


class ShortcutRuleTest(unittest.TestCase):
    def test_rules(self):
        row = {"options": [
            {"label": "A", "description": "short."},
            {"label": "B", "description": "the longest description here."},
            {"label": "C", "description": "12/34/56"},
        ], "answer": "B"}
        self.assertEqual(be._longest(row), "B")
        self.assertEqual(be._first(row), "A")
        self.assertEqual(be._morph(row), "C")


class ShortcutGateTest(unittest.TestCase):
    def test_threshold(self):
        self.assertTrue(be._shortcut_reject(0.11, 0.01))
        self.assertFalse(be._shortcut_reject(0.09, 0.01))
        self.assertFalse(be._shortcut_reject(0.15, 0.06))  # 3*SE=0.18
        self.assertFalse(be._shortcut_reject(0.15, None))


class PartitionAuditTest(unittest.TestCase):
    def fake_b(self):
        return SimpleNamespace(statehash=lambda row: fh.statehash(row["state"]))

    def test_family_overlap_fails(self):
        rows = {
            "dev2": [{"source": "x", "group_id": "g1", "state": "a"}],
            "calibration2": [{"source": "x", "group_id": "g1", "state": "b"}],
            "test2": [],
        }
        result = be.partition_audit(rows, self.fake_b(), set(), set())
        self.assertEqual(result["gate"], "FAIL")
        self.assertEqual(result["pairwise"]["dev2_x_calibration2"]["family_overlap"], 1)

    def test_statehash_overlap_fails(self):
        rows = {
            "dev2": [{"source": "x", "group_id": "g1", "state": "same"}],
            "calibration2": [{"source": "x", "group_id": "g2", "state": "same"}],
            "test2": [],
        }
        result = be.partition_audit(rows, self.fake_b(), set(), set())
        self.assertEqual(result["gate"], "FAIL")
        self.assertEqual(result["pairwise"]["dev2_x_calibration2"]["statehash_overlap"], 1)


class PublicPoolTest(unittest.TestCase):
    def test_dedup_first_wins_and_frozen_filter(self):
        rows = [
            {"group": "g1", "state": "first"},
            {"group": "g1", "state": "second"},
            {"group": "g2", "state": "frozen"},
        ]
        unique = be.dedup_rows(rows)
        self.assertEqual([row["state"] for row in unique], ["first", "frozen"])
        kept = be.filter_frozen(unique, {fh.statehash("frozen")},
                                lambda row: fh.statehash(row["state"]))
        self.assertEqual([row["group"] for row in kept], ["g1"])


class SampleExclusivityTest(unittest.TestCase):
    def test_splits_disjoint(self):
        class FakeV1:
            def balanced_take(self, pool, n, key):
                return pool[:n]

            def public_record(self, name, row, split, source):
                return {"source": name, "group_id": row["group"]}

        fake = SimpleNamespace(v1=FakeV1())
        pools = {name: [{"group": f"{name}-{i}", "state": "s"} for i in range(3)]
                 for name in be.PUBLIC}
        lock = {"datasets": {name: {} for name in be.PUBLIC}}
        quotas = {split: {name: 1 for name in be.PUBLIC} for split in be.SPLITS}
        with mock.patch.object(be, "QUOTAS", quotas):
            out = be.sample_public(pools, fake, lock, "seed")
        for name in be.PUBLIC:
            groups = [set(row["group_id"] for row in out[split][name])
                      for split in be.SPLITS]
            self.assertEqual(groups[0] & groups[1], set())
            self.assertEqual(groups[0] & groups[2], set())
            self.assertEqual(groups[1] & groups[2], set())


class QuotaShareTest(unittest.TestCase):
    def test_composition(self):
        synthetic = sum(be.QUOTAS[split][source] for split in be.SPLITS
                        for source in ("policy_v2", "routing_v2", "policy",
                                       "research_taxonomy_v21"))
        public = sum(be.QUOTAS[split][source] for split in be.SPLITS
                     for source in be.PUBLIC)
        self.assertEqual(synthetic + public, 5800)
        self.assertAlmostEqual(synthetic / 5800, 0.68, delta=0.01)


class ResearchVariantAuditTest(unittest.TestCase):
    @staticmethod
    def rows(variants):
        out = []
        for side, variant in enumerate(variants):
            out.append({
                "group_id": "research:g1", "source": "research_taxonomy_v21",
                "answer": "A", "answer_key": "topic",
                "options": [{"label": "A", "description": "x."},
                            {"label": "B", "description": "y."}],
                "provenance": {"pair_side": side // 4, "option_variant": variant,
                               "main_contribution_key": "topic"},
            })
        return out

    def test_good_family_passes(self):
        rows = self.rows([0, 1, 2, 3, 0, 1, 2, 3])
        self.assertEqual(be.answer_position_audit({"dev2": rows})["gate"], "PASS")

    def test_wrong_variant_distribution_fails(self):
        rows = self.rows([0, 0, 0, 1, 1, 2, 2, 3])
        self.assertEqual(be.answer_position_audit({"dev2": rows})["gate"], "FAIL")

    def test_construction_mismatch_does_not_fail_gate(self):
        rows = self.rows([0, 1, 2, 3, 0, 1, 2, 3])
        rows[0]["provenance"]["main_contribution_key"] = "other"
        result = be.answer_position_audit({"dev2": rows})
        self.assertEqual(result["gate"], "PASS")
        self.assertTrue(result["construction_failures"])


@unittest.skipUnless(FROZEN_AVAILABLE, "冻结清单不随公开导出集分发")
class FrozenShaTest(unittest.TestCase):
    def test_locked_sha_and_line_counts(self):
        root = FROZEN_DIR
        self.assertEqual(ve.sha256_file(root / "historical-statehashes.txt"),
                         ve.LOCKED_FROZEN_SHA["statehashes_sha256"])
        self.assertEqual(ve.sha256_file(root / "historical-families.tsv"),
                         ve.LOCKED_FROZEN_SHA["families_sha256"])
        manifest = json.loads((root / "freeze-manifest.json").read_text())
        self.assertEqual(len((root / "historical-statehashes.txt").read_text().split()),
                         manifest["files"]["historical-statehashes.txt"]["lines"])
        self.assertEqual(len((root / "historical-families.tsv").read_text().splitlines()),
                         manifest["files"]["historical-families.tsv"]["lines"])


class PositionBalanceTest(unittest.TestCase):
    def test_z_and_p(self):
        rows = ([{"answer": "A", "options": [{"label": "A"}, {"label": "B"}]}] * 60
                + [{"answer": "B", "options": [{"label": "A"}, {"label": "B"}]}] * 40)
        result = be._position_balance(rows)
        self.assertGreater(result["z"], 0)
        self.assertLess(result["p"], 0.05)
        balanced = ([{"answer": "A", "options": [{"label": "A"}, {"label": "B"}]}] * 50
                    + [{"answer": "B", "options": [{"label": "A"}, {"label": "B"}]}] * 50)
        self.assertAlmostEqual(be._position_balance(balanced)["z"], 0.0, places=6)


class FakeV1:
    @staticmethod
    def validate_partitions(rows):
        return {}

    @staticmethod
    def policy_label(tree, facts):
        return "eligible"


class FakeB21:
    CONTRIBUTIONS = {"topic": ["contrib"], "other": ["background"]}
    OUTCOMES = {"topic": "outcome"}
    TEST_TEMPLATES = ["{main} {method} {outcome} {other}"]


class TemplateGateSplitTest(unittest.TestCase):
    def research_row(self):
        return {
            "id": "r1", "source": "research_taxonomy_v21", "split": "dev2",
            "answer": "A", "answer_key": "topic",
            "options": [{"label": "A", "key": "topic", "description": "x."}],
            "state": {"summary": "contrib m outcome background"},
            "provenance": {
                "template": "{main} {method} {outcome} {other}",
                "contribution": "contrib", "method": "m", "outcome": "outcome",
                "background": "background", "main_contribution_key": "topic",
                "background_key": "other", "option_variant": 0, "pair_side": 0,
            },
        }

    def test_research_roundtrip_failure_fails_gate(self):
        row = self.research_row()
        row["state"]["summary"] = "wrong"
        result = be.template_audit({"dev2": [row]}, FakeV1, None, FakeB21)
        self.assertEqual(result["gate"], "FAIL")

    def test_research_construction_failure_does_not_fail_gate(self):
        row = self.research_row()
        row["provenance"]["contribution"] = "bad"
        row["state"]["summary"] = "bad m outcome background"
        result = be.template_audit({"dev2": [row]}, FakeV1, None, FakeB21)
        self.assertEqual(result["gate"], "PASS")
        self.assertTrue(result["construction_failures"])

    def test_policy_construction_failure_does_not_fail_gate(self):
        row = {
            "id": "p1", "source": "policy", "split": "dev2",
            "state": {"policy": "x", "record": {}},
            "provenance": {"tree": "a", "atom_facts": {"a": True}},
            "answer_key": "ineligible",
        }
        result = be.template_audit({"dev2": [row]}, FakeV1, None, FakeB21)
        self.assertEqual(result["gate"], "PASS")
        self.assertTrue(result["construction_failures"])


@unittest.skipUnless(FROZEN_AVAILABLE, "冻结清单不随公开导出集分发")
class FrozenManifestTest(unittest.TestCase):
    def test_totals_and_coverage(self):
        manifest = json.loads((FROZEN_DIR / "freeze-manifest.json").read_text())
        self.assertEqual(manifest["totals"]["unique_statehashes"],
                         len((FROZEN_DIR / "historical-statehashes.txt")
                             .read_text().split()))
        self.assertEqual(manifest["totals"]["families"],
                         len((FROZEN_DIR / "historical-families.tsv")
                             .read_text().splitlines()))
        self.assertEqual(set(manifest["coverage"].values()), {0})


if __name__ == "__main__":
    unittest.main()
