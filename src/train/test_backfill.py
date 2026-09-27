"""backfill_identity_fields 的单测：纯本地、临时目录，不碰真实产物。

跑法：uv run python src/train/test_backfill.py

背景（第九轮）：第八轮给回填脚本加的四项加固（整行指纹、先验后写、空值修复、
逐行类型核验）**当时零测试覆盖**——把指纹改回只哈希三个统计字段也 34 例全绿。
那正是第 4/6/7 轮反复踩的「核心修法未被测试装配」，故补此文件。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backfill_identity_fields import expected_mode, fingerprint


def row(rid: str, **extra) -> dict:
    return {"id": rid, "source": "s", "gold": "A", "correct": True,
            "ans_type": "choice", "probs": {"A": 1.0}, **extra}


class TestFingerprint(unittest.TestCase):
    """指纹必须覆盖**整行**（剔除三个回填键），否则改了统计字段也检不出来。"""

    BASE = {"id": "a", "source": "s", "gold": "A", "correct": True}

    def test_changes_in_source_or_probs_are_detected(self):
        for field, val in (("source", "other"), ("probs", {"A": 0.5}),
                           ("confidence", 0.9), ("kind", "score")):
            a = [{**self.BASE}]
            b = [{**self.BASE, field: val}]
            self.assertNotEqual(fingerprint(a), fingerprint(b), f"{field} 的改动必须被检出")

    def test_backfill_fields_are_excluded(self):
        """三个回填键本身不进指纹——否则回填前后必然不同，指纹失去意义。"""
        a = [{**self.BASE}]
        b = [{**self.BASE, "mode": "choice", "model_requested": "jev-1.13.0",
              "exam_sha256": "deadbeef"}]
        self.assertEqual(fingerprint(a), fingerprint(b))

    def test_does_not_mutate_input(self):
        rows = [{**self.BASE}]
        fingerprint(rows)
        self.assertEqual(rows, [{**self.BASE}])


class TestExpectedMode(unittest.TestCase):

    def test_choice_rows(self):
        self.assertEqual(expected_mode([row("a"), row("b")]), "choice")

    def test_native_rows(self):
        rows = [row("a", ans_type="score"), row("b", ans_type="noul")]
        self.assertEqual(expected_mode(rows), "native-kind")

    def test_mixed_types_rejected(self):
        with self.assertRaises(SystemExit):
            expected_mode([row("a", ans_type="choice"), row("b", ans_type="score")])

    def test_all_error_rows_rejected(self):
        """整文件皆故障行（无 ans_type）→ 无法判定，拒绝而非默认 choice。"""
        with self.assertRaises(SystemExit):
            expected_mode([row("a", correct=None, ans_type=None)])

    def test_invalid_answer_row_uses_raw_answer_type(self):
        """invalid_answer 行不写 ans_type，类型只在 raw_answer.type——不能被误拒。"""
        r = row("a", correct=False, ans_type=None,
                raw_answer={"type": "choice", "choice": "Z"})
        self.assertEqual(expected_mode([r, row("b")]), "choice")

    def test_judged_row_without_any_type_rejected(self):
        with self.assertRaises(SystemExit):
            expected_mode([row("a", ans_type=None)])


class TestValidationBeforeWrite(unittest.TestCase):
    """「先验后写」：校验失败时磁盘不得被改动（第九轮改为两文件统一写盘）。"""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.d = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def _write(self, stem, rows):
        (self.d / f"{stem}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        (self.d / f"{stem}-summary.json").write_text(
            json.dumps({"mode": "choice"}), encoding="utf-8")

    def test_conflicting_value_aborts_without_touching_disk(self):
        """第二份文件冲突时，第一份也不得已被写盘。"""
        import backfill_identity_fields as bf

        ok = row("a")
        bad = row("a", model_requested="jev-1.12.0")   # 非空且不符 → 必须报错
        self._write("jev-holdout", [ok])
        self._write("jev-native", [bad])
        before = (self.d / "jev-holdout.jsonl").read_bytes()

        # 测试不该依赖真实封存考卷：把它指向临时文件（main 只用它算 sha）
        fake_exam = self.d / "fake-exam.jsonl"
        fake_exam.write_text('{"id":"x"}\n', encoding="utf-8")

        orig_stems, orig_exam = bf.STEMS, bf.SEALED_EXAM
        bf.STEMS = ("jev-holdout", "jev-native")
        bf.SEALED_EXAM = fake_exam
        try:
            with self.assertRaises(SystemExit):
                import sys
                saved = sys.argv
                sys.argv = ["backfill", "--dir", str(self.d), "--check"]
                try:
                    bf.main()
                finally:
                    sys.argv = saved
        finally:
            bf.STEMS, bf.SEALED_EXAM = orig_stems, orig_exam
        self.assertEqual((self.d / "jev-holdout.jsonl").read_bytes(), before,
                         "第一份文件在校验失败前就被写盘了——整次运行不原子")


if __name__ == "__main__":
    unittest.main(verbosity=2)
