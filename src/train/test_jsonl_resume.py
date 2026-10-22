"""JSONL 续跑修复的规则层测试（不依赖 torch/modal）。

跑法：uv run python src/train/test_jsonl_resume.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jsonl_resume import read_repair_jsonl  # noqa: E402


class JsonlResumeTest(unittest.TestCase):
    def test_normal(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.jsonl"
            p.write_text('{"id":"a"}\n{"id":"b"}\n', encoding="utf-8")
            rows, note, err = read_repair_jsonl(p)
            self.assertEqual(len(rows), 2)
            self.assertIsNone(note)
            self.assertIsNone(err)

    def test_missing_final_newline_repaired(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.jsonl"
            p.write_text('{"id":"a"}\n{"id":"b"}', encoding="utf-8")
            rows, note, err = read_repair_jsonl(p)
            self.assertEqual(rows, [{"id": "a"}, {"id": "b"}])
            self.assertEqual(note, "missing_newline")
            self.assertIsNone(err)
            self.assertTrue(p.read_bytes().endswith(b"\n"))

    def test_truncated_utf8_tail(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.jsonl"
            # 第二条是不完整 UTF-8/JSON
            p.write_bytes(b'{"id":"a"}\n{"id":"' + "中".encode()[:2])
            rows, note, err = read_repair_jsonl(p)
            self.assertEqual(rows, [{"id": "a"}])
            self.assertEqual(note, "tail_truncated")
            self.assertIsNone(err)
            self.assertEqual(p.read_text(encoding="utf-8"), '{"id": "a"}\n')

    def test_bad_middle(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.jsonl"
            p.write_text('{"id":"a"}\n{broken}\n{"id":"c"}\n', encoding="utf-8")
            rows, note, err = read_repair_jsonl(p)
            self.assertEqual(rows, [])
            self.assertIsNone(note)
            self.assertEqual(err, "bad_middle")

    def test_empty(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.jsonl"
            p.write_text("", encoding="utf-8")
            self.assertEqual(read_repair_jsonl(p), ([], None, None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
