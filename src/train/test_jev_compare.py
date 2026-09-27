"""jev_compare 读侧身份闸的单测：纯本地，不读真实数据。

跑法：uv run python src/train/test_jev_compare.py

背景（第八轮）：读侧断言此前是 `main()` 里的闭包、`src/` 下无任何测试 import
`jev_compare`——把断言改成恒过也不会有测试变红，正是第 4/6 轮批过的反模式。
故提升为模块级 `assert_identity` 并在此直测。
"""

from __future__ import annotations

import unittest

from jev_api_eval import MODEL
from jev_compare import SEALED_SHA, assert_identity, check_inputs


def jrow(rid: str, mode: str = "choice", **extra) -> dict:
    """一行 Jev 侧结果（现行格式：带齐三个身份字段）。"""
    return {"id": rid, "source": "s", "gold": "A", "correct": True,
            "mode": mode, "model_requested": MODEL, "exam_sha256": SEALED_SHA, **extra}


class TestAssertIdentity(unittest.TestCase):

    def test_all_good_rows_pass(self):
        rows = {r["id"]: r for r in (jrow("a"), jrow("b"))}
        assert_identity(rows, "choice", "jev-holdout.jsonl")

    def test_all_good_native_rows_pass(self):
        rows = {r["id"]: r for r in (jrow("a", mode="native-kind"),)}
        assert_identity(rows, "native-kind", "jev-native.jsonl")

    def test_wrong_mode_rejected(self):
        """native 文件里混入 choice 行——第八轮要防的污染方向。"""
        rows = {"a": jrow("a", mode="native-kind"), "b": jrow("b")}  # b 是 choice
        with self.assertRaises(AssertionError):
            assert_identity(rows, "native-kind", "jev-native.jsonl")

    def test_missing_mode_rejected(self):
        rows = {"a": jrow("a")}
        del rows["a"]["mode"]
        with self.assertRaises(AssertionError):
            assert_identity(rows, "choice", "jev-holdout.jsonl")

    def test_wrong_model_rejected(self):
        rows = {"a": jrow("a", model_requested="jev-1.12.0")}
        with self.assertRaises(AssertionError):
            assert_identity(rows, "choice", "jev-holdout.jsonl")

    def test_missing_model_rejected(self):
        rows = {"a": jrow("a")}
        del rows["a"]["model_requested"]
        with self.assertRaises(AssertionError):
            assert_identity(rows, "choice", "jev-holdout.jsonl")

    def test_wrong_sha_rejected(self):
        rows = {"a": jrow("a", exam_sha256="cafebabe" + "0" * 56)}
        with self.assertRaises(AssertionError):
            assert_identity(rows, "choice", "jev-holdout.jsonl")

    def test_missing_or_null_sha_rejected(self):
        """缺 sha / 显式 null / 空串都拒——曾经 sha 那一维过滤了空值（第八轮）。"""
        for bad in (None, ""):
            rows = {"a": jrow("a", exam_sha256=bad)}
            with self.assertRaises(AssertionError, msg=f"sha={bad!r} 应被拒"):
                assert_identity(rows, "choice", "jev-holdout.jsonl")
        rows = {"a": jrow("a")}
        del rows["a"]["exam_sha256"]
        with self.assertRaises(AssertionError):
            assert_identity(rows, "choice", "jev-holdout.jsonl")

    def test_empty_rows_pass(self):
        """空文件（如 --native 未产出）不触发断言。"""
        assert_identity({}, "native-kind", "jev-native.jsonl")


class TestCheckInputsAssembly(unittest.TestCase):
    """测装配函数 `check_inputs` 本身。

    ⚠️ **已知残余（不要当成已闭合）**：第九轮指出「把 `main()` 里两行调用整段删掉，
    9 例仍全绿」。抽出 `check_inputs()` 只是把调用收敛成**一行、可 grep 的**装配点，
    **并没有**让「删掉那一行」变红——本类直测函数，不经过 `main()`。

    为什么 `main()` 级的测试做不了：① 它对考卷 sha 有硬断言（等于封存卷 sha，
    合成文件无法满足）；② 它断言 `len(ids) == 1892`（硬编码）。要让编排层可测，
    得先把这两处参数化——属独立改造，未做。

    故本层由 code review 而非测试保证；两侧（写侧 `resume` 的调用、读侧
    `check_inputs` 的调用）残余相同。
    """

    def test_contaminated_native_rejected(self):
        """native 里混入 choice 行——第六轮那条「伪造 native_arm 结论」的路径。"""
        jev = {"a": jrow("a")}
        native = {"a": jrow("a"), "b": jrow("b")}      # b 是 choice，混进 native
        with self.assertRaises(AssertionError):
            check_inputs(jev, native, "jev-holdout.jsonl", "jev-native.jsonl")

    def test_contaminated_jev_rejected(self):
        jev = {"a": jrow("a", mode="native-kind")}
        with self.assertRaises(AssertionError):
            check_inputs(jev, {}, "jev-holdout.jsonl", "jev-native.jsonl")

    def test_clean_inputs_pass(self):
        jev = {"a": jrow("a")}
        native = {"a": jrow("a", mode="native-kind")}
        check_inputs(jev, native, "jev-holdout.jsonl", "jev-native.jsonl")

    def test_empty_native_skipped(self):
        """--native 缺省（空）时应跳过，不报错。"""
        check_inputs({"a": jrow("a")}, {}, "jev-holdout.jsonl", "jev-native.jsonl")


if __name__ == "__main__":
    unittest.main(verbosity=2)
