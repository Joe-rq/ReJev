"""jev_api_eval 的续跑/去重/身份闸单测：纯本地，不发任何请求、不读真实数据。

跑法：uv run python src/train/test_jev_api_eval.py

背景（为什么有这个文件）：第三轮评审实测指出，早期版本「只在内存里跳过坏末行、
文件不截断就继续追加写」，会把新结果**接在半行后面**产出既解析不了、又可能因
行数恰好相等而躲过收尾重写的脏数据。

**第四轮两个谱系都指出：上一版测试只测了零件、没测装配**——把 `main()` 里
「坏行就调 `_rewrite`」那几行删掉，测试照样全绿。所以这里改为直接测 `resume()`
（`main()` 真正调用的那个入口），并按第四轮补了「可达方向」的去重用例。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jev_api_eval import MODEL, SEALED_EXAM, _rewrite, check_identity, load_prior, resume

def row(rid: str, correct=True, legacy: bool = False, **extra) -> dict:
    """legacy=True 模拟旧格式行（无 mode/model_requested，无逐行指纹）。"""
    base = {"id": rid, "source": "s", "gold": "A", "jev_label": "A" if correct else "B",
            "correct": correct}
    if not legacy:
        base["mode"] = "choice"
        base["model_requested"] = MODEL
    return {**base, **extra}


def write_raw(path: Path, text: str) -> None:
    """按字节写，用于制造「写到一半被打断」的现场（无末尾换行 / 半截 JSON）。"""
    path.write_text(text, encoding="utf-8")


def raw_of(*rows: dict) -> str:
    """连成合法的 jsonl 文本（**带**末尾换行），供再切一刀。"""
    return "".join(json.dumps(r) + "\n" for r in rows)


class TestLoadPrior(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.out = Path(self.dir.name) / "out.jsonl"

    def tearDown(self):
        self.dir.cleanup()

    def test_missing_file(self):
        prior, repairs = load_prior(self.out)
        self.assertEqual((prior, repairs), ({}, []))

    def test_partial_last_line_is_reported(self):
        # 半截行天然没有末尾换行，所以两条修复原因会同时成立——这没问题，
        # 断言只要求「损坏」这一条被报出来即可。
        sha = "deadbeef"
        write_raw(self.out, raw_of(row("a", exam_sha256=sha), row("b", exam_sha256=sha)) + '{"id": "c", "corr')
        prior, repairs = load_prior(self.out)
        self.assertEqual(set(prior), {"a", "b"})
        self.assertTrue(any("损坏" in r for r in repairs), repairs)

    def test_unterminated_last_line_is_reported(self):
        """完整 JSON 但缺末尾换行：能解析，可追加仍会与它拼接——必须同样报修。"""
        write_raw(self.out, raw_of(row("a")) + json.dumps(row("b")))  # 末行无 \n
        prior, repairs = load_prior(self.out)
        self.assertEqual(set(prior), {"a", "b"})
        self.assertEqual(len(repairs), 1)
        self.assertIn("缺换行", repairs[0])

    def test_middle_bad_line_is_fatal(self):
        """中间坏行说明文件被破坏：宁可报错，也不要带着缺口往下跑。"""
        write_raw(self.out, raw_of(row("a")) + '{"broken\n' + json.dumps(row("c")) + "\n")
        with self.assertRaises(AssertionError):
            load_prior(self.out)

    def test_u2028_inside_a_json_string_does_not_split_the_line(self):
        """只按 "\n" 切：U+2028/U+2029/\x0b 可能合法地出现在 JSON 字符串里
        （如 error_detail/raw_answer），splitlines() 会把一行劈成两截而误判「中间坏行」。
        ⚠️ 必须用 ensure_ascii=False 写盘（与生产 writer 一致）：默认 json.dumps 会把
        U+2028 转义成字面反斜杠u2028 六个 ASCII 字符，文件里根本没有真实分隔符——
        第八轮证明那样写出的测试是空转（干净的 splitlines 变异当时 25 例全绿）。
        """
        tricky = row("a", error_detail="first\u2028second\u2029third")
        raw = (json.dumps(tricky, ensure_ascii=False) + "\n"
               + json.dumps(row("b"), ensure_ascii=False) + "\n")
        # 前置自检：真实分隔符必须真的落进文件，否则本测试退化为空转（第八轮教训）。
        # 注：\x0b 等控制符被 JSON 规范强制转义、不可能裸现在产物里，风险只有 U+2028/2029。
        assert chr(0x2028) in raw and chr(0x2029) in raw
        write_raw(self.out, raw)
        prior, repairs = load_prior(self.out)
        self.assertEqual(set(prior), {"a", "b"})
        self.assertEqual(repairs, [])
        self.assertEqual(prior["a"]["error_detail"], "first\u2028second\u2029third")

    def test_non_object_json_line_is_reported_as_corrupt(self):
        """合法 JSON 但不是对象（[]、"x"）应报「行损坏」，而不是抛 TypeError 栈回溯。"""
        write_raw(self.out, json.dumps(row("a")) + "\n" + "[]")
        prior, repairs = load_prior(self.out)
        self.assertEqual(set(prior), {"a"})
        self.assertTrue(any("损坏" in r for r in repairs), repairs)

    def test_blank_line_is_reported_not_silently_dropped(self):
        """本文件由我们的 writer 产出、不应有空行；出现了就要报修，不能静默丢弃。"""
        write_raw(self.out, json.dumps(row("a")) + "\n\n" + json.dumps(row("b")) + "\n")
        prior, repairs = load_prior(self.out)
        self.assertEqual(set(prior), {"a", "b"})
        self.assertTrue(any("空白行" in r for r in repairs), repairs)

    # —— 去重语义：四个方向都测，重点是**续跑时真正可达的那一个** ——

    def test_dedup_reachable_direction_failure_then_success(self):
        """续跑里最典型的同 id 重复：先写故障行（correct=None），重投后写成功行 → 取成功行。

        注：故障→判定不是**唯一**可达的方向——smoke 那种按多维采样产生的重复 id
        会让「判定→判定」在首跑就出现（正是 records_n=16 → n_total=10 的机制）。
        """
        write_raw(self.out, raw_of(row("a", correct=None, error="URLError"), row("a")))
        prior, _ = load_prior(self.out)
        self.assertTrue(prior["a"]["correct"], "重投成功的新结果必须覆盖故障行")

    def test_dedup_judged_beats_unjudged_regardless_of_order(self):
        """反向顺序也要保住好行（少见，但语义必须对称，否则将来改动了会静默丢数据）。"""
        write_raw(self.out, raw_of(row("a"), row("a", correct=None, error="URLError")))
        prior, _ = load_prior(self.out)
        self.assertTrue(prior["a"]["correct"])

    def test_dedup_last_judged_wins(self):
        """同为已判定取后者（文件按追加＝时间顺序，后者是更新的结果）。"""
        write_raw(self.out, raw_of(row("a", jev_label="A"), row("a", jev_label="B")))
        prior, _ = load_prior(self.out)
        self.assertEqual(prior["a"]["jev_label"], "B")


class TestResumeOrchestration(unittest.TestCase):
    """测**装配**而非零件：`main()` 调的就是 `resume()`。

    第四轮评审指出，上一版测试全在测 `load_prior`/`_rewrite` 的隔离行为，
    把 main 里「坏行就调 `_rewrite`」那段接线删掉也照样绿——等于给了假保证。
    """

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.out = Path(self.dir.name) / "out.jsonl"

    def tearDown(self):
        self.dir.cleanup()

    def test_resume_repairs_partial_line_so_append_is_safe(self):
        """P1 的复现路径：坏末行 → resume → 文件必须被真正截断，追加不再拼接。"""
        sha = "deadbeef"
        write_raw(self.out, raw_of(row("a", exam_sha256=sha),
                                   row("b", exam_sha256=sha)) + '{"id": "c", "corr')

        prior = resume(self.out, "choice", sha)
        self.assertEqual(set(prior), {"a", "b"})

        # 关键断言：磁盘上不再有坏行（旧实现只跳过不截断，这里会失败）
        _, repairs = load_prior(self.out)
        self.assertEqual(repairs, [], "resume 必须把坏末行从磁盘上抹掉，而不只是内存里跳过")

        # 追加后文件应当逐行可解析
        with open(self.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row("c", exam_sha256=sha)) + "\n")
        reread, repairs2 = load_prior(self.out)
        self.assertEqual(repairs2, [])
        self.assertEqual(set(reread), {"a", "b", "c"})

    def test_resume_is_noop_on_missing_file(self):
        self.assertEqual(resume(self.out, "choice", "deadbeef"), {})
        self.assertFalse(self.out.exists(), "空文件不该被凭空创建")

    def test_resume_enforces_identity_gate(self):
        """经 resume() 走一遍身份闸——只测 check_identity 不够：接线本身也要可见。"""
        write_raw(self.out, raw_of(row("a", legacy=True)))  # 旧格式行（无 mode/model）
        with self.assertRaises(AssertionError):
            resume(self.out, "choice", "deadbeef")

    def test_resume_passes_identity_gate_for_sealed_exam(self):
        write_raw(self.out, raw_of(row("a", exam_sha256="deadbeef")))
        self.assertEqual(set(resume(self.out, "choice", "deadbeef")), {"a"})


class TestIdentityGate(unittest.TestCase):
    """直接测闸本身——上一版把判据抄一遍再断言，等于没测。"""

    SHA = "deadbeef"

    def test_mode_mismatch_rejected(self):
        with self.assertRaises(AssertionError):
            check_identity({"a": row("a", mode="native-kind")}, "choice", self.SHA)

    def test_model_mismatch_rejected(self):
        with self.assertRaises(AssertionError):
            check_identity({"a": row("a", model_requested="jev-1.12.0")}, "choice",
                           self.SHA)

    def test_exam_sha_mismatch_rejected(self):
        with self.assertRaises(AssertionError):
            check_identity({"a": row("a", exam_sha256="cafebabe")}, "choice",
                           self.SHA)

    def test_matching_rows_pass(self):
        check_identity({"a": row("a", mode="choice", model_requested=MODEL,
                                 exam_sha256=self.SHA)}, "choice", self.SHA)

    def test_fingerprintless_rows_rejected_even_against_sealed_exam(self):
        """缺逐行指纹的行，封存考卷下也一律拒。

        曾经过三代演变：先是「自定义考卷拒、封存放行」（运营事实当判据）；第八轮指出
        封存路径同样证明不了旧行属于本次考卷，而回填脚本已让所有现行产物带指纹——
        容忍再无存在理由，遂并进「三字段一律必填」。
        """
        with self.assertRaises(AssertionError):
            check_identity({"a": row("a")}, "choice", self.SHA)

    def test_fingerprintless_rows_rejected_against_custom_exam(self):
        """同一批行换到自定义考卷上必须拒——无法证明同源。

        ⚠️ 这里曾经有过一个 `--allow-custom-records` 的豁免，第五轮被两个谱系指出是死分支：
        那个 flag 对自定义考卷**本来就必加**，豁免恒为真 ⟹ 拒绝分支在 CLI 永不可达，
        等于把这条防线整个注销。故删除豁免、回到无条件拒绝。
        """
        with self.assertRaises(AssertionError):
            check_identity({"a": row("a")}, "choice", self.SHA)

    # —— 第六轮补的两处漏洞 ——

    def test_explicit_null_fingerprint_is_not_verifiable(self):
        """`exam_sha256: null`（键在值为空）必须与「缺键」同等对待。

        曾经 `stale` 排除 None、`no_sha` 只数缺键，于是显式空值两边都漏掉、被当已核验放行。
        """
        with self.assertRaises(AssertionError):
            check_identity({"a": row("a", exam_sha256=None)}, "choice", self.SHA)

    def test_missing_or_null_identity_fields_are_refused(self):
        """逐行三个身份字段「缺键」与「显式空值」同等对待——一律拒绝。

        这里连着栽过两次：先给 exam_sha256 补了「null 也算缺失」，却在
        model_requested 上留了同一个洞（`null`/`""` 被当成「没有已见模型」而放行）。
        """
        for field in ("mode", "model_requested", "exam_sha256"):
            for bad in (None, ""):
                with self.assertRaises(AssertionError, msg=f"{field}={bad!r} 应被拒"):
                    check_identity({"a": row("a", **{field: bad})}, "choice",
                                   self.SHA)
        # 缺键同理
        with self.assertRaises(AssertionError):
            check_identity({"a": row("a", legacy=True)}, "choice", self.SHA)

    def test_legacy_rows_are_refused_even_against_sealed_exam(self):
        """旧格式行（无 mode/model）不再靠「同目录汇总」之类的佐证放行。

        那条启发式曾是本轮的修法，但那份 summary 与结果行之间**没有任何绑定**——
        错置或过期的 summary 照样能骗过闸（第七轮实测：把 native 行的 mode 去掉、
        配一份 choice 的 summary，闸就放行了，尽管该行 ans_type=score）。
        改为一律拒绝；旧文件换个新 --out 即可，回填脚本见 src/train/backfill_identity_fields.py。
        """
        with self.assertRaises(AssertionError):
            check_identity({"a": row("a", legacy=True)}, "choice", self.SHA)

    def test_empty_prior_passes(self):
        check_identity({}, "choice", self.SHA)


class TestRewrite(unittest.TestCase):

    def test_rewrite_is_line_parseable(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "o.jsonl"
            _rewrite(out, {"a": row("a"), "b": row("b")})
            prior, repairs = load_prior(out)
            self.assertEqual(repairs, [])
            self.assertEqual(set(prior), {"a", "b"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
