"""exam_report.validate_arms 的单测：纯本地，不读真实考卷、不联网。

跑法：uv run python src/train/test_exam_report.py

**为什么必须有这个文件**（2026-09-28 第三轮双谱系评审 P1）：上一轮给
`validate_arms` 加了「跨臂 prompt 指纹比对」并声称已验证，但验证只做了**单侧**——
只测了「坏样本会被抓」，没测「好样本能通过」。结果是：真实五臂数据上，
`tev1` 臂走 `qwen3_5` 族 tokenizer、指纹与四个 MiniCPM 臂**天然不同**，
该比对在真实数据上**必然报错**、`main()` 直接 `return 2` 拒绝产出——
发布指引让第三方跑的那条命令**根本跑不完**。
单侧验证读起来像「已覆盖」，实际把一个功能性回归放了进去：**正例必须与负例同测**。
"""

from __future__ import annotations

import unittest

from exam_report import ARMS, TOKENIZER_FAMILY, judge, validate_arms

IDS = [f"phishing:p{i}" for i in range(4)]


def ref_of(ids: list[str] = IDS) -> dict[str, dict]:
    """考卷 inputs.json 的形状（只取 validate_arms 用到的字段）。"""
    return {i: {"gold": "A", "suite": "phishing",
                "task": {"options": [{"label": "A", "key": "phishing"},
                                     {"label": "B", "key": "legitimate"}]}}
            for i in ids}


def recs_of(arm: str, sha: str, *, ids: list[str] = IDS, suite: str = "phishing",
            model: str | None = None) -> dict[str, dict]:
    return {i: {"gold": "A", "pred": "phishing", "pred_unc": "phishing",
                "suite": suite, "n_options": 2, "prompt_sha256": sha,
                "model": arm if model is None else model}
            for i in ids}


def rows_of(arms: dict[str, dict], extra_rows: dict[str, int] | None = None) -> dict[str, int]:
    """`load_arm` 一并返回的「文件非空行数」——无重复时等于记录数。"""
    out = {a: len(r) for a, r in arms.items()}
    out.update(extra_rows or {})
    return out


class TestValidateArms(unittest.TestCase):

    def test_same_family_consistent_passes(self):
        """四个 MiniCPM 臂指纹逐题一致 → 无 problem（**正例**，上一轮缺的就是它）。"""
        arms = {a: recs_of(a, "sha-same") for a in
                ("base", "adapter", "clean_r16", "adapter_r64")}
        self.assertEqual(validate_arms(arms, ref_of(), rows_of(arms)), [])

    def test_heterogeneous_family_not_flagged(self):
        """五臂齐全：tev1 是 qwen3_5 族、指纹必然不同 → **不得**因此报错。

        这正是上一轮放进来的功能性回归：修前该用例返回 1 条 problem，main() 随即
        `return 2`，第三方按发布指引跑不出报告 §2.5 的读数。
        """
        arms = {a: recs_of(a, "sha-same") for a in
                ("base", "adapter", "clean_r16", "adapter_r64")}
        arms["tev1"] = recs_of("tev1", "sha-qwen")
        self.assertEqual(validate_arms(arms, ref_of(), rows_of(arms)), [])

    def test_same_family_drift_is_flagged(self):
        """同族臂里有一臂题面不同（非同卷）→ 必须报。
        分组不能宽到把真问题一起放过——那是把回归换成漏报。"""
        arms = {a: recs_of(a, "sha-same") for a in
                ("base", "adapter", "clean_r16", "adapter_r64")}
        arms["clean_r16"]["phishing:p2"]["prompt_sha256"] = "sha-drifted"
        problems = validate_arms(arms, ref_of(), rows_of(arms))
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("非同卷", problems[0])
        self.assertIn("clean_r16", problems[0])
        self.assertIn("minicpm", problems[0])       # 报错要说明「同族才可比」

    def test_missing_arm_is_not_a_problem_here(self):
        """缺臂由 main 只 warn；validate_arms 只管**已加载的臂之间**是否同集同题。"""
        arms = {"base": recs_of("base", "sha-same")}
        self.assertEqual(validate_arms(arms, ref_of(), rows_of(arms)), [])

    def test_no_arm_loaded(self):
        self.assertEqual(validate_arms({"base": {}, "tev1": {}}, ref_of(), {}),
                         ["一个臂的产物都没有"])

    def test_duplicate_ids_flagged(self):
        """重复 id：dict 会静默去重，故必须与**读到的行数**比——
        旧写法 `len(recs) != len(set(recs))` 恒不成立（recs 本来就是 set）。"""
        arms = {"base": recs_of("base", "sha-same")}
        problems = validate_arms(arms, ref_of(), rows_of(arms, {"base": len(IDS) + 1}))
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("重复 id", problems[0])

    def test_uncovered_and_extra_ids_flagged(self):
        ref = ref_of(IDS + ["phishing:p99"])
        arms = {"base": recs_of("base", "sha-same", ids=IDS + ["phishing:ghost"])}
        problems = validate_arms(arms, ref, rows_of(arms))
        self.assertEqual(len(problems), 2, problems)
        self.assertTrue(any("不在考卷" in p for p in problems), problems)
        self.assertTrue(any("未覆盖全卷" in p for p in problems), problems)

    def test_suite_mismatch_flagged(self):
        ref = ref_of()
        ref["phishing:p1"]["suite"] = "tool_risk"
        arms = {"base": recs_of("base", "sha-same")}
        problems = validate_arms(arms, ref, rows_of(arms))
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("source 与考卷不一致", problems[0])

    def test_model_field_mismatch_flagged(self):
        """产物自称的臂名与文件名不符＝串臂（把 base 的产物当 adapter 用）。"""
        arms = {"adapter": recs_of("adapter", "sha-same", model="base")}
        problems = validate_arms(arms, ref_of(), rows_of(arms))
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("与臂名不符", problems[0])

    def test_tev1_without_fingerprint_is_skipped(self):
        """上游产物无 `prompt_sha256` 字段：不参与比对，也不因此报错。"""
        arms = {"base": recs_of("base", "sha-same")}
        arms["tev1"] = {i: {**r, "prompt_sha256": None}
                        for i, r in recs_of("tev1", "").items()}
        self.assertEqual(validate_arms(arms, ref_of(), rows_of(arms)), [])

    # ── fail-open 三连（2026-09-28 第四轮双谱系评审 P1，A/B 两条谱系各自独立报出）──
    # 修前的写法是 `if s:` 才收进比对、`len(sha) < 2` 就整族 `continue`：同族某臂一个
    # 指纹都没有时被**静默**排除，比对照样「通过」，而发布指引对外声称「同族逐题一致」。

    def test_same_family_arm_without_any_fingerprint_is_flagged(self):
        """同族某臂**一个指纹都没有** → 必须报（而非静默排除）。"""
        arms = {a: recs_of(a, "sha-same") for a in ("base", "adapter")}
        arms["clean_r16"] = {i: {**r, "prompt_sha256": None}
                             for i, r in recs_of("clean_r16", "").items()}
        problems = validate_arms(arms, ref_of(), rows_of(arms))
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("clean_r16", problems[0])
        self.assertIn("一个都没有", problems[0])

    def test_same_family_arm_with_partial_fingerprint_is_flagged(self):
        """★ 只删掉**发生差异那几题**的指纹即可躲过交集比对——这正是漏报的形态。"""
        arms = {a: recs_of(a, "sha-same") for a in ("base", "adapter")}
        arms["clean_r16"] = recs_of("clean_r16", "sha-same")
        arms["clean_r16"]["phishing:p2"]["prompt_sha256"] = None
        problems = validate_arms(arms, ref_of(), rows_of(arms))
        # 两条检查都会响（缺指纹 + 题集不同），都是同一处缺口的正确描述
        self.assertTrue(any("无 prompt 指纹" in p for p in problems), problems)

    def test_same_family_fingerprint_id_set_mismatch_is_flagged(self):
        """有指纹的**题集**不同（差几题）同样是覆盖率缺口。"""
        arms = {a: recs_of(a, "sha-same") for a in ("base", "adapter")}
        del arms["adapter"]["phishing:p1"]["prompt_sha256"]
        del arms["adapter"]["phishing:p0"]["prompt_sha256"]
        problems = validate_arms(arms, ref_of(), rows_of(arms))
        self.assertTrue(any("题集与" in p for p in problems), problems)


class TestFamilyRegistry(unittest.TestCase):

    def test_registry_covers_every_arm(self):
        """新增臂必须同步登记 tokenizer 族——否则该臂会被静默排除在指纹比对之外。"""
        self.assertEqual(set(TOKENIZER_FAMILY), set(ARMS))

    def test_family_table_is_pinned_to_eval_cross(self):
        """★ 族表必须与 `eval_cross.MODELS` **逐值**一致，而不只是键集合一致。

        第四轮评审（B 线 P1）：此前这条测试对着**字面量** `{"tev1"}` 断言——它看起来
        像「钉住上游来源」，实际钉的是自己；上游把一个臂的族改掉而本表没跟，测试照样绿。
        现在模块在**导入期**静态解析 `eval_cross.py` 的 `MODELS` 比对，这一条改为验证
        那次比对是活的（解析得到的映射就是本表）。
        """
        import exam_report
        self.assertEqual(exam_report._UPSTREAM_FAMILY, TOKENIZER_FAMILY)
        self.assertEqual({a for a, f in TOKENIZER_FAMILY.items() if f != "minicpm"},
                         {"tev1"})


class TestJudgeB(unittest.TestCase):
    """★ 判据 B 的方向分支（2026-09-28 第五轮评审 B 线 P1）。

    修前的分支只有「`r_base < 0.10`」与「否则」两支，**从不比较方向**：输入
    base=0.20 / r16=0.30 会输出「base 0.2000 高于 r16 的 0.3000，方向与『微调引入了
    塌陷』一致」——字面是假命题，结论方向还是反的。而这组输入正落在判据 A 的
    「部分保留」档，即预注册**预期会发生**的那一档，不是理论风险。

    下面六个用例把四个分支全部钉住（含反向与相等这两个修前会错的输入）。
    配对臂 `base` / `adapter` 缺一即不可判读，这一点单列。
    """

    @staticmethod
    def _m(r_base=None, r_2b=None) -> dict:
        m: dict = {}
        if r_base is not None:
            m["base"] = {"phishing": {"recall": r_base}}
        if r_2b is not None:
            m["adapter"] = {"phishing": {"recall": r_2b}}
        return m

    def test_missing_adapter_is_unjudgeable(self):
        """缺 r16（adapter）臂 → 不可判读，**不得**拿 None 当比较对象给因果断言。"""
        self.assertIsNone(judge(self._m(r_base=0.1420))["B"])

    def test_missing_base_is_unjudgeable(self):
        self.assertIsNone(judge(self._m(r_2b=0.0000))["B"])

    def test_base_below_ten_percent_points_at_backbone(self):
        """base 也低于 0.10 → 塌陷来自底座。（该分支优先于方向比较）"""
        b = judge(self._m(r_base=0.0300, r_2b=0.0000))["B"]
        self.assertIn("底座", b)
        self.assertIn("0.0300", b)

    def test_base_above_r16_reports_matching_direction(self):
        """反向之外的正例：base 明显高于 r16 → 方向一致（本仓真实读数即此）。"""
        b = judge(self._m(r_base=0.1420, r_2b=0.0000))["B"]
        self.assertIn("高于", b)
        self.assertNotIn("低于", b)
        self.assertIn("方向与「微调引入了塌陷」一致", b)
        self.assertIn("0.1420", b)
        self.assertIn("0.0000", b)

    def test_base_below_r16_reports_opposite_direction(self):
        """★ 修前会输出「0.2000 高于 r16 的 0.3000，方向一致」的那组输入。"""
        b = judge(self._m(r_base=0.2000, r_2b=0.3000))["B"]
        self.assertIn("低于", b)
        self.assertNotIn("**高于**", b)
        self.assertIn("相反", b)
        self.assertIn("未成立", b)

    def test_equal_recall_is_unjudgeable(self):
        """相等 → 无方向差异，不得择一而断。"""
        b = judge(self._m(r_base=0.1420, r_2b=0.1420))["B"]
        self.assertIn("相等", b)
        self.assertIn("不可判读", b)

    def test_direction_is_not_inferred_from_ten_percent_floor(self):
        """★ 第六轮 A 线 P2-3 ／ B 线 P2-4：底座低于阈值**不等于**「塌陷来自底座」。

        上一版只判 `r_base < 0.10`，不看 r16 是否也塌——于是 `r_base=0.05 / r_2b=0.30`
        （微调臂根本没塌）会输出「塌陷来自底座，非微调引入」，拿一个不成立的前提做因果
        断言。这与第五轮修掉的「方向颠倒」是同一类缺陷，落在同一函数的另一个分支上。
        现在该分支**必须先确认 r16 也塌**，否则降级为「本判据不适用」。
        """
        b = judge(self._m(r_base=0.0500, r_2b=0.3000))["B"]
        self.assertIn("不成立", b)
        self.assertIn("不适用", b)
        # 不能再出现「塌陷来自底座」这个**断言**（它现在只作为被否定的引文出现在句中）
        self.assertNotIn("→ 塌陷来自", b)
        self.assertNotIn("非微调引入", b)

    def test_base_below_ten_percent_with_r16_also_collapsed(self):
        """该分支的**正例**：两臂都塌（各自 < 0.10）时才说「塌陷来自底座」。"""
        b = judge(self._m(r_base=0.0300, r_2b=0.0000))["B"]
        self.assertIn("塌陷来自**底座**", b)
        self.assertIn("非微调引入", b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
