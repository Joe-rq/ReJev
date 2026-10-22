"""S1 冻结规格的规则层单测（不加载 torch / modal、不读真实数据）。

跑法：uv run --group align python src/train/test_s1_spec.py
（transformers 在 `align` 依赖组，不是默认组；R1 P2-8。）
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src/data"))

import render  # noqa: E402
import s1_spec as S  # noqa: E402


def _clean_train_constants() -> dict:
    """从 clean_train.py 源码取常量（不 import modal）。"""
    tree = ast.parse((REPO / "src/train/clean_train.py").read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            continue
        if isinstance(target, ast.Name):
            out[target.id] = value
        elif isinstance(target, ast.Tuple) and isinstance(value, tuple):
            for name, val in zip(target.elts, value):
                if isinstance(name, ast.Name):
                    out[name.id] = val
    return out


class ConstantParityTest(unittest.TestCase):
    def setUp(self):
        self.clean = _clean_train_constants()

    def test_model_revision_data(self):
        self.assertEqual(S.MODEL, self.clean["MODEL"])
        self.assertEqual(S.REV, self.clean["REV"])
        self.assertEqual(S.TRAIN_SHA, self.clean["TRAIN_SHA"])
        self.assertEqual(S.TRAIN_ROWS, self.clean["N_TRAIN"])
        self.assertEqual(S.N_DROP, self.clean["N_DROP"])

    def test_template_verbatim(self):
        self.assertEqual(S.TRAIN_TEMPLATE, self.clean["TRAIN_TEMPLATE"])
        self.assertEqual(S.IM_END, self.clean["IM_END"])

    def test_recipe_verbatim(self):
        self.assertEqual(S.LORA_R, self.clean["LOCKED_RANK"])
        self.assertEqual(S.LORA_ALPHA, self.clean["LOCKED_ALPHA"])

    def test_manifest_training_sha(self):
        m = json.loads((REPO / "docs/plan/009-s0-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(S.TRAIN_SHA, m["training_data"]["sha256"])
        self.assertEqual(S.TRAIN_ROWS, m["training_data"]["rows"])
        self.assertEqual(S.MODEL, m["render_contract"]["model"])
        self.assertEqual(S.REV, m["render_contract"]["revision"])
        self.assertEqual(list(S.SEEDS), m["rng_protocol"]["training_seeds"])
        self.assertEqual(S.MAX_TOKENS, m["render_contract"]["max_tokens"])
        self.assertEqual({k: v[0] for k, v in m["render_contract"]["label_token_ids"].items()},
                         S.LABEL_TOKEN_IDS)


class BuildMessagesParityTest(unittest.TestCase):
    def test_synthetic_parity(self):
        rec = {
            "state": {"context": "c", "record": {"x": 1}},
            "question": "q?",
            "options": [{"label": "A", "key": "yes", "description": "Yes."},
                        {"label": "B", "key": "no", "description": "No."}],
            "answer": "A",
        }
        self.assertEqual(S.build_messages(rec), render.build_messages(rec))

    def test_eval2_sample_parity(self):
        path = REPO / "data/eval2-final-20261017/records/dev2.jsonl"
        if not path.exists():
            self.skipTest("本地 eval2 数据不存在（gitignored）")
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= 50:
                    break
                rec = json.loads(line)
                self.assertEqual(S.build_messages(rec), render.build_messages(rec),
                                 f"第 {i} 行 build_messages 漂移")

    def test_rejects_bad_label_sequence(self):
        rec = {
            "state": "s", "question": "q",
            "options": [{"label": "B", "key": "x", "description": "x"},
                        {"label": "C", "key": "y", "description": "y"}],
            "answer": "B",
        }
        with self.assertRaises(ValueError):
            S.build_messages(rec)

    def test_rejects_answer_not_in_labels(self):
        rec = {
            "state": "s", "question": "q",
            "options": [{"label": "A", "key": "x", "description": "x"},
                        {"label": "B", "key": "y", "description": "y"}],
            "answer": "C",
        }
        with self.assertRaises(ValueError):
            S.build_messages(rec)


class SeedTest(unittest.TestCase):
    def test_allowed(self):
        for seed in (42, 17, 29, 41):
            S.assert_seed_allowed(seed)
            self.assertEqual(S.run_tag(seed), f"seed{seed}")
        self.assertEqual(S.artifact_dir(17), "/vol/s1/artifacts/seed17")
        self.assertEqual(S.checkpoint_dir(17), "/vol/s1/checkpoints/seed17")

    def test_rejected(self):
        for seed in (0, 43, 18):
            with self.assertRaises(AssertionError):
                S.assert_seed_allowed(seed)

    def test_explicit_retrain_tag(self):
        self.assertEqual(S.run_tag(42, "r2"), "seed42-r2")
        self.assertEqual(S.artifact_dir(42, "r2"), "/vol/s1/artifacts/seed42-r2")
        # R1 P1：合法 tag 也不能把 seed 层抹掉
        self.assertEqual(S.run_tag(17, "seed42"), "seed17-seed42")
        self.assertNotEqual(S.run_tag(17, "seed42"), S.run_tag(42))

    def test_rejects_path_traversal_tag(self):
        for tag in ("../escape", "a/b", ".hidden", "run"):
            with self.assertRaises(AssertionError):
                S.run_tag(42, tag)

    def test_recipe_locked(self):
        r = S.recipe(29)
        self.assertEqual(r["seed"], 29)
        self.assertEqual(r["lora"], "r16 a32 d0.05 all-linear")
        self.assertEqual(r["bs"], "2x2")
        self.assertEqual(r["lr"], 2e-4)
        self.assertEqual(r["seq"], 2048)
        self.assertTrue(r["assistant_only_loss"])
        self.assertEqual(r["rng_protocol"], "①")
        self.assertEqual(r["logging_steps"], 50)

    def test_sft_config_kwargs_locked(self):
        """R1 P3：核实际传给 SFTConfig 的完整参数字典，而不只是 manifest 副本。"""
        kw = S.sft_config_kwargs(29, "/vol/ckpt", "rejev-s1-seed29")
        self.assertEqual(kw["num_train_epochs"], 1)
        self.assertEqual(kw["per_device_train_batch_size"], 2)
        self.assertEqual(kw["gradient_accumulation_steps"], 2)
        self.assertEqual(kw["learning_rate"], 2e-4)
        self.assertEqual(kw["lr_scheduler_type"], "cosine")
        self.assertEqual(kw["warmup_ratio"], 0.03)
        self.assertTrue(kw["bf16"])
        self.assertEqual(kw["max_length"], 2048)
        self.assertFalse(kw["packing"])
        self.assertTrue(kw["assistant_only_loss"])
        self.assertEqual(kw["seed"], 29)
        self.assertEqual(kw["save_steps"], 1000)
        self.assertEqual(kw["save_total_limit"], 2)

    def test_identity_payload(self):
        src = {"multiseed_train.py": "m-sha", "s1_spec.py": "s-sha"}
        a = S.identity_payload(42, "sha", 30987, src, "seed42", "plan", "s0")
        b = S.identity_payload(17, "sha", 30987, src, "seed17", "plan", "s0")
        self.assertNotEqual(a["recipe_sha256"], b["recipe_sha256"])
        self.assertEqual(a["template_sha256"], S.TEMPLATE_SHA256)
        self.assertEqual(a["source_sha256"], src)
        c = S.identity_payload(42, "sha2", 30987, src, "seed42", "plan", "s0")
        self.assertNotEqual(a["data_sha256"], c["data_sha256"])
        d = S.identity_payload(42, "sha", 30987,
                               {"multiseed_train.py": "m-sha2", "s1_spec.py": "s-sha"},
                               "seed42", "plan", "s0")
        self.assertNotEqual(a["source_sha256"], d["source_sha256"])
        e = S.identity_payload(42, "sha", 30987, src, "seed42-r2", "plan", "s0")
        self.assertNotEqual(a["run_id"], e["run_id"])
        f = S.identity_payload(42, "sha", 30987, src, "seed42", "plan2", "s0")
        self.assertNotEqual(a["plan_sha256"], f["plan_sha256"])

    def test_identity_requires_both_source_files(self):
        with self.assertRaises(AssertionError):
            S.identity_payload(42, "sha", 30987, {"multiseed_train.py": "m"},
                               "seed42", "plan", "s0")

    def test_recipe_matches_sft_config(self):
        """R1 P2-7：manifest 配方与真正传给 SFTConfig 的字典必须逐项对得上。"""
        seed = 29
        r = S.recipe(seed)
        kw = S.sft_config_kwargs(seed, "/vol/ckpt", "rejev-s1-seed29")
        self.assertEqual(r["lr"], kw["learning_rate"])
        self.assertEqual(r["epochs"], kw["num_train_epochs"])
        self.assertEqual(r["bs"], f"{kw['per_device_train_batch_size']}x"
                                  f"{kw['gradient_accumulation_steps']}")
        self.assertEqual(r["lr_scheduler"], kw["lr_scheduler_type"])
        self.assertEqual(r["warmup_ratio"], kw["warmup_ratio"])
        self.assertEqual(r["seq"], kw["max_length"])
        self.assertEqual(r["bf16"], kw["bf16"])
        self.assertEqual(r["target_modules"],
                         ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"])
        self.assertEqual(r["packing"], kw["packing"])
        self.assertEqual(r["assistant_only_loss"], kw["assistant_only_loss"])
        self.assertEqual(r["logging_steps"], kw["logging_steps"])
        self.assertEqual(r["save_steps"], kw["save_steps"])
        self.assertEqual(r["save_total_limit"], kw["save_total_limit"])
        self.assertEqual(r["save_strategy"], kw["save_strategy"])
        self.assertEqual(r["seed"], kw["seed"])
        self.assertEqual(kw["save_strategy"], "steps")
        self.assertEqual(kw["report_to"], "wandb")
        self.assertEqual(kw["run_name"], "rejev-s1-seed29")
        self.assertEqual(kw["output_dir"], "/vol/ckpt")
        self.assertEqual(set(kw), {
            "output_dir", "num_train_epochs", "per_device_train_batch_size",
            "gradient_accumulation_steps", "learning_rate", "lr_scheduler_type",
            "warmup_ratio", "bf16", "max_length", "packing",
            "assistant_only_loss", "logging_steps", "save_steps",
            "save_total_limit", "save_strategy", "seed", "report_to", "run_name",
        })

    def test_step_loss_window(self):
        # 父类返回值已按 GA 归一：同一步内求和，再对窗口内步取均值。
        # step1=0.1+0.3=0.4, step2=0.2+0.4=0.6, step3=0.5+0.7=1.2
        losses = [(1, 0.1), (1, 0.3), (2, 0.2), (2, 0.4), (3, 0.5), (3, 0.7)]
        sums = S.step_loss_sums(losses)
        self.assertEqual([s for s, _ in sums], [1, 2, 3])
        self.assertAlmostEqual(sums[0][1], 0.4)
        out = S.last_n_mean(losses, n=2)
        self.assertAlmostEqual(out["last_n_mean"], 0.9)
        self.assertEqual(out["window_start_step"], 2)
        self.assertEqual(out["window_end_step"], 3)
        self.assertEqual(out["n_steps_total"], 3)
        self.assertTrue(out["complete_50"])
        short = S.last_n_mean(losses, n=50)
        self.assertIsNone(short["last_n_mean"])
        self.assertFalse(short["complete_50"])

    def test_step_loss_validation(self):
        # 中间步缺一份贡献
        bad = [(1, 0.1), (1, 0.3), (2, 0.2), (3, 0.5), (3, 0.6)]
        ok, why = S.validate_step_losses(bad, expected_per_step=2)
        self.assertFalse(ok)
        self.assertIn("bad_count", why)
        # 非整数步号
        self.assertFalse(S.validate_step_losses([(1.5, 0.1)], 2)[0])
        # 非有限值
        self.assertFalse(S.validate_step_losses([(1, float("nan"))], 2)[0])
        # 步号不连续
        self.assertFalse(S.validate_step_losses(
            [(1, 0.1), (1, 0.2), (3, 0.3), (3, 0.4)], 2)[0])
        # 冻结配置每个优化步都应有 GA 份；首/末步也不放宽（R4 P2）
        self.assertFalse(S.validate_step_losses([(1, 0.1), (1, 0.2), (2, 0.3)], 2)[0])
        self.assertFalse(S.validate_step_losses([(1, 0.2), (2, 0.3), (2, 0.4)], 2)[0])
        # JSON 往返（list 而非 tuple）仍然有效
        self.assertTrue(S.validate_step_losses(
            [[1, 0.1], [1, 0.2], [2, 0.3], [2, 0.4]], 2)[0])
        # 非法 JSON 值不抛异常（含字符串/布尔/超大整数，R5 P3）
        for bad_entries in ([[1, None]], [[1, "bad"]], [[1, True]],
                            [[1, "0.5"]], [[1, 10 ** 400]], [], [1, 2]):
            ok, _ = S.validate_step_losses(bad_entries, 2)
            self.assertFalse(ok)
        # last_n_mean 对非法窗口返回 None，不填短窗口
        out = S.last_n_mean(bad, n=2)
        self.assertIsNone(out["last_n_mean"])
        self.assertFalse(out["complete_50"])
        self.assertIn("bad_count", out["validation_error"])

    def test_loss_window_file_validation(self):
        entries = [[s, 0.5] for s in range(951, 1001) for _ in range(2)]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "loss_window.json"
            p.write_text(json.dumps({"saved_through_step": 1000,
                                     "step_losses": entries}), encoding="utf-8")
            self.assertTrue(S.validate_loss_window(str(p), 1000, 2)[0])
            # 末步缺一份贡献
            p.write_text(json.dumps({"saved_through_step": 1000,
                                     "step_losses": entries[:-1]}), encoding="utf-8")
            self.assertFalse(S.validate_loss_window(str(p), 1000, 2)[0])
            # 顶层不是 dict
            p.write_text(json.dumps([]), encoding="utf-8")
            self.assertFalse(S.validate_loss_window(str(p), 1000, 2)[0])
            # saved_through_step 为 null / 非整数
            p.write_text(json.dumps({"saved_through_step": None,
                                     "step_losses": entries}), encoding="utf-8")
            self.assertFalse(S.validate_loss_window(str(p), 1000, 2)[0])
            p.write_text(json.dumps({"saved_through_step": 1000.9,
                                     "step_losses": entries}), encoding="utf-8")
            self.assertFalse(S.validate_loss_window(str(p), 1000, 2)[0])
            # 条目含 null / 字符串
            for bad in ([[1000, None]], [[1000, "bad"]]):
                p.write_text(json.dumps({"saved_through_step": 1000,
                                         "step_losses": bad}), encoding="utf-8")
                self.assertFalse(S.validate_loss_window(str(p), 1000, 2)[0])


class SeedTimingTest(unittest.TestCase):
    def test_recorded_lines_exist(self):
        """登记的行号必须仍指向 set_seed 调用/定义（防 transformers 升级后静默过期）。"""
        import transformers
        self.assertEqual(transformers.__version__, S.SET_SEED_TIMING["transformers"])
        root = Path(transformers.__file__).resolve().parent
        tr = (root / "trainer.py").read_text(encoding="utf-8").splitlines()
        tu = (root / "trainer_utils.py").read_text(encoding="utf-8").splitlines()
        tr_line = int(S.SET_SEED_TIMING["trainer_init_set_seed"].rsplit(":", 1)[1])
        tu_line = int(S.SET_SEED_TIMING["set_seed_body"].rsplit(":", 1)[1].split("-")[0])
        self.assertIn("set_seed(self.args.seed)", tr[tr_line - 1])
        self.assertIn("def set_seed", tu[tu_line - 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
