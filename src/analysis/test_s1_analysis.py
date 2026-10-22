"""S1 分析驱动的端到端合成测试（含身份/置换/校准校验）。

跑法：uv run python src/analysis/test_s1_analysis.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/train"))
import candidate_math as CM  # noqa: E402
import s1_spec as S  # noqa: E402

import hashlib as _hashlib
SOURCE = {k: _hashlib.sha256(k.encode()).hexdigest()
          for k in ("eval_candidates.py", "candidate_math.py", "s1_spec.py")}

ARMS = ["base-c0", "hist", "seed42", "seed17", "seed29", "seed41"]
SEEDS = [42, 17, 29, 41]
SEMANTIC = {
    "policy_v2": ["eligible", "ineligible", "unknown"],
    "boolq": ["no", "yes"],
    "mnli": ["contradicted", "entailed", "neutral"],
}


def _h(text: str) -> str:
    return _hashlib.sha256(text.encode()).hexdigest()


def hist_sha() -> str:
    return _h("hist")


def seed_sha(seed: int) -> str:
    return _h(f"seed{seed}")


def hist_cfg() -> str:
    return _h("hist-cfg")


def seed_cfg(seed: int) -> str:
    return _h(f"seed{seed}-cfg")


def make_raw(i: int, source: str, group: str, n_opt: int) -> dict:
    keys = SEMANTIC[source][:n_opt]
    opts = [{"label": CM.LETTERS[j], "key": k, "description": f"d{j}"}
            for j, k in enumerate(keys)]
    answer = opts[i % n_opt]["label"]
    return {"id": f"{source}:{group}:{i}", "source": source, "group_id": f"{source}:{group}",
            "kind": "choice", "state": "s", "question": "q?",
            "options": opts, "answer": answer,
            "answer_key": opts[i % n_opt]["key"]}


def make_split(n: int) -> list[dict]:
    rows = []
    for i in range(n):
        if i % 3 == 0:
            rows.append(make_raw(i, "policy_v2", f"g{i // 2}", 3))
        elif i % 3 == 1:
            rows.append(make_raw(i, "boolq", f"g{i}", 2))
        else:
            rows.append(make_raw(i, "mnli", f"g{i}", 3))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> tuple[str, int]:
    import hashlib
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest(), len(rows)


def score_record(raw: dict, arm: str, split_label: str, data_sha: str,
                 adapter: str | None, adapter_cfg: str | None,
                 model_kind: str, art: str | None) -> dict:
    rng = random.Random(raw["id"] + arm + split_label)
    labels = [o["label"] for o in raw["options"]]
    keys = [o["key"] for o in raw["options"]]
    logits = [rng.uniform(-1, 1) for _ in labels]
    m = max(logits)
    ex = [math.exp(v - m) for v in logits]
    s = sum(ex)
    probs = [v / s for v in ex]
    pred = max(range(len(probs)), key=lambda j: probs[j])
    gold = labels.index(raw["answer"])
    out = {
        "id": raw["id"], "source": raw["source"], "split": split_label,
        "group_id": raw["group_id"], "kind": raw.get("kind"),
        "arm": arm, "tag": f"{arm}-{split_label}", "model_kind": model_kind,
        "adapter_sha256": adapter, "adapter_config_sha256": adapter_cfg,
        "art": art,
        "source_sha256": SOURCE, "template_sha256": S.TEMPLATE_SHA256,
        "model": S.MODEL, "revision": S.REV,
        "data_sha256": data_sha,
        "option_labels": labels, "option_keys": keys,
        "logits_by_label": dict(zip(labels, logits)),
        "probs_by_label": dict(zip(labels, probs)),
        "probs_by_key": dict(zip(keys, probs)),
        "gold_label": raw["answer"], "gold_key": raw["answer_key"],
        "pred_label": labels[pred], "pred_key": keys[pred],
        "correct": pred == gold, "correct_key": keys[pred] == raw["answer_key"],
        "p_gold": probs[gold], "scoring_version": "s1-candidate-v2",
        "code_commit": "a" * 40,
        "run_fingerprint": "",
    }
    out["run_fingerprint"] = CM.fingerprint({
        "data_sha256": out["data_sha256"], "arm": out["arm"], "tag": out["tag"],
        "split": out["split"], "model_kind": out["model_kind"], "art": out["art"],
        "adapter_sha256": out["adapter_sha256"],
        "adapter_config_sha256": out["adapter_config_sha256"],
        "source_sha256": out["source_sha256"],
        "template_sha256": out["template_sha256"], "model": out["model"],
        "revision": out["revision"], "scoring_version": out["scoring_version"], "n": 0,
    })
    return out


class EndToEndTest(unittest.TestCase):
    def test_synthetic_pipeline(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            rawdir = td / "records"
            rawdir.mkdir()
            raw = {"dev2": make_split(6), "calibration2": make_split(24),
                   "test2": make_split(24)}
            raw["test2_permuted"] = [CM.permute_record(r) for r in raw["test2"]]
            meta = {}
            for key, rows in raw.items():
                name = "test2-permuted" if key == "test2_permuted" else key
                sha, n = write_jsonl(rawdir / f"{name}.jsonl", rows)
                meta[key] = {"path": str(rawdir / f"{name}.jsonl"), "sha256": sha, "rows": n}
            s1 = {
                "eval2_files": meta,
                "reference_arms": {"base-c0": {"adapter_sha256": None,
                                               "adapter_config_sha256": None},
                                   "hist": {"adapter_sha256": hist_sha(),
                                            "adapter_config_sha256": hist_cfg()}},
                "runs": [{"run_tag": f"seed{s}", "adapter_sha256": seed_sha(s),
                          "adapter_config_sha256": seed_cfg(s)} for s in SEEDS],
                "eval_code_sha256": SOURCE,
            }
            s1path = td / "s1-manifest.json"
            s1path.write_text(json.dumps(s1), encoding="utf-8")
            s0 = json.loads((REPO / "docs/plan/009-s0-manifest.json").read_text())
            s0["error_analysis_protocol"]["denominator_n_test2_families"] = 24
            s0path = td / "s0-manifest.json"
            s0path.write_text(json.dumps(s0), encoding="utf-8")

            scores_dir = td / "scores"
            scores_dir.mkdir()
            rawmaps = {k: {r["id"]: r for r in v} for k, v in raw.items()}
            split_meta = [("dev2", "dev2", "dev2"),
                          ("calibration2", "calibration2", "calibration2"),
                          ("test2", "test2", "test2"),
                          ("test2-perm", "test2_permuted", "test2-perm")]
            for arm in ARMS:
                mk = "base" if arm == "base-c0" else "adapter"
                adapter = (None if arm == "base-c0"
                           else hist_sha() if arm == "hist"
                           else seed_sha(int(arm[4:])))
                adapter_cfg = (None if arm == "base-c0"
                               else hist_cfg() if arm == "hist"
                               else seed_cfg(int(arm[4:])))
                for split_key, raw_key, split_label in split_meta:
                    art = (None if arm == "base-c0" else
                           "/vol/artifacts/clean-r16" if arm == "hist" else
                           f"/vol/s1/artifacts/{arm}")
                    rows = [score_record(r, arm, split_label,
                                         meta[raw_key]["sha256"],
                                         adapter, adapter_cfg, mk, art)
                            for r in raw[raw_key]]
                    (scores_dir / f"{arm}-{split_key}.jsonl").write_text(
                        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                        encoding="utf-8")
            out = td / "out"
            cmd = [sys.executable, str(REPO / "src/analysis/s1_analysis.py"),
                   "--scores-dir", str(scores_dir), "--out", str(out),
                   "--s0", str(s0path),
                   "--s1", str(s1path)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            res = json.loads((out / "analysis.json").read_text())
            for arm in ARMS:
                c = res["calibration"][arm]
                self.assertEqual(c["c3_argmax_mismatches"], 0)
                self.assertTrue(c["T"]["T"] > 0)
            self.assertIn("c7_hist_vs_seed42", res["seed_analysis"])
            self.assertIn("_base_hist_attribution", res["error_analysis"])
            perm = res["permutation"]["base-c0"]
            self.assertEqual(perm["n"], 24)
            self.assertIn(perm["label"], ("high_sensitivity", "low_sensitivity", "uncertain"))
            self.assertTrue(0.0 <= perm["consistency"] <= 1.0)
            self.assertTrue((out / "analysis.md").exists())

if __name__ == "__main__":
    unittest.main(verbosity=2)
