"""S1 结果分析入口（Issue #111）。

输入：从 Volume 下载的候选评分 JSONL（`{arm}-{split}.jsonl` 与
`{arm}-test2-perm.jsonl`）。输出：analysis.json / analysis.md / analysis-manifest.json。

口径全部来自 `docs/plan/009-s0-manifest.json` 与 `docs/plan/009-s1-manifest.json`：
- 主指标：同题配对 accuracy 差（每臂 − base-C0）；
- 校准：逐臂 T_a，calibration2 拟合、test2 只应用；top-label ECE 15 箱；
- 区间：family-clustered bootstrap 10,000，seed 20261018；
- 置换：按语义 key 还原的一致率/翻转率；
- 错误分析：S0 error_analysis_protocol v3，分母 1,659 families。

本脚本只读输入，不训练、不写 Volume。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from collections import Counter, defaultdict

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src/train"))
import metrics as M  # noqa: E402
import candidate_math as CM  # noqa: E402
import s1_spec as S  # noqa: E402

ARMS = ["base-c0", "hist", "seed42", "seed17", "seed29", "seed41"]
SEEDS = ["seed42", "seed17", "seed29", "seed41"]
SYNTHETIC = {"policy", "policy_v2", "routing_v2", "research_taxonomy_v21"}
PUBLIC = {"mnli", "boolq", "banking77"}


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_scores(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows, f"空文件：{path}"
    fps = {r.get("run_fingerprint") for r in rows}
    assert len(fps) == 1 and None not in fps, f"{path} 运行指纹不一致"
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), f"{path} id 重复"
    return rows


def index_by_id(rows: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in rows}


def align(rows_a: list[dict], rows_b: list[dict]) -> tuple[list[dict], list[dict]]:
    ia, ib = index_by_id(rows_a), index_by_id(rows_b)
    assert set(ia) == set(ib), (
        f"两文件 id 集不一致：仅 A {len(set(ia)-set(ib))}，仅 B {len(set(ib)-set(ia))}")
    ids = sorted(ia)
    return [ia[i] for i in ids], [ib[i] for i in ids]


def load_raw_splits(s1: dict, data_dir: Path | None = None) -> dict[str, dict[str, dict]]:
    """按 S1 manifest 的 SHA/行数加载原始切分，并逐条复算置换变体。"""
    files = s1["eval2_files"]
    out: dict[str, dict[str, dict]] = {}
    for key in ("dev2", "calibration2", "test2", "test2_permuted"):
        rel = files[key]["path"]
        p = (REPO / Path(rel)) if data_dir is None else Path(data_dir) / Path(rel).name
        raw = p.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == files[key]["sha256"], (
            f"原始切分 SHA 不符：{key} {p}")
        rows = [json.loads(l) for l in raw.decode("utf-8").split("\n") if l.strip()]
        assert len(rows) == files[key]["rows"], f"{key} 行数不符"
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids)), f"{key} id 重复"
        for r in rows:
            assert r.get("group_id"), f"{key} 行 {r['id']} 缺 group_id"
        out[key] = {r["id"]: r for r in rows}
    assert set(out["test2_permuted"]) == set(out["test2"])
    for rid, pr in out["test2_permuted"].items():
        orig = out["test2"][rid]
        exp = CM.permute_record(orig)
        assert pr["options"] == exp["options"], f"{rid} 置换选项与冻结规则不符"
        assert pr["answer"] == exp["answer"] and pr["answer_key"] == exp["answer_key"]
        assert pr["state"] == exp["state"] and pr["question"] == exp["question"]
        assert pr.get("group_id") == exp.get("group_id")
        assert pr.get("permutation") == exp.get("permutation")
    return out


def validate_score_record(r: dict, raw: dict) -> None:
    """逐题核对评分记录与原始记录/协议的自洽性。"""
    assert r["id"] == raw["id"] and r["source"] == raw["source"]
    assert r["group_id"] == raw["group_id"]
    CM.validate_options(raw["options"])
    labels = [o["label"] for o in raw["options"]]
    keys = [o["key"] for o in raw["options"]]
    assert r["option_labels"] == labels and r["option_keys"] == keys
    assert "logits_by_label" in r, f"{r['id']} 缺 logits_by_label（正式分析要求）"
    assert set(r["logits_by_label"]) == set(labels)
    assert set(r["probs_by_key"]) == set(keys)
    for lab, key in zip(labels, keys):
        assert abs(r["probs_by_key"][key] - r["probs_by_label"][lab]) < 1e-12
    assert r["gold_label"] == raw["answer"]
    assert r["gold_key"] == raw["answer_key"]
    logits = M._record_logits(r)
    probs = M.probs_from_logits(logits, 1.0)
    stored = np.array([r["probs_by_label"][lab] for lab in labels], dtype=np.float64)
    assert np.all(np.isfinite(stored)) and abs(stored.sum() - 1.0) < 1e-5
    assert np.allclose(stored, probs, atol=1e-5), f"{r['id']} probs 与 logits 不一致"
    pred = int(probs.argmax())
    gold = labels.index(r["gold_label"])
    assert r["pred_label"] == labels[pred] and r["pred_key"] == keys[pred]
    assert r["gold_key"] == keys[gold]
    assert r["correct"] == (r["pred_label"] == r["gold_label"])
    assert r.get("correct_key") == (r["pred_key"] == r["gold_key"])
    assert abs(r["p_gold"] - probs[gold]) < 1e-5
    assert r["scoring_version"] == "s1-candidate-v2"
    assert r["model"] == S.MODEL and r["revision"] == S.REV
    assert r["template_sha256"] == S.TEMPLATE_SHA256
    assert re.fullmatch(r"[0-9a-f]{40}", r["code_commit"]), "code_commit 不是 40 位 hex"
    fp_payload = {
        "data_sha256": r["data_sha256"], "arm": r["arm"], "tag": r["tag"],
        "split": r["split"], "model_kind": r["model_kind"], "art": r["art"],
        "adapter_sha256": r["adapter_sha256"],
        "adapter_config_sha256": r["adapter_config_sha256"],
        "source_sha256": r["source_sha256"],
        "template_sha256": r["template_sha256"], "model": r["model"],
        "revision": r["revision"], "scoring_version": r["scoring_version"], "n": 0,
    }
    assert r["run_fingerprint"] == CM.fingerprint(fp_payload), (
        f"{r['id']} run_fingerprint 与记录内容不一致")


def expected_adapters(s1: dict) -> dict[str, dict]:
    ref = s1.get("reference_arms") or {}
    runs = {r["run_tag"]: r for r in s1.get("runs", [])}
    hist = ref.get("hist") or {}
    assert hist.get("artifact", "/vol/artifacts/clean-r16") == "/vol/artifacts/clean-r16", (
        "reference_arms.hist.artifact 必须是 /vol/artifacts/clean-r16")
    out = {"base-c0": {"adapter_sha256": None, "adapter_config_sha256": None, "art": None},
           "hist": {"adapter_sha256": hist.get("adapter_sha256"),
                    "adapter_config_sha256": hist.get("adapter_config_sha256"),
                    "art": "/vol/artifacts/clean-r16"}}
    for seed in (42, 17, 29, 41):
        tag = f"seed{seed}"
        assert tag in runs, f"S1 manifest runs 缺 {tag}"
        assert runs[tag].get("artifact", f"/vol/s1/artifacts/{tag}") == f"/vol/s1/artifacts/{tag}", (
            f"{tag} 的 artifact 必须是 /vol/s1/artifacts/{tag}")
        assert runs[tag].get("adapter_config_sha256"), (
            f"S1 manifest runs[{tag}] 缺 adapter_config_sha256")
        out[tag] = {"adapter_sha256": runs[tag]["adapter_sha256"],
                    "adapter_config_sha256": runs[tag].get("adapter_config_sha256"),
                    "art": f"/vol/s1/artifacts/{tag}"}
    assert out["hist"]["adapter_sha256"], "S1 manifest reference_arms 缺 hist adapter SHA"
    assert out["hist"]["adapter_config_sha256"], "S1 manifest reference_arms 缺 hist config SHA"
    return out


def load_score_file(path: Path, arm: str, split_label: str, raw_map: dict[str, dict],
                    expect_sha: str, expect_rows: int, model_kind: str,
                    adapter: dict, expected_source: dict) -> list[dict]:
    # 正式评分调用 n=0（全量）；指纹重算按 n=0。
    rows = load_scores(path)
    assert len(rows) == expect_rows, f"{path} 行数 {len(rows)} ≠ {expect_rows}"
    assert set(r["id"] for r in rows) == set(raw_map), f"{path} id 集不符"
    for r in rows:
        assert r["arm"] == arm and r["split"] == split_label
        assert r["model_kind"] == model_kind
        assert r["adapter_sha256"] == adapter["adapter_sha256"]
        assert r["adapter_config_sha256"] == adapter["adapter_config_sha256"]
        assert r["art"] == adapter["art"], f"{r['id']} art 不符"
        assert r["data_sha256"] == expect_sha
        assert r["tag"] == f"{arm}-{split_label}", f"{r['id']} tag 不符"
        assert r.get("source_sha256") == expected_source, (
            f"{r['id']} 评分源码 SHA 与 S1 manifest 登记不符")
        validate_score_record(r, raw_map[r["id"]])
    return rows


def load_all_validated(scores_dir: Path, s1: dict,
                       data_dir: Path | None = None) -> dict[tuple[str, str], list[dict]]:
    raw = load_raw_splits(s1, data_dir)
    adapters = expected_adapters(s1)
    expected_source = {k: s1["eval_code_sha256"][k]
                       for k in ("eval_candidates.py", "candidate_math.py", "s1_spec.py")}
    splits = [("dev2", "dev2", "dev2"),
              ("calibration2", "calibration2", "calibration2"),
              ("test2", "test2", "test2"),
              ("test2-perm", "test2_permuted", "test2-perm")]
    out = {}
    for arm in ARMS:
        model_kind = "base" if arm == "base-c0" else "adapter"
        for split_key, raw_key, split_label in splits:
            p = scores_dir / f"{arm}-{split_key}.jsonl"
            assert p.exists(), f"缺 {p}"
            out[(arm, split_key)] = load_score_file(
                p, arm, split_label, raw[raw_key],
                s1["eval2_files"][raw_key]["sha256"],
                s1["eval2_files"][raw_key]["rows"], model_kind, adapters[arm],
                expected_source)
    return out


def source_metrics(records: list[dict], s0: dict, with_ci: bool = False) -> dict:
    by_src = defaultdict(list)
    for r in records:
        by_src[r["source"]].append(r)
    per_task = {}
    for src, rs in sorted(by_src.items()):
        labels = s0["semantic_label_sets"][src]
        per_task[src] = {
            "n": len(rs),
            "accuracy": float(np.mean([r["correct"] for r in rs])),
            "macro_f1": M.macro_f1(rs, labels),
            "per_class_recall": M.per_class_recall(rs, labels),
            "confusion": M.confusion(rs, labels).tolist(),
            "pred_label_counts": dict(Counter(r["pred_label"] for r in rs)),
            **M.summarize_records(rs),
        }
        if with_ci:
            groups = [r["group_id"] for r in rs]
            items = M.record_items(rs)
            correct = np.array([r["correct"] for r in rs], dtype=np.float64)
            per_task[src]["accuracy_ci"] = M.cluster_bootstrap_mean(correct, groups)
            per_task[src]["nll_ci"] = M.cluster_bootstrap_mean(items["nll"], groups)
            per_task[src]["brier_ci"] = M.cluster_bootstrap_mean(items["brier"], groups)
            per_task[src]["ece_ci"] = M.cluster_bootstrap_stat(
                np.zeros(len(rs)), groups,
                lambda idx: M.ece_from_items(items["pmax"][idx], items["correct"][idx]))
            per_task[src]["macro_f1_ci"] = M.cluster_bootstrap_stat(
                np.zeros(len(rs)), groups,
                lambda idx: M.macro_f1([rs[i] for i in idx], labels))
    out = dict(per_task)
    out["_task_equal_macro_f1"] = M.task_equal_weight(
        {s: v["macro_f1"] for s, v in per_task.items()})
    out["_sample_weighted_macro_f1"] = M.sample_weighted(
        {s: (v["macro_f1"], v["n"]) for s, v in per_task.items()})
    return out


def _ece_diff(records_a: list[dict], records_b: list[dict],
              T_a: float, T_b: float, groups: list[str]) -> dict:
    ia = M.record_items(records_a, T_a)
    ib = M.record_items(records_b, T_b)

    def stat(idx):
        return (M.ece_from_items(ia["pmax"][idx], ia["correct"][idx])
                - M.ece_from_items(ib["pmax"][idx], ib["correct"][idx]))

    return M.cluster_bootstrap_stat(np.zeros(len(groups)), groups, stat)


def calibration_analysis(scores: dict, s0: dict) -> dict:
    out = {}
    base_test = scores[("base-c0", "test2")]
    for arm in ARMS:
        cal = scores[(arm, "calibration2")]
        test = scores[(arm, "test2")]
        fit = M.fit_temperature_records(cal)
        raw = M.summarize_records(test, 1.0)
        calibrated = M.summarize_records(test, fit["T"])
        # C3：正标量温度不改变 argmax；逐题核验（不是抽样）。任何不一致即停止。
        mismatches = []
        for r in test:
            q = M.apply_temperature(r, fit["T"])
            if max(q, key=q.get) != r["pred_label"]:
                mismatches.append(r["id"])
        if mismatches:
            raise RuntimeError(
                f"{arm} 的 C3 逐题 argmax 不一致 {len(mismatches)} 题"
                f"（例 {mismatches[:5]}）——停止分析")
        raw_items = M.record_items(test, 1.0)
        cal_items = M.record_items(test, fit["T"])
        by_k = {}
        for k in sorted({len(r["option_labels"]) for r in test}):
            rs = [r for r in test if len(r["option_labels"]) == k]
            by_k[str(k)] = {"n": len(rs),
                            "raw": M.summarize_records(rs, 1.0),
                            "calibrated": M.summarize_records(rs, fit["T"])}
        # C2：相对 base-C0 与 calibrated−raw 的配对差 + family 聚类区间
        a, b = align(test, base_test)
        groups = [r["group_id"] for r in a]
        arm_raw = M.record_items(a, 1.0)
        arm_cal = M.record_items(a, fit["T"])
        base_raw = M.record_items(b, 1.0)
        deltas_vs_base = {}
        for metric in ("nll", "brier"):
            deltas_vs_base[f"{metric}_raw"] = M.cluster_bootstrap_paired_diff(
                arm_raw[metric], base_raw[metric], groups)
            deltas_vs_base[f"{metric}_cal"] = M.cluster_bootstrap_paired_diff(
                arm_cal[metric], base_raw[metric], groups)
        deltas_vs_base["ece_raw"] = _ece_diff(a, b, 1.0, 1.0, groups)
        deltas_vs_base["ece_cal"] = _ece_diff(a, b, fit["T"], 1.0, groups)
        cal_minus_raw = {
            "nll": M.cluster_bootstrap_paired_diff(
                arm_cal["nll"], arm_raw["nll"], groups),
            "brier": M.cluster_bootstrap_paired_diff(
                arm_cal["brier"], arm_raw["brier"], groups),
            "ece": _ece_diff(a, a, fit["T"], 1.0, groups),
        }
        out[arm] = {
            "T": fit, "raw": raw, "calibrated": calibrated,
            "c3_argmax_mismatches": len(mismatches),
            "raw_by_source": source_metrics(test, s0),
            # 分层校准指标：逐 source 分别 apply 同一个 T_a
            "calibrated_by_source": {
                src: M.summarize_records([r for r in test if r["source"] == src], fit["T"])
                for src in sorted({r["source"] for r in test})
            },
            "by_option_count": by_k,
            "selective": {
                "raw": M.selective_curve_items(raw_items["pmax"], raw_items["correct"]),
                "calibrated": M.selective_curve_items(cal_items["pmax"], cal_items["correct"]),
            },
            "deltas_vs_base": deltas_vs_base,
            "calibrated_minus_raw": cal_minus_raw,
        }
    return out


def seed_analysis(scores: dict, s0: dict) -> dict:
    base = index_by_id(scores[("base-c0", "test2")])
    out = {"per_seed": {}, "arms": {}}
    for arm in SEEDS:
        rows = scores[(arm, "test2")]
        ids = [r["id"] for r in rows]
        assert all(i in base for i in ids)
        arm_correct = np.array([r["correct"] for r in rows], dtype=np.float64)
        base_correct = np.array([base[i]["correct"] for i in ids], dtype=np.float64)
        groups = [r["group_id"] for r in rows]
        assert all(groups), "test2 缺 group_id"
        boot = M.cluster_bootstrap_paired_diff(arm_correct, base_correct, groups)
        out["per_seed"][arm] = {
            "accuracy": float(arm_correct.mean()),
            "base_accuracy_same_items": float(base_correct.mean()),
            "paired_diff": boot,
        }
        out["arms"][arm] = {"by_source": source_metrics(rows, s0)}
        base_rows = scores[("base-c0", "test2")]
        pa = {}
        for src in sorted({r["source"] for r in rows}):
            aa = [r for r in rows if r["source"] == src]
            bb = [r for r in base_rows if r["source"] == src]
            ida, idb = align(aa, bb)
            pa[src] = M.cluster_bootstrap_paired_diff(
                np.array([r["correct"] for r in ida], dtype=np.float64),
                np.array([r["correct"] for r in idb], dtype=np.float64),
                [r["group_id"] for r in ida])
        out["per_seed"][arm]["by_source_paired_diff"] = pa
    diffs = [out["per_seed"][a]["paired_diff"]["point"] for a in SEEDS]
    out["mean_diff"] = float(np.mean(diffs))
    out["sd_diff"] = float(np.std(diffs, ddof=1))
    out["positive_direction"] = all(d > 0 for d in diffs)
    out["zero_arm"] = any(d == 0 for d in diffs)
    out["no_best_seed"] = True
    # C7：hist vs seed42 同题配对（复现检查，不是 seed 方差）
    a, b = align(scores[("hist", "test2")], scores[("seed42", "test2")])
    s42 = np.array([r["correct"] for r in b], dtype=np.float64)
    hist = np.array([r["correct"] for r in a], dtype=np.float64)
    groups = [r["group_id"] for r in b]
    out["c7_hist_vs_seed42"] = {
        "paired_diff_seed42_minus_hist": M.cluster_bootstrap_paired_diff(s42, hist, groups),
        "note": ("复现检查：差异含历史未受控的 LoRA 初始化 + 环境/非确定性，"
                 "不可分解；不是 seed 方差，也不单独归因于环境。"),
    }
    return out


def permutation_analysis(scores: dict, s0: dict) -> dict:
    out = {}
    for arm in ARMS:
        orig = scores[(arm, "test2")]
        perm = scores[(arm, "test2-perm")]
        a, b = align(orig, perm)
        assert all(x["gold_key"] == y["gold_key"] for x, y in zip(a, b))
        consistent = np.array([x["pred_key"] == y["pred_key"] for x, y in zip(a, b)],
                              dtype=np.float64)
        groups = [x["group_id"] for x in a]
        boot = M.cluster_bootstrap_mean(consistent, groups)
        flip = 1.0 - float(consistent.mean())
        consistency = float(consistent.mean())
        ci_lo, ci_hi = boot["ci95"]
        flips = int(len(consistent) - int(consistent.sum()))
        label = M.permutation_label(len(consistent), flips, ci_lo)
        out[arm] = {
            "n": len(a), "consistency": consistency, "flip_rate": flip,
            "consistency_ci95": boot["ci95"], "label": label,
            "by_source": {
                src: {
                    "n": len([x for x in a if x["source"] == src]),
                    "consistency": float(np.mean(
                        [x["pred_key"] == y["pred_key"] for x, y in zip(a, b)
                         if x["source"] == src])),
                } for src in sorted({x["source"] for x in a})
            },
        }
    return out


def _constant_threshold(source: str, n: int) -> int:
    if source == "policy":
        return n  # 3/3
    if source in ("policy_v2", "routing_v2"):
        return math.ceil(5 * n / 6)
    if source == "research_taxonomy_v21":
        return math.ceil(7 * n / 8)
    raise ValueError(source)


def error_analysis(scores: dict, s0: dict) -> dict:
    """按 S0 error_analysis_protocol v3 归类；返回逐臂计数与案例。"""
    total_families = s0["error_analysis_protocol"]["denominator_n_test2_families"]
    # 置换翻转的 family（任一 sibling 语义预测翻转）
    flip_families = {}
    for arm in ARMS:
        a, b = align(scores[(arm, "test2")], scores[(arm, "test2-perm")])
        flip_families[arm] = {
            x["group_id"] for x, y in zip(a, b) if x["pred_key"] != y["pred_key"]}

    out = {}
    for arm in ARMS:
        rows = scores[(arm, "test2")]
        by_family = defaultdict(list)
        for r in rows:
            by_family[r["group_id"]].append(r)
        assert len(by_family) == total_families, (
            f"test2 family 数 {len(by_family)} ≠ S0 锁定分母 {total_families}")
        # 公开单行 family 的格子众数（排除自身）
        grid = defaultdict(list)
        for r in rows:
            if r["source"] in PUBLIC:
                grid[(r["source"], len(r["option_labels"]))].append(r)
        grid_mode = {}
        for key, rs in grid.items():
            if len(rs) < 20:
                continue
            counts = Counter(r["pred_key"] for r in rs)
            grid_mode[key] = (counts, len(rs))
        classes = Counter()
        cases = defaultdict(list)
        for gid, rs in sorted(by_family.items()):
            src = rs[0]["source"]
            n_opt = len(rs[0]["option_labels"])
            errors = [r for r in rs if not r["correct"]]
            if not errors:
                continue
            kind = None
            if any(r["pred_key"] is None for r in rs):
                kind = "状态解析失败"
            elif gid in flip_families[arm]:
                kind = "选项顺序敏感"
            elif src in SYNTHETIC:
                counts = Counter(r["pred_key"] for r in rs)
                mode_n = counts.most_common(1)[0][1] if counts else 0
                if mode_n >= _constant_threshold(src, len(rs)):
                    kind = "恒常策略"
            else:
                key = (src, n_opt)
                if key in grid_mode:
                    counts, total = grid_mode[key]
                    # 排除本 family 自身
                    own = Counter(r["pred_key"] for r in rs)
                    counts2 = counts - own
                    total2 = total - len(rs)
                    if total2 > 0:
                        mode, n_mode = counts2.most_common(1)[0]
                        if (errors[0]["pred_key"] == mode
                                and n_mode / total2 >= 0.90):
                            kind = "恒常策略"
            if kind is None:
                if any(r["pred_key"] is not None and r["pred_key"] != r["gold_key"]
                       for r in errors):
                    kind = "标签错误"
                else:
                    kind = "其他"
            classes[kind] += 1
            if len(cases[kind]) < 10:
                # 冻结规则：每 family 取 id 最小的一行作代表（不是最小错误行）。
                rep = min(rs, key=lambda r: r["id"])
                cases[kind].append({
                    "group_id": gid, "source": src, "id": rep["id"],
                    "pred_key": rep["pred_key"], "gold_key": rep["gold_key"],
                    "pred_label": rep["pred_label"], "gold_label": rep["gold_label"],
                    "correct": rep["correct"],
                    "error_ids": sorted(r["id"] for r in errors),
                })
        out[arm] = {
            "total_families": total_families,
            "error_families": sum(classes.values()),
            "by_class": dict(classes),
            "cases": dict(cases),
        }

    # 四臂共同错误 vs 臂间分歧（只对 seed 四臂）
    seed_families = {}
    for arm in SEEDS:
        by_family = defaultdict(list)
        for r in scores[(arm, "test2")]:
            by_family[r["group_id"]].append(r)
        seed_families[arm] = {g: any(not r["correct"] for r in rs)
                              for g, rs in by_family.items()}
    shared = sum(1 for g in seed_families[SEEDS[0]]
                 if all(seed_families[a].get(g, False) for a in SEEDS))
    divergent = sum(1 for g in seed_families[SEEDS[0]]
                    if any(seed_families[a].get(g, False) for a in SEEDS)
                    and not all(seed_families[a].get(g, False) for a in SEEDS))
    out["_seed_error_structure"] = {"shared_all_four": shared,
                                    "divergent": divergent}
    # base/hist 独错与组合归属（R4 eval P2-11）
    membership = {}
    for arm in ARMS:
        by_family = defaultdict(list)
        for r in scores[(arm, "test2")]:
            by_family[r["group_id"]].append(r)
        membership[arm] = {g: any(not r["correct"] for r in rs)
                           for g, rs in by_family.items()}
    families = set(membership["base-c0"])
    for arm in ARMS:
        assert set(membership[arm]) == families, f"{arm} family 集不一致"
    seeds = SEEDS
    out["_base_hist_attribution"] = {
        "base_only": sum(1 for g in families
                         if membership["base-c0"][g] and not membership["hist"][g]
                         and not any(membership[a][g] for a in seeds)),
        "hist_only": sum(1 for g in families
                         if membership["hist"][g] and not membership["base-c0"][g]
                         and not any(membership[a][g] for a in seeds)),
        "base_hist_both_wrong_seeds_right": sum(
            1 for g in families
            if membership["base-c0"][g] and membership["hist"][g]
            and not any(membership[a][g] for a in seeds)),
        "seeds_all_wrong_base_hist_right": sum(
            1 for g in families
            if all(membership[a][g] for a in seeds)
            and not membership["base-c0"][g] and not membership["hist"][g]),
    }
    return out


def _ci(d: dict) -> str:
    return f"{d['point']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"


def write_markdown(result: dict, path: Path) -> None:
    lines = ["# #111 S1 结果分析（自动生成）", ""]
    lines += ["## 校准（test2，逐臂 T_a；calibration2 拟合）", "",
              "| arm | T | T 触边界 | raw NLL | raw Brier | raw ECE | cal NLL | cal Brier | cal ECE | C3 mismatch |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        c = result["calibration"][arm]
        r, q = c["raw"], c["calibrated"]
        lines.append(
            f"| {arm} | {c['T']['T']:.4f} | {c['T']['at_boundary']} | {r['nll']:.4f} | "
            f"{r['brier']:.4f} | {r['ece_top_label']:.4f} | {q['nll']:.4f} | "
            f"{q['brier']:.4f} | {q['ece_top_label']:.4f} | {c['c3_argmax_mismatches']} |")
    lines += ["", "## C2 概率质量配对差（test2；family-clustered 10,000）", "",
              "| arm | NLL raw−base | NLL cal−base | Brier raw−base | Brier cal−base | ECE raw−base | ECE cal−base |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        d = result["calibration"][arm]["deltas_vs_base"]
        lines.append(f"| {arm} | {_ci(d['nll_raw'])} | {_ci(d['nll_cal'])} | "
                     f"{_ci(d['brier_raw'])} | {_ci(d['brier_cal'])} | "
                     f"{_ci(d['ece_raw'])} | {_ci(d['ece_cal'])} |")
    lines += ["", "| arm | NLL cal−raw | Brier cal−raw | ECE cal−raw |",
              "|---|---:|---:|---:|"]
    for arm in ARMS:
        d = result["calibration"][arm]["calibrated_minus_raw"]
        lines.append(f"| {arm} | {_ci(d['nll'])} | {_ci(d['brier'])} | {_ci(d['ece'])} |")
    lines += ["", "## 选择性预测（只报告，不调回退阈值）", "",
              "| arm | 覆盖 | raw acc | cal acc |", "|---|---:|---:|---:|"]
    for arm in ARMS:
        sel = result["calibration"][arm]["selective"]
        raw_by_cov = {x["coverage"]: x["accuracy"] for x in sel["raw"]}
        cal_by_cov = {x["coverage"]: x["accuracy"] for x in sel["calibrated"]}
        for cov in (1.0, 0.8, 0.5):
            rc = min(raw_by_cov, key=lambda x: abs(x - cov))
            lines.append(f"| {arm} | {rc:.2f} | {raw_by_cov[rc]:.4f} | "
                         f"{cal_by_cov[rc]:.4f} |")
    lines += ["", "## dev2（健康门覆盖率口径）", "",
              "| arm | n | accuracy | coverage | mean p_gold |", "|---|---:|---:|---:|---:|"]
    for arm, v in result["dev2"].items():
        lines.append(f"| {arm} | {v['n']} | {v['accuracy']:.4f} | "
                     f"{v['coverage']:.4f} | {v['mean_p_gold']:.4f} |")
    lines += ["", "## 任务语义指标（test2；任务内类别等权）", "",
              "| arm | task | n | accuracy [95% CI] | macro-F1 | macro recall（有真实样本类） |",
              "|---|---|---:|---|---:|---:|"]
    for arm in ARMS:
        for src, v in result["by_source"][arm].items():
            if src.startswith("_"):
                continue
            ci = v.get("accuracy_ci")
            acc = (f"{v['accuracy']:.4f} [{ci['ci95'][0]:.4f}, {ci['ci95'][1]:.4f}]"
                   if ci else f"{v['accuracy']:.4f}")
            mf = v.get("macro_f1_ci")
            mfs = (f"{v['macro_f1']:.4f} [{mf['ci95'][0]:.4f}, {mf['ci95'][1]:.4f}]"
                   if mf else f"{v['macro_f1']:.4f}")
            lines.append(f"| {arm} | {src} | {v['n']} | {acc} | {mfs} | "
                         f"{v['per_class_recall'].get('macro_recall_defined')} |")
    lines += ["", "## 选项数分层（seed42）", "",
              "| 选项数 | n | raw acc | raw ECE | cal acc | cal ECE |",
              "|---|---:|---:|---:|---:|---:|"]
    for k, v in result["calibration"]["seed42"]["by_option_count"].items():
        lines.append(f"| {k} | {v['n']} | {v['raw']['accuracy']:.4f} | "
                     f"{v['raw']['ece_top_label']:.4f} | {v['calibrated']['accuracy']:.4f} | "
                     f"{v['calibrated']['ece_top_label']:.4f} |")
    lines.append("")
    lines.append("六臂的完整选项数分层、混淆矩阵、raw/cal 任务指标与 per-source 配对差见 "
                 "`analysis.json`（`calibration.*.by_option_count`、`by_source.*.*.confusion`、"
                 "`by_source.*.*.{accuracy,nll,brier,ece,macro_f1}_ci`、"
                 "`calibration.*.raw_by_source`、`seed_analysis.*.by_source_paired_diff`、"
                 "`position_preference`）。")
    lines += ["", "## 多种子稳定性（test2，同题配对 vs base-C0）", "",
              "| seed | accuracy | paired diff | 95% CI |", "|---|---:|---:|---|"]
    for arm, v in result["seed_analysis"]["per_seed"].items():
        d = v["paired_diff"]
        lines.append(f"| {arm} | {v['accuracy']:.4f} | {d['point']:+.4f} | "
                     f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
    s = result["seed_analysis"]
    lines += ["", f"- 四臂均值差 {s['mean_diff']:+.4f}；SD {s['sd_diff']:.4f}；"
                  f"正向性（四臂均 >0）：**{s['positive_direction']}**"
                  f"{'（存在 0 值，视为未通过）' if s['zero_arm'] else ''}。",
              "- 3 个自由度：真实 SD 的 95% 区间约观测 SD 的 0.57–3.73 倍；"
              "训练间差异与样本抽样不确定性分开报；未挑最佳 seed。"]
    c7 = s.get("c7_hist_vs_seed42")
    if c7:
        d = c7["paired_diff_seed42_minus_hist"]
        lines.append(f"- C7 复现检查 seed42−hist：{d['point']:+.4f} "
                     f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]；"
                     f"{c7['note']}")
    lines += ["", "## seed 臂按 source 的配对 accuracy 差（vs base-C0）", "",
              "| seed | source | diff [95% CI] |", "|---|---|---|"]
    for arm in SEEDS:
        for src, d in result["seed_analysis"]["per_seed"][arm].get(
                "by_source_paired_diff", {}).items():
            lines.append(f"| {arm} | {src} | {d['point']:+.4f} "
                         f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
    lines += ["", "## 选项置换（test2 原题 vs 变体，按语义 key）", "",
              "| arm | n | 一致率 | 翻转率 | 95% CI | 标签 |",
              "|---|---:|---:|---:|---|---|"]
    for arm, v in result["permutation"].items():
        lines.append(f"| {arm} | {v['n']} | {v['consistency']:.4f} | {v['flip_rate']:.4f} | "
                     f"[{v['consistency_ci95'][0]:.4f}, {v['consistency_ci95'][1]:.4f}] | "
                     f"{v['label']} |")
    lines += ["", "## 位置偏好（test2 预测字母计数）", ""]
    for arm, counts in result["position_preference"].items():
        top = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])[:5])
        lines.append(f"- {arm}: {top}")
    lines += ["", "## 错误与行为分析（family 单位）", "",
              "| arm | 错误 family | 解析失败 | 顺序敏感 | 恒常策略 | 标签错误 | 其他 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, v in result["error_analysis"].items():
        if not isinstance(v, dict) or "by_class" not in v:
            continue
        c = v["by_class"]
        lines.append(f"| {arm} | {v['error_families']}/{v['total_families']} | "
                     f"{c.get('状态解析失败', 0)} | {c.get('选项顺序敏感', 0)} | "
                     f"{c.get('恒常策略', 0)} | {c.get('标签错误', 0)} | "
                     f"{c.get('其他', 0)} |")
    se = result["error_analysis"].get("_seed_error_structure")
    if se:
        lines += ["", f"- 四 seed 共同错误 family：{se['shared_all_four']}；臂间分歧：{se['divergent']}。"]
    bh = result["error_analysis"].get("_base_hist_attribution")
    if bh:
        lines += [f"- base/hist 归属：base-only {bh['base_only']}；hist-only {bh['hist_only']}；"
                  f"base+hist 同错而四 seed 全对 {bh['base_hist_both_wrong_seeds_right']}；"
                  f"四 seed 全错而 base/hist 全对 {bh['seeds_all_wrong_base_hist_right']}。"]
    lines += ["", "## 限制", "",
              "- 主指标为候选评分；温度是正标量，不改变 argmax，不主张校准提升准确率。",
              "- 置换一致率只支持「对选项顺序扰动的敏感性」，不主张语义机制。",
              "- test2 无 OOD 臂；历史 81.13% 分表并列，不作差值归因。",
              "- 覆盖率为构造结果；选择性曲线只报告，不据 test2 选择阈值。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="S1 结果分析")
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--data-dir", default=None,
                    help="原始切分目录；默认按 S1 manifest 的仓内路径")
    ap.add_argument("--s0", default=str(REPO / "docs/plan/009-s0-manifest.json"))
    ap.add_argument("--s1", default=str(REPO / "docs/plan/009-s1-manifest.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true", help="允许覆盖既有分析输出")
    args = ap.parse_args()
    scores_dir = Path(args.scores_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("analysis.json", "analysis.md", "analysis-manifest.json"):
        if (out_dir / name).exists() and not args.force:
            raise SystemExit(f"输出已存在，拒绝覆盖：{out_dir/name}（用 --force）")
    s0 = json.loads(Path(args.s0).read_text(encoding="utf-8"))
    s1 = json.loads(Path(args.s1).read_text(encoding="utf-8"))
    scores = load_all_validated(
        scores_dir, s1, Path(args.data_dir) if args.data_dir else None)

    # 同题集身份：每臂 test2 的 data_sha256 必须一致
    test_shas = {scores[(a, "test2")][0]["data_sha256"] for a in ARMS}
    assert len(test_shas) == 1, f"test2 数据 SHA 不一致：{test_shas}"
    for arm in ARMS:
        assert {r["id"] for r in scores[(arm, "test2")]} == {
            r["id"] for r in scores[("base-c0", "test2")]}, f"{arm} test2 id 集不一致"

    cal = calibration_analysis(scores, s0)
    seed = seed_analysis(scores, s0)
    perm = permutation_analysis(scores, s0)
    err = error_analysis(scores, s0)

    result = {
        "inputs": {f"{a}-{s}": sha_file(scores_dir / f"{a}-{s}.jsonl")
                   for a in ARMS for s in ("dev2", "calibration2", "test2", "test2-perm")},
        "s0_manifest_sha256": sha_file(Path(args.s0)),
        "s1_manifest_sha256": sha_file(Path(args.s1)),
        "dev2": {
            arm: {"n": len(scores[(arm, "dev2")]),
                  "accuracy": float(np.mean([r["correct"] for r in scores[(arm, "dev2")]])),
                  "coverage": float(np.mean([r["pred_key"] is not None
                                             for r in scores[(arm, "dev2")]])),
                  "mean_p_gold": float(np.mean([r["p_gold"] for r in scores[(arm, "dev2")]]))}
            for arm in ARMS
        },
        "by_source": {arm: source_metrics(scores[(arm, "test2")], s0, with_ci=True)
                      for arm in ARMS},
        "position_preference": {
            arm: dict(Counter(r["pred_label"] for r in scores[(arm, "test2")]))
            for arm in ARMS
        },
        "calibration": cal,
        "seed_analysis": seed,
        "permutation": perm,
        "error_analysis": err,
        "notes": [
            "主指标是候选评分的同题配对 accuracy 差；温度校准不改变 argmax。",
            "训练间差异（4 臂）与样本抽样不确定性分开报；不挑最佳 seed。",
            "置换一致率只支持「对选项顺序扰动的敏感性」，不主张语义机制。",
            "错误分析分母 1,659 families；案例在结果冻结后按 group_id 升序取前 10。",
        ],
    }
    (out_dir / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, out_dir / "analysis.md")
    manifest = {
        "created": "2026-10-18",
        "scores_dir": str(scores_dir),
        "inputs": result["inputs"],
        "s0_manifest_sha256": result["s0_manifest_sha256"],
        "s1_manifest_sha256": result["s1_manifest_sha256"],
        "bootstrap": {"n": M.N_BOOT, "seed": M.BOOT_SEED},
        "analysis_code_sha256": {
            "s1_analysis.py": sha_file(Path(__file__)),
            "metrics.py": sha_file(Path(M.__file__)),
        },
        "outputs": ["analysis.json", "analysis.md"],
        "outputs_sha256": {
            "analysis.json": sha_file(out_dir / "analysis.json"),
            "analysis.md": sha_file(out_dir / "analysis.md"),
        },
    }
    (out_dir / "analysis-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "n_arms": len(ARMS)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
