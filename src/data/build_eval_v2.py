"""S0：按 plan/009 A2′ 生成评测侧三切分（dev2 / calibration2 / test2）。

只生成评测侧数据；训练集仍用历史 `train-clean`。输入：
- vendored patched 上游生成器（`--patched`，由 `src/data/upstream_patches/0001-rejev-s0.patch`
  应用到 `resources/tev1` 的副本得到）；
- 冻结的历史 statehash / family 清单（`src/data/` 下的 `frozen/` 子目录；不随本发布集分发）。

产出（`--output` 指定，建议 `data/eval2-<seed>/`，不入库）：
- records/{dev2,calibration2,test2}.jsonl
- manifest.json（配额、覆盖矩阵、审计 1–5、RNG 协议、判读门槛、文件 SHA）
- 失败时 FAILED.json 并以非零码退出（审计 1–5 失败即停）。

用法（需要 datasets/transformers；用上游 venv，解释器路径按本机布置）：
    python src/data/build_eval_v2.py \\
        --seed rejev-eval2-probe-20261017 --tag probe \\
        --patched <patched-dir>
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import string
import sys

REPO = Path(__file__).resolve().parents[2]
DEFAULT_FROZEN = REPO / "src/data/frozen"
PATCH = REPO / "src/data/upstream_patches/0001-rejev-s0.patch"
SPLITS = ("dev2", "calibration2", "test2")
PUBLIC = ("mnli", "boolq", "banking77")
PUBLIC_SPLITS = {
    "mnli": ("validation_matched", "validation_mismatched"),
    "boolq": ("validation",),
    "banking77": ("test",),
}
# 配额按 train-clean 的源比例缩放（合成 67.96% / 公开 32.04%），再取各组粒度
# （policy_v2/routing 6、policy 3、research 8、公开源 1）；总数严格 5,800。
QUOTAS = {
    "dev2": {"policy_v2": 174, "routing_v2": 84, "policy": 24,
             "research_taxonomy_v21": 56, "mnli": 74, "boolq": 45, "banking77": 43},
    "calibration2": {"policy_v2": 522, "routing_v2": 264, "policy": 66,
                     "research_taxonomy_v21": 168, "mnli": 218, "boolq": 131, "banking77": 131},
    "test2": {"policy_v2": 1326, "routing_v2": 666, "policy": 165,
              "research_taxonomy_v21": 424, "mnli": 554, "boolq": 332, "banking77": 333},
}
TRAINING_SEEDS = (42, 17, 29, 41)
# S0 试生成后锁定的最小 family 数（每个 source × 可产生的选项数格子）。
# 观测最小值：dev2 7 / calibration2 21 / test2 53；阈值留出余量。
MIN_FAMILIES = {"dev2": 5, "calibration2": 15, "test2": 40}
EXPECTED_CELLS = {
    "mnli": (3,), "boolq": (2,), "banking77": (4, 6, 8),
    "policy_v2": (3,), "policy": (3,), "routing_v2": (5,),
    "research_taxonomy_v21": (8, 12, 24),
}
TRAIN_CLEAN = REPO / "data/clean/train-clean.jsonl"
PATCH_BARRIER_RESULT = REPO / "docs/plan/009-patch-barrier.json"
RENDER_CONTRACT_RESULT = REPO / "docs/plan/009-render-contract.json"
REBUILD_EVIDENCE_RESULT = REPO / "docs/plan/009-rebuild-evidence.json"
SCAFFOLD_MARKERS = ("{main}", "{method}", "{outcome}", "{other}",
                    "TODO", "Generate ", "<|")
RENDER_CONTRACT = {
    "model": "openbmb/MiniCPM5-2B",
    "revision": "12a3808a956f869c767195e9266b59c4d21d92e2",
    "tokenizer_note": "MiniCPM5-2B tokenizer（非上游 tev1 的 Qwen tokenizer）",
    "template": "apply_chat_template(..., enable_thinking=False)",
    "assistant_suffix": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "assistant_end": "<|im_end|>",
    "eos_token": "</s>",
    "eos_token_id": 1,
    "im_end_token_id": 130073,
    "letters": "ABCDEFGHIJKLMNOPQRSTUVWX",
    "label_token_ids": {
        c: [54 + i] for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWX")
    },
    "max_tokens": 2048,
    "generation_config_eos": [1, 130073],
    "sampling": "推理须显式 temperature=0 / do_sample=False",
    "measured_env": {"python": "3.14.3", "transformers": "5.6.2"},
    "label_ids_source": "实测 2026-10-18（AutoTokenizer, local_files_only=True）；输出 docs/plan/009-render-contract.json",
    "sources": ["src/align/v1a-report.md", "src/data/render.py"],
    "note": "S0 只登记渲染契约；S1 渲染时须逐条断言字母单 token、序列长度 ≤ max_tokens、add_special_tokens=False。",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def import_patched(patched: Path, seed: str, frozen_hashes: Path):
    """在导入前钉死环境；返回 patched 模块与上游 v1/v2/v2.1。"""
    os.environ["OPEN_JEV_ROOT"] = str(patched)
    os.environ.setdefault("HF_HOME", str(REPO / "resources" / "tev1" / ".cache" / "huggingface"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["REJEV_SEED_V1"] = seed
    os.environ["REJEV_SEED_V2"] = seed
    os.environ["REJEV_SEED_V21"] = seed
    os.environ["REJEV_FROZEN_STATEHASHES"] = str(frozen_hashes)
    sys.path.insert(0, str(patched))
    import build_dataset as v1  # noqa: E402
    import build_v2 as b  # noqa: E402
    import build_v21 as b21  # noqa: E402
    assert str(v1.SEED) == seed and b.SEED == seed and b21.SEED == seed, (
        v1.SEED, b.SEED, b21.SEED, seed)
    return v1, b, b21


def dedup_rows(rows: list[dict]) -> list[dict]:
    """按 group（= 归一化 state 哈希）去重，保留 official_splits 顺序的首见行。"""
    seen, unique = set(), []
    for row in rows:
        if row["group"] not in seen:
            seen.add(row["group"])
            unique.append(row)
    return unique


def filter_frozen(rows: list[dict], frozen: set[str], statehash_fn) -> list[dict]:
    """扣除冻结历史 statehash；与上游 `seen=set(forbidden)` 路径等价。"""
    return [row for row in rows if statehash_fn(row) not in frozen]


def load_public_pool(name: str, lock: dict, frozen: set[str], b) -> tuple[list[dict], dict]:
    """读取官方 held-out、按归一化 state 去重、扣冻结历史后返回池。"""
    from datasets import load_dataset
    source = lock["datasets"][name]
    ds = load_dataset(source["repo"], name=source["config"], revision=source["revision"])
    rows = []
    for split in PUBLIC_SPLITS[name]:
        rows.extend(b.v1.prepare_rows(name, ds[split], split, source))
    unique = dedup_rows(rows)
    hits = sum(1 for row in unique if b.statehash(row) in frozen)
    kept = filter_frozen(unique, frozen, b.statehash)
    audit = {"raw_rows": len(rows), "unique_states": len(unique),
             "frozen_statehash_hits": hits, "available": len(kept),
             "official_splits": list(PUBLIC_SPLITS[name]),
             "dedup_policy": "first-wins in official_splits order"}
    if name == "banking77":
        audit["label_names"] = list(ds[PUBLIC_SPLITS[name][0]].features["label"].names)
    return kept, audit


def sample_public(pools: dict[str, list[dict]], b, lock: dict, seed: str) -> dict[str, dict]:
    """按 dev2 → calibration2 → test2 顺序抽取；抽过的 group 从池中移除，保证切分互斥。"""
    out = {s: {} for s in SPLITS}
    for name in PUBLIC:
        pool = list(pools[name])
        for split in SPLITS:
            n = QUOTAS[split][name]
            chosen = b.v1.balanced_take(pool, n, f"{seed}:{name}:{split}:public")
            groups = {row["group"] for row in chosen}
            pool = [row for row in pool if row["group"] not in groups]
            out[split][name] = [
                b.v1.public_record(name, row, split, lock["datasets"][name]) for row in chosen
            ]
    return out


def research_rows(b21, split: str, n: int, seed: str, options, question) -> list[dict]:
    """按 family 整组取 research_v21；跨 topic 轮转，避免 dev2 只覆盖前几个 topic。"""
    assert n % 8 == 0
    target_groups = n // 8
    units = math.ceil(target_groups / 24)
    rows = b21.build_research(split, units, options, question)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    by_topic: dict[str, list[str]] = defaultdict(list)
    for gid in groups:
        topic = gid.rsplit(":", 2)[-2]
        by_topic[topic].append(gid)
    for topic in by_topic:
        by_topic[topic].sort(key=lambda g: int(g.rsplit(":", 1)[-1]))
    topics = sorted(by_topic)
    random.Random(int(b21.v.digest(f"{seed}:{split}:research-order"), 16)).shuffle(topics)
    selected: list[str] = []
    for idx in range(units):
        for topic in topics:
            if idx < len(by_topic[topic]):
                selected.append(by_topic[topic][idx])
                if len(selected) == target_groups:
                    break
        if len(selected) == target_groups:
            break
    assert len(selected) == target_groups, (split, len(selected), target_groups)
    out = [row for gid in selected for row in groups[gid]]
    assert len(out) == n, (split, len(out), n)
    # build_research 对非 train/dev 的 split 一律用 TEST_TEMPLATES；先断言再登记。
    for row in out:
        assert row["provenance"]["template"] in b21.TEST_TEMPLATES, row["id"]
        row["provenance"]["template_set"] = "test"
    return out


def build_rows(v1, b, b21, lock: dict, seed: str, cfg: dict,
               frozen: set[str]) -> dict[str, list[dict]]:
    # 直接调用 patch 后的屏障函数：既用到冻结清单，也验证 patch 真的生效。
    assert b.frozen_statehashes() == frozen, "patched frozen_statehashes 与清单不一致"
    pools, public_audits = {}, {}
    for name in PUBLIC:
        pools[name], public_audits[name] = load_public_pool(name, lock, frozen, b)
        used = sum(QUOTAS[split][name] for split in SPLITS)
        public_audits[name]["consumed_by_delivery"] = used
        public_audits[name]["remaining_after_delivery"] = public_audits[name]["available"] - used
        # probe / final-check / final 共三次从同一 held-out 池抽取；三次可能重叠，
        # 按 3×used 扣减得到 remaining 的下界（即 consumption 的上界）。
        public_audits[name]["draws_from_pool"] = 3
        # R4 P3-1：把三次抽取的名字登记出来；final-check2 与 final 逐字节一致，
        # 不是一次独立抽取，故 distinct 抽取仍是 3 次。
        public_audits[name]["pool_draws_registered"] = ["probe", "final-check", "final"]
        public_audits[name]["pool_draws_note"] = (
            "final-check2 ≡ final（同一脚本版本逐字节一致），不重复计为第 4 次抽取")
        public_audits[name]["remaining_after_all_consumed_lower_bound"] = (
            public_audits[name]["available"] - 3 * used)
    public = sample_public(pools, b, lock, seed)
    out = {}
    for split in SPLITS:
        q = QUOTAS[split]
        rows = []
        rows += b.synthetic(split, q["policy_v2"], routing=False)
        rows += b.synthetic(split, q["routing_v2"], routing=True)
        rows += v1.policy_records(split, q["policy"])
        rows += research_rows(b21, split, q["research_taxonomy_v21"], seed,
                              cfg["options"], cfg["question"])
        for name in PUBLIC:
            rows += public[split][name]
        assert len(rows) == sum(q.values()), (split, len(rows), sum(q.values()))
        ids = [row["id"] for row in rows]
        assert len(ids) == len(set(ids)), f"{split} 出现重复 id"
        random.Random(int(v1.digest(f"{seed}:{split}:output"), 16)).shuffle(rows)
        out[split] = rows
    return {"rows": out, "public_audits": public_audits}


def partition_audit(rows: dict[str, list[dict]], b, frozen_hashes: set[str],
                    frozen_families: set[tuple[str, str]]) -> dict:
    state_sets = {s: {b.statehash(row) for row in rs} for s, rs in rows.items()}
    family_sets = {s: {(row["source"], row["group_id"]) for row in rs} for s, rs in rows.items()}
    pairwise = {}
    for i, a in enumerate(SPLITS):
        for c in SPLITS[i + 1:]:
            pairwise[f"{a}_x_{c}"] = {
                "statehash_overlap": len(state_sets[a] & state_sets[c]),
                "family_overlap": len(family_sets[a] & family_sets[c]),
            }
    history = {
        "frozen_statehashes": len(frozen_hashes),
        "frozen_families": len(frozen_families),
        "statehash_overlap": {s: len(state_sets[s] & frozen_hashes) for s in SPLITS},
        "family_overlap": {s: len(family_sets[s] & frozen_families) for s in SPLITS},
        "exam_family_check": "vacuous（exam 无 group_id，由 statehash 检查覆盖）",
    }
    ok = (all(v["statehash_overlap"] == 0 and v["family_overlap"] == 0
              for v in pairwise.values())
          and all(v == 0 for v in history["statehash_overlap"].values())
          and all(v == 0 for v in history["family_overlap"].values()))
    return {"pairwise": pairwise, "history": history, "gate": "PASS" if ok else "FAIL"}


def _position_balance(rs: list[dict]) -> dict:
    """按 source 报告字母位置偏离；只作诊断，不进门。"""
    n = len(rs)
    counts = Counter(row["answer"] for row in rs)
    expected = Counter()
    for row in rs:
        for option in row["options"]:
            expected[option["label"]] += 1.0 / len(row["options"])
    letters = sorted({label for label, value in expected.items() if value > 0})
    chi2 = sum((counts[label] - expected[label]) ** 2 / expected[label]
               for label in letters if expected[label] > 0)
    df = len(letters) - 1
    max_letter = max(counts, key=counts.get)
    observed = counts[max_letter] / n
    exp_share = expected[max_letter] / n
    z = p = None
    if df == 1 and expected[max_letter] > 0:
        se = math.sqrt(expected[max_letter] * (1 - expected[max_letter] / n))
        if se:
            z = (counts[max_letter] - expected[max_letter]) / se
            p = math.erfc(abs(z) / math.sqrt(2))
    return {
        "n": n,
        "max_letter": max_letter,
        "observed_share": round(observed, 6),
        "expected_share": round(exp_share, 6),
        "delta_pp": round((observed - exp_share) * 100, 3),
        "chi2": round(chi2, 6),
        "df": df,
        "z": round(z, 3) if z is not None else None,
        "p": round(p, 6) if p is not None else None,
    }


def answer_position_audit(rows: dict[str, list[dict]]) -> dict:
    """审计 3：报告答案字母分布，并核对 family 内的语义答案组合。

    字母可因逐题独立置换而巧合相同；**构造期预置**的证据是 family 内
    `answer_key` 组合符合生成器的变体设计，而不是字母必须不同。
    `position_balance` 只报告；不并入失败即停（与 §3.3 第 5 项同款纪律）。
    """
    out = {}
    failures = []
    construction_failures = []
    for split, rs in rows.items():
        counts = Counter(row["answer"] for row in rs)
        sources = sorted({row["source"] for row in rs})
        by_source = {s: Counter(row["answer"] for row in rs if row["source"] == s)
                     for s in sources}
        families: dict[str, list[dict]] = defaultdict(list)
        for row in rs:
            families[row["group_id"]].append(row)
        multi = {gid: fam for gid, fam in families.items() if len(fam) > 1}
        for gid, fam in multi.items():
            source = fam[0]["source"]
            keys = Counter(row["answer_key"] for row in fam)
            variants = Counter(row["provenance"].get("variant") for row in fam)
            if source == "policy_v2":
                if keys != Counter({"eligible": 2, "ineligible": 2, "unknown": 2}):
                    failures.append(f"{split}/{gid}: policy_v2 语义组合异常 {dict(keys)}")
            elif source == "routing_v2":
                named = {v.split("_", 1)[1] if v not in ("ambiguous", "multi_missing") else v
                         for v in variants}
                if named != {"route_1", "route_2", "route_3", "default",
                             "ambiguous", "multi_missing"}:
                    failures.append(f"{split}/{gid}: routing_v2 变体集异常 {sorted(variants)}")
                elif sum(v.startswith("partial_") for v in variants) != 1:
                    failures.append(f"{split}/{gid}: routing_v2 partial 变体数异常")
                elif any(row["answer_key"] != row["provenance"]["variant"].split("_", 1)[1]
                         for row in fam if row["provenance"]["variant"] not in
                         ("ambiguous", "multi_missing")):
                    failures.append(f"{split}/{gid}: routing_v2 变体标签异常")
            elif source == "policy":
                if set(variants) != {"positive", "negative", "missing"}:
                    failures.append(f"{split}/{gid}: policy 变体集异常 {sorted(variants)}")
                elif keys != Counter({"eligible": 1, "ineligible": 1, "unknown": 1}):
                    failures.append(f"{split}/{gid}: policy 语义组合异常 {dict(keys)}")
            elif source == "research_taxonomy_v21":
                sides = Counter(row["provenance"].get("pair_side") for row in fam)
                option_variants = Counter(row["provenance"].get("option_variant") for row in fam)
                if sides != Counter({0: 4, 1: 4}):
                    failures.append(f"{split}/{gid}: research 两侧变体数异常 {dict(sides)}")
                elif option_variants != Counter({0: 2, 1: 2, 2: 2, 3: 2}):
                    failures.append(
                        f"{split}/{gid}: research option_variant 分布异常 {dict(option_variants)}")
                # 构造不变式：answer_key 由 main_contribution_key 同一变量写出，恒真；
                # 只作生成器自检，不作为独立标签判据。
                if any(row["answer_key"] != row["provenance"]["main_contribution_key"]
                       for row in fam):
                    construction_failures.append(
                        f"{split}/{gid}: research answer_key 与主贡献不符（构造不变式）")
        out[split] = {
            "answer_letter_counts": dict(sorted(counts.items())),
            "answer_letter_counts_by_source": {
                source: dict(sorted(c.items())) for source, c in by_source.items()},
            "position_balance": {source: _position_balance(
                [row for row in rs if row["source"] == source]) for source in sources},
            "multi_row_families": len(multi),
            "family_variant_check_failures": sum(1 for f in failures if f.startswith(split)),
        }
    return {
        "splits": out,
        "failures": failures[:20],
        "construction_failures": construction_failures[:20],
        "position_balance_gate": "report-only（不并入失败即停）",
        "gate_checks_note": "gate 内的 family 变体/语义组合检查是构造分布自检，不是独立标签判据；research/policy 的 answer_key 构造不变式只进 construction_failures。",
        "gate": "PASS" if not failures else "FAIL",
    }


def _longest(row: dict) -> str:
    return max(row["options"], key=lambda o: (len(o["description"]), -string.ascii_uppercase.index(o["label"])))["label"]


def _first(row: dict) -> str:
    return row["options"][0]["label"]


def _morph(row: dict) -> str:
    marks = set("0123456789.,;:!?%$€£¥()-+/=")
    return max(row["options"], key=lambda o: (
        sum(ch in marks for ch in o["description"]),
        -string.ascii_uppercase.index(o["label"])))["label"]


def _shortcut_reject(delta: float, se: float | None) -> bool:
    """审计 4 的拒收判据：Δ≥10pp 且 Δ≥3×family-clustered SE。"""
    return delta >= 0.10 and se is not None and delta >= 3 * se


def _shortcut_stats(rs: list[dict], rule) -> dict:
    correct = 0.0
    residual: dict[str, list[float]] = defaultdict(list)
    predicted_keys = set()
    for row in rs:
        predicted = rule(row)
        predicted_keys.add(next(o["key"] for o in row["options"]
                                if o["label"] == predicted))
        hit = 1.0 if predicted == row["answer"] else 0.0
        chance = 1.0 / len(row["options"])
        correct += hit
        residual[row["group_id"]].append(hit - chance)
    n = len(rs)
    acc = correct / n
    chance_mean = sum(1.0 / len(row["options"]) for row in rs) / n
    delta = acc - chance_mean
    fam_means = [statistics.fmean(v) for v in residual.values() if v]
    se = (statistics.stdev(fam_means) / math.sqrt(len(fam_means))
          if len(fam_means) > 1 else None)
    return {
        "accuracy": round(acc, 6),
        "chance": round(chance_mean, 6),
        "delta_pp": round(delta * 100, 3),
        "family_clustered_se_pp": round(se * 100, 3) if se is not None else None,
        "constant_class_prior": len(predicted_keys) == 1,
        "predicted_keys": sorted(predicted_keys),
        "reject": _shortcut_reject(delta, se),
    }


def shortcut_audit(rows: dict[str, list[dict]]) -> dict:
    """审计 4：表面特征捷径基线；Δ≥10pp 且 Δ≥3×family-clustered SE 即拒收。

    gate 只用聚合值；`by_source` 只报告。banking77 的选项描述由金标派生，
    该源的 longest/morph 可能反映语义而非纯表面特征。
    """
    rules = {"longest_description": _longest, "first_position": _first,
             "digit_punctuation_shape": _morph}
    out = {}
    triggered = []
    for split, rs in rows.items():
        out[split] = {name: _shortcut_stats(rs, rule) for name, rule in rules.items()}
        out[split]["by_source"] = {
            source: {name: _shortcut_stats(
                [row for row in rs if row["source"] == source], rule)
                for name, rule in rules.items()}
            for source in sorted({row["source"] for row in rs})
        }
        for name in rules:
            if out[split][name]["reject"]:
                triggered.append(f"{split}/{name}")
    return {"splits": out, "triggered": triggered,
            "gate_note": "gate 用聚合值；by_source 只报告，banking77 的选项描述由金标派生",
            "gate": "PASS" if not triggered else "FAIL"}


def template_audit(rows: dict[str, list[dict]], v1, b, b21) -> dict:
    """审计 5：结构/模板串比对 + 脚手架残留检查；结构失败即拒收。

    `legacy_whitelist_hit_rate` 恒为 1.0（生成器词汇必现），只作 sanity check；
    真正进门的是 scaffold markers（未替换占位符/TODO/`<|` 等）与 research round-trip。
    research 的金标是构造标签，没有独立预言机；answer_key == main_contribution_key
    等检查是恒真的构造不变式，只报告、不作为标签正确性的证据。
    """
    failures = []
    construction_failures = []
    scaffold_hits = []
    synthetic = 0
    legacy_hits = 0
    legacy_whitelist = ("fictional", "policy", "record", "eligible", "route",
                        "contribution", "summary", "rules")
    for split, rs in rows.items():
        for row in rs:
            source = row["source"]
            text = json.dumps(row["state"], ensure_ascii=False)
            hits = [marker for marker in SCAFFOLD_MARKERS if marker in text]
            if hits:
                scaffold_hits.append(f"{split}/{row['id']}: {hits}")
            if source in ("policy_v2", "routing_v2"):
                synthetic += 1
                legacy_hits += any(word in text.lower() for word in legacy_whitelist)
                try:
                    b.validate_synthetic(row)
                except AssertionError as exc:
                    failures.append(f"{split}/{row['id']}: {exc}")
            elif source == "policy":
                synthetic += 1
                legacy_hits += any(word in text.lower() for word in legacy_whitelist)
                p = row["provenance"]
                # policy 金标由 policy_label(tree, atom_facts) 构造；重算与
                # answer_key 同源，恒真。只作构造自检，不作为独立标签判据。
                if v1.policy_label(p["tree"], p["atom_facts"]) != row["answer_key"]:
                    construction_failures.append(
                        f"{split}/{row['id']}: policy 标签不一致（构造不变式）")
            elif source == "research_taxonomy_v21":
                synthetic += 1
                legacy_hits += any(word in text.lower() for word in legacy_whitelist)
                p = row["provenance"]
                if row["state"]["summary"] != p["template"].format(
                        main=p["contribution"], method=p["method"],
                        outcome=b21.OUTCOMES[row["answer_key"]], other=p["background"]):
                    failures.append(f"{split}/{row['id']}: 模板重建不一致")
                elif next(o["key"] for o in row["options"]
                          if o["label"] == row["answer"]) != row["answer_key"]:
                    failures.append(f"{split}/{row['id']}: answer_key 与选项不符")
                if not (row["answer_key"] == p["main_contribution_key"] != p["background_key"]):
                    construction_failures.append(
                        f"{split}/{row['id']}: research 标签关系异常（构造不变式）")
                elif p["contribution"] not in b21.CONTRIBUTIONS[row["answer_key"]]:
                    construction_failures.append(
                        f"{split}/{row['id']}: contribution 不在词表（构造不变式）")
                elif p["background"] not in b21.CONTRIBUTIONS[p["background_key"]]:
                    construction_failures.append(
                        f"{split}/{row['id']}: background 不在词表（构造不变式）")
    failures.extend(f"脚手架残留: {item}" for item in scaffold_hits)
    integrity = v1.validate_partitions(rows)
    return {
        "synthetic_rows": synthetic,
        "legacy_whitelist_hit_rate": round(legacy_hits / synthetic, 6) if synthetic else None,
        "legacy_whitelist_note": "恒真（生成器词汇必现），仅 sanity check；不进门。",
        "scaffold_markers": list(SCAFFOLD_MARKERS),
        "scaffold_hit_count": len(scaffold_hits),
        "scaffold_failures": scaffold_hits[:20],
        "construction_failures": construction_failures[:20],
        "research_label_note": "research 金标为构造标签，无独立预言机；验证覆盖模板 round-trip 与 option 映射；构造不变式恒真，只作生成器自检。",
        "policy_label_note": "policy 金标由 policy_label(tree, atom_facts) 构造；重算与 answer_key 同源，恒真；只作构造自检，不作为独立标签判据。",
        "gate_checks_note": "gate 内保留的 family 变体/语义组合检查是构造分布自检（可捕获生成器回归/损坏），不是独立标签预言机；独立标签判据仅 policy_v2/routing 的上游 bitset oracle 与公开源 gold。",
        "structural_failures": failures[:20],
        "structural_failure_count": len(failures),
        "validate_partitions": integrity,
        "gate": "PASS" if not failures else "FAIL",
    }


def coverage_matrix(rows: dict[str, list[dict]]) -> dict:
    out = {}
    for split, rs in rows.items():
        sources: dict[str, dict] = {}
        for row in rs:
            source = row["source"]
            nopt = len(row["options"])
            entry = sources.setdefault(source, {"rows": 0, "families": set(),
                                                "option_counts": Counter(),
                                                "family_options": defaultdict(set)})
            entry["rows"] += 1
            entry["families"].add(row["group_id"])
            entry["option_counts"][nopt] += 1
            entry["family_options"][nopt].add(row["group_id"])
        matrix = {}
        for source, entry in sources.items():
            matrix[source] = {
                "rows": entry["rows"],
                "families": len(entry["families"]),
                "option_counts": dict(sorted(entry["option_counts"].items())),
                "families_by_option": {n: len(g) for n, g in
                                       sorted(entry["family_options"].items())},
            }
        out[split] = matrix
    return out


def coverage_audit(matrix: dict) -> dict:
    """配额确认：每个 source × 可产生的选项数格子达到 S0 锁定的最小 family 数。"""
    failures = []
    for split, sources in matrix.items():
        for source, expected in EXPECTED_CELLS.items():
            if source not in sources:
                failures.append(f"{split}: 缺 source {source}")
                continue
            for nopt in expected:
                got = sources[source]["families_by_option"].get(nopt, 0)
                if got < MIN_FAMILIES[split]:
                    failures.append(f"{split}/{source}/{nopt}选项: {got} < {MIN_FAMILIES[split]}")
    return {"min_families_per_cell": MIN_FAMILIES, "failures": failures,
            "gate": "PASS" if not failures else "FAIL"}


def write_failed(output: Path, audits: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "FAILED.json").write_text(
        json.dumps({"status": "FAILED", "audits": audits}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, help="评测侧新 SEED（字符串；记录进 manifest）")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patched", type=Path, required=True,
                        help="已应用 0001-rejev-s0.patch 的上游副本")
    parser.add_argument("--tag", default="", help="probe / final 等标签")
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--force", action="store_true", help="允许覆盖已有输出目录")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        parser.error(f"输出目录已存在且非空：{args.output}；换新路径或加 --force")
    if not PATCH_BARRIER_RESULT.exists():
        parser.error(f"缺少 patch barrier 核验结果：{PATCH_BARRIER_RESULT}")
    patch_barrier = json.loads(PATCH_BARRIER_RESULT.read_text(encoding="utf-8"))
    if patch_barrier.get("status") != "PASS":
        parser.error(f"patch barrier 核验未通过：{patch_barrier.get('status')}")
    if not RENDER_CONTRACT_RESULT.exists():
        parser.error(f"缺少 render contract 实测结果：{RENDER_CONTRACT_RESULT}")
    render_contract = dict(RENDER_CONTRACT)
    render_contract.update(json.loads(RENDER_CONTRACT_RESULT.read_text(encoding="utf-8")))
    if not REBUILD_EVIDENCE_RESULT.exists():
        parser.error(f"缺少 rebuild evidence 实测结果：{REBUILD_EVIDENCE_RESULT}")
    rebuild_evidence = json.loads(REBUILD_EVIDENCE_RESULT.read_text(encoding="utf-8"))
    if rebuild_evidence.get("status") != "PASS":
        parser.error(f"rebuild evidence 未通过：{rebuild_evidence.get('status')}")

    frozen_hashes = set(load_lines(args.frozen / "historical-statehashes.txt"))
    frozen_families = {tuple(line.split("\t", 1)) for line in
                       load_lines(args.frozen / "historical-families.tsv")}
    v1, b, b21 = import_patched(args.patched, args.seed, args.frozen / "historical-statehashes.txt")
    lock = json.loads((args.patched / "sources.lock.json").read_text(encoding="utf-8"))
    cfg = json.loads((args.patched / "configs/research-taxonomy.json").read_text(encoding="utf-8"))

    built = build_rows(v1, b, b21, lock, args.seed, cfg, frozen_hashes)
    rows = built["rows"]
    coverage = coverage_matrix(rows)
    audits = {
        "partition": partition_audit(rows, b, frozen_hashes, frozen_families),
        "answer_position": answer_position_audit(rows),
        "shortcut": shortcut_audit(rows),
        "template": template_audit(rows, v1, b, b21),
        "coverage": coverage_audit(coverage),
    }
    audits["gate"] = ("PASS" if all(a["gate"] == "PASS" for a in audits.values()) else "FAIL")
    if audits["gate"] != "PASS":
        write_failed(args.output, audits)
        print(json.dumps({"status": "FAILED", "audits": audits}, ensure_ascii=False), file=sys.stderr)
        return 1

    out = args.output
    records = out / "records"
    records.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in SPLITS:
        path = records / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows[split]:
                fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        files[str(path.relative_to(out))] = {"sha256": sha256_file(path), "rows": len(rows[split])}
    # R4 P3-2：builder 必须断言本次写出的文件与 rebuild evidence 的 final 逐文件一致，
    # 不能只信仓库里已提交的 evidence 文件（陈旧证据风险）。
    for split in SPLITS:
        key = f"records/{split}.jsonl"
        assert files[key]["sha256"] == rebuild_evidence["final"][split], (
            f"{key} 本次 SHA {files[key]['sha256']} ≠ evidence final "
            f"{rebuild_evidence['final'][split]}——先用当前脚本重跑 rebuild evidence")

    bank_rows = [row for rs in rows.values() for row in rs if row["source"] == "banking77"]
    bank_none_correct = sum("gold_removed_none_correct" in row["provenance"].get("augmentation", [])
                            for row in bank_rows)
    bank_none_distractor = sum("none_incorrect" in row["provenance"].get("augmentation", [])
                               for row in bank_rows)
    dev2_topics = sorted({gid.rsplit(":", 2)[-2] for gid in
                          {row["group_id"] for row in rows["dev2"]
                           if row["source"] == "research_taxonomy_v21"}})
    fam_counts = Counter(row["group_id"] for row in rows["test2"])
    test2_families = len(fam_counts)
    test2_multi = sum(1 for count in fam_counts.values() if count > 1)
    test2_public_single = test2_families - test2_multi
    boolq_pb = audits["answer_position"]["splits"]["test2"]["position_balance"]["boolq"]
    boolq_first = audits["shortcut"]["splits"]["test2"]["by_source"]["boolq"]["first_position"]
    boolq_note = (
        f"boolq test2 位置诊断：A {boolq_pb['observed_share']:.3f}（n={boolq_pb['n']}，"
        f"期望 {boolq_pb['expected_share']:.3f}，Δ={boolq_pb['delta_pp']}pp，"
        f"z={boolq_pb['z']}，p={boolq_pb['p']}；z/p 只对应 test2 的 190/332）。"
        f"恒答 A 基线 accuracy {boolq_first['accuracy']:.3f}"
        f"（Δ={boolq_first['delta_pp']}pp，family SE={boolq_first['family_clustered_se_pp']}pp）。"
        f"三切分合并 A=287/508=56.5%（合并口径，不与 test2 的 z 混读）。"
        f"只报告，S1 不得据此调阈值/温度/选 seed。")
    routing_long = audits["shortcut"]["splits"]["test2"]["by_source"]["routing_v2"]["longest_description"]
    routing_note = (
        f"routing_v2 longest_description 的预测恒为 {routing_long['predicted_keys']}"
        f"（Δ={routing_long['delta_pp']}pp）；这是上游固定选项描述造成的 class prior，"
        f"不是 S0 采样伪影；family SE=0 系结构退化（族内残差全同），不是统计估计。"
        f"shortcut gate 用聚合值；S1 报告限制节须披露该 source-level 先验并给出捷径基线 accuracy。")
    research_variant_note = (
        "research_taxonomy_v21 的 variant 0 保持上游 canonical 选项顺序（test2 106/424 行，"
        "占 2.8%），答案位置由 topic 在 24 项中的规范序号决定；S1 位置偏好分析须单列 variant 0。")

    manifest = {
        "schema_version": 1,
        "created": datetime.now(timezone.utc).isoformat(),
        "updated": "2026-10-18",
        "amended_note": "S0 冻结后按评审增补登记/审计字段；records 与冻结清单 SHA 未变。",
        "tag": args.tag,
        "output": str(out),
        "data_location": "本地数据目录（不入库）；本 manifest 记录文件 SHA256",
        "route": "A2′：训练集＝历史 train-clean；只换 SEED 重建评测侧三切分",
        "seed": args.seed,
        "roles": {
            "train": "历史 train-clean（30,987，完整 SHA 见 training_data），不重建",
            "dev2": "实现诊断、吞吐预试验；不择优 checkpoint",
            "calibration2": "逐臂拟合单一标量温度 T_a；不用于终评",
            "test2": "冻结后一次性终评；不训练、不调参、不拟合温度",
        },
        "training_data": {
            "path": str(TRAIN_CLEAN.relative_to(REPO)),
            "sha256": sha256_file(TRAIN_CLEAN),
            "rows": sum(1 for line in TRAIN_CLEAN.read_text(encoding="utf-8").splitlines()
                        if line.strip()),
            "role": "seed 42/17/29/41 训练集；原样复用，不重建",
            "note": "本文件在 data/（gitignore）下；此字段是仓内可追踪的完整 SHA。",
        },
        "upstream": {
            "repo": "togethercomputer/tev1",
            "commit": "1dde7782382c9f49d627153759b8d1deab426ce0",
            "patch": str(PATCH.relative_to(REPO)),
            "patch_sha256": sha256_file(PATCH),
            "patched_dir": str(args.patched),
            "patched_dir_note": "临时副本；按 patch + 上游 commit 可重建",
            "patched_file_sha256": {
                name: sha256_file(args.patched / name)
                for name in ("build_dataset.py", "build_v2.py", "build_v21.py", "build_new_v1.py")
            },
            "env": {
                "python": sys.version.split()[0],
                "datasets": __import__("datasets").__version__,
                "transformers": __import__("transformers").__version__,
                "hf_home": os.environ.get("HF_HOME"),
                "env_note": "S0 只用 datasets 读取公开源；此处记录数据构建环境。S1 训练环境按 §3.2 另锁 transformers==5.6.2 与镜像。",
            },
        },
        "render_contract": render_contract,
        "patch_barrier": patch_barrier,
        "statehash_note": "S1 泄漏检查用 freeze_history.statehash（与上游 build_v2 逐字一致）；不要用 render.statehash（ensure_ascii=False，对非 ASCII 不同）。",
        "frozen": {
            "dir": str(args.frozen.relative_to(REPO)),
            "statehashes_sha256": sha256_file(args.frozen / "historical-statehashes.txt"),
            "statehashes": len(frozen_hashes),
            "families_sha256": sha256_file(args.frozen / "historical-families.tsv"),
            "families": len(frozen_families),
            "manifest_sha256": sha256_file(args.frozen / "freeze-manifest.json"),
            "derived_minus_covered": json.loads(
                (args.frozen / "freeze-manifest.json").read_text(encoding="utf-8")
            )["coverage"]["derived_minus_covered"],
        },
        "rebuild_evidence_check": rebuild_evidence,
        "consumed_versions": [
            {"tag": "probe", "seed": "rejev-eval2-probe-20261017",
             "dir": "data/eval2-probe-20261017", "status": "PASS-not-delivered",
             "note": "配额/审计预演；其 held-out 抽取已消耗，不再使用"},
            {"tag": "final-check", "seed": "rejev-eval2-20261017-1",
             "dir": "data/eval2-final-check", "status": "obsolete-intermediate",
             "note": "脚本改动前（b28e09f 之前）的中间产物；已作废，不用于确定性证明"},
            {"tag": "final-check2", "seed": "rejev-eval2-20261017-1",
             "dir": "data/eval2-final-check2", "status": "reproducibility-check",
             "note": "与交付版同一脚本版本；逐字节一致"},
            {"tag": "final", "seed": "rejev-eval2-20261017-1",
             "dir": str(out.relative_to(REPO)), "status": "delivered",
             "note": "交付版；与 final-check2 逐字节一致，不重复计为独立抽取"},
        ],
        "rebuild_evidence": [
            {"dir": "data/eval2-final-20261017", "created": "2026-10-17T15:56:04Z",
             "tag": "final", "sha256": {k: v["sha256"] for k, v in files.items()},
             "note": "交付版"},
            {"dir": "data/eval2-final-check2", "created": "2026-10-17T15:56:23Z",
             "tag": "final-recheck",
             "sha256": {"records/dev2.jsonl": "1a59a3138dd87573f2abf4ea8d1df1190b93d7a2383bc054cad2120f8bdfc152",
                        "records/calibration2.jsonl": "2c62e5f15a160330ee7a11410021543b373a7bf0c39f52f2c6aa31940b42f9fd",
                        "records/test2.jsonl": "1d59fa224dc28fb60139d91c332123e6650b4b1bcf5a50909f657489f361d9e5"},
             "note": "与交付版同一脚本版本；三文件逐字节一致"},
            {"dir": "data/eval2-final-check", "created": "2026-10-17T15:49:28Z",
             "tag": "final-recheck",
             "sha256": {"records/dev2.jsonl": "005aff6c85f79d16348815ef3f706aa79ad45d64fd2537a38e3b3889284c8ad6",
                        "records/calibration2.jsonl": "8c5a3b90fd7ad33d3a7fa58b020afea3d5ad63d504c4c68e1991226e9afd7ca7",
                        "records/test2.jsonl": "96113c64a6ed95d4fb0710cdb55a5227d69cfa83a019336d0610851e4f9beaa5"},
             "note": "脚本改动前的中间产物；已作废，不用于确定性证明"},
        ],
        "quotas": QUOTAS,
        "source_allocation": {
            "public": "官方 held-out：mnli validation_matched + validation_mismatched、boolq validation、banking77 test；扣冻结历史 statehash 后按 label 均衡抽取",
            "synthetic": "按 split 命名空间（dev2/calibration2/test2）用 patched 生成器新生成；按 family 整组",
        },
        "budget": {
            "cap_usd_credit": 30,
            "breakdown_usd": {
                "training_4_arms": 15.4,
                "test2_six_arms_eval": 2.3,
                "calibration2_dev2": 1.2,
                "permutation_and_extra_inference": [3, 5],
            },
            "point_estimate_usd": [22, 24],
            "with_reserve_usd": 26,
            "with_one_retrain_usd": 27,
            "with_two_retrains_usd": 31,
            "stop_condition": "实际已花＋预计超过 $30 credit 即停并修订方案；不静默缩评测或省一臂",
            "accounting": "S1 逐臂记账；每次远程运行前实时核对 credit/spend limit/活动任务，结束后核对 Modal 实际账单并确认无遗留任务",
        },
        "public_pool_audits": built["public_audits"],
        "semantic_label_sets": {
            "mnli": ["contradicted", "entailed", "neutral"],
            "boolq": ["no", "yes"],
            "banking77": sorted(set(built["public_audits"]["banking77"]["label_names"] + ["none"])),
            "policy": ["eligible", "ineligible", "unknown"],
            "policy_v2": ["eligible", "ineligible", "unknown"],
            "routing_v2": ["default", "route_1", "route_2", "route_3", "unknown"],
            "research_taxonomy_v21": sorted(b21.CONTRIBUTIONS),
        },
        "coverage": coverage,
        "audits": audits,
        "disclosures": [
            "公开源预训练接触：新终评约 32% 来自真实公开语料（mnli/boolq/banking77），无法排除已被底座预训练接触；合成源约 68% 不受此限。",
            "模板/词表/规则与历史同源：合成源沿用同一批生成器。",
            "research_taxonomy_v21 的三切分均由上游 TEST_TEMPLATES 生成（split 名称非 train/dev）；模板集合相同，实例与 family 不同；实际模板集已写入 provenance.template_set。provenance.template_split 沿用上游字段记录 split 名，引用时以 template_set 为准。",
            "research_taxonomy_v21 的金标是构造标签（authorship contrast），没有独立标签预言机；审计只验证模板 round-trip 与 option 映射，不验证真实论文归属。",
            f"banking77 上游增广：{bank_none_correct} 行金标被改写为 none（gold_removed_none_correct），{bank_none_distractor} 行以 none 作错误干扰项（none_incorrect）；逐行标记在 provenance.augmentation；semantic_label_sets.banking77 含 none 共 78 类，none 非数据集原生类别。",
            f"dev2 的 research_taxonomy_v21 覆盖 {len(dev2_topics)}/24 个 topic（{', '.join(dev2_topics)}）；做 dev2 诊断时须注意其余 topic 形态未在 dev2 出现。",
            boolq_note,
            routing_note,
            research_variant_note,
            "本 manifest 含 test2 金标字母分布；S1 读数期间不得据此调整任何阈值、温度拟合口径或 seed 选择。",
            "近重复未做：只做精确 statehash 与登记 family 检查；理由见 plan/009 §3.3。",
            "exam 无 group_id：family 检查对 exam 为 vacuous，由 statehash 覆盖并披露。",
        ],
        "rng_protocol": {
            "protocol": "①：在 get_peft_model 前显式 torch.manual_seed(seed)（控制 LoRA 初始化）；SFTConfig(seed=seed) 控制数据顺序；不另设 cuda 种子——Trainer.__init__ 的 set_seed(args.seed) 会重置 random/numpy/torch/cuda（与历史同机制）。报告声明对历史「不另设 torch/cuda 种子」纪律的偏离。",
            "evaluation_seed": args.seed,
            "training_seeds": list(TRAINING_SEEDS),
            "set_seed_timing": "未逐行核实（R5 P2-1 保留项）：S1 入口实现前须按 transformers==5.6.2 的 Trainer.__init__ → set_seed 调用链逐行确认时序，并把结论写入 S1 manifest。",
            "note": "S0 只锁定协议文本与 seed 集合；S1 入口 multiseed_train.py 必须按此实现，并在 manifest 中记录实际实现。",
        },
        "thresholds": {
            "run_health": {
                "execution_or_protocol_failure": "NaN/OOM/超时/数据或模板错配：可续跑或重试，完整记录",
                "complete_but_poor": "完整执行但表现差：属于研究结果，进主统计与失效计数，不重训、不剔除",
                "first_complete_run": "不得替换；重训单列登记",
                "retry_cap": "S1 全程最多 1 次完整重训，且仅当实际已花＋预计 ≤ $30；否则停",
                "loss_window": "末 50 步滑动均值（与 exp006 一致），并报告完整曲线",
            },
            "stability": {
                "positive_direction": "四臂相对 base-C0 的主指标差值均为正；0 视为未通过",
                "consistency": "报 seed 间 SD 与配对区间；本轮不另设 SD 阈值、不设实用改善界",
                "no_best_seed": "终评不挑最佳 seed；报告全部运行、每 seed 成绩、均值与 SD",
            },
            "uncertainty": {
                "seed_df": 3,
                "sd_95_interval_factor": [0.57, 3.73],
                "clustering": "按 family 聚类；训练间差异与样本抽样不确定性分开报，不合成一个区间",
            },
            "s1_precondition": "本 manifest 的 thresholds 与 rng_protocol 在 S1 开训前复核后生效；未锁定前不得启动训练。",
        },
        "error_analysis_protocol": {
            "unit": "family",
            "denominator": "test2 全部 family；逐臂报告 error families / total families",
            "denominator_n_test2_families": test2_families,
            "denominator_note": f"{test2_families} = {test2_multi} 多行族 + {test2_public_single} 公开单行族（test2 内）",
            "type_order": ["状态解析失败", "选项顺序敏感", "恒常策略", "标签错误", "其他"],
            "definitions": {
                "状态解析失败": "候选评分无有效选项 mask，或渲染/解析失败（正常应为 0）",
                "选项顺序敏感": "该 family 在 §3.6 置换检查中至少一个 sibling 的语义预测发生翻转",
                "恒常策略": "合成 family：该臂在同一语义 key 上的预测数达到该源 family 规模阈值（policy 3/3；policy_v2/routing_v2 ≥5/6；research_taxonomy_v21 ≥7/8），且该 family 至少一行是错例。公开单行 family：在 test2 内该 source×选项数格子（≥20 行）中，该 family 的预测=格子众数、格子众数占比 ≥0.90、且该 family 是错例；S1 计算众数时把该 family 自身排除。",
                "标签错误": "预测为有效选项、语义 key ≠ gold，且不满足上述更高优先级类型",
                "其他": "无法归入上述类型或数据缺失",
            },
            "assignment": "按 type_order 优先级单标签归类",
            "case_sampling": "每类按 group_id 升序取前 10 个 error family；每个 family 取 id 最小的一行作代表案例；全部结果冻结后抽取",
            "note": "共同错误与臂间分歧错误分列计数；base-C0 / hist 独错 family 入池并单列归属。",
            "frozen_at": "2026-10-18",
            "revision": 3,
            "revision_note": "v3 按 R2 评审：公开源格子限定 test2 内、S1 众数排除自身；锁定 test2 family 分母。",
        },
        "permutation_protocol": {
            "variant_rule": "对每个 test2 record 做一次确定性循环移位：shift = 1 + (int(sha256(id.encode('utf-8')),16) mod (n_options-1))；保留语义 key 与 description，按新顺序重标 A..；原记录与变体同题配对",
            "metric": "按语义 key 还原后的预测一致率、翻转率、按 family 聚类的 95% 区间",
            "consistency_definition": "原始记录与变体的预测语义 key 相同 = 一致",
            "labels": {
                "high_sensitivity": "翻转率点估计 ≥ 0.20",
                "low_sensitivity": "一致率 ≥ 0.95 且 family-clustered 95% 区间下界 ≥ 0.90",
                "otherwise": "不确定",
            },
            "claim_limit": "只主张对选项顺序扰动的敏感性；不主张决策由语义内容决定。",
        },
        "interval_protocol": {
            "family_clustered_bootstrap": "按 group_id 有放回重采样 10,000 次，取 2.5/97.5 百分位",
            "bootstrap_seed": 20261018,
            "paired": "同题配对差值在每次重采样内重算",
            "seed_variance": "4 臂均值与样本 SD（3 自由度）；真实 SD 的 95% 区间约观测 SD 的 0.57–3.73 倍",
            "separation": "训练间差异与样本抽样不确定性分开报，不合成一个区间",
        },
        "files": files,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "output": str(out),
        "counts": {s: len(rows[s]) for s in SPLITS},
        "gate": audits["gate"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
