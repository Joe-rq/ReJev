"""Generate report/paper tables from the archived S1 analysis.

This module does **not** recompute any metric. It only reshapes
``docs/experiments/exp008-s1-multiseed/analysis.json`` into the exact tables
and figure series used by the v1.1.0 report and paper, so the prose numbers
and the archived analysis cannot drift apart silently.

Usage::

    uv run python src/analysis/report_assets.py \
        --analysis docs/experiments/exp008-s1-multiseed/analysis.json \
        --out <out-dir>
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ARMS = ["base-c0", "hist", "seed42", "seed17", "seed29", "seed41"]
SEED_ARMS = ["seed42", "seed17", "seed29", "seed41"]
# test2 is frozen at 3,800 items; calibration2 (T fitting) is 1,500 and must not
# be used as the coverage denominator (R3 review).
TEST2_SIZE = 3800
ERROR_KEYS = ["状态解析失败", "选项顺序敏感", "恒常策略", "标签错误", "其他"]
TASKS = [
    "banking77",
    "boolq",
    "mnli",
    "policy",
    "policy_v2",
    "research_taxonomy_v21",
    "routing_v2",
]


@contextmanager
def _locked(out: Path):
    """Serialize writers targeting the same output directory (temp lock)."""
    key = __import__("hashlib").sha256(str(out.resolve()).encode()).hexdigest()[:12]
    lock_path = Path(tempfile.gettempdir()) / f"rejev-assets-{key}.lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.",
                                    suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def pct(x: float, digits: int = 2) -> str:
    return f"{100.0 * x:.{digits}f}"


def pp(x: float, digits: int = 2) -> str:
    return f"{100.0 * x:+.{digits}f}"


def ci(interval: list[float], digits: int = 2) -> str:
    return f"[{pp(interval[0], digits)}, {pp(interval[1], digits)}]"


def num(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def _cal(analysis: dict[str, Any], arm: str, kind: str) -> dict[str, Any]:
    return analysis["calibration"][arm][kind]


def build_assets(analysis: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    # ── main results ────────────────────────────────────────────────────
    main = []
    for arm in ARMS:
        raw = analysis["calibration"][arm]["raw"]
        acc = raw["accuracy"]
        cov = raw["n"] / TEST2_SIZE
        assert 0 < cov <= 1.0, f"coverage out of range for {arm}: {cov}"
        row = {
            "arm": arm,
            "accuracy": acc,
            "accuracy_pct": pct(acc),
            "coverage": cov,
        }
        if arm in SEED_ARMS:
            diff = analysis["seed_analysis"]["per_seed"][arm]["paired_diff"]
            row["paired_diff"] = diff["point"]
            row["paired_diff_pp"] = pp(diff["point"])
            row["ci95_pp"] = ci(diff["ci95"])
        main.append(row)
    out["main"] = main

    # ── seed stability ─────────────────────────────────────────────────
    seed_rows = []
    for arm in SEED_ARMS:
        p = analysis["seed_analysis"]["per_seed"][arm]
        seed_rows.append(
            {
                "arm": arm,
                "accuracy": p["accuracy"],
                "accuracy_pct": pct(p["accuracy"]),
                "paired_diff_pp": pp(p["paired_diff"]["point"]),
                "ci95_pp": ci(p["paired_diff"]["ci95"]),
            }
        )
    sa = analysis["seed_analysis"]
    out["seed_stability"] = {
        "rows": seed_rows,
        "mean_diff_pp": pp(sa["mean_diff"]),
        "sd_diff_pp": pct(sa["sd_diff"]),
        "positive_direction": sa["positive_direction"],
        "zero_arm": sa["zero_arm"],
        # The report quotes the SD rounded to 1.78pp; use that rounded value so
        # the 0.57x/3.73x interval matches the text (1.01--6.64pp, not 1.02).
        "sd_range_low": f"{round(100.0 * sa['sd_diff'], 2) * 0.57:.2f}",
        "sd_range_high": f"{round(100.0 * sa['sd_diff'], 2) * 3.73:.2f}",
        "c7": {
            "diff_pp": pp(sa["c7_hist_vs_seed42"]["paired_diff_seed42_minus_hist"]["point"]),
            "ci95_pp": ci(sa["c7_hist_vs_seed42"]["paired_diff_seed42_minus_hist"]["ci95"]),
        },
    }

    # ── calibration ────────────────────────────────────────────────────
    cal_rows = []
    for arm in ARMS:
        c = analysis["calibration"][arm]
        cal_rows.append(
            {
                "arm": arm,
                "T": num(c["T"]["T"], 4),
                "at_boundary": c["T"]["at_boundary"],
                "raw_nll": num(c["raw"]["nll"], 4),
                "raw_brier": num(c["raw"]["brier"], 4),
                "raw_ece": num(c["raw"]["ece_top_label"], 4),
                "cal_nll": num(c["calibrated"]["nll"], 4),
                "cal_brier": num(c["calibrated"]["brier"], 4),
                "cal_ece": num(c["calibrated"]["ece_top_label"], 4),
                "c3_mismatch": c["c3_argmax_mismatches"],
            }
        )
    out["calibration"] = cal_rows

    # ── permutation ────────────────────────────────────────────────────
    perm_rows = []
    for arm in ARMS:
        p = analysis["permutation"][arm]
        perm_rows.append(
            {
                "arm": arm,
                "n": p["n"],
                "consistency_pct": pct(p["consistency"]),
                "flip_pct": pct(p["flip_rate"]),
                "ci95_pct": f"[{pct(p['consistency_ci95'][0])}, {pct(p['consistency_ci95'][1])}]",
                "label": p["label"],
            }
        )
    out["permutation"] = perm_rows

    # ── errors (family unit) ───────────────────────────────────────────
    err_rows = []
    for arm in ARMS:
        e = analysis["error_analysis"][arm]
        by = e["by_class"]
        unknown = set(by) - set(ERROR_KEYS)
        assert not unknown, f"{arm}: unknown error classes {unknown}"
        # analysis.json omits zero-count classes; absent known class means 0.
        err_rows.append(
            {
                "arm": arm,
                "error_families": e["error_families"],
                "total_families": e["total_families"],
                "parse_failure": by.get("状态解析失败", 0),
                "order_sensitive": by.get("选项顺序敏感", 0),
                "constant_policy": by.get("恒常策略", 0),
                "label_error": by.get("标签错误", 0),
                "other": by.get("其他", 0),
            }
        )
    struct = analysis["error_analysis"]["_seed_error_structure"]
    attr = analysis["error_analysis"]["_base_hist_attribution"]
    out["errors"] = {
        "rows": err_rows,
        "shared_all_four": struct["shared_all_four"],
        "divergent": struct["divergent"],
        "base_only": attr["base_only"],
        "hist_only": attr["hist_only"],
        "base_hist_both_wrong_seeds_right": attr["base_hist_both_wrong_seeds_right"],
        "seeds_all_wrong_base_hist_right": attr["seeds_all_wrong_base_hist_right"],
    }

    # ── per-task table ─────────────────────────────────────────────────
    pertask = []
    for arm in ARMS:
        for task in TASKS:
            t = analysis["by_source"][arm][task]
            pertask.append(
                {
                    "arm": arm,
                    "task": task,
                    "n": t["n"],
                    "accuracy_pct": pct(t["accuracy"]),
                    "accuracy_ci95_pp": ci(t["accuracy_ci"]["ci95"]),
                    "macro_f1": num(t["macro_f1"], 4),
                    "macro_f1_ci95": f"[{num(t['macro_f1_ci']['ci95'][0], 4)}, {num(t['macro_f1_ci']['ci95'][1], 4)}]",
                    "macro_recall": num(t["per_class_recall"]["macro_recall_defined"], 4),
                }
            )
    out["pertask"] = pertask

    # ── option-count strata (all arms) ─────────────────────────────────
    option_rows = []
    for arm in ARMS:
        for n_opt, t in analysis["calibration"][arm]["by_option_count"].items():
            option_rows.append(
                {
                    "arm": arm,
                    "n_options": int(n_opt),
                    "n": t["n"],
                    "raw_accuracy_pct": pct(t["raw"]["accuracy"]),
                    "raw_ece": num(t["raw"]["ece_top_label"], 4),
                    "cal_accuracy_pct": pct(t["calibrated"]["accuracy"]),
                    "cal_ece": num(t["calibrated"]["ece_top_label"], 4),
                }
            )
    out["option_count"] = option_rows

    # ── selective prediction ───────────────────────────────────────────
    selective = {}
    for arm in ARMS:
        selective[arm] = {
            kind: [
                {
                    "coverage_pct": pct(p["coverage"]),
                    "n": p["n"],
                    "accuracy_pct": pct(p["accuracy"], 3),
                }
                for p in analysis["calibration"][arm]["selective"][kind]
            ]
            for kind in ("raw", "calibrated")
        }
    out["selective"] = selective

    # ── single-class recall declines (the 30 cells) ─────────────────────
    declines = []
    for arm in SEED_ARMS:
        for task in TASKS:
            base = analysis["by_source"]["base-c0"][task]["per_class_recall"]
            tuned = analysis["by_source"][arm][task]["per_class_recall"]
            for cls, b in base.items():
                if cls in ("macro_recall_defined", "n_defined"):
                    continue
                a = tuned.get(cls)
                if a is None or b is None:
                    continue
                if a < b:
                    declines.append(
                        {
                            "arm": arm,
                            "task": task,
                            "class": cls,
                            "base": round(b, 4),
                            "tuned": round(a, 4),
                        }
                    )
    out["single_class_declines"] = {
        "count": len(declines),
        "by_arm": {a: sum(1 for d in declines if d["arm"] == a) for a in SEED_ARMS},
        "rows": declines,
    }

    return out


def render_markdown(assets: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("## Main results (test2, candidate scoring)\n")
    lines.append("| arm | accuracy | coverage | paired diff vs base | 95% CI |")
    lines.append("|---|---:|---:|---:|---|")
    for r in assets["main"]:
        diff = r.get("paired_diff_pp", "—")
        ci_s = r.get("ci95_pp", "—")
        lines.append(
            f"| {r['arm']} | {r['accuracy_pct']}% | {r['coverage']:.1f} | {diff} | {ci_s} |"
        )

    s = assets["seed_stability"]
    lines.append("\n## Seed stability\n")
    lines.append(
        f"- mean paired diff {s['mean_diff_pp']}pp, SD {s['sd_diff_pp']}pp, "
        f"positive_direction={s['positive_direction']}, zero_arm={s['zero_arm']}"
    )
    lines.append(
        f"- 3 df: true SD 95% interval ≈ {s['sd_range_low']}–{s['sd_range_high']}pp"
    )
    lines.append(
        f"- C7 seed42−hist {s['c7']['diff_pp']}pp {s['c7']['ci95_pp']}"
    )

    lines.append("\n## Calibration (test2)\n")
    lines.append("| arm | T | raw NLL | raw Brier | raw ECE | cal NLL | cal Brier | cal ECE | C3 mismatch |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in assets["calibration"]:
        lines.append(
            f"| {r['arm']} | {r['T']} | {r['raw_nll']} | {r['raw_brier']} | {r['raw_ece']} "
            f"| {r['cal_nll']} | {r['cal_brier']} | {r['cal_ece']} | {r['c3_mismatch']} |"
        )

    lines.append("\n## Permutation (test2)\n")
    lines.append("| arm | n | consistency | flip | 95% CI | label |")
    lines.append("|---|---:|---:|---:|---|---|")
    for r in assets["permutation"]:
        lines.append(
            f"| {r['arm']} | {r['n']} | {r['consistency_pct']}% | {r['flip_pct']}% "
            f"| {r['ci95_pct']} | {r['label']} |"
        )

    e = assets["errors"]
    lines.append("\n## Error families (test2)\n")
    lines.append("| arm | error/total | state-parse failure | order | constant | label | other |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in e["rows"]:
        lines.append(
            f"| {r['arm']} | {r['error_families']}/{r['total_families']} | {r['parse_failure']} "
            f"| {r['order_sensitive']} | {r['constant_policy']} | {r['label_error']} | {r['other']} |"
        )
    lines.append(
        f"- shared all four: {e['shared_all_four']}; divergent: {e['divergent']}; "
        f"base-only: {e['base_only']}; hist-only: {e['hist_only']}; "
        f"base+hist wrong & seeds right: {e['base_hist_both_wrong_seeds_right']}; "
        f"seeds all wrong & base+hist right: {e['seeds_all_wrong_base_hist_right']}"
    )

    d = assets["single_class_declines"]
    lines.append(
        f"\n## Single-class recall declines: {d['count']} "
        f"(seed42/17/29/41 = {d['by_arm']['seed42']}/{d['by_arm']['seed17']}/"
        f"{d['by_arm']['seed29']}/{d['by_arm']['seed41']})\n"
    )
    for r in d["rows"]:
        lines.append(
            f"- {r['arm']} {r['task']} {r['class']}: {r['base']:.2f} → {r['tuned']:.2f}"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--analysis",
        default="docs/experiments/exp008-s1-multiseed/analysis.json",
        type=Path,
    )
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    assets = build_assets(analysis)
    args.out.mkdir(parents=True, exist_ok=True)
    with _locked(args.out):
        _atomic_write(args.out / "assets.json",
                      json.dumps(assets, ensure_ascii=False, indent=2) + "\n")
        _atomic_write(args.out / "tables.md", render_markdown(assets))
    print(f"wrote {args.out/'assets.json'} and {args.out/'tables.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
