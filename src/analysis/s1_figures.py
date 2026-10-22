"""Generate the three S1 result figures for report v1.1.0 / paper v1.1.0.

All values come from the frozen ``analysis.json``; the script never recomputes a
metric.  It writes PDF (for LaTeX) and PNG (for the Markdown report) into
``docs/paper/figures/`` and a companion ``fig4-6-numbers.json`` with every
plotted value, so the captions and the figures cannot drift silently.

Usage::

    uv run --group figures python src/analysis/s1_figures.py \
        --analysis docs/experiments/exp008-s1-multiseed/analysis.json \
        --out docs/paper/figures --force

Requires the optional ``figures`` dependency group (matplotlib). The delivery
directory is frozen and requires ``--force``; prefer a temporary ``--out`` for
experiments. The committed PNGs are pinned in
``docs/paper/figures/SHA256SUMS``; regenerating on a different font stack can
change the bytes even when the plotted values are identical.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from figure_style import ACCENT, BASELINE, CAT, GRID, INK, INK_2, MUTED, apply

ROOT = Path(__file__).resolve().parents[2]
ARMS = ["base-c0", "hist", "seed42", "seed17", "seed29", "seed41"]
SEEDS = ["seed42", "seed17", "seed29", "seed41"]
FINE = ["hist", "seed42", "seed17", "seed29", "seed41"]


def pct(x: float) -> float:
    return 100.0 * x


@contextmanager
def _locked(out: Path):
    """Serialize writers targeting the same output directory (temp lock)."""
    key = hashlib.sha256(str(out.resolve()).encode()).hexdigest()[:12]
    lock_path = Path(tempfile.gettempdir()) / f"rejev-figures-{key}.lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _save(fig, out: Path, stem: str) -> None:
    # P1: matplotlib's PDF backend writes CreationDate/ModDate by default, so
    # two runs of identical code produce different bytes and break SHA256SUMS.
    # P2: unique temp file + atomic replace; the caller holds the output lock.
    for suffix, fmt, meta in ((".pdf", "pdf", {"CreationDate": None, "ModDate": None}),
                              (".png", "png", None)):
        target = out / f"{stem}{suffix}"
        fd, tmp_name = tempfile.mkstemp(dir=out, prefix=f".{stem}{suffix}.", suffix=".tmp")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            fig.savefig(tmp, format=fmt, metadata=meta)
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    plt.close(fig)


def fig_seed_stability(a: dict[str, Any], out: Path) -> dict[str, Any]:
    acc = {arm: pct(a["calibration"][arm]["raw"]["accuracy"]) for arm in ARMS}
    diffs = {
        s: (
            pct(a["seed_analysis"]["per_seed"][s]["paired_diff"]["point"]),
            [pct(x) for x in a["seed_analysis"]["per_seed"][s]["paired_diff"]["ci95"]],
        )
        for s in SEEDS
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    order = ARMS
    y = np.arange(len(order))
    colors = [BASELINE if arm == "base-c0" else (MUTED if arm == "hist" else ACCENT)
              for arm in order]
    ax1.barh(y, [acc[arm] for arm in order], color=colors, height=0.55)
    ax1.set_yticks(y, order)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("test2 accuracy (%)")
    ax1.axvline(acc["base-c0"], color=INK_2, lw=0.8, ls=(0, (3, 2)))
    for yi, arm in zip(y, order):
        ax1.text(acc[arm] + 1.0, yi, f"{acc[arm]:.2f}", va="center", fontsize=7)
    ax1.set_title("accuracy (base dashed)", loc="left", fontsize=8)

    y2 = np.arange(len(SEEDS))
    pts = np.array([diffs[s][0] for s in SEEDS])
    lo = np.array([diffs[s][1][0] for s in SEEDS])
    hi = np.array([diffs[s][1][1] for s in SEEDS])
    ax2.errorbar(pts, y2, xerr=[pts - lo, hi - pts], fmt="o", color=CAT[0],
                 ecolor=CAT[0], elinewidth=1.2, capsize=3, ms=4)
    ax2.axvline(0, color=INK_2, lw=0.8)
    ax2.set_yticks(y2, SEEDS)
    ax2.invert_yaxis()
    ax2.set_xlabel("paired accuracy difference vs base (pp)")
    for yi, s in zip(y2, SEEDS):
        ax2.text(hi[list(SEEDS).index(s)] + 0.6, yi,
                 f"{pts[list(SEEDS).index(s)]:+.2f} [{lo[list(SEEDS).index(s)]:+.2f}, {hi[list(SEEDS).index(s)]:+.2f}]",
                 va="center", fontsize=6.5)
    ax2.set_xlim(min(lo) - 3, max(hi) + 14)
    ax2.set_title("paired difference, family-clustered 95% CI", loc="left", fontsize=8)
    fig.tight_layout(w_pad=1.5)
    _save(fig, out, "fig4_seed_stability")
    return {"accuracy_pct": acc, "paired_diff_pp": {s: list(diffs[s][0:1]) + [diffs[s][1]] for s in SEEDS}}


def fig_calibration(a: dict[str, Any], out: Path) -> dict[str, Any]:
    metrics = [
        ("nll", "NLL", "raw NLL", "cal NLL"),
        ("brier", "Brier", "raw Brier", "cal Brier"),
        ("ece_top_label", "ECE", "raw ECE", "cal ECE"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.8))
    for col, (key, label, raw_label, cal_label) in enumerate(metrics):
        for row, arms in enumerate((ARMS, FINE)):
            ax = axes[row, col]
            y = np.arange(len(arms))
            raw = [a["calibration"][arm]["raw"][key] for arm in arms]
            cal = [a["calibration"][arm]["calibrated"][key] for arm in arms]
            ax.hlines(y, np.minimum(raw, cal), np.maximum(raw, cal), color=GRID, lw=1.0)
            ax.plot(raw, y, "o", mfc="white", mec=INK_2, ms=4, label="raw")
            ax.plot(cal, y, "o", color=CAT[0], ms=4, label="calibrated")
            ax.set_yticks(y, arms)
            ax.invert_yaxis()
            ax.set_title(f"{label} — {'all arms' if row == 0 else 'fine-tuned only'}",
                         loc="left", fontsize=7.5)
            for yi, r, c in zip(y, raw, cal):
                ax.annotate(f"{r:.4f} → {c:.4f}", xy=(max(r, c), yi),
                            xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=5.6)
            if row == 0:
                ax.set_xlim(0, max(raw) * 1.65)
            else:
                lo = min(min(raw), min(cal))
                hi = max(max(raw), max(cal))
                if key == "ece_top_label":
                    ax.set_xlim(0.010, hi * 1.28)
                else:
                    ax.set_xlim(lo * 0.92, hi * 1.28)
            if row == 1:
                ax.set_xlabel(label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out, "fig5_calibration")
    vals = {
        arm: {
            "raw": {k: a["calibration"][arm]["raw"][k] for k, *_ in metrics},
            "calibrated": {k: a["calibration"][arm]["calibrated"][k] for k, *_ in metrics},
            "T": a["calibration"][arm]["T"]["T"],
        }
        for arm in ARMS
    }
    return vals


def fig_permutation_errors(a: dict[str, Any], out: Path) -> dict[str, Any]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    y = np.arange(len(ARMS))
    cons = [pct(a["permutation"][arm]["consistency"]) for arm in ARMS]
    lo = [pct(a["permutation"][arm]["consistency_ci95"][0]) for arm in ARMS]
    hi = [pct(a["permutation"][arm]["consistency_ci95"][1]) for arm in ARMS]
    labels = [a["permutation"][arm]["label"] for arm in ARMS]
    colors = []
    for arm in ARMS:
        if arm == "base-c0":
            colors.append(MUTED)
        elif labels[ARMS.index(arm)] == "uncertain":
            colors.append(CAT[3])
        else:
            colors.append(CAT[0])
    for yi, c, l, h, color in zip(y, cons, lo, hi, colors):
        ax1.errorbar([c], [yi], xerr=[[c - l], [h - c]], fmt="none",
                     ecolor=color, elinewidth=1.2, capsize=3)
        ax1.scatter([c], [yi], color=color, s=16, zorder=3)
    ax1.set_yticks(y, ARMS)
    ax1.invert_yaxis()
    ax1.set_xlim(55, 100)
    ax1.axvline(95, color=GRID, ls=(0, (3, 2)), lw=0.8, zorder=1)
    ax1.set_xlabel("semantic-key consistency (%)")
    for yi, arm, lab, l, h in zip(y, ARMS, labels, lo, hi):
        if arm == "base-c0":
            ax1.text(h + 1.0, yi, lab.replace("_", " "), ha="left",
                     va="center", fontsize=5.6, color=INK_2)
        else:
            ax1.text(l - 2.0, yi, lab.replace("_", " "), ha="right",
                     va="center", fontsize=5.6, color=INK_2)
    ax1.set_title("option-permutation consistency, 95% CI", loc="left", fontsize=8)

    cats = [("order_sensitive", "order sensitive", CAT[1]),
            ("label_error", "label error", CAT[0]),
            ("constant_policy", "constant policy", CAT[2])]
    error_keys = {"order_sensitive": "选项顺序敏感",
                  "label_error": "标签错误",
                  "constant_policy": "恒常策略"}
    for arm in ARMS:
        by = a["error_analysis"][arm]["by_class"]
        unknown = set(by) - set(error_keys.values())
        assert not unknown, f"{arm}: unknown error classes {unknown}"
    left = np.zeros(len(ARMS))
    for key, lab, color in cats:
        vals = np.array([a["error_analysis"][arm]["by_class"].get(error_keys[key], 0)
                         for arm in ARMS])
        ax2.barh(y, vals, left=left, color=color, height=0.55, label=lab)
        left += vals
    ax2.set_yticks(y, ARMS)
    ax2.invert_yaxis()
    ax2.set_xlabel("error families (test2; state-parse failure and other = 0)")
    for yi, arm in zip(y, ARMS):
        ax2.text(left[ARMS.index(arm)] + 8, yi,
                 f"{a['error_analysis'][arm]['error_families']}", va="center", fontsize=6.5)
    ax2.set_xlim(0, max(left) * 1.18)
    ax2.legend(loc="lower right", fontsize=6, frameon=False)
    ax2.set_title("error families by type", loc="left", fontsize=8)
    fig.tight_layout(w_pad=1.5)
    _save(fig, out, "fig6_permutation_errors")
    return {
        "permutation": {arm: {
            "consistency_pct": pct(a["permutation"][arm]["consistency"]),
            "ci95": [pct(x) for x in a["permutation"][arm]["consistency_ci95"]],
            "label": a["permutation"][arm]["label"],
        } for arm in ARMS},
        "errors": {arm: {
            "error_families": a["error_analysis"][arm]["error_families"],
            "by_class": a["error_analysis"][arm]["by_class"],
        } for arm in ARMS},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="docs/experiments/exp008-s1-multiseed/analysis.json",
                    type=Path)
    ap.add_argument("--out", default="docs/paper/figures", type=Path)
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting the frozen delivery figures")
    args = ap.parse_args()
    out = args.out.resolve()
    pinned = (ROOT / "docs/paper/figures").resolve()
    if out == pinned and not args.force:
        existing = list(out.glob("fig[4-6]_*.pdf")) + list(out.glob("fig[4-6]_*.png"))
        if existing:
            raise SystemExit(
                "refusing to overwrite pinned figures in docs/paper/figures; "
                "pass --force (or use a temporary --out for experiments)")
    apply()
    raw = args.analysis.read_bytes()
    analysis = json.loads(raw)
    args.out.mkdir(parents=True, exist_ok=True)
    with _locked(args.out):
        numbers = {
            "analysis_sha256": hashlib.sha256(raw).hexdigest(),
            "fig4_seed_stability": fig_seed_stability(analysis, args.out),
            "fig5_calibration": fig_calibration(analysis, args.out),
            "fig6_permutation_errors": fig_permutation_errors(analysis, args.out),
        }
        numbers_path = args.out / "fig4-6-numbers.json"
        fd, numbers_tmp = tempfile.mkstemp(dir=args.out, prefix=".fig4-6-numbers.",
                                           suffix=".tmp")
        os.close(fd)
        tmp_path = Path(numbers_tmp)
        try:
            tmp_path.write_text(
                json.dumps(numbers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp_path, numbers_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    print(f"wrote 3 figures + fig4-6-numbers.json to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
