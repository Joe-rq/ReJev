"""验收对比：base vs adapter 的逐题四象限 + McNemar + 分 source 增益。

输入两个 eval jsonl（同 id 集），输出对照报告（003 验收判据的支撑材料）。
零 GPU、纯本地。

用法：uv run python src/train/compare.py <base.jsonl> <adapter.jsonl> [--mode constrained]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    return {json.loads(l)["id"]: json.loads(l)
            for l in path.read_text(encoding="utf-8").splitlines() if l}


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """精确二项检验（b=base对/adapter错，c=base错/adapter对）。返回 (chi2_approx, p_exact)。"""
    n = b + c
    if n == 0:
        return 0.0, 1.0
    # 精确 p：双侧
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    p = min(p, 1.0)
    chi2 = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    return chi2, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("adapter")
    ap.add_argument("--mode", default="constrained",
                    choices=["constrained", "unconstrained"])
    args = ap.parse_args()

    B = load(Path(args.base))
    A = load(Path(args.adapter))
    ids = sorted(set(B) & set(A))
    mode = args.mode

    quad = {"both_right": 0, "both_wrong": 0, "base_only": 0, "adapter_only": 0}
    per_src = defaultdict(lambda: {"n": 0, "base_ok": 0, "adapter_ok": 0})
    adapter_invalid = base_invalid = 0
    for i in ids:
        br = B[i][f"pred_{mode}"] == B[i]["gold"]
        ar = A[i][f"pred_{mode}"] == A[i]["gold"]
        base_invalid += B[i][f"pred_{mode}"] is None
        adapter_invalid += A[i][f"pred_{mode}"] is None
        key = ("both_right" if br and ar else "both_wrong" if not br and not ar
               else "base_only" if br else "adapter_only")
        quad[key] += 1
        s = per_src[A[i]["source"]]
        s["n"] += 1
        s["base_ok"] += br
        s["adapter_ok"] += ar

    n = len(ids)
    base_acc = (quad["both_right"] + quad["base_only"]) / n
    adap_acc = (quad["both_right"] + quad["adapter_only"]) / n
    chi2, p = mcnemar(quad["base_only"], quad["adapter_only"])

    print(f"口径：{mode} · 样本 {n}")
    print(f"base    准确率：{base_acc:.4f}（无效 {base_invalid}）")
    print(f"adapter 准确率：{adap_acc:.4f}（无效 {adapter_invalid}）")
    print(f"增益：{(adap_acc - base_acc) * 100:+.2f} pp")
    print(f"\n四象限：都对 {quad['both_right']} · 都错 {quad['both_wrong']} · "
          f"仅base对 {quad['base_only']} · 仅adapter对 {quad['adapter_only']}")
    print(f"McNemar：chi2≈{chi2:.1f}，精确双侧 p={p:.2e}"
          f"{'（显著 p<0.05）' if p < 0.05 else '（不显著）'}")
    print("\n分 source（按增益排序）：")
    rows = sorted(per_src.items(),
                  key=lambda kv: (kv[1]["adapter_ok"] - kv[1]["base_ok"]) / kv[1]["n"],
                  reverse=True)
    for s, v in rows:
        b = v["base_ok"] / v["n"]
        a = v["adapter_ok"] / v["n"]
        print(f"  {s:24s} n={v['n']:4d} base={b:.3f} adapter={a:.3f} Δ={(a-b)*100:+.1f}pp")

    report = {"mode": mode, "n": n, "base_acc": round(base_acc, 4),
              "adapter_acc": round(adap_acc, 4),
              "delta_pp": round((adap_acc - base_acc) * 100, 2),
              "quadrants": quad, "mcnemar": {"chi2": round(chi2, 2), "p_exact": p},
              "per_source": {s: v for s, v in per_src.items()}}
    out = Path(args.adapter).with_name(f"compare-{mode}.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
