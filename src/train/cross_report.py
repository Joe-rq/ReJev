"""exp003 三方同考卷对照汇总（零 GPU、纯本地）。

输入：目录下若干 eval jsonl，由 `modal volume get rejev /eval/...` 取回：
- `cross-{model}-{paper}.jsonl`  exp003 新跑
- `{model}-holdout.jsonl`        exp002 已验收的 base/adapter holdout 结果（沿用，不重跑）

**完整性是第一道关**——任一条不满足即拒绝出表（防止残缺/错配被当成三方对照）：
1. 每张考卷三方**齐全**（缺任一方不出版）；
2. 每个文件条数 == 封存值，且**无重复 id**（重复会被 dict 静默覆盖，故按原始行数查）；
3. 必需字段齐备（schema 漂移会让 accuracy 静默变 0）；
4. 同一考卷上三方的 id 集合、逐题 gold / source / paper_split **完全一致**
   （id 相同不等于同一张考卷）。

主口径为 **constrained**——tev1 官方经 Together API 用 regex 约束解码（README Protocol），
约束候选集为「该题实际选项」；unconstrained 并列报告。

用法：uv run python src/train/cross_report.py --dir <结果目录> [--md-out <文件>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWX"
# 官方公布数（`resources/tev1/evaluation/new-v1-4b/README.md`，同一 results.jsonl 的汇总）
OFFICIAL = {
    "test": {"correct": 880, "n": 1000, "label": "主考 1,000"},
    "policy_transfer": {"correct": 300, "n": 300, "label": "迁移 300"},
}
MODEL_LABEL = {"base": "MiniCPM5-2B base", "adapter": "ReJev-2B（全量训练 r16）",
               "tev1": "Tev1-4B（官方权重）", "adapter_r64": "ReJev-2B（r64 探针）",
               "clean_r16": "ReJev-2B（006 剔源 r16·已发布）"}
# 每张考卷期望的模型集：缺任一方即拒绝出表（exp003 的 P0-c 要求）。
# 004 容量探针在 holdout2 上不与 tev1 同台（该集只用于 rank 16 vs 64 的对比；
# 注意该对比**不是单变量**——两臂的训练切分与评测集也不同，2026-09-28 更正）。
MODELS_BY_PAPER = {
    "tev1paper": ("base", "adapter", "tev1"),
    "holdout": ("base", "adapter", "tev1"),
    "holdout2": ("base", "adapter", "adapter_r64"),
    # 007 OOD 客卷：五臂。plan/005 原写四臂，**加 clean_r16 是刻意的偏离**——
    # plan 写于 2026-09-25，当时 006 的干净权重轮尚未训练；而 clean_r16 才是
    # 已发布到 HF/魔搭的模型，报告缺了它给出的就不是已发布模型的 OOD 表现。
    # adapter_r64 已知在 holdout2 上崩坏（004）——本卷上它回答「OOD 是否同样崩」，
    # 属 #15 的线索，不参与主判据。
    "exam": ("base", "adapter", "clean_r16", "adapter_r64", "tev1"),
}
EXPECTED_N = {"tev1paper": 1300, "holdout": 1892,
              "holdout2": 1802,   # 封存条数（holdout2 为 004 新切分）
              "exam": 2087}       # 007 OOD 客卷（第三方 2,087 题）
REQUIRED_FIELDS = {"id", "source", "gold", "pred_constrained", "pred_unconstrained"}


def load(path: Path) -> dict[str, dict]:
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l]
    rows = [json.loads(l) for l in lines]
    ids = [r["id"] for r in rows]
    dup = len(ids) - len(set(ids))
    assert dup == 0, f"{path.name} 有 {dup} 个重复 id（会被字典静默覆盖，拒绝出表）"
    return {r["id"]: r for r in rows}


def is_valid(r: dict, mode: str) -> bool:
    """有效 = 预测是该题**实际存在**的选项。exp002 产物无 n_options，按当时 A–X 口径回退。"""
    p = r.get(f"pred_{mode}")
    if p is None:
        return False
    idx = LETTERS.find(p)
    return 0 <= idx < r.get("n_options", len(LETTERS))


def metrics(rows: list[dict], mode: str) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    correct = sum(1 for r in rows if r.get(f"pred_{mode}") == r["gold"])
    return {"n": n, "correct": correct, "accuracy": round(correct / n, 4),
            "valid_rate": round(sum(1 for r in rows if is_valid(r, mode)) / n, 4)}


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="含 cross-*.jsonl 的目录")
    ap.add_argument("--md-out", default=None, help="同时写出 Markdown 片段")
    args = ap.parse_args()

    d = Path(args.dir)
    files = {p.name: p for p in sorted(d.glob("*.jsonl"))}

    def find(model: str, paper: str):
        for name in (f"cross-{model}-{paper}.jsonl", f"{model}-{paper}.jsonl"):
            if name in files:
                return files[name]
        return None

    def check_meta(f: Path) -> dict:
        """读 sidecar meta 并核对约束协议身份。

        没有这一步，一份「旧版全 A–X 约束」但与当前同名、条数与标签齐全的文件会通过
        全部结构校验，然后被报告注记为「按题实际选项」——审计上不可接受。历史 exp002
        产物无 meta，不拒绝但显式标注其协议未经校验。
        """
        mp = f.with_name(f.stem + ".meta.json")
        if not mp.exists():
            return {"约束协议": "⚠️ 无 meta，未经校验（历史/手工产物）"}
        m = json.loads(mp.read_text(encoding="utf-8"))
        assert m.get("constraint") == "per-item-options", (
            f"{f.name} 的 meta 记录约束模式 {m.get('constraint')!r} ≠ 当前 per-item-options，"
            f"拒绝出表（否则会把旧协议的产物标成按题约束）")
        return {"约束协议": m.get("constraint"), "模型revision": (m.get("revision") or "")[:12],
                "adapter_sha256": (m.get("adapter_sha256") or "—")[:12],
                "n_target": m.get("n_target"), "code_sha256": (m.get("code_sha256") or "")[:12]}

    # ── 载入 ＋ 完整性校验（任一条不过即拒绝出表）──
    data: dict[tuple[str, str], dict[str, dict]] = {}
    sources_used = {}
    for paper in EXPECTED_N:
        for model in MODELS_BY_PAPER[paper]:
            f = find(model, paper)
            if f is None:
                continue
            rows = load(f)
            n_exp = EXPECTED_N[paper]
            assert len(rows) == n_exp, (
                f"{f.name} 条数 {len(rows)} ≠ 封存 {n_exp}——产物不完整或用了子集，拒绝出表")
            for rid, r in rows.items():  # 逐行检查：只看首行会漏掉后续行的字段漂移
                missing = REQUIRED_FIELDS - set(r)
                assert not missing, (
                    f"{f.name} 的 {rid} 缺必需字段 {sorted(missing)}（schema 漂移），拒绝出表")
            data[(model, paper)] = rows
            sources_used[f"{model}/{paper}"] = {"file": f.name, **check_meta(f)}
    for paper in EXPECTED_N:
        expected = MODELS_BY_PAPER[paper]
        present = [m for m in expected if (m, paper) in data]
        if not present:
            continue   # 本目录完全不涉及这张考卷
        absent = [m for m in expected if (m, paper) not in data]
        assert not absent, (
            f"考卷 {paper} 只跑了一部分：缺 {absent}。"
            f"已有：{present}——模型集不齐不出对照表")
    for paper in EXPECTED_N:
        expected = MODELS_BY_PAPER[paper]
        if not any((m, paper) in data for m in expected):
            continue
        ref_model = expected[0]
        ref = data[(ref_model, paper)]
        for m in expected[1:]:
            other = data[(m, paper)]
            for i in ref:
                a, b = ref[i], other[i]
                # 只比**任务语义**字段：prompt_tokens 出自各自 tokenizer（MiniCPM vs Qwen
                # 同题长度必然不同），拿它做跨模型相等断言会让报告永远无法出表——
                # 这正是第四轮评审抓到的缺陷。n_options 与 tokenizer 无关，可比。
                for field in ("gold", "source", "n_options"):
                    av, bv = a.get(field), b.get(field)
                    if av is None or bv is None:
                        continue  # exp002 产物无 n_options；缺值即无从核对
                    assert av == bv, (
                        f"{paper} 上 {i} 的 {field} 不一致：{ref_model}={av!r} vs {m}={bv!r}"
                        f"——id 相同不等于同一张考卷，拒绝出表")
                assert a.get("paper_split") == b.get("paper_split"), (
                    f"{paper} 上 {i} 的 paper_split 不一致，拒绝出表")
    print(f"完整性校验通过 ✅（三方齐全 / 条数 / 唯一 id / schema / 逐题 gold+source 一致）")
    print(f"文件来源：{json.dumps(sources_used, ensure_ascii=False)}\n")

    # ── 表 1：三方 × 考卷 × 双口径 ──
    rows_out = ["## 三方同考卷对照\n",
                "| 模型 | 考卷 | n | 约束准确率 | 无约束准确率 | 无效率(约束/无约束) | 被约束改写 |",
                "|---|---|---:|---:|---:|---:|---:|"]
    table1 = {}
    for paper in EXPECTED_N:
        if not any((m, paper) in data for m in MODELS_BY_PAPER[paper]):
            continue
        for model in MODELS_BY_PAPER[paper]:
            if (model, paper) not in data:
                continue
            rows = list(data[(model, paper)].values())
            if paper == "tev1paper":
                groups = {k: [r for r in rows if r.get("paper_split") == k]
                          for k in ("test", "policy_transfer")}
                odd = [r for r in rows if r.get("paper_split") not in ("test", "policy_transfer")]
                if odd:  # 不静默丢弃：未知 split 单独成桶并标警
                    groups[f"⚠️未知split"] = odd
            else:
                groups = {"全部": rows}
            for gname, grows in groups.items():
                if not grows:
                    continue
                c, u = metrics(grows, "constrained"), metrics(grows, "unconstrained")
                touched = sum(1 for r in grows
                              if r.get("pred_constrained") != r.get("pred_unconstrained"))
                label = f"{paper} · {gname}" if paper == "tev1paper" else paper
                rows_out.append(
                    f"| {MODEL_LABEL[model]} | {label} | {c['n']} | **{pct(c['accuracy'])}** | "
                    f"{pct(u['accuracy'])} | {1 - c['valid_rate']:.2%} / {1 - u['valid_rate']:.2%} | "
                    f"{touched} |")
                table1[f"{model}/{paper}/{gname}"] = {
                    "constrained": c, "unconstrained": u, "constraint_touched": touched}
            # 分桶总数须等于全量，防止静默丢弃
            covered = sum(len(v) for v in groups.values())
            assert covered == len(rows), f"{model}/{paper} 分桶覆盖 {covered} ≠ {len(rows)}"

    # ── 表 2：Tev1-4B 本地权重 vs 官方公布 ──
    if ("tev1", "tev1paper") in data:
        rows_out += ["\n## Tev1-4B 本地权重 vs 官方公布（同考卷，近似同协议）\n",
                 "> 该差值**不等于**考卷重建忠实度：它混合了（a）考卷题面差异、（b）本地权重与",
                 "> 历史端点权重差异、（c）解码实现差异（官方 regex 约束 vs 本仓状态机约束）。",
                 "> 可用作「复现是否落在合理范围」的检查，不能单独归因。\n",
                 "| 考卷 | 官方公布（Together API，regex 约束） | 本次实测（本地权重，按题约束） | 差 |",
                     "|---|---:|---:|---:|"]
    table2 = {}
    for split, off in (OFFICIAL.items() if ("tev1", "tev1paper") in data else []):
        grow = [r for r in data[("tev1", "tev1paper")].values()
                if r.get("paper_split") == split]
        assert len(grow) == off["n"], (
            f"表 2 的 {split} 组条数 {len(grow)} ≠ 官方 {off['n']}——分母不可比，拒绝出表")
        m = metrics(grow, "constrained")
        off_acc = off["correct"] / off["n"]
        delta = (m["accuracy"] - off_acc) * 100
        rows_out.append(f"| {off['label']} | {off['correct']}/{off['n']}（{pct(off_acc)}） | "
                        f"**{m['correct']}/{m['n']}（{pct(m['accuracy'])}）** | {delta:+.2f}pp |")
        table2[split] = {"official": off_acc, "measured": m["accuracy"],
                         "delta_pp": round(delta, 2),
                         "flag": "⚠️ 偏差>5pp，先查协议差异再下结论" if abs(delta) > 5 else "ok"}

    # ── 表 3：分 source（约束口径；每张考卷一个独立小表，列随该卷模型集）──
    table3 = {}
    table3_skipped = {}   # 静默跳过会让「某 source 全错」这类信号消失，故显式记账
    for paper in EXPECTED_N:
        models = MODELS_BY_PAPER[paper]
        if not any((m, paper) in data for m in models):
            continue
        key = "paper_split" if paper == "tev1paper" else None
        buckets: dict[tuple, dict[str, list]] = {}
        for m in models:
            for r in data[(m, paper)].values():
                g = r.get(key, "全部") if key else "全部"
                buckets.setdefault((g, r["source"]), {}).setdefault(m, []).append(r)
        rows_out += [f"\n### {paper}（{', '.join(MODEL_LABEL[m] for m in models)}）\n",
                     "| 分组 | source | n | " + " | ".join(MODEL_LABEL[m] for m in models) + " |",
                     "|---|---|---:|" + "---:|" * len(models)]
        for (g, src), cell in sorted(buckets.items(),
                                     key=lambda kv: (-len(kv[1][models[0]]), kv[0])):
            n = len(cell[models[0]])
            if n < 25:  # 过小单元格不出表（噪声），但记账不静默
                table3_skipped[f"{paper}/{g}/{src}"] = n
                continue
            vals = [pct(metrics(cell[m], "constrained")["accuracy"]) if m in cell else "—"
                    for m in models]
            rows_out.append(f"| {g} | {src} | {n} | " + " | ".join(vals) + " |")
            table3[f"{paper}/{g}/{src}"] = {
                "n": n, **{m: metrics(cell[m], "constrained")["accuracy"]
                           for m in models if m in cell}}

    md = "\n".join(rows_out)
    print(md)

    payload = {"table1_three_way": table1,
               "table2_local_vs_official": table2,   # 曾名 fidelity：会误导成「重建忠实度」
               "table3_by_source": table3, "table3_skipped": table3_skipped,
               "official_reference": OFFICIAL,
               "sources_used": sources_used,
               "note": "主口径=constrained（候选集为该题实际选项，对应官方 regex per option list）；"
                       "考卷为 tev1 开发集，Tev1-4B 有主场效应；协议差异见 exp003 README。"
                       "已知限制：exp002 历史产物无 n_options，其有效率为 A–X 口径；"
                       "已核验该批 1,892×2 条预测**无一超出实际选项**，故数值不受影响"}
    (d / "cross-report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.md_out:
        Path(args.md_out).write_text(md + "\n", encoding="utf-8")
        print(f"\n→ {args.md_out}")
    print(f"→ {d / 'cross-report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
