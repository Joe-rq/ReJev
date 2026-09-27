"""Issue #2 对照分析：同一张考卷（ReJev 封存 holdout 1,892 题）上的多方结果合表。

输入（逐题结果，均本地留存、不进 Git）：
  --jev     data/exp005/jev-holdout.jsonl        真 Jev（TypeSafe jev-1.13.0，choice 口径）
  --adapter data/exp005/adapter-holdout.jsonl    ReJev-2B（exp002，约束口径）
  --base    data/exp005/base-holdout.jsonl       MiniCPM5-2B base
  --native  data/exp005/jev-native.jsonl         Jev 原生 score/noul 口径（探索性，可缺）
  --tev1    <file>                               Tev1-4B（Issue #1；未产出则整列留空）

产出：控制台 markdown + `--out` 指定的 JSON（供 exp005 归档引用）。

统计口径（**两条腿都报，不混为一谈**）：
  ① 逐题口径：准确率 + Wilson 95% 区间、配对 McNemar 精确检验——**假定 1,892 题相互独立**。
  ② 分组口径（主口径）：考卷实际只有 **924 个 group_id**（同源变体同组，policy_v2 的 600 题
     只来自 100 组），组内变体高度相关 → 独立同分布假设不成立。故另给按 group_id 整组重抽的
     聚类 bootstrap 区间与配对差值区间。两者不一致时，结论以分组口径为准。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
Z = 1.95996398454
EXAM = REPO / "data/rejev/records/rejev-holdout.jsonl"
SEALED_SHA = "2601d649573f2b9940584ae521ad05a3ea52003ac91eb9df51557f09e0a79063"

from jev_api_eval import MODEL  # noqa: E402  本目录同源；钉死的 jev-1.13.0


def assert_identity(rows: dict[str, dict], want_mode: str, what: str) -> None:
    """读侧身份闸：jev / native 文件逐行必须与本次对照的身份**完全一致**。

    写侧的续跑闸只防「往错文件里写」，防不住「把错的文件读进来」——而本脚本是
    产出结论的那一环：若 native 文件被 choice 结果污染，`native_arm` 会静默变成
    全 source 0.00pp，伪造出「原生措辞无增益」的结论。

    三个维度（mode / model_requested / exam_sha256）**缺值或空值都拒**，与
    `jev_api_eval.check_identity` 同口径——曾经 sha 那一维过滤了空值、模型那一维
    完全没查（第八轮两谱系各自指出）。

    缺值与错值**分开断言**（第九轮）：曾经合成一条消息，唯一缺陷是缺失时会渲染出
    「非 X 的行（[]）」这种自相矛盾的文案。

    注意范围：只适用于 Jev 侧产物（jev / native）。base / adapter / tev1 是本地
    评测脚本产出，本就没有这些字段，**不得**纳入本断言。
    """
    for field, want, label in (("mode", want_mode, "模式"),
                               ("model_requested", MODEL, "模型"),
                               ("exam_sha256", SEALED_SHA, "考卷指纹")):
        missing = sum(1 for r in rows.values() if not r.get(field))
        assert not missing, (
            f"{what} 里有 {missing} 行缺 {field}（{label}无法核验）——拒绝出结论")
        wrong = {r[field] for r in rows.values()} - {want}
        assert not wrong, (
            f"{what} 里有 {len(wrong)} 种不符的 {field}（{label}）："
            f"{sorted(str(w)[:12] for w in wrong)}——拒绝出结论")


def check_inputs(jev: dict[str, dict], native: dict[str, dict],
                 jev_path: str, native_path: str) -> None:
    """读侧闸的**装配点**：`main()` 只调这一个函数，测试也直测它。

    单独抽出来是因为「闸本身测了」不等于「闸真的被挂上」——第九轮实测：把 main()
    里两行调用整段删掉，9 例全绿。这是第 4/6/7 轮三次踩过的同一个坑，故把装配
    收敛到一处、可被直接调用。
    """
    assert_identity(jev, "choice", jev_path)
    if native:
        assert_identity(native, "native-kind", native_path)


def load(path: Path) -> dict[str, dict]:
    """读逐题结果并归一出 `correct`。

    本地侧（评估脚本产物）只存 `pred_constrained`/`pred_unconstrained`，无 `correct`；
    Jev 侧直接存 `correct`。两种都在此处统一，顺带断言双解码口径逐位一致
    （exp002 已证约束器从未触发，这里把它变成每次运行的检查而不是记忆）。
    """
    if not path.exists():
        return {}
    # 注：此处不套 load_prior 的行级容错——本函数读的是**本地评测产物**（base/adapter），
    # 不是 Jev 侧结果，中段坏行会以 JSONDecodeError 响亮失败（非静默），够用（第九轮 P3-5）。
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]
    for r in rows:
        if "correct" not in r:
            assert r["pred_constrained"] == r["pred_unconstrained"], \
                f"{r['id']} 双解码口径不一致——须分开记录无效率，不可混算"
            r["correct"] = r["pred_constrained"] == r["gold"]
    return {r["id"]: r for r in rows}


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + Z * Z / n
    mid = (p + Z * Z / (2 * n)) / den
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return [round(mid - half, 4), round(mid + half, 4)]


def acc(rows: list[bool]) -> dict:
    n = len(rows)
    k = sum(rows)
    return {"n": n, "k": k, "acc": round(k / n, 4) if n else None,
            "wilson95": wilson(k, n) if n else None}


def mcnemar_exact(b: int, c: int) -> dict:
    """精确双侧 McNemar：b/c 为两方向的不一致计数。"""
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_exact_two_sided": 1.0}
    # 双尾 = 2 * P(X <= min(b,c)), X ~ Binomial(n, .5)，封顶 1
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    # 不四舍五入：极端显著时 p 会小到 1e-96 量级，round 到小数位会变成 0.0 丢掉量级
    return {"b": b, "c": c, "p_exact_two_sided": min(1.0, 2 * tail)}


def cluster_bootstrap(groups: dict[str, str], ids: list[str], stat_fn, n_boot: int = 2000,
                      seed: int = 20260925) -> list[float]:
    """按 group_id **整组重抽**的聚类 bootstrap 95% 区间。

    考卷里同源变体共享 group_id（policy_v2 的 600 题只来自 100 组），组内变体高度相关。
    逐题独立假设会低估方差、把区间报窄；整组重抽是这类「成组变体」题集的标准处置。
    """
    by_group: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        by_group[groups[i]].append(i)
    gkeys = sorted(by_group)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        drawn = rng.choices(gkeys, k=len(gkeys))
        sample = [i for g in drawn for i in by_group[g]]
        stats.append(stat_fn(sample))
    stats.sort()
    return [round(stats[int(0.025 * n_boot)], 4), round(stats[min(n_boot - 1, int(0.975 * n_boot))], 4)]


def origin(rec: dict) -> str:
    """gold 的来源：公开数据集（原始人工标注）vs 合成（生成规则/作者设定）。"""
    p = rec.get("provenance", {})
    if "dataset" in p:
        return "public:" + p["dataset"].split("/")[-1]
    return "synthetic:" + str(p.get("generator", "?"))


def rate(rows: dict[str, dict], ids: list[str]) -> float:
    """某个模型在一批 id 上的准确率（bootstrap 里反复调用，故写得轻）。"""
    xs = [bool(rows[i]["correct"]) for i in ids if rows[i].get("correct") is not None]
    return sum(xs) / len(xs) if xs else 0.0


def auc_confidence(rows: list[dict]) -> float | None:
    """confidence 区分对错的 AUC（Mann-Whitney，正确为正类；并列计 0.5）。"""
    pos = [r["confidence"] for r in rows if r["correct"] and r.get("confidence") is not None]
    neg = [r["confidence"] for r in rows if not r["correct"] and r.get("confidence") is not None]
    if not pos or not neg:
        return None
    pairs = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return round(pairs / (len(pos) * len(neg)), 4)


def binned(rows: list[dict], edges: list[float], getter) -> tuple[list[dict], float | None]:
    """把某标量按区间分箱，报每箱「均值 vs 实际准确率」及 ECE（按箱样本数加权）。

    注意：只有**该标量本身是「答对的概率」**时，算出来的才叫校准误差（ECE）。
    TypeSafe 的 confidence 是「选项概率分布的集中度」，不是所选选项的概率，
    所以两者必须分开算、分开命名（见 main 里的两张表）。
    """
    bins = []
    for lo, hi in zip(edges, edges[1:]):
        sub = [r for r in rows
               if (v := getter(r)) is not None and lo <= v < hi]
        if sub:
            accv = sum(bool(r["correct"]) for r in sub) / len(sub)
            mean = sum(getter(r) for r in sub) / len(sub)
            bins.append({"range": f"[{lo},{hi})", "n": len(sub),
                         "mean_value": round(mean, 4),
                         "accuracy": round(accv, 4),
                         "gap": round(mean - accv, 4)})
    n = sum(b["n"] for b in bins)
    ece = sum(b["n"] / n * abs(b["gap"]) for b in bins) if n else None
    return bins, (round(ece, 4) if ece is not None else None)


def chosen_prob(r: dict) -> float | None:
    """Jev 给自己所选选项的概率——**这个**才是可以做校准检验的量。"""
    p = (r.get("probs") or {}).get(r.get("jev_label"))
    return float(p) if isinstance(p, (int, float)) else None


def shape(rows: list[dict]) -> dict:
    """概率分布形态（我们只出字母时看不到的那部分）。"""
    tops, ent, margins = [], [], []
    for r in rows:
        probs = sorted((r.get("probs") or {}).values(), reverse=True)
        if len(probs) < 2:
            continue
        tops.append(probs[0])
        margins.append(probs[0] - probs[1])
        ent.append(-sum(p * math.log(p) for p in probs if p > 0))
    mean = lambda v: round(sum(v) / len(v), 4) if v else None
    return {"n": len(tops), "mean_top_prob": mean(tops), "mean_margin": mean(margins),
            "mean_entropy_nats": mean(ent),
            "top_prob_hist": dict(sorted(Counter(round(t, 1) for t in tops).items()))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jev", default=str(REPO / "data/exp005/jev-holdout.jsonl"))
    ap.add_argument("--adapter", default=str(REPO / "data/exp005/adapter-holdout.jsonl"))
    ap.add_argument("--base", default=str(REPO / "data/exp005/base-holdout.jsonl"))
    ap.add_argument("--native", default=str(REPO / "data/exp005/jev-native.jsonl"))
    ap.add_argument("--tev1", default=None, help="Tev1-4B 逐题结果（Issue #1；缺则留空）")
    ap.add_argument("--exam", default=str(EXAM), help="考卷原文（取 group_id 与 gold 来源用）")
    ap.add_argument("--boot", type=int, default=2000, help="聚类 bootstrap 次数")
    ap.add_argument("--out", default=str(REPO / "data/exp005/compare.json"))
    args = ap.parse_args()

    jev = load(Path(args.jev))
    adapter = load(Path(args.adapter))
    base = load(Path(args.base))
    native = load(Path(args.native))
    tev1 = load(Path(args.tev1)) if args.tev1 else {}
    if not jev:
        raise SystemExit("缺 Jev 结果文件")

    # 读侧身份断言（装配点见 check_inputs 的 docstring）。
    check_inputs(jev, native, args.jev, args.native)

    import hashlib
    exam_path = Path(args.exam)
    exam_sha = hashlib.sha256(exam_path.read_bytes()).hexdigest()
    assert exam_sha == SEALED_SHA, f"考卷哈希与封存不符：{exam_sha}（结果不可与 exp002 并表）"
    exam = {r["id"]: r for r in
            (json.loads(l) for l in exam_path.read_text(encoding="utf-8").split("\n") if l.strip())}
    groups = {i: exam[i]["group_id"] for i in exam}

    ids = sorted(set(adapter) & set(base) & set(jev))
    assert len(ids) == len(jev) == 1892, f"对齐异常：交集 {len(ids)} / jev {len(jev)}"
    models = {"base_2b": base, "rejev_2b": adapter, "jev": jev}
    if tev1:
        missing = set(ids) - set(tev1)
        assert not missing, (
            f"Tev1-4B 结果缺 {len(missing)} 个 id（例：{sorted(missing)[:3]}）——"
            f"不能整列并表，否则 gold 循环会以 KeyError 崩溃而非清晰报错")
        models["tev1_4b"] = tev1
    for i in ids:  # 逐题各源 gold 必须一致，否则不是同一张卷
        g = {rows[i]["gold"] for rows in models.values()}
        assert len(g) == 1, f"{i} gold 不一致 {g}"
        assert models["jev"][i]["gold"] == exam[i]["answer"], f"{i} 考卷 gold 与结果不符"

    n_groups = len({groups[i] for i in ids})
    out: dict = {"exam": "rejev-holdout (1892)", "n": len(ids), "n_groups": n_groups,
                 "exam_sha256": exam_sha,
                 "jev_model_reported": sorted({r.get("model_reported") for r in jev.values()
                                               if r.get("model_reported")}),
                 "overall": {}, "by_source": {}, "by_origin": {}, "paired": {},
                 "confidence": {}, "calibration": {}, "native_arm": {}, "notes": {}}

    print(f"# 同考卷对照（{len(ids)} 题 / **{n_groups} 组**）\n")
    print("## 总表\n")
    print("> 逐题口径与分组口径并报。考卷实际只有 "
          f"{n_groups} 个 group_id（同源变体同组），组内相关 → **以分组口径为准**。\n")
    print("| 模型 | 准确率 | 逐题 Wilson 95% | 分组聚类 bootstrap 95% | n |")
    print("|---|---:|---|---|---:|")
    for name, rows in models.items():
        res = acc([bool(rows[i]["correct"]) for i in ids if rows[i].get("correct") is not None])
        res["cluster_bootstrap95"] = cluster_bootstrap(
            groups, ids, lambda s: rate(rows, s), args.boot)
        out["overall"][name] = res
        ci, cb = res["wilson95"], res["cluster_bootstrap95"]
        print(f"| {name} | {res['acc']*100:.2f}% | [{ci[0]*100:.2f}, {ci[1]*100:.2f}] | "
              f"[{cb[0]*100:.2f}, {cb[1]*100:.2f}] | {res['n']} |")
    if not tev1:
        print("| ~~tev1_4b~~ | 待 Issue #1 | — | — | — |")
        out["notes"]["tev1_4b"] = ("本次未并表：Tev1-4B 逐题结果不在 --tev1 指定路径。"
                                   "该结果由 Issue #1 / exp003 产出（本地副本不进 Git），"
                                   "缺失时退回三方表。")

    print("\n## 分 source\n")
    print("> 点估计 + **分组聚类**区间，与主表同口径。注意两种区间各有退化情形："
          "全对的分组会让聚类区间塌成 [100,100]（不含未见组的风险），"
          "而组数少的 source 上逐题 Wilson 会偏窄。两者冲突时不要单取其一。\n")
    hdr = ["source", "n", "组"] + list(models)
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    by_src: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        by_src[adapter[i]["source"]].append(i)
    sub_boot = max(400, args.boot // 2)
    for src in sorted(by_src, key=lambda s: -len(by_src[s])):
        sid = by_src[src]
        cells = [f"{src}", f"{len(sid)}", f"{len({groups[i] for i in sid})}"]
        row = {"n": len(sid), "groups": len({groups[i] for i in sid})}
        for name, rows in models.items():
            r = acc([bool(rows[i]["correct"]) for i in sid if rows[i].get("correct") is not None])
            r["cluster_bootstrap95"] = cluster_bootstrap(
                groups, sid, lambda s, rr=rows: rate(rr, s), sub_boot)
            row[name] = r
            cb = r["cluster_bootstrap95"]
            cells.append("—" if r["acc"] is None else
                         f"{r['acc']*100:.1f}% [{cb[0]*100:.1f},{cb[1]*100:.1f}]")
        print("| " + " | ".join(cells) + " |")
        out["by_source"][src] = row

    print("\n## 分 gold 来源（合成 vs 公开）\n")
    org_ids: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        org_ids[origin(exam[i])].append(i)
    print("| gold 来源 | n | 组 | " + " | ".join(models) + " |")
    print("|---|---:|---:|" + "---:|" * len(models))
    for org in sorted(org_ids, key=lambda s: -len(org_ids[s])):
        oid = org_ids[org]
        cells = [org, str(len(oid)), str(len({groups[i] for i in oid}))]
        row = {"n": len(oid), "groups": len({groups[i] for i in oid})}
        for name, rows in models.items():
            r = acc([bool(rows[i]["correct"]) for i in oid if rows[i].get("correct") is not None])
            r["cluster_bootstrap95"] = cluster_bootstrap(
                groups, oid, lambda s, rr=rows: rate(rr, s), sub_boot)
            row[name] = r
            cb = r["cluster_bootstrap95"]
            cells.append("—" if r["acc"] is None else
                         f"{r['acc']*100:.1f}% [{cb[0]*100:.1f},{cb[1]*100:.1f}]")
        print("| " + " | ".join(cells) + " |")
        out["by_origin"][org] = row
    for band, pref in (("合成合计", "synthetic"), ("公开合计", "public")):
        bid = [i for k, v in org_ids.items() if k.startswith(pref) for i in v]
        cells = [f"**{band}**", str(len(bid)), str(len({groups[i] for i in bid}))]
        for name, rows in models.items():
            r = acc([bool(rows[i]["correct"]) for i in bid if rows[i].get("correct") is not None])
            r["cluster_bootstrap95"] = cluster_bootstrap(
                groups, bid, lambda s, rr=rows: rate(rr, s), sub_boot)
            cb = r["cluster_bootstrap95"]
            cells.append("—" if r["acc"] is None else
                         f"**{r['acc']*100:.2f}%** [{cb[0]*100:.1f},{cb[1]*100:.1f}]")
            out["by_origin"].setdefault(
                band, {"n": len(bid), "groups": len({groups[i] for i in bid})})[name] = r
        print("| " + " | ".join(cells) + " |")
    syn = sum(len(v) for k, v in org_ids.items() if k.startswith("synthetic"))
    print(f"\n合成任务 {syn}/{len(ids)}（{syn/len(ids)*100:.1f}%）的 gold 来自生成规则或作者设定，"
          f"不是独立人工标注；公开源取自各数据集 `train` split 的原始标签。"
          f"两者性质不同，报告须分层呈现，**不可只报混合加权总分**。")

    print("\n## 配对检验（同卷同题）\n")
    print("> **推断只以「分组聚类 bootstrap 差值」为准**；逐题 McNemar 假定 1,892 题独立，"
          "在 924 组的成组变体上偏激进（p 偏小），仅作描述用。")
    print("> 六组比较未做多重校正。\n")
    print("| 对照 | 仅 A 对 | 仅 B 对 | 两者都对 | 两者都错 | McNemar 精确双侧 p（**描述用**） | "
          "**分组聚类 bootstrap 差值 95%（推断用）** |")
    print("|---|---:|---:|---:|---:|---|---|")
    names = list(models)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            na, nb = names[a], names[b]
            ra, rb = models[na], models[nb]
            # 配对分析只在**两模型都判了**的题上做；API 失败(correct=None)既不算对也不算错。
            # 本卷全部成功，但把这条写成断言而不是靠运气——
            # 否则一旦有失败响应，分母会悄悄变小、失败会被当成答错。
            pair = [i for i in ids
                    if isinstance(ra[i].get("correct"), bool) and isinstance(rb[i].get("correct"), bool)]
            assert len(pair) == len(ids), f"{na}/{nb} 有 {len(ids)-len(pair)} 题无有效判定"
            only_a = sum(1 for i in pair if ra[i]["correct"] and not rb[i]["correct"])
            only_b = sum(1 for i in pair if rb[i]["correct"] and not ra[i]["correct"])
            both_ok = sum(1 for i in pair if ra[i]["correct"] and rb[i]["correct"])
            both_no = sum(1 for i in pair if not ra[i]["correct"] and not rb[i]["correct"])
            m = mcnemar_exact(only_a, only_b)
            diff_ci = cluster_bootstrap(
                groups, pair, lambda s: rate(ra, s) - rate(rb, s), args.boot)
            print(f"| {na} vs {nb} | {only_a} | {only_b} | {both_ok} | {both_no} | "
                  f"{m['p_exact_two_sided']:.3e} | [{diff_ci[0]*100:+.2f}, {diff_ci[1]*100:+.2f}] pp |")
            out["paired"][f"{na}_vs_{nb}"] = {
                **m, "only_a": only_a, "only_b": only_b,
                "both_correct": both_ok, "both_wrong": both_no,
                # 字段名带 pp，值就必须是百分点——曾经这里直接存比例（−0.0289 实为 −2.89pp），
                # 下游按名字读会再缩小 100 倍。
                "diff_cluster_bootstrap95_pp": [round(x * 100, 4) for x in diff_ci]}

    print("\n## Jev 的 confidence 与正确率（附加观测，不设判据）\n")
    print("> TypeSafe 的 `confidence` 是**选项概率分布的集中度**，不是「答对的概率」。")
    print("> 因此它是**排序信号**（看区分力），不能直接当校准概率用；校准另算下表。\n")
    jrows = [jev[i] for i in ids if jev[i].get("confidence") is not None]
    edges = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001]
    cbins, _ = binned(jrows, edges, lambda r: r.get("confidence"))
    print("| confidence 区间 | n | 均值 | 实际准确率 | 差 |")
    print("|---|---:|---:|---:|---:|")
    for b in cbins:
        print(f"| {b['range']} | {b['n']} | {b['mean_value']:.3f} | "
              f"{b['accuracy']*100:.1f}% | {b['gap']:+.3f} |")
    pbins, pece = binned(jrows, edges, chosen_prob)
    print("\n真实校准（按**所选选项的概率**分箱；只有这一张能算 ECE）：\n")
    print("| 所选选项概率区间 | n | 均值 | 实际准确率 | 差 |")
    print("|---|---:|---:|---:|---:|")
    for b in pbins:
        print(f"| {b['range']} | {b['n']} | {b['mean_value']:.3f} | "
              f"{b['accuracy']*100:.1f}% | {b['gap']:+.3f} |")
    sh = shape(jrows)
    n_eq = sum(1 for r in jrows if r.get("confidence") == chosen_prob(r))
    print(f"\n> 以下均为**点估计，无置信区间**，且建于成组变体数据上；本节不设判据。")
    print(f"\n- AUC(confidence → 对错) = **{auc_confidence(jrows)}**（0.5=无区分力）")
    print(f"- 校准 ECE（所选选项概率口径）= **{pece}**；"
          f"confidence 与所选选项概率相等的行数 {n_eq}/{len(jrows)}"
          f"（说明两者确非同一个量）")
    print(f"- 平均 top 概率 {sh['mean_top_prob']}，平均 top1−top2 边际 {sh['mean_margin']}，"
          f"平均熵 {sh['mean_entropy_nats']} nats")
    print(f"- top 概率分布：{sh['top_prob_hist']}")
    out["confidence"] = {"bins": cbins, "auc": auc_confidence(jrows), **sh,
                         "n_equal_to_chosen_prob": n_eq}
    out["calibration"] = {"bins_on_chosen_prob": pbins, "ece_on_chosen_prob": pece}

    if native:
        print("\n## 探索性附加组：原生 score/noul 口径（非预注册协议）\n")
        print("| source | n | Choice 口径 | 原生口径 | Δ |")
        print("|---|---:|---:|---:|---:|")
        nat: dict[str, dict] = {}
        for src in sorted({r["source"] for r in native.values()}):
            sids = [i for i in ids if jev[i]["source"] == src and i in native]
            c = acc([bool(jev[i]["correct"]) for i in sids])
            nv = acc([bool(native[i]["correct"]) for i in sids])
            nat[src] = {"n": len(sids), "choice": c, "native": nv,
                        "delta_pp": round((nv["acc"] - c["acc"]) * 100, 2)}
            print(f"| {src} | {len(sids)} | {c['acc']*100:.1f}% | {nv['acc']*100:.1f}% | "
                  f"{nat[src]['delta_pp']:+.2f}pp |")
        tot_c = acc([bool(jev[i]["correct"]) for i in native])
        tot_n = acc([bool(native[i]["correct"]) for i in native])
        d = [i for i in native if jev[i]["correct"] != native[i]["correct"]]
        c_only = sum(1 for i in d if jev[i]["correct"])
        print(f"\n合计（{len(native)} 题）：Choice {tot_c['acc']*100:.1f}% vs 原生 "
              f"{tot_n['acc']*100:.1f}% · 两种问法**判定不同的只有 {len(d)} 题**"
              f"（Choice 独对 {c_only} / 原生独对 {len(d)-c_only}）。")
        print("**结论边界**：样本仅 250 题、分歧仅个位数，**不足以断言「原生措辞无增益」**，"
              "更不足以据此排除措辞影响；它只说明这 250 题上没有看到大幅改善。"
              "另：Score 的官方输出是**连续加权分数**，本脚本按概率众数离散化——"
              "换用「期望分数四舍五入」会有若干题改变预测，离散化规则本身即影响比较，"
              "故本组不作为主表口径。")
        out["native_arm"] = {"per_source": nat, "total_choice": tot_c, "total_native": tot_n,
                             "n_disagree": len(d), "choice_only_right": c_only,
                             "caveat": "250 题 / 分歧个位数，不足以断言措辞无影响；"
                                       "Score 离散化规则未预注册"}

    out["notes"]["protocol"] = (
        "一题一请求（与 tev1 仓库内第三方对照同款纪律）；模型钉 jev-1.13.0；"
        "HTTP 错误至多重试一次（本次 1,892 题中 1 题触发一次重试并成功）；"
        "选项按我们考卷里的原始顺序序列化后发出、**不洗牌**——这是为与 ReJev 侧看同一顺序"
        "而固定的协议条件，不等于验证过 Jev 对顺序不敏感；"
        "主场效应：考卷为 tev1 分布，ReJev-2B 训练于此；**假设** Jev 不在该分布上"
        "（其训练数据无从查证，属推断）——结论限定为「本考卷上」，不做跨分布的一般化判断。"
        "两模型看到的题目形态并不对称：ReJev 侧读的是渲染成文本的 prompt，"
        "Jev 侧 `state` 按 JSON 对象原生直传（与上游第三方对照同款），此差异未做消融。")
    out["notes"]["gold_provenance"] = (
        f"1,892 题中 {1_167} 题为合成任务，gold 由生成规则或作者设定产生，非独立人工标注；"
        "725 题为公开数据集原始标签（取自各数据集 train split）。"
        "三方 gold 逐题一致只证明抄录一致，不构成标注正确性证据；"
        "完全排除预训练接触公开样本亦不可能。")
    out["notes"]["cost"] = {"jev_choice_est_usd": 0.05268, "jev_native_est_usd": 0.004179,
                            "note": "TypeSafe 直连响应不含 cost 字段（OpenRouter 才有），"
                                    "以账号实际扣费为准"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # 重抽是随机的：把次数与种子落进产物，否则同一个脚本两次跑出来的端点会有细小差异而无法追溯
    out["bootstrap"] = {"n_boot_main": args.boot, "n_boot_sub": sub_boot,
                        "seed": 20260925, "note": "分表用 n_boot_sub，主表与配对用 n_boot_main"}
    out["notes"]["bootstrap_caveat"] = (
        "聚类 bootstrap 在**全部组都答对**的退化情形下必然每次抽到全对，区间塌成 "
        "[100.0,100.0]——它不含「未见过的组答错」的风险，不可读作「能力确定为 100%」。"
        "注意：这不等于「换回逐题 Wilson 就对了」——该分组仅 24 组，Wilson 的 "
        "98.04% 下端同样无法界定未见组风险（它假定题间独立，与成组事实不符）。"
        "两种现有区间都不足以刻画未见组风险，只能明说这一限制。")
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[json] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
