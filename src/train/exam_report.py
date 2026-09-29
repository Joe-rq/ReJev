"""exp007 OOD 客卷分析（plan/005 第 7 步；零 GPU、纯本地）。

回答 plan/005 预注册的可证伪问题：**tev1 这套配方训练出来的模型，出了训练分布还剩多少？**
具体到一点：ReJev-2B 的塌陷方向是否与官方 4B 复刻一致——即 phishing 的 recall 是否也塌到个位数。

**为什么另写而不是复用 cross_report.py**：本卷的主口径不是准确率，而是
**phishing 二分类的 recall / FPR**（判据 A–D 全压在这两个数上）；cross_report 只出
准确率与无效率。两者互补：cross_report 出五臂 × 双口径的总表，本脚本出判据所需的
混淆矩阵与逐条判读。

**自检是硬前提**（plan/005 第 7 步）：把上游随卷附带的 `qwen.jsonl` / `jev.jsonl`
喂进本脚本，必须逐位复现上游公布数字（50.65% / 62.95% / recall 1.3% / 42.7% /
FPR 0.0% / 16.8%）。**对不上就是分析器错了，不是上游错了**——自检不过则拒绝产出。

**分母按 gold 实际计数，不写死**：上游 `analyze.py` 把 recall/FPR 的分母写成常量
`/1000`。本卷恰为 1000/1000 平衡，所以其公布数字**碰巧正确**；但那是巧合不是设计，
本脚本按实际计数，并在自检里顺带验证两种算法在本卷同值。

用法：uv run python src/train/exam_report.py --dir <结果目录> [--md-out <文件>]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/train"))
# 统计口径与 exp005 同源，import 复用不复制（#25 纪律）
from jev_compare import mcnemar_exact, wilson  # noqa: E402

EXAM_DIR = REPO / "resources/tev1/evaluation/public-third-party"
INPUTS = EXAM_DIR / "inputs.json"


def set_exam_dir(p: Path) -> None:
    """覆盖考卷目录（对应 `--exam-dir`）。

    存在意义：本仓的 `resources/` **不随仓库发布**，而本文件在发布集里。路径若写死，
    第三方拿到的是一个**跑不起来**的分析器（2026-09-28 首轮双谱系评审 P1）。改由参数
    注入后，第三方可指向自备副本，并用 `--selfcheck-only` 先验证副本正确。
    """
    global EXAM_DIR, INPUTS
    EXAM_DIR = p
    INPUTS = p / "inputs.json"

# 四臂（v1 部署）+ 第五臂（v2 部署，见 plan 偏离记录）
ARMS = ("base", "adapter", "clean_r16", "adapter_r64", "tev1")
# tokenizer 族——**只用于判定「哪些臂之间可以比对 prompt 指纹」**，取值的唯一来源是
# `src/train/eval_cross.py` 的 `MODELS[*]["family"]`（那边决定加载路径与词表常量）。
# 为什么必须有它（2026-09-28 第三轮评审 P1）：`prompt_sha256` 是对**各臂自己 tokenizer
# 渲染后 decode 的文本**取哈希，而 `tev1` 走 `qwen3_5` 族（EOS/PAD/chat template 全不同）
# ——本仓已在 `docs/experiments/exp003-three-way/implementation-notes.md` 登记
# 「`prompt_sha256` 不能跨模型比对」。不按族分组时，五臂齐全会让指纹比对**必然报错**、
# `main()` 直接 `return 2` 拒绝产出，第三方按发布指引**根本跑不出报告 §2.5 的读数**。
TOKENIZER_FAMILY = {
    "base": "minicpm", "adapter": "minicpm", "clean_r16": "minicpm",
    "adapter_r64": "minicpm", "tev1": "qwen3_5",
}
# 新增臂时必须同步登记族：漏登记会让该臂落进「未知族」而被静默排除在指纹比对之外
# （fail-open）。宁可在导入时就炸。用显式 raise 而非 assert——`-O` 会把 assert 剥离。
if set(TOKENIZER_FAMILY) != set(ARMS):
    raise RuntimeError(
        f"TOKENIZER_FAMILY 与 ARMS 不同步：{sorted(set(ARMS) ^ set(TOKENIZER_FAMILY))}"
        f"——新增臂须同时登记 tokenizer 族，否则该臂的指纹比对会被静默跳过")


def _upstream_families() -> dict[str, str]:
    """从 `eval_cross.py` 的 `MODELS` 里取各臂的族——**静态解析，不 import**。

    为什么不要手工复制的那份（2026-09-28 第四轮双谱系评审 P1）：上面那张表自称
    「取值的唯一来源是 eval_cross.MODELS」，实际是**手抄**的复制品——只校验键集合，
    不校验值。eval_cross 改一个臂的族而不动本表，导入期断言不会响，于是本脚本会按
    **错误的族**分组比指纹：把异族当同族（必然全题指纹不同→拒绝产出）或把同族当异族
    （**静默跳过比对**）。这两种都是静默的，正是本仓反复出现的形态。

    不直接 `import eval_cross`：那个模块 `import modal` 并在模块级建 App，本地分析
    脚本不该被它拖下水。改用 `ast` 读字面量——MODELS 全是字面量，解析失败即报错。
    """
    import ast
    src = (REPO / "src/train/eval_cross.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "MODELS" for t in node.targets):
            mods = ast.literal_eval(node.value)
            return {a: spec["family"] for a, spec in mods.items()}
    raise RuntimeError("eval_cross.py 里找不到 MODELS 字面量——本脚本的族表无从校验")


_UPSTREAM_FAMILY = _upstream_families()
if set(_UPSTREAM_FAMILY) - set(TOKENIZER_FAMILY):
    raise RuntimeError(
        f"eval_cross.MODELS 里有本表未覆盖的臂：{sorted(set(_UPSTREAM_FAMILY) - set(TOKENIZER_FAMILY))}"
        f"——本脚本会把它当成未知族")
_drift = {a: (_UPSTREAM_FAMILY[a], TOKENIZER_FAMILY[a])
          for a in TOKENIZER_FAMILY if _UPSTREAM_FAMILY.get(a) != TOKENIZER_FAMILY[a]}
if _drift:
    raise RuntimeError(
        f"TOKENIZER_FAMILY 与 eval_cross.MODELS 的族不一致：{_drift}"
        f"——按错的族分组会让指纹比对静默跳过（fail-open）或误报全题不同（fail-closed）")
ARM_LABEL = {
    "base": "MiniCPM5-2B base", "adapter": "ReJev-2B r16（exp002）",
    "clean_r16": "ReJev-2B 剔源 r16（exp006·**已发布**）",
    "adapter_r64": "ReJev-2B r64 探针（exp004）", "tev1": "Tev1-4B（官方权重）",
}
# 上游公布的 phishing 锚（README + report.json）——自检的期望值，**不是**本仓结论
UPSTREAM_ANCHOR = {
    "qwen": {"acc": 1013 / 2000, "recall": 0.013, "fpr": 0.0, "model": "hassan/Qwen3.5-4B-v1-new-69617472-bdc3c2fc"},
    "jev": {"acc": 1259 / 2000, "recall": 0.427, "fpr": 0.168, "model": "typesafe/jev-1.13-20260917"},
}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWX"


def load_inputs() -> dict[str, dict]:
    rows = json.loads(INPUTS.read_text(encoding="utf-8"))
    return {r["id"]: r for r in rows}


def key2label(row: dict) -> dict[str, str]:
    return {o["key"]: o["label"] for o in row["task"]["options"]}


def load_upstream(name: str, ref: dict[str, dict]) -> dict[str, dict]:
    """上游逐题产物 → 统一 {gold, pred, suite}。

    **一律用 key 空间**（`phishing` / `legitimate` / `readonly`…），不用字母 label：
    正类是「phishing」这个概念，它落在哪个字母由选项顺序决定——按 label 判正类会在
    选项顺序不同的卷子上静默判错。上游产物本就存 key，直接用。
    """
    out = {}
    for line in (EXAM_DIR / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        a = json.loads(line)
        if not a.get("ok"):
            continue
        row = ref[a["id"]]
        out[a["id"]] = {"gold": row["gold"], "pred": a["prediction"],
                        "suite": row["suite"], "n_options": len(row["task"]["options"])}
    return out


def load_arm(d: Path, arm: str, ref: dict[str, dict]) -> tuple[dict[str, dict], int]:
    """本仓评测产物 → key 空间。主口径 constrained（对应官方 regex per option list）。

    本仓产物存的是字母 label，故经该题选项表反查回 key；反查不到（无效输出）即为 None。

    返回 `(按 id 索引的记录, 文件里的非空行数)`。**行数必须带出来**：返回的是 dict，
    一旦文件里有重复 id，后一条会**静默覆盖**前一条，下游只能看见「行数变少」——
    把行数一并返回，`validate_arms` 才有东西可对（第三轮评审 P2）。
    """
    f = d / f"cross-{arm}-exam.jsonl"
    if not f.exists():
        return {}, 0
    out = {}
    n_rows = 0
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n_rows += 1
        r = json.loads(line)
        row = ref[r["id"]]
        l2k = {o["label"]: o["key"] for o in row["task"]["options"]}
        out[r["id"]] = {"gold": row["gold"], "pred": l2k.get(r["pred_constrained"]),
                        "pred_unc": l2k.get(r["pred_unconstrained"]), "suite": r["source"],
                        "n_options": r.get("n_options"),
                        # 保留 prompt 指纹与臂名：前者是「各臂同卷」的机械证明，
                        # 后者用于抓「产物张冠李戴」（把 base 的当 adapter 用）。
                        "prompt_sha256": r.get("prompt_sha256"), "model": r.get("model")}
    return out, n_rows


def validate_arms(arms: dict[str, dict], ref: dict[str, dict],
                  rows: dict[str, int]) -> list[str]:
    """各臂完整性校验。缺臂在 main 里只 warn，但**已加载的臂之间**必须同集同题。

    为什么必须有这一步：`load_arm` 对不存在的产物返回 `{}`（静默），而下游
    `paired()` 只在两臂**共有**的题上比对。若某臂只跑了一半，配对检验会**静默地**
    在子集上算出「无差异」——读起来像结论，实际只是**数据不全**。
    """
    problems: list[str] = []
    present = {a: r for a, r in arms.items() if r}
    if not present:
        return ["一个臂的产物都没有"]
    ref_ids = set(ref)
    for a, recs in sorted(present.items()):
        # 重复 id：dict 会静默去重，故与**读到的行数**比，而不是与自身比（旧写法
        # `len(recs) != len(set(recs))` 恒不成立——recs 本来就是 set）。
        n_rows = rows.get(a, len(recs))
        if n_rows != len(recs):
            problems.append(
                f"{a}: 存在重复 id（读到 {n_rows} 行 / {len(recs)} 个唯一 id）"
                f"——重复行会被静默覆盖，配对检验读到的不是文件全部内容")
        extra, lack = set(recs) - ref_ids, ref_ids - set(recs)
        if extra:
            problems.append(f"{a}: {len(extra)} 个 id 不在考卷 inputs.json 中")
        if lack:
            problems.append(f"{a}: 缺 {len(lack)}/{len(ref_ids)} 题（未覆盖全卷）")
        bad = [i for i, r in recs.items() if i in ref and r["suite"] != ref[i]["suite"]]
        if bad:
            problems.append(f"{a}: {len(bad)} 题的 source 与考卷不一致")
        # 产物自称的臂名必须与文件名一致——防串臂
        named = [recs[i].get("model") for i in recs]
        wrong = {m for m in named if m and m != a}
        if wrong:
            problems.append(f"{a}: 产物内 `model` 字段为 {sorted(wrong)}，与臂名不符")
    # 各臂的 prompt 指纹必须逐题一致——同卷同题的机械证明。
    # **只在同一 tokenizer 族内比**（见 TOKENIZER_FAMILY）：跨族渲染文本本就不同，
    # 比对只会产生假阳性。上游产物（tev1）无 `prompt_sha256` 字段，故还要有字段才比。
    #
    # ⚠️ 2026-09-28 第四轮双谱系评审 P1（A、B 两条谱系各自独立报出）：此前的写法是
    # `if s:` 才把该臂放进比对、`len(sha) < 2` 就整族 `continue`——**两个 fail-open**：
    #   ① 同族某臂一个指纹都没有 → 该臂被静默排除，比对照样「通过」；
    #   ② 某臂只删掉**发生差异那几题**的指纹 → 交集比对看不见，而覆盖检查（按 id）
    #      只看记录在不在，不看指纹在不在 → 照样通过。
    # 发布指引却对外声称「同族各臂逐题指纹一致」。改为**同族内一律要求指纹完整且题集
    # 相同**，缺一个就报 problem（缺指纹不是「没法比」，而是「这一臂的产物不完整」）。
    by_family: dict[str, dict[str, dict[str, str]]] = {}
    for a, recs in present.items():
        s = {i: r["prompt_sha256"] for i, r in recs.items() if r.get("prompt_sha256")}
        by_family.setdefault(TOKENIZER_FAMILY[a], {})[a] = s
    for fam, sha in sorted(by_family.items()):
        # 族内只有一个臂 → 没有可比对象。异族臂（qwen3_5）本就无该字段，属已知且正常。
        if len(sha) < 2:
            continue
        anchor = max(sha, key=lambda a: len(sha[a]))
        for a, s in sorted(sha.items()):
            if not s:
                problems.append(
                    f"{a}: 同族（{fam}）其它臂有逐题 prompt 指纹，本臂**一个都没有**"
                    f"——无法证明同卷，比对会被静默跳过")
                continue
            if len(s) < len(present[a]):
                problems.append(
                    f"{a}: {len(present[a]) - len(s)}/{len(present[a])} 题无 prompt 指纹"
                    f"——只删掉有差异那几题的指纹即可躲过比对")
            if set(s) != set(sha[anchor]):
                problems.append(
                    f"{a}: 有指纹的题集与 {anchor} 不同（差 {len(set(sha[anchor]) - set(s))} 题）")
            if a == anchor:
                continue
            diff = [i for i in set(s) & set(sha[anchor]) if s[i] != sha[anchor][i]]
            if diff:
                problems.append(
                    f"{a}: {len(diff)} 题的 prompt 指纹与 {anchor} 不同（非同卷；"
                    f"同属 {fam} 族，可逐题比对）")
    return problems


def valid(r: dict, mode: str = "pred") -> bool:
    """有效 = 预测落在该题**实际存在**的选项上。

    key 空间下这就是「反查得到」：`label→key` 表只含该题选项，字母超出选项范围时
    查不到 → None。（等价于 cross_report.is_valid 的 `LETTERS.find(p) < n_options`。）
    """
    return r.get(mode) is not None


def rate(recs: list[dict], mode: str = "pred") -> dict:
    n = len(recs)
    if n == 0:
        return {"n": 0}
    k = sum(1 for r in recs if r.get(mode) == r["gold"])
    return {"n": n, "correct": k, "accuracy": round(k / n, 6), "wilson95": wilson(k, n),
            "valid_rate": round(sum(1 for r in recs if valid(r, mode)) / n, 6)}


def phishing_rates(recs: list[dict], mode: str = "pred") -> dict:
    """正类 = `phishing`。分母按 gold 实际计数（**不写死**）。

    无效输出（pred 不是合法 label）既不是 TP 也不是 FP——它落入 FN/TN，即拉低
    recall、不抬高 FPR。这与上游 analyze.py 同口径（它只数 =='phishing' 的行）。
    """
    pos = [r for r in recs if r["gold"] == "phishing"]
    neg = [r for r in recs if r["gold"] == "legitimate"]
    tp = sum(1 for r in pos if r.get(mode) == "phishing")
    fn = len(pos) - tp
    fp = sum(1 for r in neg if r.get(mode) == "phishing")
    tn = len(neg) - fp
    if len(pos) + len(neg) != len(recs):
        # 同 selfcheck：不用 assert（`-O` 会剥离），否则分母会**静默**算错。
        raise ValueError(
            f"phishing 卷出现非二值 gold：正类 {len(pos)} + 负类 {len(neg)} ≠ {len(recs)}"
            f"——分母错了，整张混淆矩阵不可用")
    return {
        "n_pos": len(pos), "n_neg": len(neg),
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "recall": round(tp / len(pos), 6) if pos else None,
        "fpr": round(fp / len(neg), 6) if neg else None,
        "precision": round(tp / (tp + fp), 6) if (tp + fp) else None,
    }


def paired(a: dict, b: dict, mode: str = "pred") -> dict:
    """配对 McNemar（逐题，**假定逐题独立**——本卷无分组键，故无更宽的区间可用，
    该限制在报告中显式声明）。"""
    ids = sorted(set(a) & set(b))
    b_, c_ = 0, 0
    for i in ids:
        ca, cb = a[i].get(mode) == a[i]["gold"], b[i].get(mode) == b[i]["gold"]
        if ca and not cb:
            b_ += 1
        elif cb and not ca:
            c_ += 1
    return {"n_paired": len(ids), "a_only_correct": b_, "b_only_correct": c_,
            **mcnemar_exact(b_, c_)}


# ── 自检：上游产物喂进本分析器，必须复现上游公布数字 ──
def selfcheck(ref: dict[str, dict]) -> dict:
    got = {}
    for name, want in UPSTREAM_ANCHOR.items():
        recs = load_upstream(name, ref)
        ph = [r for r in recs.values() if r["suite"] == "phishing"]
        m, pr = rate(ph), phishing_rates(ph)
        checks = {
            "accuracy": (m["accuracy"], want["acc"]),
            "recall": (pr["recall"], want["recall"]),
            "fpr": (pr["fpr"], want["fpr"]),
        }
        for k, (g, w) in checks.items():
            # 用显式分支而非 `assert`：`python -O` / `PYTHONOPTIMIZE` 会把 assert
            # **整条剥离**，「自检不过则拒绝产出」的承诺随之失效（第三轮评审 P3）。
            # 发布闸不能建在可被优化掉的语句上。
            if abs(g - w) >= 5e-7:
                print(f"✗ 自检失败 [{name}/{k}]：本分析器算出 {g!r} ≠ 上游公布 {w!r}——"
                      f"**是分析器错了，不是上游错了**，拒绝产出")
                return None
        # 登记：上游 analyze.py 把分母写死 /1000；本卷恰为 1000/1000 平衡，故其数字
        # **碰巧正确**。这里如实记录「巧合」这一事实，而不是把它当成上游的设计正确。
        n_pos = pr["n_pos"]
        denom_note = ("n_pos==1000：上游写死 /1000 与本脚本按实际计数同值（巧合）"
                      if n_pos == 1000 else f"n_pos={n_pos}≠1000：上游写死的分母会给错数")
        got[name] = {"n": m["n"], "accuracy": m["accuracy"], "recall": pr["recall"],
                     "fpr": pr["fpr"], "confusion": pr["confusion"], "n_pos": n_pos,
                     "denominator_note": denom_note}
    return got


def judge(m: dict) -> dict:
    """plan/005 判据 A–D 的**机械判读**（规则训前锁定，此处只做查表，不做解释）。"""
    r_2b = m.get("adapter", {}).get("phishing", {}).get("recall")
    r_base = m.get("base", {}).get("phishing", {}).get("recall")
    r_tev1 = m.get("tev1", {}).get("phishing", {}).get("recall")
    acc_2b = m.get("adapter", {}).get("phishing", {}).get("accuracy")
    out = {"A": None, "B": None, "C": None, "D": None}
    if r_2b is not None:
        if r_2b < 0.10:
            # ⚠️ 2026-09-28 第三轮双谱系评审 P2：判据 A 原先在括号里写「配方层面的
            # 分布外缺陷」，那是**因果归因**；判据 C 已按同一理由收窄，A 却漏改。
            # 判据 A 的字面（recall < 10%）只支持「同向塌陷」，不支持「原因是配方」。
            out["A"] = (f"recall {r_2b:.4f} < 10% → 与官方 4B 复刻同向塌陷"
                        f"（本仓可复现该塌陷；**不作因果归因**——理由见判据 C）")
        elif r_2b <= 0.427:
            out["A"] = f"recall {r_2b:.4f} ∈ [10%, 42.7%] → 部分保留，介于 4B 复刻与真 Jev 之间"
        else:
            out["A"] = f"recall {r_2b:.4f} > 42.7% → 优于真 Jev；**须先排除底座偏置**再看判据 B"
    # 判据 B 是**配对**判据（base vs r16），两臂缺一不可判读。
    # ⚠️ 2026-09-28 第四轮双谱系评审 P2（A 线）：此前只检查 `r_base is not None`，
    # 缺 `adapter` 时会输出「base recall 0.xxxx 明显高于 r16 的 **None** → 微调引入了
    # 塌陷」——一个拿 None 当比较对象、却给出因果断言的句子。且「明显高于」是预注册
    # 的措辞，但代码从未定义阈值、也没做检验，等于把「高于」偷换成「明显高于」。
    if r_base is None or r_2b is None:
        out["B"] = None
    elif r_base < 0.10 and r_2b < 0.10:
        out["B"] = f"base recall {r_base:.4f} 也 < 10% → 塌陷来自**底座**，非微调引入"
    elif r_base < 0.10:
        # ⚠️ 2026-09-29 第六轮双谱系评审（A 线 P2-3 ＋ B 线 P2-4，**两条线各自独立报出**）：
        # 上一版这个分支只判 `r_base < 0.10`，**不看 r16 是否也塌**。于是
        # `r_base=0.05 / r_2b=0.30`（微调臂根本没塌）会输出「塌陷来自底座，非微调引入」
        # ——一个拿不成立的前提做的因果断言，与第五轮修掉的「方向颠倒」是**同一类**缺陷，
        # 只是落在同一个函数的**另一个分支**上。此处降级为只陈述观测到的事实。
        out["B"] = (f"base recall {r_base:.4f} < 10%，但 r16 的 {r_2b:.4f} **未塌陷** → "
                    f"「塌陷来自底座」这一读法**不成立**，本判据在本次读数上**不适用**")
    elif r_base > r_2b:
        out["B"] = (f"base recall {r_base:.4f} **高于** r16 的 {r_2b:.4f}，方向与「微调引入了"
                    f"塌陷」一致。**注意**：预注册写的是「明显高于」，而本脚本既未定义阈值"
                    f"也未做显著性检验——本行只报观测方向，不构成「明显」的判定")
    elif r_base < r_2b:
        # ⚠️ 2026-09-28 第五轮评审 B 线 P1：上面的分支原先只有「r_base >= 0.10」这一个
        # 条件，**从不比较方向**——于是 r_base=0.20 / r_2b=0.30 会输出「base 0.2000 高于
        # r16 的 0.3000，方向与『微调引入了塌陷』一致」，字面是假命题、结论方向还反了。
        # 该分支的可达区间正是判据 A 的「10%–42.7% 部分保留」档，即预注册**预期会发生**
        # 的那一档——不是理论风险。补成三分支，并由 test_exam_report.py 三个方向用例钉住。
        out["B"] = (f"base recall {r_base:.4f} **低于** r16 的 {r_2b:.4f} → 方向与「微调引入了"
                    f"塌陷」**相反**（微调在该卷上保留的正类更多）。本判据的预注册形态"
                    f"（「base 明显高于 r16」）**未成立**")
    else:
        out["B"] = (f"base recall {r_base:.4f} 与 r16 的 {r_2b:.4f} **相等** → 无方向差异，"
                    f"本判据**不可判读**")
    if r_2b is not None and r_tev1 is not None:
        # ⚠️ 2026-09-28 首轮双谱系评审 P1：原输出写「结论升级为『tev1 配方整体出了
        # 分布就塌』」是**因果归因**，而证据不支持它——本轮没有「未微调的 4B」对照臂，
        # 若 4B 底座本身在这张卷上就差，塌陷便与配方无关。改为只声称证据支持的部分。
        both = r_2b < 0.10 and r_tev1 < 0.10
        out["C"] = (f"r16 recall {r_2b:.4f} 与 Tev1-4B recall {r_tev1:.4f} "
                    + ("**同向塌陷** → 塌陷**非 2B 容量特有**（同一配方在 4B 上同样塌陷）。"
                       "注意本判据**不含**「未微调的 4B」对照臂，故它只排除「2B 太小」，"
                       "**不构成「塌陷由配方造成」的归因**。"
                       if both else "不同向")
                    + "（上游 4B 复刻 recall = 0.013）")
    if acc_2b is not None:
        d = abs(acc_2b - 0.5065) * 100
        out["D"] = (f"|{acc_2b:.4f} − 0.5065| = {d:.2f}pp "
                    f"{'≤ 10pp → 2B 与 4B 复刻在分布外同档' if d <= 10 else '> 10pp → 不同档'}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None,
                    help="含 cross-{arm}-exam.jsonl 的目录。`--selfcheck-only` 时不需要"
                         "——自检只用客卷，不碰本仓臂产物，故不逼用户瞎填一个目录")
    ap.add_argument("--exam-dir", default=None,
                    help="第三方客卷目录（含 inputs.json 与上游逐题产物）。默认指向"
                         "本仓的 resources/tev1/evaluation/public-third-party，"
                         "**该目录不随本仓发布**——在干净环境里请用本参数指向自备副本，"
                         "并核对 --selfcheck-only 能逐位复现上游公布数字")
    ap.add_argument("--md-out", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--selfcheck-only", action="store_true",
                    help="只跑上游自检（不依赖任何本仓臂产物），用于验证分析器本身")
    args = ap.parse_args()
    if not args.selfcheck_only and not args.dir:
        ap.error("--dir 是必需的（除非用 --selfcheck-only）")
    d = Path(args.dir) if args.dir else None
    if args.exam_dir:
        set_exam_dir(Path(args.exam_dir))

    # 干净环境的第一道门（2026-09-28 第四轮双谱系评审 B 线 P3）：默认目录不随本仓发布，
    # 缺失时给一句人话，而不是让 load_inputs() 抛裸 FileNotFoundError。缺哪个列哪个。
    need = [INPUTS.name, *[f"{n}.jsonl" for n in UPSTREAM_ANCHOR]]
    absent = [f for f in need if not (EXAM_DIR / f).exists()]
    if absent:
        print(f"✗ 客卷目录不可用：{EXAM_DIR}\n"
              f"  缺少：{', '.join(absent)}\n"
              f"  该目录 **不随本仓发布**（见 REPRODUCING.md §6.1）。请自备一份与上游\n"
              f"  逐字节相同的副本，用 --exam-dir 指向它；并先用 --selfcheck-only 验副本真伪。",
              file=sys.stderr)
        return 2

    ref = load_inputs()
    print("── 自检：上游产物喂进本分析器 ──")
    sc = selfcheck(ref)
    if sc is None:
        return 2
    for name, v in sc.items():
        print(f"  {name:>5}: n={v['n']} acc={v['accuracy']:.4f} recall={v['recall']:.4f} "
              f"fpr={v['fpr']:.4f} 混淆{tv(v['confusion'])}")
    print("  自检 ✅ 全部逐位复现上游公布数字（50.65% / 62.95% / 1.3% / 42.7% / 0.0% / 16.8%）\n")
    if args.selfcheck_only:
        return 0

    arms: dict[str, dict] = {}
    rows: dict[str, int] = {}
    for a in ARMS:
        arms[a], rows[a] = load_arm(d, a, ref)
    present = [a for a in ARMS if arms[a]]
    missing = [a for a in ARMS if not arms[a]]
    if missing:
        print(f"⚠️ 缺臂：{missing}（已有的：{present}）")
    if not present:
        print("✗ 一个臂的产物都没有——无内容可分析")
        return 2

    # 完整性校验**中止**而非 warn：不完整的臂会让配对检验在子集上静默出结论。
    problems = validate_arms(arms, ref, rows)
    if problems:
        print("✗ 臂完整性校验未过：")
        for p in problems:
            print(f"    {p}")
        print("  （不完整的臂会让下游配对检验静默地在子集上比对——拒绝产出）")
        return 2

    # ── 主表：五臂 × phishing 混淆矩阵 ──
    # 缺臂时把缺哪几臂**写进产物本身**（第四轮评审 A 线 P2）：只在终端 warn 的话，
    # 留档的 md 读起来与跑满五臂没有区别，而判据 A–D 里有几条依赖特定臂。
    out_md = ["## exp007 · OOD 客卷（第三方 2,087 题）判读\n",
              "> 主口径 = phishing（2,000 题，二选一，随机基线 50%）；constrained 解码。"
              "`tool_risk` / `ticket_routing` 只报不分判（plan/005）。\n"]
    if missing:
        out_md.append(f"> ⚠️ **本产物不完整**：缺 {', '.join(missing)} 臂"
                      f"（已有 {', '.join(present)}）。标「缺臂，无法判读」的判据即因此产生，"
                      f"**不得**把它读成「该判据未成立」。\n")
    out_md += ["| 臂 | 准确率 | Wilson 95% | recall | FPR | 混淆（TP/FN/FP/TN） | 有效率 |",
               "|---|---:|---|---:|---:|---|---:|"]
    summary = {}
    for a in present:
        ph = [r for r in arms[a].values() if r["suite"] == "phishing"]
        m, pr = rate(ph), phishing_rates(ph)
        summary[a] = {"phishing": {**m, **pr},
                      "overall": rate(list(arms[a].values())),
                      "by_suite": {s: rate([r for r in arms[a].values() if r["suite"] == s])
                                   for s in ("phishing", "tool_risk", "ticket_routing")}}
        out_md.append(
            f"| {ARM_LABEL[a]} | **{m['accuracy'] * 100:.2f}%** | "
            f"[{m['wilson95'][0] * 100:.2f}, {m['wilson95'][1] * 100:.2f}] | "
            f"**{pr['recall'] * 100:.2f}%** | {pr['fpr'] * 100:.2f}% | "
            f"{tv(pr['confusion'])} | {m['valid_rate'] * 100:.2f}% |")

    # ── 上游参照行（非本仓臂，标清出处）──
    out_md += ["\n**上游公布参照**（第三方客卷自带的上游逐题产物，非本仓实测）\n",
               "| 模型 | 准确率 | recall | FPR |", "|---|---:|---:|---:|"]
    for name, v in sc.items():
        out_md.append(f"| {UPSTREAM_ANCHOR[name]['model']} | {v['accuracy'] * 100:.2f}% | "
                      f"{v['recall'] * 100:.2f}% | {v['fpr'] * 100:.2f}% |")

    # ── 判据机械判读 ──
    jd = judge(summary)
    out_md += ["\n## 预注册判据（plan/005，训前锁定）的机械判读\n"]
    for k in ("A", "B", "C", "D"):
        out_md.append(f"- **判据 {k}**：{jd[k] or '（缺臂，无法判读）'}")

    # ── 配对检验（vs tev1 与 vs base）──
    # **只在 phishing 子集上做**：主表与判据 A–D 全压在这一套上，若混入
    # tool_risk / ticket_routing（60 + 27 条），全卷口径下 tev1 会因后两个 suite
    # 净赚若干题——读起来像是它在 phishing 上的优势，实际不是。
    ph_arms = {a: {i: r for i, r in arms[a].items() if r["suite"] == "phishing"}
               for a in present}
    out_md += ["\n## 配对检验（phishing 2,000 题；逐题 McNemar，**假定逐题独立**——本卷无分组键）\n",
               "| 对照 | A 仅对 | B 仅对 | p（精确双侧） |", "|---|---:|---:|---:|"]
    pairs = {}
    if "adapter" in present:
        for b in ("base", "clean_r16", "tev1", "adapter_r64"):
            if b in present:
                p = paired(ph_arms["adapter"], ph_arms[b])
                pairs[f"adapter_vs_{b}"] = p
                out_md.append(f"| r16(exp002) vs {b} | {p['a_only_correct']} | "
                              f"{p['b_only_correct']} | {p['p_exact_two_sided']:.3g} |")

    # ── 分 suite（只报不分判）──
    out_md += ["\n## 分 suite（**不作主口径**：tool_risk 60 条含主观类、ticket_routing 27 条有天花板）\n",
               "| 臂 | phishing (2000) | tool_risk (60) | ticket_routing (27) | 全卷 (2087) |",
               "|---|---:|---:|---:|---:|"]
    for a in present:
        s = summary[a]["by_suite"]
        out_md.append(f"| {ARM_LABEL[a]} | {s['phishing']['accuracy'] * 100:.2f}% | "
                      f"{s['tool_risk']['accuracy'] * 100:.2f}% | "
                      f"{s['ticket_routing']['accuracy'] * 100:.2f}% | "
                      f"{summary[a]['overall']['accuracy'] * 100:.2f}% |")

    md = "\n".join(out_md)
    print(md)

    payload = {
        "exam": "public-third-party", "n": 2087, "primary_suite": "phishing",
        "arms": summary, "upstream_anchor": sc, "selfcheck": "PASS（逐位复现上游公布）",
        "judgements": jd, "paired": pairs,
        "missing_arms": missing,
        "caveats": [
            "本卷无分组键 → 无聚类 bootstrap，逐题区间偏乐观（plan/005 风险 5）",
            "协议不可比：上游 Qwen 用原生字母 wrapper + per-task regex、Jev 用 native choice API、"
            "本仓用状态机约束 → 只做「同考卷、近似同协议」声称（plan/005 风险 1）",
            "phishing 为公开合成邮件，无法排除任一模型预训练阶段见过（plan/005 风险 2）",
            "gold 来自数据集构造规则，非独立人工标注；上游自述简单 URL 启发式即 91.6%——"
            "该数字是**数据集属性**，不是部署级安全保证（plan/005 风险 4）",
            "本卷一次性使用：结果不得用于调参（plan/005 风险 7）",
        ],
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
        print(f"\n→ {args.json_out}")
    if args.md_out:
        Path(args.md_out).write_text(md + "\n", encoding="utf-8")
        print(f"→ {args.md_out}")
    return 0


def tv(c: dict) -> str:
    return f"{c['tp']}/{c['fn']}/{c['fp']}/{c['tn']}"


if __name__ == "__main__":
    sys.exit(main())
