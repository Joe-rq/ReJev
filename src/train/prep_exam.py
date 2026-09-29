"""exp007 OOD 客卷准备（零 GPU）：第三方客卷 → 本项目评测格式。

考卷取自 `resources/tev1/evaluation/public-third-party/inputs.json`（2,087 题）
——上游 tev1 仓库内保存的三家第三方（anisselbd / themsquared / WallerChen）
构造的客卷，与 t** 的差异见 plan/005。**不是**本仓自建，故为分布外材料。

与 `prep_paper.py` 的四处关键差异（plan/005 §三）：
1. **`gold` 是选项 key，本仓要求 `answer` 是字母 label** → 须做 `{key: label}` 映射
   （`render.build_messages` 对 `answer ∉ labels` 直接抛错）
2. **`source := suite`**：`eval_cross.py:281` 直接取 `r["source"]`，缺即 KeyError
3. **补 `n_options`**：`eval_cross.py` 虽从 messages 现推，但 manifest 需登记
4. **泄漏校验去掉 `(source, group_id)` 口径**：本卷无分组键，该口径会退化成单键、
   等于没查；只留 `statehash` + `id` 两口径

另：本卷**无分组键**（每题独立），故 manifest 显式记 `"groups": null`——
不记会被读成 0 或被 `render.py:112` 那类 `len({...})` 塌成 1。

用法（`--exam-dir` 用于 `resources/` 不随本仓发布的场景，见下）：
    uv run python src/train/prep_exam.py
    uv run python src/train/prep_exam.py --exam-dir /path/to/public-third-party

⚠️ 自备副本必须与上游**逐字节相同**：`校验 0` 拿 `inputs.json` 的 sha256 与同一目录下
`manifest.json` 里记录的 `inputs_sha256` 比对，不符即中止。故 `--exam-dir` 指向的目录
要同时有 `inputs.json` 与上游 `manifest.json`（两者取自上游仓库同一 commit）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

# `-O` / `PYTHONOPTIMIZE=1` 会把本文件的全部 `assert` **整条剥离**，而本文件的
# 「校验 0–3 不过即中止」正是用 assert 实现的——剥离后脚本会一路跑到底、把一份
# **未经验证**的考卷写成「正式考卷」。2026-09-28 第四轮双谱系评审 P1/P2（两条谱系
# 各自独立报出）。整条改成显式 raise 要动 ~12 处、且将来新增的 assert 又会重新打开
# 这个洞；这里改用**更彻底**的形态：检测到 -O 就拒绝运行（`__debug__` 在 -O 下为
# False，而这个 `if` 本身不会被剥离）。
if not __debug__:
    raise RuntimeError(
        "本脚本的校验用 assert 实现，`python -O` / `PYTHONOPTIMIZE=1` 会把它们整条剥离，"
        "校验将全部静默失效。请用普通模式运行（`uv run python src/train/prep_exam.py`）。")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/data"))
from render import build_messages, statehash  # noqa: E402

EXAM_DIR = REPO / "resources/tev1/evaluation/public-third-party"
INPUTS = EXAM_DIR / "inputs.json"
UPSTREAM_MANIFEST = EXAM_DIR / "manifest.json"
OUT_DIR = REPO / "data/exam"
OUT_PATH = OUT_DIR / "exam.jsonl"


def set_exam_dir(p: Path) -> None:
    """覆盖客卷目录（对应 `--exam-dir`）。

    存在意义与 `exam_report.set_exam_dir` 相同：本仓的 `resources/` **不随仓库发布**，
    而本文件在发布集里。路径写死会让第三方拿到一个**跑不起来**的准备脚本——而
    `REPRODUCING.md` 的 §6.1 复现链第一步就是它（2026-09-28 第三轮评审 P1）。
    """
    global EXAM_DIR, INPUTS, UPSTREAM_MANIFEST
    EXAM_DIR = p
    INPUTS = p / "inputs.json"
    UPSTREAM_MANIFEST = p / "manifest.json"

# 上游 README 的公布数字（自检锚）：suite → (n, qwen_correct, jev_correct)
UPSTREAM_PUBLISHED = {
    "phishing": (2000, 1013, 1259),
    "tool_risk": (60, 56, 56),
    "ticket_routing": (27, 26, 27),
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam-dir", default=None,
                    help="第三方客卷目录（含 inputs.json 与上游 manifest.json）。默认指向"
                         "本仓的 resources/tev1/evaluation/public-third-party，**该目录不随"
                         "本仓发布**——干净环境里请用本参数指向自备副本（须与上游逐字节相同）")
    args = ap.parse_args()
    if args.exam_dir:
        set_exam_dir(Path(args.exam_dir))

    # 干净环境的第一道门（2026-09-28 第四轮双谱系评审 B 线 P3）：默认目录不随本仓发布，
    # 缺失时给一句人话，而不是让 INPUTS.read_bytes() 抛裸 FileNotFoundError。缺哪个列哪个。
    # ⚠️ 2026-09-28 第五轮评审（A 线 P2）：预检**只列了前两个文件**，而下面还会读
    # `qwen.jsonl` / `jev.jsonl`（校验 2）——缺它们仍会抛裸异常。故预检必须覆盖本脚本
    # **实际读取的每一个文件**，一次列全。
    need = [INPUTS.name, UPSTREAM_MANIFEST.name, "qwen.jsonl", "jev.jsonl"]
    absent = [f for f in need if not (EXAM_DIR / f).exists()]
    if absent:
        print(f"✗ 客卷目录不可用：{EXAM_DIR}\n"
              f"  缺少：{', '.join(absent)}\n"
              f"  该目录 **不随本仓发布**（见 REPRODUCING.md §6.1）。请自备一份与上游逐字节\n"
              f"  相同的副本（须同时含 inputs.json、上游 manifest.json 与 qwen/jev 两份逐题\n"
              f"  产物），用 --exam-dir 指向它。",
              file=sys.stderr)
        return 2

    # ── 校验 0：用的是上游封存的那一份 inputs ──
    raw = INPUTS.read_bytes()
    up_man = json.loads(UPSTREAM_MANIFEST.read_text(encoding="utf-8"))
    actual_sha = hashlib.sha256(raw).hexdigest()
    assert actual_sha == up_man["inputs_sha256"], (
        f"inputs.json 与上游 manifest 不符：{actual_sha} ≠ {up_man['inputs_sha256']}")
    print(f"校验 0 ✅ inputs.json sha256 与上游 manifest 一致（{actual_sha[:16]}…）")

    rows = json.loads(raw.decode("utf-8"))
    assert len(rows) == 2087, f"条数异常 {len(rows)}"

    # ── 校验 1：gold(key) → answer(label) 映射可行 ──
    mapped: list[dict] = []
    for r in rows:
        opts = r["task"]["options"]
        labels = [o["label"] for o in opts]
        assert labels == list("ABCDEFGHIJKLMNOPQRSTUVWX")[:len(labels)], (
            f"label 非连续 A 起：{labels} @ {r['id']}")
        key2label = {o["key"]: o["label"] for o in opts}
        assert len(key2label) == len(opts), f"key 有重复：{r['id']}"
        assert r["gold"] in key2label, f"gold {r['gold']!r} 不在 key 集 @ {r['id']}"
        rec = dict(r["task"])
        rec["id"] = r["id"]
        rec["source"] = r["suite"]          # 差异 2：source := suite
        rec["answer"] = key2label[r["gold"]]
        rec["n_options"] = len(opts)
        rec["suite"] = r["suite"]
        if "difficulty" in r:
            rec["difficulty"] = r["difficulty"]
        mapped.append(rec)
    print(f"校验 1 ✅ gold→label 映射 2,087/2,087 全部可行"
          f"（key 无重复、label 连续 A 起、gold 全在 key 集内）")

    dist = Counter((r["suite"], r["n_options"]) for r in mapped)
    print(f"           分 suite 选项数：{dict(dist)}")
    assert dict(Counter(r["suite"] for r in mapped)) == {
        "phishing": 2000, "tool_risk": 60, "ticket_routing": 27}, "suite 分布不符"

    # ── 校验 2：用上游逐题结果反证 key→label 映射（双向，不是自证）──
    # 上游 analyze.py 判的是 `correct == (prediction == gold)`，**两侧都是 key**。
    # 这里改到 label 空间重算一遍：`key2label[prediction] == 我们的 answer`。
    # 两者必须逐题一致——因为 key2label 题内是双射，这等价于 `prediction == gold`；
    # 若我把映射方向写反成 {label: key}，本校验会立刻炸（KeyError 或结果不符）。
    exam_ids = {r["id"] for r in mapped}
    for name in ("qwen", "jev"):
        attempts = [json.loads(l) for l in
                    (EXAM_DIR / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l]
        rr = {a["id"]: a for a in attempts if a.get("ok")}
        assert set(rr) == exam_ids, f"{name} 的 id 集与考卷不符"
        bad, by_suite = [], Counter()
        for r in mapped:
            a = rr[r["id"]]
            k2l = {o["key"]: o["label"] for o in r["options"]}
            recomputed = k2l.get(a["prediction"]) == r["answer"]
            if a.get("correct") != recomputed:
                bad.append(r["id"])
            if recomputed:
                by_suite[r["suite"]] += 1
        assert not bad, f"{name}: correct 与 label 空间复算不符 {len(bad)} 条：{bad[:5]}"
        for suite, (n, q_c, j_c) in UPSTREAM_PUBLISHED.items():
            expect = q_c if name == "qwen" else j_c
            assert by_suite[suite] == expect, (
                f"{name}/{suite}: 复算 correct {by_suite[suite]} ≠ 上游公布 {expect}")
        print(f"校验 2 ✅ {name}：2,087 题逐题 correct 与 label 空间复算一致；"
              f"6 项逐 suite 计数与上游 report.json 同值（{dict(by_suite)}）")

    # ── 校验 3：与 ReJev 全部数据集零重叠（statehash + id 两口径）──
    # 差异 4：不查 (source, group_id)——本卷无分组键，查了等于没查
    #
    # ⚠️ 2026-09-28 第四轮双谱系评审 P1/P2（两条谱系各自独立报出）：这六个输入都在
    # `data/` 下，而**本仓不发布数据**——第三方按发布指引从干净树跑这一步时必然
    # `FileNotFoundError`，于是「第 0 步」根本走不完。改为：**文件不在就记成「未检查」**，
    # 逐项打印、并写进产物 manifest 的 `leak_check_missing`。刻意不做成静默跳过——
    # 「未检查」与「检查过且为零」必须可区分（本仓发布闸的同一原则）。
    exam_sh = {statehash(r["state"]) for r in mapped}
    exam_ids = {r["id"] for r in mapped}
    leak: dict[str, dict] = {}
    leak_missing: list[str] = []
    for name, rel_path in (
            ("rejev-train", "data/rejev/records/rejev-train.jsonl"),
            ("rejev-holdout", "data/rejev/records/rejev-holdout.jsonl"),
            ("rejev-dev", "data/rejev/records/rejev-dev.jsonl"),
            ("rejev2-train", "data/rejev2/records/rejev2-train.jsonl"),
            ("rejev2-holdout", "data/rejev2/records/rejev2-holdout.jsonl"),
            ("tev1-paper", "data/paper/tev1-paper.jsonl")):
        path = REPO / rel_path
        if not path.exists():
            leak_missing.append(f"{name}（{rel_path}）")
            continue
        d = load_jsonl(path)
        d_sh = {statehash(r["state"]) for r in d if "state" in r}
        d_ids = {r["id"] for r in d if "id" in r}
        leak[name] = {"n": len(d), "statehash": len(exam_sh & d_sh), "id": len(exam_ids & d_ids)}
        assert not (leak[name]["statehash"] or leak[name]["id"]), f"{name} 重叠：{leak[name]}"
    if leak_missing:
        print(f"⚠️ 校验 3 **未覆盖**以下口径（输入不在本机，本仓不发布数据）："
              f"{'、'.join(leak_missing)}——它们**未被检查**，不等于通过；"
              f"该缺口已写入 manifest 的 `leak_check_missing`")
        # ⚠️ 2026-09-28 第五轮评审（两谱系各自独立报出）：此前这里**无条件**打印
        # 「✅ 已查口径两口径零泄漏」。缺输入时上面的 warning 存在，末行却是成功标记、
        # manifest 的 `note` 还写「对本项目全部数据集两口径零泄漏」——两者给出相反结论。
        # 「未检查」必须与「检查过且为零」可区分，故末行随 leak_missing 切换。
        print(f"校验 3 ⬜ **未完整检查**：已查 {len(leak)} 个口径两口径零重叠；"
              f"另有 {len(leak_missing)} 个口径缺输入、**未检查**。"
              f"备齐 data/ 下各数据集后重跑，方可称「零泄漏」")
    else:
        print(f"校验 3 ✅ 全部 6 个口径两口径零泄漏：{json.dumps(leak, ensure_ascii=False)}")

    # ── 构造 records 并做长度断言（两套 tokenizer）──
    records = [{"id": r["id"], "source": r["source"], "answer": r["answer"],
                "suite": r["suite"], "messages": build_messages(r)[:-1]}
               for r in mapped]
    lengths = {}
    for label, repo_id, rev, hf_home in (
            ("minicpm", "openbmb/MiniCPM5-2B", "12a3808a956f869c767195e9266b59c4d21d92e2", None),
            ("tev1", "togethercomputer/Tev1-4B-experimental",
             "0b7becf017daa0e5eb222f8ce7483c8c8259c52f", str(REPO / "resources/tev1/.cache"))):
        import os
        if hf_home:
            os.environ.setdefault("HF_HOME", hf_home)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(repo_id, revision=rev)
        per = {s: [] for s in ("phishing", "tool_risk", "ticket_routing")}
        overall = []
        for r in records:
            n = len(tok.apply_chat_template(
                r["messages"], tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_dict=True)["input_ids"])
            overall.append(n)
            per[r["suite"]].append(n)
            assert n < 2048, f"prompt 超长 {n}：{r['id']}（{label}）"

        def q(xs, p):
            xs = sorted(xs)
            return xs[min(len(xs) - 1, int(len(xs) * p))]

        lengths[label] = {
            "overall": {"p50": q(overall, .5), "p95": q(overall, .95),
                        "p99": q(overall, .99), "max": max(overall)},
            **{s: {"p50": q(v, .5), "p99": q(v, .99), "max": max(v)} for s, v in per.items()},
        }
    print(f"长度（tokenizer 后，门槛 2048）：{json.dumps(lengths, ensure_ascii=False)}")

    # ── 全部校验通过，落盘 ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"→ {OUT_PATH}（{len(records)} 条）")

    manifest = {
        "exam": "public-third-party",
        "n": len(records),
        "suites": dict(Counter(r["suite"] for r in records)),
        "n_options": {f"{s}:{n}": c for (s, n), c in
                      Counter((r["suite"], r["n_options"]) for r in mapped).items()},
        "gold_labels": {f"{s}:{lab}": c for (s, lab), c in
                        Counter((r["suite"], r["answer"]) for r in mapped).items()},
        "sha256": hashlib.sha256(OUT_PATH.read_bytes()).hexdigest(),
        "upstream": {
            "inputs": "resources/tev1/evaluation/public-third-party/inputs.json",
            "inputs_sha256": actual_sha,
            "sources": up_man["sources"],
            "models": {"qwen": "hassan/Qwen3.5-4B-v1-new-69617472-bdc3c2fc",
                       "jev": "typesafe/jev-1.13-20260917"},
        },
        "checks": {"inputs_sha256": "match", "gold_to_label": "2087/2087",
                   "upstream_correct_recomputed": "25/25 逐 suite 一致",
                   "leak": leak,
                   # 非空 = 这些口径**没查**（输入不在本机）。「未检查」≠「通过」。
                   "leak_check_missing": leak_missing},
        # 本卷无分组键：显式记 null，别让它被读成 0 或塌成 1
        "groups": None,
        "prompt_len": lengths,
        "tokenizers": {
            "minicpm": "openbmb/MiniCPM5-2B@12a3808a956f869c767195e9266b59c4d21d92e2",
            "tev1": "togethercomputer/Tev1-4B-experimental@0b7becf017daa0e5eb222f8ce7483c8c8259c52f",
        },
        "note": ("第三方公开客卷。"
                 # 摘要必须与上面的 leak_check_missing 一致：缺输入时**不得**写成
                 # 「全部零泄漏」（第五轮评审两谱系各自独立报出的同一处）。
                 + (f"⚠️ 零泄漏**未完整检查**：以下口径缺输入、未检查——"
                    f"{'、'.join(leak_missing)}；已查 {len(leak)} 个口径两口径零重叠。"
                    if leak_missing else "对本项目全部数据集两口径零泄漏。")
                 + "本卷一次性使用：结果不得用于调参（plan/005 风险 7）"),
    }
    (OUT_DIR / "exam-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
