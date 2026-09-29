"""exp003 考卷准备（零 GPU）：tev1 官方考卷 → 本项目评测格式。

考卷本体取自 `resources/tev1/data/v1/records/{test,policy_transfer}.jsonl`
（v1 test 1,000 ＋ policy_transfer 300），**不是**按 id 在重建数据中反查——
实测该反查交集为 0（官方考卷取 test split，我们重建的是 train/dev）。

三重校验（任一失败即中止）：
1. 与官方逐题结果 `evaluation/new-v1-4b/results.jsonl` 按 id 对齐，gold 逐条一致
2. 分 source 计数与官方 README 的 main breakdown 一致
3. 与 ReJev train/holdout/dev 三口径（group / statehash / id）零重叠

产出 `data/paper/tev1-paper.jsonl`：{id, source, answer, messages, paper_split}
（messages 只含 system+user，与 holdout-eval.jsonl 同构；评测时按各自 tokenizer 渲染）。

用法（`--tev1-dir` 用于 `resources/` 不随本仓发布的场景，见下）：
    uv run python src/train/prep_paper.py
    uv run python src/train/prep_paper.py --tev1-dir /path/to/tev1-checkout/resources/tev1

⚠️ 本仓的 `resources/` **不随仓库发布**，而本文件在发布集里——路径写死会让第三方拿到
一个**跑不起来**的准备脚本，而 `REPRODUCING.md` 的 §6.1 复现链第一步就是它
（2026-09-28 第三轮评审 P1）。`--tev1-dir` 指向的目录须是上游 tev1 仓库的
`resources/tev1` 布局（`data/v1/records/`、`data/v1/instruction/`、`evaluation/new-v1-4b/`）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# `-O` / `PYTHONOPTIMIZE=1` 会把本文件的全部 `assert` **整条剥离**，而本文件对上游
# results 的条数/gold/split/渲染逐字一致性检查全部用 assert 实现——剥离后脚本会一路
# 跑到底，把一份**未经验证**的考卷当「已对齐上游」写出。2026-09-28 第四轮双谱系评审
# P1/P2（两条谱系各自独立报出）。见 `prep_exam.py` 同一处注释：检测到 -O 即拒绝运行，
# 比逐条改 raise 更彻底（将来新增的 assert 也一并覆盖）。
if not __debug__:
    raise RuntimeError(
        "本脚本的校验用 assert 实现，`python -O` / `PYTHONOPTIMIZE=1` 会把它们整条剥离，"
        "校验将全部静默失效。请用普通模式运行（`uv run python src/train/prep_paper.py`）。")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/data"))
from render import build_messages, statehash  # noqa: E402

TEV1 = REPO / "resources/tev1"
OFFICIAL = TEV1 / "evaluation/new-v1-4b/results.jsonl"


def set_tev1_dir(p: Path) -> None:
    """覆盖上游 tev1 目录（对应 `--tev1-dir`）。"""
    global TEV1, OFFICIAL
    TEV1 = p
    OFFICIAL = p / "evaluation/new-v1-4b/results.jsonl"


def rel(p: Path) -> str:
    """manifest 里记路径：在仓内记相对路径，覆盖到仓外时如实记绝对路径。"""
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)
# 官方 README 的 main test breakdown（1,000 条的 source 计数）
OFFICIAL_MAIN_BREAKDOWN = {"ag_news": 100, "banking77": 200, "boolq": 200,
                           "mnli": 250, "policy": 150, "sst5": 100}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def main() -> int:
    import hashlib

    ap = argparse.ArgumentParser()
    ap.add_argument("--tev1-dir", default=None,
                    help="上游 tev1 的 resources/tev1 目录。默认取本仓的 resources/tev1，"
                         "**该目录不随本仓发布**——干净环境里请用本参数指向自备副本")
    args = ap.parse_args()
    if args.tev1_dir:
        set_tev1_dir(Path(args.tev1_dir))

    # 干净环境的第一道门（2026-09-28 第五轮评审 A 线 P2）：`prep_exam.py` 上一轮补了缺项
    # 预检，本脚本当时**没跟上**——缺上游 records / results 会直接抛裸 FileNotFoundError。
    # 与本文件对上游依赖同源的还有 `data/rejev/records/*`（校验 3），但那条已按「未检查」
    # 处置（输入是**本仓自产**，不是上游资产），故这里只预检上游门。
    #
    # ⚠️ 2026-09-29 第六轮评审 B 线 P2-1：上一版只列了 **3** 个文件，而脚本实际读 **5** 个
    # ——`data/v1/instruction/{split}.jsonl` 是在 `label == "tev1"` 分支里读的（逐题渲染
    # 一致性），该分支可达（只要 tokenizer 能加载，真实复现时必然发生）。缺这两个文件时
    # 仍会抛裸 `FileNotFoundError`，即 A 线 P2 的原形态没修干净。**预检清单必须与实读
    # 集合逐一相等**——这是一条可机械核对的判据，别再靠眼看。
    need = [TEV1 / "data/v1/records/test.jsonl",
            TEV1 / "data/v1/records/policy_transfer.jsonl",
            OFFICIAL] + [TEV1 / f"data/v1/instruction/{s}.jsonl"
                         for s in ("test", "policy_transfer")]
    absent = [str(p) for p in need if not p.exists()]
    if absent:
        print("✗ 上游 tev1 目录不可用：" + str(TEV1) + "\n  缺少：\n" +
              "".join(f"    {a}\n" for a in absent) +
              "  该目录 **不随本仓发布**（见 REPRODUCING.md §6.1）。请自备一份与上游逐字节\n"
              "  相同的副本（须含 data/v1/records/{test,policy_transfer}.jsonl、\n"
              "  data/v1/instruction/{test,policy_transfer}.jsonl 与\n"
              "  evaluation/new-v1-4b/results.jsonl），用 --tev1-dir 指向它。",
              file=sys.stderr)
        return 2

    src = {"test": TEV1 / "data/v1/records/test.jsonl",
           "policy_transfer": TEV1 / "data/v1/records/policy_transfer.jsonl"}
    paper: list[dict] = []
    for split, path in src.items():
        rows = load_jsonl(path)
        for r in rows:
            r["paper_split"] = split
        paper.extend(rows)
    print(f"考卷载入：{len(paper)} 条 "
          f"({Counter(r['paper_split'] for r in paper)})")

    # ── 校验 1：与官方逐题结果对齐 ──
    official = {json.loads(l)["id"]: json.loads(l) for l in OFFICIAL.open(encoding="utf-8")}
    assert len(official) == 1300, f"官方 results 条数异常 {len(official)}"
    ours = {r["id"]: r for r in paper}
    assert set(ours) == set(official), (
        f"id 集不一致：缺 {len(set(official) - set(ours))}，多 {len(set(ours) - set(official))}")
    gold_bad = [i for i in official if official[i]["gold"] != ours[i]["answer"]]
    assert not gold_bad, f"gold 不一致 {len(gold_bad)} 条：{gold_bad[:5]}"
    # paper_split 与官方 split 字段一致
    split_bad = [i for i in official if official[i]["split"] != ours[i]["paper_split"]]
    assert not split_bad, f"split 标注不一致 {len(split_bad)} 条：{split_bad[:5]}"
    print(f"校验 1 ✅ id/gold 与官方逐条一致（{len(official)}/1300）")

    # ── 校验 2：main breakdown ──
    main_src = Counter(r["source"] for r in paper if r["paper_split"] == "test")
    assert dict(main_src) == OFFICIAL_MAIN_BREAKDOWN, (
        f"main source 分布不符：{dict(main_src)}")
    print(f"校验 2 ✅ main 分 source 与官方 README 一致：{dict(main_src)}")

    # ── 校验 3：与 ReJev 三个切分零重叠 ──
    # 与 `prep_exam.py` 校验 3 同一处置：输入在 `data/` 下、本仓不发布数据，第三方
    # 可能没有。**文件不在就记成「未检查」**（逐项打印 + 进 manifest），
    # 「未检查」与「检查过且为零」必须可区分。
    leak = {}
    leak_missing: list[str] = []
    for name in ("train", "holdout", "dev"):
        path = REPO / f"data/rejev/records/rejev-{name}.jsonl"
        if not path.exists():
            leak_missing.append(f"{name}（data/rejev/records/rejev-{name}.jsonl）")
            continue
        rows = load_jsonl(path)
        g = {(r["source"], r["group_id"]) for r in rows} & {(r["source"], r["group_id"]) for r in paper}
        s = {statehash(r["state"]) for r in rows} & {statehash(r["state"]) for r in paper}
        i_ = {r["id"] for r in rows} & set(ours)
        leak[name] = {"group": len(g), "state": len(s), "id": len(i_)}
        assert not (g or s or i_), f"{name} 与考卷重叠：{leak[name]}"
    if leak_missing:
        print(f"⚠️ 校验 3 **未覆盖**：{'、'.join(leak_missing)}——未被检查，不等于通过"
              f"（已写入 manifest 的 `leak_check_missing`）")
        # 同 `prep_exam.py` 的处置（第五轮评审两谱系各自独立报出）：缺输入时末行**不能**
        # 打成功标记，否则与上面的 warning、以及 manifest 的 `note` 三者互相矛盾。
        print(f"校验 3 ⬜ **未完整检查**：已查 {len(leak)} 个口径三口径零重叠；"
              f"另有 {len(leak_missing)} 个口径缺输入、**未检查**。"
              f"备齐 data/rejev 后重跑，方可称「零泄漏」")
    else:
        print(f"校验 3 ✅ 全部 3 个口径三口径零泄漏：{leak}")

    # ── 构造 records（messages 只含 system+user）──
    # **先不落盘**：渲染一致性校验未过之前，不得在磁盘上留下「正式考卷」，
    # 否则后续上传流程会拿到没通过校验的文件
    out_dir = REPO / "data/paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tev1-paper.jsonl"
    records = [{"id": r["id"], "source": r["source"], "answer": r["answer"],
                "paper_split": r["paper_split"], "messages": build_messages(r)[:-1]}
               for r in paper]

    # ── prompt 长度分布 ＋ 渲染一致性校验（两套 tokenizer）──
    # 渲染校验补的是「id/gold 相同但正文被改」这个盲区：用 Tev1-4B 自带 tokenizer
    # 渲染我们的 messages，须与官方 instruction 层的 prompt 逐字一致——这样被验证的
    # 就不只是 id 与答案，而是真正送进模型的输入。
    lengths = {}
    render_check = {}
    for label, repo_id, rev, hf_home in (
            ("minicpm", "openbmb/MiniCPM5-2B", "12a3808a956f869c767195e9266b59c4d21d92e2", None),
            ("tev1", "togethercomputer/Tev1-4B-experimental",
             "0b7becf017daa0e5eb222f8ce7483c8c8259c52f", str(TEV1 / ".cache"))):
        try:
            import os
            if hf_home:
                os.environ.setdefault("HF_HOME", hf_home)
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(repo_id, revision=rev)
            ls = sorted(len(tok.apply_chat_template(
                r["messages"], tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_dict=True)["input_ids"]) for r in records)
            q = lambda x: ls[min(len(ls) - 1, int(len(ls) * x))]
            lengths[label] = {"p50": q(0.5), "p95": q(0.95), "p99": q(0.99), "max": ls[-1]}
            if label == "tev1":
                # 行序与 records 一致（官方 exports 与 records 同序，已逐条核过）
                bad = []
                for split, path in src.items():
                    off_rows = load_jsonl(
                        TEV1 / f"data/v1/instruction/{split}.jsonl")
                    mine = [r for r in records if r["paper_split"] == split]
                    assert len(off_rows) == len(mine), (
                        f"{split}: 官方 instruction {len(off_rows)} 条 ≠ 我方 {len(mine)} 条"
                        f"——zip 会静默截断，拒绝继续")
                    for r, o in zip(mine, off_rows):
                        ours = tok.apply_chat_template(
                            r["messages"], tokenize=False, add_generation_prompt=True,
                            enable_thinking=False)
                        if ours != o["prompt"]:
                            bad.append(r["id"])
                render_check = {"n_checked": len(records), "prompt_mismatch": len(bad),
                                "sample": bad[:3]}
                assert not bad, f"渲染与官方 instruction 不一致：{len(bad)} 条，{bad[:5]}"
        except AssertionError:
            raise
        except Exception as e:  # noqa: BLE001
            if label == "tev1":
                # tev1 的逐字渲染校验是「同考卷」忠实度的核心证据，不可静默跳过：
                # tokenizer 拉不到就当作没有证据，中止产出（而非写出无证据的考卷）
                raise RuntimeError(
                    f"Tev1-4B 渲染校验未能执行（{type(e).__name__}: {str(e)[:200]}）——"
                    f"考卷忠实度无证据，中止") from e
            lengths[label] = f"跳过：{type(e).__name__}: {str(e)[:120]}"
    print(f"prompt 长度：{json.dumps(lengths, ensure_ascii=False)}")
    print(f"渲染一致性：{json.dumps(render_check, ensure_ascii=False)}")
    assert render_check.get("prompt_mismatch") == 0 and render_check.get("n_checked"), (
        f"渲染校验未通过或未执行：{render_check}——不产出考卷")

    # ── 全部校验通过，落盘 ──
    with out_path.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"→ {out_path}（{len(records)} 条）")

    manifest = {
        "paper": "tev1-official",
        "n": len(records),
        "splits": dict(Counter(r["paper_split"] for r in records)),
        "sources": dict(Counter(r["source"] for r in records)),
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "official_source": f"{rel(TEV1)}/data/v1/records/{{test,policy_transfer}}.jsonl",
        "official_results": rel(OFFICIAL),
        "checks": {"id_gold_alignment": "1300/1300", "breakdown": "match",
                   "leak": leak,
                   # 非空 = 这些口径**没查**（输入不在本机）。「未检查」≠「通过」。
                   "leak_check_missing": leak_missing,
                   "render_vs_official": render_check or "未执行"},
        "prompt_len": lengths,
        "tokenizers": {  # 渲染/校验所用 tokenizer 的钉定 revision（审计用）
            "minicpm": "openbmb/MiniCPM5-2B@12a3808a956f869c767195e9266b59c4d21d92e2",
            "tev1": "togethercomputer/Tev1-4B-experimental@0b7becf017daa0e5eb222f8ce7483c8c8259c52f",
        },
        "note": ("考卷为 tev1 官方开发集；"
                 # 与上面的 leak_check_missing 一致：缺输入时不得写成「三口径零泄漏」。
                 + (f"⚠️ 零泄漏**未完整检查**：以下口径缺输入、未检查——"
                    f"{'、'.join(leak_missing)}；已查 {len(leak)} 个口径三口径零重叠，"
                    if leak_missing else "对本项目 train/holdout/dev 三口径零泄漏，")
                 + "对 Tev1-4B 是其自身开发集（有主场效应，报告须标注）"),
    }
    (out_dir / "paper-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
