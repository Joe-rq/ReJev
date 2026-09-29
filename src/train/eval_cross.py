"""exp003 三方同考卷对照评测（Issue #1）。

同一张考卷上跑三方：`base`（MiniCPM5-2B）· `adapter`（ReJev-2B 全量训练）·
`tev1`（Tev1-4B 官方权重）。双解码口径（constrained / unconstrained）与 exp002
**逐位同协议**，故 base/adapter 在 holdout 上的历史结果可直接并表。

考卷：
- `tev1paper`  tev1 官方考卷 1,300（main 1,000 ＋ policy_transfer 300）
- `holdout`    ReJev 封存 holdout 1,892

任务语义层（messages）由 `src/data/render.py` 的 `build_messages` 产出，两族模型
共用同一份 messages，各自套自己的 chat template —— 这是「同考卷」的落点；
模板差异属模型固有要求，协议差异（官方经 Together API 用 regex 约束）在报告中
显式记录，不做完全同等条件声称。

**约束解码的候选集＝该题实际选项**（`LETTERS[:len(options)]`），照官方
「regex per option list」协议（README Protocol）。exp002 版允许全 A–X——实测其
约束器从未触发（约束/无约束逐位相同），故口径变化对历史结果无影响；但对选项数
< 24 的题目（官方主考有 200 道二选一）必须按题约束才与官方可比。

**续跑安全**：每个输出文件配一个 `.meta.json` 记录考卷 sha256 / 模型 revision /
adapter sha / 目标条数。续跑前比对，不符即拒绝——防止 sanity 子集、换权重或
换考卷的产物混入同一次汇总。

Tev1-4B 为多模态架构（`Qwen3_5ForConditionalGeneration`），推理只用文本路径。
权重许可 **未定**（R8）：仅本仓内部研究对照，不进 Git、不随结果发布、不上传。

用法：
    modal run src/train/eval_cross.py --model base --papers tev1paper
    modal run src/train/eval_cross.py --model tev1 --papers tev1paper,holdout
    modal run src/train/eval_cross.py --model tev1 --papers tev1paper --n 5   # sanity
"""

from __future__ import annotations

import json

import modal

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWX"
STUCK_SEC = 900      # 零写入自杀阈值（看门狗线程，覆盖首次 generate 卡死）
MAX_RUN_SEC = 10800  # 整次调用墙钟硬上限（成本有界；远低于 6h function timeout）

# family: 决定加载路径与词表常量；revision 一律钉死
MODELS = {
    "base": {"repo": "openbmb/MiniCPM5-2B", "rev": "12a3808a956f869c767195e9266b59c4d21d92e2",
             "family": "minicpm", "adapter": None},
    "adapter": {"repo": "openbmb/MiniCPM5-2B", "rev": "12a3808a956f869c767195e9266b59c4d21d92e2",
                "family": "minicpm", "adapter": "/vol/artifacts/full-v1"},
    "tev1": {"repo": "togethercomputer/Tev1-4B-experimental",
             "rev": "0b7becf017daa0e5eb222f8ce7483c8c8259c52f",
             "family": "qwen3_5", "adapter": None},
    # 004 容量探针：同底座，仅 LoRA rank 16→64（见 docs/plan/004_weakness-probe.md）
    "adapter_r64": {"repo": "openbmb/MiniCPM5-2B",
                    "rev": "12a3808a956f869c767195e9266b59c4d21d92e2",
                    "family": "minicpm", "adapter": "/vol/artifacts/probe-r64"},
    # 006 干净权重轮：剔源重训的 r16（**已发布到 HF/魔搭的就是它**，见 plan/007/008）。
    # plan/005 写四臂时它还不存在——OOD 读数若缺这一臂，报告给出的就不是**已发布模型**的表现。
    "clean_r16": {"repo": "openbmb/MiniCPM5-2B",
                  "rev": "12a3808a956f869c767195e9266b59c4d21d92e2",
                  "family": "minicpm", "adapter": "/vol/artifacts/clean-r16"},
}

# 显式常量，绝不用 tok.eos_token（MiniCPM 上是 </s>，Qwen 上是 <|im_end|>）
EOS_IDS = {"minicpm": [130073, 1], "qwen3_5": [248046, 248044]}
PAD_IDS = {"minicpm": 1, "qwen3_5": 248044}

PAPERS = {
    "tev1paper": {"path": "/vol/data/tev1-paper.jsonl", "n": 1300},
    "holdout": {"path": "/vol/data/holdout-eval.jsonl", "n": 1892},
    # 004 新封存集（原 holdout 已降级为开发集）
    "holdout2": {"path": "/vol/data/holdout2-eval.jsonl", "n": 1802},
    # 007 OOD 客卷：第三方三家构造的 2,087 题（plan/005；对本仓训练分布外）
    "exam": {"path": "/vol/data/exam.jsonl", "n": 2087},
}

app = modal.App("rejev-eval-cross-v2")
# ⚠️ 版本后缀是**规避 Modal 部署缓存**（教训来自内部权重发布脚本，不随本发布集分发）：实测同名的
# app 在代码改动后仍可能跑旧版。**改了本文件就 +1**，并把触发脚本的 from_name 同步改名。
# v2：MODELS 增 clean_r16（006 已发布模型）——plan/005 的四臂里没有它。

# 依赖钉版本：环境指纹须可复现（同代码+同权重重跑应得同结果）
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.7.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("transformers==5.6.2", "peft==0.21.0", "accelerate==1.15.0",
                 "pillow==12.3.0", "sentencepiece==0.2.2")
)

vol = modal.Volume.from_name("rejev", create_if_missing=True)


def _n_options(messages: list[dict]) -> int:
    """该题选项数：约束候选集＝实际选项，照官方 regex per option list。

    messages[1]（user）是 {"state","question","options"} 的 JSON；label 须为连续
    从 A 起（render.py 已断言），故长度即候选字母数。
    """
    payload = json.loads(messages[1]["content"])
    labels = [o["label"] for o in payload["options"]]
    assert labels == list(LETTERS[:len(labels)]), f"选项 label 非连续：{labels[:6]}"
    return len(labels)


def _load_model(spec: dict, device):
    """按 family 加载。返回 (model, tokenizer, adapter_sha)。"""
    import hashlib

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["rev"])
    adapter_sha = None
    if spec["adapter"]:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            spec["repo"], revision=spec["rev"], torch_dtype=torch.bfloat16,
            attn_implementation="sdpa")
        adapter_sha = hashlib.sha256(
            open(f"{spec['adapter']}/adapter_model.safetensors", "rb").read()).hexdigest()
        model = PeftModel.from_pretrained(base, spec["adapter"])
    else:
        # Tev1-4B 的 config 是 Qwen3_5ForConditionalGeneration（多模态），
        # 实测 AutoModelForCausalLM 即可加载文本路径；保留显式兜底。
        errors = []
        model = None
        for name in ("AutoModelForCausalLM", "Qwen3_5ForConditionalGeneration"):
            try:
                if name == "AutoModelForCausalLM":
                    cls = AutoModelForCausalLM
                else:
                    import transformers
                    cls = getattr(transformers, name)
                model = cls.from_pretrained(
                    spec["repo"], revision=spec["rev"], torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa")
                print(f"[load] {spec['repo']} via {name}", flush=True)
                break
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}: {type(e).__name__}: {str(e)[:200]}")
                print(f"[load] {name} 失败：{errors[-1]}", flush=True)
        if model is None:
            raise RuntimeError("模型加载全链路失败：\n" + "\n".join(errors))
    # 显式选设备：from_pretrained 默认在 CPU，必须是真搬移而非 no-op。
    # 事故记录（2026-09-24）：`.to(model.device)` 在 CPU 加载时是 no-op，prefill 等同挂死。
    model = model.eval().to(device)
    return model, tok, adapter_sha


@app.function(
    image=image, gpu="L4:1", memory=32768, timeout=21600,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("hf-token")],
)
def eval_run(model: str = "base", papers: str = "tev1paper", n: int = 0,
             code_commit: str = "unknown", code_sha256: str = "unknown") -> dict:
    """model: base|adapter|tev1；papers: 逗号分隔的考卷名；n>0 只跑前 n 条（sanity，独立产物）。"""
    import hashlib
    import json
    import os
    import threading
    import time

    import torch

    # 整次调用的墙钟上限从**函数入口**起算：早先版本在每张考卷循环里重置 t0，
    # 于是 2 张考卷各可跑到上限，整次调用不受该上限约束。
    t_run_start = time.time()

    # 看门狗：独立线程监控零写入，覆盖「首次 generate 卡死」这类主循环检查不到的挂起。
    # 线程**永不退出**——早先版本写 `while not watch["off"]`，置 off 后线程即终结、
    # 无法复活，第二张考卷便失去保护（双谱系评审都抓到）。改为 pause 期间重置计时。
    # 放在函数**最前**：模型加载、考卷读取等前置阶段同样可能卡死。
    watch = {"t": time.time(), "pause": True, "start": t_run_start}

    def _watchdog():
        while True:
            time.sleep(15)
            # 总时限检查先于暂停分支：收尾汇总期间 pause=True，但整次调用上限仍然有效
            if time.time() - watch["start"] > MAX_RUN_SEC:
                print(f"[watchdog] 达到整次调用上限 {MAX_RUN_SEC}s，强制停止", flush=True)
                os._exit(1)
            if watch["pause"]:
                watch["t"] = time.time()
                continue
            if time.time() - watch["t"] > STUCK_SEC:
                print(f"[watchdog] {STUCK_SEC}s 零写入，强制自杀止损", flush=True)
                os._exit(1)

    threading.Thread(target=_watchdog, daemon=True).start()

    assert model in MODELS, f"未知模型 {model}"
    spec = MODELS[model]
    paper_names = [p.strip() for p in papers.split(",") if p.strip()]
    assert all(p in PAPERS for p in paper_names), f"未知考卷 {paper_names}"
    assert 0 <= n, "n 须非负"

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    assert device.type != "cpu", "评测函数必须跑在 GPU 上（Modal 环境）"

    # ── 启动断言：考卷完整性（在加载模型前，避免白烧）──
    payloads = {}
    for p in paper_names:
        spec_p = PAPERS[p]
        rows = [json.loads(l) for l in open(spec_p["path"], encoding="utf-8")]
        assert len(rows) == spec_p["n"], f"{p} 行数 {len(rows)} ≠ 封存 {spec_p['n']}"
        payloads[p] = rows[:n] if n > 0 else rows
        print(f"[assert] 考卷 {p}: {len(payloads[p])} 条", flush=True)


    model_obj, tok, adapter_sha = _load_model(spec, device)

    # 字母单 token 断言（两族词表都须成立）
    letter_ids = []
    for c in LETTERS:
        ids = tok.encode(c, add_special_tokens=False)
        assert len(ids) == 1, f"字母 {c} 非单 token：{ids}"
        letter_ids.append(ids[0])
    eos_ids, pad_id = EOS_IDS[spec["family"]], PAD_IDS[spec["family"]]
    print(f"[assert] 字母 {len(letter_ids)}/24 单 token · eos={eos_ids} · pad={pad_id}",
          flush=True)

    os.makedirs("/vol/eval", exist_ok=True)
    summaries = {}
    for p in paper_names:
        rows = payloads[p]
        # sanity（n>0）写独立产物，绝不与全量共用文件
        suffix = f"-dev{n}" if n > 0 else ""
        out_path = f"/vol/eval/cross-{model}-{p}{suffix}.jsonl"
        meta_path = f"/vol/eval/cross-{model}-{p}{suffix}.meta.json"
        meta = {
            "runner": "eval_cross.py", "paper": p,
            "paper_sha256": hashlib.sha256(open(PAPERS[p]["path"], "rb").read()).hexdigest(),
            "model": model, "repo": spec["repo"], "revision": spec["rev"],
            "adapter_sha256": adapter_sha, "n_target": len(rows),
            "constraint": "per-item-options", "code_sha256": code_sha256,
        }
        # 续跑身份校验：meta 必须逐字段相符，否则拒绝（防子集/换权重/换考卷混入）
        # 有 jsonl 却无 meta = 来源不明（旧版脚本或手工产物），一律拒绝而非补一份新身份
        if os.path.exists(out_path) and not os.path.exists(meta_path):
            raise RuntimeError(
                f"{out_path} 存在但缺 {meta_path}，来源不明，拒绝续跑；"
                f"请先删除该 .jsonl 或补齐其 meta")
        try:
            with open(meta_path, "x", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2, ensure_ascii=False)
        except FileExistsError:
            old = json.load(open(meta_path, encoding="utf-8"))
            # 双向比较且区分「缺键」与「显式 null」（用哨兵），否则 old 多出值为
            # null 的键、或新 meta 缺值为 null 的键，都会被误判为相同
            MISSING = "<缺>"
            keys = set(meta) | set(old)
            diff = {k: (old.get(k, MISSING), meta.get(k, MISSING)) for k in keys
                    if old.get(k, MISSING) != meta.get(k, MISSING)}
            if diff:
                raise RuntimeError(
                    f"{out_path} 属于另一次运行，拒绝续跑。差异：{diff}。"
                    f"如确认要重跑，请先删除该 .jsonl 与 .meta.json") from None
        done = set()
        try:
            done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8")}
            print(f"[resume] {model}/{p} 跳过已完成 {len(done)} 条", flush=True)
        except FileNotFoundError:
            pass

        last_write = time.time()
        watch["t"] = last_write
        watch["pause"] = False  # 前置阶段结束，开始监控
        t0 = time.time()
        n_new = 0
        with open(out_path, "a", encoding="utf-8") as out:
            for i, r in enumerate(rows):
                if r["id"] in done:
                    continue
                if time.time() - last_write > STUCK_SEC:
                    raise RuntimeError(f"stuck：{STUCK_SEC}s 零写入，自杀止损（{model}/{p}）")
                if time.time() - t_run_start > MAX_RUN_SEC:
                    raise RuntimeError(
                        f"达到整次调用墙钟上限 {MAX_RUN_SEC}s，主动停止（成本有界）")
                n_opt = _n_options(r["messages"])
                assert n_opt >= 2, f"选项数 {n_opt} < 2（数据异常）：{r['id']}"
                allowed_letters = letter_ids[:n_opt]
                enc = tok.apply_chat_template(
                    r["messages"], tokenize=True, add_generation_prompt=True,
                    enable_thinking=False, return_dict=True)
                ids = enc["input_ids"]
                prompt_len = len(ids)
                prompt_sha = hashlib.sha256(tok.decode(ids).encode()).hexdigest()
                assert prompt_len < 2048, f"prompt 超长 {prompt_len}: {r['id']}"

                rec = {"id": r["id"], "paper": p, "source": r["source"],
                       "gold": r["answer"], "n_options": n_opt,
                       "prompt_tokens": prompt_len, "prompt_sha256": prompt_sha,
                       "model": model, "adapter_sha256": adapter_sha}
                if "paper_split" in r:
                    rec["paper_split"] = r["paper_split"]
                for mode in ("unconstrained", "constrained"):
                    if mode == "constrained":
                        def allowed(_b, inp, pl=prompt_len, L=allowed_letters):
                            return L if len(inp) - pl == 0 else eos_ids
                        kw = {"prefix_allowed_tokens_fn": allowed}
                    else:
                        kw = {}
                    if n_new == 0 and i == 0:
                        print(f"[i0] {mode} start (prompt={prompt_len})", flush=True)
                        t_m = time.time()
                    input_ids = torch.tensor([ids], device=device)
                    with torch.no_grad():
                        out_ids = model_obj.generate(
                            input_ids,
                            attention_mask=torch.ones_like(input_ids),
                            do_sample=False, max_new_tokens=8,
                            eos_token_id=eos_ids, pad_token_id=pad_id, **kw)
                    if n_new == 0 and i == 0:
                        print(f"[i0] {mode} done {time.time()-t_m:.1f}s", flush=True)
                    gen = out_ids[0][prompt_len:].tolist()
                    text = tok.decode(gen, skip_special_tokens=True)
                    t = text.strip()
                    rec[f"gen_ids_{mode}"] = gen
                    rec[f"text_{mode}"] = text[:32]
                    rec[f"pred_{mode}"] = t if len(t) == 1 and t in LETTERS else None
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                last_write = watch["t"] = time.time()
                n_new += 1
                if n_new % 50 == 0:
                    el = time.time() - t0
                    print(f"[{i+1}/{len(rows)}] {el:.0f}s · {el/n_new:.2f}s/条 · "
                          f"ETA {el/n_new*(len(rows)-i-1):.0f}s", flush=True)
                    vol.commit()
        watch["pause"] = True  # 本考卷跑完，暂停看门狗（避免收尾阶段误杀）

        results = [json.loads(l) for l in open(out_path, encoding="utf-8")]
        elapsed = round(time.time() - t0, 1)
        summary = {"model": model, "paper": p, "n": len(results), "n_new": n_new,
                   "code_commit": code_commit, "code_sha256": code_sha256,
                   "adapter_sha256": adapter_sha,
                   "repo": spec["repo"], "revision": spec["rev"],
                   "constraint": "per-item-options",
                   "elapsed_sec": elapsed, "per_item_sec": round(elapsed / max(n_new, 1), 2)}
        for mode in ("unconstrained", "constrained"):
            preds = [(r[f"pred_{mode}"], r["gold"]) for r in results]
            # 有效 = 预测是该题**实际存在**的选项（二选一题输出 C 应记无效）。
            # exp002 历史产物无 n_options 字段，按其当时的 A–X 口径回退。
            n_valid = sum(1 for r in results
                          if r.get(f"pred_{mode}") is not None
                          and LETTERS.index(r[f"pred_{mode}"]) < r.get("n_options", len(LETTERS)))
            summary[mode] = {
                "accuracy": round(sum(1 for x, g in preds if x == g) / len(preds), 4),
                "valid_rate": round(n_valid / len(preds), 4)}
        summary["constraint_touched"] = sum(
            1 for r in results
            if r["pred_constrained"] != r["pred_unconstrained"])
        # 分组（tev1paper 按 main/xfer；其余按 source，n≥50 单列）
        key = "paper_split" if any("paper_split" in r for r in results) else "source"
        groups = {}
        for r in results:
            groups.setdefault(r.get(key, "?"), []).append(r)
        tbl = {}
        for g, rs in sorted(groups.items()):
            if len(rs) < 50:
                continue
            for mode in ("unconstrained", "constrained"):
                acc = sum(1 for r in rs if r[f"pred_{mode}"] == r["gold"]) / len(rs)
                tbl[f"{g}:{mode}"] = {"n": len(rs), "accuracy": round(acc, 3)}
        summary[f"by_{key}"] = tbl

        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        with open(f"/vol/eval/cross-{model}-{p}{suffix}-summary.json", "w",
                  encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        vol.commit()
        summaries[p] = summary
        # 收尾计算期间暂停看门狗；还有下一张考卷才恢复
        if p != paper_names[-1]:
            watch["pause"] = False
            watch["t"] = time.time()
    watch["pause"] = True
    return summaries


@app.local_entrypoint()
def main(model: str = "base", papers: str = "tev1paper", n: int = 0) -> None:
    import hashlib
    import json
    import subprocess
    from pathlib import Path

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "uncommitted"
    # 脚本自身未提交时，rev-parse 返回的是**上一个** commit，会误导审计——标 -dirty
    dirty = subprocess.run(["git", "status", "--porcelain", "--", __file__],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        commit += "-dirty"
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = eval_run.remote(model=model, papers=papers, n=n,
                             code_commit=commit, code_sha256=sha)
    print(json.dumps(result, indent=2, ensure_ascii=False))
