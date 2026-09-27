"""ReJev 003 评测：holdout/dev 双口径（v2 方案，双谱系评审修订版）。

支持 base 与 adapter（训练后）两种模型、holdout/dev 两个切分；约束口径为
状态机 prefix_allowed_tokens_fn（第 0 步只许 A–X，之后只许 EOS）；逐题落盘、
断点续跑（带运行指纹校验）、stuck 自杀；启动断言：封存文件行数＋prompt 长度 p99<1900。

**启动方式（项目铁律）**：`modal deploy` + `.spawn()`——`modal run` 会把任务绑在本地
进程上，本地被终止会向远程发 cancellation 杀掉评测（2026-09-25 用事故换来的）。
本文件的 `local_entrypoint` 仅供本地 dry 调试，不作为正式启动路径。

多臂同集评测（006）：`--data/--expect-n/--tag/--art` 四件套，**每臂必须给不同 tag**。
"""

from __future__ import annotations

import json

import modal

MODEL = "openbmb/MiniCPM5-2B"
REV = "12a3808a956f869c767195e9266b59c4d21d92e2"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWX"

app = modal.App("rejev-eval")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.7.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("transformers==5.6.2", "peft==0.21.0", "accelerate", "wandb")
)

vol = modal.Volume.from_name("rejev", create_if_missing=True)


@app.function(
    image=image, gpu="L4:1", memory=16384, timeout=21600,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("hf-token"), modal.Secret.from_name("llm101")],
)
def eval_run(model_kind: str = "base", split: str = "holdout", n: int = 0,
             data_path: str = "", tag: str = "", art: str = "/vol/artifacts/full-v1",
             expect_n: int = 0, expect_sha: str = "", code_commit: str = "unknown") -> dict:
    """model_kind: base|adapter；split: holdout|dev；n>0 时只跑前 n 条（sanity 用）。

    006 起新增：data_path 指定评测集 jsonl、expect_n 指定封存行数断言、
    tag 决定输出文件名与 wandb run 名（默认 = model_kind）、art 指定 adapter 目录。
    **多 adapter 同集评测必须给不同 tag**，否则断点续跑会串味。

    默认值下，数据路径、输出文件名、1,892 条断言与模型选择均与 003 一致；但**输出侧
    有几处新增**（wandb tags `003`→`006`、summary 多 `by_restricted` 与展平项、逐题多
    `tag`/`data_path`/`run_fingerprint`、未传 `--data` 时打印 `[info]`、旧格式结果归档重跑）。
    故「默认行为逐字不变」不成立——完整清单见 docs/plan/006_clean-weights.md「未决与风险」。
    """
    import hashlib
    import os
    import time
    import uuid

    os.environ.setdefault("WANDB_PROJECT", "rejev")
    import wandb

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert model_kind in ("base", "adapter") and split in ("holdout", "dev")
    tag = tag or model_kind

    # ── 启动断言 1：封存文件完整性（行数；封存 sha 校验在 prep 阶段已完成并随 prep-report 在档）──
    explicit_data = bool(data_path)
    if not data_path and split == "holdout":
        print("[info] 未指定 --data → 使用 003 口径的默认集 holdout-eval.jsonl（1,892 条）。"
              "本轮（006）请显式传 --data /vol/data/eval-set.jsonl --expect-n 3694 "
              "--expect-sha <封存 sha>。", flush=True)
    data_path = data_path or (
        f"/vol/data/{'holdout-eval.jsonl' if split == 'holdout' else 'dev-sample-200.jsonl'}")
    # 显式指定 --data 即 006 路径：封存集身份**不可省**。漏传会让下面的比对静默跳过，
    # 于是「同样 3,694 行的错文件」照样产出成绩（复审 P2）。003 默认路径不受影响。
    assert not (explicit_data and not expect_sha), (
        "显式指定 --data 时必须同时给 --expect-sha——封存集身份不能省，"
        "否则封存校验会静默跳过")
    # 读一次字节：内容指纹与解析同源（与 clean_train.py 同构的修复）。
    # **行数断言挡不住「同样 3,694 行的另一份文件」**——四臂若读到的不是同一份封存集，
    # 数字不可直接比较。expect_sha 由调用方从封存 manifest 硬编码传入（006 必传）。
    raw = open(data_path, "rb").read()
    file_sha = hashlib.sha256(raw).hexdigest()
    if expect_sha:
        assert file_sha == expect_sha, (
            f"评测集内容指纹不符：{file_sha} ≠ 封存 {expect_sha}——"
            f"{data_path} 不是封存的那份评测集，拒绝评测")
    rows = [json.loads(l) for l in raw.decode("utf-8").split("\n") if l.strip()]
    if split == "holdout":
        expected_n = expect_n or 1892
        assert len(rows) == expected_n, f"holdout 行数 {len(rows)} ≠ 封存 {expected_n}"
    if n > 0:
        rows = rows[:n]

    # ── 模型加载 ──
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REV, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    adapter_sha = None
    if model_kind == "adapter":
        from peft import PeftModel
        adapter_sha = hashlib.sha256(open(f"{art}/adapter_model.safetensors", "rb").read()).hexdigest()
        model = PeftModel.from_pretrained(model, art)
    # 显式选设备：from_pretrained 默认在 CPU，必须是真搬移而非 no-op。
    # 事故记录（2026-09-24）：曾写 `.to(model.device)`＝`.to("cpu")` no-op，
    # 1482-token prefill 在 CPU 上 bf16 等同挂死（白烧 ~$0.5）。
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = model.eval().to(device)
    assert device.type != "cpu", "评测函数必须跑在 GPU 上（Modal 环境）"

    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    # 生成参数集中定义：指纹与 generate 共用同一组变量。写成两处字面量会漂移——
    # 改了 generate 却忘改指纹，跨参数的续跑会静默沿用旧预测（复审 P2）。
    eos_ids = [130073, 1]
    max_new_tokens, do_sample, pad_token_id = 8, False, 1

    # ── 启动断言 2：prompt 长度 p99（抽 150）──
    import random as _r
    lens = sorted(len(tok.apply_chat_template(
        r["messages"], tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=True)["input_ids"])
        for r in _r.Random(7).sample(rows, min(150, len(rows))))
    p99 = lens[min(len(lens) - 1, int(len(lens) * 0.99))]
    assert p99 < 1900, f"prompt p99={p99} 超限"
    print(f"[assert] 行数 {len(rows)} · prompt p99={p99} · {model_kind}/{split}", flush=True)
    run = wandb.init(project="rejev", name=f"eval-{tag}-{split}",
                     config={"model_kind": model_kind, "split": split, "tag": tag,
                             "art": art if model_kind == "adapter" else None,
                             "n_target": len(rows),
                             "model": MODEL, "revision": REV, "code_commit": code_commit},
                     tags=["006", tag, split])

    # ── 逐题评测（双口径；断点续跑）──
    # 续跑指纹覆盖「数据内容 + 模型身份 + 影响推理的配置」。**只比路径不够**：
    # 同一路径被换成另一份内容时路径不变，旧预测会被静默沿用（复审 P1，2026-09-25）。
    # 指纹用**同一份已读入内存的字节**算出的 file_sha（与清单同源，且不重读文件）；
    # 截断范围由单列的 "n" 字段表达。
    fingerprint = hashlib.sha256(json.dumps(
        {"data_sha256": file_sha, "model_kind": model_kind,
         "art": art if model_kind == "adapter" else None,
         "adapter_sha256": adapter_sha, "model": MODEL, "revision": REV,
         "max_new_tokens": max_new_tokens, "eos_ids": eos_ids, "n": n,
         "do_sample": do_sample, "pad_token_id": pad_token_id},
        sort_keys=True).encode()).hexdigest()
    def _archive(path: str, kind: str) -> str:
        """把 path 挪到一个**保证不覆盖既有文件**的新名字，返回新路径。

        不直接 rename：POSIX 的 rename 会静默**覆盖**已存在的目标——同秒两次归档
        就会丢掉前一份证据。这里先探存在、再改名，重试到找到空位（复审 P2）。
        """
        for _ in range(100):
            cand = f"{path}.{kind}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
            if not os.path.exists(cand):
                os.rename(path, cand)
                return cand
        raise RuntimeError(f"归档失败：100 次重试仍撞名（{path}）")

    out_path = f"/vol/eval/{tag}-{split}.jsonl"
    done = set()
    prev = []
    try:
        with open(out_path, encoding="utf-8") as fh:
            prev = [json.loads(l) for l in fh if l.strip()]
    except FileNotFoundError:
        pass
    except json.JSONDecodeError as e:
        # 坏行**不能原地追加**：新结果会写在坏行之后，最终汇总解析必再炸。
        bad = _archive(out_path, "bad")
        vol.commit()
        print(f"[resume] 既有结果含坏行（{e}）→ 归档 {bad}，从头重跑", flush=True)
        prev = []
    if prev:
        prev_fp = prev[0].get("run_fingerprint")
        if prev_fp is None:
            # 旧格式（003 及更早）无指纹字段：无法确认它属于哪次运行 → 归档重跑，不静默沿用。
            bad = _archive(out_path, "legacy")
            vol.commit()
            print(f"[resume] 既有结果无运行指纹（旧格式）→ 归档 {bad}，从头重跑", flush=True)
        else:
            # 校验**每一条**而非只看首行：混入其它运行的记录会让 done 收进错配的 id，
            # 那些题会被静默跳过、预测混进汇总。
            mismatched = [r["id"] for r in prev if r.get("run_fingerprint") != fingerprint]
            assert not mismatched, (
                f"{len(mismatched)} 条既有记录的运行指纹与本次不符（例：{mismatched[:3]}）——"
                f"{out_path} 疑似混入其它运行的结果，拒绝续跑。先归档旧结果再重跑。")
            done = {r["id"] for r in prev}
            print(f"[resume] 跳过已完成 {len(done)} 条（指纹逐条校验通过）", flush=True)

    last_write = time.time()
    t0 = time.time()
    os.makedirs("/vol/eval", exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as out:
        for i, r in enumerate(rows):
            if r["id"] in done:
                continue
            if time.time() - last_write > 600:
                raise RuntimeError("stuck：10 分钟零写入，自杀止损")
            enc = tok.apply_chat_template(
                r["messages"], tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_dict=True)
            ids = enc["input_ids"]
            prompt_len = len(ids)
            prompt_sha = hashlib.sha256(tok.decode(ids).encode()).hexdigest()

            rec = {"id": r["id"], "source": r["source"], "gold": r["answer"],
                   "prompt_tokens": prompt_len, "prompt_sha256": prompt_sha,
                   "model": model_kind, "tag": tag, "data_path": data_path,
                   "run_fingerprint": fingerprint,
                   "adapter_sha256": adapter_sha}
            for mode in ("unconstrained", "constrained"):
                if mode == "constrained":
                    def allowed(_b, inp, pl=prompt_len):
                        return letter_ids if len(inp) - pl == 0 else eos_ids
                    kw = {"prefix_allowed_tokens_fn": allowed}
                else:
                    kw = {}
                if i == 0:
                    print(f"[i0] {mode} start (prompt={prompt_len})", flush=True)
                    t_m = time.time()
                input_ids = torch.tensor([ids], device=device)
                with torch.no_grad():
                    out_ids = model.generate(
                        input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        do_sample=do_sample, max_new_tokens=max_new_tokens,
                        eos_token_id=eos_ids, pad_token_id=pad_token_id, **kw)
                if i == 0:
                    print(f"[i0] {mode} done {time.time()-t_m:.1f}s", flush=True)
                gen = out_ids[0][prompt_len:].tolist()
                text = tok.decode(gen, skip_special_tokens=True)
                t = text.strip()
                pred = t if len(t) == 1 and t in LETTERS else None
                rec[f"gen_ids_{mode}"] = gen
                rec[f"text_{mode}"] = text[:32]
                rec[f"pred_{mode}"] = pred
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            last_write = time.time()
            wandb.log({
                f"{mode}/correct": int(rec.get(f"pred_{mode}") == rec["gold"])
                for mode in ("unconstrained", "constrained")
            } | {
                f"{mode}/valid": int(rec.get(f"pred_{mode}") is not None)
                for mode in ("unconstrained", "constrained")
            } | {"throughput_per_min": round((i + 1) / (time.time() - t0) * 60, 1)})
            if (i + 1) % 50 == 0:
                el = time.time() - t0
                n_done = i + 1 - len(done)
                print(f"[{i+1}/{len(rows)}] {el:.0f}s · {el/max(n_done,1):.2f}s/条 · "
                      f"ETA {el/max(n_done,1)*(len(rows)-i-1):.0f}s", flush=True)
                vol.commit()  # 每 50 条落一次盘，中断可从断点续跑

    # ── 汇总 ──
    results = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    elapsed = round(time.time() - t0, 1)
    summary = {"model": model_kind, "tag": tag, "split": split, "n": len(results),
               "data_path": data_path, "data_sha256": file_sha,
               "expect_sha": expect_sha, "sha_verified": bool(expect_sha),
               "art": art if model_kind == "adapter" else None,
               "code_commit": code_commit, "adapter_sha256": adapter_sha,
               "prompt_p99": p99, "elapsed_sec": elapsed,
               "per_item_sec": round(elapsed / max(len(results) - len(done), 1), 2)}
    for mode in ("unconstrained", "constrained"):
        preds = [(r[f"pred_{mode}"], r["gold"]) for r in results]
        valid = [p for p, _ in preds if p is not None]
        acc = sum(1 for p, g in preds if p == g) / len(preds)  # 分母全集，invalid 计错
        summary[mode] = {"accuracy": round(acc, 4),
                         "valid_rate": round(len(valid) / len(preds), 4)}
    # 按 source（n≥50 单列）
    by_src = {}
    for r in results:
        by_src.setdefault(r["source"], []).append(r)
    src_tbl = {}
    for s, rs in sorted(by_src.items()):
        if len(rs) < 50:
            continue
        for mode in ("unconstrained", "constrained"):
            acc = sum(1 for r in rs if r[f"pred_{mode}"] == r["gold"]) / len(rs)
            src_tbl[f"{s}:{mode}"] = {"n": len(rs), "accuracy": round(acc, 3)}
    summary["by_source"] = src_tbl

    # 三档子集（plan 006 验收判据）：受限源 vs 保留源——这是「剔源代价」的直接读数。
    # 事后拿逐题 jsonl 拼装容易出错（换个人、换个口径就变了），在脚本里一次算清。
    restricted = ("ag_news", "sst5")
    subsets = {}
    for label, keep in (("restricted", True), ("retained", False)):
        sub = [r for r in results if (r["source"] in restricted) == keep]
        if not sub:
            continue
        for mode in ("unconstrained", "constrained"):
            acc = sum(1 for r in sub if r[f"pred_{mode}"] == r["gold"]) / len(sub)
            subsets[f"{label}:{mode}"] = {"n": len(sub), "accuracy": round(acc, 4)}
    summary["by_restricted"] = subsets

    # ── dev sanity 对比（adapter 在 dev 上跑完后自动对照 base）──
    if model_kind == "adapter" and split == "dev":
        try:
            base_rows = {json.loads(l)["id"]: json.loads(l)
                         for l in open("/vol/eval/base-dev.jsonl", encoding="utf-8")}
            n_worse = sum(1 for r in results
                          if base_rows.get(r["id"], {}).get("pred_unconstrained") == r["gold"]
                          and r["pred_unconstrained"] != r["gold"])
            summary["dev_sanity"] = {"adapter_worse_than_base": n_worse}
        except FileNotFoundError:
            summary["dev_sanity"] = "base-dev 不存在，跳过对比"

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    with open(f"/vol/eval/{tag}-{split}-summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    vol.commit()
    wandb.summary.update({f"final/{k}": v for k, v in summary.items()
                          if isinstance(v, (int, float))})
    # by_restricted / by_source 是嵌套 dict，上面那行会整块过滤掉——显式展平成标量上传，
    # 否则「受限源 vs 保留源」这个剔源代价的直接读数只能在 jsonl 里离线翻。
    for group in ("by_restricted", "by_source"):
        wandb.summary.update({
            f"final/{group}_{k.replace(':', '_')}_{kk}": vv
            for k, v in (summary.get(group) or {}).items() if isinstance(v, dict)
            for kk, vv in v.items() if isinstance(vv, (int, float))})
    run.finish()
    return summary


@app.local_entrypoint()
def main(model_kind: str = "base", split: str = "holdout", n: int = 0,
         data_path: str = "", tag: str = "", art: str = "/vol/artifacts/full-v1",
         expect_n: int = 0, expect_sha: str = "") -> None:
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "uncommitted"
    result = eval_run.remote(model_kind=model_kind, split=split, n=n, data_path=data_path,
                             tag=tag, art=art, expect_n=expect_n, expect_sha=expect_sha,
                             code_commit=commit)
    print(json.dumps(result, indent=2, ensure_ascii=False))
