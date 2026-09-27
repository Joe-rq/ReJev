"""006 干净权重轮训练（Issue #8）：Modal · L4 · 30,987 条 · 剔 ag_news/sst5。

**单一变量 = 训练集**（rejev2-train 剔源 3,159 条）。配方与 004 探针（`probe_train.py`）
逐字相同，含 LoRA rank —— 这是为了让 `clean-r*` 与 `probe-r*` 构成**纯剔源消融**
（同源同切分，唯一差异是那 3,159 条）。理由见 docs/plan/006_clean-weights.md。

**数据**：`/vol/data/train-clean.jsonl`（30,987 条，由 `src/train/prep_clean.py` 产出）。
**产物**：`/vol/artifacts/clean-r{rank}`；checkpoint `/vol/checkpoints-clean-r{rank}`
（路径自带 rank，防不同 rank 互踩幂等检查）。

用法（铁律：deploy + spawn，不用 `modal run`）：
    modal deploy src/train/clean_train.py
    # 再由秒级触发脚本 Function.from_name("rejev-clean-train","train").spawn(rank=..., ...)
"""

from __future__ import annotations

import json

import modal

MODEL = "openbmb/MiniCPM5-2B"
REV = "12a3808a956f869c767195e9266b59c4d21d92e2"
N_TRAIN = 30987  # rejev2-train 34,146 剔 ag_news(1,354)+sst5(1,805)
DATA_PATH = "/vol/data/train-clean.jsonl"
N_DROP = 3159
# 训练集内容指纹，与 data/clean/clean-split-manifest.json 的 files.train-clean.jsonl.sha256 一致。
# **只断条数是不够的**：同样 30,987 行但含 ag_news/sst5 的文件会静默通过，产物却声称已剔源。
TRAIN_SHA = "bce0ae242975f23854a34341a11275de4cec2f7ec5d889cc3f726c78fe5b664d"
# 训前锁定：**退回 exp002 的 r16/α32**。
# 原计划跟随 004 探针用 r64——但 004 判读显示 r64 在相同配方下**全面崩坏**
# （holdout2 上 27.91% vs base 51.22%；research_taxonomy_v21 93.5%→8.2%；
#  train_loss 0.880 vs exp002 的 0.457），该配方不可用。
# 代价：对照臂只能用 exp002-r16（35,948 / 切分 A），与本轮（30,987 / 切分 B）
# 差「剔源 ＋ 切分」两个变量 → **消融不主张单一变量归因**，报告须明写。
LOCKED_RANK, LOCKED_ALPHA = 16, 32

app = modal.App("rejev-clean-train")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.7.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("transformers==5.6.2", "trl==1.13.0", "peft==0.21.0",
                 "datasets", "accelerate", "wandb")
)

vol = modal.Volume.from_name("rejev", create_if_missing=True)

# 与 precheck/smoke/eval 一致的训练专用模板（官方 trl.md 结构重建）。
TRAIN_TEMPLATE = (
    "{{- bos_token }}\n"
    "{%- for message in messages %}\n"
    "{%- if message['role'] == 'assistant' %}\n"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- generation %}"
    "{{- message['content'] + '<|im_end|>' }}"
    "{%- endgeneration %}\n"
    "{{- '\\n' }}\n"
    "{%- else %}\n"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}\n"
    "{%- endif %}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "{{- '<|im_start|>assistant\\n' }}\n"
    "{%- endif %}"
)
IM_END = "<|im_end|>"


@app.function(
    image=image, gpu="L4:1", memory=16384, timeout=21600,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("hf-token"), modal.Secret.from_name("llm101")],
)
def train(rank: int = 64, alpha: int = 128, code_commit: str = "unknown") -> dict:
    """rank/alpha 默认取 004 探针配置（r64/a128）——训前锁定，见 plan 006。"""
    import hashlib
    import os
    import shutil
    import time

    assert (rank, alpha) == (LOCKED_RANK, LOCKED_ALPHA), (
        f"plan 006 训前锁定 r{LOCKED_RANK}/a{LOCKED_ALPHA}，收到 r{rank}/a{alpha}——"
        f"改 rank 会让 clean 与 probe 的对比不再是纯剔源消融（消融前提：配方逐字相同）")

    os.environ.setdefault("WANDB_PROJECT", "rejev")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainerCallback)
    from trl import SFTConfig, SFTTrainer

    art = f"/vol/artifacts/clean-r{rank}"
    ckpt_dir = f"/vol/checkpoints-clean-r{rank}"

    # 数据指纹**每次调用都重算**，且必须排在幂等分支**之前**——manifest 存在不代表
    # 卷上的训练集没被替换过。若把校验放在幂等 return 之后，重传数据后的重跑会静默
    # 返回旧 manifest，让审计误信「已剔源」（两轮复审连续抓到此项，切勿再挪后）。
    raw = open(DATA_PATH, "rb").read()
    data_sha = hashlib.sha256(raw).hexdigest()
    assert data_sha == TRAIN_SHA, (
        f"训练集内容指纹不符：{data_sha} ≠ 封存 {TRAIN_SHA}——"
        f"{DATA_PATH} 不是 prep_clean 产出的剔源集，拒绝训练"
        f"（防「条数凑巧相同但含受限源」）")
    # 训练行**从同一份已校验的字节**解析：若先哈希、后重新打开文件读，两次读取之间
    # 文件可能被换掉，就变成「用未校验的内容训练、却写下合格指纹」（复审 P1）。
    # 用 split("\n") 而非 splitlines()：后者按 Unicode 行边界切分，会把 JSON 字符串里
    # 合法的 U+2028/U+2029 也切开——prep_clean 以 ensure_ascii=False 写文件，这类字符
    # 原样落盘，切错会让合法记录解析失败（复审 P2）。
    rows = [json.loads(l) for l in raw.decode("utf-8").split("\n") if l.strip()]

    # 幂等：final adapter 已存在则直接返回（防收尾中断后重跑重复训练）。
    # 清残留 checkpoint 之前**必须先验产物**——manifest 损坏或 adapter 缺失时若先删
    # checkpoint，可恢复的材料就没了。
    if os.path.exists(f"{art}/manifest.json"):
        prev = json.load(open(f"{art}/manifest.json", encoding="utf-8"))
        artifact_sha = prev.get("artifact_sha256") or {}
        assert artifact_sha, (
            "manifest 缺 artifact_sha256——无法确认产物完整，拒绝清理 checkpoint")
        required = {"adapter_model.safetensors", "adapter_config.json"}
        assert required <= set(artifact_sha), (
            f"manifest 的产物清单缺少必需项 {sorted(required - set(artifact_sha))}——"
            f"拒绝清理 checkpoint")
        for name, sha in artifact_sha.items():
            p = os.path.join(art, name)
            assert os.path.isfile(p), f"manifest 声称有 {name} 但缺失——拒绝清理 checkpoint"
            got = hashlib.sha256(open(p, "rb").read()).hexdigest()
            assert got == sha, f"{name} 与 manifest 记录不符——拒绝清理 checkpoint"
        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir)
            vol.commit()
            print(f"[idempotent] 清掉残留 {ckpt_dir}（产物已校验）", flush=True)
        print(f"[idempotent] clean-r{rank} 已完成，直接返回 manifest", flush=True)
        return prev

    # ── 1. 模型与 tokenizer ──
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REV, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ── 2. 远程协议断言（与本地 precheck 同口径；不过即中止）──
    assert len(rows) == N_TRAIN, f"训练集条数异常 {len(rows)}（应为剔源后 {N_TRAIN}）"
    for r in rows[:3]:
        enc = tok.apply_chat_template(
            r["messages"], chat_template=TRAIN_TEMPLATE, tokenize=True,
            return_assistant_tokens_mask=True, return_dict=True)
        ids, amask = enc["input_ids"], enc["assistant_masks"]
        sup = [i for i, m in enumerate(amask) if m]
        assert ids[0] == 0 and ids[1] != 0, f"双BOS或无BOS: {ids[:4]}"
        assert len(sup) == 2 and tok.decode(ids[sup[0]]) == r["messages"][-1]["content"] \
            and tok.decode([ids[sup[-1]]]) == IM_END, f"监督span异常: sup={sup}"
    print(f"[precheck] 远程断言 3/3 通过 · 训练集 {len(rows)} 条", flush=True)

    # ── 3. LoRA 与训练配置（与 004 探针逐字相同，除 rank 传参）──
    # ⚠️ **不要在此处额外设 torch/cuda 随机种子**（复审 P1，2026-09-25）：probe_train.py
    # 没有这一步，而 clean 与 probe 的 LoRA 随机初始化必须来自同一时序，才能保证两臂
    # 「只有数据集不同」。复现性由 SFTConfig(seed=42) 承担——两臂一致。
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    tok.chat_template = TRAIN_TEMPLATE
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    cfg = SFTConfig(
        output_dir=ckpt_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        max_length=2048,
        packing=False,
        assistant_only_loss=True,
        logging_steps=50,
        save_steps=1000,
        save_total_limit=2,
        save_strategy="steps",
        seed=42,
        report_to="wandb",
        run_name=f"rejev-clean-r{rank}",
    )

    class V2Callback(TrainerCallback):
        """每 200 步 ETA 打点；每次 checkpoint 落盘即 commit Volume（断点持久化）。"""

        def __init__(self):
            self.t0 = time.time()

        def on_step_end(self, args, state, control, **kw):
            if state.global_step % 200 == 0 and state.global_step > 0:
                el = time.time() - self.t0
                per = el / state.global_step
                eta = per * max(state.max_steps - state.global_step, 0)
                print(f"[eta] step {state.global_step}/{state.max_steps} · "
                      f"{per:.2f}s/step · 已 {el/60:.0f}min · ETA {eta/60:.0f}min", flush=True)

        def on_save(self, args, state, control, **kw):
            vol.commit()
            print(f"[ckpt] step {state.global_step} 已 commit Volume", flush=True)

    trainer = SFTTrainer(model=model, train_dataset=ds, args=cfg, processing_class=tok,
                         callbacks=[V2Callback()])

    # ── 4. 实弹检查：首个 batch 监督 span ──
    batch = next(iter(trainer.get_train_dataloader()))
    sup_ids = [int(t) for t in batch["labels"][0] if t != -100]
    assert len(sup_ids) == 2 and tok.decode([sup_ids[0]]) in "ABCDEFGHIJKLMNOPQRSTUVWX" \
        and tok.decode([sup_ids[1]]) == IM_END, f"batch 监督异常: {tok.decode(sup_ids)!r}"
    print(f"[batch-check] 监督 span decode = {tok.decode(sup_ids)!r}", flush=True)

    # ── 5. 训练（自动从 ckpt_dir 最新检查点续跑）──
    resume = os.path.isdir(ckpt_dir) and any(
        n.startswith("checkpoint-") for n in os.listdir(ckpt_dir))
    print(f"[train] resume_from_checkpoint={resume}", flush=True)
    t0 = time.time()
    stats = trainer.train(resume_from_checkpoint=resume if resume else None)
    train_secs = round(time.time() - t0, 1)
    peak_gb = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    fin_loss = stats.metrics.get("train_loss")
    print(f"[train] done loss={fin_loss} · {train_secs}s · 峰值 {peak_gb}GB", flush=True)

    # ── 6. 原子收尾：adapter → SHA256 → manifest → 才清 checkpoint ──
    model.save_pretrained(art)
    files = {}
    for name in sorted(os.listdir(art)):
        p = os.path.join(art, name)
        if os.path.isfile(p) and name.endswith((".safetensors", ".json")):
            files[name] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    manifest = {
        "run": f"rejev-clean-r{rank}", "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plan": "docs/plan/006_clean-weights.md（干净权重轮：仅训练集变化）",
        "code_commit": code_commit, "model": MODEL, "revision": REV,
        "data": {"train": "train-clean.jsonl（rejev2-train 剔 ag_news/sst5）",
                 "n": len(rows), "sha256": data_sha, "expected_sha256": TRAIN_SHA,
                 "sha_verified": data_sha == TRAIN_SHA,
                 "dropped_expected": N_DROP,
                 "dropped_note": "剔除 ag_news 1,354 + sst5 1,805；训练集身份由 sha256 自证"},
        "data_path": DATA_PATH,
        "config": {"lora": f"r{rank} a{alpha} d0.05 all-linear", "epochs": 1, "bs": "2x2",
                   "lr": 2e-4, "seq": 2048, "assistant_only_loss": True, "seed": 42},
        "metrics": {"train_loss": fin_loss, "train_secs": train_secs,
                    "peak_gpu_gb": peak_gb, "steps": stats.metrics.get("global_step")},
        "checkpoint_contents": ["adapter", "optimizer", "scheduler", "trainer_state"],
        "artifact_sha256": files,
    }
    with open(f"{art}/manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    vol.commit()
    print(f"[artifact] adapter+manifest → {art}（SHA256 已录）", flush=True)

    # 验 SHA256 后才清 checkpoint
    for name, sha in files.items():
        got = hashlib.sha256(open(os.path.join(art, name), "rb").read()).hexdigest()
        assert got == sha, f"收尾校验失败: {name}"
    if os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir)
        vol.commit()
        print(f"[cleanup] {ckpt_dir} 已清（final adapter 校验通过后）", flush=True)

    return manifest


@app.local_entrypoint()
def main(rank: int = 64, alpha: int = 128) -> None:
    """**仅供本地 dry 调试。** 正式启动一律走 `modal deploy` + `.spawn()`——
    `modal run` 会把任务绑在本地进程上，本地被终止会向远程发 cancellation 杀掉训练。"""
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "uncommitted"
    result = train.remote(rank=rank, alpha=alpha, code_commit=commit)
    print(json.dumps(result, indent=2, ensure_ascii=False))
