"""004 弱项探针：容量探针训练（LoRA r=16 → r=64）。

**各项配置**：LoRA rank 与 alpha 为 r64/a128；其余训练超参与 exp002 相同
（lr 2e-4、1 epoch、seq 2048、d0.05、all-linear、seed 42、L4、MiniCPM5-2B @ 12a3808）。
⚠️ **但本探针不是「只换 rank」的单变量对照**——训练集也不是 exp002 那一份：它取自
rejev2-train 另切 5% 新封存集后的剩余部分（见下行），而评测集同样换了。
故 exp004 的读数**不作 rank 归因**（2026-09-28 更正，见 docs/plan/004 的偏离记录）。

**数据**：`/vol/data/train-probe.jsonl`（rejev2-train，34,146 条）——从原 train 另切 5%
新封存集后的剩余部分；原 holdout 已降级为开发集（见 docs/plan/004_weakness-probe.md）。

**产物**：`/vol/artifacts/probe-r64`（幂等：manifest 在即返回）。checkpoint 独立命名
（`/vol/checkpoints-r64`），不与 exp002 的 full-v1 冲突。
"""

from __future__ import annotations

import json

import modal

MODEL = "openbmb/MiniCPM5-2B"
REV = "12a3808a956f869c767195e9266b59c4d21d92e2"

app = modal.App("rejev-probe-r64")

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
def train(code_commit: str = "unknown") -> dict:
    import os
    import shutil
    import time

    os.environ.setdefault("WANDB_PROJECT", "rejev")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainerCallback)
    from trl import SFTConfig, SFTTrainer

    art = "/vol/artifacts/probe-r64"
    ckpt_dir = "/vol/checkpoints-r64"

    # 幂等：final adapter 已存在则直接返回（防收尾中断后重跑重复训练）
    if os.path.exists(f"{art}/manifest.json"):
        print("[idempotent] probe-r64 已完成，直接返回 manifest", flush=True)
        return json.load(open(f"{art}/manifest.json", encoding="utf-8"))

    # ── 1. 模型与 tokenizer ──
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REV, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ── 2. 远程协议断言（与本地 precheck 同口径；不过即中止）──
    rows = [json.loads(l) for l in open("/vol/data/train-probe.jsonl", encoding="utf-8")]
    assert len(rows) == 34146, f"训练集条数异常 {len(rows)}（应为 rejev2-train 34,146）"
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

    # ── 3. LoRA 与训练配置（超参面板沿用 exp001；rank/alpha 非单变量对照，见文件头）──
    model = get_peft_model(model, LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
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
        run_name="rejev-full-v1",
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

    # ── 5. 训练（自动从 /vol/checkpoints 最新检查点续跑）──
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
    import hashlib
    model.save_pretrained(art)
    files = {}
    for name in sorted(os.listdir(art)):
        p = os.path.join(art, name)
        if os.path.isfile(p) and name.endswith((".safetensors", ".json")):
            files[name] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    manifest = {
        "run": "rejev-probe-r64", "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plan": "docs/plan/004_weakness-probe.md（容量探针：rank 16→64 与切分/评测集共变，"
                "**非单变量对照**——本轮读数不作 rank 归因）",
        "code_commit": code_commit, "model": MODEL, "revision": REV,
        "data": {"train": "train-probe.jsonl (rejev2-train，004 新切分)", "n": len(rows)},
        "config": {"lora": "r64 a128 d0.05 all-linear", "epochs": 1, "bs": "2x2",
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
        print("[cleanup] /vol/checkpoints-r64 已清（final adapter 校验通过后）", flush=True)

    return manifest


@app.local_entrypoint()
def main() -> None:
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "uncommitted"
    result = train.remote(code_commit=commit)
    print(json.dumps(result, indent=2, ensure_ascii=False))
