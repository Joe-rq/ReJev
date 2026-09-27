"""ReJev 首轮 smoke 训练（002 方案，已批准）：Modal · L4 · ≤$1。

可证伪问题：管线端到端通＋协议零错误（单 BOS、loss 只落「字母＋<|im_end|>」）。
数据：data/smoke/smoke-500.jsonl（precheck.py 已断言）；评测：holdout-5.jsonl。
环境：模型缓存走容器临时盘（零 Volume 存储残留）；adapter 存 /artifacts。
wandb 只记指标（loss/lr/显存/耗时），不上传样本正文。

用法：uv run python src/train/smoke_train.py        # 经 modal run 触发远程
"""

from __future__ import annotations

import json
import subprocess

import modal

MODEL = "openbmb/MiniCPM5-2B"
REV = "12a3808a956f869c767195e9266b59c4d21d92e2"

app = modal.App("rejev-smoke-v1")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.7.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("transformers==5.6.2", "trl==1.13.0", "peft==0.21.0",
                 "datasets", "accelerate", "wandb")
)

vol = modal.Volume.from_name("rejev", create_if_missing=True)

# 与 src/train/precheck.py 完全一致的训练专用模板（官方 trl.md 结构重建）。
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
    image=image,
    gpu="L4:1",
    memory=16384,
    timeout=1800,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("hf-token"), modal.Secret.from_name("llm101")],
)
def train(code_commit: str = "unknown") -> dict:
    import os

    os.environ.setdefault("WANDB_PROJECT", "rejev")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # ── 1. 模型与 tokenizer（钉 revision；缓存走容器临时盘，用后即弃）──
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REV, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ── 2. 远程协议断言（与本地 precheck 同口径；不过即中止，不进训练）──
    rows = [json.loads(l) for l in open("/vol/data/smoke-500.jsonl", encoding="utf-8")]
    for r in rows[:3]:
        enc = tok.apply_chat_template(
            r["messages"], chat_template=TRAIN_TEMPLATE, tokenize=True,
            return_assistant_tokens_mask=True, return_dict=True)
        ids, amask = enc["input_ids"], enc["assistant_masks"]
        sup = [i for i, m in enumerate(amask) if m]
        assert ids[0] == 0 and ids[1] != 0, f"双BOS或无BOS: {ids[:4]}"
        assert len(sup) == 2 and tok.decode(ids[sup[0]]) == r["messages"][-1]["content"] \
            and tok.decode([ids[sup[-1]]]) == IM_END, f"监督span异常: sup={sup}"
    print("[precheck] 远程断言 3/3 通过：单BOS、监督=字母+<|im_end|>", flush=True)

    # ── 3. LoRA（官方 trl.md 配方）──
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    # ── 4. 数据与训练配置（002 方案：官方值，smoke 不调参）──
    tok.chat_template = TRAIN_TEMPLATE
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    cfg = SFTConfig(
        output_dir="/tmp/outputs",
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
        logging_steps=5,
        save_strategy="no",
        seed=42,
        report_to="wandb",
        run_name="rejev-smoke-v1",
    )
    trainer = SFTTrainer(model=model, train_dataset=ds, args=cfg, processing_class=tok)

    # ── 5. 训练前实弹检查：collator 出的第一个 batch，labels≠-100 恰为字母+im_end ──
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"][0]
    sup_ids = [int(t) for t in labels if t != -100]
    assert len(sup_ids) == 2 and tok.decode([sup_ids[0]]) in "ABCDEFGHIJKLMNOPQRSTUVWX" \
        and tok.decode([sup_ids[1]]) == IM_END, \
        f"batch 监督异常: decode={tok.decode(sup_ids)!r}"
    print(f"[batch-check] 监督 span decode = {tok.decode(sup_ids)!r}", flush=True)

    # ── 6. 训练 ──
    import time
    t0 = time.time()
    stats = trainer.train()
    train_secs = round(time.time() - t0, 1)
    peak_gb = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[train] loss={stats.metrics.get('train_loss')} · {train_secs}s · "
          f"峰值显存 {peak_gb}GB · {gpu_name}", flush=True)

    # ── 7. 生成冒烟：holdout 5 条，greedy、显式 eos，恢复推理模板 ──
    orig_template = AutoTokenizer.from_pretrained(MODEL, revision=REV).chat_template
    tok.chat_template = orig_template
    model.eval()
    peft_model = model
    peft_model.get_base_model().config.use_cache = True
    ho = [json.loads(l) for l in open("/vol/data/holdout-5.jsonl", encoding="utf-8")]
    results = []
    for r in ho:
        prompt_ids = tok.apply_chat_template(
            r["messages"], tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True)["input_ids"]
        device = next(peft_model.parameters()).device  # 显式取模型所在设备，防 no-op
        with torch.no_grad():
            out = peft_model.generate(
                torch.tensor([prompt_ids], device=device),
                do_sample=False, max_new_tokens=8,
                eos_token_id=[1, 130073], pad_token_id=1)
        # skip_special_tokens=True 剥掉 <|im_end|>——False 会留下字面量导致
        # 「字母+结束标记」被误判 invalid（exp001 实测教训：模型输出协议全对，解析器误杀）
        text = tok.decode(out[0][len(prompt_ids):], skip_special_tokens=True)
        letter = text.strip() if len(text.strip()) == 1 and text.strip() in "ABCDEFGHIJKLMNOPQRSTUVWX" else None
        results.append({"id": r.get("id"), "raw": repr(text[:24]), "valid": letter is not None,
                        "letter": letter, "gold": r.get("answer")})
    n_valid = sum(x["valid"] for x in results)
    print(f"[gen] 5 条 holdout 有效率 {n_valid}/5：{[(x['raw'], x['valid']) for x in results]}", flush=True)

    # ── 8. adapter 与 manifest 存 Volume（只留 adapter，几十 MB；模型缓存已走临时盘）──
    art = "/vol/artifacts/smoke-v1"
    peft_model.save_pretrained(art)
    manifest = {
        "run": "rejev-smoke-v1", "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plan": "docs/plan/002_first-smoke-training.md（已批准）",
        "code_commit": code_commit, "model": MODEL, "revision": REV,
        "data": {"train": "smoke-500 (rejev-train seed42)", "n": len(rows)},
        "config": {"lora": "r16 a32 d0.05 all-linear", "epochs": 1, "bs": "2x2",
                   "lr": 2e-4, "seq": 2048, "assistant_only_loss": True, "seed": 42},
        "metrics": {"train_loss": stats.metrics.get("train_loss"),
                    "train_secs": train_secs, "peak_gpu_gb": peak_gb,
                    "gpu": gpu_name, "gen_valid": f"{n_valid}/5"},
    }
    with open(f"{art}/manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    vol.commit()
    print(f"[artifact] adapter+manifest → Volume rejev:/artifacts/smoke-v1", flush=True)

    return manifest


@app.local_entrypoint()
def main() -> None:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "uncommitted"
    result = train.remote(code_commit=commit)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    with app.run():
        main()
