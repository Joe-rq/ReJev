"""S1 多种子训练入口（Issue #111）：同一份历史 train-clean，seed 42/17/29/41。

**设计意图**：四臂同数据、同配方、仅 seed 不同；RNG 协议 ① 在 `get_peft_model`
之前显式 `torch.manual_seed(seed)`（控制 LoRA 初始化），`SFTConfig(seed=seed)`
控制数据顺序。历史 `clean_train.py` 与 `artifacts/clean-r16` 保持不动，本入口
只写 `/vol/s1/` 下的新产物。

**启动方式（项目铁律）**：`modal deploy` + `src/train/trigger_multiseed.py`
的 `.spawn()`，不用 `modal run`。本文件不提供 local_entrypoint，防误用。

产物：
    /vol/s1/artifacts/{run_tag}/adapter_model.safetensors + manifest.json + loss_curve.json
    /vol/s1/checkpoints/{run_tag}/（训练中；final adapter 校验通过后清理）
    /vol/s1/attempts/{run_tag}/{attempt_id}.json（started/source_ok/identity_ok/finished/failed/finished_reused）
    /vol/s1/failures/{run_tag}/*.json（执行/协议故障记录；checkpoint 保留）

`run_tag` 始终带 `seed{N}` 前缀（R1 P1-3）；显式重训 tag 只追加后缀，不能抹掉
seed 层。续跑前逐项比对 `run-identity.json` 的 seed/数据行数/模型/revision/
模板/配方/RNG/max_length/**部署源码 SHA**（`code_commit` 只作 provenance，
不进身份——R1 P1-1）。
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid

import modal

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:  # Modal 容器内可能没有 __file__；`s1_spec` 已由 image 挂到 PYTHONPATH
    pass
import s1_spec as S  # noqa: E402

app = modal.App("rejev-multiseed-train")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.7.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("transformers==5.6.2", "trl==1.13.0", "peft==0.21.0",
                 "datasets", "accelerate", "wandb")
    .add_local_python_source("s1_spec")
    # 远端自校验冻结件（R1 P3-5）：调用方传的 SHA 不再只是自述值。
    .add_local_file("docs/plan/009_report-enhancement-experiments.md", "/root/plan009.md")
    .add_local_file("docs/plan/009-s0-manifest.json", "/root/s0-manifest.json")
    .add_local_file("docs/plan/009-s1-manifest.json", "/root/s1-manifest.json")
)

vol = modal.Volume.from_name("rejev", create_if_missing=True)

TRAINER_CHECKPOINT_REQUIRED = S.TRAINER_CHECKPOINT_FILES
CHECKPOINT_REQUIRED = S.CHECKPOINT_FILES
COMPLETE_MARKER = "_COMPLETE"
STAGING_ROOT = "/vol/s1/_staging"
TRASH_ROOT = "/vol/s1/_trash"
HEARTBEAT_ROOT = "/vol/s1/heartbeats"


def _sha_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: str, obj) -> None:
    """同目录临时文件 + fsync + os.replace；避免截断 JSON 卡死幂等入口。"""
    tmp = f"{path}.tmp-{uuid.uuid4().hex[:8]}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _checkpoint_complete(path: str) -> bool:
    """checkpoint 必须文件齐全、非空、state/step/窗口/RNG/标记全部一致。

    任何非法 JSON/类型都返回 False（不向恢复扫描抛异常，R4 P2）。
    """
    import json as _json
    for name in CHECKPOINT_REQUIRED:
        p = os.path.join(path, name)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            return False
    marker_path = os.path.join(path, COMPLETE_MARKER)
    if not os.path.isfile(marker_path):
        return False
    try:
        step = int(path.rsplit("-", 1)[-1])
        marker = _json.load(open(marker_path, encoding="utf-8"))
        state = _json.load(open(os.path.join(path, "trainer_state.json"), encoding="utf-8"))
        if int(marker.get("step", -1)) != step:
            return False
        if int(state.get("global_step", -1)) != step:
            return False
        if marker.get("trainer_state_sha256") != _sha_file(
                os.path.join(path, "trainer_state.json")):
            return False
        ok, _ = S.validate_loss_window(
            os.path.join(path, "loss_window.json"), step, S.GRAD_ACCUM)
        return ok
    except Exception:
        return False


def _prepare_checkpoints(ckpt_dir: str) -> str | None:
    """把不完整 checkpoint 改名留档；返回最新完整 checkpoint 路径或 None。"""
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(TRASH_ROOT, exist_ok=True)
    complete = []
    for name in sorted(os.listdir(ckpt_dir)):
        path = os.path.join(ckpt_dir, name)
        if not (name.startswith("checkpoint-") and os.path.isdir(path)):
            continue
        if _checkpoint_complete(path):
            complete.append(path)
        else:
            # R1 P2-10：不完整目录移出 ckpt 根，避免被 glob/轮转误认。
            new = os.path.join(
                TRASH_ROOT, f"incomplete-{os.path.basename(ckpt_dir)}-{name}-{int(time.time())}")
            os.rename(path, new)
            print(f"[resume] 不完整 checkpoint 移入 trash：{new}", flush=True)
    if not complete:
        return None
    return max(complete, key=lambda p: int(p.rsplit("-", 1)[-1]))


def _write_failure(seed: int, tag: str, data_sha: str | None, code_commit: str,
                   plan_sha: str, s0_sha: str, exc: Exception,
                   ckpt_dir: str, art: str, started_at: str,
                   attempt_id: str | None = None, call_id: str | None = None,
                   attempt_path: str | None = None) -> str:
    """执行/协议故障登记；不写题面或数据内容。"""
    import traceback
    fail_dir = f"/vol/s1/failures/{tag}"
    os.makedirs(fail_dir, exist_ok=True)
    path = f"{fail_dir}/{int(time.time())}-{uuid.uuid4().hex[:6]}.json"
    _atomic_json(path, {
        "seed": seed, "tag": tag, "data_sha256": data_sha,
        "code_commit": code_commit, "plan_sha256": plan_sha,
        "s0_manifest_sha256": s0_sha,
        "attempt_id": attempt_id, "call_id": call_id, "attempt_path": attempt_path,
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:500],
        "traceback": traceback.format_exc()[:2000],
        "checkpoint_dir": ckpt_dir, "artifact_dir": art,
        "started_at": started_at, "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "执行/协议故障；checkpoint 保留，可续跑/重试。完整执行但表现差不走此路径。",
    })
    vol.commit()
    return path


def _validate_all_rows(tok, rows: list[dict]) -> dict:
    """逐条断言：不截断、单 BOS、监督 span 恰为答案字母 + <|im_end|>。"""
    max_len = 0
    over_len = 0
    bad = []
    for i, r in enumerate(rows):
        enc = tok.apply_chat_template(
            r["messages"], chat_template=S.TRAIN_TEMPLATE, tokenize=True,
            return_assistant_tokens_mask=True, return_dict=True)
        ids, amask = enc["input_ids"], enc["assistant_masks"]
        sup = [j for j, m in enumerate(amask) if m]
        answer = r["messages"][-1]["content"]
        over_len += int(len(ids) > S.MAX_LENGTH)
        ok = (len(ids) <= S.MAX_LENGTH and len(ids) > 1
              and ids[0] == 0 and ids[1] != 0 and len(sup) == 2
              and answer in S.LABEL_TOKEN_IDS
              and ids[sup[0]] == S.LABEL_TOKEN_IDS[answer]
              and tok.decode(ids[sup[0]]) == answer
              and tok.decode([ids[sup[-1]]]) == S.IM_END)
        if not ok:
            bad.append(i)
        max_len = max(max_len, len(ids))
    assert not bad, (
        f"逐条校验失败 {len(bad)}/{len(rows)} 行（示例下标 {bad[:5]}；"
        f"max_len={max_len}，超 {S.MAX_LENGTH} 的 {over_len} 行）——"
        f"截断、双 BOS、字母 token id 或监督 span 异常；拒绝训练")
    return {"n": len(rows), "max_len": max_len,
            "over_max_length": over_len, "passed": True}


@app.function(
    image=image, gpu="L4:1", memory=16384, timeout=21600, max_containers=1,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("hf-token"), modal.Secret.from_name("llm101")],
)
def train(seed: int = 42, tag: str = "", code_commit: str = "",
          plan_sha256: str = "", s0_manifest_sha256: str = "",
          s1_manifest_sha256: str = "",
          source_multiseed_sha256: str = "",
          source_s1_spec_sha256: str = "") -> dict:
    """Train one seed arm. `tag` non-empty = explicit retrain, registered separately."""
    import hashlib
    import shutil

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    wall_t0 = time.time()
    data_sha = None
    # 前置断言失败也要有 failure 记录（R1 P3-2）：先用安全默认路径。
    run_tag = f"seed{seed}" if seed in S.SEEDS else f"invalid-seed{seed}"
    art = f"{S.ARTIFACT_ROOT}/{run_tag}"
    ckpt_dir = f"{S.CHECKPOINT_ROOT}/{run_tag}"
    call_id = attempt_id = attempt_path = None

    try:
        S.assert_seed_allowed(seed)
        assert re.fullmatch(r"[0-9a-f]{40}", code_commit), (
            "必须传入完整 40 位 git commit（不接受 unknown/local-dry/任意字符串）")
        assert plan_sha256 and s0_manifest_sha256 and s1_manifest_sha256, (
            "必须传入 plan/009、009-s0-manifest 与 009-s1-manifest 的 SHA256")
        assert all(re.fullmatch(r"[0-9a-f]{64}", x) for x in
                   (plan_sha256, s0_manifest_sha256, s1_manifest_sha256,
                    source_multiseed_sha256, source_s1_spec_sha256)), (
            "冻结件/源码 SHA 必须是 64 位 hex，不接受占位值")
        vol.reload()  # 防止复用容器读取旧 Volume 快照（R1 P1-4）
        run_tag = S.run_tag(seed, tag)
        art = S.artifact_dir(seed, tag)
        ckpt_dir = S.checkpoint_dir(seed, tag)
        os.environ.setdefault("WANDB_PROJECT", "rejev")
        os.makedirs("/vol/s1/artifacts", exist_ok=True)
        os.makedirs("/vol/s1/checkpoints", exist_ok=True)
        os.makedirs(STAGING_ROOT, exist_ok=True)
        os.makedirs(TRASH_ROOT, exist_ok=True)

        # R2 P2-7 / R3-03：重工作前（含重依赖导入）持久化 attempt-start。
        try:
            call_id = modal.current_function_call_id()
        except Exception:
            call_id = None
        attempt_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        attempt_dir = f"/vol/s1/attempts/{run_tag}"
        attempt_path = f"{attempt_dir}/{attempt_id}.json"
        os.makedirs(attempt_dir, exist_ok=True)
        _atomic_json(attempt_path, {
            "attempt_id": attempt_id, "call_id": call_id,
            "seed": seed, "tag": tag, "run_tag": run_tag,
            "code_commit": code_commit,
            "plan_sha256": plan_sha256,
            "s0_manifest_sha256": s0_manifest_sha256,
            "s1_manifest_sha256": s1_manifest_sha256,
            "started_at": started_at, "status": "started",
        })
        vol.commit()

        # 远端自校验冻结件（R1 P3-5）：SHA 不再只是调用方自述。
        assert _sha_file("/root/plan009.md") == plan_sha256, (
            "远端 plan/009 与传入 SHA 不符——部署与触发基线不一致")
        assert _sha_file("/root/s0-manifest.json") == s0_manifest_sha256, (
            "远端 009-s0-manifest 与传入 SHA 不符")
        assert _sha_file("/root/s1-manifest.json") == s1_manifest_sha256, (
            "远端 009-s1-manifest 与传入 SHA 不符")

        # 重依赖放在 try 内：导入失败也要进入 failure 记录（R2 P2-7）。
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
        from trl import SFTConfig, SFTTrainer

        # R1 P1-1 / P1-2：远程容器自算源码 SHA，并与触发方提交树 SHA 绑定；
        # 脏树 deploy 或旧部署会在这里被拒，而不是混进 manifest。
        source_sha = {
            "multiseed_train.py": _sha_file(__file__),
            "s1_spec.py": _sha_file(S.__file__),
        }
        expected_source = {
            "multiseed_train.py": source_multiseed_sha256,
            "s1_spec.py": source_s1_spec_sha256,
        }
        assert source_sha == expected_source, (
            f"部署源码 SHA 与触发基线不符：remote={source_sha} "
            f"expected={expected_source}——拒绝训练（先干净树重新 deploy）")
        att = json.load(open(attempt_path, encoding="utf-8"))
        att["source_sha256"] = source_sha
        att["status"] = "source_ok"
        _atomic_json(attempt_path, att)
        vol.commit()

        # 数据指纹每次调用都重算，且排在幂等分支之前（防卷上训练集被替换后静默复用旧产物）。
        raw = open(S.TRAIN_PATH, "rb").read()
        data_sha = hashlib.sha256(raw).hexdigest()
        assert data_sha == S.TRAIN_SHA, (
            f"训练集内容指纹不符：{data_sha} ≠ 封存 {S.TRAIN_SHA}——"
            f"{S.TRAIN_PATH} 不是 S0 锁定的 train-clean，拒绝训练")
        rows = [json.loads(l) for l in raw.decode("utf-8").split("\n") if l.strip()]
        assert len(rows) == S.TRAIN_ROWS, f"训练集条数异常 {len(rows)}（应为 {S.TRAIN_ROWS}）"

        # R1 P1-1：运行身份绑部署源码，不绑调用方自述的 commit。
        identity = S.identity_payload(seed, data_sha, len(rows), source_sha,
                                      run_tag, plan_sha256, s0_manifest_sha256)
        att = json.load(open(attempt_path, encoding="utf-8"))
        att["run_identity"] = identity
        att["status"] = "identity_ok"
        _atomic_json(attempt_path, att)
        vol.commit()
        recovered = None

        # ── 幂等：只信「manifest 可解析 + 身份一致 + 产物 SHA 全对」的已完产物 ──
        # R2 P1-2：已完成产物损坏时**不得自动从头重训并覆盖首次运行**；停下，等
        # 所有者决定是否用独立 retrain tag。art 只会在成功发布后出现（staging rename）。
        manifest_path = f"{art}/manifest.json"
        if os.path.exists(art):
            if not os.path.exists(manifest_path):
                raise RuntimeError(
                    f"产物目录 {art} 存在但没有 manifest——疑似损坏/中断的已完成产物；"
                    f"不自动重训覆盖。请复核后用显式 retrain tag（如 r2）重跑")
            try:
                prev = json.load(open(manifest_path, encoding="utf-8"))
            except Exception as exc:
                bad = f"{TRASH_ROOT}/manifest-unreadable-{run_tag}-{int(time.time())}.json"
                os.rename(manifest_path, bad)
                vol.commit()
                raise RuntimeError(
                    f"manifest 不可解析，已归档到 {bad}；不自动重训覆盖首次运行。"
                    f"请复核后用显式 retrain tag") from exc
            assert prev.get("run_identity") == identity, (
                "已存在的 manifest 运行身份与本次不符（seed/数据/模型/模板/配方/RNG/源码）"
                "——拒绝复用；重训请给显式 tag")
            artifact_sha = prev.get("artifact_sha256") or {}
            required = {"adapter_model.safetensors", "adapter_config.json"}
            assert required <= set(artifact_sha), (
                f"manifest 缺必需产物 {sorted(required - set(artifact_sha))}")
            bad_files = [name for name, sha in artifact_sha.items()
                         if not os.path.isfile(os.path.join(art, name))
                         or _sha_file(os.path.join(art, name)) != sha]
            if bad_files:
                raise RuntimeError(
                    f"产物 SHA 不符（{bad_files[:5]}）——已完成产物损坏，不自动重训覆盖；"
                    f"请复核后用显式 retrain tag")
            if os.path.isdir(ckpt_dir):
                shutil.rmtree(ckpt_dir)
                vol.commit()
            if attempt_path and os.path.exists(attempt_path):
                att = json.load(open(attempt_path, encoding="utf-8"))
                att.update({"status": "finished_reused", "artifact_dir": art,
                            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                _atomic_json(attempt_path, att)
                vol.commit()
            print(f"[idempotent] {run_tag} 已完成，直接返回 manifest", flush=True)
            return prev

        # ── 运行身份：首次训练写入 checkpoint 目录；续跑逐项比对 ──
        os.makedirs(ckpt_dir, exist_ok=True)
        ident_path = f"{ckpt_dir}/run-identity.json"
        if os.path.exists(ident_path):
            old = json.load(open(ident_path, encoding="utf-8"))
            assert old == identity, (
                "checkpoint 运行身份与本次不符（seed/数据/模型/模板/配方/RNG/commit）——"
                "拒绝续跑；请换 tag 或恢复旧产物")
        else:
            _atomic_json(ident_path, identity)
            vol.commit()
        resume = _prepare_checkpoints(ckpt_dir)
        vol.commit()

        # ── 模型与 tokenizer ──
        tok = AutoTokenizer.from_pretrained(S.MODEL, revision=S.REV)
        model = AutoModelForCausalLM.from_pretrained(
            S.MODEL, revision=S.REV, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", trust_remote_code=False)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

        # ── 全量数据检查（R1 P2-8）：逐条证明不截断、监督 span 未被吞 ──
        scan = _validate_all_rows(tok, rows)
        print(f"[data-check] {scan['n']} 行全过 · max_len={scan['max_len']}", flush=True)

        # ── RNG 协议 ①：LoRA 初始化前显式设种子 ──
        torch.manual_seed(seed)
        model = get_peft_model(model, LoraConfig(
            r=S.LORA_R, lora_alpha=S.LORA_ALPHA, lora_dropout=S.LORA_DROPOUT, bias="none",
            task_type="CAUSAL_LM", target_modules=S.TARGET_MODULES))
        model.print_trainable_parameters()

        tok.chat_template = S.TRAIN_TEMPLATE
        ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])
        cfg = SFTConfig(**S.sft_config_kwargs(
            seed, ckpt_dir, f"rejev-s1-{run_tag}"))

        class S1Trainer(SFTTrainer):
            """记录原始 per-step loss（Trainer 的 train_loss 是累计均值，不能冒充末 50 步）。"""

            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.step_losses = []
                self.nonfinite_steps = []

            def training_step(self, *args, **kwargs):
                loss = super().training_step(*args, **kwargs)
                step = int(self.state.global_step) + 1
                val = float(loss.detach().float().cpu())
                self.step_losses.append((step, val))
                if not bool(torch.isfinite(loss).all()):
                    self.nonfinite_steps.append(step)
                    raise RuntimeError(f"non-finite loss at step {step}: {val}")
                return loss

        class CheckpointCallback(TrainerCallback):
            """checkpoint 落盘后校验必需文件并写完整标记；再 commit Volume。"""

            def __init__(self):
                self.t0 = time.time()

            def on_step_end(self, args, state, control, **kw):
                if state.global_step % 200 == 0 and state.global_step > 0:
                    el = time.time() - self.t0
                    per = el / state.global_step
                    eta = per * max(state.max_steps - state.global_step, 0)
                    print(f"[eta] step {state.global_step}/{state.max_steps} · "
                          f"{per:.2f}s/step · 已 {el/60:.0f}min · ETA {eta/60:.0f}min",
                          flush=True)
                    # R1 P2-5：超时/取消时 SIGKILL 不会跑 except；心跳是最后进度。
                    os.makedirs(HEARTBEAT_ROOT, exist_ok=True)
                    _atomic_json(f"{HEARTBEAT_ROOT}/{run_tag}.json", {
                        "attempt_id": attempt_id,
                        "step": state.global_step, "max_steps": state.max_steps,
                        "status": "running",
                        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    })
                    vol.commit()

            def on_save(self, args, state, control, **kw):
                path = os.path.join(ckpt_dir, f"checkpoint-{state.global_step}")
                # R3 P0：先只检查 Trainer 自己写的文件；loss_window.json 是本回调
                # 随后才写的，不能放在这一步的清单里（否则首次 checkpoint 必失败）。
                missing = [n for n in TRAINER_CHECKPOINT_REQUIRED
                           if not os.path.isfile(os.path.join(path, n))
                           or os.path.getsize(os.path.join(path, n)) == 0]
                if missing:
                    raise RuntimeError(
                        f"checkpoint-{state.global_step} 缺 Trainer 文件 {missing}——"
                        f"不写完整标记")
                if getattr(self, "trainer", None) is None:
                    raise RuntimeError("CheckpointCallback 未绑定 trainer——无法持久化 loss 窗口")
                window_path = os.path.join(path, "loss_window.json")
                # 500 条 = GA=2 下 250 个优化步；消费者只需末 50 步（R4 P3-3）。
                _atomic_json(window_path, {
                    "schema_version": 1,
                    "saved_through_step": state.global_step,
                    "step_losses": self.trainer.step_losses[-500:],
                })
                ok, why = S.validate_loss_window(window_path, state.global_step,
                                                 S.GRAD_ACCUM)
                if not ok:
                    raise RuntimeError(
                        f"checkpoint-{state.global_step} 的 loss 窗口无效（{why}）——不写完整标记")
                # 窗口写好后，再核对完整清单（此时应包含 loss_window.json）。
                missing = [n for n in CHECKPOINT_REQUIRED
                           if not os.path.isfile(os.path.join(path, n))
                           or os.path.getsize(os.path.join(path, n)) == 0]
                if missing:
                    raise RuntimeError(
                        f"checkpoint-{state.global_step} 完整清单仍缺 {missing}")
                _atomic_json(os.path.join(path, COMPLETE_MARKER), {
                    "step": state.global_step,
                    "trainer_state_sha256": _sha_file(
                        os.path.join(path, "trainer_state.json")),
                })
                vol.commit()
                print(f"[ckpt] step {state.global_step} 已校验并 commit Volume", flush=True)

        ckpt_cb = CheckpointCallback()
        trainer = S1Trainer(model=model, train_dataset=ds, args=cfg,
                            processing_class=tok, callbacks=[ckpt_cb])
        ckpt_cb.trainer = trainer
        # 续跑时恢复 checkpoint 里的 per-step loss 窗口（R1 P1-1 / R2 P2-3）。
        resumed_window = False
        loss_window_restore_reason = None
        if resume:
            resume_step = int(resume.rsplit("-", 1)[-1])
            lw_path = os.path.join(resume, "loss_window.json")
            ok, why = S.validate_loss_window(lw_path, resume_step, S.GRAD_ACCUM)
            if ok:
                lw = json.load(open(lw_path, encoding="utf-8"))
                trainer.step_losses = [
                    (int(s), float(v)) for s, v in lw["step_losses"]
                    if int(s) <= resume_step]
                resumed_window = True
            else:
                loss_window_restore_reason = why
                print(f"[resume] loss_window 不可用（{why}），"
                      f"末 50 步窗口从续跑步起算", flush=True)
        # TRL 会在准备阶段截断并过滤全掩码样本；行数不一致即拒绝训练。
        assert len(trainer.train_dataset) == len(rows), (
            f"SFTTrainer 处理后行数 {len(trainer.train_dataset)} ≠ 原始 {len(rows)}——"
            f"发生过滤，拒绝训练")
        # R1 P2-1：训练前用真实 dataloader 长度核对步数公式，别把公式错误留到 4 小时后。
        train_dl = trainer.get_train_dataloader()
        dl_len = len(train_dl)
        expected_steps = (dl_len // S.GRAD_ACCUM) * S.EPOCHS
        formula_steps = (math.ceil(len(rows) / S.PER_DEVICE_BATCH) // S.GRAD_ACCUM) * S.EPOCHS
        assert expected_steps == formula_steps and expected_steps > 0, (
            f"步数公式不一致：dataloader 长度 {dl_len} → {expected_steps}，"
            f"行数公式 → {formula_steps}——拒绝训练")
        print(f"[steps] dataloader={dl_len} expected_steps={expected_steps}", flush=True)

        # 实弹检查：首个 batch 监督 span
        batch = next(iter(train_dl))
        sup_ids = [int(t) for t in batch["labels"][0] if t != -100]
        assert len(sup_ids) == 2 and tok.decode([sup_ids[0]]) in S.LETTERS \
            and tok.decode([sup_ids[1]]) == S.IM_END, f"batch 监督异常: {tok.decode(sup_ids)!r}"
        print(f"[batch-check] 监督 span decode = {tok.decode(sup_ids)!r}", flush=True)

        print(f"[train] seed={seed} tag={run_tag} resume={resume} "
              f"expected_steps={expected_steps}", flush=True)
        resume_step = int(resume.rsplit("-", 1)[-1]) if resume else 0
        os.makedirs(HEARTBEAT_ROOT, exist_ok=True)
        _atomic_json(f"{HEARTBEAT_ROOT}/{run_tag}.json", {
            "attempt_id": attempt_id,
            "step": resume_step, "max_steps": expected_steps, "status": "starting",
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        vol.commit()
        train_t0 = time.time()
        stats = trainer.train(resume_from_checkpoint=resume)
        trainer_secs = round(time.time() - train_t0, 1)
        peak_gb = round(torch.cuda.max_memory_reserved() / 1024**3, 2)

        # ── 健康闸（R1 P1-2）：NaN/Inf、步数、adapter 有限性 ──
        nonfinite = list(trainer.nonfinite_steps)
        assert not nonfinite, f"出现 non-finite loss 的步：{nonfinite[:10]}"
        assert trainer.state.global_step == expected_steps, (
            f"实际步数 {trainer.state.global_step} ≠ 预期 {expected_steps}——协议故障")
        n_checked = n_bad = 0
        for name, p in model.named_parameters():
            if "lora_" in name:
                n_checked += 1
                if not bool(torch.isfinite(p).all()):
                    n_bad += 1
        assert n_bad == 0, f"{n_bad}/{n_checked} 个 LoRA 张量含非有限值"
        loss_window = S.last_n_mean(trainer.step_losses, 50)
        assert loss_window["complete_50"] and loss_window["last_n_mean"] is not None, (
            f"末 50 步窗口不完整（{loss_window['n_steps_in_window']} 步，"
            f"原因 {loss_window_restore_reason}）——拒绝把不足窗口写成 last_50")
        assert loss_window["window_end_step"] == expected_steps, (
            f"per-step loss 末步 {loss_window['window_end_step']} ≠ 预期 {expected_steps}")
        _atomic_json(f"{HEARTBEAT_ROOT}/{run_tag}.json", {
            "attempt_id": attempt_id,
            "step": trainer.state.global_step, "max_steps": expected_steps,
            "status": "trained",
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        vol.commit()

        # ── 原子收尾：staging 目录 → 校验 → 发布；manifest 原子写 ──
        step_sums = S.step_loss_sums(trainer.step_losses)
        # R1 P2-10：staging/superseded 不落在 artifacts 根下，评测侧按精确 tag 拼路径。
        staging = f"{STAGING_ROOT}/{run_tag}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        model.save_pretrained(staging)
        _atomic_json(os.path.join(staging, "loss_curve.json"), {
            "seed": seed, "run_tag": run_tag,
            "per_step_from": step_sums[0][0] if step_sums else None,
            "per_step_steps": [{"step": s, "loss": v} for s, v in step_sums],
            "log_history": [
                {k: e.get(k) for k in ("step", "loss", "train_loss", "learning_rate")
                 if k in e}
                for e in trainer.state.log_history
                if "loss" in e or "train_loss" in e
            ],
            "note": ("per_step_steps = 同一优化步内 Trainer.training_step 贡献量之和"
                     "（父类已按 GA/组内 token 归一；续跑时从续跑步开始）；"
                     "log_history 覆盖整个训练，但只在 logging_steps 区间记录。"),
        })
        files = {}
        for name in sorted(os.listdir(staging)):
            if name.endswith((".safetensors", ".json")) and name != "manifest.json":
                files[name] = _sha_file(os.path.join(staging, name))

        import accelerate as _accelerate
        import datasets as _datasets
        import peft as _peft
        import transformers as _transformers
        import transformers.trainer as _transformers_trainer
        import trl as _trl
        try:
            import wandb
            wandb_run = ({"id": wandb.run.id, "url": wandb.run.url, "name": wandb.run.name}
                         if wandb.run is not None else None)
        except Exception:
            wandb_run = None

        def _src_sha(mod_file):
            try:
                return _sha_file(mod_file)
            except Exception:
                return None

        manifest = {
            "run": f"rejev-s1-{run_tag}", "run_tag": run_tag, "seed": seed,
            "date": started_at, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "plan": "docs/plan/009_report-enhancement-experiments.md（S1；A2′）",
            "code_commit": code_commit, "plan_sha256": plan_sha256,
            "s0_manifest_sha256": s0_manifest_sha256,
            "s1_manifest_path": "docs/plan/009-s1-manifest.json",
            "s1_manifest_sha256": s1_manifest_sha256,
            "source_sha256": source_sha,
            "modal_call_id": call_id,
            "attempt_id": attempt_id, "attempt_path": attempt_path,
            "attempts": sorted(
                p.name for p in Path(f"/vol/s1/attempts/{run_tag}").glob("*.json")
            ) if Path(f"/vol/s1/attempts/{run_tag}").is_dir() else [],
            "model": S.MODEL, "revision": S.REV,
            "data": {"train": "train-clean.jsonl（rejev2-train 剔 ag_news/sst5）",
                     "n": len(rows), "sha256": data_sha,
                     "expected_sha256": S.TRAIN_SHA,
                     "sha_verified": data_sha == S.TRAIN_SHA,
                     "all_rows_validated": scan,
                     "dropped_expected": S.N_DROP,
                     "dropped_note": "剔除 ag_news 1,354 + sst5 1,805；身份由 sha256 自证"},
            "data_path": S.TRAIN_PATH,
            "config": S.recipe(seed),
            "rng_protocol": S.RNG_PROTOCOL,
            "set_seed_timing": S.SET_SEED_TIMING,
            "run_identity": identity,
            "resume_from": resume,
            "recovered": recovered,
            "env": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "transformers": _transformers.__version__,
                "trl": _trl.__version__,
                "peft": _peft.__version__,
                "datasets": _datasets.__version__,
                "accelerate": _accelerate.__version__,
                "transformers_trainer_sha256": _src_sha(_transformers_trainer.__file__),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            "data_residency": (
                "训练集位于 Modal 私有 Volume rejev（/vol/data/train-clean.jsonl），"
                "非公开；与 #111「混合训练数据留本地」的字面要求存在偏离，报告限制节披露。"
            ),
            "metrics": {
                # R1 P2-2：不再保留含义模糊的 train_loss；两个口径分开命名。
                "train_loss_cumulative_trainer_return": stats.metrics.get("train_loss"),
                "train_loss_last_50": loss_window["last_n_mean"],
                "loss_window": loss_window,
                "loss_window_restored_from_resume": resumed_window,
                "loss_window_restore_reason": loss_window_restore_reason,
                "loss_window_status": ("complete" if loss_window["complete_50"]
                                       else "incomplete"),
                "loss_curve_file": "loss_curve.json",
                "loss_curve_sha256": files.get("loss_curve.json"),
                "trainer_secs": trainer_secs,
                "train_runtime": stats.metrics.get("train_runtime"),
                "wall_secs": round(time.time() - wall_t0, 1),
                "peak_gpu_gb": peak_gb,
                "steps": trainer.state.global_step,
                "steps_expected": expected_steps,
                "billing_note": "trainer_secs 是训练循环时间；实际账单以 Modal billing 为准",
            },
            "health_gate": {
                "execution_or_protocol_failure": None,
                "checks": {"nonfinite_steps": nonfinite,
                           "steps_expected": expected_steps,
                           "steps_actual": trainer.state.global_step,
                           "adapter_tensors_checked": n_checked,
                           "adapter_nonfinite": n_bad},
                "complete_but_poor": "recorded, not retrained (plan §5.1-1)",
                "first_complete_run": not tag,
                "run_kind": "first" if not tag else "retrain",
                "retrain_note": "重训必须用显式 tag 单列登记，不与本结果合并",
            },
            "checkpoint_contents": ["adapter", "optimizer", "scheduler", "trainer_state"],
            "failure_records": sorted(
                p.name for p in Path(f"/vol/s1/failures/{run_tag}").glob("*.json")
            ) if Path(f"/vol/s1/failures/{run_tag}").is_dir() else [],
            "heartbeat_path": f"{HEARTBEAT_ROOT}/{run_tag}.json",
            "last_heartbeat": (
                json.load(open(f"{HEARTBEAT_ROOT}/{run_tag}.json", encoding="utf-8"))
                if os.path.isfile(f"{HEARTBEAT_ROOT}/{run_tag}.json") else None),
            "wandb": wandb_run,
            "artifact_sha256": files,
        }
        _atomic_json(os.path.join(staging, "manifest.json"), manifest)
        for name, sha in files.items():
            assert _sha_file(os.path.join(staging, name)) == sha, f"收尾校验失败: {name}"
        if os.path.exists(art):
            os.rename(art, f"{TRASH_ROOT}/superseded-{run_tag}-{int(time.time())}")
        os.rename(staging, art)
        vol.commit()
        print(f"[artifact] adapter+manifest → {art}（SHA256 已录）", flush=True)
        for name, sha in files.items():
            assert _sha_file(os.path.join(art, name)) == sha, f"发布后校验失败: {name}"
        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir)
            vol.commit()
            print(f"[cleanup] {ckpt_dir} 已清（final adapter 校验通过后）", flush=True)
        _atomic_json(f"{HEARTBEAT_ROOT}/{run_tag}.json", {
            "attempt_id": attempt_id,
            "step": trainer.state.global_step, "max_steps": expected_steps,
            "status": "finished",
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        if attempt_path and os.path.exists(attempt_path):
            att = json.load(open(attempt_path, encoding="utf-8"))
            att.update({"status": "finished", "artifact_dir": art,
                        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            _atomic_json(attempt_path, att)
        vol.commit()
        return manifest

    except Exception as exc:
        if attempt_path and os.path.exists(attempt_path):
            try:
                att = json.load(open(attempt_path, encoding="utf-8"))
                att.update({"status": "failed",
                            "error_type": type(exc).__name__,
                            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                _atomic_json(attempt_path, att)
                vol.commit()
            except Exception:
                pass
        try:
            _write_failure(seed, run_tag, data_sha, code_commit, plan_sha256,
                           s0_manifest_sha256, exc, ckpt_dir, art, started_at,
                           attempt_id=attempt_id, call_id=call_id,
                           attempt_path=attempt_path)
        except Exception as fe:
            print(f"[failure-record] 写入失败：{type(fe).__name__}: {fe}", flush=True)
        raise


