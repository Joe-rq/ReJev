"""S1 候选评分入口（Issue #111）：最后一个 prompt 位置的选项 logits → softmax。

主口径锁定候选评分（plan/009 §3.5.1-1 的 A）。**不生成文本**；只算
`logits[0, -1]` 在本题有效选项字母上的分布，供校准、配对分析与置换检查使用。

启动方式：`modal deploy src/train/eval_candidates.py` + `.spawn()`（项目铁律）。

五项前置验证见 `precheck()`：答案位置、有效选项 mask、标签映射、保存重载、
与既有生成预测的对应关系。
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import uuid

import modal

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:
    pass
import candidate_math as CM  # noqa: E402
import jsonl_resume  # noqa: E402
import s1_spec as S  # noqa: E402

SCORING_VERSION = "s1-candidate-v2"


def _sha_file(path: str) -> str:
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

app = modal.App("rejev-eval-candidates")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.7.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("transformers==5.6.2", "peft==0.21.0", "accelerate", "wandb")
    .add_local_python_source("s1_spec", "candidate_math", "jsonl_resume")
)

vol = modal.Volume.from_name("rejev", create_if_missing=True)


def _archive(path: str, kind: str) -> str:
    import os
    import time
    for _ in range(100):
        cand = f"{path}.{kind}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        if not os.path.exists(cand):
            os.rename(path, cand)
            return cand
    raise RuntimeError(f"归档失败：100 次重试仍撞名（{path}）")


def _options_from_row(row: dict) -> list[dict]:
    if isinstance(row.get("options"), list):
        return row["options"]
    msgs = row.get("messages") or []
    if not msgs:
        raise ValueError(f"行 {row.get('id')} 既无 options 也无 messages")
    payload = json.loads(msgs[-1]["content"])
    return payload["options"]


def _messages_from_row(row: dict) -> list[dict]:
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        return msgs
    return S.build_messages(row)[:-1]


def _assert_row_consistency(row: dict) -> None:
    """同一行既有 messages 又有 options 时，两者必须指向同一 prompt。"""
    if "messages" in row and "options" in row:
        assert _messages_from_row(row) == S.build_messages(row)[:-1], (
            f"{row.get('id')} 的 messages 与 options 不一致——拒绝评分")


def _load_model(art: str, model_kind: str):
    import hashlib
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(S.MODEL, revision=S.REV)
    model = AutoModelForCausalLM.from_pretrained(
        S.MODEL, revision=S.REV, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    adapter_sha = None
    if model_kind == "adapter":
        from peft import PeftModel
        adapter_sha = hashlib.sha256(
            open(f"{art}/adapter_model.safetensors", "rb").read()).hexdigest()
        model = PeftModel.from_pretrained(model, art)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = model.eval().to(device)
    assert device.type != "cpu", "评测函数必须跑在 GPU 上（Modal 环境）"
    return tok, model, device, adapter_sha


def _score_one(tok, model, device, messages: list[dict], options: list[dict]) -> dict:
    """返回候选分布与预测；不做任何生成。"""
    import hashlib
    import torch

    labels = CM.validate_options(options)
    enc = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=True)
    ids = enc["input_ids"]
    assert len(ids) < S.MAX_TOKENS, f"prompt 长度 {len(ids)} ≥ {S.MAX_TOKENS}"
    letter_ids = []
    for label in labels:
        enc_ids = tok.encode(label, add_special_tokens=False)
        assert len(enc_ids) == 1, f"选项字母 {label} 非单 token: {enc_ids}"
        assert enc_ids[0] == S.LABEL_TOKEN_IDS[label], (
            f"选项 {label} 的 token id {enc_ids[0]} ≠ render contract "
            f"{S.LABEL_TOKEN_IDS[label]}")
        letter_ids.append(enc_ids[0])

    input_ids = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids))
    logits = out.logits[0, -1, :].float()
    mask_logits = logits[torch.tensor(letter_ids, device=device)]
    probs = torch.softmax(mask_logits, dim=-1).tolist()
    pred_i = CM.argmax(probs)
    return {
        "prompt_tokens": len(ids),
        "prompt_sha256": hashlib.sha256(tok.decode(ids).encode()).hexdigest(),
        "option_labels": labels,
        "option_keys": [o["key"] for o in options],
        "logits_by_label": {lab: float(v) for lab, v in zip(labels, mask_logits.tolist())},
        "probs_by_label": {lab: float(v) for lab, v in zip(labels, probs)},
        "pred_label": labels[pred_i],
        "pred_key": options[pred_i]["key"],
    }


@app.function(
    image=image, gpu="L4:1", memory=16384, timeout=21600, max_containers=1,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("hf-token"), modal.Secret.from_name("llm101")],
)
def score(arm: str = "base-c0", art: str = "", data_path: str = "", split: str = "",
          expect_n: int = 0, expect_sha: str = "", tag: str = "",
          code_commit: str = "unknown", n: int = 0,
          expect_adapter_sha256: str = "",
          expect_adapter_config_sha256: str = "",
          source_eval_sha256: str = "", source_math_sha256: str = "",
          source_spec_sha256: str = "") -> dict:
    """Score one arm on one frozen split. `tag` must be unique per arm/split/variant."""
    import hashlib
    import os
    import time

    import torch  # noqa: F401  (确保 GPU 环境已就绪)

    vol.reload()

    assert expect_sha, "必须给 --expect-sha（封存集身份不可省）"
    assert data_path and split and tag, "data_path / split / tag 必填"
    for what, value in (("arm", arm), ("tag", tag), ("split", split)):
        assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", value), (
            f"{what} 只允许字母数字/._-（防路径穿越），收到 {value!r}")
    model_kind = "base" if arm == "base-c0" else "adapter"
    # 六臂封闭映射：arm 与 art 不允许串用（eval R1 P1-2）
    if arm == "base-c0":
        assert not art and not expect_adapter_sha256 and not expect_adapter_config_sha256, (
            "base-c0 不能带 adapter")
    elif arm == "hist":
        assert art == "/vol/artifacts/clean-r16", f"hist 的 art 固定为 clean-r16，收到 {art}"
        assert re.fullmatch(r"[0-9a-f]{64}", expect_adapter_sha256), (
            "hist 必须给 64 位 expect_adapter_sha256")
        assert re.fullmatch(r"[0-9a-f]{64}", expect_adapter_config_sha256), (
            "hist 必须给 64 位 expect_adapter_config_sha256")
    elif arm in ("seed42", "seed17", "seed29", "seed41"):
        assert art == f"/vol/s1/artifacts/{arm}", (
            f"{arm} 的 art 必须是 /vol/s1/artifacts/{arm}，收到 {art}")
        assert re.fullmatch(r"[0-9a-f]{64}", expect_adapter_sha256), (
            f"{arm} 必须给 64 位 expect_adapter_sha256")
        assert re.fullmatch(r"[0-9a-f]{64}", expect_adapter_config_sha256), (
            f"{arm} 必须给 64 位 expect_adapter_config_sha256")
    else:
        raise AssertionError(f"未知 arm：{arm}")

    # 评分源码身份：远程自算并与触发方提交树绑定
    source_sha = {
        "eval_candidates.py": _sha_file(__file__),
        "candidate_math.py": _sha_file(CM.__file__),
        "s1_spec.py": _sha_file(S.__file__),
    }
    expected_source = {
        "eval_candidates.py": source_eval_sha256,
        "candidate_math.py": source_math_sha256,
        "s1_spec.py": source_spec_sha256,
    }
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in expected_source.values()), (
        "评分源码 SHA 必须是 64 位 hex")
    assert source_sha == expected_source, (
        f"评分源码 SHA 与触发基线不符：remote={source_sha} expected={expected_source}")
    assert re.fullmatch(r"[0-9a-f]{40}", code_commit), "必须传完整 40 位 git commit"

    raw = open(data_path, "rb").read()
    file_sha = hashlib.sha256(raw).hexdigest()
    assert file_sha == expect_sha, (
        f"评测集内容指纹不符：{file_sha} ≠ 封存 {expect_sha}——{data_path} 不是锁定文件")
    rows = [json.loads(l) for l in raw.decode("utf-8").split("\n") if l.strip()]
    if expect_n:
        assert len(rows) == expect_n, f"{split} 行数 {len(rows)} ≠ 封存 {expect_n}"
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), f"{split} 内 id 重复"
    if n > 0:
        rows = rows[:n]

    if model_kind == "adapter":
        pre_adapter_sha = _sha_file(f"{art}/adapter_model.safetensors")
        pre_config_sha = _sha_file(f"{art}/adapter_config.json")
        assert pre_adapter_sha == expect_adapter_sha256, (
            f"{arm} adapter SHA 不符：{pre_adapter_sha} ≠ {expect_adapter_sha256}")
        assert pre_config_sha == expect_adapter_config_sha256, (
            f"{arm} adapter_config SHA 不符：{pre_config_sha} ≠ "
            f"{expect_adapter_config_sha256}")
    else:
        pre_adapter_sha = pre_config_sha = None
    tok, model, device, adapter_sha = _load_model(art, model_kind)
    if model_kind == "adapter":
        assert adapter_sha == pre_adapter_sha
        adapter_config_sha = pre_config_sha
    else:
        adapter_config_sha = None
    fp = CM.fingerprint({
        "data_sha256": file_sha, "arm": arm, "tag": tag, "split": split,
        "model_kind": model_kind, "art": art if model_kind == "adapter" else None,
        "adapter_sha256": adapter_sha, "adapter_config_sha256": adapter_config_sha,
        "source_sha256": source_sha, "template_sha256": S.TEMPLATE_SHA256,
        "model": S.MODEL, "revision": S.REV,
        "scoring_version": SCORING_VERSION, "n": n,
    })

    out_path = f"/vol/eval/s1/{tag}.jsonl"
    os.makedirs("/vol/eval/s1", exist_ok=True)
    done = set()
    prev, repair, err = jsonl_resume.read_repair_jsonl(out_path)
    if err:
        bad = _archive(out_path, "bad")
        vol.commit()
        print(f"[resume] 既有结果中间含坏行 → 归档 {bad}，从头重跑", flush=True)
        prev = []
    elif repair:
        vol.commit()
        print(f"[resume] 修复 {repair}，保留 {len(prev)} 条", flush=True)
    if prev:
        mismatched = [r["id"] for r in prev if r.get("run_fingerprint") != fp]
        assert not mismatched, (
            f"{len(mismatched)} 条既有记录指纹不符（例：{mismatched[:3]}）——"
            f"{out_path} 疑似混入其它运行，拒绝续跑")
        done = {r["id"] for r in prev}
        print(f"[resume] 跳过已完成 {len(done)} 条（指纹逐条校验通过）", flush=True)

    last_write = time.time()
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            if row["id"] in done:
                continue
            if time.time() - last_write > 600:
                raise RuntimeError("stuck：10 分钟零写入，自杀止损")
            _assert_row_consistency(row)
            options = _options_from_row(row)
            messages = _messages_from_row(row)
            sc = _score_one(tok, model, device, messages, options)
            gold_label = row["answer"]
            labels = sc["option_labels"]
            assert gold_label in labels, f"行 {row['id']} 的 answer 不在选项 label 集"
            gold_key = options[labels.index(gold_label)]["key"]
            if row.get("answer_key") is not None:
                assert gold_key == row["answer_key"], (
                    f"行 {row['id']} 的 answer_key 与 answer 不一致")
            rec = {
                "id": row["id"], "source": row["source"], "split": split,
                "group_id": row.get("group_id"), "kind": row.get("kind"),
                "arm": arm, "tag": tag, "model_kind": model_kind,
                "art": art if model_kind == "adapter" else None,
                "adapter_sha256": adapter_sha,
                "adapter_config_sha256": adapter_config_sha,
                "source_sha256": source_sha, "template_sha256": S.TEMPLATE_SHA256,
                "model": S.MODEL, "revision": S.REV,
                "data_path": data_path, "data_sha256": file_sha,
                "gold_label": gold_label, "gold_key": gold_key,
                "correct": sc["pred_label"] == gold_label,
                "correct_key": sc["pred_key"] == gold_key,
                "p_gold": sc["probs_by_label"][gold_label],
                "prompt_tokens": sc["prompt_tokens"],
                "prompt_sha256": sc["prompt_sha256"],
                "option_labels": labels, "option_keys": sc["option_keys"],
                "logits_by_label": sc["logits_by_label"],
                "probs_by_label": sc["probs_by_label"],
                "probs_by_key": CM.semantic_scores(sc["probs_by_label"], options),
                "pred_label": sc["pred_label"], "pred_key": sc["pred_key"],
                "scoring_version": SCORING_VERSION,
                "run_fingerprint": fp, "code_commit": code_commit,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            last_write = time.time()
            if (i + 1) % 50 == 0:
                el = time.time() - t0
                print(f"[{i+1}/{len(rows)}] {el:.0f}s · {el/max(i+1-len(done),1):.2f}s/条 "
                      f"· ETA {el/max(i+1-len(done),1)*(len(rows)-i-1):.0f}s", flush=True)
                vol.commit()

    results = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    elapsed = round(time.time() - t0, 1)
    by_src = {}
    for r in results:
        by_src.setdefault(r["source"], []).append(r)
    summary = {
        "arm": arm, "tag": tag, "split": split, "n": len(results),
        "data_path": data_path, "data_sha256": file_sha, "expect_sha": expect_sha,
        "sha_verified": True, "expect_n": expect_n,
        "model": S.MODEL, "revision": S.REV, "scoring_version": SCORING_VERSION,
        "model_kind": model_kind, "art": art if model_kind == "adapter" else None,
        "adapter_sha256": adapter_sha, "adapter_config_sha256": adapter_config_sha,
        "source_sha256": source_sha, "template_sha256": S.TEMPLATE_SHA256,
        "code_commit": code_commit,
        "accuracy": round(sum(r["correct"] for r in results) / len(results), 6),
        "coverage": round(sum(r["pred_key"] is not None for r in results) / len(results), 6),
        "mean_p_gold": round(sum(r["p_gold"] for r in results) / len(results), 6),
        "by_source": {
            s: {"n": len(rs),
                "accuracy": round(sum(r["correct"] for r in rs) / len(rs), 6),
                "mean_p_gold": round(sum(r["p_gold"] for r in rs) / len(rs), 6)}
            for s, rs in sorted(by_src.items())
        },
        "elapsed_sec": elapsed,
        "per_item_sec": round(elapsed / max(len(results) - len(done), 1), 3),
        "prompt_p99": sorted(r["prompt_tokens"] for r in results)[
            int(len(results) * 0.99)] if results else None,
        "run_fingerprint": fp,
        "coverage_note": "候选评分对每道题都有有效选项 mask；coverage=1 是构造结果，不是格式能力",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    with open(f"/vol/eval/s1/{tag}-summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    vol.commit()
    return summary


@app.function(
    image=image, gpu="L4:1", memory=16384, timeout=21600, max_containers=1,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("hf-token"), modal.Secret.from_name("llm101")],
)
def precheck(arm: str = "base-c0", art: str = "", n: int = 200,
             code_commit: str = "unknown",
             expect_adapter_sha256: str = "",
             expect_adapter_config_sha256: str = "",
             source_eval_sha256: str = "", source_math_sha256: str = "",
             source_spec_sha256: str = "") -> dict:
    """五项前置验证：在历史 holdout1 上比对候选评分与既有生成预测。"""
    import hashlib
    import os

    vol.reload()
    assert arm in ("base-c0", "hist"), "precheck 只支持 base-c0 / hist"
    assert re.fullmatch(r"[0-9a-f]{40}", code_commit), "必须传完整 40 位 git commit"
    model_kind = "base" if arm == "base-c0" else "adapter"
    if model_kind == "base":
        assert not art and not expect_adapter_sha256 and not expect_adapter_config_sha256
    else:
        assert art == "/vol/artifacts/clean-r16", (
            "hist 必须给 --art /vol/artifacts/clean-r16")
        assert re.fullmatch(r"[0-9a-f]{64}", expect_adapter_sha256)
        assert re.fullmatch(r"[0-9a-f]{64}", expect_adapter_config_sha256)
    source_sha = {
        "eval_candidates.py": _sha_file(__file__),
        "candidate_math.py": _sha_file(CM.__file__),
        "s1_spec.py": _sha_file(S.__file__),
    }
    expected_source = {
        "eval_candidates.py": source_eval_sha256,
        "candidate_math.py": source_math_sha256,
        "s1_spec.py": source_spec_sha256,
    }
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in expected_source.values())
    assert source_sha == expected_source, (
        f"precheck 源码 SHA 与触发基线不符：remote={source_sha} expected={expected_source}")
    holdout_path = "/vol/data/holdout-eval.jsonl"
    hist_path = ("/vol/eval/006-base-holdout.jsonl" if model_kind == "base"
                 else "/vol/eval/006-clean-r16-holdout.jsonl")
    raw = open(holdout_path, "rb").read()
    holdout_sha = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(l) for l in raw.decode("utf-8").split("\n") if l.strip()]
    hist = {json.loads(l)["id"]: json.loads(l)
            for l in open(hist_path, encoding="utf-8") if l.strip()}
    rows = rows[:n]
    if model_kind == "adapter":
        pre_adapter_sha = _sha_file(f"{art}/adapter_model.safetensors")
        pre_config_sha = _sha_file(f"{art}/adapter_config.json")
        assert pre_adapter_sha == expect_adapter_sha256
        assert pre_config_sha == expect_adapter_config_sha256
    else:
        pre_adapter_sha = pre_config_sha = None
    tok, model, device, adapter_sha = _load_model(art, model_kind)
    if model_kind == "adapter":
        assert adapter_sha == pre_adapter_sha
        adapter_config_sha = pre_config_sha
    else:
        adapter_config_sha = None
    out_path = f"/vol/eval/s1/precheck-{arm}.jsonl"
    summary_path = f"/vol/eval/s1/precheck-{arm}-summary.json"
    os.makedirs("/vol/eval/s1", exist_ok=True)
    for p in (out_path, summary_path):
        if os.path.exists(p):
            bad = _archive(p, "prev")
            vol.commit()
            print(f"[precheck] 旧证据归档：{bad}", flush=True)
    prompt_match = 0
    agree_c = agree_u = valid = 0
    mismatches = []
    written = []
    with open(out_path, "w", encoding="utf-8") as out:
        for row in rows:
            _assert_row_consistency(row)
            options = _options_from_row(row)
            sc = _score_one(tok, model, device, _messages_from_row(row), options)
            h = hist[row["id"]]
            if sc["prompt_sha256"] == h["prompt_sha256"]:
                prompt_match += 1
            else:
                mismatches.append({"id": row["id"], "kind": "prompt_sha"})
            gc = sc["pred_label"] == h.get("pred_constrained")
            gu = sc["pred_label"] == h.get("pred_unconstrained")
            agree_c += int(gc)
            agree_u += int(gu)
            valid += 1
            rec = {
                "id": row["id"], "arm": arm, "prompt_sha256": sc["prompt_sha256"],
                "hist_prompt_sha256": h["prompt_sha256"],
                "pred_label": sc["pred_label"],
                "hist_constrained": h.get("pred_constrained"),
                "hist_unconstrained": h.get("pred_unconstrained"),
                "agree_constrained": gc, "agree_unconstrained": gu,
                "probs_by_label": sc["probs_by_label"],
                "adapter_sha256": adapter_sha,
            }
            written.append(rec)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if not gc and len(mismatches) < 20:
                mismatches.append({"id": row["id"], "kind": "pred",
                                   "ours": sc["pred_label"],
                                   "hist": h.get("pred_constrained")})
    vol.commit()

    # 保存/重载核对：读回必须与**写入前**逐 id 一致（不是只自洽，R4 eval P2-5）。
    reloaded = [json.loads(l) for l in open(out_path, encoding="utf-8") if l.strip()]
    assert len(reloaded) == valid
    reloaded_by_id = {r["id"]: r for r in reloaded}
    assert len(reloaded_by_id) == len(reloaded)
    assert set(reloaded_by_id) == {w["id"] for w in written}
    for w in written:
        rr = reloaded_by_id[w["id"]]
        for k in ("pred_label", "hist_constrained", "hist_unconstrained",
                  "prompt_sha256", "adapter_sha256"):
            assert rr[k] == w[k], f"{w['id']} 保存/重载后 {k} 不一致"
        for lab, p in w["probs_by_label"].items():
            assert abs(rr["probs_by_label"][lab] - p) < 1e-12, (
                f"{w['id']} 保存/重载后概率不一致")
    summary = {
        "arm": arm, "n": valid, "holdout_sha256": holdout_sha,
        "hist_path": hist_path, "adapter_sha256": adapter_sha,
        "prompt_sha_matches": prompt_match,
        "agreement_constrained": round(agree_c / valid, 6),
        "agreement_unconstrained": round(agree_u / valid, 6),
        "mismatches": mismatches, "code_commit": code_commit,
        "scoring_version": SCORING_VERSION,
        "source_sha256": source_sha,
        "adapter_config_sha256": adapter_config_sha,
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    vol.commit()
    assert prompt_match == valid, (
        f"{valid - prompt_match} 条 prompt SHA 与历史生成不一致——答案位置/渲染怀疑有误")
    assert agree_c / valid >= 0.95, (
        f"候选评分与历史 constrained 生成一致率 {agree_c}/{valid} < 0.95——停止，排查口径")
    return summary


@app.local_entrypoint()
def main(arm: str = "base-c0", art: str = "", data_path: str = "", split: str = "",
         expect_n: int = 0, expect_sha: str = "", tag: str = "",
         code_commit: str = "unknown", n: int = 0) -> None:
    """仅供本地 dry 调试；正式启动走 deploy + spawn。"""
    print(json.dumps(score.remote(
        arm=arm, art=art, data_path=data_path, split=split, expect_n=expect_n,
        expect_sha=expect_sha, tag=tag, code_commit=code_commit, n=n),
        indent=2, ensure_ascii=False))
