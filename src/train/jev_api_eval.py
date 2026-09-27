"""ReJev × 真 Jev 对照评测（Issue #2）：holdout 逐题发 TypeSafe systemone API。

映射：record.state → state；record.question → Choice.instructions；
options[{key,description}] → criteria{key: description}。返回所选 criteria key
映回 label 与 gold 比对。模型钉 jev-1.13.0；一题一请求；断点续跑（幂等跳过已完成）。
协议与响应 schema 依据：docs.typesafe.ai/api.md（端点/字段）、本仓上游第三方
实测原始响应（`resources/tev1/evaluation/public-third-party/jev.jsonl`）。

`--native-kind`：对 `kind` 为 score/noul 的记录（sst5/boolq）改用 TypeSafe 原生
问题类型发问——**探索性对照，不属预注册协议**，用来量化「统一按 Choice 发问」
给 Jev 造成的损失。默认关闭。

零新依赖（urllib 标准库）；key 从环境变量或 .env 读（永不打印、永不入库）。
用法：
  uv run python src/train/jev_api_eval.py --preflight     # 零成本验 key/额度
  uv run python src/train/jev_api_eval.py --smoke 3 --records <任意小卷> \
      --allow-custom-records --out data/exp005/smoke-$(date +%s).jsonl
  #  smoke 必须给**新 --out**：默认 smoke 路径已有产物（另一份考卷的指纹），
  #  续跑会被身份闸正确拒绝；自定义小卷还须显式 --allow-custom-records
  uv run python src/train/jev_api_eval.py                 # 全量 1,892 题（Choice）
  uv run python src/train/jev_api_eval.py --native-kind --out data/exp005/jev-native.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEALED_EXAM = REPO / "data/rejev/records/rejev-holdout.jsonl"  # 决策 8 授权出境的那一份
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODELS_ENDPOINT = "https://api.typesafe.ai/v1/models"
MODEL = "jev-1.13.0"  # 钉版本，防 latest 别名漂移（Issue #2 协议）
INPUT_USD_PER_MTOK = 0.042  # $42/Btok（官方 models 页，2026-09-24 读）
CONCURRENCY = 8
MAX_RETRY = 1  # Issue #2 纪律：HTTP 错误至多重试一次，原始终局失败保留
RETRY_DELAY_CAP = 30.0  # 尊重 retry-after，但封顶，防单题拖死整轮
NATIVE_KINDS = {"score", "noul"}  # 原生类型可用的 kind（仅 --native-kind 时生效）


def load_prior(out_path: Path) -> tuple[dict[str, dict], list[str]]:
    """读已有结果，返回 (按 id 去重的行, 需要修复的原因清单)。

    规则：
    - 同一 id 重复时**已判定的行胜过未判定的行**，同为已判定则**后写的胜**
      （文件按追加顺序＝时间顺序，后写的是重投后的新结果）。
      不是简单 last-wins——那会让「先写好行、后写坏行」的顺序把好行丢掉。
    - 只容忍**末行**是写到一半的半行；中间出现坏行说明文件被破坏，宁可直接报错。
    - **末行可解析但文件不以换行结尾**同样要报修：追加写会把它和下一行拼成一行。
      这条路径极难命中（每次 write 都带 `\\n` 且行短），但代价为零，就不留这个口子。
    """
    prior: dict[str, dict] = {}
    repairs: list[str] = []
    if not out_path.exists():
        return prior, repairs
    raw = out_path.read_text(encoding="utf-8")
    # 只按 "\n" 切：str.splitlines() 还会在 U+2028/U+2029/\x0b/\x0c/\x85 处断开，
    # 而这些字符可能合法地出现在 JSON 字符串里（如 error_detail/raw_answer），
    # 那会把一行完整记录劈成两截不可解析的"半行"，误触"中间坏行"而拒绝续跑。
    parts = raw.split("\n")
    if parts and parts[-1] == "":
        parts.pop()  # 末尾换行产生的空项，不是内容
    # 空白物理行**不能静默丢弃**：本文件由我们自己的 writer 产出，不应有空行；
    # 出现了就说明被人为编辑过或写坏了，按「报修」处理（收尾重写会清掉它）。
    blanks = sum(1 for p in parts if not p.strip())
    lines = [l.rstrip("\r") for l in parts if l.strip()]
    bad_lines = 0
    for n, line in enumerate(lines):
        try:
            row = json.loads(line)
            rid = row["id"]  # 合法 JSON 但非对象（如 []、"x"）会在这里抛 TypeError
        except (json.JSONDecodeError, TypeError, KeyError):
            assert n == len(lines) - 1, f"结果文件第 {n+1} 行损坏，非末行，拒绝续跑"
            bad_lines += 1
            continue
        if rid not in prior or row.get("correct") is not None:
            prior[rid] = row
    if bad_lines:
        repairs.append(f"{bad_lines} 行损坏的末行（截断或非对象）")
    if blanks:
        repairs.append(f"{blanks} 个空白行")
    if raw and not raw.endswith("\n"):
        repairs.append("末行缺换行符（追加会与它拼接）")
    return prior, repairs


def resume(out_path: Path, mode: str, exam_sha: str) -> dict[str, dict]:
    """续跑入口：读旧结果 → 抹除坏行 → 校验身份 → 返回可安全追加的 prior。

    抹除这一步是**必须**的：只在内存里跳过坏末行而不截断文件，下一次追加写会把新行
    接在半行后面，产出既解析不了、又可能因行数恰好相等而躲过收尾重写。
    """
    prior, repairs = load_prior(out_path)
    if repairs:
        print(f"[resume] 修复结果文件：{'；'.join(repairs)}（重写）", flush=True)
        _rewrite(out_path, prior)
    check_identity(prior, mode, exam_sha)
    return prior


def check_identity(prior: dict[str, dict], mode: str, exam_sha: str) -> None:
    """复用旧结果前，先证明它跟本次跑的是**同一次运行**。

    模式、模型、考卷任一**核验不了或不符**都拒绝混写——否则 `--out` 指向另一模式的
    历史文件时，旧行会被静默跳过、而汇总标成新模式（甚至把付费请求追加进错误的文件）。

    设计取「宁可拒」而非逐维打启发式补丁，这一点是有教训的：
      · 先给 `exam_sha256` 补了「显式 null 也算缺失」，却在 `model_requested` 上
        留了同一个洞——`null`/`""` 被当成「没有已见模型」而放行；
      · 又试过「拿同目录汇总的顶层 mode 当佐证」，但那份 summary 与结果行之间
        **没有任何绑定**，错置或过期的 summary 照样能骗过闸；
      · 还留过「封存考卷容忍缺指纹」的口子——第八轮指出它同样证明不了旧行属于
        本次考卷，而回填脚本已让所有现行产物带指纹，容忍再无存在理由。
    统一成「逐行三字段必须齐备且值相符」之后，这些洞一起消失，代价只是旧格式文件
    不能直接续跑——换个新 `--out`，或先跑 `src/train/backfill_identity_fields.py` 回填。

    抽成独立函数是为了能被直接测到：这段判据曾以「只锁判据可计算性」的方式被测，
    等于给了一个假保证——把这几个 assert 写坏，测试照样绿。
    """
    if not prior:
        return
    # 逐行三个身份字段**必须齐备且为真值**：`{"x": null}` 与缺键同等对待，无任何豁免。
    missing = {k: sum(1 for r in prior.values() if not r.get(k))
               for k in ("mode", "model_requested", "exam_sha256")}
    assert not any(missing.values()), (
        f"结果文件里 {'、'.join(f'{v} 行缺 {k}' for k, v in missing.items() if v)}"
        f"（旧格式或空值）——无法证明它们与本次（{mode} · {MODEL} · 考卷 {exam_sha[:12]}…）"
        f"同源，拒绝混写。请换 --out，或跑 src/train/backfill_identity_fields.py 回填后重跑。")
    seen_mode = {r["mode"] for r in prior.values()} - {mode}
    assert not seen_mode, (
        f"结果文件里已有 {sorted(seen_mode)} 模式的行，与本次 {mode} 不符——换 --out 后再跑")
    seen_model = {r["model_requested"] for r in prior.values()} - {MODEL}
    assert not seen_model, (
        f"结果文件里已有 {sorted(seen_model)} 的行，与本次 {MODEL} 不符")
    stale = {r["exam_sha256"] for r in prior.values()} - {exam_sha}
    assert not stale, (
        f"结果文件里已有其它考卷的行（sha {sorted(s[:12] for s in stale)}），"
        f"本次考卷为 {exam_sha[:12]}…——不可混写")


def _rewrite(out_path: Path, rows: dict[str, dict]) -> None:
    """按 id 把结果文件原子重写成 rows（同目录建 tmp 再 replace，避免半截文件）。"""
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows.values()),
                   encoding="utf-8")
    tmp.replace(out_path)


def verify_exam(records: list[dict], path: Path) -> str:
    """发送前闸：把「送出去的到底是哪 1,892 题」变成可核验的断言。

    出境授权是按**封存考卷**给的（决策 8），所以必须先证明手上这份就是封存的那份，
    再用 manifest 的逐 source 计数交叉核对。返回考卷 sha256，供归档引用。
    """
    manifest = json.loads((REPO / "data/rejev/holdout-manifest.json").read_text(encoding="utf-8"))
    want = manifest["files"]["rejev-holdout.jsonl"]["sha256"]
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    assert got == want, f"考卷哈希不符：{got} ≠ {want}（拒绝发送未授权的题集）"
    assert len(records) == manifest["split"]["holdout_n"], f"题数 {len(records)} 与封存不符"
    assert len({r["id"] for r in records}) == len(records), "考卷有重复 id"
    assert len({r["group_id"] for r in records}) == manifest["split"]["holdout_groups"], \
        "分组数与封存不符"
    assert Counter(r["source"] for r in records) == Counter(manifest["split"]["holdout_sources"]), \
        "逐 source 计数与封存不符"
    print(f"[guard] 考卷与封存 manifest 逐项一致 · sha256 {got[:12]}… · "
          f"{len(records)} 题 / {manifest['split']['holdout_groups']} 组")
    return got


def load_key() -> str:
    if os.environ.get("TYPESAFE_API_KEY", "").strip():
        return os.environ["TYPESAFE_API_KEY"].strip()
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("TYPESAFE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
                if key:
                    return key
    raise SystemExit("未找到 TYPESAFE_API_KEY（环境变量或 .env；键值不会被打印）")


def build_payload(record: dict, native: bool = False) -> dict:
    """Choice：所有记录一律按「从给定选项里选一个」发问（预注册协议）。

    native=True 时，kind 为 score/noul 的记录改用原生问题类型（探索性对照）：
      score → criteria 为有序数组；noul → criteria {true,false}，取 yes/no 两选项描述。
    """
    kind = record.get("kind", "choice")
    opts = record["options"]
    if native and kind in NATIVE_KINDS:
        if kind == "score":
            q = {"type": "score", "instructions": record["question"],
                 "criteria": [o["description"] for o in opts]}
        else:  # noul：仅支持 yes/no 两选项
            labels = {o["key"]: o["label"] for o in opts}
            if set(labels) != {"yes", "no"}:
                raise ValueError(f"noul 期望 yes/no 两选项，实为 {sorted(labels)}")
            by_key = {o["key"]: o["description"] for o in opts}
            q = {"type": "noul", "instructions": record["question"],
                 "criteria": {"true": by_key["yes"], "false": by_key["no"]}}
    else:
        q = {"type": "choice", "instructions": record["question"],
             "criteria": {o["key"]: o["description"] for o in opts}}
    return {"model": MODEL, "state": record["state"], "questions": {"decision": q}}


def _request(url: str, key: str, payload: dict | None, timeout: int = 90):
    req = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode() if payload else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": "rejev-exp005/0.1"})
    return urllib.request.urlopen(req, timeout=timeout)


def retry_delay(headers, attempt: int) -> tuple[float, str]:
    """算重试等待。Retry-After 可能是秒数，也可能是 HTTP 日期——两种都认，认不出就退避。"""
    raw = (headers.get("retry-after") if headers else None) or ""
    raw = raw.strip()
    if raw:
        try:
            return min(float(raw), RETRY_DELAY_CAP), "retry-after(秒)"
        except ValueError:
            try:
                when = parsedate_to_datetime(raw)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                secs = (when - datetime.now(timezone.utc)).total_seconds()
                return max(0.0, min(secs, RETRY_DELAY_CAP)), "retry-after(日期)"
            except (TypeError, ValueError):
                pass
        return min(2.0 ** attempt, RETRY_DELAY_CAP), "退避(Retry-After 无法解析)"
    return min(2.0 ** attempt, RETRY_DELAY_CAP), "退避"


def call_api(payload: dict, key: str) -> tuple[dict, int]:
    """返回 (响应体, 尝试次数)。HTTP/网络错误按 Issue #2 纪律重试一次。"""
    last: Exception | None = None
    for attempt in range(1, MAX_RETRY + 2):
        try:
            with _request(ENDPOINT, key, payload) as resp:
                return json.load(resp), attempt
        except urllib.error.HTTPError as exc:
            last = exc
            if attempt > MAX_RETRY:
                break
            delay, why = retry_delay(exc.headers, attempt)
            print(f"[retry] HTTP {exc.code} · 第 {attempt} 次失败 · 等 {delay:.1f}s（{why}）",
                  flush=True)
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            if attempt > MAX_RETRY:
                break
            delay = min(2.0 ** attempt, RETRY_DELAY_CAP)
            print(f"[retry] {type(exc).__name__} · 第 {attempt} 次失败 · 等 {delay:.1f}s", flush=True)
            time.sleep(delay)
    assert last is not None
    raise last


def preflight(key: str) -> int:
    """GET /v1/models：零成本验证 key 有效（不打印 key）。"""
    with _request(MODELS_ENDPOINT, key, None, timeout=30) as resp:
        body = json.load(resp)
    names = [m.get("name") for m in body.get("models", [])]
    print(f"[preflight] 鉴权通过 · 别名列表 {names} · 本次钉版本 {MODEL}")
    if MODEL not in names:
        print("[preflight] 注：版本号未出现在别名列表（官方文档明示仍可直用，仅别名会被列出）")
    return 0


def parse_answer(record: dict, ans: dict, native: bool) -> tuple[str, str, dict]:
    """把 TypeSafe 答案归一成 (所选 label, 所选 option key, {label: 概率})。

    三种问题类型都归一到 label 空间，便于下游统一做校准分析。
    任何取不到/越界的情形抛 ValueError——由调用方标错并留原始响应，不静默判错。
    """
    opts = record["options"]
    kind = record.get("kind", "choice")
    probs_raw = ans.get("probabilities") or {}
    if native and kind in NATIVE_KINDS and ans.get("type") == kind:
        if kind == "score":
            if probs_raw:
                idx = max(probs_raw, key=lambda k: probs_raw[k])
            elif ans.get("score") is not None:
                idx = str(int(round(float(ans["score"]))))
            else:
                raise ValueError("score 答案既无 probabilities 也无 score")
            i = int(idx)
            if not 0 <= i < len(opts):
                raise ValueError(f"score 下标 {i} 越界（共 {len(opts)} 级）")
            chosen = opts[i]
        else:  # noul
            p = ans.get("noul")
            if not isinstance(p, (int, float)):
                raise ValueError(f"noul 答案缺 noul 字段：{sorted(ans)}")
            chosen = next(o for o in opts if o["key"] == ("yes" if p >= 0.5 else "no"))
    else:
        chosen_key = ans.get("choice")
        by_key = {o["key"]: o for o in opts}
        if chosen_key not in by_key:
            raise ValueError(f"choice={chosen_key!r} 不在选项 key 内：{sorted(by_key)}")
        chosen = by_key[chosen_key]

    label_of = {o["key"]: o["label"] for o in opts}
    if kind == "noul" and native and ans.get("type") == "noul":
        p = float(ans["noul"])
        probs = {label_of["yes"]: p, label_of["no"]: round(1 - p, 6)}
    elif kind == "score" and native and ans.get("type") == "score":
        probs = {opts[int(k)]["label"]: v for k, v in probs_raw.items()}
    else:
        probs = {label_of[k]: v for k, v in probs_raw.items() if k in label_of}
    return chosen["label"], chosen["key"], probs


def run_one(record: dict, key: str, native: bool = False) -> dict:
    out = {"id": record["id"], "source": record["source"],
           "kind": record.get("kind", "choice"), "gold": record["answer"],
           "mode": "native-kind" if native else "choice",
           "model_requested": MODEL}
    try:
        body, attempts = call_api(build_payload(record, native), key)
        ans = body["answers"]["decision"]
        label, picked_key, probs = parse_answer(record, ans, native)
        out.update({
            "jev_choice_key": picked_key,
            "jev_label": label,
            "confidence": ans.get("confidence"),
            "probs": probs,
            "ans_type": ans.get("type"),
            "model_reported": body.get("model"),
            "usage": body.get("usage"),
            "attempts": attempts,
        })
        out["correct"] = label == record["answer"]
    except Exception as exc:  # noqa: BLE001 —— 任何异常都要落盘成一行可审计的记录
        detail = getattr(exc, "code", "")
        raw = locals().get("ans")
        if isinstance(raw, dict):
            out["raw_answer"] = raw
        if isinstance(exc, ValueError) and isinstance(raw, dict):
            # 传输通了、模型也答了，只是答案不在选项集内（或类型不符）——这是**答错**，
            # 不是 API 故障。计错并留在分母，与本地侧「无效输出判错」同口径；
            # 若当故障剔除，会因「模型乱答」而虚高准确率。
            out["error"] = f"invalid_answer:{exc}"
            out["correct"] = False
        else:
            out["error"] = f"{type(exc).__name__}{':' + str(detail) if detail else ''}"
            out["error_detail"] = str(exc)[:200]
            out["correct"] = None
    return out


def wilson(k: int, n: int, z: float = 1.95996398454) -> list[float]:
    """Wilson 95% 区间（与上游第三方对照同口径，便于并表）。"""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return [round(mid - half, 4), round(mid + half, 4)]


def summarize(rows: list[dict], elapsed: float) -> dict:
    """按**全量文件**统计（续跑不漏历史错误）。rows 为已落盘的全部行。"""
    scored = [r for r in rows if r.get("correct") is not None]
    errs = [r for r in rows if r.get("correct") is None]
    invalid = [r for r in scored if str(r.get("error", "")).startswith("invalid_answer")]
    n = len(scored)
    k = sum(bool(r["correct"]) for r in scored)
    tok_in = sum((r.get("usage") or {}).get("input_tokens", 0) for r in rows)
    cost_api = sum((r.get("usage") or {}).get("cost") or 0 for r in rows)

    def acc_of(sub: list[dict]) -> dict:
        s = [r for r in sub if r.get("correct") is not None]
        kk = sum(bool(r["correct"]) for r in s)
        return {"n": len(s), "acc": round(kk / len(s), 4) if s else None,
                "wilson95": wilson(kk, len(s)) if s else None}

    conf_ok = [r["confidence"] for r in scored if r["correct"] and r.get("confidence") is not None]
    conf_bad = [r["confidence"] for r in scored if not r["correct"] and r.get("confidence") is not None]
    by_src = {s: acc_of([r for r in rows if r.get("source") == s])
              for s in sorted({r.get("source") for r in rows})}
    by_kind = {s: acc_of([r for r in rows if r.get("kind") == s])
               for s in sorted({r.get("kind") for r in rows})}
    return {
        "model_requested": MODEL,
        "model_reported": sorted({r["model_reported"] for r in scored if r.get("model_reported")}),
        "n_total": len(rows), "n_scored": n, "n_error": len(errs),
        "n_invalid_answer": len(invalid),
        # 分母口径必须显式写出：accuracy 的分母是 n_scored（= 拿到合法/可判定答案的题），
        # 与本地侧「无效输出判错并留在分母」不是同一个口径，并表时须对齐后再比。
        "accuracy_denominator": n,
        "denominator_note": "accuracy = k/n_scored；n_error（传输/协议失败）已移出分母，"
                            "n_invalid_answer（模型答了但不在选项集内）计错并留在分母",
        "success_rate": round(n / len(rows), 4) if rows else None,
        "accuracy": round(k / n, 4) if n else None,
        "accuracy_wilson95": wilson(k, n) if n else None,
        "by_source": by_src, "by_kind": by_kind,
        "error_tally": {e: sum(1 for r in errs if r.get("error") == e)
                        for e in sorted({r.get("error") for r in errs})},
        "failed_ids": [r["id"] for r in errs],
        "input_tokens": tok_in,
        # TypeSafe 直连响应不含 cost 字段（OpenRouter 才有）：缺失就报 null，不假装是 0；
        # 估算只覆盖已成功响应的输入 token，是**下界**（不含失败重试那次可能的计费）。
        "cost_api_reported_usd": round(cost_api, 6) if cost_api else None,
        "cost_est_usd_lower_bound": round(tok_in / 1e6 * INPUT_USD_PER_MTOK, 6),
        "confidence_mean_correct": round(sum(conf_ok) / len(conf_ok), 4) if conf_ok else None,
        "confidence_mean_wrong": round(sum(conf_bad) / len(conf_bad), 4) if conf_bad else None,
        "attempts_total": sum(r.get("attempts") or 0 for r in rows),
        "elapsed_sec": round(elapsed, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="只跑前 N 题（验证 schema 用）")
    ap.add_argument("--preflight", action="store_true", help="只验 key/额度，不发题")
    ap.add_argument("--native-kind", action="store_true",
                    help="kind=score/noul 的记录改用原生类型（探索性，非预注册协议）")
    ap.add_argument("--records", default=str(SEALED_EXAM))
    ap.add_argument("--allow-custom-records", action="store_true",
                    help="显式承认考卷不是封存 holdout——出境授权不覆盖此类题集，后果自负")
    ap.add_argument("--out", default=None, help="默认按模式选：全量 / smoke / native")
    args = ap.parse_args()

    key = load_key()
    if args.preflight:
        return preflight(key)

    default_out = {
        (False, False): "data/exp005/jev-holdout.jsonl",
        (True, False): "data/exp005/jev-holdout-smoke.jsonl",
        (False, True): "data/exp005/jev-native.jsonl",
        (True, True): "data/exp005/jev-native-smoke.jsonl",
    }[(bool(args.smoke), args.native_kind)]
    out_path = Path(args.out or (REPO / default_out))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rec_path = Path(args.records).resolve()
    records = [json.loads(l) for l in rec_path.read_text(encoding="utf-8").split("\n") if l.strip()]
    if rec_path == SEALED_EXAM.resolve():
        exam_sha = verify_exam(records, rec_path)
    else:
        assert args.allow_custom_records, (
            f"考卷 {rec_path} 不是封存 holdout。出境授权按封存考卷给出，"
            f"发送其它题集须显式加 --allow-custom-records 并自行确认授权。")
        exam_sha = hashlib.sha256(rec_path.read_bytes()).hexdigest()
        print(f"[warn] 非封存考卷 · sha256 {exam_sha[:12]}… · 出境授权不覆盖本清单")
    if args.native_kind:
        records = [r for r in records if r.get("kind") in NATIVE_KINDS]
    mode = "native-kind" if args.native_kind else "choice"
    # 续跑只跳过**已判定**的行：correct is None 的行是故障行，必须重投，
    # 否则一次网络中断会在结果里永久留洞、且被当作「已完成」。
    prior = resume(out_path, mode, exam_sha)

    done = {i for i, r in prior.items() if r.get("correct") is not None}
    todo = [r for r in records if r["id"] not in done]
    if args.smoke:
        todo = todo[:args.smoke]
    assert todo or done, "无可跑记录：records 为空或全部已完成"
    print(f"[start] 模式 {mode} · 考卷 {len(records)} · 已完成 {len(done)} · 本轮 {len(todo)} "
          f"· 模型 {MODEL} · 输出 {out_path}", flush=True)

    t0 = time.time()
    n_err = 0
    with open(out_path, "a", encoding="utf-8") as fh, \
         ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futs = {pool.submit(run_one, r, key, args.native_kind): r for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            res["exam_sha256"] = exam_sha  # 逐行可自证属于哪份考卷（续跑校验用）
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            fh.flush()
            prior[res["id"]] = res  # 追加写只为崩溃安全；权威副本是这份按 id 去重的
            if res["correct"] is None:
                n_err += 1
                print(f"[error] {res['id']} · {res['error']} · {res.get('error_detail','')[:120]}",
                      flush=True)
            if i % 100 == 0:
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] {el:.0f}s · {el/i:.2f}s/题 · 本轮错误 {n_err}", flush=True)

    # 收尾：按 id 去重后**无条件**重写整个文件。重投过的故障行只留最后一次结果，
    # 避免重复 id 混进分母把 n_total 撑大。
    # 曾经这里加了「行数不等才重写」的短路，但行数相等不等于内容一致——
    # 半行拼上一条新结果就可能凑出相同的行数。结果文件很小（约 700 KB），不值得为省一次写盘冒这个险。
    _rewrite(out_path, prior)
    print(f"[sync] 结果文件已按 id 同步：{len(prior)} 行（本轮新增 {len(todo)}）", flush=True)

    rows = list(prior.values())
    summary = summarize(rows, time.time() - t0)
    summary["mode"] = mode
    summary["out_file"] = out_path.name
    summary["exam_path"] = str(rec_path)
    summary["exam_sha256"] = exam_sha
    summary["records_n"] = len(records)  # 本模式**参与**的记录数（去重前；native 已按 kind 过滤）
    summary["records_note"] = ("records_n 为本模式参与记录数（native-kind 会先按 kind 过滤；"
                              "去重前），n_total 为去重后落盘行数。"
                              "两者不等说明参与记录里有重复 id（如 smoke 的多维采样）或历史重投行。")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    (out_path.parent / (out_path.stem + "-summary.json")).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if summary["n_error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
