"""S1 六臂候选评分/前置验证触发（deploy + spawn）。

用法（仓库根）：
    modal deploy src/train/eval_candidates.py
    python src/train/trigger_eval.py base-c0 test2
    python src/train/trigger_eval.py hist test2
    python src/train/trigger_eval.py seed42 test2-perm
    python src/train/trigger_eval.py hist --precheck [n]

split ∈ dev2 / calibration2 / test2 / test2-perm；--precheck 只支持 base-c0/hist。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import modal

REPO = Path(__file__).resolve().parents[2]
SPLITS = {
    "dev2": ("dev2", "dev2.jsonl", "dev2"),
    "calibration2": ("calibration2", "calibration2.jsonl", "calibration2"),
    "test2": ("test2", "test2.jsonl", "test2"),
    "test2-perm": ("test2_permuted", "test2-permuted.jsonl", "test2-perm"),
}
SEED_ARMS = ("seed42", "seed17", "seed29", "seed41")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arm_identity(arm: str, s1: dict) -> tuple[str, str, str]:
    """返回 (art, adapter_sha, adapter_config_sha)；base 全空。"""
    ref = s1.get("reference_arms") or {}
    runs = {r["run_tag"]: r for r in s1.get("runs", [])}
    if arm == "base-c0":
        return "", "", ""
    if arm == "hist":
        h = ref.get("hist") or {}
        art = "/vol/artifacts/clean-r16"
        a, c = h.get("adapter_sha256", ""), h.get("adapter_config_sha256", "")
        assert a and c, "S1 manifest reference_arms 缺 hist adapter/config SHA"
        return art, a, c
    if arm in SEED_ARMS:
        assert arm in runs, f"S1 manifest runs 缺 {arm}"
        art = f"/vol/s1/artifacts/{arm}"
        a = runs[arm]["adapter_sha256"]
        c = runs[arm].get("adapter_config_sha256", "")
        assert a and c, f"S1 manifest runs[{arm}] 缺 adapter/config SHA"
        return art, a, c
    raise SystemExit(f"未知 arm：{arm}")


def main() -> int:
    dry = "--dry-run" in sys.argv
    precheck = "--precheck" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    assert args, "需要 arm（base-c0/hist/seed42/...）"
    arm = args[0]
    s1 = json.loads((REPO / "docs/plan/009-s1-manifest.json").read_text(encoding="utf-8"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).decode().strip()
    if dirty:
        print("WARNING: 工作树不干净：")
        print(dirty)
        if not dry:
            raise SystemExit("工作树不干净，拒绝 spawn（先 commit 并干净树重新 deploy）")
    src_eval = sha(REPO / "src/train/eval_candidates.py")
    src_math = sha(REPO / "src/train/candidate_math.py")
    src_spec = sha(REPO / "src/train/s1_spec.py")

    if precheck:
        assert arm in ("base-c0", "hist"), "precheck 只支持 base-c0/hist"
        n = int(args[1]) if len(args) > 1 else 200
        art, adapter, adapter_cfg = arm_identity(arm, s1)
        print(f"precheck arm={arm} art={art!r} n={n} commit={commit}")
        print(f"adapter={adapter or '(base)'} config={adapter_cfg or '(base)'}")
        print(f"source eval={src_eval} math={src_math} spec={src_spec}")
        if dry:
            return 0
        fn = modal.Function.from_name("rejev-eval-candidates", "precheck")
        call = fn.spawn(arm=arm, art=art, n=n, code_commit=commit,
                        expect_adapter_sha256=adapter,
                        expect_adapter_config_sha256=adapter_cfg,
                        source_eval_sha256=src_eval, source_math_sha256=src_math,
                        source_spec_sha256=src_spec)
        print(f"spawned precheck call={call.object_id}")
        return 0

    assert len(args) >= 2, "需要 arm 与 split"
    split_key = args[1]
    assert split_key in SPLITS, f"split 必须是 {sorted(SPLITS)}"
    raw_key, filename, split_label = SPLITS[split_key]
    files = s1["eval2_files"]
    art, adapter, adapter_cfg = arm_identity(arm, s1)
    tag = f"{arm}-{split_label}"
    data_path = f"/vol/data/eval2/records/{filename}"
    print(f"arm={arm} split={split_label} tag={tag} commit={commit}")
    print(f"data={data_path} sha={files[raw_key]['sha256']} rows={files[raw_key]['rows']}")
    print(f"art={art!r} adapter_sha256={adapter or '(base)'} "
          f"config={adapter_cfg or '(base)'}")
    print(f"source eval={src_eval} math={src_math} spec={src_spec}")
    if dry:
        return 0
    fn = modal.Function.from_name("rejev-eval-candidates", "score")
    call = fn.spawn(
        arm=arm, art=art, data_path=data_path, split=split_label,
        expect_n=files[raw_key]["rows"], expect_sha=files[raw_key]["sha256"],
        tag=tag, code_commit=commit, n=0,
        expect_adapter_sha256=adapter,
        expect_adapter_config_sha256=adapter_cfg,
        source_eval_sha256=src_eval, source_math_sha256=src_math,
        source_spec_sha256=src_spec)
    print(f"spawned score call={call.object_id}")
    ledger = REPO / "<run-ledger>.csv"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if not ledger.exists():
        ledger.write_text("time,arm,split,tag,commit,call_id,adapter_sha256,"
                          "adapter_config_sha256,source_eval,source_math,source_spec,"
                          "data_sha,rows\n", encoding="utf-8")
    import time as _time
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(",".join([_time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                           arm, split_label, tag, commit, call.object_id, adapter,
                           adapter_cfg, src_eval, src_math, src_spec,
                           files[raw_key]["sha256"], str(files[raw_key]["rows"])]) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
