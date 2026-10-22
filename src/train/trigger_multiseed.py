"""S1 训练触发：`modal deploy` 后秒级 spawn，调用方退出不影响远程任务。

用法（仓库根）：
    modal deploy src/train/multiseed_train.py
    python src/train/trigger_multiseed.py 42
    # 可选第二参数为重训 tag；--dry-run 只打印实算 SHA，不 spawn。

铁律：不用 `modal run`（本地进程被终止会向远程发 cancellation）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

import modal

REPO = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    dry = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    seed = int(args[0]) if args else 42
    tag = args[1] if len(args) > 1 else ""
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO).decode().strip()
    if dirty:
        print("WARNING: 工作树不干净；deploy 的代码可能与 commit 不一致：")
        print(dirty)
        if not dry:
            raise SystemExit("工作树不干净，拒绝 spawn（先 commit 并干净树重新 deploy）")
    plan_sha = sha(REPO / "docs/plan/009_report-enhancement-experiments.md")
    s0_sha = sha(REPO / "docs/plan/009-s0-manifest.json")
    s1_sha = sha(REPO / "docs/plan/009-s1-manifest.json")
    src_m = sha(REPO / "src/train/multiseed_train.py")
    src_s = sha(REPO / "src/train/s1_spec.py")
    print(f"seed={seed} tag={tag!r} commit={commit}")
    print(f"plan_sha256={plan_sha}")
    print(f"s0_manifest_sha256={s0_sha}")
    print(f"s1_manifest_sha256={s1_sha}")
    print(f"source_multiseed_sha256={src_m}")
    print(f"source_s1_spec_sha256={src_s}")
    if dry:
        return 0
    fn = modal.Function.from_name("rejev-multiseed-train", "train")
    call = fn.spawn(seed=seed, tag=tag, code_commit=commit,
                    plan_sha256=plan_sha, s0_manifest_sha256=s0_sha,
                    s1_manifest_sha256=s1_sha,
                    source_multiseed_sha256=src_m,
                    source_s1_spec_sha256=src_s)
    print(f"spawned call={call.object_id}")
    # 本地运行台账（gitignored）：平台终止后用于核销 attempt/账单。
    ledger = REPO / "<run-ledger>.csv"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if not ledger.exists():
        ledger.write_text(
            "time,seed,tag,commit,call_id,plan_sha256,s0_manifest_sha256,"
            "s1_manifest_sha256,source_multiseed_sha256,source_s1_spec_sha256\n",
            encoding="utf-8")
    import time as _time
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(",".join([_time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                           str(seed), tag or "-", commit, call.object_id,
                           plan_sha, s0_sha, s1_sha, src_m, src_s]) + "\n")
    print(f"ledger={ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
