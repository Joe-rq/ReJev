"""秒级触发脚本：spawn OOD 客卷多臂评测后**立刻返回**（deploy + spawn 解耦模式）。

机制说明见同目录 `trigger_clean.py`：不要用 `modal run <评测脚本>` 直接跑，
本地终止会取消远程任务。

**客卷不随本仓发布**（第三方随其评测代码公开，见 `REPRODUCING.md` §6.1）——先自备
一份与上游逐字节相同的副本，本地转换后上传：

  1. `uv run python src/train/prep_exam.py --exam-dir <你的客卷目录>`
  2. `modal volume put rejev <转出的 exam.jsonl> /vol/data/exam.jsonl`
  3. `modal deploy src/train/eval_cross.py`（得到 `rejev-eval-cross-v2` app）

五臂（`docs/plan/005_ood-exam-eval.md` 所有者选定）：base · adapter · clean_r16 ·
adapter_r64 · tev1。**每臂产物自带模型名**（`cross-{model}-exam.jsonl`），互不覆盖；
续跑有 `.meta.json` 身份闸门（考卷 sha256 / 模型 revision / adapter sha 不符即拒绝续写）。

用法：modal run tools/trigger_exam.py                        # 全部臂
      modal run tools/trigger_exam.py --models base,adapter  # 指定臂
"""

import hashlib
from pathlib import Path

import modal

app = modal.App("rejev-trigger-exam")

REPO = Path(__file__).resolve().parents[1]
EVAL_SRC = REPO / "src/train/eval_cross.py"
# 把评测脚本自身的 sha256 随调用送进远程产物的 meta——事后可核对「跑的是哪一版代码」。
# commit 由 deploy 侧记录（触发端不假设 git 可用）。
CODE_SHA = hashlib.sha256(EVAL_SRC.read_bytes()).hexdigest()

APP_NAME = "rejev-eval-cross-v2"
ARMS = ["base", "adapter", "clean_r16", "adapter_r64", "tev1"]


@app.local_entrypoint()
def main(models: str = ",".join(ARMS), papers: str = "exam") -> None:
    fn = modal.Function.from_name(APP_NAME, "eval_run")
    picked = [m for m in models.split(",") if m]
    for m in picked:
        call = fn.spawn(model=m, papers=papers, n=0,
                        code_commit="spawn-detached", code_sha256=CODE_SHA)
        print(f"已提交 {m:<12} papers={papers:<6} call_id = {call.object_id}")
    print(f"\ncode_sha256 = {CODE_SHA[:16]}…（评测脚本本身）")
    print("本地即将退出；远程各臂独立执行（各自独立容器）。")
