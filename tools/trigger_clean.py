"""秒级触发脚本：spawn 训练任务后**立刻返回**（deploy + spawn 解耦模式）。

**为什么不能直接用 `modal run <训练脚本>`**：`modal run` 把本地进程绑在远程 app 上——
本地被终止（换对话、断网、Ctrl-C）时，Modal 会把 cancellation 信号送到远程，
**杀掉正在跑的训练**。加 `--detach` 也不行：实测本地终止仍会取消远程任务
（本项目 2026-09-25 用一次真实事故换来的教训：训练跑到 90 步被自己的后台清理操作取消）。

**正确做法**：① `modal deploy src/train/clean_train.py`，得到一个**持久 app**；
② 用本脚本 spawn —— 脚本自身秒级返回、本地进程正常退出，训练与调用方**完全解耦**。
此时 `modal app list` 里该 app 的 State 是 `deployed`（而非 `ephemeral`）。

前置（缺一不可）：
  1. `modal deploy src/train/clean_train.py`
  2. `modal volume put rejev data/clean/train-clean.jsonl /data/`   ← 训练集
  3. `modal volume put rejev data/clean/eval-set.jsonl /data/`      ← 评测集（评测阶段用）
  4. 复核账户 credit 余量（避免任务中途因额度耗尽失败）

用法：modal run tools/trigger_clean.py

监控：`modal app list`（State / Tasks）· `modal app logs <app-id>` ·
      `modal volume ls rejev /checkpoints-clean-r16`（**checkpoint 才是保险**）
"""

import os

import modal

app = modal.App("rejev-trigger-clean")


@app.local_entrypoint()
def main() -> None:
    fn = modal.Function.from_name("rejev-clean-train", "train")
    # r16/α32：exp002 配方。更高 rank（r64）在本项目实测为优化失败（见 exp004），
    # 故默认锁此配方；改 rank 前请先读 docs/experiments/exp004-capacity-probe/。
    # 要试其他配置时用环境变量覆盖，不必改源码：
    #     RANK=32 ALPHA=64 modal run tools/trigger_clean.py
    rank = int(os.environ.get("RANK", "16"))
    alpha = int(os.environ.get("ALPHA", "32"))
    call = fn.spawn(rank=rank, alpha=alpha, code_commit="spawn-detached")
    print(f"已提交训练 rank={rank} alpha={alpha}，call_id = {call.object_id}")
    print("本地即将退出；远程独立执行。")
