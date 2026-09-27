"""秒级触发脚本：spawn 同集多臂评测后**立刻返回**（deploy + spawn 解耦模式）。

机制说明见同目录 `trigger_clean.py`：不要用 `modal run <评测脚本>` 直接跑，
本地终止会取消远程任务。

**每臂必须给不同 `tag`**：逐题结果按 tag 落盘，同 tag 会互相覆盖、断点续跑会串味。
**必须核对封存集指纹**（下方 `EXPECT_SHA`，可用同名环境变量覆盖）：行数断言挡不住
「同样行数的另一份文件」——若各臂读到的不是同一份封存集，数字不可比且事后无从察觉。

前置：
  1. `modal deploy src/train/eval_holdout.py`（得到 `rejev-eval` app）
  2. `modal volume put rejev data/clean/eval-set.jsonl /data/`
  3. 确认封存集 SHA256 与下方 `EXPECT_SHA` 一致

用法：modal run tools/trigger_eval.py            # 全部臂
      modal run tools/trigger_eval.py --arms 006-clean-r16   # 指定臂
"""

import os

import modal

app = modal.App("rejev-trigger-eval")

EVAL_SET = "/vol/data/eval-set.jsonl"
EXPECT_N = 3694
# 封存集内容指纹。它是「各臂读的是同一份集」的唯一保证——行数挡不住「同样行数的
# 另一份文件」。换了数据就要同步更新；也可用环境变量临时覆盖，例如：
#     sha256sum data/clean/eval-set.jsonl
#     EXPECT_SHA=<你的指纹> modal run tools/trigger_eval.py
EXPECT_SHA = os.environ.get(
    "EXPECT_SHA",
    "8423a838db7a9217db8e6010e47a51d52b2dcfba7d439be743a503e1eb720f96",
)

COMMON = dict(
    split="holdout",
    data_path=EVAL_SET,
    expect_n=EXPECT_N,
    expect_sha=EXPECT_SHA,
    code_commit="spawn-detached",
)

# tag 带 006- 前缀：避免与更早轮次的同名 tag 在 Volume 上串味。
ARMS = [
    dict(model_kind="base", tag="006-base", art="/vol/artifacts/full-v1"),
    dict(model_kind="adapter", tag="006-exp002-r16", art="/vol/artifacts/full-v1"),
    dict(model_kind="adapter", tag="006-clean-r16", art="/vol/artifacts/clean-r16"),
]


@app.local_entrypoint()
def main(arms: str = "all") -> None:
    fn = modal.Function.from_name("rejev-eval", "eval_run")
    picked = ARMS if arms == "all" else [a for a in ARMS if a["tag"] in arms.split(",")]
    assert picked, f"没匹配到臂：{arms}"
    for a in picked:
        call = fn.spawn(**{**COMMON, **a})
        print(f"已提交 {a['tag']:<11} art={a['art']:<28} call_id = {call.object_id}")
    print("本地即将退出；远程独立执行。")
