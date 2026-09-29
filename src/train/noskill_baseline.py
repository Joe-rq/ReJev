"""No-skill baselines for a choice exam: what a model that learns nothing scores.

为什么要单独有这个脚本（2026-09-28 第三轮评审 P2）：技术报告 §2.1 引了四个无技能
基线（逐题随机 27.66% / 恒定最常见字母 27.91% / 恒定最常见语义 key 17.18% /
按选项数取该组最常见位置 29.86%），§2.4 又引了一个同形的 27.30%。**这五个数在全仓
只有报告本身出现过**——既没有计算代码，实验记录里也没有。读者无法复算、无法核对，
与报告 §7 自己写的「把每个数字与其来源放在一起」直接抵触。本脚本就是它们的来源。

口径（**逐条写死，改口径必须同时改报告**）：
  1. 逐题随机     mean(1 / n_options)：每题均匀随机猜一个位置的期望正确率
  2. 恒定字母     全卷 gold 里出现最多的**字母标签**，所有题都答它
  3. 恒定语义 key 全卷 gold 里出现最多的**语义 key**，所有题都答它
     （语义 key 的取值空间比字母大得多——24 个字母、上百个 key——故它通常更低）
  4. 按位置分组   对每个 n_options 分组，取该组内 gold 出现最多的**位置序号**，
                  该组所有题都答这个位置（比全局恒定字母更强：它允许「二选一卷偏 A、
                  四选一卷偏 C」这种按卷面结构变化的最优常数策略）

⚠️ **口径 2–4 是「事后描述上界」，不是无技能基线**：它们的取值（众数）是在**同一张
留出集**上拟合出来的，再拿同一批标签评分，故**系统性偏高**。它们回答的是「一个只会
输出常数的策略最好能到多少」，不能当作「随机猜的期望」。只有口径 1 不含拟合。
报告 §2.1 引用它们时必须带上这个限定（已加）。

用法：
    uv run python src/train/noskill_baseline.py                       # 默认两张封存留出集
    uv run python src/train/noskill_baseline.py <records.jsonl> ...
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT = [
    REPO / "data/rejev/records/rejev-holdout.jsonl",
    REPO / "data/rejev2/records/rejev2-holdout.jsonl",
]


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]


def key_of(r: dict) -> str | None:
    """从 `options` 反查 gold 的**语义 key**（`answer` 是 label，key 挂在选项上）。

    刻意**不读** record 级的 `answer_key` 字段（2026-09-28 第四轮评审 P1）：那个字段
    在本仓 `data/` 的产物里确实有，但**源码树里没有任何生产者**（全仓仅引用一处，
    就是本函数原来的写法），而文档化的 records 口径是 `state/question/options/answer`。
    第三方按文档重建数据时，读 `answer_key` 会 KeyError——一个跑不起来的「数字来源」
    比没有来源更糟。两者在本仓两张留出集上**逐条一致**（实测 1,892 ＋ 1,802 条零差异），
    故改用反查不改变任何数字。
    """
    return next((o.get("key") for o in r["options"] if o["label"] == r["answer"]), None)


def baselines(rows: list[dict]) -> dict:
    n = len(rows)
    letters = Counter(r["answer"] for r in rows)
    # ⚠️ 2026-09-29 第五轮评审 B 线 P3：`key_of` 在 gold 不在该题选项里时返回 `None`。
    # 此前 `None` 会作为**一个正常取值**参与众数——若这类题占多数，口径 3 会报出一个
    # 「以 None 为众数」的百分比，读起来像「最常见语义 key 的基线」，实际什么 key 都不是。
    # 这里显式过滤，并把未解析条数记进 detail 供上游核对；一条都解析不出时报「不可判读」
    # 而不是报一个假数字（分母仍取 n：答不出 key 的题在恒定 key 策略下就是答错）。
    keys = Counter(k for k in (key_of(r) for r in rows) if k is not None)
    # 分组「最常见位置」：把 gold 字母映射回它在该题选项表里的**位置序号**，
    # 再按 n_options 分组取组内众数。选项表的 label 是 A 起的连续字母。
    by_n: dict[int, list[int]] = {}
    for r in rows:
        labels = [o["label"] for o in r["options"]]
        pos = labels.index(r["answer"]) if r["answer"] in labels else None
        if pos is not None:
            by_n.setdefault(len(labels), []).append(pos)
    pick = {k: Counter(v).most_common(1)[0][0] for k, v in by_n.items()}

    def _hit(r: dict) -> bool:
        """该题在「按选项数取组内最常见位置」策略下是否命中。

        ⚠️ 2026-09-29：补 `pick.get(...)`。上面按 `n_options` 分组时**跳过**了 gold 不在
        选项里的题（`pos is None`），而这里原来对**全部**题取 `pick[n_options]`——只要数据
        里有一条 gold 落空的题，就是 `KeyError` 而不是一个可解释的结果。同一函数里
        「口径 3 过滤了 None」，口径 4 却会炸，属同源残留。
        """
        labels = [o["label"] for o in r["options"]]
        p = pick.get(len(labels))
        return p is not None and r["answer"] == labels[p]

    return {
        "n": n,
        "random_mean_1_over_n": round(sum(1 / len(r["options"]) for r in rows) / n, 6),
        "constant_most_frequent_letter": round(letters.most_common(1)[0][1] / n, 6),
        "constant_most_frequent_key": (round(keys.most_common(1)[0][1] / n, 6)
                                       if keys else None),
        "per_option_count_most_frequent_position": round(
            sum(1 for r in rows if _hit(r)) / n, 6),
        "detail": {"letter": letters.most_common(1)[0][0],
                   "key": keys.most_common(1)[0][0] if keys else None,
                   "key_unresolved": n - sum(keys.values()),
                   "n_options": len(by_n)},
    }


def main() -> int:
    paths = [Path(a) for a in sys.argv[1:]] or DEFAULT
    for p in paths:
        if not p.exists():
            # 与 prep_exam / prep_paper 同一处置：缺文件时给一句人话 + 该去哪儿造，
            # 而不是抛裸 FileNotFoundError。报告 §2.1 承诺「不带参数跑一次即可复现」，
            # 那条承诺的前提正是这两份文件在盘上（第五轮评审 B 线 P2）。
            print(f"✗ 找不到 {p}\n"
                  f"  本脚本默认读两张**封存留出集**，而本仓不发布数据。先按\n"
                  f"  REPRODUCING.md §2.4 造 data/rejev/（split_holdout.py）、\n"
                  f"  §2.5 造 data/rejev2/（split_holdout_v2.py）；\n"
                  f"  或直接把自定义 records 路径当参数传进来。",
                  file=sys.stderr)
            return 2
        b = baselines(load(p))
        d = b.pop("detail")
        print(f"\n{p.relative_to(REPO) if p.is_relative_to(REPO) else p}  n={b.pop('n')}"
              f"  （最常见字母 {d['letter']} / 最常见 key {d['key']} / {d['n_options']} 种选项数）")
        if d["key_unresolved"]:
            print(f"    ⚠️ 有 {d['key_unresolved']} 条的 gold 不在其选项里，"
                  f"其语义 key 无法解析——它们**未参与**众数统计，但在恒定 key 口径下计为答错")
        for k, v in b.items():
            if v is None:
                print(f"    {k:42s} 不可判读（没有任何一条能解析出语义 key）")
                continue
            print(f"    {k:42s} {v * 100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
