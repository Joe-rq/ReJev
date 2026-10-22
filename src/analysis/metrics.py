"""S1 分析口径（Issue #111）：校准指标、温度拟合、选择性预测、聚类 bootstrap。

纯 numpy 实现，不依赖 torch/modal；口径锁定于 `docs/plan/009-s0-manifest.json`：
- top-label ECE，15 等宽箱；
- Brier 为多分类求和式；
- NLL 为样本平均；
- 温度逐臂一个 T_a，calibration2 拟合，区间 [0.05, 20]；
- family-clustered bootstrap 10,000 次，seed 20261018，取 2.5/97.5 百分位。
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

import numpy as np

BOOT_SEED = 20261018
N_BOOT = 10_000
ECE_BINS = 15
T_RANGE = (0.05, 20.0)
T_GRID = 4001


def _record_logits(r: dict) -> np.ndarray:
    """取本题选项顺序的 logits；缺失时才退回 log(probs)（测试/旧记录）。"""
    labels = r["option_labels"]
    if "logits_by_label" in r:
        vals = [r["logits_by_label"][lab] for lab in labels]
    else:
        vals = [math.log(max(float(r["probs_by_label"][lab]), 1e-300)) for lab in labels]
    a = np.asarray(vals, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"non-finite logits in {r.get('id')}")
    return a


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits, axis=-1, keepdims=True)
    return logits - m - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))


def probs_from_logits(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    if T <= 0:
        raise ValueError("温度必须为正")
    return np.exp(_log_softmax(logits / T))


def nll_from_logits(logits: np.ndarray, gold_idx: int, T: float = 1.0) -> float:
    if T <= 0:
        raise ValueError("温度必须为正")
    return float(-_log_softmax(logits / T)[gold_idx])


def _probs(records: list[dict]) -> tuple[np.ndarray, list[list[str]], np.ndarray]:
    """返回 (probs[n, k] 对齐的 option_labels, gold_idx[n])；要求每题 label 顺序一致。"""
    labels = records[0]["option_labels"]
    for r in records:
        if r["option_labels"] != labels:
            raise ValueError("同一批记录的 option_labels 不一致；不能拼成矩阵")
    probs = np.array([[r["probs_by_label"][lab] for lab in labels] for r in records],
                     dtype=np.float64)
    gold = np.array([labels.index(r["gold_label"]) for r in records], dtype=np.int64)
    return probs, labels, gold


def nll(probs: np.ndarray, gold_idx: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(gold_idx)), gold_idx], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def brier(probs: np.ndarray, gold_idx: np.ndarray) -> float:
    y = np.zeros_like(probs)
    y[np.arange(len(gold_idx)), gold_idx] = 1.0
    return float(np.mean(np.sum((probs - y) ** 2, axis=1)))


def ece_top_label(probs: np.ndarray, gold_idx: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """top-label ECE：按 max prob 等宽分箱，|acc - mean conf| 加权。"""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == gold_idx).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # p=1.0 落最后一箱
    bins = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    total = len(conf)
    e = 0.0
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            continue
        e += (m.sum() / total) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def temperature_scale(probs: np.ndarray, T: float) -> np.ndarray:
    if T <= 0:
        raise ValueError("温度必须为正")
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / T
    logits -= logits.max(axis=1, keepdims=True)
    z = np.exp(logits)
    return z / z.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, gold_idx: np.ndarray,
                    t_range: tuple[float, float] = T_RANGE,
                    n_grid: int = T_GRID) -> dict:
    """在 log 空间网格上最小化 hard-label NLL；T=1 必在网格内；记录边界。"""
    grid = np.geomspace(t_range[0], t_range[1], n_grid)
    # 确保 1.0 在网格内
    if not np.any(np.isclose(grid, 1.0, rtol=0, atol=1e-12)):
        grid = np.sort(np.append(grid, 1.0))
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    gold_logp = logp[np.arange(len(gold_idx)), gold_idx]
    best_t, best_nll = None, math.inf
    for T in grid:
        scaled = logp / T
        m = scaled.max(axis=1, keepdims=True)
        lse = (m[:, 0] + np.log(np.exp(scaled - m).sum(axis=1)))
        val = float(-np.mean(gold_logp / T - lse))
        if val < best_nll:
            best_nll, best_t = val, float(T)
    at_boundary = bool(np.isclose(best_t, t_range[0]) or np.isclose(best_t, t_range[1]))
    return {"T": best_t, "nll": best_nll, "at_boundary": at_boundary,
            "range": list(t_range), "n_grid": len(grid)}


def selective_curve(probs: np.ndarray, gold_idx: np.ndarray,
                    coverages: Iterable[float] = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)
                    ) -> list[dict]:
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == gold_idx).astype(np.float64)
    return selective_curve_items(conf, correct, coverages)


def selective_curve_items(pmax: np.ndarray, correct: np.ndarray,
                          coverages: Iterable[float] = (0.05, 0.1, 0.2, 0.3,
                                                        0.5, 0.7, 0.9, 1.0)
                          ) -> list[dict]:
    """按 p_max 排序的覆盖率—错误率曲线（只报告，不调回退阈值）。"""
    pmax = np.asarray(pmax, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    order = np.argsort(-pmax, kind="stable")
    out = []
    n = len(pmax)
    for cov in coverages:
        k = max(1, int(math.ceil(cov * n)))
        out.append({"coverage": round(k / n, 4), "n": k,
                    "accuracy": round(float(correct[order[:k]].mean()), 6)})
    return out


def _record_groups(records: list[dict]):
    """按 option_labels 分组，返回 [(labels, logits[n,k], gold[n])]。"""
    groups: dict[tuple[str, ...], list[dict]] = {}
    for r in records:
        groups.setdefault(tuple(r["option_labels"]), []).append(r)
    out = []
    for labels, rs in groups.items():
        L = np.array([_record_logits(r) for r in rs], dtype=np.float64)
        gold = np.array([list(labels).index(r["gold_label"]) for r in rs], dtype=np.int64)
        out.append((list(labels), L, gold))
    return out


def fit_temperature_records(records: list[dict],
                            t_range: tuple[float, float] = T_RANGE,
                            n_grid: int = T_GRID) -> dict:
    """逐臂温度拟合；允许同一臂内不同题有不同选项数。"""
    groups = _record_groups(records)
    grid = np.geomspace(t_range[0], t_range[1], n_grid)
    if not np.any(np.isclose(grid, 1.0, rtol=0, atol=1e-12)):
        grid = np.sort(np.append(grid, 1.0))
    n_total = sum(len(gold) for _, _, gold in groups)
    best_t, best_nll = None, math.inf
    for T in grid:
        total = 0.0
        for _, L, gold in groups:
            logp = _log_softmax(L / T)
            total += float(-logp[np.arange(len(gold)), gold].sum())
        val = total / n_total
        if val < best_nll:
            best_nll, best_t = val, float(T)
    at_boundary = bool(np.isclose(best_t, t_range[0]) or np.isclose(best_t, t_range[1]))
    return {"T": best_t, "nll": best_nll, "at_boundary": at_boundary,
            "range": list(t_range), "n_grid": len(grid), "n_records": n_total}


def apply_temperature(record_or_probs: dict, T: float) -> dict[str, float]:
    """对单题应用温度。传 record（含 option_labels/logits）时用 logits/T；
    传旧的 probs dict 时退回 log(probs)/T（仅兼容测试）。"""
    if "option_labels" in record_or_probs:
        labels = record_or_probs["option_labels"]
        logits = _record_logits(record_or_probs)
        q = probs_from_logits(logits, T)
        return {k: float(v) for k, v in zip(labels, q)}
    labels = list(record_or_probs)
    p = np.array([record_or_probs[k] for k in labels], dtype=np.float64)
    q = temperature_scale(p.reshape(1, -1), T)[0]
    return {k: float(v) for k, v in zip(labels, q)}


def record_items(records: list[dict], T: float = 1.0) -> dict:
    """逐题 NLL/Brier/top-label 置信度/正确性；NLL 用 logits 稳定计算。"""
    nlls, briers, pmaxs, corrects = [], [], [], []
    for r in records:
        labels = r["option_labels"]
        gold = labels.index(r["gold_label"])
        logits = _record_logits(r)
        vec = probs_from_logits(logits, T)
        y = np.zeros_like(vec)
        y[gold] = 1.0
        nlls.append(nll_from_logits(logits, gold, T))
        briers.append(float(np.sum((vec - y) ** 2)))
        pred = int(vec.argmax())
        pmaxs.append(float(vec[pred]))
        corrects.append(int(pred == gold))
    return {"nll": np.array(nlls), "brier": np.array(briers),
            "pmax": np.array(pmaxs), "correct": np.array(corrects)}


def ece_from_items(pmax: np.ndarray, correct: np.ndarray, n_bins: int = ECE_BINS) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(pmax, edges[1:-1], right=False), 0, n_bins - 1)
    total = len(pmax)
    e = 0.0
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            continue
        e += (m.sum() / total) * abs(correct[m].mean() - pmax[m].mean())
    return float(e)


def summarize_records(records: list[dict], T: float = 1.0) -> dict:
    items = record_items(records, T)
    return {"n": len(records),
            "accuracy": float(items["correct"].mean()),
            "nll": float(items["nll"].mean()),
            "brier": float(items["brier"].mean()),
            "ece_top_label": ece_from_items(items["pmax"], items["correct"]),
            "mean_pmax": float(items["pmax"].mean())}


def confusion(records: list[dict], labels: list[str]) -> np.ndarray:
    idx = {lab: i for i, lab in enumerate(labels)}
    c = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for r in records:
        c[idx[r["gold_key"]], idx[r["pred_key"]]] += 1
    return c


def macro_f1(records: list[dict], labels: list[str]) -> float:
    """任务内固定类别集；缺失类 F1 记 0（plan §3.5.1-3）。"""
    c = confusion(records, labels)
    tp = np.diag(c).astype(np.float64)
    fp = c.sum(axis=0) - tp
    fn = c.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def per_class_recall(records: list[dict], labels: list[str]) -> dict:
    """真实类无样本记 NA、不计入平均（与 F1 的缺失记 0 不同）。"""
    c = confusion(records, labels)
    tp = np.diag(c).astype(np.float64)
    true_count = c.sum(axis=1).astype(np.float64)
    rec = {}
    vals = []
    for i, lab in enumerate(labels):
        if true_count[i] == 0:
            rec[lab] = None
        else:
            v = tp[i] / true_count[i]
            rec[lab] = float(v)
            vals.append(v)
    rec["macro_recall_defined"] = float(np.mean(vals)) if vals else None
    rec["n_defined"] = len(vals)
    return rec


def permutation_label(n: int, flips: int, ci_low: float) -> str:
    """按冻结阈值判读置换敏感性；用整数比较避免 0.2 浮点边界误差。"""
    if n <= 0:
        raise ValueError("n 必须为正")
    if 5 * flips >= n:          # 翻转率 >= 20%
        return "high_sensitivity"
    if 100 * (n - flips) >= 95 * n and ci_low >= 0.90:  # 一致率 >= 95% 且下界 >= 0.90
        return "low_sensitivity"
    return "uncertain"


def task_equal_weight(per_task: dict[str, float]) -> float | None:
    vals = [v for v in per_task.values() if v is not None]
    return float(np.mean(vals)) if vals else None


def sample_weighted(per_task: dict[str, tuple[float, int]]) -> float | None:
    """per_task: task -> (value, n)"""
    denom = sum(n for _, n in per_task.values())
    if denom == 0:
        return None
    return float(sum(v * n for v, n in per_task.values()) / denom)


def _group_index(groups: list[str]) -> tuple[list[str], list[np.ndarray]]:
    uniq = sorted(set(groups))
    idx = {g: i for i, g in enumerate(uniq)}
    buckets = [[] for _ in uniq]
    for i, g in enumerate(groups):
        buckets[idx[g]].append(i)
    return uniq, [np.asarray(b, dtype=np.int64) for b in buckets]


def cluster_bootstrap_mean(values: np.ndarray, groups: list[str],
                           n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """按 group 整组重抽的均值区间；values 为逐题值（0/1 或连续）。"""
    uniq, buckets = _group_index(groups)
    sums = np.array([values[b].sum() for b in buckets], dtype=np.float64)
    counts = np.array([len(b) for b in buckets], dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        out[i] = sums[pick].sum() / counts[pick].sum()
    return {"point": float(values.mean()),
            "ci95": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))],
            "n_boot": n_boot, "seed": seed, "n_groups": len(uniq)}


def cluster_bootstrap_paired_diff(arm_values: np.ndarray, base_values: np.ndarray,
                                  groups: list[str], n_boot: int = N_BOOT,
                                  seed: int = BOOT_SEED) -> dict:
    """同题配对差值 = arm - base；每次重采样内重算（plan interval_protocol）。"""
    diff = np.asarray(arm_values, dtype=np.float64) - np.asarray(base_values, dtype=np.float64)
    uniq, buckets = _group_index(groups)
    sums = np.array([diff[b].sum() for b in buckets], dtype=np.float64)
    counts = np.array([len(b) for b in buckets], dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        out[i] = sums[pick].sum() / counts[pick].sum()
    return {"point": float(diff.mean()),
            "ci95": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))],
            "n_boot": n_boot, "seed": seed, "n_groups": len(uniq)}


def cluster_bootstrap_stat(values: np.ndarray, groups: list[str],
                           stat_fn: Callable[[np.ndarray], float],
                           n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """通用按组重抽：stat_fn 收到被重采样拼出的逐题数组。用于 ECE/macro-F1。"""
    assert len(values) == len(groups), "values 与 groups 长度必须一致"
    uniq, buckets = _group_index(groups)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([buckets[j] for j in pick])
        out[i] = stat_fn(idx)
    return {"point": float(stat_fn(np.arange(len(values)))),
            "ci95": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))],
            "n_boot": n_boot, "seed": seed, "n_groups": len(uniq)}
