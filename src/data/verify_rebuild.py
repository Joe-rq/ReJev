"""只读验证历史 records 重建副本；仅输出聚合诊断，不输出样本。

默认锚点来自随仓发布的 docs/reproducibility/history-hashes.json。
文件级字节哈希、行数、分源/分组计数及切分不相交同时成立才通过。
不校验模型成绩或跨 tokenizer 渲染文本，不修改数据或封存状态。
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

from render import statehash

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "docs/reproducibility/history-hashes.json"
FILES = (
    "upstream/train.jsonl", "upstream/dev.jsonl",
    "rejev/rejev-train.jsonl", "rejev/rejev-holdout.jsonl", "rejev/rejev-dev.jsonl",
    "rejev2/rejev2-train.jsonl", "rejev2/rejev2-holdout.jsonl",
)
MAX_BYTES = 256 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024
MAX_ROWS = 50_000
MAX_MANIFEST_BYTES = 64 * 1024
SHA = re.compile(r"[0-9a-f]{64}")


class Invalid(ValueError):
    """可以公开打印的聚合错误。"""


def require(condition, message):
    if not condition:
        raise Invalid(message)


def validate_manifest(m):
    """字段封闭：候选公开清单不能混入题干、逐题 ID 或其它字段。"""
    require(isinstance(m, dict) and set(m) == {
        "schema_version", "upstream_commit", "split_parameters", "files"}, "清单顶层字段异常")
    require(type(m["schema_version"]) is int and m["schema_version"] == 1, "清单版本异常")
    require(isinstance(m["upstream_commit"], str) and
            re.fullmatch(r"[0-9a-f]{40}", m["upstream_commit"]), "上游 commit 异常")
    require(m["split_parameters"] == {
        "holdout1": {"seed": 20260924, "group_fraction": 0.05},
        "holdout2": {"seed": 20260925, "group_fraction": 0.05}}, "历史切分参数异常")
    require(isinstance(m["files"], dict) and set(m["files"]) == set(FILES), "清单文件集合异常")
    for name, e in m["files"].items():
        fields = {"sha256", "rows", "groups", "sources"}
        if name == "rejev/rejev-holdout.jsonl":
            fields.add("statehashes_sha256")
        require(isinstance(e, dict) and set(e) == fields, f"{name}: 清单字段异常")
        for key in ("sha256", "statehashes_sha256"):
            if key in e:
                require(isinstance(e[key], str) and SHA.fullmatch(e[key]), f"{name}: 哈希格式异常")
        for key in ("rows", "groups"):
            require(type(e[key]) is int and 0 < e[key] <= MAX_ROWS, f"{name}: 计数异常")
        require(e["groups"] <= e["rows"], f"{name}: 分组计数异常")
        require(isinstance(e["sources"], dict) and e["sources"] and all(
            isinstance(s, str) and type(n) is int and 0 < n <= MAX_ROWS
            for s, n in e["sources"].items()), f"{name}: 分源计数异常")
        require(sum(e["sources"].values()) == e["rows"], f"{name}: 分源总数异常")
    return m


def load_manifest(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "清单有重复 JSON 键")
            result[key] = value
        return result

    with path.open("rb") as fh:
        raw = fh.read(MAX_MANIFEST_BYTES + 1)
    require(len(raw) <= MAX_MANIFEST_BYTES, "清单过大")
    return json.loads(raw, object_pairs_hook=unique_object)


def read_records(path, name, expected):
    require(path.is_file() and not path.is_symlink(), f"{name}: 缺普通文件或为软链")
    require(path.stat().st_size <= MAX_BYTES, f"{name}: 文件过大")
    digest = hashlib.sha256()
    ids, groups, states = set(), set(), set()
    sources = Counter()
    n = 0
    total_bytes = 0
    with path.open("rb") as fh:
        while raw := fh.readline(MAX_LINE_BYTES + 1):
            require(len(raw) <= MAX_LINE_BYTES, f"{name}: 行过大")
            total_bytes += len(raw)
            require(total_bytes <= MAX_BYTES, f"{name}: 实际读取过大")
            digest.update(raw)
            n += 1
            require(n <= MAX_ROWS, f"{name}: 行数过多")
            try:
                r = json.loads(raw)
            except (ValueError, UnicodeError, RecursionError):
                raise Invalid(f"{name}: 第 {n} 行 JSON 无效") from None
            require(isinstance(r, dict) and all(k in r for k in ("id", "source", "group_id", "state")),
                    f"{name}: 第 {n} 行结构无效")
            require(all(isinstance(r[k], str) and r[k] for k in ("id", "source", "group_id")) and
                    isinstance(r["state"], (str, dict)), f"{name}: 第 {n} 行字段类型无效")
            require(r["id"] not in ids, f"{name}: 存在重复 ID")
            ids.add(r["id"])
            groups.add((r["source"], r["group_id"]))
            sources[r["source"]] += 1
            states.add(statehash(r["state"]))
    require(digest.hexdigest() == expected["sha256"], f"{name}: SHA256 不匹配")
    require(n == expected["rows"], f"{name}: 行数不匹配")
    require(len(groups) == expected["groups"], f"{name}: 分组计数不匹配")
    require(dict(sources) == expected["sources"], f"{name}: 分源计数不匹配")
    if "statehashes_sha256" in expected:
        h = hashlib.sha256("\n".join(sorted(states)).encode()).hexdigest()
        require(h == expected["statehashes_sha256"], f"{name}: state 集摘要不匹配")
    return {"rows": n, "ids": ids, "groups": groups, "states": states}


def check_partition(parent, children, name):
    require(sum(c["rows"] for c in children) == parent["rows"], f"{name}: 行数不守恒")
    for key in ("ids", "groups", "states"):
        union = set()
        for child in children:
            require(not union & child[key], f"{name}: {key} 跨侧重叠")
            union.update(child[key])
        require(union == parent[key], f"{name}: {key} 集合不守恒")


def verify(manifest, roots):
    validate_manifest(manifest)
    summaries = {}
    for name in FILES:
        folder, filename = name.split("/")
        base = roots[folder].resolve()
        path = base / filename
        summaries[name] = read_records(path, name, manifest["files"][name])
    check_partition(summaries["upstream/train.jsonl"], [summaries["rejev/rejev-train.jsonl"],
                    summaries["rejev/rejev-holdout.jsonl"]], "第一次切分")
    check_partition(summaries["rejev/rejev-train.jsonl"], [summaries["rejev2/rejev2-train.jsonl"],
                    summaries["rejev2/rejev2-holdout.jsonl"]], "第二次切分")
    require(summaries["upstream/dev.jsonl"] == summaries["rejev/rejev-dev.jsonl"], "dev 保留不一致")
    require(not summaries["upstream/train.jsonl"]["states"] & summaries["upstream/dev.jsonl"]["states"],
            "上游 train/dev state 跨侧重叠")
    return {name: manifest["files"][name]["rows"] for name in FILES}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream-records", required=True, type=Path, help="上游 new-v1/records 目录")
    ap.add_argument("--rejev-records", required=True, type=Path, help="第一次切分的 records 目录")
    ap.add_argument("--rejev2-records", required=True, type=Path, help="第二次切分的 records 目录")
    args = ap.parse_args()
    try:
        manifest = load_manifest(MANIFEST)
        result = verify(manifest, {"upstream": args.upstream_records,
                        "rejev": args.rejev_records, "rejev2": args.rejev2_records})
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        # 不把系统路径、JSON 原文或题面夹进异常日志。
        print(f"FAIL: {exc if isinstance(exc, Invalid) else '清单或输入读取失败'}", file=sys.stderr)
        return 1
    for name, rows in result.items():
        print(f"PASS {name}: {rows} records")
    print("REBUILD-CHECK: PASS（仅 records 与历史切分；非 GPU 成绩验证）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
