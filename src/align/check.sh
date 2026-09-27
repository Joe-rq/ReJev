#!/usr/bin/env bash
# 负例检查（主题见同目录 negatives.md）
# 用法：check.sh <要检查的文件> [更多文件...]
# 读同目录 negatives.md：PATTERNS 段逐条查，命中即失败（退 1）；
# CHECKLIST 段打印成人工核对清单（脚本不判）。
# 检查器自己出错（坏正则、文件读不了）时退 2——出错不算通过。
#
# 它只查得了字面。语气、结构、逻辑查不了——
#    查不了的那部分，就是你每次仍要亲自看一眼的部分。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEG="$HERE/negatives.md"

if [ $# -lt 1 ]; then
  echo "用法：$(basename "$0") <要检查的文件> [更多文件...]" >&2
  exit 2
fi
if [ ! -f "$NEG" ]; then
  echo "找不到 $NEG" >&2
  exit 2
fi

# 目标必须是可读的普通文件——目录、不存在、读不了都算检查器错误，不算通过
for f in "$@"; do
  if [ ! -f "$f" ] || [ ! -r "$f" ]; then
    echo "检查不了：$f 不是可读的普通文件。" >&2
    exit 2
  fi
done

# 取围栏段之间的行；跳过空行与 # 开头的注释
fence() {
  awk -v tag="$1" '$0=="```"tag{f=1;next} /^```$/{f=0} f' "$NEG" \
    | grep -v '^[[:space:]]*$' | grep -v '^[[:space:]]*#'
}
PATTERNS="$(fence PATTERNS)"
CHECKS="$(fence CHECKLIST)"

if [ -z "$PATTERNS" ]; then
  echo "negatives.md 的 PATTERNS 段是空的——还没有护栏可查。" >&2
  echo "空的检查必然通过，那不叫通过，叫没查。" >&2
  exit 2
fi

fails=0
total=0
while IFS=$'\t' read -r pat note; do
  [ -z "${pat:-}" ] && continue
  total=$((total + 1))
  # 模式经变量传入 grep -e，不拼进命令行求值——防止模式被 shell 当命令展开。
  # 退出码：0=命中（护栏被触发）；1=未命中；其余=检查器出错，立即失败。
  hits="$(grep -nEH -e "$pat" -- "$@")"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    fails=$((fails + 1))
    printf '✗ %s　%s\n' "$pat" "${note:-}"
    printf '%s\n' "$hits" | sed 's/^/    /' | head -5
  elif [ "$rc" -ne 1 ]; then
    echo "第 $total 条模式无法执行（多半是正则写错了）：$pat" >&2
    exit 2
  fi
done <<< "$PATTERNS"

echo
if [ -n "$CHECKS" ]; then
  echo "人工核对清单（脚本不判，逐条自己看）："
  printf '%s\n' "$CHECKS" | sed 's/^/  □ /'
  echo
fi

if [ "$fails" -gt 0 ]; then
  echo "不通过：$total 条护栏里 $fails 条被触发。"
  exit 1
fi
echo "通过：$total 条护栏全部未触发。"
echo "它只查了字面。语气、结构、逻辑仍要你自己看一眼。"
exit 0
