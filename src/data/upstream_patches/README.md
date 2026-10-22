# upstream_patches — 上游生成器 vendored patch

`0001-rejev-s0.patch` 应用于 `resources/tev1`（`togethercomputer/tev1`
`1dde7782382c9f49d627153759b8d1deab426ce0`，MIT）的**副本**；`resources/tev1`
本身保持只读，不修改。

## 改动

| 文件 | 改动 | 目的 |
|---|---|---|
| `build_dataset.py` | `SEED` 改为 `REJEV_SEED_V1` 环境变量 | 换 SEED 生成新实例 |
| `build_v2.py` | `SEED` 改为 `REJEV_SEED_V2`；`public_data` 的 `forbidden` 并入 `REJEV_FROZEN_STATEHASHES` 冻结清单 | 新实例；公开源屏障覆盖全部历史材料 |
| `build_v21.py` | `SEED` 改为 `REJEV_SEED_V21`；`research-v21` 的 `group_id` 加入 SEED | 新实例；family 标识与历史不相交 |
| `build_new_v1.py` | 行序 seed 改为 `REJEV_SEED_ORDER` | 可复现（本轮不执行 new-v1 阶段） |

环境变量缺省值均等于上游原常量，但**不设环境变量时并不完全等价于上游**，有两处例外：

1. `build_v2.public_data` 在缺 `REJEV_FROZEN_STATEHASHES` 时**明确报错**，而不是退回只挡 `data/v1` 的旧屏障（fail-closed）。
2. `build_v21` 的 `research-v21` group_id **无条件**加入 `SEED`：即使 `REJEV_SEED_V21` 未设，也会从
   `research-v21:{split}:{topic}:{i}` 变为 `research-v21:paper-feedback-v21-1:{split}:{topic}:{i}`。
   这是 family 隔离所需；若裸跑 patched 副本，v2.1 的 group_id 与官方不同，`build_new_v1.py` 的
   membership 键语义随之改变。用于复现上游原版时请用未打 patch 的 `resources/tev1`。

另：`build_dataset.py` 的默认 SEED 以字符串形式读取（`"20260920"`）；对 RNG 行为无影响
（`rng_for` 只做 `f"{SEED}:{value}"`），但若运行上游 `build_dataset.py`，其 manifest 的 `seed`
字段会写成字符串而非整数。

## 应用与核验

```bash
# 复制上游文件到仓外/临时目录（不要改 resources/tev1）
# 然后在副本根目录执行：
patch -p1 < src/data/upstream_patches/0001-rejev-s0.patch

# 指纹（写入 S0 manifest）
shasum -a 256 src/data/upstream_patches/0001-rejev-s0.patch
```

S0 交付版生成时，patch SHA256 为
`0b210096eac05774c7ae3d19be0b4c8d617df1f980ed9dd6339c8f6afda40d1e`。

端到端核验（2026-10-17；2026-10-18 补脚本留档）：用 `src/data/verify_patch_barrier.py`
跑 patched `build_v2.public_data`，生成 v2 公开切分（train/dev/calibration/test 共 9,000 行），
冻结清单命中 **0**，且缺 `REJEV_FROZEN_STATEHASHES` 时抛 `RuntimeError`（fail-closed）。
结果存 `docs/plan/009-patch-barrier.json`。
