# Selective Risk 指标补缺（RFC §6.1/§6.2/§11 收口）

- 日期：2026-09-24
- 状态：定稿（auto 模式决策）
- 前置：`docs/rfc-0001-zjev-decision-runtime.md` §6.1/§6.2/§11；基线 main @ 691aa1d（V0.2 合并后，109 单测 + 10 conformance 全绿）

## 1. 背景与缺口

RFC 四处要求 selective risk，实现为零：

1. §1.1："`src/calib`：……Brier / ECE / MCE / NLL / **selective risk** 全套指标"；
2. §6.1：`stats.zig` 职责含 "selective risk @ coverage"；
3. §6.2：profile metrics 示例含 `"selective_risk@0.9": 0.05`；
4. §11：口径 "selective risk 在 coverage ∈ {0.5,0.7,0.9,0.95} 各算一档"。

## 2. 语义定义（唯一定义，消除歧义）

输入：`conf: []const f32`（排序依据，逐条预测的置信度）、`ok: []const bool`（该条预测是否正确）、`coverages: []const f64`（档位，实现固定 {0.5, 0.7, 0.9, 0.95}）。

对每个 coverage c：

1. 排序：按 conf 降序；conf 相等时按原始下标升序（确定性，与样本顺序无关的是"分数"而非稳定性承诺）。
2. 保留数 `k = ceil(c · n)`（向上取整，保证至少覆盖比例 c；n=0 时整档跳过）。
3. `risk = 1 − mean(ok[0..k])`（保留子集的错误率）。
4. `threshold = conf_sorted[k−1]`（该档保留的最小置信度——消费方可直接用作 `Gate.threshold` 的调参参考）。

abstention 交互：selective risk 按 engine 上报的 confidence 原始值排序，不特判弃权样本（abstention accuracy 已是独立指标）。此口径写入文档，避免与"先剔除弃权再排序"的另一种合理实现产生静默分歧。

## 3. 交付

| 位置 | 变更 |
|---|---|
| `src/calib/stats.zig` | 追加 `pub const RiskPoint = struct { coverage: f64, keep: usize, n: usize, risk: f64, threshold: f32 }` 与 `pub fn selectiveRisk(a: alloc.Allocator, conf: []const f32, ok: []const bool, coverages: []const f64) error{OutOfMemory}![]RiskPoint` |
| `tools/bench.zig` | 每组输出追加 `"selective_risk":[{coverage,keep,risk,threshold}]×4` |
| `tools/traj.zig` | 轨迹级（按 path_prob 排序，即对低势轨迹弃权的口径）追加同名块；`by_node` 不动 |
| `tools/fit.zig` | profile metrics 增补 `selective_risk@0.5/0.7/0.9/0.95` 四个键（§6.2 示例的补全） |

`src/zjev.zig` 已导出 `stats`（V0.1），无需新增导出行；若新增 `risk` 别名导出则必须加 refAllDecls 覆盖——本设计不加别名，直接扩 stats.zig。

## 4. 测试

- `stats.zig` 单测：n=10 构造已知 conf/ok，手算验证 4 档的 keep/risk/threshold；并列 conf 的确定性（同分数不同顺序输入 → 同结果）；n=1、n<档位所需最小样本（如 n=2 跑 0.5/0.7 档）边界。
- bench：对 `datasets/calibration_sample.jsonl` 重跑，检查新字段出现且数值手算一致（抽一组）。
- traj：对 `datasets/traj_sample.jsonl` 重跑，轨迹级 4 档手算（5 条样本 k=3,4,5,5）。

## 5. 非目标

- 服务端 selective-prediction API（超出 RFC，YAGNI）。
- coverage 可调参数化（档位按 RFC 固定四档；函数已接受 coverages 参数，工具层写死常量）。
- risk-coverage 曲线输出（AURC 等）：待真模型数据有需要再加。

## 6. 出口标准

`zig build test` 全绿（净增 ≥4 例）；bench/traj 冒烟输出含新块且手算一致；fit 产出的 profile JSON 含四键；ReleaseSafe 编译通过。
