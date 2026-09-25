# M4 温度标定 设计

日期：2026-09-25
状态：已批准（auto 模式 inline 决策）
前置：`benchmarks/traj_laya_ft_2026-09-25.md`（M3 修复图读数）、
`docs/reports/2026-09-25-laya-head-training-report.md`

## 1. 问题

M3 修复图读数：escalate 节点 acc 0.785 但 **ece 0.586**——过拟合导致 softmax
饱和，置信度与对错严重脱节（selective_risk coverage 0.5 仍有 risk 0.12，
顶部置信含信息但整体温度过高）。不重训的改法就是温度标定：每决策拟合一个
softmax 温度 T，写入 profiles，运行时在 `finishOne` 应用。

## 2. 方案

工具链已存在（`tools/fit.zig` = zjev-fit、`calib/temperature.zig`、
`calib/profile.zig`、`model/calibration/` 样例），缺的是**对 ONNX bundle
模型的支持**：

1. **`tools/fit.zig` 扩展**：
   - `--model <path.onnx> [--sessions n] [--ort-extensions lib]`：经
     `zjev.factory.open` 加载（同 traj 的错误处理：Unsupported → 打印重建
     命令 exit 2）。
   - `--bundle '<json>'`：ONNX bundle 的完整 schema 集（engine.run 要求
     ΣlogitCount = 图宽，单决策调用必然 BadModelIO）。每条记录先按其
     decision id 在 bundle 中定位段下标，跑 `decideRaw(bundle_schemas)`
     一次，切出该段 logits 进拟合组。
   - **state 缓存**：按 state 文本缓存上次前向的 flat logits——标定集
     每记录展开 3 行（同 state 连续），无缓存要 3× 前向（约 5.4h），
     缓存后回到 5652 次（约 108 min）。
   - mock 默认路径不变（回归）。
2. **`export/laya/build_calib.py`**：`support_bundle_eval_flat.jsonl` →
   `datasets/support_bundle_calib.jsonl`（zjev-fit 格式：每行
   `{"state":{...},"decision":<RawDecision>,"label":<value>}`，每记录 3 行，
   label 类型按 decision 类型：noul bool / choice string / score 整数 bucket
   下标——对齐 `dataset.zig labelIndex`）。
3. **拟合**：`zjev-fit --dataset ... --model laya.onnx --bundle '<3 决策>'` →
   `model/calibration/laya_{noul,choice,score}_{2,3,3}_general.json`
   （temperature.fit 逐组最小化 NLL，输出带 ece/brier/selective_risk 指标）。
4. **评估**：`zjev-traj --profiles-dir model/calibration --model-name laya`
   复测，与 M3 无标定读数并排；关注 escalate ece 0.586 → ?、
   trajectory_accuracy 0.754 变化（noul 的 value 判定走温度后 softmax，
   边界样本可能翻转，通常微升）。

## 3. 边界与决策

- 标定集 = eval 集（训练未见过）。escalate=false 的记录在图上会被剪枝，
  但标定是**单跳**层面做的（每决策独立），与图激活无关——这正是
  "单跳校准"语义，与 traj 的轨迹级读数互补。
- 温度上下限：`temperature.fit` 内部有界（沿用，不额外约束）。
- 拟合分组键 = (task, num)：`noul/2`、`choice/3`、`score/3`，与
  `profile.lookup(model_name, schema, domain)` 的匹配键一致（沿用既有
  mock profile 的命名：`model/calibration/mock_noul_2_general.json` 格式）。

## 4. 改动面

- `tools/fit.zig`（--model/--sessions/--ort-extensions/--bundle + state 缓存）
- 新增 `export/laya/build_calib.py`
- 新增 `datasets/support_bundle_calib.jsonl`
- 新增 `model/calibration/laya_*.json`（3 个 profile）
- 新增 `benchmarks/temp_laya_2026-09-25.md`（前后对比）
- README 标定命令一段

## 5. 验收

1. `zig build test`（默认 + onnx）+ conformance 全绿。
2. mock 回归：`zjev-fit --dataset datasets/calibration_sample.jsonl
   --mock-mode sequence` 行为不变（产 mock profile 到临时目录比对既有样例）。
3. `--bundle` 宽度不匹配时该记录 skip 并计数（不静默吞错——打印
   `skipped: N` 汇总）。
4. 3 个 laya profile 产出，escalate 的 T 预期 >2（高温退烧），其 profile
   ece 显著低于 0.586。
5. traj --profiles 读数：escalate ece 显著下降；trajectory_accuracy 不降
   （±2pt 内）；并排写入 benchmark。
