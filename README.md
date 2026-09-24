# ZJEV

Typed Probabilistic Decision Runtime —— 把非结构化状态转换为经过概率校准的结构化决策（Zig 0.17 实现）。

协议与理论见 `docs/rfc-0001-zjev-decision-runtime.md`；设计规格见 `docs/specs/2026-09-24-zjev-v0.2-decision-graph-design.md`；实现任务拆解见 `docs/plans/`。

## 构建与测试

```bash
zig build                          # 构建 zjev-serve / zjev-fit / zjev-bench / zjev-traj / zjev-conformance
zig build test                     # 单元测试
zig build test-conformance         # 协议 conformance fixtures
```

## 运行

```bash
zig-out/bin/zjev-serve --port 9377                 # 直连引擎
zig-out/bin/zjev-serve --port 9377 --scheduler --cache   # Queue 批处理 + 单飞缓存
curl -s -X POST localhost:9377/v1/decide -d @examples/request.json
curl -s -X POST localhost:9377/v1/decide/batch -d @examples/batch_request.json
curl -s -X POST localhost:9377/v1/execute -d @examples/execute_request.json   # V0.2 Decision Graph
```

CLI 选项：`--bind` `--port` `--mock-mode uniform|peaked|sequence` `--profiles-dir <dir>` `--scheduler` `--cache`。

## 工具

```bash
zig-out/bin/zjev-fit --dataset datasets/calibration_sample.jsonl   # 拟合 temperature，写 model/calibration/*.json（含 ece/brier/selective_risk@0.5/0.7/0.9/0.95）
zig-out/bin/zjev-bench --dataset <jsonl> [--profiles-dir model/calibration]  # 输出 accuracy/brier/ece/mce + selective_risk 四档
zig-out/bin/zjev-traj --dataset datasets/traj_sample.jsonl --mock-mode sequence  # 轨迹级校准报告（node vs trajectory 并排 + selective_risk）
```

selective risk 口径：按置信度降序（并列按下标）取前 ⌈coverage·n⌉ 条，risk = 保留子集错误率，threshold = 该档最小保留置信度（可直接作 `Gate.threshold` 调参参考）。

## Decision Graph（V0.2）

图执行端点 `POST /v1/execute`：请求体 = DecisionRequest + `graph`（nodes/edges/可选 gate），
条件表达式（如 `risk_level == high and risk.confidence > 0.5`）在请求解析期编译为 AST
并做静态类型检查，运行期沿 DAG frontier 批量执行（共享一次 encoder 前向），
响应为 `trajectory[] + skipped[] + path_prob`。节点可挂策略门 `Gate{threshold, action_above, action_below, action_abstain}`。

`zjev-traj` 度量 quest1.md §12 的命题（单跳 calibrated ⇏ 轨迹 calibrated）：
同一数据集并排输出 node-level 与 trajectory-level 的 accuracy/ECE/Brier。
详见 `docs/specs/2026-09-24-zjev-v0.2-decision-graph-design.md` §2.6。

## ONNX 后端（可选）

导出约定：图内包含 tokenizer；输入 string tensor `text`，输出 float tensor `logits`
（长度 = Σ logitCount(decisions)，schema 主序）。session intra-op 线程固定为 1，
由 `std.Io` 调度层并行。

```bash
zig build -Donnx=true     # 需要系统安装 onnxruntime 动态库
```

未安装时跳过；`-Donnx=true` 的链接错误是预期行为。

## 目录

```
src/core      State / DecisionSchema / DecisionResult / 校验
src/calib     softmax / 统计 / brier / ece / temperature / profile
src/model     vtable 接口 / mock / onnx（extern 绑定，vendor 头文件生成）
src/graph     图类型 / 条件 AST / gate / trajectory / frontier 执行器 / 轨迹报告（V0.2）
src/runtime   engine / scheduler（std.Io.Queue 批处理）/ cache
src/api       std.json 解析 + 手写序列化 + std.http.Server(std.Io.Threaded)
tools/        conformance runner / fit / bench / traj / onnx_api 生成器
test/conformance/*.json
datasets/     calibration_sample.jsonl / traj_sample.jsonl
```

V0.3+：概率分支遍历（若轨迹度量表明有必要）、WebSocket、WebGPU/Metal 后端（见 RFC §7/§12 与 spec §3 方案 C）。
