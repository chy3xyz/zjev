# ZJEV

Typed Probabilistic Decision Runtime —— 把非结构化状态转换为经过概率校准的结构化决策（Zig 0.17 实现）。

协议与理论见 `docs/rfc-0001-zjev-decision-runtime.md`；实现任务拆解见 `docs/plans/2026-09-24-zjev-v0.1-implementation.md`。

## 构建与测试

```bash
zig build                          # 构建 zjev-serve / zjev-fit / zjev-bench / zjev-conformance
zig build test                     # 单元测试
zig build test-conformance         # 协议 conformance fixtures
```

## 运行

```bash
zig-out/bin/zjev-serve --port 9377                 # 直连引擎
zig-out/bin/zjev-serve --port 9377 --scheduler --cache   # Queue 批处理 + 单飞缓存
curl -s -X POST localhost:9377/v1/decide -d @examples/request.json
curl -s -X POST localhost:9377/v1/decide/batch -d @examples/batch_request.json
```

CLI 选项：`--bind` `--port` `--mock-mode uniform|peaked|sequence` `--profiles-dir <dir>` `--scheduler` `--cache`。

## 工具

```bash
zig-out/bin/zjev-fit --dataset datasets/calibration_sample.jsonl   # 拟合 temperature，写 model/calibration/*.json
zig-out/bin/zjev-bench --dataset <jsonl> [--profiles-dir model/calibration]  # 输出 accuracy/brier/ece/mce 指标 JSON
```

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
src/runtime   engine / scheduler（std.Io.Queue 批处理）/ cache
src/api       std.json 解析 + 手写序列化 + std.http.Server(std.Io.Threaded)
tools/        conformance runner / fit / bench / onnx_api 生成器
test/conformance/*.json
```

V0.2+：Decision Graph 执行器、trajectory 校准、WebSocket、WebGPU/Metal 后端（见 RFC §7/§12）。
