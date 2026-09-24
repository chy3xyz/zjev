# Laya 权重导出 + 本机 onnxruntime 设计（方案 A）

日期：2026-09-24
状态：已批准（v0 纯 plumbing，head 随机初始化）

## 背景与目标

ZJEV 运行时已完成 `--model`/`--sessions` 接线（CLI 可消费任意满足导出约定的 ONNX 图），
但真模型三输入全缺。本里程碑打通其中两个：

1. 本机安装 onnxruntime 动态库，`-Donnx=true` 链接通过
2. 将 Laya 英文 checkpoint（`convaiinnovations/laya`，ModernBERT-large 421M encoder）
   改造导出为符合 ZJEV 导出约定的 ONNX 图，端到端跑通 `/v1/execute`

**成功标准（v0 = plumbing 证据，非质量）**：
`zig build -Donnx=true` 链接成功；`zjev-serve --model laya.onnx --ort-extensions <lib>`
启动后 `/v1/execute` 带固定决策束返回 200 + trajectory + 概率；
决策束 Σ logitCount ≠ 8 时返回 4xx。

## 非目标

- 输出概率的质量（Laya base 零样本 near-chance，head 又是随机初始化；质量归 M2 自训模型）
- 搬运/逆向 Laya 预训练决策头（其 prompt 结构与 ZJEV 单前向契约不兼容）
- 决策束参数化（v0 固定 8-logit 示例束）
- 工具（fit/bench/traj）接 `--model`
- 远端推送

## 导出约定（现有，不变）

图内包含 tokenizer；输入 string tensor `text`（shape [1]），输出 float tensor `logits`；
输出 shape 必须**静态**（`onnx.zig:72-85` 用 `GetTensorShapeElementCount` 读取）。
session 单线程，并行由 `std.Io` 调度层负责。
图 = 固定决策束：请求 schema 的 Σ logitCount 必须等于图静态输出长度（`onnx.zig:166`）。

## 固定决策束（8 logits）

| # | id | type | 参数 | logitCount |
|---|----|------|------|-----------|
| 1 | escalate | noul | yes/no | 2 |
| 2 | topic | choice | options=[billing, bug, other] | 3 |
| 3 | urgency | score | scale=[lo, mid, hi] | 3 |

schema 主序 = 上表顺序（escalate, topic, urgency），Σ = 8。

## 架构

### 1. 本机环境

- `brew install onnxruntime`（1.30，`/opt/homebrew/lib/libonnxruntime.dylib`）
- `build.zig` 新增 build option `-Donnx-lib-dir <path>`（默认 `/opt/homebrew/lib`）；
  `-Donnx=true` 时 `lib_module.addLibraryPath` + `addRPathOnce`，保证链接与运行时解析
- libortextensions：从 GitHub `microsoft/onnxruntime-extensions` releases 下载
  `libortextensions-osx-arm64-*.dylib` 到 `export/laya/lib/`（不进 zig 链接，运行时 dlopen 式注册）

### 2. Zig 运行时扩展

- CLI 新增 `--ort-extensions <path>`（可选；仅 `--model` 模式生效）
- `factory.Config` 新增 `ort_extensions: ?[]const u8 = null`
- `onnx.zig`：`openOnnx(a, io, model_path, num_sessions, extensions_path: ?[]const u8)`；
  `createSession` 在每个 session 的 options 创建后、`CreateSession` 前调
  `ort.RegisterCustomOpsLibrary(opts, path, null)`（extern 已在 `onnx_api.zig:100`）
- 注册失败 → `error.OrtInitFailed` + `std.log.err` 打印 onnxruntime 错误信息，
  提示检查扩展库与 onnxruntime 版本匹配

### 3. 导出脚本 `export/laya/`（Python，不进 zig build）

```
export/laya/
├── requirements.txt      # torch / transformers / onnx / onnxscript / onnxruntime-extensions
├── export_laya.py        # 产出 out/laya.onnx
├── smoke_check.py        # 加载 laya.onnx 验证 shape/有限性/确定性
├── lib/                  # libortextensions dylib（gitignore）
└── out/                  # laya.onnx（gitignore）
```

`export_laya.py` 图结构（数据流）：

```
text: string[1]
  → BertTokenizer op (ai.onnx.contrib, vocab/配置取自 ModernBERT tokenizer)
  → input_ids, attention_mask: int64[1, seq]
  → ModernBERT encoder（权重 = Laya 英文 checkpoint 的 encoder 部分）
  → [CLS] hidden (1024)
  → Linear(1024→8)（随机初始化，固定 seed=42，仅 plumbing）
  → logits: float[1, 8]（静态 shape）
```

实现要点：

- 权重加载：优先 `pip install laya` 用其加载器；若 checkpoint 即标准 ModernBERT 结构，
  回退 `transformers.AutoModel.from_pretrained`。脚本第一步打印 state_dict 键名清单用于映射核对
- tokenizer 子图：onnxruntime-extensions python 从 HF tokenizer 生成；encoder 子图：
  torch.onnx 导出；二者 onnx.compose 拼接，中间 tensor 名对齐（ids/mask）
- 静态输出 shape [1,8] 是硬约束（`outputLogitCount` 要求）
- 图导出后立即用 `onnx.checker.check_model` + `smoke_check.py` 自检

### 4. 端到端验收

```bash
zig build -Donnx=true
./zig-out/bin/zjev-serve --model export/laya/out/laya.onnx \
    --ort-extensions export/laya/lib/libortextensions.dylib --port 9377
# /v1/execute 带 8-logit 决策束 → 200 + trajectory + 概率
# 带 Σ≠8 的决策束 → 4xx（BadModelIO 路径）
```

## 错误处理

| 场景 | 行为 |
|------|------|
| onnx=false 传 `--model` | 启动报错提示 `-Donnx=true`（已有） |
| `--ort-extensions` 路径错/版本不匹配 | `RegisterCustomOpsLibrary` 返回 status → `OrtInitFailed` + 错误日志 |
| 图缺扩展 op 直接 Run | `RunFailed` + onnxruntime 日志（提示缺 RegisterCustomOps） |
| 决策束 Σ≠8 | `BadModelIO` → 4xx（已有路径） |
| `--ort-extensions` 在 mock 模式 | 解析接受，忽略 |

## 测试

- Zig 单测：`parseCli --ort-extensions` 用例
- 链接测试：`zig build -Donnx=true` 本身 + 已有 `test "ort api base"`（onnx=true 时真实执行）
- 导出自检：`smoke_check.py`（输出 shape [8]、有限值、同输入两次前向一致）
- 端到端：`/v1/execute` 正确束 200 / 错误束 4xx

## 风险

1. **扩展库 ↔ brew ORT 1.30 版本耦合**：RegisterCustomOpsLibrary 是稳定 C API，大概率可用；
   失败时下载与 ORT 1.30 同期的 extensions release，或源码编译（记入 README）
2. **图拼接脆弱**：tokenizer 子图与 encoder 子图的 compose 最易碎，smoke_check 兜底
3. **概率无意义**：v0 head 随机初始化，验收只看链路与形状，汇报与文档明示
4. **checkpoint 结构未知**：Laya checkpoint 的 state_dict 键名需实地下载后核对；
   脚本着重打印与映射回退

## 开放事项

- libortextensions 具体 release 版本以下载时与 ORT 1.30 的匹配为准
- Laya checkpoint 是否含完整 ModernBERT 还是裁剪版 —— 实地下载后确认（风险 4）
