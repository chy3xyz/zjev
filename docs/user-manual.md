# ZJEV 使用手册

Typed Probabilistic Decision Runtime —— 把非结构化状态（文本）转换为
**经过概率校准**的结构化决策。Zig 0.17 实现，可选 ONNX 真模型后端。

- 协议与理论：`docs/rfc-0001-zjev-decision-runtime.md`、`docs/prd.md`、`docs/quest1.md`
- 设计规格：`docs/specs/2026-09-24-zjev-v0.2-decision-graph-design.md`
- 训练报告：`docs/reports/2026-09-25-laya-head-training-report.md`
- 实测记录：`benchmarks/`

---

## 1. 安装与构建

### 环境要求

| 依赖 | 版本 | 说明 |
|---|---|---|
| Zig | 0.17.0-dev（经 zigup 安装） | 必须 |
| onnxruntime + onnxruntime-extensions 动态库 | 1.30 / 0.15.2 | 仅真模型后端需要，已随仓库提供于 `export/laya/lib/`（版本自洽的一对，勿混用） |
| Python venv + torch | 见 `export/laya/requirements.txt` | 仅训练/导出模型需要 |

### 构建

```bash
zig build                              # 默认构建：zjev-serve / zjev-fit / zjev-bench / zjev-traj / zjev-conformance
zig build test                         # 单元测试
zig build test-conformance             # 协议 conformance fixtures（应 10 pass 0 fail）
zig build -Donnx=true -Donnx_lib_dir=$PWD/export/laya/lib   # ONNX 后端构建
```

> **陷阱**：`zig build`（不带 onnx flag）会覆盖 onnx 构建出的二进制。
> 跑任何带 `--model` 的工具/serve 前，必须显式用 `-Donnx=true` 重建，
> 否则启动即报「--model requires an onnx build」并提示重建命令。

---

## 2. 五分钟上手（mock 模式，无需模型）

```bash
zig build
zig-out/bin/zjev-serve --port 9377
curl -s -X POST localhost:9377/v1/decide -d @examples/request.json
```

`examples/request.json`：

```json
{
  "state": { "text": "用户近 24h 链上交易 17 笔，余额 3.2 SOL，合约交互 8 次" },
  "decisions": [
    { "id": "risk_level", "type": "choice", "options": ["low", "medium", "high"], "abstain": true },
    { "id": "churn_7d", "type": "score", "scale": { "min": 1, "max": 5 } }
  ]
}
```

mock 概率分布可用 `--mock-mode uniform|peaked|sequence` 切换；
`--scheduler --cache` 开启 Queue 批处理 + 单飞缓存。

---

## 3. 核心概念

**State（状态）**：非结构化输入。`text` 字段喂给模型（图内 tokenizer）；
`id/data/embeddings/timestamp/source` 为元数据。目前真模型只用 `text`。

**Decision schema（决策模式）**：声明决策类型与合法输出空间，是请求、
图、模型束三方的契约。类型见 §5.1。

**Uncertainty（不确定性）**：每个决策结果携带：
- `confidence`：置信度。choice/score/rank 为预测类最大概率；**noul 为 P(yes)**
  （正类检测分——Gate 阈值语义依赖这一点）
- `entropy` / `variance`（score）：分布弥散度
- `abstention`：仅当 `"abstain": true` 时存在，模型给「拒答」列的概率

**Temperature / Profiles（温度标定）**：不重训修正过自信的 softmax。
serve 带 `--profiles-dir` 时，按 (model_name, decision task, num_options,
domain) 查 `model/calibration/*.json` 里的 temperature，对 logits 除 T
再 softmax。T>1 软化（治过自信），T<1 锐化。判定结果（argmax）不变。

**Decision Graph（V0.2）**：决策按 DAG 连接，边带 `when` 条件表达式；
条件不成立的支路整支跳过（响应 `skipped[]`）。节点可挂 Gate 策略门。
单请求内全图共享一次 encoder 前向（全量预取 + 激活波模拟）。

**Gate（策略门）**：

```
confidence >= threshold        → action_above
confidence <  threshold        → action_below
abstention >= threshold（若有）→ action_abstain（优先）
```

响应里 trajectory step 的 `action` 字段即门输出。阈值调参参考：
`zjev-fit`/`zjev-traj` 输出的 selective_risk 各档 threshold。

---

## 4. 快速路径选择

| 你想做什么 | 去哪 |
|---|---|
| 试试 API、看响应长什么样 | §2 mock 上手 |
| 接真模型做决策（英文工单场景） | §6 Laya 链路 + §5 API |
| 决策有依赖/要按条件分支、挂策略门 | §5.3 Decision Graph |
| 置信度太满/太虚，想修校准 | §7 温度标定 |
| 评估模型 + 标定效果 | §8 评估工具 |
| 接入自己的模型（非 Laya） | §5.4 模型契约 |
| 出错排查 | §9 FAQ |

---

## 5. HTTP API 参考

启动：`./zig-out/bin/zjev-serve --port 9377 [选项]`（选项表见 §8.1）。

### 5.1 决策类型速查

| type | value | 概率 | logitCount（模型束） | 说明 |
|---|---|---|---|---|
| `noul` | `true/false` | `probability` = P(yes) | 2（`abstain:true` 时 3） | 是/否。**注意默认 `abstain:true`** |
| `choice` | 选项字符串（或 `"__abstain__"`） | `probabilities{选项: p}` | options 数（+1 若 abstain） | 多选一 |
| `score` | 期望分 float | `probabilities{档: p}` | 档数（+1 若 abstain） | `scale.labels` 或 `scale.{min,max}` |
| `rank` | `[{id, score}]` 降序 | `probabilities{item: p}` | items 数 | 排序 |

### 5.2 `POST /v1/decide` —— 平面决策

一次前向算全部决策，无图无分支。请求 = `state` + `decisions[]`
（+ 可选 `domain`，默认 `"general"`；`policy` 字段暂不支持）。

响应：

```json
{
  "results": [
    {
      "id": "escalate", "type": "noul", "value": false,
      "probability": 0.027567,
      "uncertainty": { "confidence": 0.027567 },
      "latency_us": 12834059
    },
    {
      "id": "topic", "type": "choice", "value": "bug",
      "probabilities": { "billing": 0.0, "bug": 0.999914, "other": 0.000086 },
      "uncertainty": { "entropy": 0.000892, "confidence": 0.999914 },
      "latency_us": 12834059
    }
  ],
  "calibration": "matched"
}
```

- `calibration`：`"matched"` = profiles 命中并应用了温度；`"default"` = 无。
- 带 `"abstain": true` 的决策，`probabilities` 多出 `"__abstain__"` 键，
  `uncertainty` 多出 `abstention` 字段。

**`POST /v1/decide/batch`**：请求体 `{"requests": [<decide 请求>, ...]}`，
返回数组。配 `--scheduler` 走批处理队列。

### 5.3 `POST /v1/execute` —— Decision Graph

请求在 decide 基础上加 `graph`：

```json
{
  "state": { "text": "..." },
  "decisions": [
    {"id":"escalate","type":"noul","abstain":false},
    {"id":"topic","type":"choice","options":["billing","bug","other"]},
    {"id":"urgency","type":"score","scale":{"labels":["low","medium","high"]}}
  ],
  "graph": {
    "nodes": [
      {"id": "esc", "decision": "escalate"},
      {"id": "topic", "decision": "topic",
       "gate": {"threshold": 0.6, "action_above": "route_human",
                "action_below": "auto_reply", "action_abstain": "review"}},
      {"id": "urg", "decision": "urgency"}
    ],
    "edges": [
      {"from": "esc", "to": "topic", "when": "escalate == true"},
      {"from": "esc", "to": "urg", "when": "escalate == true"}
    ]
  }
}
```

- 节点 `decision` 必须能在 `decisions[]` 里按 id 找到；可选 `gate`。
- 边 `when` 为条件表达式，请求解析期编译成 AST 并做静态类型检查。
  语法：`ident == value`、`and/or/not`、比较 `> >= < <= == !=`、
  字段引用 `decision_id.confidence` / `.value` / `.abstention`
  （如 `risk.confidence > 0.5 and risk_level == high`）。
- 执行：一次性前向算全图节点，再沿 frontier 模拟激活波；
  条件为 false 的下游节点不执行，记入 `skipped[]`。

响应：

```json
{
  "trajectory": [
    {"node_id": "esc", "decision_id": "escalate", "result": { ... }, "action": "route_human"},
    {"node_id": "topic", "decision_id": "topic", "result": { ... }}
  ],
  "skipped": ["urg"],
  "path_prob": 0.6123,
  "calibration": "matched"
}
```

`path_prob` = 实际走过路径上各决策条件概率的连乘（轨迹级置信度）。

### 5.4 错误码

| HTTP | code | 含义 |
|---|---|---|
| 400 | `invalid_request` / `InvalidJson` / `MissingField` | 请求体非法/缺字段 |
| 400 | `BadModelIO` | ΣlogitCount 与图宽不一致（最常见：noul 没写 `"abstain": false`，默认 true 多 1 列） |
| 400 | `invalid_graph` / `InvalidGraph` | 图结构/条件表达式/gate 阈值非法 |
| 400 | `unsupported` | 传了 `policy` 等未支持字段 |
| 503 | `overloaded` | 队列满 |
| 500 | `internal` | 内部错误 |

错误体统一：`{"error":{"code":"...","message":"..."}}`。

---

## 6. ONNX 后端与模型契约

### 6.1 模型契约（自己接入模型时必读）

- 图内包含 tokenizer：输入 string tensor `text`，输出 float tensor `logits`。
- **输出 shape 必须静态**；`logits` 宽度 = 固定决策束的 Σ logitCount。
- 请求 `decisions[]` 的 Σ logitCount 必须与图一致，否则 400 BadModelIO。
  logitCount：noul=2（abstain:true→3），choice=|options|（+1），
  score=档数（+1），rank=|items|。
- logits 按决策在束中的声明顺序分段。
- 图含 ai.onnx.contrib 自定义 op（HfJsonTokenizer）时必须传
  `--ort-extensions` 指向 libortextensions 动态库。

### 6.2 动态库配对

`export/laya/lib/` 是版本自洽的一对（onnxruntime 1.30 取自 pip 轮，
libortextensions 0.15.2 取自 NuGet），**无 python 依赖**。
**brew 的 onnxruntime 1.30 在 `RegisterCustomOpsLibrary` 路径上会段错误，
勿混用。**

```bash
zig build -Donnx=true -Donnx_lib_dir=$PWD/export/laya/lib
./zig-out/bin/zjev-serve --model <m.onnx> --ort-extensions export/laya/lib/libortextensions.dylib
```

`--model` 的文件名（去扩展名）即 `model_name`，profiles 按它匹配。

---

## 7. Laya 决策束全链路（仓库内置真模型）

模型：ModernBERT-large（HuggingFace `convaiinnovations/laya`）+ 自训
Linear(1024,8) 决策头，束 = escalate-noul(无 abstain) / topic-choice3 /
urgency-score3（ΣlogitCount=8）。英文场景。

### 7.1 起服务

```bash
zig build -Donnx=true -Donnx_lib_dir=$PWD/export/laya/lib
./zig-out/bin/zjev-serve --model export/laya/out/laya.onnx \
    --ort-extensions export/laya/lib/libortextensions.dylib \
    --profiles-dir model/calibration --port 9377
```

注意：请求束必须正好 8 logits——noul 必须显式 `"abstain": false`。

### 7.2 从头训练（可选，读数见训练报告）

```bash
export/laya/.venv/bin/pip install -r export/laya/requirements.txt
export/laya/.venv/bin/python export/laya/build_dataset.py   # HF 工单 → 8-logit 束（train/eval）
export/laya/.venv/bin/python export/laya/train_head.py      # 冻结 encoder 训 head（MPS，~小时级）
export/laya/.venv/bin/python export/laya/export_laya.py --head export/laya/out/head.pt
export/laya/.venv/bin/python export/laya/smoke_check.py
```

### 7.3 当前读数（5652 条 eval，修复图，带 M4 温度标定）

| 指标 | 无标定 | 标定后 |
|---|---|---|
| trajectory_accuracy | 0.754 | 0.754（不变） |
| traj_ece | 0.200 | **0.106** |
| escalate acc / ece | 0.785 / 0.586 | 0.785 / 0.393 |
| topic acc / ece | 0.888 / 0.093 | 0.888 / 0.045 |

口径注意：traj 的 noul 节点 ece 用 P(yes)（Gate 语义），与 fit 的
max-prob ece 不同口径；urgency 在升级路径上金标恒 high，其 ece 是
退化读数。详见 `benchmarks/temp_laya_2026-09-25.md`。

### 7.4 效果演示

```bash
# 实例 A（带 profiles）:8790，实例 B（不带）:8791，然后：
python3 export/laya/demo_profiles.py
# 同一批工单两边对照，直接看置信度从 0.000x/0.999x 饱和区拉回工作区
```

---

## 8. CLI 工具参考

### 8.1 `zjev-serve`

| 选项 | 默认 | 说明 |
|---|---|---|
| `--bind` / `--port` | 127.0.0.1 / 9377 | 监听地址 |
| `--mock-mode` | peaked | uniform / peaked / sequence |
| `--profiles-dir` | 关 | 温度标定目录 |
| `--scheduler` / `--cache` | 关 | Queue 批处理 / 单飞缓存 |
| `--model` | 无 | ONNX 模型路径（需 onnx 构建） |
| `--sessions` | 0=默认 | onnxruntime 会话数（intra-op 线程固定 1，靠调度层并行） |
| `--ort-extensions` | 无 | 图含自定义 op 时必传 |

### 8.2 `zjev-fit` —— 温度拟合

```bash
# mock 教学：
zig-out/bin/zjev-fit --dataset datasets/calibration_sample.jsonl [--out model/calibration]
# 真模型束（按 (task,num_options) 分组拟合 T，min NLL）：
zig-out/bin/zjev-fit --dataset <jsonl> --model m.onnx \
    [--sessions n] [--ort-extensions lib] --bundle '<完整 schema JSON>' \
    [--model-name n] [--domain d] [--out dir]
```

- `--bundle`：束 schema JSON（与 serve 请求同构），给出段表；ΣlogitCount
  必须等于图宽，单决策跑会 BadModelIO。
- 数据集每行：`{"state": {...}, "decision": {...}, "label": <bool|string>}`。
  score 的 label 用字符串档名（`"low"/"medium"/"high"`），整数桶下标会被拒。
- 同 state 连续记录复用一次前向（state 缓存）；结束打印
  `fitted: N skipped: M`。
- 产出 `model/calibration/<model>_<task>_<num>_<domain>.json`：
  temperature + ece/brier/nll + selective_risk@0.5/0.7/0.9/0.95。

### 8.3 `zjev-traj` —— 轨迹级评估（quest1 §12 实验台）

```bash
zig-out/bin/zjev-traj --dataset <jsonl> [--mock-mode m] [--profiles-dir d] \
    [--model m.onnx [--sessions n] [--ort-extensions lib]]
```

数据集每行：`{"state":..., "decisions":[...], "graph":{...}, "expected":{decision_id: gold}}`。
读数走 **stderr**（`2>readings.json`）。输出：trajectory_accuracy /
traj_brier / traj_ece / traj_mce / selective_risk 四档 / by_node
（每节点 acc + ece）。CPU + 大模型约 2h/5652 条。

### 8.4 `zjev-bench`

```bash
zig-out/bin/zjev-bench --dataset <jsonl> [--profiles-dir dir]
```

平面校准报告：accuracy / brier / ece / mce + selective_risk 四档。

### 8.5 `zjev-conformance`

协议 fixtures 校验，`zig build test-conformance` 即跑（应 10 pass 0 fail）。

---

## 9. 温度标定工作流（M4）

**什么时候需要**：单跳 ece 高（置信度挤在 0/1）、selective_risk 阈值
落在≈1.0 饱和区、Gate 阈值没法调。

**步骤**：

1. 造拟合集：覆盖各决策、带金标 label（同分布、与评估集互斥最佳——
   本仓库 M4 用 eval split 展开，见 `export/laya/build_calib.py`）。
2. 拟合：`zjev-fit --model ... --bundle ...`（真模型）。
3. 挂 profiles 起 serve / 复测 traj。
4. 对比读数：acc 应不变（T 保序），ece/brier 应降，selective_risk
   threshold 应离开饱和区。

本仓库实例：escalate T=9.61 / topic T=4.82 / urgency T=7.97，
traj_ece 0.200→0.106，benchmark 见 `benchmarks/temp_laya_2026-09-25.md`。

---

## 10. 故障排查 FAQ

| 症状 | 原因与处理 |
|---|---|
| `--model requires an onnx build` | 二进制被普通 `zig build` 覆盖了；`zig build -Donnx=true -Donnx_lib_dir=...` 重建 |
| 400 BadModelIO | ΣlogitCount ≠ 图宽。最常见：noul 漏写 `"abstain": false`（默认 true 多一列） |
| serve 启动段错误（ RegisterCustomOpsLibrary 附近） | onnxruntime 与 ortextensions 版本不配/混用了 brew 的库；用 `export/laya/lib/` 这一对 |
| 400 InvalidJson | 请求体不是合法 JSON 或字段类型错（如 score 的 label 传了整数桶下标） |
| 400 InvalidGraph | 条件表达式语法错、引用了不存在的 decision、gate threshold 不在 [0,1] |
| traj/fit 半天没输出 | 不是卡死：日志只在批末打印；看 `ps cputime` 是否在涨 |
| topic/urgency 结果看着像瞎猜 | 已知模型局限（合成模板训练数据），见 §11 |
| urgency acc 恒 1.0 | 升级路径金标恒 high 的退化读数，非模型逆天 |

---

## 11. 已知限制

1. **合成模板训练数据**：Laya head 训在模板化工单上，自然文本上会有
   判错（标定只管置信度诚实，管不了判错）；换真实分布数据是最高优先
   后续项。
2. **escalate/urgency 标签共线**：escalate=true ⇒ 金标 urgency=high。
3. **noul 的 confidence 语义是 P(yes)**：traj by_node ece 与 fit ece
   不同口径，对比时先看 `benchmarks/temp_laya_2026-09-25.md` 的说明。
4. traj/fit 在 CPU 上串行，大模型约 2h/5652 条——交互评估的瓶颈。

---

## 12. 版本里程碑

| 里程碑 | 内容 |
|---|---|
| V0.1 | 核心运行时 + mock/ONNX 后端 + 校准协议 |
| V0.2 | Decision Graph（条件分支 + Gate + 全量预取多 wave） |
| M2 | Laya 冻结 head 训练 + 导出 + 实测（坏图读数已勘误） |
| M3 | 解冻末 2 层微调 + tokenizer [CLS]/[SEP] 修复图 |
| M4 | 温度标定端到端（本文 §9） |
