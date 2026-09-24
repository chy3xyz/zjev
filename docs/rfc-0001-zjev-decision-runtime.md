# ZJEV RFC-0001：Typed Probabilistic Decision Runtime —— 技术规格（Zig 0.17 实现稿）

- 状态：Draft（待评审）
- 日期：2026-09-24
- 上游文档：`docs/quest1.md`（原理还原）、`docs/prd.md`（产品级 RFC 草案）
- 目标工具链：Zig `0.17.0-dev.2151+2ec5523d5`（本机已装 nightly；0.16.0 已于 2026-04 发布，0.17 为当前开发线）。本文所有标准库断言均已对照该 nightly 实际编译验证。
- 本文定位：把 `prd.md` 的概念协议推进到**可直接开工**的接口与工程规格。协议数学不再重复推导，只给实现级定义。

---

## 0. 三个已拍板的技术取舍（含备选方案）

### 0.1 模型接入：手写 `extern` 绑定 ONNX Runtime C API（方案 A，采纳）

- **A. 手写 extern 声明（采纳）**。ONNX Runtime 的 C API 是自 1.x 起稳定的 vtable  ABI（`OrtApi` 获取函数表），V0.1 只需要约 30–40 个符号（Env/Session/Run/Tensor/Allocator 生命周期）。手写 `extern fn` + 精确的结构体布局完全可控、可审计。
- B. `addTranslateC` 构建步骤翻译 `onnxruntime_c_api.h`（0.16 起 `@cImport` 已废弃并在 0.17 nightly 中彻底移除，实测编译报错 `invalid builtin function: '@cImport'`）。备选保留：绑定面变大时切换到 B。
- C. Python 推理 sidecar（ONNX 跑在独立进程，Zig 走 IPC）。**否决**：引入第二个运行时与序列化开销，违背边缘侧低延迟目标；仅作为 ORT 绑定严重阻塞时的临时降级路径。

### 0.2 服务协议：HTTP + JSON 资源风格（采纳），非 JSON-RPC

- `POST /v1/decide`（单条）、`POST /v1/decide/batch`（批量）。schema 驱动的类型化响应本质上是"资源提交"，REST 形状比 JSON-RPC 更直读，curl 可调试。
- WebSocket 流式推送图执行进度推迟到 V0.2（`std.http.Server` 已内置 `respondWebSocket`/`readSmallMessage`/`writeMessage`，本机编译验证，届时无需引入依赖）。

### 0.3 并发模型：`std.Io` fiber + Group/Queue（采纳），不用线程池

- 0.16 已移除 `std.Thread.Pool`，官方替代是 `std.Io` 并发原语：`std.Io.Threaded`（执行器）、`std.Io.Group`（任务组）、`std.Io.Queue(T)`（多生产者多消费者阻塞队列，天然适合做 batching 调度）。
- 同步原语全部迁移：`std.Io.Mutex/Condition/RwLock/Semaphore`；锁自由结构不需要 `Io`。
- ONNX Runtime 自带 intra-op 线程池：在 ZJEV 调度下将其线程数设为 1，用 `std.Io.Queue` 做请求分片，避免嵌套过订阅。

---

## 1. 范围

### 1.1 V0.1 交付物（对应 prd.md M1+M3+M4+M5）

1. `src/core`：State / DecisionSchema / DecisionResult 完整类型系统 + JSON 编解码 + 校验。
2. `src/calib`：数值稳定 softmax、temperature scaling、Brier / ECE / MCE / NLL / selective risk 全套指标。
3. `src/model`：vtable 模型接口 + `mock.zig`（M1 纯引擎）+ `onnx.zig`（ONNX Runtime 绑定）。
4. `src/runtime`：engine（决策执行）、scheduler（批量调度）、cache（单飞缓存）。
5. `src/api`：`zjev-serve` HTTP 服务（`/v1/decide`、`/v1/decide/batch`、`GET /health`、`GET /v1/schema`）。
6. `tools/`：`zjev-bench`（评估）、`zjev-fit`（温度拟合 + calibration profile 落盘）。
7. 一致性测试：协议 conformance fixtures（JSON 文件驱动）。

### 1.2 非目标（V0.1 明确不做）

- GPU/WebGPU/Metal/CUDA 后端（V0.2–V0.5 路线，见 prd.md §23）。
- 训练与数据生产（Python 侧流水线，输出物仅为 `datasets/*.jsonl`；本仓库只消费）。
- Decision Graph 执行器与 trajectory 校准（M7，V0.2）。接口在 `src/graph` 预留，V0.1 不实现。
- 分布式、持久化、鉴权。单机进程内运行时。

---

## 2. 总体架构

```text
                HTTP / JSON
                     │
              ┌──────▼───────┐
              │  api/server  │  std.Io.Threaded + std.http.Server
              └──────┬───────┘
                     │ DecisionRequest (parsed & validated)
              ┌──────▼───────┐
              │   runtime    │  engine → scheduler → batch → cache
              │  ┌────────┐  │
              │  │ model  │──┼── mock.zig (M1)
              │  │ encoder│──┼── onnx.zig  (M3, ONNX Runtime C API)
              │  │ heads  │  │
              │  └────────┘  │
              │  ┌────────┐  │
              │  │ calib  │  │  softmax / temperature / metrics
              │  └────────┘  │
              └──────┬───────┘
                     │ DecisionResult[]
                     ▼
                  JSON out
```

一次请求的数据流：`parse → validate → encode(state) → per-schema head → logits → (profile) temperature → softmax → uncertainty 统计 → assemble result → serialize`。除 encode/head 依赖模型外，其余全部纯函数、可单测。

---

## 3. 核心类型（`src/core`）

### 3.1 设计约束

- 所有集合用 0.17 unmanaged 容器（`std.ArrayList(T)` 无 allocator 字段，`.empty` 初始化，方法显式传 `gpa`；`std.StringHashMap` 在本 nightly 仍是 managed 形态 `.init(gpa)`——以源码为准，两处形态并存，统一由 `core/alloc.zig` 提供别名收敛）。
- 每个公开函数显式接收 `std.mem.Allocator`；禁止全局分配器。
- 错误用显式 error set，跨模块组合。API 边界内不得 panic。

### 3.2 类型定义（草码，均按 0.17 语法编写）

```zig
// core/state.zig
pub const State = struct {
    id: ?[]const u8 = null,
    text: ?[]const u8 = null,
    data: ?std.json.Value = null,      // 任意 JSON 对象/标量
    embeddings: ?[]const f32 = null,
    timestamp_ms: ?i64 = null,
    source: ?[]const u8 = null,

    pub fn validate(s: *const State) Error!void; // 至少 text/data/embeddings 之一非空
};

// core/schema.zig
pub const DecisionType = enum { choice, noul, score, rank };

pub const ChoiceSchema = struct {
    id: []const u8,
    options: []const []const u8,       // 1..255 个；不允许 "__abstain__"
    abstain: bool = false,
};

pub const NoulSchema = struct {
    id: []const u8,
    abstain: bool = true,              // noul 默认允许弃权
};

pub const ScoreSchema = struct {
    id: []const u8,
    scale: union(enum) {               // score 是有序离散分布，不是回归标量
        int: struct { min: i16, max: i16 },          // max-min ≤ 255
        labels: []const []const u8,                 // 有序标签，≤255 个
    },
    abstain: bool = false,
};

pub const RankSchema = struct {
    id: []const u8,
    items: []const []const u8,        // 2..255 个，互不相同
};

pub const DecisionSchema = union(DecisionType) {
    choice: ChoiceSchema,
    noul: NoulSchema,
    score: ScoreSchema,
    rank: RankSchema,

    pub fn id(self: DecisionSchema) []const u8;
    pub fn validate(self: DecisionSchema, arena: std.mem.Allocator) Error!void; // 查重/查保留字/查上限
};

// core/result.zig
pub const Uncertainty = struct {
    entropy: ?f32 = null,              // choice/score/rank 有分布时计算
    variance: ?f32 = null,             // score 有
    confidence: f32,                   // = max prob（choice）/ 目标类 prob（noul）/ 期望 prob（score）
    abstention: ?f32 = null,           // abstain 开启时必给
};

pub const DecisionResult = struct {
    id: []const u8,
    type: DecisionType,
    value: Value,                      // 见下
    probability: ?f32 = null,          // noul: P(yes)
    probabilities: ?[]const f32 = null,// choice/score/rank: 与 options/桶/items 对齐
    labels: ?[]const []const u8 = null,// score(labels)/rank 时携带对齐标签
    uncertainty: Uncertainty,
    latency_us: u64,
};

pub const Value = union(enum) {
    choice: []const u8,                // 选中的 option id
    noul: bool,
    score: f32,                        // E[X]
    rank: []const RankEntry,           // 降序
};

pub const RankEntry = struct { id: []const u8, score: f32 };
```

### 3.3 保留字与校验规则（实现必须逐条落实）

1. `"__abstain__"` 为保留 id，不得出现在任何 `options/items/labels` 中。
2. `choice.options`、`rank.items`、`score.labels` 内部不得重复；`rank.items` 还需互异。
3. `score.int`：min < max，max-min ≤ 255；输出桶为闭区间整数 `[min..max]` 全部取整点。
4. 同一请求内所有 decision `id` 唯一。
5. 单 state 决策数上限：V0.1 为 **64**（prd 目标 ≥32，留出余量，硬上限防滥用）。
6. `state` 必须至少携带 `text/data/embeddings` 之一。

### 3.4 数值约定

- logits 一律 `f32`；概率输出 `f32`，四舍五入到 6 位小数后输出。
- softmax 必须做 max-subtraction 稳定化：`p_i = exp((z_i - z_max)/T) / Σ`。
- temperature 在 softmax **之前**作用于 logits；`T ≤ 0` 视为协议错误。
- 全零 logits + `T=1` 必须退化为均匀分布（M1 conformance case）。
- entropy 以自然对数计算；`p=0` 的项贡献为 0。

---

## 4. 决策语义（实现级定义）

### 4.1 choice

- head 输出 `options.len + (abstain?1:0)` 个 logits；最后一个 logit 对应 `__abstain__`。
- `value` = argmax（不含弃权时的常规项）；若弃权项 argmax 胜出，`value = "__abstain__"`。
- `uncertainty.confidence` = 常规项最大概率（不含弃权）。
- 输出 JSON 的 `probabilities` 为对象（id → prob），含弃权键（开启时）。

### 4.2 noul

- head 输出 2（+1 弃权）个 logits，sigmoid 化看待：`P(yes)=exp(z_yes/T)/(exp(z_yes/T)+exp(z_no/T))`。
- `value` = `P(yes) ≥ 0.5`；`probability` = `P(yes)`；`abstention` = `P(abstain)`。

### 4.3 score

- head 输出 `桶数 + (abstain?1:0)` 个 logits，softmax 得到有序离散分布。
- `value` = `E[X] = Σ x_i·p_i`（int 桶用整数值，labels 桶用序号 1..n 的映射仅用于内部统计，输出仍给标签分布）。
- `variance = E[X²] − E[X]²`（clamp 到 ≥0 防浮点负数）；`entropy` 按分布计算。
- 输出 `probabilities`：int 桶时为对象（桶值字符串 → prob），labels 桶时为对象（标签 → prob）。

### 4.4 rank

- V0.1：**逐项 sigmoid** 独立打分（每 item 一个 logit → sigmoid → `[0,1]`），`rank` = 按分降序（同分按 id 字典序稳定排序）。
- `probabilities` 与 `labels`（items 对齐，原始顺序）同时输出；`confidence` = 最高分；`entropy` 对 items 归一化 softmax 后计算。
- rank 在 V0.1 **不内置弃权**（items 集合本身可包含 `"none"` 之类的显式项来表达"都不选"）。

---

## 5. 模型层（`src/model`）

### 5.1 接口：vtable 运行时多态

模型需要在运行时切换（mock / onnx / 未来 native），故不用 comptime duck typing，用 vtable：

```zig
// model/encoder.zig
pub const HiddenState = opaque {};

pub const Encoder = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        encode: *const fn (ptr: *anyopaque, alloc: std.mem.Allocator, state: *const State) Error!*HiddenState,
        deinit: *const fn (ptr: *anyopaque, alloc: std.mem.Allocator, hidden: *HiddenState) void,
    };

    pub fn encode(self: Encoder, alloc: std.mem.Allocator, state: *const State) Error!*HiddenState {
        return self.vtable.encode(self.ptr, alloc, state);
    }
};

// model/head.zig —— 一次 head 调用可同时服务请求内所有同型 schema（摊薄 encoder 成本）
pub const HeadKind = DecisionType;
pub const Head = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// 返回 logits 布局：schema_index 主序的连续 f32，
        /// 每段长度 = schema 的 logit 数（含弃权位）。
        decide: *const fn (
            ptr: *anyopaque,
            alloc: std.mem.Allocator,
            hidden: *HiddenState,
            schemas: []const DecisionSchema,
        ) Error![]f32,
    };
};
```

要点：

- **encoder 与 head 彻底解耦**（prd.md §11）：`HiddenState` 为 opaque，由模型实现持有（如 ONNX 的 last_hidden_state tensor）。
- 一次 `encode` 服务请求内全部 schema——这是 "one state → many decisions" 的成本模型根基（quest1.md §9）。
- `model/logits.zig` 提供 logits 视口类型（`LogitsView{ base: []f32, schema: DecisionSchema }`），把连续 buffer 按 schema 切分。

### 5.2 mock.zig（M1）

- 不做任何 I/O：用 `state.id` 的 FNV-1a 哈希作种子，走 `std.Random` 生成确定性强伪随机 logits（同输入必同输出，conformance 可复现）。
- 提供 `mock.uniform` / `mock.peaked` / `mock.sequence`（按请求内 schema 序号轮换尖峰位置）三种模式，build option 选择，供校准数学与调度测试使用。

### 5.3 onnx.zig（M3）：ONNX Runtime C API 绑定

- 手写 `extern struct` + `extern fn`，核心符号：`OrtGetApiBase` → `OrtApi` 函数表（结构体布局逐字段对齐官方 C 头，只写用到的字段前缀并静态断言 `sizeOf`）。
- 关键 C 符号（V0.1 子集）：`CreateEnv`、`CreateSessionOptions`、`SetIntraOpNumThreads(1)`、`CreateSession`、`Run`、`GetTensorTypeAndShape`、`CreateCpuMemoryInfo`、`Release*` 全套。
- tokenizer 不入 runtime：V0.1 要求导出的 ONNX 图内部包含 tokenize（ModernBERT 导出时前置 WordPiece 实现为图内 op，或离线预 tokenize 为 input_ids 传入——**采用前者**：tokenizer 固化进图，Zig 侧只传 string tensor，API 边界保持"State 进、Logits 出"）。
- 多线程：ORT session 的 intra-op 线程设 1；跨请求并行由 `std.Io` 调度 + 多 session（每 worker fiber 一个 session 实例，N = CPU 核数一半，启动时建立池）。
- 构建：`build.zig` 新增 option `-Donnx=true|false`（默认 false，M1 无外部依赖可跑全部测试）；true 时 `exe.linkSystemLibrary("onnxruntime")` 并链接 `lib/` 路径。

### 5.4 Model 工厂

```zig
// model/factory.zig
pub const Config = struct {
    kind: enum { mock, onnx },
    mock_mode: MockMode = .peaked,
    model_dir: ?[]const u8 = null,   // onnx 时必填：*.onnx + calibration profiles
    num_sessions: u16 = 0,           // 0 = auto( max(1, cores/2) )
};

pub fn open(alloc: std.mem.Allocator, io: std.Io, cfg: Config) Error!Model;
pub const Model = struct {
    encoder: Encoder,
    heads: [4]Head,                  // 按 DecisionType 索引
    deinit: fn(...) void,
};
```

---

## 6. 校准系统（`src/calib`）

### 6.1 模块

| 文件 | 职责 |
|---|---|
| `softmax.zig` | 稳定 softmax（含 temperature）、log-softmax |
| `temperature.zig` | 应用/拟合 T（网格搜索 + 二分精调，NLL 目标） |
| `brier.zig` | BS / 多类 BS |
| `ece.zig` | ECE / MCE（默认 15 个等宽置信区间） |
| `stats.zig` | NLL、selective risk @ coverage、abstention accuracy、entropy/variance |
| `profile.zig` | profile 加载/查询/落盘 |

### 6.2 Calibration Profile（落实 prd.md §13）

```json
{
  "model": "zjev-150m-v0.1",
  "task": "choice",
  "num_options": 4,
  "domain": "general",
  "temperature": 1.37,
  "metrics": { "ece": 0.034, "brier": 0.081, "nll": 0.42, "mce": 0.11,
               "abstention_accuracy": 0.87, "selective_risk@0.9": 0.05 },
  "fitted_at": "2026-09-24T00:00:00Z",
  "val_size": 12000
}
```

- 查询键 `(model, task, num_options, domain)`；`domain` 缺省 `"general"`。
- 运行时加载整个 profile 目录（`model/calibration/*.json`）到 `std.StringHashMap`，key 为上述四元组拼串。
- 未命中 profile 时：**不加温（T=1）并在结果 `calibration` 字段标记 `"default"`**，绝不失败。
- `zjev-fit` 工具：`--model --task --domain --dataset datasets/calibration.jsonl --out model/calibration/`，输出上述 profile 文件。

---

## 7. 决策图与策略门（`src/graph`，V0.1 仅接口 + 类型，V0.2 实现）

- `node.zig`：图节点 = `{ id, decision: schema_id }`；`edge.zig`：边 = `{ from, to, when: Condition }`。
- `condition.zig`：条件表达式编译期解析为 AST（禁止运行时字符串 eval）。支持：`result_id.value == "x"`、`!=`、`>`、`<`、`>=`、`<=`、`confidence`、`abstention` 字段引用、`and/or/not`。
- `executor.zig`（V0.2）：拓扑遍历 + 条件短路；沿路径累积记录 trajectory。
- **trajectory 校准**（quest1.md §12–13）：整轨迹执行后输出 trajectory-level ECE/Brier，验证校准不可简单串联；执行策略遵循 deferred crispification——在 policy gate 之前一律保持概率形态，门控处才 crispify。V0.1 仅在 `gate.zig` 提供单机策略门原语：`Gate{ threshold, action_above, action_below, action_abstain }`，供 M6 Agent 集成直接使用。

---

## 8. Runtime 与并发（`src/runtime`）

### 8.1 结构

- `engine.zig`：纯逻辑。输入 `(State, []DecisionSchema, Model, Profiles)` → `[]DecisionResult`。无 I/O，可任意并发重入（所有状态经参数/栈传递）。
- `scheduler.zig`：`std.Io.Queue(Pending)` 为入口；按 `max_batch`（默认 8，build option 可调）聚合后整批送一个 session 执行；超时 `batch_timeout_us`（默认 500µs）不满批也发。
- `cache.zig`：key = `hash(model_id ‖ state.id ‖ schema_set_hash)`；`std.Io.Mutex` 保护的 `std.StringHashMap(CacheEntry)`；TTL 默认 0（关闭）。
- 会话池：`sessions: std.ArrayList(*Session)` + `std.Io.Queue(*Session)` 空闲队列；worker fiber 数 = `num_sessions`。

### 8.2 server 主循环（已对 nightly 编译验证的骨架）

```zig
pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var model = try model_mod.open(gpa, io, cfg);
    defer model.deinit();

    var group: std.Io.Group = .init;
    defer group.cancel(io);

    const addr = try std.Io.net.IpAddress.parse(bind_host, bind_port);
    var listener = try addr.listen(io, .{ .mode = .stream });
    defer listener.deinit(io);

    for (0..num_workers) |_| group.async(io, connWorker, .{ io, &listener, &shared });
    try group.await(io);
}
```

- 每连接一 fiber；fiber 内 `std.http.Server.init(&reader.interface, &writer.interface)` → `receiveHead()` 循环 → `respond()`。keep-alive 默认开。
- 请求内存：每请求一个 `std.heap.ArenaAllocator`，请求结束整体释放；解析用 `std.json.parseFromSliceLeaky` 进 arena。

### 8.3 背压与超时

- `Queue(T)` 容量 = `max_queue`（默认 1024）；满则新请求直接 503。
- 请求级 deadline 默认 5000ms；超时应答 `{"error": {"code": "timeout", ...}}`。

---

## 9. API 协议（`src/api`）

### 9.1 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | `{"status":"ok","model":"...","version":"..."}` |
| GET | `/v1/schema` | 返回 DecisionSchema 的 JSON-Schema 描述（自描述协议） |
| POST | `/v1/decide` | 单 DecisionRequest → DecisionResponse |
| POST | `/v1/decide/batch` | `{"requests":[...]}` → `{"responses":[...]}`，独立失败互不影响 |

### 9.2 请求（与 prd.md §2 一致，补两条规则）

- `decisions[]` 空数组 → 400。
- `policy` 字段 V0.1 仅接受 null/缺省（图与策略编排未上线，显式拒绝防误解）。

### 9.3 响应

成功：`{"results":[DecisionResult...],"calibration":"matched"|"default"}`。
`DecisionResult` 字段与 prd.md §8 对齐（`probabilities` 输出为对象；数组形态仅供 `labels` 对齐内部使用，不出现在 JSON）。

### 9.4 错误映射

| HTTP | code | 触发 |
|---|---|---|
| 400 | `invalid_request` | JSON 解析失败 / 校验规则（§3.3）违反 |
| 400 | `unsupported` | `type` 或 `policy` 不被 V0.1 支持 |
| 408 | `timeout` | 超过 deadline |
| 429 | `overloaded` | 队列满 |
| 500 | `internal` | 模型/运行时错误（附 `message`，不泄露路径） |

错误体统一：`{"error":{"code":"...","message":"...","request_id":"..."}}`。

---

## 10. 工程结构

```text
zjev/
├── build.zig                  # 见 §10.2
├── build.zig.zon
├── src/
│   ├── main.zig               # zjev-serve 入口（juicy main）
│   ├── core/
│   │   ├── alloc.zig          # 容器别名与分配器约定
│   │   ├── state.zig
│   │   ├── schema.zig
│   │   ├── result.zig
│   │   └── error.zig          # 全局 error set + API 映射
│   ├── model/
│   │   ├── encoder.zig / head.zig / logits.zig
│   │   ├── factory.zig
│   │   ├── mock.zig
│   │   └── onnx.zig           # extern 绑定（-Donnx 时编译）
│   ├── calib/
│   │   ├── softmax.zig / temperature.zig / brier.zig
│   │   ├── ece.zig / stats.zig / profile.zig
│   ├── runtime/
│   │   ├── engine.zig / scheduler.zig / batch.zig / cache.zig
│   ├── graph/                 # V0.1 仅类型与 gate.zig
│   │   ├── node.zig / edge.zig / condition.zig / executor.zig / gate.zig
│   └── api/
│       ├── json.zig           # 协议编解码（std.json 之上）
│       ├── server.zig         # 连接循环
│       └── routes.zig         # 端点分发
├── tools/
│   ├── bench.zig              # zjev-bench
│   ├── fit.zig                # zjev-fit
│   └── conformance.zig        # fixtures  runner
├── model/                     # *.onnx + calibration/*.json（git 不存权重）
├── datasets/                  # *.jsonl（git 只存小样例）
├── benchmarks/                # 结果输出目录
├── examples/
└── test/
    └── conformance/           # JSON fixtures
```

### 10.2 build.zig 选项（0.17 注意：`b.args` 已移除，命令行参数经 `init.minimal.args` 进程序；构建期开关用 `-D`）

```zig
// 关键片段
const onnx = b.option(bool, "onnx", "Enable ONNX Runtime backend") orelse false;
const mock_mode = b.option(enum { uniform, peaked, sequence }, "mock_mode", "") orelse .peaked;

const exe = b.addExecutable(.{
    .name = "zjev-serve",
    .root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{},
    }),
});
if (onnx) {
    exe.root_module.addCMacro("ZJEV_ONNX", "1");
    exe.linkSystemLibrary("onnxruntime");
}
```

- 测试步骤：`zig build test`（0.16 起支持单元测试超时，默认 60s/文件，conformance 慢测单独 step `zig build test-conformance`）。
- 本地包覆盖用 0.16 的 "Override Packages Locally"（`zig fetch --save` + 本地路径覆盖），V0.1 无外部 Zig 依赖，预留。

### 10.3 测试策略

1. 单元测试：每个数值模块（softmax/brier/ece/stats）带已知答案用例（含全零 logits、单桶、全弃权边界）。
2. 协议 conformance：`test/conformance/*.json`（请求 + 期望响应），`zig build test-conformance` 以 mock 模式跑全量；CI 必跑。
3. 属性测试：schema 校验器随机生成非法输入断言必拒。
4. 集成：`zjev-serve` 起在随机端口，python/httpie 风格脚本打 `/v1/decide` 断言端到端（tools 内 zig 实现，不引入 Python 依赖）。

---

## 11. 数据集与评估（消费侧）

- 格式沿用 prd.md §15 的 `datasets/*.jsonl`（state/decision/label[/teacher]）。`zjev-bench --dataset ... --model ... --profile-dir ...` 输出 metrics JSON（§6.2 的 metrics 字段全集）+ 逐条预测 dump。
- 指标口径：Brier 多类定义、ECE=15 等宽区间、selective risk 在 coverage ∈ {0.5,0.7,0.9,0.95} 各算一档；abstention accuracy 仅在数据集带弃权标签时输出。
- 报告写入 `benchmarks/<model>_<dataset>_<ts>.json`，`zjev-fit` 直接消费其 dump 拟合 profile。

---

## 12. 里程碑（V0.1 收口定义）

| 里程碑 | 内容 | 出口标准 |
|---|---|---|
| M0 | 本 RFC + `core` 类型与校验 | conformance 校验用例全绿 |
| M1 | mock + calib + engine | 数值单测全绿；mock 全链路出正确 DecisionResult |
| M2 | （数据侧，仓库外） | 产出 `datasets/*.jsonl` |
| M3 | onnx.zig 接通 ModernBERT 导出图 | 真模型输出与 mock 走同一 conformance 管道 |
| M4 | zjev-fit + profiles + bench | 输出带 ECE/Brier 的 profile 文件；bench 可复现 |
| M5 | server + scheduler + cache | 压测：batch=8、2K context 下 P50 ≤ 20ms（CPU）、无泄漏（DebugAllocator 报告干净） |
| M6 | zharness 集成（System1/System2） | 在 zharness 示例 agent 中以 gate 方式调用 zjev |
| M7 | graph executor + trajectory 校准（V0.2） | 图执行 + 轨迹级 ECE 报告 |

V0.1 = M0–M5 + M6 的 gate 子集。

---

## 13. Zig 0.17 适配要点汇总（已验证清单）

> 工具链：`0.17.0-dev.2151+2ec5523d5`（macOS aarch64）。以下条目均经本机编译验证或官方 0.16.0 release notes / 2026-08 devlog 核对。

| 主题 | 0.17 现实 | 对本项目的影响 |
|---|---|---|
| `@cImport` | **已移除**（编译报错） | ONNX 绑定手写 extern（§5.3） |
| `std.Thread.Pool` | 0.16 已移除 | 用 `std.Io.Group`/`std.Io.async`（§8） |
| 线程同步 | `std.Thread.Mutex` 等迁往 `std.Io.Mutex/Condition/RwLock/Semaphore`；锁自由原语无需 Io | 调度器/缓存同步全部走 std.Io |
| 并发队列 | `std.Io.Queue(T)` MPMC、阻塞、容量运行时可调 | scheduler 入口（§8.1） |
| 执行器 | `std.Io.Threaded`（线程池+ fiber）；另有 `std.Io.Evented`（kqueue/uring） | serve/批处理统一 Threaded；Evented 留作 Linux 高性能变体 |
| main 签名 | `pub fn main(init: std.process.Init) !void`，`init.gpa/io/arena/minimal.args` | 入口与测试 CLI 统一 |
| 网络 | `std.net` 已删除；`std.Io.net.IpAddress.parse(host,port)`、`listen(io,.{.mode=.stream})`、`Server.accept(io)` | server 骨架（§8.2） |
| HTTP | `std.http.Client` 全异步重写；`std.http.Server` 存在但按连接手工驱动：`init(&reader.interface,&writer.interface)`、`receiveHead()`、`respond()`；内置 WebSocket | API 层按 §8.2 骨架实现 |
| 容器 | `std.ArrayList` 等迁 unmanaged（`.empty` + 显式 allocator）；ArrayList 新增 pointer-stability lock（0.17 devlog）；`std.StringHashMap` 本 nightly 仍为 managed 形态 | `core/alloc.zig` 收敛形态差异 |
| 分配器 | `std.heap.smp_allocator`、`DebugAllocator`、`ArenaAllocator`（线程安全无锁）、`c_allocator` | release 用 smp，debug/test 用 DebugAllocator 查泄漏 |
| `std.json` | `parseFromSlice`/`parseFromValue`/`Stringify`/`fmt` 健在 | 协议编解码直接可用 |
| 语言 | `**` 数组重复运算符移除（用 `@splat`）；`@Type` 移除（用 `@Int/@Enum 等`）；packed union 限制收紧；lazy field analysis | 写码时规避，code review checklist 加这几条 |
| 构建 | `b.args` 移除；`b.addTranslateC` 替代 `@cImport`；包可本地化覆盖 | build.zig 按 §10.2 |
| 类型反射 | `std.builtin.Type` 本 nightly 仍可用（`std.lang.Type` 并存，0.17 定型中） | 仅测试代码使用，集中隔离 |

风险登记：① `std.http.Server` 在 0.17 正式版可能继续演进——API 层集中 `api/server.zig` 单文件隔离，变动只改一处；② `std.StringHashMap` unmanaged 化若落地，只需改 `core/alloc.zig`；③ ORT 的 macOS arm64 动态库分发不入库，README 给安装命令，`zig build -Donnx=true` 校验链接。

---

## 14. 附录：一次 /v1/decide 的完整样例

请求：

```json
{
  "state": { "text": "用户近 24h 链上交易 17 笔，余额 3.2 SOL，合约交互 8 次" },
  "decisions": [
    { "id": "risk_level", "type": "choice",
      "options": ["low", "medium", "high"], "abstain": true },
    { "id": "churn_7d", "type": "score", "scale": { "min": 1, "max": 5 } }
  ]
}
```

响应（mock 模式，T 来自 profile 命中）：

```json
{
  "results": [
    {
      "id": "risk_level", "type": "choice", "value": "medium",
      "probabilities": { "low": 0.12, "medium": 0.73, "high": 0.09, "__abstain__": 0.06 },
      "uncertainty": { "entropy": 0.802, "confidence": 0.73, "abstention": 0.06 },
      "latency_us": 384
    },
    {
      "id": "churn_7d", "type": "score", "value": 3.63,
      "probabilities": { "1": 0.02, "2": 0.11, "3": 0.24, "4": 0.48, "5": 0.15 },
      "uncertainty": { "entropy": 1.208, "variance": 0.92, "confidence": 0.48 },
      "latency_us": 96
    }
  ],
  "calibration": "matched"
}
```

---

## 15. 评审检查单（合并前必答）

1. §3.3 的硬上限（choice ≤255、单请求 ≤64 decisions）是否与 prd.md §26 指标表一致？——一致（255 对齐；64 ≥ 32 达标）。
2. §4.3 score 的 labels 桶输出形态（标签→prob 对象）是否满足"数"的语义？——满足，E[X] 以序号映射内部计算并在文档明示。
3. §5.3 tokenizer 固化进 ONNX 图的方案是否可接受？——若坚持运行时 tokenizer，需新增 `model/tokenize.zig`（推迟到 V0.2 再评估）。
4. §9.2 对 `policy` 显式拒绝的策略是否有异议？
5. §12 M5 的 P50 ≤ 20ms 目标是否合理（待 M3 后首次真机测量复核，此处为先验值）。
