# Executor 全量预取（prefetch）设计

日期：2026-09-24
状态：已批准（auto 模式 inline 决策）
前置：`docs/superpowers/specs/2026-09-24-laya-onnx-export-design.md`（bundled 头契约）

## 1. 问题

`executor.execute`（`src/graph/executor.zig`）沿 DAG frontier 分批调用 `engine.run`：
每个 wave 只把该波激活节点的 schema 传给模型。这对 mock 头可行，但与 bundled 头
契约冲突——bundled 头（ONNX 导出，`src/model/head.zig` 的 `bundled=true`）要求
单次 `decide()` 消费**全图 schema 集**，输出 flat logits 宽度 = Σ logitCount(全图)。
多 wave（含条件边）图中任一批的 Σ logitCount ≠ 图宽 → `BadModelIO` → 400。

后果：ONNX 后端只能跑平图（edges 为空），条件图（V0.2 的核心价值）不可用。

## 2. 方案：executor 全量预取

把「分批跑模型」改为「一次跑全图，再模拟激活波」：

1. **构建运行集**：按图节点顺序（`g.nodes` 主序）收集每个节点的 decision 对应
   schema。`graph.validate` 已保证 node→decision 一对一（`DuplicateDecisionRef`
   拒绝复用），且图非空（`EmptyGraph` 拒绝）、无环，因此运行集大小恒等于节点数，
   无需去重。
2. **单次前向**：一次 `engine.run(a, model, s, run_schemas, run_temps)`。
   - bundled 头：一次 `decide()` 消费全图 schema，天然匹配导出契约。
   - 非 bundled 头：走 engine 内既有按 type 分组路径（≤4 次 decide，全量传入）。
3. **模拟 wave**：用预取结果跑原有 frontier 激活模拟（indeg-0 入波、wave 内按
   节点 id 排序、逐节点出边条件求值、`gate.apply`、`path_prob` 连乘
   `trajectory.massFor`）。每节点结果取自预取集（节点序下标）。

`temps` 映射：运行集第 j 项 = 节点 j 的 decision 在请求 schema set 中的下标 →
`run_temps[j] = temps[setIdx]`。

### 2.1 语义等价论证

- 真模型（encoder+head）：每个 schema 的 logits 只依赖 hidden state，
  与批组成/批内位置无关 → 逐 schema 结果与分批调用相同。
- trajectory 顺序 = wave 序（wave 内按节点 id 字典序），与现实现一致。
- `path_prob` = 已执行节点 massFor 连乘，节点集合与相乘顺序不变。
- `skipped` = 未执行节点，图序，不变。
- mock `.sequence` 模式：wave-2+ 节点的批内下标不再每波重置（峰值位置变为
  全局节点序）。仅影响测试可见行为；已逐个核对现有 5 个 executor 测试与 5 个
  graph conformance 夹具，全部保持绿色（见 §5 验收）。
- 错误时机：模型 IO 错误（如 bundled 宽度不匹配）在任何 wave 模拟前抛出
  （fail-fast）。响应本来就是 all-or-nothing，无响应级差异；且对 bundled 而言
  宽度错误本就该拒绝。

### 2.2 边界

- 请求 schema set 中未被图引用的额外 schema：旧实现不运行它们，新实现同样
  排除在运行集外（运行集由图节点驱动）。若 bundled 图宽与运行集 Σ logitCount
  不等，仍 400 BadModelIO——契约不变。
- 被剪枝分支的节点也被计算（浪费前向，换取 bundled 兼容）。整图 encoder 前向
  从「每 wave 一次」降为「全图一次」，净前向次数不增。
- 空图不可达（validate 拒绝）。

## 3. 否决的替代方案

1. **executor 内按头类型分叉**（bundled 预取 / 非 bundled 维持分批）：
   保留两条执行路径，行为随模型后端静默改变，长期 divergence 风险。否决。
2. **有状态头缓存**（bundled 头一次算全量 logits 缓存，后续 wave 切片）：
   在头里藏请求级状态，破坏头的无状态契约，并发下需额外同步。否决。

## 4. 改动面

- `src/graph/executor.zig`：重写 execute 主循环为「构建运行集 → 单次
  engine.run → wave 模拟」。`ExecError`、Outcome 结构、对外签名不变。
- `src/runtime/engine.zig`、`src/model/head.zig`：不改。
- 新增测试（executor.zig 内）：
  - bundled mock 头 + 多 wave 条件图 → 单次 decide 调用、trajectory/skipped 正确；
  - 未引用额外 schema 不参与运行；
  - 既有 5 测试保持绿（mock sequence 行为核对）。
- README：删除「多 wave ONNX 暂不支持」限制段，补预取语义一句。
- e2e：Laya ONNX + 两 wave 条件图（noul 显式 `"abstain":false`）→ 200 +
  trajectory 两步（此前 400）。

## 5. 验收

1. `zig build test` exit 0（含新增测试）。
2. `zig build -Donnx=true -Donnx_lib_dir=$PWD/export/laya/lib test` exit 0。
3. `test-conformance` 10 pass 0 fail。
4. e2e 条件图 200（命令见 README runbook；serve 起后 curl 两 wave 图）。
5. README 与实际行为一致。
