# Executor 全量预取 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** executor 改为全量预取（一次前向算全图节点，再模拟激活波），使 bundled ONNX 头支持多 wave 条件图。

**Architecture:** `execute` 先按图节点主序构建运行 schema 集并单次调用 `engine.run`，然后用返回的逐节点结果跑原有 frontier 激活模拟。对外签名与 Outcome 不变；engine/head 不动。

**Tech Stack:** Zig 0.17.0-dev.2151+2ec5523d5（zigup 管理的 `zig`）。

## Global Constraints

- 测试命令：`zig build test`（exit 0）；改了 `src/model/onnx.zig` 时还需 `zig build -Donnx=true -Donnx_lib_dir=$PWD/export/laya/lib test`。本计划不碰 onnx.zig，但 e2e 需 onnx 构建。
- conformance：`zig build test-conformance` → 输出 `10 pass, 0 fail`。
- 分支流：每 Task 一 commit；全部完成后 `--no-ff` 合 main、删分支。
- 测试输出重定向到文件再查 `$?`，避免长输出刷屏。
- commit 不 add `zig-out/` 产物。
- 设计契约全文：`docs/superpowers/specs/2026-09-24-executor-prefetch-design.md`（尤其 §2.1 语义等价论证）。

---

### Task 1: executor 预取重写 + 单元测试

**Files:**
- Modify: `src/graph/executor.zig`（`execute` 函数体重写，第 32–119 行；测试追加文件尾部）
- Test: `src/graph/executor.zig` 内新增两个测试

**Interfaces:**
- Consumes: `engine.run(a, model, s, schemas, temps) !RunOutcome`（`src/runtime/engine.zig:18`，`RunOutcome.results` 长度 == 传入 schemas 长度，同下标对应）；`factory.Model.heads: std.EnumArray(schema.DecisionType, head.Head)`（`src/model/factory.zig:11`）；`head.Head{ ptr, vtable, bundled }`（`src/model/head.zig:16`）。
- Produces: `execute(a, model, s, schemas, g, temps) ExecError!Outcome` 签名与语义不变（`src/graph/executor.zig:32`）；新增测试可见的约定：bundled 头对任意图只被调用一次、传入 schema 数 == 图节点数。

- [ ] **Step 1: 写失败测试**

在 `src/graph/executor.zig` 文件尾追加（helpers 与既有测试同名函数复用：`cmpEq`、`two_choice_schemas`、`newArena`）。
注意：不能用既有 helper `execIn`——它内部自建 mock 模型，无法注入带 counting 头的模型；
两个新测试都直接调 `execute` 传自建模型。

```zig
const model_encoder = @import("../model/encoder.zig");
const model_head = @import("../model/head.zig");
const model_logits = @import("../model/logits.zig");

const Count = struct {
    calls: usize = 0,
    last_schema_count: usize = 0,
};

fn countingDecide(
    ptr: *anyopaque,
    a: alloc.Allocator,
    hidden: *model_encoder.HiddenState,
    schemas: []const schema.DecisionSchema,
) model_head.Error![]f32 {
    _ = hidden;
    const self: *Count = @ptrCast(@alignCast(ptr));
    self.calls += 1;
    self.last_schema_count = schemas.len;
    var total: usize = 0;
    for (schemas) |sc| total += model_logits.logitCount(sc);
    const buf = try a.alloc(f32, total);
    @memset(buf, 0);
    var off: usize = 0;
    for (schemas) |sc| {
        const n = model_logits.logitCount(sc);
        buf[off] = 4.0; // 每个 schema 峰在下标 0 → choice 取首选项 / noul=true
        off += n;
    }
    return buf;
}

const counting_vtable: model_head.VTable = .{ .decide = countingDecide };

test "execute prefetches whole graph in one bundled call" {
    var arena = newArena();
    defer arena.deinit();
    const a = arena.allocator();
    var count: Count = .{};
    var m = try factory.mockModel(.sequence, a);
    defer m.deinit(a);
    m.heads.set(.noul, .{ .ptr = &count, .vtable = &counting_vtable, .bundled = true });

    const nodes = [_]graph_mod.Node{
        .{ .id = "r", .decision = "c1" },
        .{ .id = "hit", .decision = "c2" },
        .{ .id = "miss", .decision = "c2b" },
    };
    var schemas: [4]schema.DecisionSchema = undefined;
    schemas[0] = two_choice_schemas[0];
    schemas[1] = two_choice_schemas[1];
    schemas[2] = .{ .choice = .{ .id = "c2b", .options = &.{ "u", "v" }, .abstain = false } };
    schemas[3] = .{ .noul = .{ .id = "unused-noul", .abstain = false } };
    const edges = [_]graph_mod.Edge{
        .{ .from = "r", .to = "hit", .when = cmpEq("c1", "a") },
        .{ .from = "r", .to = "miss", .when = cmpEq("c1", "b") },
    };
    const g: graph_mod.Graph = .{ .nodes = &nodes, .edges = &edges };
    const s: state.State = .{ .text = "x" };

    const out = try execute(a, &m, &s, &schemas, g, null);
    try std.testing.expectEqual(@as(usize, 1), count.calls);
    try std.testing.expectEqual(@as(usize, 3), count.last_schema_count);
    try std.testing.expectEqual(@as(usize, 2), out.steps.len);
    try std.testing.expectEqualStrings("hit", out.steps[1].node_id);
    try std.testing.expectEqual(@as(usize, 1), out.skipped.len);
    try std.testing.expectEqualStrings("miss", out.skipped[0]);
}
```

（再补一个非 bundled 回归测试，锁定「额外 schema 不被运行」在非 bundled 路径同样成立：）

```zig
test "execute skips request schemas not referenced by graph" {
    // 非 bundled：运行集 = 图节点引用的 schema；额外 schema 不进入任何 head 调用。
    var arena = newArena();
    defer arena.deinit();
    const a = arena.allocator();
    var count: Count = .{};
    var m = try factory.mockModel(.sequence, a);
    defer m.deinit(a);
    m.heads.set(.choice, .{ .ptr = &count, .vtable = &counting_vtable, .bundled = false });

    const nodes = [_]graph_mod.Node{
        .{ .id = "n1", .decision = "c1" },
        .{ .id = "n2", .decision = "c2" },
    };
    var schemas: [3]schema.DecisionSchema = undefined;
    schemas[0] = two_choice_schemas[0];
    schemas[1] = two_choice_schemas[1];
    schemas[2] = .{ .noul = .{ .id = "unused-noul", .abstain = false } };
    const edges = [_]graph_mod.Edge{
        .{ .from = "n1", .to = "n2", .when = cmpEq("c1", "a") },
    };
    const g: graph_mod.Graph = .{ .nodes = &nodes, .edges = &edges };
    const s: state.State = .{ .text = "x" };

    const out = try execute(a, &m, &s, &schemas, g, null);
    // 非 bundled 路径按 type 分组调用 choice 头：全量传入 2 个图引用 schema，一次调用
    try std.testing.expectEqual(@as(usize, 1), count.calls);
    try std.testing.expectEqual(@as(usize, 2), count.last_schema_count);
    try std.testing.expectEqual(@as(usize, 2), out.steps.len);
}
```

- [ ] **Step 2: 跑测试确认红**

Run: `zig build test > /tmp/prefetch-red.log 2>&1; echo "exit=$?"; grep -E "prefetch|skips request|passed|failed" /tmp/prefetch-red.log | tail -20`
Expected: exit 非 0；`execute prefetches whole graph in one bundled call` FAIL（旧代码每 wave 调一次 bundled 头：先 1 个 schema、再 1 个 → `count.calls == 2 != 1`）。

- [ ] **Step 3: 重写 `execute` 为全量预取**

把 `src/graph/executor.zig` 中 `execute` 函数体（第 32–119 行）替换为：

```zig
pub fn execute(
    a: alloc.Allocator,
    model: *const factory.Model,
    s: *const state.State,
    schemas: []const schema.DecisionSchema,
    g: graph_mod.Graph,
    temps: ?[]const f32,
) ExecError!Outcome {
    try s.validate();
    try schema.validateSet(schemas, a);
    try graph_mod.validate(g, schemas);

    const n = g.nodes.len;

    // 全量预取：按图节点主序单次前向。bundled 头契约要求一次 decide 消费
    // 全图 schema（图宽 = Σ logitCount）；validate 保证 node→decision 一一对应，
    // 故 run_schemas[i] 对应 g.nodes[i]，结果按下标直取。
    var run_schemas: std.ArrayList(schema.DecisionSchema) = .empty;
    var run_temps: std.ArrayList(f32) = .empty;
    for (g.nodes) |nd| {
        const si = schemaById(schemas, nd.decision).?;
        try run_schemas.append(a, schemas[si]);
        try run_temps.append(a, if (temps) |ts| ts[si] else 1.0);
    }
    const outcome = try engine.run(a, model, s, run_schemas.items, run_temps.items);

    const activated = try a.alloc(bool, n);
    @memset(activated, false);
    const executed = try a.alloc(bool, n);
    @memset(executed, false);

    var frontier: std.ArrayList(usize) = .empty;
    for (g.nodes, 0..) |nd, i| {
        var indeg: usize = 0;
        for (g.edges) |e| {
            if (std.mem.eql(u8, e.to, nd.id)) indeg += 1;
        }
        if (indeg == 0) {
            activated[i] = true;
            try frontier.append(a, i);
        }
    }

    var steps: std.ArrayList(trajectory.Step) = .empty;
    var done_results: std.ArrayList(result.DecisionResult) = .empty;
    var skipped: std.ArrayList([]const u8) = .empty;
    var path_prob: f32 = 1.0;

    while (frontier.items.len > 0) {
        std.mem.sort(usize, frontier.items, g.nodes, byId);

        for (frontier.items) |ni| {
            executed[ni] = true;
            const r = outcome.results[ni];
            var step: trajectory.Step = .{
                .node_id = g.nodes[ni].id,
                .decision_id = r.id,
                .result = r,
            };
            if (g.nodes[ni].gate) |gt| step.action = gt.apply(r);
            try steps.append(a, step);
            try done_results.append(a, r);
            path_prob *= try trajectory.massFor(a, r, run_schemas.items[ni]);

            for (g.edges) |e| {
                if (!std.mem.eql(u8, e.from, g.nodes[ni].id)) continue;
                const t_idx = for (g.nodes, 0..) |nd, i| {
                    if (std.mem.eql(u8, nd.id, e.to)) break i;
                } else unreachable; // validate 已保证
                if (activated[t_idx] or executed[t_idx]) continue;
                if (try condition.eval(e.when, done_results.items)) {
                    activated[t_idx] = true;
                }
            }
        }

        frontier.clearRetainingCapacity();
        for (0..n) |i| {
            if (activated[i] and !executed[i]) try frontier.append(a, i);
        }
    }

    for (g.nodes, 0..) |nd, i| {
        if (!executed[i]) try skipped.append(a, nd.id);
    }

    return .{
        .steps = try steps.toOwnedSlice(a),
        .skipped = try skipped.toOwnedSlice(a),
        .path_prob = path_prob,
    };
}
```

函数级 import 若无则补（文件已有 `std`、`alloc`、`state`、`schema`、`result`、`err`、`factory`、`engine`、`graph_mod`、`condition`、`trajectory`）：`model_encoder`/`model_head`/`model_logits` 只在测试用，放文件尾测试区即可。

- [ ] **Step 4: 跑全量测试确认绿**

Run: `zig build test > /tmp/prefetch-green.log 2>&1; echo "exit=$?"; tail -5 /tmp/prefetch-green.log`
Expected: exit 0（含既有 5 个 execute 测试 + 2 个新测试）。
再跑 conformance：Run: `zig build test-conformance > /tmp/conf.log 2>&1; echo "exit=$?"; tail -3 /tmp/conf.log`
Expected: exit 0，`conformance: 10 pass, 0 fail`。

- [ ] **Step 5: Commit**

```bash
git add src/graph/executor.zig
git commit -m "feat(graph): executor full-prefetch — single forward for whole graph, enabling bundled ONNX heads on multi-wave conditional graphs"
```

---

### Task 2: e2e 条件图 + README + 合并

**Files:**
- Modify: `README.md`（第 41 行 frontier 描述；第 80–82 行删除多 wave 限制段）

**Interfaces:**
- Consumes: Task 1 的 executor（多 wave + bundled 不再 400）。
- Produces: e2e 证据（两 wave 条件图 200 + 3 步 trajectory）；README 与行为一致。

- [ ] **Step 1: e2e——Laya ONNX + 两 wave 条件图**

```bash
zig build -Donnx=true -Donnx_lib_dir=$PWD/export/laya/lib > /tmp/e2e-build.log 2>&1 && echo build-ok
./zig-out/bin/zjev-serve --model export/laya/out/laya.onnx \
    --ort-extensions export/laya/lib/libortextensions.dylib --sessions 2 --port 18080 \
    > /tmp/e2e-serve.log 2>&1 &
sleep 2
curl -s -X POST localhost:18080/v1/execute -H 'content-type: application/json' -d '{
  "state": {"text": "my bill looks wrong"},
  "decisions": [
    {"id":"escalate-noul","type":"noul","abstain":false},
    {"id":"topic-choice","type":"choice","options":["billing","bug","other"]},
    {"id":"urgency-score","type":"score","scale":{"labels":["low","medium","high"]}}
  ],
  "graph": {
    "nodes": [
      {"id":"esc","decision":"escalate-noul"},
      {"id":"topic","decision":"topic-choice"},
      {"id":"urg","decision":"urgency-score"}
    ],
    "edges": [{"from":"esc","to":"topic","when":"flag == true"}]
  }
}' | tee /tmp/e2e-resp.json | head -40
kill %1
```

Expected: HTTP 200，JSON 含 `trajectory`（3 步：wave1 = esc, urg；wave2 = topic）、`path_prob`；无 `BadModelIO`。（节点序必须保持 escalate→topic→urgency，匹配导出契约的主序。）

- [ ] **Step 2: 更新 README**

第 41 行附近改为：

```markdown
并做静态类型检查；运行期一次性前向算全图节点，再沿 DAG frontier 模拟激活波
（整图共享一次 encoder 前向，bundled ONNX 头同样支持多 wave 条件图），
```

删除第 80–82 行限制段（「图含条件边（多 wave）时 ONNX 头暂不支持……见 …laya-onnx-export-design.md」），替换为：

```markdown
多 wave 条件图自 executor 全量预取起同样支持（单次前向算全图，再模拟激活波），
见 `docs/superpowers/specs/2026-09-24-executor-prefetch-design.md`。
```

- [ ] **Step 3: 全量验证 + commit + 合并**

```bash
zig build test > /tmp/final-test.log 2>&1; echo "test exit=$?"
zig build test-conformance > /tmp/final-conf.log 2>&1; echo "conf exit=$?"
git add README.md
git commit -m "docs(readme): multi-wave conditional graphs work with bundled ONNX heads via executor prefetch"
git checkout main && git merge --no-ff feat/executor-prefetch -m "merge: executor full-prefetch — bundled ONNX heads on multi-wave graphs" && git branch -d feat/executor-prefetch
```

Expected: 两个 exit 均为 0；merge commit 落在 main。
