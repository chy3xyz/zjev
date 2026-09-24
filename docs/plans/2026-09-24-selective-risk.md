# Selective Risk 补缺实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `stats.zig` 实现 selective risk（spec §2 语义），接入 bench/traj 输出，并让 fit 填充 profile 的 ece/brier/selective_risk@4 键，收口 RFC §6.1/§6.2/§11。

**Architecture:** 单一纯函数 `selectiveRisk`（排序在副本上进行，不改动输入），三个工具各自调用；Profile 用 Zig 引号标识符字段直接序列化为 `selective_risk@0.9` 风格 JSON 键。

**Tech Stack:** Zig 0.17 nightly（同前）；无新依赖。

**Spec:** `docs/specs/2026-09-24-selective-risk.md`（§3 交付表）。实现期修正：fit 除 selective_risk 四键外同时填充 ece/brier（Profile 结构早已支持但 fit 从未填，借本次收口；nll 无公开求值函数，保持 null）。

## Global Constraints

- 工具链 `zig 0.17.0-dev.2151+2ec5523d5`；显式 allocator；unmanaged 容器；测试红绿验证必须重定向查退出码（管道掩盖退出码的坑）。
- coverage 档位固定 {0.5, 0.7, 0.9, 0.95}（工具层常量；函数接受 coverages 参数）。
- `k = ceil(c·n)`；并列 conf 按下标升序（确定性）；risk = 1 − mean(ok[0..k])；threshold = conf_sorted[k−1]。
- 每 Task 一个 commit；分支 `feat/selective-risk`；收尾合并 main。

---

### Task 1: stats.zig——selectiveRisk 纯函数

**Files:**
- Modify: `src/calib/stats.zig`（文件尾追加）

**Interfaces:**
- Produces:
  - `pub const RiskPoint = struct { coverage: f64, keep: usize, n: usize, risk: f64, threshold: f32 };`
  - `pub fn selectiveRisk(a: alloc.Allocator, conf: []const f32, ok: []const bool, coverages: []const f64) error{OutOfMemory}![]RiskPoint`
  - `pub const default_coverages = [_]f64{ 0.5, 0.7, 0.9, 0.95 };`（工具层引用，避免各处硬编码）

- [ ] **Step 1: 写失败测试**（追加到 stats.zig）

```zig
const alloc = @import("../core/alloc.zig");

test "selective risk four coverage tiers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const conf = [_]f32{ 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05 };
    const ok = [_]bool{ true, true, true, true, false, true, false, false, true, false };
    const pts = try selectiveRisk(a, &conf, &ok, &default_coverages);
    try std.testing.expectEqual(@as(usize, 4), pts.len);
    // k = ceil(c*10): 5, 7, 9, 10
    try std.testing.expectEqual(@as(usize, 5), pts[0].keep);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), pts[0].risk, 1e-9); // 前5: TTTTF
    try std.testing.expectEqual(@as(f32, 0.5), pts[0].threshold);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 7.0), pts[1].risk, 1e-9); // 前7: TTTTFTF
    try std.testing.expectEqual(@as(f32, 0.3), pts[1].threshold);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), pts[2].risk, 1e-9); // 前9
    try std.testing.expectEqual(@as(f32, 0.1), pts[2].threshold);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), pts[3].risk, 1e-9); // 前10: 6T
    try std.testing.expectEqual(@as(f32, 0.05), pts[3].threshold);
}

test "selective risk tie breaks by original index" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const conf = [_]f32{ 0.5, 0.5, 0.9 };
    const ok = [_]bool{ true, false, true };
    const cov = [_]f64{0.5};
    const pts = try selectiveRisk(a, &conf, &ok, &cov);
    try std.testing.expectEqual(@as(usize, 2), pts[0].keep); // ceil(1.5)
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), pts[0].risk, 1e-9); // 0.9→T, 0.5(idx0)→T
    try std.testing.expectEqual(@as(f32, 0.5), pts[0].threshold);
}

test "selective risk single sample" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const conf = [_]f32{0.7};
    const ok = [_]bool{false};
    const pts = try selectiveRisk(a, &conf, &ok, &default_coverages);
    try std.testing.expectEqual(@as(usize, 1), pts[0].keep);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), pts[0].risk, 1e-9);
    try std.testing.expectEqual(@as(f32, 0.7), pts[0].threshold);
}
```

- [ ] **Step 2: 确认红**（`zig build test` 重定向查 exit=1，`selectiveRisk` 未声明）

- [ ] **Step 3: 实现**（追加到 stats.zig）

```zig
pub const RiskPoint = struct {
    coverage: f64,
    keep: usize,
    n: usize,
    risk: f64,
    threshold: f32,
};

pub const default_coverages = [_]f64{ 0.5, 0.7, 0.9, 0.95 };

pub fn selectiveRisk(a: alloc.Allocator, conf: []const f32, ok: []const bool, coverages: []const f64) error{OutOfMemory}![]RiskPoint {
    std.debug.assert(conf.len == ok.len);
    const n = conf.len;
    var idx: std.ArrayList(usize) = .empty;
    defer idx.deinit(a);
    for (0..n) |i| try idx.append(a, i);
    const Ctx = struct { conf: []const f32 };
    const ctx: Ctx = .{ .conf = conf };
    std.mem.sort(usize, idx.items, ctx, struct {
        fn less(c: Ctx, x: usize, y: usize) bool {
            if (c.conf[x] != c.conf[y]) return c.conf[x] > c.conf[y];
            return x < y;
        }
    }.less);
    const pts = try a.alloc(RiskPoint, coverages.len);
    for (coverages, 0..) |c, i| {
        const k: usize = if (n == 0) 0 else @intFromFloat(@ceil(c * @as(f64, @floatFromInt(n))));
        const kk = @min(k, n);
        var correct: usize = 0;
        for (idx.items[0..kk]) |j| {
            if (ok[j]) correct += 1;
        }
        pts[i] = .{
            .coverage = c,
            .keep = kk,
            .n = n,
            .risk = if (kk == 0) 0 else 1.0 - @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(kk)),
            .threshold = if (kk == 0) 0 else conf[idx.items[kk - 1]],
        };
    }
    return pts;
}
```

- [ ] **Step 4: 确认绿**（exit=0）

- [ ] **Step 5: Commit** `feat(calib): selective risk @ coverage（ceil 保留 + 下标确定性 tie-break）`

---

### Task 2: Profile 增 selective_risk@ 键 + fit 填充 metrics

**Files:**
- Modify: `src/calib/profile.zig`（Profile 追加 4 个引号标识符可选字段）
- Modify: `tools/fit.zig`（拟合后计算并填充 ece/brier/sr）
- Test: profile.zig 追加 JSON 往返测试

**Interfaces:**
- Consumes: Task 1 的 `selectiveRisk/default_coverages`、`zjev.ece.compute`、`zjev.brier.score`
- Produces: `Profile.@"selective_risk@0.5"/@0.7/@0.9/@0.95: ?f32 = null`（序列化键名即 `selective_risk@0.5` 等）

- [ ] **Step 1: 写失败测试**（追加 profile.zig）

```zig
test "profile serializes selective risk keys" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const p: Profile = .{
        .model = "m",
        .task = "choice",
        .num_options = 2,
        .temperature = 1.2,
        .ece = 0.1,
        .@"selective_risk@0.9" = 0.05,
    };
    const s = try std.json.Stringify.valueAlloc(a, p, .{});
    try std.testing.expect(std.mem.indexOf(u8, s, "\"selective_risk@0.9\":0.05") != null);
    const back = try std.json.parseFromSlice(Profile, a, s, .{});
    try std.testing.expectEqual(@as(?f32, 0.05), back.value.@"selective_risk@0.9");
    try std.testing.expectEqual(@as(?f32, null), back.value.@"selective_risk@0.5");
}
```

- [ ] **Step 2: 确认红**（引号标识符字段不存在 → 编译错）

- [ ] **Step 3: 改 Profile 字段**

```zig
pub const Profile = struct {
    model: []const u8,
    task: []const u8,
    num_options: u16,
    domain: []const u8 = "general",
    temperature: f32,
    ece: ?f32 = null,
    brier: ?f32 = null,
    nll: ?f32 = null,
    @"selective_risk@0.5": ?f32 = null,
    @"selective_risk@0.7": ?f32 = null,
    @"selective_risk@0.9": ?f32 = null,
    @"selective_risk@0.95": ?f32 = null,
    fitted_at: ?[]const u8 = null,
};
```

（loadDir 已用 `.ignore_unknown_fields = true`，旧文件无这些键兼容。）

- [ ] **Step 4: 改 fit.zig 填充**（`for (groups.items) |*g|` 循环内、`Profile{...}` 构造之前插入计算）

```zig
        // 用拟合后的 T 计算校准指标（spec §3；ece/brier 顺带收口 §6.2）
        const cal_conf = try a.alloc(f32, g.zs.items.len);
        const cal_ok = try a.alloc(bool, g.zs.items.len);
        const cal_probs: std.ArrayList([]const f32) = .empty;
        _ = cal_probs;
        var probs_rows: std.ArrayList([]const f32) = .empty;
        defer probs_rows.deinit(a);
        var label_idx: std.ArrayList(usize) = .empty;
        for (g.zs.items, g.labels.items, 0..) |z, li, ri| {
            const row = try a.alloc(f32, z.len);
            zjev.softmax.apply(z, t, row) catch {
                cal_conf[ri] = 0;
                cal_ok[ri] = false;
                continue;
            };
            try probs_rows.append(a, row);
            try label_idx.append(a, li);
            cal_conf[ri] = blk: {
                var m: f32 = 0;
                for (row) |p| m = @max(m, p);
                break :blk m;
            };
            cal_ok[ri] = zjev.stats.argmax(row) == li;
        }
        const ece_r = zjev.ece.compute(cal_conf[0..probs_rows.items.len], cal_ok[0..probs_rows.items.len], 15);
        const brier_v = zjev.brier.score(probs_rows.items, label_idx.items);
        const sr = try zjev.stats.selectiveRisk(a, cal_conf[0..probs_rows.items.len], cal_ok[0..probs_rows.items.len], &zjev.stats.default_coverages);
```

并将 Profile 构造改为：

```zig
        const profile = zjev.profile.Profile{
            .model = model_name,
            .task = @tagName(g.task),
            .num_options = g.num,
            .domain = domain,
            .temperature = t,
            .ece = @floatCast(ece_r.ece),
            .brier = @floatCast(brier_v),
            .@"selective_risk@0.5" = @floatCast(sr[0].risk),
            .@"selective_risk@0.7" = @floatCast(sr[1].risk),
            .@"selective_risk@0.9" = @floatCast(sr[2].risk),
            .@"selective_risk@0.95" = @floatCast(sr[3].risk),
            .fitted_at = "zjev-fit",
        };
```

注意：probs_rows 与 g.zs 对齐可能被 softmax 失败打散——实现时将 cal_conf/cal_ok 也改 append 到 ArrayList（与 probs_rows 同推退），保持三数组等长；`num = switch (s)` 处 noul 现有 num=2 口径不变。`zjev.softmax.apply` 签名（V0.1）：`apply(z: []const f32, temp: f32, out: []f32) !void`。

- [ ] **Step 5: 验证** `zig build test` exit=0；实跑 `zig-out/bin/zjev-fit --dataset datasets/calibration_sample.jsonl --out /tmp/fit_out` 并 `cat /tmp/fit_out/*.json` 确认五键存在。

- [ ] **Step 6: Commit** `feat(fit): profile 填充 ece/brier/selective_risk@4（§6.2 收口）`

---

### Task 3: bench 每组输出 selective_risk

**Files:**
- Modify: `tools/bench.zig`（组输出处）

**Interfaces:**
- Consumes: Task 1 `selectiveRisk/default_coverages`；既有 `ac.conf/ac.ok`
- Produces: 每组 JSON 追加 `"selective_risk":[{"coverage":0.5,"keep":k,"n":N,"risk":r,"threshold":t},...]`

- [ ] **Step 1: 修改输出**（`try w.print("{{\"task\":...` 那一行之后、`if (ac.abstain_total > 0)` 之前插入）

```zig
        const sr = zjev.stats.selectiveRisk(ac.conf.items, ac.ok.items, &zjev.stats.default_coverages) catch null;
        if (sr) |pts| {
            try w.writeAll(",\"selective_risk\":[");
            for (pts, 0..) |pt, pi| {
                if (pi > 0) try w.writeByte(',');
                try w.print("{{\"coverage\":{d:.2},\"keep\":{d},\"n\":{d},\"risk\":{d:.6},\"threshold\":{d:.6}}}", .{ pt.coverage, pt.keep, pt.n, pt.risk, pt.threshold });
            }
            try w.writeByte(']');
        }
```

（无独立测试基建的工具：以实跑验证代替——见 Step 2。）

- [ ] **Step 2: 实跑验证** `zig-out/bin/zjev-bench --dataset datasets/calibration_sample.jsonl --mock-mode sequence` 输出含 selective_risk 块；抽一组手算 keep/risk 与输出一致。

- [ ] **Step 3: Commit** `feat(bench): 每组输出 selective_risk 四档`

---

### Task 4: traj 轨迹级 selective_risk

**Files:**
- Modify: `tools/traj.zig`

**Interfaces:**
- Consumes: Task 1；既有 `p_list/y_list`
- Produces: 顶层 JSON 追加 `"selective_risk":[...]`（按 path_prob 排序口径，文档明示）

- [ ] **Step 1: 修改输出**（`"skipped_records":{d}` 之后、`"by_node"` 之前插入）

```zig
    const sr = zjev.stats.selectiveRisk(a, p_list.items, y_list.items, &zjev.stats.default_coverages) catch null;
```

并在 JSON 中 `,"selective_risk":[...]` 紧跟 skipped_records 之后输出（字段顺序：...,"skipped_records":S,"selective_risk":[...],"by_node":[...]）。

- [ ] **Step 2: 实跑验证**（5 条样本手算：k=3/4/5/5，y=[1,1,0,0,1]，全部并列 conf=0.947312 → 原序；risk 分别 1/3、0.5、0.4、0.4，threshold 均 0.947312）

- [ ] **Step 3: Commit** `feat(traj): 轨迹级 selective_risk（低势轨迹弃权口径）`

---

### Task 5: 文档 + 全量验证 + 合并

- [ ] **Step 1:** README 工具节两处输出说明补 selective_risk 行。
- [ ] **Step 2:** `zig build test`、`zig build test-conformance`、`zig build -Doptimize=ReleaseSafe` 全绿。
- [ ] **Step 3:** 合并 main（--no-ff）、删分支、main 上复跑测试。

---

## Self-Review 记录

- **Spec 覆盖**：§3 表四行 → Task 1/2/3/4；§4 测试 → Task 1 三例 + Task 2 往返 + Task 3/4 手算；§6 出口 → Task 5。fit 增补 ece/brier 已记为 spec §3 实现期修正。
- **占位符**：无；全部代码给出完整文本。
- **类型一致性**：`RiskPoint{coverage:f64,keep:usize,n:usize,risk:f64,threshold:f32}` 与 `default_coverages` 在 Task 1 定义、Task 2/3/4 引用一致；Profile 引号标识符字段在 Task 2 定义并同任务内序列化验证；traj 手算值与 spec §2 语义一致（并列保序）。风险：`std.json.Stringify` 对 `@"selective_risk@0.9"` 字段名是否原样输出（含 @ 与点号）——Task 2 Step 1 的往返测试即验证此点，若 Stringify 转义不符则改用 `std.json.Value` 手动构造 metrics 对象（回退方案，同任务内完成）。
