const std = @import("std");

pub fn entropy(probs: []const f32) f32 {
    var h: f32 = 0;
    for (probs) |p| {
        if (p > 0) h -= p * @log(p);
    }
    return h;
}

pub fn expectation(values: []const f32, probs: []const f32) f32 {
    std.debug.assert(values.len == probs.len);
    var sum: f32 = 0;
    for (values, probs) |v, p| sum += v * p;
    return sum;
}

pub fn variance(values: []const f32, probs: []const f32) f32 {
    std.debug.assert(values.len == probs.len);
    const mean = expectation(values, probs);
    var second: f32 = 0;
    for (values, probs) |v, p| second += v * v * p;
    return @max(0, second - mean * mean);
}

pub fn argmax(items: []const f32) usize {
    std.debug.assert(items.len > 0);
    var best: usize = 0;
    for (items[1..], 1..) |v, i| {
        if (v > items[best]) best = i;
    }
    return best;
}

pub fn sigmoid(z: f32) f32 {
    return 1.0 / (1.0 + @exp(-z));
}

test "uniform entropy is ln n" {
    const p = [_]f32{ 0.25, 0.25, 0.25, 0.25 };
    try std.testing.expectApproxEqAbs(@log(@as(f32, 4)), entropy(&p), 1e-5);
}

test "zero prob contributes nothing" {
    const p = [_]f32{ 1.0, 0.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 0), entropy(&p), 1e-6);
}

test "expectation and variance of known distribution" {
    const values = [_]f32{ 1, 2, 3, 4, 5 };
    const probs = [_]f32{ 0.02, 0.11, 0.24, 0.48, 0.15 };
    const mean = expectation(&values, &probs);
    try std.testing.expectApproxEqAbs(@as(f32, 3.63), mean, 1e-4);
    const v = variance(&values, &probs);
    const second: f32 = blk: {
        var s: f32 = 0;
        for (values, probs) |x, p| s += x * x * p;
        break :blk s;
    };
    try std.testing.expectApproxEqAbs(second - mean * mean, v, 1e-4);
    try std.testing.expect(v >= 0);
}

test "argmax" {
    const p = [_]f32{ 0.1, 0.7, 0.2 };
    try std.testing.expectEqual(1, argmax(&p));
}

test "sigmoid endpoints" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), sigmoid(0), 1e-6);
    try std.testing.expect(sigmoid(10) > 0.999);
    try std.testing.expect(sigmoid(-10) < 0.001);
}

const alloc = @import("../core/alloc.zig");

test "selective risk four coverage tiers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const conf = [_]f32{ 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05 };
    const ok = [_]bool{ true, true, true, true, false, true, false, false, true, false };
    const pts = try selectiveRisk(a, &conf, &ok, &default_coverages);
    try std.testing.expectEqual(@as(usize, 4), pts.len);
    try std.testing.expectEqual(@as(usize, 5), pts[0].keep);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), pts[0].risk, 1e-9);
    try std.testing.expectEqual(@as(f32, 0.5), pts[0].threshold);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 7.0), pts[1].risk, 1e-9);
    try std.testing.expectEqual(@as(f32, 0.3), pts[1].threshold);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), pts[2].risk, 1e-9);
    try std.testing.expectEqual(@as(f32, 0.1), pts[2].threshold);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), pts[3].risk, 1e-9);
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
    try std.testing.expectEqual(@as(usize, 2), pts[0].keep);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), pts[0].risk, 1e-9);
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
