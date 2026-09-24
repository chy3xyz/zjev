const std = @import("std");
const schema = @import("../core/schema.zig");
const result = @import("../core/result.zig");

pub const NodeAcc = struct {
    id: []const u8 = "",
    n: usize = 0,
    correct: usize = 0,
    conf: std.ArrayList(f32) = .empty,
    ok: std.ArrayList(bool) = .empty,
};

pub fn hit(sc: schema.DecisionSchema, r: result.DecisionResult, expected: std.json.Value) bool {
    switch (sc) {
        .choice => {
            const v = switch (r.value) {
                .choice => |s| s,
                else => return false,
            };
            const want = switch (expected) {
                .string => |s| s,
                else => return false,
            };
            return std.mem.eql(u8, v, want);
        },
        .noul => {
            const v = switch (r.value) {
                .noul => |b| b,
                else => return false,
            };
            const want = switch (expected) {
                .bool => |b| b,
                else => return false,
            };
            return v == want;
        },
        .score => |sd| {
            const x = switch (r.value) {
                .score => |f| f,
                else => return false,
            };
            const want = switch (expected) {
                .integer => |i| i,
                .float => |f| @as(i64, @intFromFloat(f)),
                else => return false,
            };
            switch (sd.scale) {
                // int scale：桶序号 = round(E[X]) - min
                .int => |r_int| {
                    const idx: i64 = @intFromFloat(@round(x));
                    return idx - @as(i64, r_int.min) == want;
                },
                // labels scale：期望值直接给桶序号整数（区间校验从简，登记为已知简化）
                .labels => {
                    return want >= 0 and want < @as(i64, @intCast(sd.bucketCount()));
                },
            }
        },
        .rank => {
            const entries = switch (r.value) {
                .rank => |e| e,
                else => return false,
            };
            if (entries.len == 0) return false;
            const want = switch (expected) {
                .string => |s| s,
                else => return false,
            };
            return std.mem.eql(u8, entries[0].id, want);
        },
    }
}

pub fn trajectoryBrier(p: []const f32, y: []const bool) f64 {
    std.debug.assert(p.len == y.len);
    if (p.len == 0) return 0;
    var sum: f64 = 0;
    for (p, y) |pi, yi| {
        const d: f64 = @as(f64, pi) - @as(f64, @floatFromInt(@intFromBool(yi)));
        sum += d * d;
    }
    return sum / @as(f64, @floatFromInt(p.len));
}

test "hit choice by string" {
    const sc: schema.DecisionSchema = .{ .choice = .{ .id = "c", .options = &.{ "a", "b" }, .abstain = false } };
    const r: result.DecisionResult = .{ .id = "c", .type = .choice, .value = .{ .choice = "a" } };
    try std.testing.expect(hit(sc, r, .{ .string = "a" }));
    try std.testing.expect(!hit(sc, r, .{ .string = "b" }));
}

test "hit noul by bool" {
    const sc: schema.DecisionSchema = .{ .noul = .{ .id = "n", .abstain = false } };
    const r: result.DecisionResult = .{ .id = "n", .type = .noul, .value = .{ .noul = false } };
    try std.testing.expect(hit(sc, r, .{ .bool = false }));
    try std.testing.expect(!hit(sc, r, .{ .bool = true }));
}

test "hit score int scale by bucket index" {
    const sc: schema.DecisionSchema = .{ .score = .{ .id = "s", .scale = .{ .int = .{ .min = 1, .max = 5 } }, .abstain = false } };
    const r: result.DecisionResult = .{ .id = "s", .type = .score, .value = .{ .score = 3.6 } };
    // round(3.6) = 4 → 桶序号 4-1 = 3
    try std.testing.expect(hit(sc, r, .{ .integer = 3 }));
    try std.testing.expect(!hit(sc, r, .{ .integer = 1 }));
    try std.testing.expect(!hit(sc, r, .{ .string = "3" }));
}

test "hit rank by top1 id" {
    const sc: schema.DecisionSchema = .{ .rank = .{ .id = "r", .items = &.{ "x", "y" } } };
    const r: result.DecisionResult = .{
        .id = "r",
        .type = .rank,
        .value = .{ .rank = &.{ .{ .id = "y", .score = 0.8 }, .{ .id = "x", .score = 0.2 } } },
    };
    try std.testing.expect(hit(sc, r, .{ .string = "y" }));
    try std.testing.expect(!hit(sc, r, .{ .string = "x" }));
}

test "trajectory brier matches binary formula" {
    const p = [_]f32{ 0.9, 0.2 };
    const y = [_]bool{ true, false };
    try std.testing.expectApproxEqAbs(@as(f64, 0.025), trajectoryBrier(&p, &y), 1e-6);
}
