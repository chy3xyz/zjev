const std = @import("std");
const alloc = @import("../core/alloc.zig");
const schema = @import("../core/schema.zig");
const result = @import("../core/result.zig");
const err = @import("../core/error.zig");
const logits_mod = @import("../model/logits.zig");

pub const Step = struct {
    node_id: []const u8,
    decision_id: []const u8,
    result: result.DecisionResult,
    action: ?[]const u8 = null,
};

pub const Trajectory = struct {
    steps: []Step,
    path_prob: f32,
};

pub fn massFor(a: alloc.Allocator, r: result.DecisionResult, sc: schema.DecisionSchema) err.GraphError!f32 {
    if (!std.mem.eql(u8, r.id, sc.id())) return error.UnknownDecision;
    return switch (sc) {
        .choice => |c| blk: {
            const probs = r.probabilities orelse return error.UnknownDecision;
            const taken = switch (r.value) {
                .choice => |v| v,
                else => return error.UnknownDecision,
            };
            var idx: usize = c.options.len; // 默认 abstain 槽
            for (c.options, 0..) |o, i| {
                if (std.mem.eql(u8, o, taken)) {
                    idx = i;
                    break;
                }
            }
            break :blk probs[idx];
        },
        .noul => switch (r.value) {
            .noul => |b| if (b) r.probability.? else 1.0 - r.probability.?,
            else => return error.UnknownDecision,
        },
        .score => blk: {
            const probs = r.probabilities orelse return error.UnknownDecision;
            const x = switch (r.value) {
                .score => |v| v,
                else => return error.UnknownDecision,
            };
            const view: logits_mod.View = .{ .data = probs, .schema = sc };
            const centers = view.bucketValues(a) catch return error.UnknownDecision;
            var best: usize = 0;
            var best_d: f32 = @floatCast(@abs(centers[0] - x));
            for (centers[1..], 1..) |cv, i| {
                const d: f32 = @floatCast(@abs(cv - x));
                if (d < best_d) {
                    best_d = d;
                    best = i;
                }
            }
            break :blk probs[best];
        },
        .rank => blk: {
            const probs = r.probabilities orelse return error.UnknownDecision;
            const top = switch (r.value) {
                .rank => |entries| if (entries.len == 0) return error.UnknownDecision else entries[0].id,
                else => return error.UnknownDecision,
            };
            var total: f32 = 0;
            for (probs) |p| total += p;
            if (total <= 0) return error.UnknownDecision;
            var idx: ?usize = null;
            for (r.labels orelse return error.UnknownDecision, 0..) |lb, i| {
                if (std.mem.eql(u8, lb, top)) idx = i;
            }
            break :blk probs[idx.?] / total;
        },
    };
}

test "massFor choice returns probability of taken option" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const sc: schema.DecisionSchema = .{ .choice = .{ .id = "c", .options = &.{ "a", "b" }, .abstain = false } };
    const r: result.DecisionResult = .{
        .id = "c",
        .type = .choice,
        .value = .{ .choice = "b" },
        .probabilities = &.{ 0.7, 0.3 },
        .labels = &.{ "a", "b" },
    };
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), try massFor(arena.allocator(), r, sc), 1e-6);
}

test "massFor choice abstain winner uses tail probability" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const sc: schema.DecisionSchema = .{ .choice = .{ .id = "c", .options = &.{"a"}, .abstain = true } };
    const r: result.DecisionResult = .{
        .id = "c",
        .type = .choice,
        .value = .{ .choice = "__abstain__" },
        .probabilities = &.{ 0.4, 0.6 },
        .labels = &.{"a"},
    };
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), try massFor(arena.allocator(), r, sc), 1e-6);
}

test "massFor noul false uses complement" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const sc: schema.DecisionSchema = .{ .noul = .{ .id = "n", .abstain = false } };
    const r: result.DecisionResult = .{
        .id = "n",
        .type = .noul,
        .value = .{ .noul = false },
        .probability = 0.3,
    };
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), try massFor(arena.allocator(), r, sc), 1e-6);
}

test "massFor score picks nearest bucket probability" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const sc: schema.DecisionSchema = .{ .score = .{ .id = "s", .scale = .{ .int = .{ .min = 1, .max = 5 } }, .abstain = false } };
    const r: result.DecisionResult = .{
        .id = "s",
        .type = .score,
        .value = .{ .score = 3.6 },
        .probabilities = &.{ 0.1, 0.2, 0.3, 0.3, 0.1 },
        .labels = &.{ "1", "2", "3", "4", "5" },
    };
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), try massFor(arena.allocator(), r, sc), 1e-6);
}

test "massFor rank normalizes top1 sigmoid" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const sc: schema.DecisionSchema = .{ .rank = .{ .id = "r", .items = &.{ "x", "y", "z" } } };
    const r: result.DecisionResult = .{
        .id = "r",
        .type = .rank,
        .value = .{ .rank = &.{.{ .id = "y", .score = 0.8 }} },
        .probabilities = &.{ 0.2, 0.8, 0.5 },
        .labels = &.{ "x", "y", "z" },
    };
    try std.testing.expectApproxEqAbs(@as(f32, 0.53333333), try massFor(arena.allocator(), r, sc), 1e-4);
}

test "massFor rejects decision id mismatch" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const sc: schema.DecisionSchema = .{ .noul = .{ .id = "n", .abstain = false } };
    const r: result.DecisionResult = .{ .id = "other", .type = .noul, .value = .{ .noul = true }, .probability = 0.9 };
    try std.testing.expectError(error.UnknownDecision, massFor(arena.allocator(), r, sc));
}
