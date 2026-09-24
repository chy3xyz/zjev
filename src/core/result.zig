const schema = @import("schema.zig");

test "defaults" {
    const r: DecisionResult = .{
        .id = "d",
        .type = .noul,
        .value = .{ .noul = true },
    };
    try std.testing.expectEqual(@as(?f32, null), r.probability);
    try std.testing.expectEqual(@as(u64, 0), r.latency_us);
    try std.testing.expectEqual(@as(?f32, null), r.uncertainty.abstention);
}

const std = @import("std");

pub const Uncertainty = struct {
    entropy: ?f32 = null,
    variance: ?f32 = null,
    confidence: f32 = 0,
    abstention: ?f32 = null,
};

pub const RankEntry = struct {
    id: []const u8,
    score: f32,
};

pub const Value = union(enum) {
    choice: []const u8,
    noul: bool,
    score: f32,
    rank: []const RankEntry,
};

pub const DecisionResult = struct {
    id: []const u8,
    type: schema.DecisionType,
    value: Value,
    probability: ?f32 = null,
    probabilities: ?[]const f32 = null,
    labels: ?[]const []const u8 = null,
    uncertainty: Uncertainty = .{},
    latency_us: u64 = 0,
};
