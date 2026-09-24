const std = @import("std");
const result = @import("../core/result.zig");

pub const Gate = struct {
    threshold: f32,
    action_above: []const u8,
    action_below: []const u8,
    action_abstain: []const u8,

    pub fn apply(g: Gate, r: result.DecisionResult) []const u8 {
        if (r.uncertainty.abstention) |ab| {
            if (ab >= g.threshold) return g.action_abstain;
        }
        return if (r.uncertainty.confidence >= g.threshold) g.action_above else g.action_below;
    }
};

pub fn validateThreshold(t: f32) bool {
    return t >= 0.0 and t <= 1.0;
}

fn res(conf: f32, abst: ?f32) result.DecisionResult {
    return .{
        .id = "d",
        .type = .noul,
        .value = .{ .noul = true },
        .uncertainty = .{ .confidence = conf, .abstention = abst },
    };
}

test "gate abstention takes priority" {
    const g: Gate = .{ .threshold = 0.5, .action_above = "go", .action_below = "stop", .action_abstain = "review" };
    try std.testing.expectEqualStrings("review", g.apply(res(0.9, 0.6)));
}

test "gate above threshold" {
    const g: Gate = .{ .threshold = 0.5, .action_above = "go", .action_below = "stop", .action_abstain = "review" };
    try std.testing.expectEqualStrings("go", g.apply(res(0.7, 0.1)));
}

test "gate below threshold" {
    const g: Gate = .{ .threshold = 0.5, .action_above = "go", .action_below = "stop", .action_abstain = "review" };
    try std.testing.expectEqualStrings("stop", g.apply(res(0.3, null)));
}

test "gate boundary is inclusive" {
    const g: Gate = .{ .threshold = 0.5, .action_above = "go", .action_below = "stop", .action_abstain = "review" };
    try std.testing.expectEqualStrings("go", g.apply(res(0.5, null)));
    try std.testing.expectEqualStrings("review", g.apply(res(0.9, 0.5)));
}

test "threshold validation" {
    try std.testing.expect(validateThreshold(0.5));
    try std.testing.expect(!validateThreshold(-0.1));
    try std.testing.expect(!validateThreshold(1.1));
}
