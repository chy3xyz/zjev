const std = @import("std");

pub fn score(probs: []const []const f32, label_index: []const usize) f64 {
    std.debug.assert(probs.len == label_index.len);
    if (probs.len == 0) return 0;
    var sum: f64 = 0;
    for (probs, label_index) |p, li| {
        for (p, 0..) |p_k, k| {
            const y: f32 = if (k == li) 1 else 0;
            const d = p_k - y;
            sum += @as(f64, d) * d;
        }
    }
    return sum / @as(f64, @floatFromInt(probs.len));
}

test "perfect prediction zero brier" {
    const probs = [_][]const f32{ &.{ 0.0, 1.0, 0.0 }, &.{ 1.0, 0.0, 0.0 } };
    const labels = [_]usize{ 1, 0 };
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), score(&probs, &labels), 1e-9);
}

test "known brier value" {
    const probs = [_][]const f32{&.{ 0.7, 0.3 }};
    const labels = [_]usize{1};
    try std.testing.expectApproxEqAbs(@as(f64, 0.98), score(&probs, &labels), 1e-6);
}
