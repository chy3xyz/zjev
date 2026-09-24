const std = @import("std");

pub const Result = struct {
    ece: f64,
    mce: f64,
};

pub fn compute(confidence: []const f32, correct: []const bool, bin_count: usize) Result {
    std.debug.assert(confidence.len == correct.len);
    std.debug.assert(bin_count > 0 and bin_count <= 100);
    var acc: [100]f64 = @splat(0);
    var conf_sum: [100]f64 = @splat(0);
    var cnt: [100]u64 = @splat(0);
    for (confidence, correct) |c, ok| {
        const b = @min(bin_count - 1, @as(usize, @intFromFloat(c * @as(f32, @floatFromInt(bin_count)))));
        acc[b] += if (ok) 1 else 0;
        conf_sum[b] += c;
        cnt[b] += 1;
    }
    const n: f64 = @floatFromInt(confidence.len);
    var ece: f64 = 0;
    var mce: f64 = 0;
    for (0..bin_count) |b| {
        if (cnt[b] == 0) continue;
        const bin_acc = acc[b] / @as(f64, @floatFromInt(cnt[b]));
        const bin_conf = conf_sum[b] / @as(f64, @floatFromInt(cnt[b]));
        const gap = @abs(bin_acc - bin_conf);
        ece += @as(f64, @floatFromInt(cnt[b])) / n * gap;
        mce = @max(mce, gap);
    }
    return .{ .ece = ece, .mce = mce };
}

test "perfectly calibrated two points" {
    const conf = [_]f32{ 0.5, 0.5 };
    const correct = [_]bool{ true, false };
    const r = compute(&conf, &correct, 10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), r.ece, 1e-9);
}

test "overconfident wrong" {
    const conf = [_]f32{ 0.9, 0.9, 0.9, 0.9 };
    const correct = [_]bool{ false, false, false, false };
    const r = compute(&conf, &correct, 10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9), r.ece, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9), r.mce, 1e-6);
}
