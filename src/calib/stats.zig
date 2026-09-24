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
