const std = @import("std");

pub const Error = error{BadTemperature};

pub fn apply(logits: []const f32, temperature: f32, out: []f32) Error!void {
    if (!(temperature > 0) or std.math.isNan(temperature)) return error.BadTemperature;
    std.debug.assert(out.len == logits.len);
    var max: f32 = -std.math.inf(f32);
    for (logits) |z| max = @max(max, z);
    var sum: f32 = 0;
    for (logits, out) |z, *o| {
        const e = @exp((z - max) / temperature);
        o.* = e;
        sum += e;
    }
    const inv = 1.0 / sum;
    for (out) |*o| o.* *= inv;
}

test "zero logits uniform" {
    const z = [_]f32{ 0, 0, 0, 0 };
    var out: [4]f32 = undefined;
    try apply(&z, 1.0, &out);
    for (out) |p| try std.testing.expectApproxEqAbs(@as(f32, 0.25), p, 1e-6);
}

test "known vector" {
    const z = [_]f32{ 1, 2, 3 };
    var out: [3]f32 = undefined;
    try apply(&z, 1.0, &out);
    const e1 = @exp(@as(f32, 1) - 3);
    const e2 = @exp(@as(f32, 2) - 3);
    const sum = e1 + e2 + 1.0;
    try std.testing.expectApproxEqAbs(e1 / sum, out[0], 1e-6);
    try std.testing.expectApproxEqAbs(1.0 / sum, out[2], 1e-6);
}

test "temperature sharpens" {
    const z = [_]f32{ 1, 2 };
    var cold: [2]f32 = undefined;
    var hot: [2]f32 = undefined;
    try apply(&z, 0.1, &cold);
    try apply(&z, 10.0, &hot);
    try std.testing.expect(cold[1] > 0.99);
    try std.testing.expect(hot[1] < 0.6);
}

test "bad temperature rejected" {
    var out: [2]f32 = undefined;
    const z = [_]f32{ 1, 2 };
    try std.testing.expectError(error.BadTemperature, apply(&z, 0.0, &out));
    try std.testing.expectError(error.BadTemperature, apply(&z, -1.0, &out));
    try std.testing.expectError(error.BadTemperature, apply(&z, std.math.nan(f32), &out));
}

test "extreme logits no overflow" {
    const z = [_]f32{ 1e30, -1e30, 0 };
    var out: [3]f32 = undefined;
    try apply(&z, 1.0, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[1], 1e-6);
}
