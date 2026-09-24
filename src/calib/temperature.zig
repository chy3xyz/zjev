const std = @import("std");
const softmax = @import("softmax.zig");

pub fn nllOf(logits: []const f32, temperature: f32, label: usize) f64 {
    var buf: [300]f32 = undefined;
    const probs = buf[0..logits.len];
    softmax.apply(logits, temperature, probs) catch return std.math.inf(f64);
    const p = probs[label];
    if (p <= 0) return std.math.inf(f64);
    return -@log(@as(f64, p));
}

pub fn fit(zs: []const []const f32, labels: []const usize) f32 {
    std.debug.assert(zs.len == labels.len);
    var lo: f32 = 0.05;
    var hi: f32 = 20.0;
    const gr: f32 = 0.6180339887498949;
    var c = hi - gr * (hi - lo);
    var d = lo + gr * (hi - lo);
    var fc = totalNll(zs, labels, c);
    var fd = totalNll(zs, labels, d);
    for (0..60) |_| {
        if (fc < fd) {
            hi = d;
            d = c;
            fd = fc;
            c = hi - gr * (hi - lo);
            fc = totalNll(zs, labels, c);
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + gr * (hi - lo);
            fd = totalNll(zs, labels, d);
        }
    }
    return (lo + hi) / 2;
}

fn totalNll(zs: []const []const f32, labels: []const usize, temperature: f32) f64 {
    var sum: f64 = 0;
    for (zs, labels) |z, li| sum += nllOf(z, temperature, li);
    return sum;
}

test "fit sharpens when labels match argmax" {
    const base = [_]f32{ 2.0, 0.5, -1.0 };
    var zs = [_][]const f32{ &base, &base, &base, &base };
    const labels = [_]usize{ 0, 0, 0, 0 };
    const t = fit(&zs, &labels);
    try std.testing.expect(t < 1.0);
}

test "fit flattens when labels disagree with argmax" {
    const base = [_]f32{ 2.0, 0.5, -1.0 };
    var zs = [_][]const f32{ &base, &base, &base, &base };
    const labels = [_]usize{ 2, 2, 2, 2 };
    const t = fit(&zs, &labels);
    try std.testing.expect(t > 1.0);
}
