const std = @import("std");
const alloc = @import("../core/alloc.zig");
const schema = @import("../core/schema.zig");

pub fn logitCount(s: schema.DecisionSchema) usize {
    return switch (s) {
        .choice => |c| c.options.len + @intFromBool(c.abstain),
        .noul => |n| 2 + @as(usize, @intFromBool(n.abstain)),
        .score => |sc| sc.bucketCount() + @intFromBool(sc.abstain),
        .rank => |r| r.items.len,
    };
}

pub const View = struct {
    data: []const f32,
    schema: schema.DecisionSchema,

    pub fn regular(self: View) []const f32 {
        const n = switch (self.schema) {
            .choice => |c| c.options.len,
            .noul => 2,
            .score => |sc| sc.bucketCount(),
            .rank => |r| r.items.len,
        };
        return self.data[0..n];
    }

    pub fn abstain(self: View) ?f32 {
        const has = switch (self.schema) {
            .choice => |c| c.abstain,
            .noul => |n| n.abstain,
            .score => |sc| sc.abstain,
            .rank => false,
        };
        if (!has) return null;
        return self.data[self.data.len - 1];
    }

    pub fn bucketValues(self: View, a: alloc.Allocator) ![]f32 {
        const sc = switch (self.schema) {
            .score => |x| x,
            else => unreachable,
        };
        const n = sc.bucketCount();
        const vals = try a.alloc(f32, n);
        switch (sc.scale) {
            .int => |r| {
                for (vals, 0..) |*v, i| v.* = @floatFromInt(@as(i32, r.min) + @as(i32, @intCast(i)));
            },
            .labels => {
                for (vals, 0..) |*v, i| v.* = @floatFromInt(i + 1);
            },
        }
        return vals;
    }
};

test "logit counts" {
    const c: schema.DecisionSchema = .{ .choice = .{ .id = "c", .options = &.{ "a", "b" }, .abstain = true } };
    try std.testing.expectEqual(3, logitCount(c));
    const n: schema.DecisionSchema = .{ .noul = .{ .id = "n" } };
    try std.testing.expectEqual(3, logitCount(n));
    const s: schema.DecisionSchema = .{ .score = .{ .id = "s", .scale = .{ .int = .{ .min = 1, .max = 5 } }, .abstain = true } };
    try std.testing.expectEqual(6, logitCount(s));
    const r: schema.DecisionSchema = .{ .rank = .{ .id = "r", .items = &.{ "x", "y" } } };
    try std.testing.expectEqual(2, logitCount(r));
}

test "view splits abstain" {
    const c: schema.DecisionSchema = .{ .choice = .{ .id = "c", .options = &.{ "a", "b" }, .abstain = true } };
    const v: View = .{ .data = &.{ 1, 2, 3 }, .schema = c };
    try std.testing.expectEqualSlices(f32, &.{ 1, 2 }, v.regular());
    try std.testing.expectEqual(@as(?f32, 3), v.abstain());
}

test "bucket values" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const s: schema.DecisionSchema = .{ .score = .{ .id = "s", .scale = .{ .int = .{ .min = 3, .max = 5 } } } };
    const v: View = .{ .data = &.{ 0, 0, 0 }, .schema = s };
    const vals = try v.bucketValues(arena.allocator());
    try std.testing.expectEqualSlices(f32, &.{ 3, 4, 5 }, vals);
}
