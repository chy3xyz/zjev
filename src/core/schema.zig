const std = @import("std");
const alloc = @import("alloc.zig");
const err = @import("error.zig");

test "reserved abstain id rejected" {
    var a = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer a.deinit();
    const s: DecisionSchema = .{ .choice = .{
        .id = "d",
        .options = &.{ "a", "__abstain__" },
    } };
    try std.testing.expectError(error.ReservedId, s.validateOne(a.allocator()));
}

test "duplicate options rejected" {
    var a = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer a.deinit();
    const s: DecisionSchema = .{ .choice = .{ .id = "d", .options = &.{ "a", "a" } } };
    try std.testing.expectError(error.DuplicateId, s.validateOne(a.allocator()));
}

test "duplicate decision ids rejected" {
    var a = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer a.deinit();
    const set = [_]DecisionSchema{
        .{ .noul = .{ .id = "x" } },
        .{ .noul = .{ .id = "x" } },
    };
    try std.testing.expectError(error.DuplicateId, validateSet(&set, a.allocator()));
}

test "empty decisions rejected" {
    var a = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer a.deinit();
    try std.testing.expectError(error.NoDecisions, validateSet(&.{}, a.allocator()));
}

test "over 64 decisions rejected" {
    var a = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer a.deinit();
    var set: [65]DecisionSchema = undefined;
    for (&set, 0..) |*s, i| {
        var buf: [8]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "d{d}", .{i});
        s.* = .{ .noul = .{ .id = try a.allocator().dupe(u8, name) } };
    }
    try std.testing.expectError(error.TooManyDecisions, validateSet(&set, a.allocator()));
}

test "bad score range rejected" {
    var a = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer a.deinit();
    const s: DecisionSchema = .{ .score = .{
        .id = "s",
        .scale = .{ .int = .{ .min = 5, .max = 2 } },
    } };
    try std.testing.expectError(error.BadRange, s.validateOne(a.allocator()));
}

test "rank needs at least 2 items" {
    var a = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer a.deinit();
    const s: DecisionSchema = .{ .rank = .{ .id = "r", .items = &.{"only"} } };
    try std.testing.expectError(error.BadItemCount, s.validateOne(a.allocator()));
}

test "score bucket count" {
    const s: DecisionSchema = .{ .score = .{
        .id = "s",
        .scale = .{ .int = .{ .min = 1, .max = 5 } },
    } };
    try std.testing.expectEqual(5, s.score.bucketCount());
    const l: DecisionSchema = .{ .score = .{
        .id = "s",
        .scale = .{ .labels = &.{ "a", "b", "c" } },
    } };
    try std.testing.expectEqual(3, l.score.bucketCount());
}

pub const DecisionType = enum { choice, noul, score, rank };

pub const ChoiceSchema = struct {
    id: []const u8,
    options: []const []const u8,
    abstain: bool = false,
};

pub const NoulSchema = struct {
    id: []const u8,
    abstain: bool = true,
};

pub const IntRange = struct {
    min: i16,
    max: i16,
};

pub const Scale = union(enum) {
    int: IntRange,
    labels: []const []const u8,
};

pub const ScoreSchema = struct {
    id: []const u8,
    scale: Scale,
    abstain: bool = false,

    pub fn bucketCount(s: *const ScoreSchema) usize {
        return switch (s.scale) {
            .int => |r| blk: {
                if (r.max <= r.min) break :blk 0;
                break :blk @intCast(@as(i32, r.max) - r.min + 1);
            },
            .labels => |l| l.len,
        };
    }
};

pub const RankSchema = struct {
    id: []const u8,
    items: []const []const u8,
};

pub const DecisionSchema = union(DecisionType) {
    choice: ChoiceSchema,
    noul: NoulSchema,
    score: ScoreSchema,
    rank: RankSchema,

    pub fn id(self: DecisionSchema) []const u8 {
        return switch (self) {
            inline else => |s| s.id,
        };
    }

    pub fn validateOne(self: DecisionSchema, a: alloc.Allocator) err.ValidateError!void {
        switch (self) {
            .choice => |c| try checkIds(a, c.options, 1, 255, error.BadOptionCount),
            .noul => {},
            .score => |s| {
                const n = s.bucketCount();
                if (n == 0 or n > 255) return error.BadRange;
                switch (s.scale) {
                    .int => |r| if (r.max <= r.min) return error.BadRange,
                    .labels => |l| try checkIds(a, l, 1, 255, error.BadOptionCount),
                }
            },
            .rank => |r| try checkIds(a, r.items, 2, 255, error.BadItemCount),
        }
    }
};

pub fn validateSet(schemas: []const DecisionSchema, a: alloc.Allocator) err.ValidateError!void {
    if (schemas.len == 0) return error.NoDecisions;
    if (schemas.len > 64) return error.TooManyDecisions;
    var seen = alloc.StringMap(void).init(a);
    defer seen.deinit();
    for (schemas) |s| {
        if (seen.contains(s.id())) return error.DuplicateId;
        try seen.put(s.id(), {});
        try s.validateOne(a);
    }
}

fn checkIds(
    a: alloc.Allocator,
    ids: []const []const u8,
    comptime min: usize,
    comptime max: usize,
    too_few: err.ValidateError,
) err.ValidateError!void {
    if (ids.len < min or ids.len > max) return too_few;
    var seen = alloc.StringMap(void).init(a);
    defer seen.deinit();
    for (ids) |x| {
        if (std.mem.eql(u8, x, "__abstain__")) return error.ReservedId;
        if (seen.contains(x)) return error.DuplicateId;
        try seen.put(x, {});
    }
}
