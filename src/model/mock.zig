const std = @import("std");
const alloc = @import("../core/alloc.zig");
const state = @import("../core/state.zig");
const schema = @import("../core/schema.zig");
const encoder = @import("encoder.zig");
const head = @import("head.zig");
const logits = @import("logits.zig");
const factory = @import("factory.zig");

pub const Mode = enum { uniform, peaked, sequence };

const Mock = struct { mode: Mode };
const MockHidden = struct { seed: u64 };

fn fnv1a(bytes: []const u8) u64 {
    var hash: u64 = 0xcbf29ce484222325;
    for (bytes) |b| {
        hash ^= b;
        hash *%= 0x100000001b3;
    }
    return hash;
}

fn encodeImpl(ptr: *anyopaque, a: alloc.Allocator, s: *const state.State) encoder.Error!*encoder.HiddenState {
    _ = ptr;
    const seed = fnv1a(s.id orelse s.text orelse "");
    const h = try a.create(MockHidden);
    h.* = .{ .seed = seed };
    return @ptrCast(h);
}

fn deinitImpl(ptr: *anyopaque, a: alloc.Allocator, h: *encoder.HiddenState) void {
    _ = ptr;
    a.destroy(@as(*MockHidden, @ptrCast(@alignCast(h))));
}

const encoder_vtable: encoder.VTable = .{ .encode = encodeImpl, .deinit = deinitImpl };

fn decideImpl(
    ptr: *anyopaque,
    a: alloc.Allocator,
    hidden: *encoder.HiddenState,
    schemas: []const schema.DecisionSchema,
) head.Error![]f32 {
    const self: *Mock = @ptrCast(@alignCast(ptr));
    const h: *MockHidden = @ptrCast(@alignCast(hidden));
    var total: usize = 0;
    for (schemas) |s| total += logits.logitCount(s);
    const buf = try a.alloc(f32, total);
    var prng = std.Random.DefaultPrng.init(h.seed);
    const rand = prng.random();
    var off: usize = 0;
    for (schemas, 0..) |s, i| {
        const n = logits.logitCount(s);
        const slice = buf[off .. off + n];
        switch (self.mode) {
            .uniform => @memset(slice, 0),
            .peaked => {
                @memset(slice, 0);
                slice[rand.uintLessThan(usize, n)] = 4.0;
            },
            .sequence => {
                @memset(slice, 0);
                slice[i % n] = 4.0;
            },
        }
        off += n;
    }
    return buf;
}

const head_vtable: head.VTable = .{ .decide = decideImpl };

fn modelDeinit(ptr: *anyopaque, a: alloc.Allocator) void {
    a.destroy(@as(*Mock, @ptrCast(@alignCast(ptr))));
}

pub fn model(mode: Mode, a: alloc.Allocator) error{OutOfMemory}!factory.Model {
    const m = try a.create(Mock);
    m.* = .{ .mode = mode };
    return .{
        .ptr = m,
        .deinitFn = modelDeinit,
        .encoder = .{ .ptr = m, .vtable = &encoder_vtable },
        .heads = .initFill(.{ .ptr = m, .vtable = &head_vtable }),
    };
}

test "mock determinism" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var m = try model(.peaked, a);
    defer m.deinit(a);
    const s: state.State = .{ .id = "seed-me", .text = "x" };
    const schemas = [_]schema.DecisionSchema{
        .{ .choice = .{ .id = "c", .options = &.{ "a", "b" }, .abstain = true } },
    };
    const h1 = try m.encoder.encode(a, &s);
    const z1 = try m.heads.get(.choice).decide(a, h1, &schemas);
    const h2 = try m.encoder.encode(a, &s);
    const z2 = try m.heads.get(.choice).decide(a, h2, &schemas);
    try std.testing.expectEqualSlices(f32, z1, z2);
}

test "mock uniform gives zero logits" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var m = try model(.uniform, a);
    defer m.deinit(a);
    const s: state.State = .{ .text = "x" };
    const schemas = [_]schema.DecisionSchema{
        .{ .noul = .{ .id = "n" } },
    };
    const h = try m.encoder.encode(a, &s);
    const z = try m.heads.get(.noul).decide(a, h, &schemas);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0 }, z);
}

test "mock sequence rotates peak" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var m = try model(.sequence, a);
    defer m.deinit(a);
    const s: state.State = .{ .text = "x" };
    const schemas = [_]schema.DecisionSchema{
        .{ .choice = .{ .id = "c1", .options = &.{ "a", "b" } } },
        .{ .choice = .{ .id = "c2", .options = &.{ "a", "b" } } },
    };
    const h = try m.encoder.encode(a, &s);
    const z = try m.heads.get(.choice).decide(a, h, &schemas);
    try std.testing.expectEqual(@as(f32, 4.0), z[0]);
    try std.testing.expectEqual(@as(f32, 0.0), z[1]);
    try std.testing.expectEqual(@as(f32, 0.0), z[2]);
    try std.testing.expectEqual(@as(f32, 4.0), z[3]);
}
