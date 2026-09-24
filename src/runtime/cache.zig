const std = @import("std");
const alloc = @import("../core/alloc.zig");

pub const Cache = struct {
    enabled: bool,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    map: alloc.StringMap([]const u8),
    arena: std.heap.ArenaAllocator,

    pub fn init(a: alloc.Allocator, io: std.Io, enabled: bool) Cache {
        return .{
            .enabled = enabled,
            .io = io,
            .map = alloc.StringMap([]const u8).init(a),
            .arena = std.heap.ArenaAllocator.init(a),
        };
    }

    pub fn deinit(self: *Cache) void {
        self.map.deinit();
        self.arena.deinit();
    }

    pub fn get(self: *Cache, key: []const u8) ?[]const u8 {
        if (!self.enabled) return null;
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.map.get(key);
    }

    pub fn put(self: *Cache, key: []const u8, response: []const u8) !void {
        if (!self.enabled) return;
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        const k = try self.arena.allocator().dupe(u8, key);
        const v = try self.arena.allocator().dupe(u8, response);
        try self.map.put(k, v);
    }
};

test "cache roundtrip" {
    var threaded: std.Io.Threaded = .init(std.testing.allocator, .{});
    defer threaded.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c = Cache.init(a, threaded.io(), true);
    defer c.deinit();
    try c.put("m|general|abc", "{\"ok\":1}");
    const hit = c.get("m|general|abc").?;
    try std.testing.expectEqualStrings("{\"ok\":1}", hit);
    try std.testing.expectEqual(@as(?[]const u8, null), c.get("missing"));
}

test "disabled cache misses" {
    var threaded: std.Io.Threaded = .init(std.testing.allocator, .{});
    defer threaded.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c = Cache.init(a, threaded.io(), false);
    defer c.deinit();
    try c.put("k", "v");
    try std.testing.expectEqual(@as(?[]const u8, null), c.get("k"));
}
