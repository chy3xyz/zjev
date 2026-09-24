const std = @import("std");
const alloc = @import("../core/alloc.zig");
const schema = @import("../core/schema.zig");

pub const Profile = struct {
    model: []const u8,
    task: []const u8,
    num_options: u16,
    domain: []const u8 = "general",
    temperature: f32,
    ece: ?f32 = null,
    brier: ?f32 = null,
    nll: ?f32 = null,
    fitted_at: ?[]const u8 = null,
};

pub const Profiles = struct {
    map: alloc.StringMap(Profile),
    parsed_list: alloc.List(std.json.Parsed(Profile)),
    a: alloc.Allocator,

    pub fn init(a: alloc.Allocator) Profiles {
        return .{ .map = alloc.StringMap(Profile).init(a), .parsed_list = .empty, .a = a };
    }

    pub fn deinit(self: *Profiles) void {
        for (self.parsed_list.items) |p| p.deinit();
        self.parsed_list.deinit(self.a);
        self.map.deinit();
    }

    fn keyFor(buf: []u8, model: []const u8, task: []const u8, num: u16, domain: []const u8) ![]u8 {
        return std.fmt.bufPrint(buf, "{s}|{s}|{d}|{s}", .{ model, task, num, domain });
    }

    pub fn put(self: *Profiles, p: Profile) !void {
        var buf: [256]u8 = undefined;
        const key = try keyFor(&buf, p.model, p.task, p.num_options, p.domain);
        try self.map.put(try self.a.dupe(u8, key), p);
    }

    pub fn lookup(self: *const Profiles, model: []const u8, s: schema.DecisionSchema, domain: []const u8) ?f32 {
        var buf: [256]u8 = undefined;
        const num: u16 = switch (s) {
            .choice => |c| @intCast(c.options.len),
            .noul => 2,
            .score => |sc| @intCast(sc.bucketCount()),
            .rank => |r| @intCast(r.items.len),
        };
        const key = keyFor(&buf, model, @tagName(s), num, domain) catch return null;
        const p = self.map.get(key) orelse return null;
        return p.temperature;
    }

    pub fn loadDir(self: *Profiles, io: std.Io, dir_path: []const u8) !void {
        var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
        defer dir.close(io);
        var it = dir.iterate();
        while (try it.next(io)) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.name, ".json")) continue;
            const bytes = try dir.readFileAlloc(io, self.a, entry.name, .{ .limit = .limited(1 << 20) });
            const parsed = try std.json.parseFromSlice(Profile, self.a, bytes, .{ .ignore_unknown_fields = true });
            try self.parsed_list.append(self.a, parsed);
            try self.put(parsed.value);
        }
    }
};

test "put and lookup" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var ps = Profiles.init(a);
    try ps.put(.{
        .model = "zjev-150m-v0.1",
        .task = "choice",
        .num_options = 4,
        .temperature = 1.37,
    });
    const s: schema.DecisionSchema = .{ .choice = .{ .id = "c", .options = &.{ "a", "b", "c", "d" } } };
    try std.testing.expectEqual(@as(?f32, 1.37), ps.lookup("zjev-150m-v0.1", s, "general"));
    try std.testing.expectEqual(@as(?f32, null), ps.lookup("other", s, "general"));
}

test "lookup per option count" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var ps = Profiles.init(a);
    try ps.put(.{ .model = "m", .task = "choice", .num_options = 2, .temperature = 2.0 });
    const two: schema.DecisionSchema = .{ .choice = .{ .id = "c", .options = &.{ "a", "b" } } };
    const three: schema.DecisionSchema = .{ .choice = .{ .id = "c", .options = &.{ "a", "b", "c" } } };
    try std.testing.expectEqual(@as(?f32, 2.0), ps.lookup("m", two, "general"));
    try std.testing.expectEqual(@as(?f32, null), ps.lookup("m", three, "general"));
}
