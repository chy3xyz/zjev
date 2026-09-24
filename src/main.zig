const std = @import("std");
const zjev = @import("zjev");

pub const Cli = struct {
    bind: []const u8 = "127.0.0.1",
    port: u16 = 9377,
    mock_mode: []const u8 = "peaked",
    profiles_dir: ?[]const u8 = null,
    use_scheduler: bool = false,
    cache_enabled: bool = false,
    model_path: ?[]const u8 = null,
    num_sessions: u16 = 0,
};

pub fn parseCli(args: []const []const u8) error{ InvalidPort, InvalidSessions }!Cli {
    var cli = Cli{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--bind") and i + 1 < args.len) {
            i += 1;
            cli.bind = args[i];
        } else if (std.mem.eql(u8, arg, "--port") and i + 1 < args.len) {
            i += 1;
            cli.port = std.fmt.parseInt(u16, args[i], 10) catch return error.InvalidPort;
        } else if (std.mem.eql(u8, arg, "--mock-mode") and i + 1 < args.len) {
            i += 1;
            cli.mock_mode = args[i];
        } else if (std.mem.eql(u8, arg, "--profiles-dir") and i + 1 < args.len) {
            i += 1;
            cli.profiles_dir = args[i];
        } else if (std.mem.eql(u8, arg, "--model") and i + 1 < args.len) {
            i += 1;
            cli.model_path = args[i];
        } else if (std.mem.eql(u8, arg, "--sessions") and i + 1 < args.len) {
            i += 1;
            cli.num_sessions = std.fmt.parseInt(u16, args[i], 10) catch return error.InvalidSessions;
        } else if (std.mem.eql(u8, arg, "--scheduler")) {
            cli.use_scheduler = true;
        } else if (std.mem.eql(u8, arg, "--cache")) {
            cli.cache_enabled = true;
        } else {
            std.log.warn("unknown arg: {s}", .{arg});
        }
    }
    return cli;
}

pub fn modelNameFromPath(path: []const u8) []const u8 {
    const base = if (std.mem.lastIndexOfScalar(u8, path, '/')) |idx| path[idx + 1 ..] else path;
    if (std.mem.lastIndexOfScalar(u8, base, '.')) |dot| return base[0..dot];
    return base;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    const cli = parseCli(args) catch |err| {
        std.log.err("invalid arguments: {s}", .{@errorName(err)});
        return err;
    };

    const mode: zjev.mock.Mode = if (std.mem.eql(u8, cli.mock_mode, "uniform"))
        .uniform
    else if (std.mem.eql(u8, cli.mock_mode, "sequence"))
        .sequence
    else
        .peaked;

    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const tio = threaded.io();

    var model = zjev.factory.open(gpa, tio, .{
        .kind = if (cli.model_path != null) .onnx else .mock,
        .mock_mode = mode,
        .model_path = cli.model_path,
        .num_sessions = cli.num_sessions,
    }) catch |err| {
        if (err == error.Unsupported) {
            std.log.err("--model requires onnx support; rebuild with: zig build -Donnx=true", .{});
        } else {
            std.log.err("failed to open model '{s}': {s}", .{ cli.model_path orelse "mock", @errorName(err) });
        }
        return err;
    };
    defer model.deinit(gpa);

    var profiles_storage: zjev.profile.Profiles = undefined;
    var profiles_ptr: ?*zjev.profile.Profiles = null;
    if (cli.profiles_dir) |dir| {
        profiles_storage = zjev.profile.Profiles.init(gpa);
        try profiles_storage.loadDir(tio, dir);
        profiles_ptr = &profiles_storage;
    }

    const sched_storage = try gpa.alloc(zjev.scheduler.Job, zjev.scheduler.max_queue);
    defer gpa.free(sched_storage);

    var shared: zjev.server.Shared = .{
        .alloc = gpa,
        .io = tio,
        .model = &model,
        .profiles = profiles_ptr,
        .model_name = if (cli.model_path) |p| modelNameFromPath(p) else "mock",
        .scheduler = null,
    };

    var sched = zjev.scheduler.Scheduler.init(sched_storage, &shared, tio, cli.cache_enabled);
    var sgroup: std.Io.Group = .init;
    if (cli.use_scheduler) {
        shared.scheduler = &sched;
        sgroup.async(tio, schedRunner, .{ tio, &sched });
    }

    const addr = try std.Io.net.IpAddress.parse(cli.bind, cli.port);
    var listener = try addr.listen(tio, .{ .mode = .stream });
    defer listener.deinit(tio);

    std.log.info("zjev-serve listening on {s}:{d} (scheduler={})", .{ cli.bind, cli.port, cli.use_scheduler });
    try zjev.server.run(tio, &shared, &listener);
}

fn schedRunner(io: std.Io, sched: *zjev.scheduler.Scheduler) void {
    sched.start(io, @max(1, (std.Thread.getCpuCount() catch 4) / 2));
}

test "parseCli defaults" {
    const cli = try parseCli(&.{"zjev-serve"});
    try std.testing.expectEqualStrings("127.0.0.1", cli.bind);
    try std.testing.expectEqual(@as(u16, 9377), cli.port);
    try std.testing.expectEqualStrings("peaked", cli.mock_mode);
    try std.testing.expectEqual(@as(?[]const u8, null), cli.model_path);
    try std.testing.expectEqual(@as(u16, 0), cli.num_sessions);
    try std.testing.expect(!cli.use_scheduler);
    try std.testing.expect(!cli.cache_enabled);
}

test "parseCli model and sessions" {
    const cli = try parseCli(&.{
        "zjev-serve", "--model", "/models/zjev-v1.onnx", "--sessions", "8", "--port", "18080",
    });
    try std.testing.expectEqualStrings("/models/zjev-v1.onnx", cli.model_path.?);
    try std.testing.expectEqual(@as(u16, 8), cli.num_sessions);
    try std.testing.expectEqual(@as(u16, 18080), cli.port);
}

test "parseCli invalid port" {
    try std.testing.expectError(error.InvalidPort, parseCli(&.{ "zjev-serve", "--port", "abc" }));
}

test "parseCli invalid sessions" {
    try std.testing.expectError(error.InvalidSessions, parseCli(&.{ "zjev-serve", "--sessions", "x" }));
}

test "modelNameFromPath" {
    try std.testing.expectEqualStrings("zjev-v1", modelNameFromPath("/models/zjev-v1.onnx"));
    try std.testing.expectEqualStrings("laya", modelNameFromPath("laya.onnx"));
    try std.testing.expectEqualStrings("c.tar", modelNameFromPath("/a/b/c.tar.onnx"));
    try std.testing.expectEqualStrings("noext", modelNameFromPath("/x/noext"));
    try std.testing.expectEqualStrings("", modelNameFromPath(""));
}
