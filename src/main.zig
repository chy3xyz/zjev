const std = @import("std");
const zjev = @import("zjev");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var bind: []const u8 = "127.0.0.1";
    var port: u16 = 9377;
    var mock_mode: []const u8 = "peaked";
    var profiles_dir: ?[]const u8 = null;
    var use_scheduler = false;
    var cache_enabled = false;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--bind") and i + 1 < args.len) {
            i += 1;
            bind = args[i];
        } else if (std.mem.eql(u8, arg, "--port") and i + 1 < args.len) {
            i += 1;
            port = try std.fmt.parseInt(u16, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--mock-mode") and i + 1 < args.len) {
            i += 1;
            mock_mode = args[i];
        } else if (std.mem.eql(u8, arg, "--profiles-dir") and i + 1 < args.len) {
            i += 1;
            profiles_dir = args[i];
        } else if (std.mem.eql(u8, arg, "--scheduler")) {
            use_scheduler = true;
        } else if (std.mem.eql(u8, arg, "--cache")) {
            cache_enabled = true;
        } else {
            std.log.warn("unknown arg: {s}", .{arg});
        }
    }

    const mode: zjev.mock.Mode = if (std.mem.eql(u8, mock_mode, "uniform"))
        .uniform
    else if (std.mem.eql(u8, mock_mode, "sequence"))
        .sequence
    else
        .peaked;

    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const tio = threaded.io();

    var model = try zjev.mock.model(mode, gpa);
    defer model.deinit(gpa);

    var profiles_storage: zjev.profile.Profiles = undefined;
    var profiles_ptr: ?*zjev.profile.Profiles = null;
    if (profiles_dir) |dir| {
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
        .model_name = "mock",
        .scheduler = null,
    };

    var sched = zjev.scheduler.Scheduler.init(sched_storage, &shared, tio, cache_enabled);
    var sgroup: std.Io.Group = .init;
    if (use_scheduler) {
        shared.scheduler = &sched;
        sgroup.async(tio, schedRunner, .{ tio, &sched });
    }

    const addr = try std.Io.net.IpAddress.parse(bind, port);
    var listener = try addr.listen(tio, .{ .mode = .stream });
    defer listener.deinit(tio);

    std.log.info("zjev-serve listening on {s}:{d} (scheduler={})", .{ bind, port, use_scheduler });
    try zjev.server.run(tio, &shared, &listener);
}

fn schedRunner(io: std.Io, sched: *zjev.scheduler.Scheduler) void {
    sched.start(io, @max(1, (std.Thread.getCpuCount() catch 4) / 2));
}
