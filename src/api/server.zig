const std = @import("std");
const alloc = @import("../core/alloc.zig");
const factory = @import("../model/factory.zig");
const profile = @import("../calib/profile.zig");
const routes = @import("routes.zig");
const scheduler = @import("../runtime/scheduler.zig");

pub const Shared = struct {
    alloc: alloc.Allocator,
    io: std.Io,
    model: *factory.Model,
    profiles: ?*profile.Profiles,
    model_name: []const u8,
    scheduler: ?*scheduler.Scheduler,
};

fn connWorker(io: std.Io, listener: *std.Io.net.Server, shared: *Shared) void {
    connWorkerErr(io, listener, shared) catch {};
}

fn connWorkerErr(io: std.Io, listener: *std.Io.net.Server, shared: *Shared) !void {
    while (true) {
        var stream = listener.accept(io) catch return;
        var rbuf: [8192]u8 = undefined;
        var wbuf: [8192]u8 = undefined;
        var rs = stream.reader(io, &rbuf);
        var ws = stream.writer(io, &wbuf);
        var http_server: std.http.Server = .init(&rs.interface, &ws.interface);
        while (true) {
            var req = http_server.receiveHead() catch break;
            var arena = std.heap.ArenaAllocator.init(shared.alloc);
            defer arena.deinit();
            const t_start = std.Io.Timestamp.now(io, .real);
            routes.handle(shared, &req, &arena, t_start) catch break;
        }
        stream.close(io);
    }
}

pub fn run(io: std.Io, shared: *Shared, listener: *std.Io.net.Server) !void {
    var group: std.Io.Group = .init;
    defer group.cancel(io);
    const workers = @max(1, std.Thread.getCpuCount() catch 4);
    for (0..workers) |_| {
        group.async(io, connWorker, .{ io, listener, shared });
    }
    try group.await(io);
}
