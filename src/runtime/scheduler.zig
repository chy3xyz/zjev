const std = @import("std");
const alloc = @import("../core/alloc.zig");
const api_json = @import("../api/json.zig");
const server = @import("../api/server.zig");
const engine = @import("engine.zig");
const cache_mod = @import("cache.zig");
const json = @import("../api/json.zig");

pub const max_batch = 8;
pub const max_queue = 1024;

pub const Slot = struct {
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    done: bool = false,
    response: []const u8 = &.{},
};

pub const Job = struct {
    arena: std.heap.ArenaAllocator,
    parsed: api_json.Request,
    body: []const u8,
    slot: *Slot,
};

pub const Scheduler = struct {
    queue: std.Io.Queue(Job),
    shared: *server.Shared,
    cache: cache_mod.Cache,
    io: std.Io,

    pub fn init(storage: []Job, shared: *server.Shared, io: std.Io, cache_enabled: bool) Scheduler {
        return .{
            .queue = std.Io.Queue(Job).init(storage),
            .shared = shared,
            .cache = cache_mod.Cache.init(shared.alloc, io, cache_enabled),
            .io = io,
        };
    }

    pub fn submit(self: *Scheduler, io: std.Io, job: Job) error{ Overloaded, Closed, Canceled }!void {
        var one = [1]Job{job};
        const n = try self.queue.put(io, &one, 0);
        if (n == 0) return error.Overloaded;
    }

    pub fn start(self: *Scheduler, io: std.Io, worker_count: usize) void {
        for (0..worker_count) |_| {
            _ = std.Io.async(io, worker, .{ io, self });
        }
    }

    fn worker(io: std.Io, self: *Scheduler) void {
        workerErr(io, self) catch {};
    }

    fn workerErr(io: std.Io, self: *Scheduler) !void {
        var buf: [max_batch]Job = undefined;
        while (true) {
            const n1 = self.queue.get(io, &buf, 1) catch return;
            const n = n1 + (self.queue.get(io, buf[n1..], 0) catch 0);
            for (buf[0..n]) |*job| {
                processJob(io, self, job);
            }
        }
    }
};

fn cacheKey(a: alloc.Allocator, model_name: []const u8, domain: []const u8, body: []const u8) ![]u8 {
    var h = std.hash.Wyhash.init(0);
    h.update(model_name);
    h.update(domain);
    h.update(body);
    return std.fmt.allocPrint(a, "{s}|{s}|{x}", .{ model_name, domain, h.final() });
}

fn processJob(io: std.Io, self: *Scheduler, job: *Job) void {
    const a = job.arena.allocator();

    const key = cacheKey(a, self.shared.model_name, job.parsed.domain, job.body) catch null;
    if (key) |k| {
        if (self.cache.get(k)) |hit| {
            const response = a.dupe(u8, hit) catch {
                return failSlot(io, job, "oom");
            };
            return finishSlot(io, job, response);
        }
    }

    var aw: std.Io.Writer.Allocating = .init(a);
    const results = engine.decide(
        a,
        self.shared.model,
        self.shared.profiles,
        self.shared.model_name,
        &job.parsed.state,
        job.parsed.schemas,
        job.parsed.domain,
    ) catch {
        return failSlot(io, job, "engine failed");
    };
    json.writeResponse(&aw, results, if (self.shared.profiles != null) "matched" else "default") catch {
        return failSlot(io, job, "encode failed");
    };
    const response = aw.toOwnedSlice() catch {
        return failSlot(io, job, "oom");
    };
    if (key) |k| {
        self.cache.put(k, response) catch {};
    }
    finishSlot(io, job, response);
}

fn finishSlot(io: std.Io, job: *Job, response: []const u8) void {
    job.slot.mutex.lockUncancelable(io);
    defer job.slot.mutex.unlock(io);
    job.slot.response = response;
    job.slot.done = true;
    job.slot.cond.signal(io);
}

fn failSlot(io: std.Io, job: *Job, msg: []const u8) void {
    const a = job.arena.allocator();
    var aw: std.Io.Writer.Allocating = .init(a);
    json.writeErrorBody(&aw, "internal", msg) catch return;
    const response = aw.toOwnedSlice() catch return;
    finishSlot(io, job, response);
}

test "batch idiom drains available" {
    var threaded: std.Io.Threaded = .init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var storage: [16]Job = undefined;
    var q: std.Io.Queue(Job) = .init(&storage);
    var slot: Slot = .{};
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const dummy_req = api_json.Request{
        .state = .{ .text = "x" },
        .schemas = &.{},
    };
    try q.putOne(io, .{ .arena = arena, .parsed = dummy_req, .body = "", .slot = &slot });
    var buf: [max_batch]Job = undefined;
    const n1 = try q.get(io, &buf, 1);
    const n = n1 + try q.get(io, buf[n1..], 0);
    try std.testing.expectEqual(1, n);
    q.close(io);
}
