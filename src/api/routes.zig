const std = @import("std");
const alloc = @import("../core/alloc.zig");
const engine = @import("../runtime/engine.zig");
const json = @import("json.zig");
const server = @import("server.zig");
const schema = @import("../core/schema.zig");
const scheduler_mod = @import("../runtime/scheduler.zig");
const executor = @import("../graph/executor.zig");

const StatusCode = std.http.Status;

fn respondErr(
    req: *std.http.Server.Request,
    a: alloc.Allocator,
    status: StatusCode,
    code: []const u8,
    message: []const u8,
) !void {
    var aw: std.Io.Writer.Allocating = .init(a);
    defer aw.deinit();
    try json.writeErrorBody(&aw, code, message);
    const body = try aw.toOwnedSlice();
    try req.respond(body, .{
        .status = status,
        .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }},
    });
}

fn statusFor(e: json.Error) StatusCode {
    return switch (e) {
        error.InvalidJson, error.MissingField, error.BadAbstain => .bad_request,
        error.Unsupported, error.InvalidGraph => .bad_request,
        error.OutOfMemory => .internal_server_error,
    };
}

fn codeFor(e: json.Error) []const u8 {
    return switch (e) {
        error.Unsupported => "unsupported",
        error.InvalidGraph => "invalid_graph",
        else => "invalid_request",
    };
}

pub fn handle(
    shared: *server.Shared,
    req: *std.http.Server.Request,
    arena: *std.heap.ArenaAllocator,
    t_start: std.Io.Timestamp,
) !void {
    _ = t_start;
    const a = arena.allocator();

    const method = req.head.method;
    const target = req.head.target;

    if (method == .GET and std.mem.eql(u8, target, "/health")) {
        var aw: std.Io.Writer.Allocating = .init(a);
        defer aw.deinit();
        const w = &aw.writer;
        try w.print("{{\"status\":\"ok\",\"model\":\"{s}\",\"version\":\"0.1.0\"}}", .{shared.model_name});
        const body = try aw.toOwnedSlice();
        try req.respond(body, .{ .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }} });
        return;
    }

    if (method == .GET and std.mem.eql(u8, target, "/v1/schema")) {
        const body = "{\"state\":{\"text\":\"string\"},\"decisions\":[{\"id\":\"string\",\"type\":\"choice|noul|score|rank\"}]}";
        try req.respond(body, .{ .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }} });
        return;
    }

    const is_execute = method == .POST and std.mem.eql(u8, target, "/v1/execute");
    const is_decide = method == .POST and std.mem.eql(u8, target, "/v1/decide");
    const is_batch = method == .POST and std.mem.eql(u8, target, "/v1/decide/batch");
    if (!is_decide and !is_batch and !is_execute) {
        return respondErr(req, a, .not_found, "invalid_request", "unknown route");
    }

    const limit: usize = 1 << 20;
    var rbuf: [4096]u8 = undefined;
    const body = req.readerExpectNone(&rbuf).allocRemaining(a, .limited(limit)) catch {
        return respondErr(req, a, .bad_request, "invalid_request", "body read failed");
    };

    if (is_execute) {
        const parsed = json.parseExecuteRequest(a, body) catch |e| {
            return respondErr(req, a, statusFor(e), codeFor(e), @errorName(e));
        };
        parsed.state.validate() catch |e| {
            return respondErr(req, a, .bad_request, "invalid_request", @errorName(e));
        };
        schema.validateSet(parsed.schemas, a) catch |e| {
            return respondErr(req, a, .bad_request, "invalid_request", @errorName(e));
        };
        var temps: ?[]f32 = null;
        if (shared.profiles) |ps| {
            temps = try a.alloc(f32, parsed.schemas.len);
            for (parsed.schemas, 0..) |sc, i| {
                temps.?[i] = ps.lookup(shared.model_name, sc, parsed.domain) orelse 1.0;
            }
        }
        const t0 = std.Io.Timestamp.now(shared.io, .real);
        const outcome = executor.execute(a, shared.model, &parsed.state, parsed.schemas, parsed.graph, temps) catch |e| {
            return switch (e) {
                error.OutOfMemory => respondErr(req, a, .internal_server_error, "internal", @errorName(e)),
                error.ModelFailed, error.BadTemperature, error.BadState => respondErr(req, a, .internal_server_error, "internal", @errorName(e)),
                else => respondErr(req, a, .bad_request, "invalid_graph", @errorName(e)),
            };
        };
        const elapsed = t0.durationTo(std.Io.Timestamp.now(shared.io, .real)).toMicroseconds();
        for (outcome.steps) |*st| st.result.latency_us = @intCast(@max(0, elapsed));
        var aw: std.Io.Writer.Allocating = .init(a);
        defer aw.deinit();
        try json.writeExecuteResponse(&aw, outcome, if (shared.profiles != null) "matched" else "default");
        const out = try aw.toOwnedSlice();
        try req.respond(out, .{ .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }} });
        return;
    }

    if (is_decide) {
        const parsed = json.parseRequest(a, body) catch |e| {
            return respondErr(req, a, statusFor(e), codeFor(e), @errorName(e));
        };
        if (shared.scheduler) |sched| {
            var slot: scheduler_mod.Slot = .{};
            sched.submit(shared.io, .{
                .arena = arena.*,
                .parsed = parsed,
                .body = body,
                .slot = &slot,
            }) catch {
                return respondErr(req, a, .service_unavailable, "overloaded", "queue full");
            };
            slot.mutex.lockUncancelable(shared.io);
            defer slot.mutex.unlock(shared.io);
            while (!slot.done) {
                slot.cond.wait(shared.io, &slot.mutex) catch {};
            }
            return req.respond(slot.response, .{
                .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }},
            });
        }
        try executeOne(shared, req, a, parsed);
        return;
    }

    const batch = json.parseBatch(a, body) catch {
        return respondErr(req, a, .bad_request, "invalid_request", "bad batch body");
    };
    var aw: std.Io.Writer.Allocating = .init(a);
    defer aw.deinit();
    const w = &aw.writer;
    try w.writeAll("{\"responses\":[");
    for (batch, 0..) |one, i| {
        if (i > 0) try w.writeByte(',');
        const results = engine.decide(a, shared.model, shared.profiles, shared.model_name, &one.state, one.schemas, one.domain) catch {
            try json.writeErrorBody(&aw, "internal", "engine failed");
            continue;
        };
        try json.writeResponse(&aw, results, if (shared.profiles != null) "matched" else "default");
    }
    try w.writeAll("]}");
    const out = try aw.toOwnedSlice();
    try req.respond(out, .{ .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }} });
}

fn executeOne(
    shared: *server.Shared,
    req: *std.http.Server.Request,
    a: alloc.Allocator,
    parsed: json.Request,
) !void {
    parsed.state.validate() catch |e| {
        return respondErr(req, a, .bad_request, "invalid_request", @errorName(e));
    };
    schema.validateSet(parsed.schemas, a) catch |e| {
        return respondErr(req, a, .bad_request, "invalid_request", @errorName(e));
    };
    const t0 = std.Io.Timestamp.now(shared.io, .real);
    const results = engine.decide(a, shared.model, shared.profiles, shared.model_name, &parsed.state, parsed.schemas, parsed.domain) catch |e| {
        return respondErr(req, a, .internal_server_error, "internal", @errorName(e));
    };
    const elapsed = t0.durationTo(std.Io.Timestamp.now(shared.io, .real)).toMicroseconds();
    for (results) |*r| r.latency_us = @intCast(@max(0, elapsed));
    var aw: std.Io.Writer.Allocating = .init(a);
    defer aw.deinit();
    try json.writeResponse(&aw, results, if (shared.profiles != null) "matched" else "default");
    const body = try aw.toOwnedSlice();
    try req.respond(body, .{
        .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }},
    });
}

test "graph errors map to 400 invalid_graph" {
    try std.testing.expect(statusFor(error.InvalidGraph) == .bad_request);
    try std.testing.expectEqualStrings("invalid_graph", codeFor(error.InvalidGraph));
    try std.testing.expect(statusFor(error.OutOfMemory) == .internal_server_error);
    try std.testing.expectEqualStrings("unsupported", codeFor(error.Unsupported));
}
