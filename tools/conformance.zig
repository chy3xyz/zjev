const std = @import("std");
const zjev = @import("zjev");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;

    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const tio = threaded.io();

    var dir = try std.Io.Dir.cwd().openDir(tio, "test/conformance", .{ .iterate = true });
    defer dir.close(tio);

    var pass: usize = 0;
    var fail: usize = 0;

    var it = dir.iterate();
    while (try it.next(tio)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".json")) continue;
        const bytes = try dir.readFileAlloc(tio, entry.name, gpa, .limited(1 << 20));
        defer gpa.free(bytes);
        const ok = runFixture(gpa, bytes) catch |e| blk: {
            std.debug.print("FAIL {s}: {s}\n", .{ entry.name, @errorName(e) });
            break :blk false;
        };
        if (ok) {
            pass += 1;
            std.debug.print("PASS {s}\n", .{entry.name});
        } else {
            fail += 1;
        }
    }
    std.debug.print("conformance: {d} pass, {d} fail\n", .{ pass, fail });
    if (fail > 0) std.process.exit(1);
}

const Fixture = struct {
    name: []const u8,
    mode: []const u8,
    request: zjev.api_json.RawRequest,
    graph: ?zjev.api_json.RawGraph = null,
    expect_error: ?[]const u8 = null,
    expect: ?Expect = null,
    expect_path: ?[]const []const u8 = null,
    expect_skipped: ?[]const []const u8 = null,
    expect_step0_action: ?[]const u8 = null,
};

const Expect = struct {
    first: ?First = null,
};

const First = struct {
    type: []const u8,
    value_in: ?[]const []const u8 = null,
    has_abstention: ?bool = null,
    has_variance: ?bool = null,
    rank_len: ?usize = null,
    probs_sum: ?f64 = null,
};

fn runFixture(gpa: std.mem.Allocator, bytes: []const u8) !bool {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const fx = try std.json.parseFromSliceLeaky(Fixture, a, bytes, .{});

    const mode: zjev.mock.Mode = if (std.mem.eql(u8, fx.mode, "uniform"))
        .uniform
    else if (std.mem.eql(u8, fx.mode, "sequence"))
        .sequence
    else
        .peaked;

    var model = try zjev.mock.model(mode, a);
    defer model.deinit(a);

    const parsed = zjev.api_json.fromRaw(a, fx.request) catch |e| {
        if (fx.expect_error != null) return true;
        std.debug.print("  parse error: {s}\n", .{@errorName(e)});
        return false;
    };
    if (fx.graph) |rg| {
        const ex_req = zjev.api_json.fromRawExecute(a, .{
            .state = fx.request.state,
            .decisions = fx.request.decisions,
            .domain = fx.request.domain,
            .policy = fx.request.policy,
            .graph = rg,
        }) catch {
            return fx.expect_error != null;
        };
        if (fx.expect_error != null) return false;
        const outcome = zjev.executor.execute(a, &model, &ex_req.state, ex_req.schemas, ex_req.graph, null) catch {
            return fx.expect_error != null;
        };
        if (fx.expect_path) |want| {
            if (outcome.steps.len != want.len) return false;
            for (outcome.steps, want) |st, w| {
                if (!std.mem.eql(u8, st.node_id, w)) return false;
            }
        }
        if (fx.expect_skipped) |want| {
            if (outcome.skipped.len != want.len) return false;
            for (outcome.skipped, want) |sk, w| {
                if (!std.mem.eql(u8, sk, w)) return false;
            }
        }
        if (fx.expect_step0_action) |want| {
            if (outcome.steps.len == 0) return false;
            const act = outcome.steps[0].action orelse return false;
            if (!std.mem.eql(u8, act, want)) return false;
        }
        return true;
    }

    if (fx.expect_error != null) {
        zjev.schema.validateSet(parsed.schemas, a) catch return true;
        std.debug.print("  expected error but validated\n", .{});
        return false;
    }

    const results = try zjev.engine.decide(a, &model, null, "mock", &parsed.state, parsed.schemas, parsed.domain);
    const exp = fx.expect orelse return false;
    const first = exp.first orelse return true;
    const r = results[0];

    if (!std.mem.eql(u8, first.type, @tagName(r.type))) return false;
    if (first.value_in) |allowed| {
        const got = switch (r.value) {
            .choice => |v| v,
            else => return false,
        };
        var found = false;
        for (allowed) |x| {
            if (std.mem.eql(u8, x, got)) found = true;
        }
        if (!found) return false;
    }
    if (first.has_abstention) |want| {
        if ((r.uncertainty.abstention != null) != want) return false;
    }
    if (first.has_variance) |want| {
        if ((r.uncertainty.variance != null) != want) return false;
    }
    if (first.rank_len) |want| {
        switch (r.value) {
            .rank => |entries| if (entries.len != want) return false,
            else => return false,
        }
    }
    if (first.probs_sum) |want| {
        var sum: f32 = 0;
        for (r.probabilities orelse return false) |p| sum += p;
        if (@abs(sum - want) > 1e-3) return false;
    }
    return true;
}
