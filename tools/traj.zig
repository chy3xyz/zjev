const std = @import("std");
const zjev = @import("zjev");

const Record = struct {
    state: zjev.api_json.RawState,
    decisions: []zjev.api_json.RawDecision,
    graph: zjev.api_json.RawGraph,
    expected: std.json.Value,
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var dataset_path: ?[]const u8 = null;
    var profiles_dir: ?[]const u8 = null;
    var model_name: []const u8 = "mock";
    var domain: []const u8 = "general";
    var mock_mode: []const u8 = "peaked";
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--dataset") and i + 1 < args.len) {
            i += 1;
            dataset_path = args[i];
        } else if (std.mem.eql(u8, arg, "--profiles-dir") and i + 1 < args.len) {
            i += 1;
            profiles_dir = args[i];
        } else if (std.mem.eql(u8, arg, "--model-name") and i + 1 < args.len) {
            i += 1;
            model_name = args[i];
        } else if (std.mem.eql(u8, arg, "--domain") and i + 1 < args.len) {
            i += 1;
            domain = args[i];
        } else if (std.mem.eql(u8, arg, "--mock-mode") and i + 1 < args.len) {
            i += 1;
            mock_mode = args[i];
        }
    }
    const path = dataset_path orelse {
        std.debug.print("usage: zjev-traj --dataset <jsonl> [--mock-mode m] [--profiles-dir d]\n", .{});
        std.process.exit(2);
    };

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
    defer gpa.free(bytes);

    const mode: zjev.mock.Mode = if (std.mem.eql(u8, mock_mode, "uniform"))
        .uniform
    else if (std.mem.eql(u8, mock_mode, "sequence"))
        .sequence
    else
        .peaked;
    var model = try zjev.mock.model(mode, a);
    defer model.deinit(a);

    var profiles: ?zjev.profile.Profiles = null;
    if (profiles_dir) |dir| {
        profiles = zjev.profile.Profiles.init(a);
        try profiles.?.loadDir(io, dir);
    }

    var p_list: std.ArrayList(f32) = .empty;
    var y_list: std.ArrayList(bool) = .empty;
    var correct: usize = 0;
    var total: usize = 0;
    var skipped: usize = 0;
    var node_accs: std.ArrayList(zjev.report.NodeAcc) = .empty;

    var lines = std.mem.splitScalar(u8, bytes, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) continue;
        const rec = std.json.parseFromSliceLeaky(Record, a, trimmed, .{}) catch {
            skipped += 1;
            continue;
        };
        const req = zjev.api_json.fromRawExecute(a, .{
            .state = rec.state,
            .decisions = rec.decisions,
            .graph = rec.graph,
        }) catch {
            skipped += 1;
            continue;
        };
        if (rec.expected.object.count() == 0) {
            skipped += 1;
            continue;
        }
        var temps: ?[]f32 = null;
        if (profiles) |*ps| {
            temps = try a.alloc(f32, req.schemas.len);
            for (req.schemas, 0..) |sc, si| {
                temps.?[si] = ps.lookup(model_name, sc, domain) orelse 1.0;
            }
        }
        const outcome = zjev.executor.execute(a, &model, &req.state, req.schemas, req.graph, temps) catch {
            skipped += 1;
            continue;
        };
        var all_ok = true;
        for (rec.expected.object.keys(), rec.expected.object.values()) |k, v| {
            const si_opt = for (req.schemas, 0..) |sc, si| {
                if (std.mem.eql(u8, sc.id(), k)) break si;
            } else null;
            if (si_opt == null) {
                all_ok = false;
                continue;
            }
            const step = for (outcome.steps) |st| {
                if (std.mem.eql(u8, st.decision_id, k)) break st;
            } else {
                all_ok = false;
                continue;
            };
            if (!zjev.report.hit(req.schemas[si_opt.?], step.result, v)) all_ok = false;
        }
        try p_list.append(a, outcome.path_prob);
        try y_list.append(a, all_ok);
        if (all_ok) correct += 1;
        total += 1;
        for (outcome.steps) |st| {
            const ev = rec.expected.object.get(st.decision_id) orelse continue;
            const si_opt = for (req.schemas, 0..) |sc, si| {
                if (std.mem.eql(u8, sc.id(), st.decision_id)) break si;
            } else null;
            if (si_opt == null) continue;
            const acc = try findOrAdd(&node_accs, a, st.decision_id);
            const ok = zjev.report.hit(req.schemas[si_opt.?], st.result, ev);
            acc.n += 1;
            if (ok) acc.correct += 1;
            try acc.conf.append(a, st.result.uncertainty.confidence);
            try acc.ok.append(a, ok);
        }
    }

    const tb = zjev.report.trajectoryBrier(p_list.items, y_list.items);
    const te = zjev.ece.compute(p_list.items, y_list.items, 15);
    const sr = zjev.stats.selectiveRisk(a, p_list.items, y_list.items, &zjev.stats.default_coverages) catch null;
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    const w = &out.writer;
    const acc_f: f64 = if (total > 0) @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(total)) else 0;
    try w.print("{{\"n\":{d},\"trajectory_accuracy\":{d:.6},\"traj_brier\":{d:.6},\"traj_ece\":{d:.6},\"traj_mce\":{d:.6},\"skipped_records\":{d},\"selective_risk\":[", .{ total, acc_f, tb, te.ece, te.mce, skipped });
    if (sr) |pts| {
        for (pts, 0..) |pt, pi| {
            if (pi > 0) try w.writeByte(',');
            try w.print("{{\"coverage\":{d:.2},\"keep\":{d},\"n\":{d},\"risk\":{d:.6},\"threshold\":{d:.6}}}", .{ pt.coverage, pt.keep, pt.n, pt.risk, pt.threshold });
        }
    }
    try w.writeAll("],\"by_node\":[");
    for (node_accs.items, 0..) |*ac, idx| {
        if (idx > 0) try w.writeByte(',');
        const e = zjev.ece.compute(ac.conf.items, ac.ok.items, 15);
        const af: f64 = if (ac.n > 0) @as(f64, @floatFromInt(ac.correct)) / @as(f64, @floatFromInt(ac.n)) else 0;
        try w.print("{{\"id\":\"{s}\",\"n\":{d},\"accuracy\":{d:.6},\"ece\":{d:.6}}}", .{ ac.id, ac.n, af, e.ece });
    }
    try w.writeAll("]}");
    std.debug.print("{s}\n", .{try out.toOwnedSlice()});
}

fn findOrAdd(accs: *std.ArrayList(zjev.report.NodeAcc), a: std.mem.Allocator, id: []const u8) !*zjev.report.NodeAcc {
    for (accs.items) |*ac| {
        if (std.mem.eql(u8, ac.id, id)) return ac;
    }
    try accs.append(a, .{ .id = id });
    return &accs.items[accs.items.len - 1];
}
