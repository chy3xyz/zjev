const std = @import("std");
const zjev = @import("zjev");
const dataset = @import("dataset.zig");

const Accum = struct {
    task: zjev.schema.DecisionType,
    num: u16,
    n: usize = 0,
    correct: usize = 0,
    conf: std.ArrayList(f32) = .empty,
    ok: std.ArrayList(bool) = .empty,
    probs: std.ArrayList([]const f32) = .empty,
    labels: std.ArrayList(usize) = .empty,
    abstain_correct: usize = 0,
    abstain_total: usize = 0,
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
        std.debug.print("usage: zjev-bench --dataset <jsonl> [--profiles-dir dir]\n", .{});
        std.process.exit(2);
    };

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const records = try dataset.load(a, io, path);
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

    var accs: std.ArrayList(Accum) = .empty;
    var skipped_rank: usize = 0;

    for (records) |rec| {
        const rd = std.json.parseFromValueLeaky(zjev.api_json.RawDecision, a, rec.decision, .{}) catch continue;
        var raws = [1]zjev.api_json.RawDecision{rd};
        const raw_req = zjev.api_json.RawRequest{
            .state = .{ .id = rec.state_id, .text = rec.state_text },
            .decisions = &raws,
        };
        const parsed = zjev.api_json.fromRaw(a, raw_req) catch continue;
        const s = parsed.schemas[0];
        if (s == .rank) {
            skipped_rank += 1;
            continue;
        }
        const li = dataset.labelIndex(s, rec.label) catch continue;
        const results = zjev.engine.decide(a, &model, if (profiles) |*ps| ps else null, model_name, &parsed.state, parsed.schemas, domain) catch continue;
        const r = results[0];
        const num: u16 = switch (s) {
            .choice => |c| @intCast(c.options.len),
            .noul => 2,
            .score => |sc| @intCast(sc.bucketCount()),
            .rank => unreachable,
        };
        const acc = try findOrAdd(&accs, a, std.meta.activeTag(s), num);
        acc.n += 1;
        var noul_probs: [2]f32 = undefined;
        const ps: []const f32 = r.probabilities orelse switch (r.value) {
            .noul => blk: {
                noul_probs = .{ r.probability.?, 1.0 - r.probability.? };
                break :blk &noul_probs;
            },
            else => continue,
        };
        const pred = zjev.stats.argmax(ps);
        const hit = pred == li;
        if (hit) acc.correct += 1;
        var conf: f32 = 0;
        for (ps) |p| conf = @max(conf, p);
        try acc.conf.append(a, conf);
        try acc.ok.append(a, hit);
        try acc.probs.append(a, ps);
        try acc.labels.append(a, li);
        if (r.uncertainty.abstention != null) {
            acc.abstain_total += 1;
            const is_abstain = switch (r.value) {
                .choice => |v| std.mem.eql(u8, v, "__abstain__"),
                else => false,
            };
            const label_str: ?[]const u8 = switch (rec.label) {
                .string => |v| v,
                else => null,
            };
            const label_abstain = label_str != null and std.mem.eql(u8, label_str.?, "__abstain__");
            if (is_abstain == label_abstain) acc.abstain_correct += 1;
        }
    }

    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    const w = &out.writer;
    try w.writeAll("{\"groups\":[");
    for (accs.items, 0..) |*ac, gi| {
        if (gi > 0) try w.writeByte(',');
        const acc_f: f64 = if (ac.n > 0) @as(f64, @floatFromInt(ac.correct)) / @as(f64, @floatFromInt(ac.n)) else 0;
        const brier_v = zjev.brier.score(ac.probs.items, ac.labels.items);
        const ece_r = zjev.ece.compute(ac.conf.items, ac.ok.items, 15);
        try w.print("{{\"task\":\"{s}\",\"num_options\":{d},\"n\":{d},\"accuracy\":{d:.6},\"brier\":{d:.6},\"ece\":{d:.6},\"mce\":{d:.6}", .{
            @tagName(ac.task), ac.num, ac.n, acc_f, brier_v, ece_r.ece, ece_r.mce,
        });
        if (ac.abstain_total > 0) {
            const aa: f64 = @as(f64, @floatFromInt(ac.abstain_correct)) / @as(f64, @floatFromInt(ac.abstain_total));
            try w.print(",\"abstention_accuracy\":{d:.6}", .{aa});
        }
        try w.writeByte('}');
    }
    if (skipped_rank > 0) {
        try w.print("],\"skipped_rank\":{d}}}", .{skipped_rank});
    } else {
        try w.writeAll("]}");
    }
    std.debug.print("{s}\n", .{try out.toOwnedSlice()});
}

fn findOrAdd(accs: *std.ArrayList(Accum), a: std.mem.Allocator, task: zjev.schema.DecisionType, num: u16) !*Accum {
    for (accs.items) |*ac| {
        if (ac.task == task and ac.num == num) return ac;
    }
    try accs.append(a, .{ .task = task, .num = num });
    return &accs.items[accs.items.len - 1];
}
