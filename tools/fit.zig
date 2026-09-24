const std = @import("std");
const zjev = @import("zjev");
const dataset = @import("dataset.zig");

const Group = struct {
    task: zjev.schema.DecisionType,
    num: u16,
    zs: std.ArrayList([]const f32) = .empty,
    labels: std.ArrayList(usize) = .empty,
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var dataset_path: ?[]const u8 = null;
    var model_name: []const u8 = "mock";
    var domain: []const u8 = "general";
    var out_dir: []const u8 = "model/calibration";
    var mock_mode: []const u8 = "peaked";
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--dataset") and i + 1 < args.len) {
            i += 1;
            dataset_path = args[i];
        } else if (std.mem.eql(u8, arg, "--model-name") and i + 1 < args.len) {
            i += 1;
            model_name = args[i];
        } else if (std.mem.eql(u8, arg, "--domain") and i + 1 < args.len) {
            i += 1;
            domain = args[i];
        } else if (std.mem.eql(u8, arg, "--out") and i + 1 < args.len) {
            i += 1;
            out_dir = args[i];
        } else if (std.mem.eql(u8, arg, "--mock-mode") and i + 1 < args.len) {
            i += 1;
            mock_mode = args[i];
        }
    }
    const path = dataset_path orelse {
        std.debug.print("usage: zjev-fit --dataset <jsonl> [--model-name name] [--domain d] [--out dir] [--mock-mode m]\n", .{});
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

    var groups: std.ArrayList(Group) = .empty;

    for (records) |rec| {
        const rd = std.json.parseFromValueLeaky(zjev.api_json.RawDecision, a, rec.decision, .{}) catch continue;
        var raws = [1]zjev.api_json.RawDecision{rd};
        const raw_req = zjev.api_json.RawRequest{
            .state = .{ .id = rec.state_id, .text = rec.state_text },
            .decisions = &raws,
        };
        const parsed = zjev.api_json.fromRaw(a, raw_req) catch continue;
        const s = parsed.schemas[0];
        const li = dataset.labelIndex(s, rec.label) catch continue;
        const outcome = zjev.engine.decideRaw(a, &model, &parsed.state, parsed.schemas) catch continue;
        const num: u16 = switch (s) {
            .choice => |c| @intCast(c.options.len),
            .noul => 2,
            .score => |sc| @intCast(sc.bucketCount()),
            .rank => |r| @intCast(r.items.len),
        };
        const gop = findOrAdd(&groups, a, std.meta.activeTag(s), num) catch continue;
        try gop.zs.append(a, outcome.logits);
        try gop.labels.append(a, li);
    }

    std.Io.Dir.cwd().createDir(io, out_dir, .default_dir) catch {};
    var out_dir_handle = std.Io.Dir.cwd().openDir(io, out_dir, .{}) catch std.Io.Dir.cwd();

    for (groups.items) |*g| {
        if (g.zs.items.len == 0) continue;
        const t = zjev.temperature.fit(g.zs.items, g.labels.items);
        var cal_conf: std.ArrayList(f32) = .empty;
        var cal_ok: std.ArrayList(bool) = .empty;
        var probs_rows: std.ArrayList([]const f32) = .empty;
        var label_idx: std.ArrayList(usize) = .empty;
        for (g.zs.items, g.labels.items) |z, li| {
            const row = try a.alloc(f32, z.len);
            zjev.softmax.apply(z, t, row) catch continue;
            try probs_rows.append(a, row);
            try label_idx.append(a, li);
            var m: f32 = 0;
            for (row) |p| m = @max(m, p);
            try cal_conf.append(a, m);
            try cal_ok.append(a, zjev.stats.argmax(row) == li);
        }
        const ece_r = zjev.ece.compute(cal_conf.items, cal_ok.items, 15);
        const brier_v = zjev.brier.score(probs_rows.items, label_idx.items);
        const sr = try zjev.stats.selectiveRisk(a, cal_conf.items, cal_ok.items, &zjev.stats.default_coverages);
        const profile = zjev.profile.Profile{
            .model = model_name,
            .task = @tagName(g.task),
            .num_options = g.num,
            .domain = domain,
            .temperature = t,
            .ece = @floatCast(ece_r.ece),
            .brier = @floatCast(brier_v),
            .@"selective_risk@0.5" = @floatCast(sr[0].risk),
            .@"selective_risk@0.7" = @floatCast(sr[1].risk),
            .@"selective_risk@0.9" = @floatCast(sr[2].risk),
            .@"selective_risk@0.95" = @floatCast(sr[3].risk),
            .fitted_at = "zjev-fit",
        };
        const json_text = try std.json.Stringify.valueAlloc(a, profile, .{});
        const fname = try std.fmt.allocPrint(a, "{s}_{s}_{d}_{s}.json", .{ model_name, @tagName(g.task), g.num, domain });
        try out_dir_handle.writeFile(io, .{ .sub_path = fname, .data = json_text });
        std.debug.print("fitted {s}: T={d:.4} (n={d})\n", .{ fname, t, g.zs.items.len });
    }
}

fn findOrAdd(groups: *std.ArrayList(Group), a: std.mem.Allocator, task: zjev.schema.DecisionType, num: u16) !*Group {
    for (groups.items) |*g| {
        if (g.task == task and g.num == num) return g;
    }
    try groups.append(a, .{ .task = task, .num = num });
    return &groups.items[groups.items.len - 1];
}
