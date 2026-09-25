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
    var model_path: ?[]const u8 = null;
    var num_sessions: u16 = 0;
    var ort_extensions: ?[]const u8 = null;
    var bundle_json: ?[]const u8 = null;
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
        } else if (std.mem.eql(u8, arg, "--model") and i + 1 < args.len) {
            i += 1;
            model_path = args[i];
        } else if (std.mem.eql(u8, arg, "--sessions") and i + 1 < args.len) {
            i += 1;
            num_sessions = try std.fmt.parseInt(u16, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--ort-extensions") and i + 1 < args.len) {
            i += 1;
            ort_extensions = args[i];
        } else if (std.mem.eql(u8, arg, "--bundle") and i + 1 < args.len) {
            i += 1;
            bundle_json = args[i];
        }
    }
    const path = dataset_path orelse {
        std.debug.print("usage: zjev-fit --dataset <jsonl> [--mock-mode m] [--model-name n] [--domain d] [--out dir] [--model p.onnx [--sessions n] [--ort-extensions lib] --bundle '<json>']]\n", .{});
        std.process.exit(2);
    };
    if (model_path == null and bundle_json != null) {
        std.debug.print("--bundle requires --model (ONNX bundle)\n", .{});
        std.process.exit(2);
    }

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
    var model = if (model_path) |mp| blk: {
        const m = zjev.factory.open(a, io, .{
            .kind = .onnx,
            .model_path = mp,
            .num_sessions = num_sessions,
            .ort_extensions = ort_extensions,
        }) catch |e| {
            if (e == error.Unsupported) {
                std.debug.print("--model requires an onnx build: zig build -Donnx=true -Donnx_lib_dir=<dir>\n", .{});
                std.process.exit(2);
            }
            std.debug.print("failed to open model '{s}': {s}\n", .{ mp, @errorName(e) });
            std.process.exit(1);
        };
        break :blk m;
    } else try zjev.mock.model(mode, a);
    defer model.deinit(a);

    var bundle_schemas: ?[]zjev.schema.DecisionSchema = null;
    var seg_starts: []usize = &.{};
    if (bundle_json) |bj| {
        const raw = std.json.parseFromSliceLeaky([]zjev.api_json.RawDecision, a, bj, .{}) catch {
            std.debug.print("invalid --bundle json\n", .{});
            std.process.exit(2);
        };
        const parsed = zjev.api_json.fromRaw(a, .{ .state = .{ .text = "x" }, .decisions = raw }) catch {
            std.debug.print("invalid --bundle schemas\n", .{});
            std.process.exit(2);
        };
        bundle_schemas = parsed.schemas;
        var starts: std.ArrayList(usize) = .empty;
        var off: usize = 0;
        for (parsed.schemas) |sc| {
            try starts.append(a, off);
            off += zjev.logits.logitCount(sc);
        }
        seg_starts = starts.items;
    }

    var groups: std.ArrayList(Group) = .empty;

    const total_recs = records.len;
    var skipped: usize = 0;
    var cached_text: ?[]const u8 = null;
    var cached_logits: []const f32 = &.{};

    for (records) |rec| {
        const rd = std.json.parseFromValueLeaky(zjev.api_json.RawDecision, a, rec.decision, .{}) catch {
            skipped += 1;
            continue;
        };
        var raws = [1]zjev.api_json.RawDecision{rd};
        const raw_req = zjev.api_json.RawRequest{
            .state = .{ .id = rec.state_id, .text = rec.state_text },
            .decisions = &raws,
        };
        const parsed = zjev.api_json.fromRaw(a, raw_req) catch {
            skipped += 1;
            continue;
        };
        const s = parsed.schemas[0];
        const li = dataset.labelIndex(s, rec.label) catch {
            skipped += 1;
            continue;
        };

        var seg: []const f32 = undefined;
        if (bundle_schemas) |bs| {
            const key = rec.state_text orelse rec.state_id orelse "";
            if (cached_text == null or !std.mem.eql(u8, cached_text.?, key)) {
                const full = zjev.engine.decideRaw(a, &model, &parsed.state, bs) catch {
                    skipped += 1;
                    continue;
                };
                cached_text = try a.dupe(u8, key);
                cached_logits = full.logits;
            }
            const si = blk: {
                for (bs, 0..) |bsc, idx| {
                    if (std.mem.eql(u8, bsc.id(), s.id())) break :blk idx;
                }
                skipped += 1;
                continue;
            };
            const st = seg_starts[si];
            const n = zjev.logits.logitCount(s);
            if (st + n > cached_logits.len) {
                skipped += 1;
                continue;
            }
            seg = cached_logits[st .. st + n];
        } else {
            const outcome = zjev.engine.decideRaw(a, &model, &parsed.state, parsed.schemas) catch {
                skipped += 1;
                continue;
            };
            seg = outcome.logits;
        }

        const num: u16 = switch (s) {
            .choice => |c| @intCast(c.options.len),
            .noul => 2,
            .score => |sc| @intCast(sc.bucketCount()),
            .rank => |r| @intCast(r.items.len),
        };
        const gop = findOrAdd(&groups, a, std.meta.activeTag(s), num) catch continue;
        try gop.zs.append(a, seg);
        try gop.labels.append(a, li);
    }
    std.debug.print("fitted: {d} skipped: {d}\n", .{ total_recs - skipped, skipped });

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
