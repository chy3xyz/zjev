const std = @import("std");
const alloc = @import("../core/alloc.zig");
const err = @import("../core/error.zig");
const state = @import("../core/state.zig");
const schema = @import("../core/schema.zig");
const result = @import("../core/result.zig");
const softmax = @import("../calib/softmax.zig");
const stats = @import("../calib/stats.zig");
const profile = @import("../calib/profile.zig");
const factory = @import("../model/factory.zig");
const logits_mod = @import("../model/logits.zig");

pub const RunOutcome = struct {
    results: []result.DecisionResult,
    logits: []f32,
};

pub fn run(
    a: alloc.Allocator,
    model: *const factory.Model,
    s: *const state.State,
    schemas: []const schema.DecisionSchema,
    temps: ?[]const f32,
) err.EngineError!RunOutcome {
    try s.validate();
    try schema.validateSet(schemas, a);

    const hidden = try model.encoder.encode(a, s);
    defer model.encoder.deinit(a, hidden);

    const flat = try a.alloc(f32, blk: {
        var total: usize = 0;
        for (schemas) |sc| total += logits_mod.logitCount(sc);
        break :blk total;
    });
    const results = try a.alloc(result.DecisionResult, schemas.len);

    if (model.heads.get(.noul).bundled) {
        // bundle-shaped graph: one decide() over the full schema set,
        // flat logits in schema order.
        const z = try model.heads.get(.noul).decide(a, hidden, schemas);
        var off: usize = 0;
        for (schemas, 0..) |sc, i| {
            const n = logits_mod.logitCount(sc);
            const dst_off = flatOffset(schemas, i);
            @memcpy(flat[dst_off .. dst_off + n], z[off .. off + n]);
            const temp: f32 = if (temps) |ts| ts[i] else 1.0;
            results[i] = try finishOne(a, sc, flat[dst_off .. dst_off + n], temp);
            off += n;
        }
        return .{ .results = results, .logits = flat };
    }

    var by_type: [4]alloc.List(usize) = .{ .empty, .empty, .empty, .empty };
    defer for (&by_type) |*l| l.deinit(a);
    for (schemas, 0..) |sc, i| {
        try by_type[@intFromEnum(std.meta.activeTag(sc))].append(a, i);
    }

    inline for (0..4) |t| {
        const dt: schema.DecisionType = @enumFromInt(t);
        const idxs = by_type[t].items;
        if (idxs.len != 0) {
            const group = try a.alloc(schema.DecisionSchema, idxs.len);
            for (idxs, 0..) |si, j| group[j] = schemas[si];
            const z = try model.heads.get(dt).decide(a, hidden, group);
            var off: usize = 0;
            for (idxs, 0..) |si, j| {
                const n = logits_mod.logitCount(group[j]);
                const dst_off = flatOffset(schemas, si);
                const temp: f32 = if (temps) |ts| ts[si] else 1.0;
                @memcpy(flat[dst_off .. dst_off + n], z[off .. off + n]);
                results[si] = try finishOne(a, group[j], flat[dst_off .. dst_off + n], temp);
                off += n;
            }
        }
    }
    return .{ .results = results, .logits = flat };
}

fn flatOffset(schemas: []const schema.DecisionSchema, index: usize) usize {
    var off: usize = 0;
    for (schemas[0..index]) |sc| off += logits_mod.logitCount(sc);
    return off;
}

fn bucketLabelStrings(a: alloc.Allocator, sc: schema.ScoreSchema) ![]const []const u8 {
    const n = sc.bucketCount();
    const labels = try a.alloc([]const u8, n);
    switch (sc.scale) {
        .int => |r| {
            for (labels, 0..) |*l, i| {
                l.* = try std.fmt.allocPrint(a, "{d}", .{@as(i32, r.min) + @as(i32, @intCast(i))});
            }
        },
        .labels => |ls| {
            for (labels, ls) |*l, x| l.* = x;
        },
    }
    return labels;
}

fn entryLess(_: void, x: result.RankEntry, y: result.RankEntry) bool {
    if (x.score != y.score) return x.score > y.score;
    return std.mem.order(u8, x.id, y.id) == .lt;
}

fn finishOne(
    a: alloc.Allocator,
    sc: schema.DecisionSchema,
    z: []const f32,
    temp: f32,
) err.EngineError!result.DecisionResult {
    const view: logits_mod.View = .{ .data = z, .schema = sc };
    const probs = try a.alloc(f32, z.len);
    try softmax.apply(z, temp, probs);

    switch (sc) {
        .choice => |c| {
            const reg = probs[0..c.options.len];
            const winner = stats.argmax(probs);
            const value: []const u8 = if (winner == c.options.len) "__abstain__" else c.options[winner];
            var conf: f32 = 0;
            for (reg) |p| conf = @max(conf, p);
            return .{
                .id = c.id,
                .type = .choice,
                .value = .{ .choice = value },
                .probabilities = probs,
                .labels = c.options,
                .uncertainty = .{
                    .entropy = stats.entropy(reg),
                    .confidence = conf,
                    .abstention = if (c.abstain) probs[probs.len - 1] else null,
                },
            };
        },
        .noul => |n| {
            const yes = probs[0];
            return .{
                .id = n.id,
                .type = .noul,
                .value = .{ .noul = yes >= 0.5 },
                .probability = yes,
                .uncertainty = .{
                    .confidence = yes,
                    .abstention = if (n.abstain) probs[probs.len - 1] else null,
                },
            };
        },
        .score => |sc_def| {
            const reg = probs[0..sc_def.bucketCount()];
            const values = try view.bucketValues(a);
            return .{
                .id = sc_def.id,
                .type = .score,
                .value = .{ .score = stats.expectation(values, reg) },
                .probabilities = probs,
                .labels = try bucketLabelStrings(a, sc_def),
                .uncertainty = .{
                    .entropy = stats.entropy(reg),
                    .variance = stats.variance(values, reg),
                    .confidence = blk: {
                        var c: f32 = 0;
                        for (reg) |p| c = @max(c, p);
                        break :blk c;
                    },
                    .abstention = if (sc_def.abstain) probs[probs.len - 1] else null,
                },
            };
        },
        .rank => |r| {
            const entries = try a.alloc(result.RankEntry, r.items.len);
            const scores = try a.alloc(f32, r.items.len);
            for (r.items, 0..) |id, i| {
                const item_score = stats.sigmoid(z[i]);
                entries[i] = .{ .id = id, .score = item_score };
                scores[i] = item_score;
            }
            std.sort.block(result.RankEntry, entries, {}, entryLess);
            var total: f32 = 0;
            for (scores) |sv| total += sv;
            const norm = try a.dupe(f32, scores);
            for (norm) |*p| p.* /= total;
            return .{
                .id = r.id,
                .type = .rank,
                .value = .{ .rank = entries },
                .probabilities = scores,
                .labels = r.items,
                .uncertainty = .{
                    .entropy = stats.entropy(norm),
                    .confidence = blk: {
                        var c: f32 = 0;
                        for (scores) |sv| c = @max(c, sv);
                        break :blk c;
                    },
                },
            };
        },
    }
}

pub fn decide(
    a: alloc.Allocator,
    model: *const factory.Model,
    profiles: ?*const profile.Profiles,
    model_name: []const u8,
    s: *const state.State,
    schemas: []const schema.DecisionSchema,
    domain: []const u8,
) err.EngineError![]result.DecisionResult {
    const temps = try a.alloc(f32, schemas.len);
    for (schemas, 0..) |sc, i| {
        temps[i] = if (profiles) |ps| (ps.lookup(model_name, sc, domain) orelse 1.0) else 1.0;
    }
    const outcome = try run(a, model, s, schemas, temps);
    return outcome.results;
}

pub fn decideRaw(
    a: alloc.Allocator,
    model: *const factory.Model,
    s: *const state.State,
    schemas: []const schema.DecisionSchema,
) err.EngineError!RunOutcome {
    return run(a, model, s, schemas, null);
}

test "choice with abstain winner" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var m = try factory.mockModel(.sequence, a);
    const s: state.State = .{ .text = "x" };
    const schemas = [_]schema.DecisionSchema{
        .{ .choice = .{ .id = "c", .options = &.{ "a", "b" }, .abstain = true } },
    };
    const outcome = try run(a, &m, &s, &schemas, null);
    const r = outcome.results[0];
    try std.testing.expectEqualStrings("c", r.id);
    try std.testing.expectEqual(schema.DecisionType.choice, r.type);
    try std.testing.expect(r.probabilities.?.len == 3);
    try std.testing.expect(r.uncertainty.abstention != null);
    var sum: f32 = 0;
    for (r.probabilities.?) |p| sum += p;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}

test "noul value and probability" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var m = try factory.mockModel(.sequence, a);
    const s: state.State = .{ .text = "x" };
    const schemas = [_]schema.DecisionSchema{
        .{ .noul = .{ .id = "n", .abstain = false } },
    };
    const outcome = try run(a, &m, &s, &schemas, null);
    const r = outcome.results[0];
    try std.testing.expect(r.probability != null);
    try std.testing.expect(r.probability.? >= 0 and r.probability.? <= 1);
    switch (r.value) {
        .noul => |b| try std.testing.expect(b == (r.probability.? >= 0.5)),
        else => return error.TestUnexpectedResult,
    }
}

test "score expectation over buckets" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var m = try factory.mockModel(.sequence, a);
    const s: state.State = .{ .text = "x" };
    const schemas = [_]schema.DecisionSchema{
        .{ .score = .{ .id = "sc", .scale = .{ .int = .{ .min = 1, .max = 5 } } } },
    };
    const outcome = try run(a, &m, &s, &schemas, null);
    const r = outcome.results[0];
    try std.testing.expect(r.value.score >= 1.0 and r.value.score <= 5.0);
    try std.testing.expect(r.uncertainty.variance != null);
    try std.testing.expect(r.probabilities.?.len == 5);
    try std.testing.expectEqualStrings("1", r.labels.?[0]);
    try std.testing.expectEqualStrings("5", r.labels.?[4]);
}

test "rank sorted with labels in original order" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var m = try factory.mockModel(.peaked, a);
    const s: state.State = .{ .text = "x" };
    const schemas = [_]schema.DecisionSchema{
        .{ .rank = .{ .id = "r", .items = &.{ "x", "y", "z" } } },
    };
    const outcome = try run(a, &m, &s, &schemas, null);
    const r = outcome.results[0];
    try std.testing.expectEqual(@as(usize, 3), r.value.rank.len);
    try std.testing.expectEqualSlices([]const u8, &.{ "x", "y", "z" }, r.labels.?);
}

test "decide applies profile temperature" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var m = try factory.mockModel(.uniform, a);
    var ps = profile.Profiles.init(a);
    try ps.put(.{ .model = "m", .task = "noul", .num_options = 2, .temperature = 2.0 });
    const s: state.State = .{ .text = "x" };
    const schemas = [_]schema.DecisionSchema{
        .{ .noul = .{ .id = "n", .abstain = false } },
    };
    const results = try decide(a, &m, &ps, "m", &s, &schemas, "general");
    try std.testing.expectEqual(@as(f32, 0.5), results[0].probability.?);
}
