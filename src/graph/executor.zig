const std = @import("std");
const alloc = @import("../core/alloc.zig");
const state = @import("../core/state.zig");
const schema = @import("../core/schema.zig");
const result = @import("../core/result.zig");
const err = @import("../core/error.zig");
const factory = @import("../model/factory.zig");
const engine = @import("../runtime/engine.zig");
const graph_mod = @import("types.zig");
const condition = @import("condition.zig");
const trajectory = @import("trajectory.zig");

pub const ExecError = err.EngineError || err.GraphError || condition.EvalError;

pub const Outcome = struct {
    steps: []trajectory.Step,
    skipped: []const []const u8,
    path_prob: f32,
};

fn byId(nodes: []const graph_mod.Node, a_idx: usize, b_idx: usize) bool {
    return std.mem.order(u8, nodes[a_idx].id, nodes[b_idx].id) == .lt;
}

fn schemaById(schemas: []const schema.DecisionSchema, id: []const u8) ?usize {
    for (schemas, 0..) |sc, i| {
        if (std.mem.eql(u8, sc.id(), id)) return i;
    }
    return null;
}

pub fn execute(
    a: alloc.Allocator,
    model: *const factory.Model,
    s: *const state.State,
    schemas: []const schema.DecisionSchema,
    g: graph_mod.Graph,
    temps: ?[]const f32,
) ExecError!Outcome {
    try s.validate();
    try schema.validateSet(schemas, a);
    try graph_mod.validate(g, schemas);

    const n = g.nodes.len;
    const activated = try a.alloc(bool, n);
    @memset(activated, false);
    const executed = try a.alloc(bool, n);
    @memset(executed, false);

    var frontier: std.ArrayList(usize) = .empty;
    for (g.nodes, 0..) |nd, i| {
        var indeg: usize = 0;
        for (g.edges) |e| {
            if (std.mem.eql(u8, e.to, nd.id)) indeg += 1;
        }
        if (indeg == 0) {
            activated[i] = true;
            try frontier.append(a, i);
        }
    }

    var steps: std.ArrayList(trajectory.Step) = .empty;
    var done_results: std.ArrayList(result.DecisionResult) = .empty;
    var skipped: std.ArrayList([]const u8) = .empty;
    var path_prob: f32 = 1.0;

    while (frontier.items.len > 0) {
        std.mem.sort(usize, frontier.items, g.nodes, byId);

        var batch_schemas: std.ArrayList(schema.DecisionSchema) = .empty;
        var batch_temps: std.ArrayList(f32) = .empty;
        for (frontier.items) |ni| {
            const si = schemaById(schemas, g.nodes[ni].decision).?;
            try batch_schemas.append(a, schemas[si]);
            try batch_temps.append(a, if (temps) |ts| ts[si] else 1.0);
        }

        const outcome = try engine.run(a, model, s, batch_schemas.items, batch_temps.items);

        for (frontier.items, outcome.results, batch_schemas.items) |ni, r, sc| {
            executed[ni] = true;
            var step: trajectory.Step = .{
                .node_id = g.nodes[ni].id,
                .decision_id = r.id,
                .result = r,
            };
            if (g.nodes[ni].gate) |gt| step.action = gt.apply(r);
            try steps.append(a, step);
            try done_results.append(a, r);
            path_prob *= try trajectory.massFor(a, r, sc);

            for (g.edges) |e| {
                if (!std.mem.eql(u8, e.from, g.nodes[ni].id)) continue;
                const t_idx = for (g.nodes, 0..) |nd, i| {
                    if (std.mem.eql(u8, nd.id, e.to)) break i;
                } else unreachable; // validate 已保证
                if (activated[t_idx] or executed[t_idx]) continue;
                if (try condition.eval(e.when, done_results.items)) {
                    activated[t_idx] = true;
                }
            }
        }

        frontier.clearRetainingCapacity();
        for (0..n) |i| {
            if (activated[i] and !executed[i]) try frontier.append(a, i);
        }
    }

    for (g.nodes, 0..) |nd, i| {
        if (!executed[i]) try skipped.append(a, nd.id);
    }

    return .{
        .steps = try steps.toOwnedSlice(a),
        .skipped = try skipped.toOwnedSlice(a),
        .path_prob = path_prob,
    };
}

fn litStr(s: []const u8) condition.Operand {
    return .{ .lit = .{ .str = s } };
}

fn fld(decision: []const u8) condition.Operand {
    return .{ .field = .{ .decision = decision, .field = .value } };
}

fn cmpEq(decision: []const u8, value: []const u8) condition.Cond {
    return .{ .cmp = .{ .op = .eq, .lhs = fld(decision), .rhs = litStr(value) } };
}

const two_choice_schemas = [_]schema.DecisionSchema{
    .{ .choice = .{ .id = "c1", .options = &.{ "a", "b" }, .abstain = false } },
    .{ .choice = .{ .id = "c2", .options = &.{ "x", "y" }, .abstain = false } },
};

fn execIn(a: alloc.Allocator, g: graph_mod.Graph, schemas: []const schema.DecisionSchema) !Outcome {
    var m = try factory.mockModel(.sequence, a);
    defer m.deinit(a);
    const s: state.State = .{ .text = "x" };
    return execute(a, &m, &s, schemas, g, null);
}

fn newArena() std.heap.ArenaAllocator {
    return std.heap.ArenaAllocator.init(std.testing.allocator);
}

test "execute linear chain runs both nodes in order" {
    const nodes = [_]graph_mod.Node{
        .{ .id = "n1", .decision = "c1" },
        .{ .id = "n2", .decision = "c2" },
    };
    const edges = [_]graph_mod.Edge{
        .{ .from = "n1", .to = "n2", .when = cmpEq("c1", "a") },
    };
    const g: graph_mod.Graph = .{ .nodes = &nodes, .edges = &edges };
    var arena = newArena();
    defer arena.deinit();
    const out = try execIn(arena.allocator(), g, &two_choice_schemas);
    try std.testing.expectEqual(@as(usize, 2), out.steps.len);
    try std.testing.expectEqualStrings("n1", out.steps[0].node_id);
    try std.testing.expectEqualStrings("n2", out.steps[1].node_id);
    try std.testing.expectEqual(@as(usize, 0), out.skipped.len);
    try std.testing.expect(out.path_prob > 0 and out.path_prob <= 1.0);
}

test "execute prunes branch when condition false" {
    // sequence 模式：批内第 i 个 schema 峰落在 i%n → c1（批内 i=0）恒为 "a"
    const nodes = [_]graph_mod.Node{
        .{ .id = "r", .decision = "c1" },
        .{ .id = "hit", .decision = "c2" },
        .{ .id = "miss", .decision = "c2b" },
    };
    var schemas: [3]schema.DecisionSchema = undefined;
    schemas[0] = two_choice_schemas[0];
    schemas[1] = two_choice_schemas[1];
    schemas[2] = .{ .choice = .{ .id = "c2b", .options = &.{ "u", "v" }, .abstain = false } };
    const edges = [_]graph_mod.Edge{
        .{ .from = "r", .to = "hit", .when = cmpEq("c1", "a") },
        .{ .from = "r", .to = "miss", .when = cmpEq("c1", "b") },
    };
    const g: graph_mod.Graph = .{ .nodes = &nodes, .edges = &edges };
    var arena = newArena();
    defer arena.deinit();
    const out = try execIn(arena.allocator(), g, &schemas);
    try std.testing.expectEqual(@as(usize, 2), out.steps.len);
    try std.testing.expectEqualStrings("hit", out.steps[1].node_id);
    try std.testing.expectEqual(@as(usize, 1), out.skipped.len);
    try std.testing.expectEqualStrings("miss", out.skipped[0]);
}

test "execute is deterministic across runs" {
    const nodes = [_]graph_mod.Node{
        .{ .id = "n1", .decision = "c1" },
        .{ .id = "n2", .decision = "c2" },
    };
    const edges = [_]graph_mod.Edge{
        .{ .from = "n1", .to = "n2", .when = cmpEq("c1", "a") },
    };
    const g: graph_mod.Graph = .{ .nodes = &nodes, .edges = &edges };
    var arena = newArena();
    defer arena.deinit();
    const a = arena.allocator();
    const o1 = try execIn(a, g, &two_choice_schemas);
    const p1 = o1.path_prob;
    const o2 = try execIn(a, g, &two_choice_schemas);
    try std.testing.expectEqual(o1.steps.len, o2.steps.len);
    try std.testing.expectEqual(p1, o2.path_prob);
    for (o1.steps, o2.steps) |s1, s2| {
        try std.testing.expectEqualStrings(s1.node_id, s2.node_id);
    }
}

test "execute records gate action" {
    const nodes = [_]graph_mod.Node{
        .{ .id = "n1", .decision = "c1", .gate = .{
            .threshold = 0.01,
            .action_above = "go",
            .action_below = "stop",
            .action_abstain = "review",
        } },
    };
    const g: graph_mod.Graph = .{ .nodes = &nodes, .edges = &.{} };
    var arena = newArena();
    defer arena.deinit();
    const out = try execIn(arena.allocator(), g, &two_choice_schemas);
    try std.testing.expectEqualStrings("go", out.steps[0].action.?);
}

test "execute OR activation via two incoming edges" {
    const nodes = [_]graph_mod.Node{
        .{ .id = "r", .decision = "c1" },
        .{ .id = "s", .decision = "c2" },
        .{ .id = "join", .decision = "c2b" },
    };
    var schemas: [3]schema.DecisionSchema = undefined;
    schemas[0] = two_choice_schemas[0];
    schemas[1] = two_choice_schemas[1];
    schemas[2] = .{ .choice = .{ .id = "c2b", .options = &.{"u"}, .abstain = false } };
    const edges = [_]graph_mod.Edge{
        .{ .from = "r", .to = "join", .when = cmpEq("c1", "a") },
        .{ .from = "s", .to = "join", .when = cmpEq("c2", "y") }, // sequence 下 c2（批内 i=1）恒为 "y"，OR 激活仍只执行一次
    };
    const g: graph_mod.Graph = .{ .nodes = &nodes, .edges = &edges };
    var arena = newArena();
    defer arena.deinit();
    const out = try execIn(arena.allocator(), g, &schemas);
    try std.testing.expectEqual(@as(usize, 3), out.steps.len);
    try std.testing.expectEqualStrings("join", out.steps[2].node_id);
}

