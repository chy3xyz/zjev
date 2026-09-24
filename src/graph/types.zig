const std = @import("std");
const schema = @import("../core/schema.zig");
const err = @import("../core/error.zig");

pub const Node = struct {
    id: []const u8,
    decision: []const u8,
};

pub const Edge = struct {
    from: []const u8,
    to: []const u8,
    when: []const u8,
};

pub const Graph = struct {
    nodes: []const Node,
    edges: []const Edge,
};

fn choiceGraph() Graph {
    return .{
        .nodes = &.{.{ .id = "n1", .decision = "c1" }},
        .edges = &.{},
    };
}

test "validate accepts single node" {
    const schemas = [_]schema.DecisionSchema{
        .{ .choice = .{ .id = "c1", .options = &.{ "a", "b" }, .abstain = false } },
    };
    try validate(choiceGraph(), &schemas);
}

test "validate rejects empty graph" {
    const g: Graph = .{ .nodes = &.{}, .edges = &.{} };
    try std.testing.expectError(error.EmptyGraph, validate(g, &.{}));
}

test "validate rejects duplicate node id" {
    const nodes = [_]Node{ .{ .id = "n", .decision = "c1" }, .{ .id = "n", .decision = "c1" } };
    const schemas = [_]schema.DecisionSchema{
        .{ .choice = .{ .id = "c1", .options = &.{ "a", "b" }, .abstain = false } },
    };
    const g: Graph = .{ .nodes = &nodes, .edges = &.{} };
    try std.testing.expectError(error.DuplicateNodeId, validate(g, &schemas));
}

test "validate rejects unknown decision" {
    const g: Graph = .{ .nodes = &.{.{ .id = "n1", .decision = "nope" }}, .edges = &.{} };
    try std.testing.expectError(error.UnknownDecision, validate(g, &.{}));
}

test "validate rejects one decision referenced twice" {
    const nodes = [_]Node{ .{ .id = "n1", .decision = "c1" }, .{ .id = "n2", .decision = "c1" } };
    const schemas = [_]schema.DecisionSchema{
        .{ .choice = .{ .id = "c1", .options = &.{ "a", "b" }, .abstain = false } },
    };
    const g: Graph = .{ .nodes = &nodes, .edges = &.{} };
    try std.testing.expectError(error.DuplicateDecisionRef, validate(g, &schemas));
}

test "validate rejects edge to unknown node" {
    const g: Graph = .{
        .nodes = &.{.{ .id = "n1", .decision = "c1" }},
        .edges = &.{.{ .from = "n1", .to = "ghost", .when = "c1 == a" }},
    };
    const schemas = [_]schema.DecisionSchema{
        .{ .choice = .{ .id = "c1", .options = &.{ "a", "b" }, .abstain = false } },
    };
    try std.testing.expectError(error.UnknownEdgeNode, validate(g, &schemas));
}

test "validate rejects cycle" {
    const nodes = [_]Node{
        .{ .id = "n1", .decision = "c1" },
        .{ .id = "n2", .decision = "c2" },
    };
    const edges = [_]Edge{
        .{ .from = "n1", .to = "n2", .when = "c1 == a" },
        .{ .from = "n2", .to = "n1", .when = "c2 == a" },
    };
    const schemas = [_]schema.DecisionSchema{
        .{ .choice = .{ .id = "c1", .options = &.{"a"}, .abstain = false } },
        .{ .choice = .{ .id = "c2", .options = &.{"a"}, .abstain = false } },
    };
    const g: Graph = .{ .nodes = &nodes, .edges = &edges };
    try std.testing.expectError(error.Cycle, validate(g, &schemas));
}

test "validate rejects self loop" {
    const g: Graph = .{
        .nodes = &.{.{ .id = "n1", .decision = "c1" }},
        .edges = &.{.{ .from = "n1", .to = "n1", .when = "c1 == a" }},
    };
    const schemas = [_]schema.DecisionSchema{
        .{ .choice = .{ .id = "c1", .options = &.{"a"}, .abstain = false } },
    };
    try std.testing.expectError(error.SelfLoop, validate(g, &schemas));
}

test "validate rejects more than 64 nodes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const nodes = try a.alloc(Node, 65);
    for (nodes, 0..) |*nd, i| {
        nd.* = .{ .id = try std.fmt.allocPrint(a, "n{d}", .{i}), .decision = "c1" };
    }
    const g: Graph = .{ .nodes = nodes, .edges = &.{} };
    try std.testing.expectError(error.TooManyNodes, validate(g, &.{}));
}

pub fn validate(g: Graph, schemas: []const schema.DecisionSchema) err.GraphError!void {
    if (g.nodes.len == 0) return error.EmptyGraph;
    if (g.nodes.len > 64) return error.TooManyNodes;
    for (g.nodes) |n| {
        if (n.id.len == 0) return error.EmptyNodeId;
        if (schemaById(schemas, n.decision) == null) return error.UnknownDecision;
    }
    for (g.nodes, 0..) |n, i| {
        for (g.nodes[i + 1 ..]) |m| {
            if (std.mem.eql(u8, n.id, m.id)) return error.DuplicateNodeId;
            if (std.mem.eql(u8, n.decision, m.decision)) return error.DuplicateDecisionRef;
        }
    }
    for (g.edges) |e| {
        if (std.mem.eql(u8, e.from, e.to)) return error.SelfLoop;
        if (nodeIndex(g, e.from) == null or nodeIndex(g, e.to) == null) return error.UnknownEdgeNode;
    }
    try acyclic(g);
}

fn schemaById(schemas: []const schema.DecisionSchema, id: []const u8) ?schema.DecisionSchema {
    for (schemas) |sc| {
        if (std.mem.eql(u8, sc.id(), id)) return sc;
    }
    return null;
}

fn nodeIndex(g: Graph, id: []const u8) ?usize {
    for (g.nodes, 0..) |n, i| {
        if (std.mem.eql(u8, n.id, id)) return i;
    }
    return null;
}

fn acyclic(g: Graph) err.GraphError!void {
    var indeg: [64]usize = @splat(0);
    for (g.edges) |e| {
        indeg[nodeIndex(g, e.to).?] += 1;
    }
    var queue: [64]usize = undefined;
    var head: usize = 0;
    var tail: usize = 0;
    for (g.nodes, 0..) |_, i| {
        if (indeg[i] == 0) {
            queue[tail] = i;
            tail += 1;
        }
    }
    var seen: usize = 0;
    while (head < tail) {
        const cur = queue[head];
        head += 1;
        seen += 1;
        for (g.edges) |e| {
            if (std.mem.eql(u8, e.from, g.nodes[cur].id)) {
                const t = nodeIndex(g, e.to).?;
                indeg[t] -= 1;
                if (indeg[t] == 0) {
                    queue[tail] = t;
                    tail += 1;
                }
            }
        }
    }
    if (seen != g.nodes.len) return error.Cycle;
}
