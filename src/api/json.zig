const std = @import("std");
const alloc = @import("../core/alloc.zig");
const state = @import("../core/state.zig");
const schema = @import("../core/schema.zig");
const result = @import("../core/result.zig");

pub const Error = error{
    InvalidJson,
    MissingField,
    Unsupported,
    BadAbstain,
    OutOfMemory,
};

pub const Request = struct {
    state: state.State,
    schemas: []schema.DecisionSchema,
    domain: []const u8 = "general",
};

pub const RawScale = struct {
    min: ?i16 = null,
    max: ?i16 = null,
    labels: ?[]const []const u8 = null,
};

pub const RawDecision = struct {
    id: []const u8,
    type: []const u8,
    options: ?[]const []const u8 = null,
    items: ?[]const []const u8 = null,
    abstain: ?bool = null,
    scale: ?RawScale = null,
};

pub const RawState = struct {
    id: ?[]const u8 = null,
    text: ?[]const u8 = null,
    data: ?std.json.Value = null,
    embeddings: ?[]const f32 = null,
    timestamp: ?f64 = null,
    source: ?[]const u8 = null,
};

pub const RawRequest = struct {
    state: RawState,
    decisions: []RawDecision,
    domain: ?[]const u8 = null,
    policy: ?std.json.Value = null,
};

fn convertRaw(a: alloc.Allocator, raw: RawRequest) Error!Request {
    if (raw.policy != null) return error.Unsupported;
    const schemas = try a.alloc(schema.DecisionSchema, raw.decisions.len);
    for (raw.decisions, 0..) |d, i| {
        schemas[i] = try convert(a, d);
    }
    return .{
        .state = .{
            .id = raw.state.id,
            .text = raw.state.text,
            .data = raw.state.data,
            .embeddings = raw.state.embeddings,
            .timestamp_ms = if (raw.state.timestamp) |t| @intFromFloat(t) else null,
            .source = raw.state.source,
        },
        .schemas = schemas,
        .domain = raw.domain orelse "general",
    };
}

pub fn parseRequest(a: alloc.Allocator, body: []const u8) Error!Request {
    const raw = std.json.parseFromSliceLeaky(RawRequest, a, body, .{}) catch return error.InvalidJson;
    return convertRaw(a, raw);
}

pub fn fromRaw(a: alloc.Allocator, raw: RawRequest) Error!Request {
    return convertRaw(a, raw);
}

pub fn parseBatch(a: alloc.Allocator, body: []const u8) Error![]Request {
    const RawBatch = struct { requests: []RawRequest };
    const raw = std.json.parseFromSliceLeaky(RawBatch, a, body, .{}) catch return error.InvalidJson;
    const out = try a.alloc(Request, raw.requests.len);
    for (raw.requests, 0..) |rr, i| {
        out[i] = try convertRaw(a, rr);
    }
    return out;
}

fn convert(a: alloc.Allocator, d: RawDecision) Error!schema.DecisionSchema {
    _ = a;
    if (std.mem.eql(u8, d.type, "choice")) {
        const options = d.options orelse return error.MissingField;
        return .{ .choice = .{ .id = d.id, .options = options, .abstain = d.abstain orelse false } };
    }
    if (std.mem.eql(u8, d.type, "noul")) {
        return .{ .noul = .{ .id = d.id, .abstain = d.abstain orelse true } };
    }
    if (std.mem.eql(u8, d.type, "score")) {
        const scale = d.scale orelse return error.MissingField;
        if (scale.labels) |ls| {
            return .{ .score = .{ .id = d.id, .scale = .{ .labels = ls }, .abstain = d.abstain orelse false } };
        }
        const mn = scale.min orelse return error.MissingField;
        const mx = scale.max orelse return error.MissingField;
        return .{ .score = .{ .id = d.id, .scale = .{ .int = .{ .min = mn, .max = mx } }, .abstain = d.abstain orelse false } };
    }
    if (std.mem.eql(u8, d.type, "rank")) {
        const items = d.items orelse return error.MissingField;
        return .{ .rank = .{ .id = d.id, .items = items } };
    }
    return error.Unsupported;
}

fn writeJsonString(w: *std.Io.Writer, s: []const u8) std.Io.Writer.Error!void {
    try w.writeByte('"');
    for (s) |ch| {
        switch (ch) {
            '"' => try w.writeAll("\\\""),
            '\\' => try w.writeAll("\\\\"),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            '\t' => try w.writeAll("\\t"),
            else => {
                if (ch < 0x20) {
                    try w.print("\\u{x:0>4}", .{ch});
                } else {
                    try w.writeByte(ch);
                }
            },
        }
    }
    try w.writeByte('"');
}

fn writeUncertainty(w: *std.Io.Writer, u: result.Uncertainty) std.Io.Writer.Error!void {
    try w.writeByte('{');
    var first = true;
    if (u.entropy) |e| {
        try w.print("\"entropy\":{d:.6}", .{e});
        first = false;
    }
    if (u.variance) |v| {
        if (!first) try w.writeByte(',');
        try w.print("\"variance\":{d:.6}", .{v});
        first = false;
    }
    if (!first) try w.writeByte(',');
    try w.print("\"confidence\":{d:.6}", .{u.confidence});
    if (u.abstention) |ab| {
        try w.print(",\"abstention\":{d:.6}", .{ab});
    }
    try w.writeByte('}');
}

fn writeResult(w: *std.Io.Writer, r: result.DecisionResult) std.Io.Writer.Error!void {
    try w.print("{{\"id\":", .{});
    try writeJsonString(w, r.id);
    try w.print(",\"type\":\"{s}\",\"value\":", .{@tagName(r.type)});
    switch (r.value) {
        .choice => |v| try writeJsonString(w, v),
        .noul => |v| try w.writeAll(if (v) "true" else "false"),
        .score => |v| try w.print("{d:.6}", .{v}),
        .rank => |entries| {
            try w.writeByte('[');
            for (entries, 0..) |e, i| {
                if (i > 0) try w.writeByte(',');
                try w.print("{{\"id\":", .{});
                try writeJsonString(w, e.id);
                try w.print(",\"score\":{d:.6}}}", .{e.score});
            }
            try w.writeByte(']');
        },
    }
    if (r.probability) |p| {
        try w.print(",\"probability\":{d:.6}", .{p});
    }
    if (r.probabilities) |ps| {
        try w.writeAll(",\"probabilities\":{");
        if (r.labels) |labels| {
            for (ps[0..labels.len], 0..) |p, i| {
                if (i > 0) try w.writeByte(',');
                try writeJsonString(w, labels[i]);
                try w.print(":{d:.6}", .{p});
            }
            if (ps.len > labels.len) {
                try w.writeAll(",\"__abstain__\":");
                try w.print("{d:.6}", .{ps[labels.len]});
            }
        }
        try w.writeByte('}');
    }
    try w.writeAll(",\"uncertainty\":");
    try writeUncertainty(w, r.uncertainty);
    try w.print(",\"latency_us\":{d}}}", .{r.latency_us});
}

pub fn writeResponse(
    aw: *std.Io.Writer.Allocating,
    results: []const result.DecisionResult,
    calibration: []const u8,
) !void {
    const w = &aw.writer;
    try w.writeAll("{\"results\":[");
    for (results, 0..) |r, i| {
        if (i > 0) try w.writeByte(',');
        try writeResult(w, r);
    }
    try w.writeAll("],\"calibration\":");
    try writeJsonString(w, calibration);
    try w.writeByte('}');
}

pub fn writeErrorBody(aw: *std.Io.Writer.Allocating, code: []const u8, message: []const u8) !void {
    const w = &aw.writer;
    try w.writeAll("{\"error\":{\"code\":");
    try writeJsonString(w, code);
    try w.writeAll(",\"message\":");
    try writeJsonString(w, message);
    try w.writeAll("}}");
}

test "parse minimal choice request" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const req = try parseRequest(a,
        \\{"state":{"text":"hi"},"decisions":[{"id":"c","type":"choice","options":["a","b"]}]}
    );
    try std.testing.expectEqualStrings("hi", req.state.text.?);
    try std.testing.expectEqual(@as(usize, 1), req.schemas.len);
    try std.testing.expectEqualStrings("general", req.domain);
}

test "policy rejected" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.Unsupported, parseRequest(arena.allocator(),
        \\{"state":{"text":"x"},"decisions":[],"policy":{}}
    ));
}

test "unknown type rejected" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.Unsupported, parseRequest(arena.allocator(),
        \\{"state":{"text":"x"},"decisions":[{"id":"d","type":"wat"}]}
    ));
}

test "score with labels" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const req = try parseRequest(arena.allocator(),
        \\{"state":{"text":"x"},"decisions":[{"id":"s","type":"score","scale":{"labels":["lo","hi"]}}],"domain":"web3"}
    );
    try std.testing.expectEqualStrings("web3", req.domain);
    try std.testing.expectEqual(@as(usize, 2), req.schemas[0].score.bucketCount());
}

test "write response shape" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const results = [_]result.DecisionResult{
        .{
            .id = "risk_level",
            .type = .choice,
            .value = .{ .choice = "medium" },
            .probabilities = &.{ 0.12, 0.73, 0.09, 0.06 },
            .labels = &.{ "low", "medium", "high" },
            .uncertainty = .{ .entropy = 0.700779, .confidence = 0.73, .abstention = 0.06 },
            .latency_us = 384,
        },
    };
    var aw: std.Io.Writer.Allocating = .init(a);
    defer aw.deinit();
    try writeResponse(&aw, &results, "matched");
    const s = try aw.toOwnedSlice();
    try std.testing.expect(std.mem.indexOf(u8, s, "\"value\":\"medium\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "\"__abstain__\":0.060000") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "\"calibration\":\"matched\"") != null);
}
