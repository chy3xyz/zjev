const std = @import("std");
const zjev = @import("zjev");

pub const Record = struct {
    state_id: ?[]const u8,
    state_text: ?[]const u8,
    decision: std.json.Value,
    label: std.json.Value,
};

const RawRecord = struct {
    state: struct {
        id: ?[]const u8 = null,
        text: ?[]const u8 = null,
    },
    decision: std.json.Value,
    label: std.json.Value,
};

pub fn load(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]Record {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30));
    var out: std.ArrayList(Record) = .empty;
    var it = std.mem.splitScalar(u8, bytes, '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        const raw = try std.json.parseFromSliceLeaky(RawRecord, a, line, .{});
        try out.append(a, .{
            .state_id = raw.state.id,
            .state_text = raw.state.text,
            .decision = raw.decision,
            .label = raw.label,
        });
    }
    return out.items;
}

pub fn labelIndex(schema_def: zjev.schema.DecisionSchema, label: std.json.Value) error{BadLabel}!usize {
    switch (schema_def) {
        .choice => |c| {
            const want = switch (label) {
                .string => |v| v,
                else => return error.BadLabel,
            };
            for (c.options, 0..) |o, i| {
                if (std.mem.eql(u8, o, want)) return i;
            }
            if (c.abstain and std.mem.eql(u8, want, "__abstain__")) return c.options.len;
            return error.BadLabel;
        },
        .noul => {
            const b = switch (label) {
                .bool => |v| v,
                else => return error.BadLabel,
            };
            return if (b) 0 else 1;
        },
        .score => |sc| {
            switch (sc.scale) {
                .int => |r| {
                    const v = switch (label) {
                        .integer => |x| x,
                        else => return error.BadLabel,
                    };
                    const idx = v - r.min;
                    if (idx < 0 or idx >= sc.bucketCount()) return error.BadLabel;
                    return @intCast(idx);
                },
                .labels => |ls| {
                    const want = switch (label) {
                        .string => |v| v,
                        else => return error.BadLabel,
                    };
                    for (ls, 0..) |l, i| {
                        if (std.mem.eql(u8, l, want)) return i;
                    }
                    return error.BadLabel;
                },
            }
        },
        .rank => return error.BadLabel,
    }
}
