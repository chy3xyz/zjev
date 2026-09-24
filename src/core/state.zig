const std = @import("std");
const err = @import("error.zig");

pub const State = struct {
    id: ?[]const u8 = null,
    text: ?[]const u8 = null,
    data: ?std.json.Value = null,
    embeddings: ?[]const f32 = null,
    timestamp_ms: ?i64 = null,
    source: ?[]const u8 = null,

    pub fn validate(s: *const State) err.ValidateError!void {
        if (s.text == null and s.data == null and s.embeddings == null) {
            return error.EmptyState;
        }
    }
};

test "empty state rejected" {
    const s: State = .{};
    try std.testing.expectError(error.EmptyState, s.validate());
}

test "text-only state accepted" {
    const s: State = .{ .text = "hello" };
    try s.validate();
}

test "data-only state accepted" {
    const s: State = .{ .data = .{ .integer = 42 } };
    try s.validate();
}
