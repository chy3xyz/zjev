const std = @import("std");
const alloc = @import("../core/alloc.zig");
const schema = @import("../core/schema.zig");
const encoder = @import("encoder.zig");
const head = @import("head.zig");

pub const Model = struct {
    ptr: *anyopaque,
    deinitFn: *const fn (ptr: *anyopaque, a: alloc.Allocator) void,
    encoder: encoder.Encoder,
    heads: std.EnumArray(schema.DecisionType, head.Head),

    pub fn deinit(self: *Model, a: alloc.Allocator) void {
        self.deinitFn(self.ptr, a);
    }
};

pub fn mockModel(mode: @import("mock.zig").Mode, a: alloc.Allocator) error{OutOfMemory}!Model {
    return @import("mock.zig").model(mode, a);
}

pub const Config = struct {
    kind: enum { mock, onnx } = .mock,
    mock_mode: @import("mock.zig").Mode = .peaked,
    model_path: ?[]const u8 = null,
    num_sessions: u16 = 0,
};

pub fn open(a: alloc.Allocator, io: std.Io, cfg: Config) !Model {
    return switch (cfg.kind) {
        .mock => @import("mock.zig").model(cfg.mock_mode, a),
        .onnx => blk: {
            if (!@import("build_options").onnx) return error.Unsupported;
            break :blk @import("onnx.zig").openOnnx(a, io, cfg.model_path orelse return error.MissingModelPath, cfg.num_sessions);
        },
    };
}
