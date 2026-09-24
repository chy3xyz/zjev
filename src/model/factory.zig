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
