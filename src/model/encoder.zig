const alloc = @import("../core/alloc.zig");
const state = @import("../core/state.zig");

pub const Error = error{ ModelFailed, OutOfMemory, BadState };
pub const HiddenState = opaque {};

pub const VTable = struct {
    encode: *const fn (ptr: *anyopaque, a: alloc.Allocator, s: *const state.State) Error!*HiddenState,
    deinit: *const fn (ptr: *anyopaque, a: alloc.Allocator, h: *HiddenState) void,
};

pub const Encoder = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub fn encode(self: Encoder, a: alloc.Allocator, s: *const state.State) Error!*HiddenState {
        return self.vtable.encode(self.ptr, a, s);
    }

    pub fn deinit(self: Encoder, a: alloc.Allocator, h: *HiddenState) void {
        self.vtable.deinit(self.ptr, a, h);
    }
};
