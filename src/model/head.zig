const alloc = @import("../core/alloc.zig");
const encoder = @import("encoder.zig");
const schema = @import("../core/schema.zig");

pub const Error = encoder.Error;

pub const VTable = struct {
    decide: *const fn (
        ptr: *anyopaque,
        a: alloc.Allocator,
        hidden: *encoder.HiddenState,
        schemas: []const schema.DecisionSchema,
    ) Error![]f32,
};

pub const Head = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub fn decide(
        self: Head,
        a: alloc.Allocator,
        hidden: *encoder.HiddenState,
        schemas: []const schema.DecisionSchema,
    ) Error![]f32 {
        return self.vtable.decide(self.ptr, a, hidden, schemas);
    }
};
