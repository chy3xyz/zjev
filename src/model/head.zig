const alloc = @import("../core/alloc.zig");
const encoder = @import("encoder.zig");
const schema = @import("../core/schema.zig");

pub const Error = encoder.Error || error{BadModelIO};

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
    /// true: decide() consumes the WHOLE schema set in one call and returns
    /// flat logits in schema order (bundle-shaped graphs, e.g. ONNX export).
    /// false: per-decision-type calls with same-type schema groups (mock).
    bundled: bool = false,

    pub fn decide(
        self: Head,
        a: alloc.Allocator,
        hidden: *encoder.HiddenState,
        schemas: []const schema.DecisionSchema,
    ) Error![]f32 {
        return self.vtable.decide(self.ptr, a, hidden, schemas);
    }
};
