const std = @import("std");

pub const Allocator = std.mem.Allocator;
pub const ArenaAllocator = std.heap.ArenaAllocator;

pub fn List(comptime T: type) type {
    return std.ArrayList(T);
}

pub fn StringMap(comptime V: type) type {
    return std.StringHashMap(V);
}
