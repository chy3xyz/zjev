const std = @import("std");
const alloc = @import("../core/alloc.zig");
const api = @import("onnx_api.zig");
const state = @import("../core/state.zig");
const schema = @import("../core/schema.zig");
const encoder = @import("encoder.zig");
const head = @import("head.zig");
const logits = @import("logits.zig");
const factory = @import("factory.zig");
const build_options = @import("build_options");

pub const Error = error{
    Unsupported,
    OrtInitFailed,
    SessionCreateFailed,
    RunFailed,
    BadModelIO,
    OutOfMemory,
};

const Session = struct {
    handle: *api.OrtSession,
    total_logits: usize,
};

const Onnx = struct {
    a: alloc.Allocator,
    ort: *const api.OrtApi,
    env: *api.OrtEnv,
    allocator: *api.OrtAllocator,
    sessions: []*Session,
    idle: std.Io.Queue(*Session),
    idle_storage: []*Session,
    io: std.Io,
    in_name: [*:0]const u8,
    out_name: [*:0]const u8,
};

const OnnxHidden = struct {
    onnx: *Onnx,
    session: *Session,
    value: *api.OrtValue,
    data: []f32,
};

fn checkStatus(ort: *const api.OrtApi, st: ?*api.OrtStatus, err: Error) Error!void {
    if (st) |s| {
        std.log.err("onnxruntime: {s}", .{ort.GetErrorMessage(s)});
        ort.ReleaseStatus(s);
        return err;
    }
}

fn createEnv(ort: *const api.OrtApi) Error!*api.OrtEnv {
    var env: ?*api.OrtEnv = null;
    try checkStatus(ort, ort.CreateEnv(.warning, "zjev", &env), error.OrtInitFailed);
    return env.?;
}

fn createSession(ort: *const api.OrtApi, a: alloc.Allocator, env: *api.OrtEnv, path: [:0]const u8, extensions_path: ?[]const u8) Error!*api.OrtSession {
    var opts: ?*api.OrtSessionOptions = null;
    try checkStatus(ort, ort.CreateSessionOptions(&opts), error.SessionCreateFailed);
    defer ort.ReleaseSessionOptions(opts);
    try checkStatus(ort, ort.SetIntraOpNumThreads(opts, 1), error.SessionCreateFailed);
    try checkStatus(ort, ort.SetSessionLogSeverityLevel(opts, 3), error.SessionCreateFailed);
    if (extensions_path) |ext| {
        const zext = try a.dupeSentinel(u8, ext, 0);
        defer a.free(zext);
        var lib_handle: ?*anyopaque = null;
        checkStatus(ort, ort.RegisterCustomOpsLibrary(opts, zext.ptr, &lib_handle), error.OrtInitFailed) catch |err| {
            std.log.err("RegisterCustomOpsLibrary failed for '{s}': check version match with onnxruntime", .{ext});
            return err;
        };
    }
    var sess: ?*api.OrtSession = null;
    try checkStatus(ort, ort.CreateSession(env, path, opts, &sess), error.SessionCreateFailed);
    return sess.?;
}

fn outputLogitCount(ort: *const api.OrtApi, sess: *api.OrtSession) Error!usize {
    var out_count: usize = 0;
    try checkStatus(ort, ort.SessionGetOutputCount(sess, &out_count), error.BadModelIO);
    if (out_count != 1) return error.BadModelIO;
    var type_info: ?*api.OrtTypeInfo = null;
    try checkStatus(ort, ort.SessionGetOutputTypeInfo(sess, 0, &type_info), error.BadModelIO);
    defer ort.ReleaseTypeInfo(type_info);
    var tensor_info: ?*const api.OrtTensorTypeAndShapeInfo = null;
    try checkStatus(ort, ort.CastTypeInfoToTensorInfo(type_info, &tensor_info), error.BadModelIO);
    // tensor_info is borrowed from type_info (ORT C API): do NOT release it.
    var count: usize = 0;
    try checkStatus(ort, ort.GetTensorShapeElementCount(tensor_info, &count), error.BadModelIO);
    return count;
}

fn nameIs(ort: *const api.OrtApi, allocator: *api.OrtAllocator, sess: *api.OrtSession, index: usize, input: bool, expect: []const u8) Error!void {
    var name_ptr: [*:0]u8 = undefined;
    const st = if (input)
        ort.SessionGetInputName(sess, index, allocator, @ptrCast(&name_ptr))
    else
        ort.SessionGetOutputName(sess, index, allocator, @ptrCast(&name_ptr));
    try checkStatus(ort, st, error.BadModelIO);
    defer _ = ort.AllocatorFree(allocator, name_ptr);
    const got = std.mem.span(name_ptr);
    if (!std.mem.eql(u8, got, expect)) return error.BadModelIO;
}

fn encodeImpl(ptr: *anyopaque, a: alloc.Allocator, s: *const state.State) encoder.Error!*encoder.HiddenState {
    const self: *Onnx = @ptrCast(@alignCast(ptr));
    const text = s.text orelse s.id orelse "";
    const ztext = a.dupeSentinel(u8, text, 0) catch return error.OutOfMemory;

    const session = self.idle.getOne(self.io) catch return error.ModelFailed;
    errdefer self.idle.putOneUncancelable(self.io, session) catch {};

    var input_value: ?*api.OrtValue = null;
    const dims = [1]i64{1};
    checkStatus(self.ort, self.ort.CreateTensorAsOrtValue(self.allocator, &dims, 1, .string, &input_value), error.RunFailed) catch return error.ModelFailed;
    defer self.ort.ReleaseValue(input_value);
    var strs = [1][*:0]const u8{ztext.ptr};
    checkStatus(self.ort, self.ort.FillStringTensor(input_value, &strs, 1), error.RunFailed) catch return error.ModelFailed;

    var output_value: ?*api.OrtValue = null;
    var in_names = [1][*:0]const u8{self.in_name};
    var out_names = [1][*:0]const u8{self.out_name};
    var inputs = [1]?*const api.OrtValue{input_value};
    checkStatus(self.ort, self.ort.Run(session.handle, null, &in_names, &inputs, 1, &out_names, 1, @ptrCast(&output_value)), error.RunFailed) catch return error.ModelFailed;

    var data_ptr: ?*anyopaque = null;
    checkStatus(self.ort, self.ort.GetTensorMutableData(output_value, @ptrCast(&data_ptr)), error.RunFailed) catch {
        self.ort.ReleaseValue(output_value);
        return error.ModelFailed;
    };
    const n = session.total_logits;
    const raw: [*]f32 = @ptrCast(@alignCast(data_ptr.?));
    const copied = a.dupe(f32, raw[0..n]) catch {
        self.ort.ReleaseValue(output_value);
        return error.OutOfMemory;
    };

    const h = a.create(OnnxHidden) catch {
        self.ort.ReleaseValue(output_value);
        return error.OutOfMemory;
    };
    h.* = .{
        .onnx = self,
        .session = session,
        .value = output_value.?,
        .data = copied,
    };
    return @ptrCast(h);
}

fn deinitImpl(ptr: *anyopaque, a: alloc.Allocator, h: *encoder.HiddenState) void {
    _ = ptr;
    _ = a;
    const hidden: *OnnxHidden = @ptrCast(@alignCast(h));
    hidden.onnx.ort.ReleaseValue(hidden.value);
    hidden.onnx.idle.putOneUncancelable(hidden.onnx.io, hidden.session) catch {};
    // note: OnnxHidden itself lives in the request arena; no explicit destroy
}

const encoder_vtable: encoder.VTable = .{ .encode = encodeImpl, .deinit = deinitImpl };

fn decideImpl(
    ptr: *anyopaque,
    a: alloc.Allocator,
    hidden: *encoder.HiddenState,
    schemas: []const schema.DecisionSchema,
) head.Error![]f32 {
    _ = ptr;
    const h: *OnnxHidden = @ptrCast(@alignCast(hidden));
    var total: usize = 0;
    for (schemas) |s| total += logits.logitCount(s);
    if (total != h.session.total_logits) return error.BadModelIO;
    return a.dupe(f32, h.data) catch return error.OutOfMemory;
}

const head_vtable: head.VTable = .{ .decide = decideImpl };

fn modelDeinit(ptr: *anyopaque, a: alloc.Allocator) void {
    const self: *Onnx = @ptrCast(@alignCast(ptr));
    for (self.sessions) |s| self.ort.ReleaseSession(s.handle);
    self.ort.ReleaseEnv(self.env);
    a.free(self.sessions);
    a.free(self.idle_storage);
    a.destroy(self);
}

pub fn openOnnx(a: alloc.Allocator, io: std.Io, model_path: []const u8, num_sessions: u16, extensions_path: ?[]const u8) Error!factory.Model {
    if (!build_options.onnx) return error.Unsupported;

    const base = api.OrtGetApiBase();
    const ort = base.GetApi(api.ORT_API_VERSION) orelse return error.OrtInitFailed;
    const env = try createEnv(ort);

    if (extensions_path) |ext| {
        std.Io.Dir.cwd().access(io, ext, .{}) catch {
            std.log.err("--ort-extensions '{s}' not found", .{ext});
            return error.OrtInitFailed;
        };
    }

    var allocator: ?*api.OrtAllocator = null;
    try checkStatus(ort, ort.GetAllocatorWithDefaultOptions(&allocator), error.OrtInitFailed);

    const zpath = try a.dupeSentinel(u8, model_path, 0);
    defer a.free(zpath);
    const count: usize = if (num_sessions == 0)
        @max(1, (std.Thread.getCpuCount() catch 4) / 2)
    else
        num_sessions;

    const sessions = try a.alloc(*Session, count);
    errdefer a.free(sessions);

    var total_logits: usize = 0;
    for (sessions, 0..) |*sp, i| {
        const handle = try createSession(ort, a, env, zpath, extensions_path);
        const n = try outputLogitCount(ort, handle);
        if (i == 0) {
            total_logits = n;
            var in_count: usize = 0;
            try checkStatus(ort, ort.SessionGetInputCount(handle, &in_count), error.BadModelIO);
            if (in_count != 1) return error.BadModelIO;
            try nameIs(ort, allocator.?, handle, 0, true, "text");
            try nameIs(ort, allocator.?, handle, 0, false, "logits");
        } else if (n != total_logits) {
            return error.BadModelIO;
        }
        sp.* = try a.create(Session);
        sp.*.* = .{ .handle = handle, .total_logits = n };
    }

    const self = try a.create(Onnx);
    const idle_storage = try a.alloc(*Session, count);
    self.* = .{
        .a = a,
        .ort = ort,
        .env = env,
        .allocator = allocator.?,
        .sessions = sessions,
        .idle = std.Io.Queue(*Session).init(idle_storage),
        .idle_storage = idle_storage,
        .io = io,
        .in_name = "text",
        .out_name = "logits",
    };
    for (sessions) |s| self.idle.putOneUncancelable(io, s) catch {};

    return .{
        .ptr = self,
        .deinitFn = modelDeinit,
        .encoder = .{ .ptr = self, .vtable = &encoder_vtable },
        .heads = .initFill(.{ .ptr = self, .vtable = &head_vtable, .bundled = true }),
    };
}

test "ort api base" {
    if (!build_options.onnx) return;
    const base = api.OrtGetApiBase();
    try std.testing.expect(base.GetApi(api.ORT_API_VERSION) != null);
}
