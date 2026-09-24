const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const onnx = b.option(bool, "onnx", "Enable ONNX Runtime backend") orelse false;
    const mock_mode = b.option([]const u8, "mock_mode", "uniform|peaked|sequence") orelse "peaked";

    const options = b.addOptions();
    options.addOption(bool, "onnx", onnx);
    options.addOption([]const u8, "mock_mode", mock_mode);

    const lib_module = b.createModule(.{
        .root_source_file = b.path("src/zjev.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_module.addOptions("build_options", options);
    if (onnx) lib_module.linkSystemLibrary("onnxruntime", .{});

    const lib = b.addLibrary(.{ .name = "zjev", .root_module = lib_module });
    b.installArtifact(lib);

    const exe_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "zjev", .module = lib_module }},
    });
    const exe = b.addExecutable(.{ .name = "zjev-serve", .root_module = exe_module });
    b.installArtifact(exe);

    const test_module = b.createModule(.{
        .root_source_file = b.path("src/zjev.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_module.addOptions("build_options", options);
    const unit_tests = b.addTest(.{ .root_module = test_module });
    const run_unit = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit.step);

    const conf_module = b.createModule(.{
        .root_source_file = b.path("tools/conformance.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "zjev", .module = lib_module }},
    });
    const conf_exe = b.addExecutable(.{ .name = "zjev-conformance", .root_module = conf_module });
    const run_conf = b.addRunArtifact(conf_exe);
    const conf_step = b.step("test-conformance", "Run conformance fixtures");
    conf_step.dependOn(&run_conf.step);

    const tools = .{
        .{ .name = "zjev-fit", .root = "tools/fit.zig" },
        .{ .name = "zjev-bench", .root = "tools/bench.zig" },
        .{ .name = "zjev-traj", .root = "tools/traj.zig" },
    };
    inline for (tools) |t| {
        const tool_module = b.createModule(.{
            .root_source_file = b.path(t.root),
            .target = target,
            .optimize = optimize,
            .imports = &.{ .{ .name = "zjev", .module = lib_module }, .{ .name = "dataset", .module = b.createModule(.{
                .root_source_file = b.path("tools/dataset.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{.{ .name = "zjev", .module = lib_module }},
            }) } },
        });
        const tool_exe = b.addExecutable(.{ .name = t.name, .root_module = tool_module });
        b.installArtifact(tool_exe);
    }
}
