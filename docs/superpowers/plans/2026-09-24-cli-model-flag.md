# CLI `--model`/`--sessions` 接线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `zjev-serve` 通过 `--model <path.onnx>` 接入真模型（ONNX 图），不带则维持 mock；`--sessions` 控制 onnxruntime 会话数；`model_name` 从模型文件名推导。

**Architecture:** `src/main.zig` 新增纯函数 `parseCli`（可单测）解析全部 CLI 参数；模型打开统一走 `zjev.factory.open`（mock/onnx 两分支已存在）；onnx 未启用或打开失败时启动即报错退出。内嵌测试放 main.zig，build.zig 增加 main 测试模块挂到 `zig build test`。

**Tech Stack:** Zig 0.17.0-dev.2151（zigup 工具链）；现有 `src/model/factory.zig` Config/onnx 分支；build_options `-Donnx`（默认 false）。

## Global Constraints

- 工具链：`zig 0.17.0-dev.2151+2ec5523d5`，**只允许 0.17 API**（无旧 std API）。
- 本机**未装 onnxruntime**：`-Donnx=true` 链接失败是预期，README 已说明；本轮只验证「默认构建（onnx=false）编译过 + 参数解析单测 + onnx=false 时 `--model` 清晰报错」。
- 测试惯例：命令**重定向到文件再查 `$?`**（管道会掩盖退出码）。
- TDD：红 → 绿 → commit；分支 `feat/cli-model` → `--no-ff` 合并 main → 删分支；每任务一 commit。
- 范围：工具（fit/bench/traj）本轮**不加** `--model`。
- 0.17 已知坑：const 字面量 `&.{}` 只配 `[]const` 切片参数；文件级声明与参数名不得同名（shadow error）。

---

### Task 1: `parseCli` 纯函数 + main.zig 测试接线

**Files:**
- Modify: `src/main.zig:8-36`（内联解析循环 → `Cli` struct + `parseCli`）
- Modify: `build.zig:25-32`（exe 模块）、`build.zig:34-43`（test step）
- Test: `src/main.zig` 内嵌 `test` 块（新测试模块挂进 `zig build test`）

**Interfaces:**
- Produces:
  - `pub const Cli` — 字段：`bind: []const u8`、`port: u16`、`mock_mode: []const u8`、`profiles_dir: ?[]const u8`、`use_scheduler: bool`、`cache_enabled: bool`、`model_path: ?[]const u8`、`num_sessions: u16`（全部有默认值）
  - `pub fn parseCli(args: []const []const u8) error{ InvalidPort, InvalidSessions }!Cli`（`args` 含 argv[0]，从 index 1 开始扫）
  - `zig build test` 现在同时跑 `src/zjev.zig` 与 `src/main.zig` 两组测试

- [ ] **Step 1: build.zig 加 main 测试模块（先接线，否则新测试无处运行）**

在 `build.zig` 的 exe 模块创建后（`build.zig:31-33` 的 `const exe = ...` 之后）加：

```zig
    const main_test_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "zjev", .module = lib_module }},
    });
```

> 注意：**不要**给 exe_module / main_test_module 调 `addOptions("build_options", ...)`——0.17 里同一 options 文件属两个模块会报 `file exists in modules` 冲突（lib_module 已持有 build_options）。

把 `build.zig:34-43` 的 test step 改为同时依赖 main 测试：

```zig
    const unit_tests = b.addTest(.{ .root_module = test_module });
    const run_unit = b.addRunArtifact(unit_tests);
    const main_unit_tests = b.addTest(.{ .root_module = main_test_module });
    const run_main_unit = b.addRunArtifact(main_unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit.step);
    test_step.dependOn(&run_main_unit.step);
```

- [ ] **Step 2: 写失败测试（src/main.zig 末尾追加 test 块）**

```zig
test "parseCli defaults" {
    const cli = try parseCli(&.{"zjev-serve"});
    try std.testing.expectEqualStrings("127.0.0.1", cli.bind);
    try std.testing.expectEqual(@as(u16, 9377), cli.port);
    try std.testing.expectEqualStrings("peaked", cli.mock_mode);
    try std.testing.expectEqual(@as(?[]const u8, null), cli.model_path);
    try std.testing.expectEqual(@as(u16, 0), cli.num_sessions);
    try std.testing.expect(!cli.use_scheduler);
    try std.testing.expect(!cli.cache_enabled);
}

test "parseCli model and sessions" {
    const cli = try parseCli(&.{
        "zjev-serve", "--model", "/models/zjev-v1.onnx", "--sessions", "8", "--port", "18080",
    });
    try std.testing.expectEqualStrings("/models/zjev-v1.onnx", cli.model_path.?);
    try std.testing.expectEqual(@as(u16, 8), cli.num_sessions);
    try std.testing.expectEqual(@as(u16, 18080), cli.port);
}

test "parseCli invalid port" {
    try std.testing.expectError(error.InvalidPort, parseCli(&.{ "zjev-serve", "--port", "abc" }));
}

test "parseCli invalid sessions" {
    try std.testing.expectError(error.InvalidSessions, parseCli(&.{ "zjev-serve", "--sessions", "x" }));
}
```

- [ ] **Step 3: 运行确认红**

```bash
zig build test > /tmp/zjev-t1-red.log 2>&1; echo "exit=$?"
```
Expected: exit≠0，编译错误 `use of undeclared identifier 'parseCli'`（红，符合预期）。

- [ ] **Step 4: 实现 parseCli**

把 `src/main.zig:8-36` 替换为：

```zig
pub const Cli = struct {
    bind: []const u8 = "127.0.0.1",
    port: u16 = 9377,
    mock_mode: []const u8 = "peaked",
    profiles_dir: ?[]const u8 = null,
    use_scheduler: bool = false,
    cache_enabled: bool = false,
    model_path: ?[]const u8 = null,
    num_sessions: u16 = 0,
};

pub fn parseCli(args: []const []const u8) error{ InvalidPort, InvalidSessions }!Cli {
    var cli = Cli{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--bind") and i + 1 < args.len) {
            i += 1;
            cli.bind = args[i];
        } else if (std.mem.eql(u8, arg, "--port") and i + 1 < args.len) {
            i += 1;
            cli.port = std.fmt.parseInt(u16, args[i], 10) catch return error.InvalidPort;
        } else if (std.mem.eql(u8, arg, "--mock-mode") and i + 1 < args.len) {
            i += 1;
            cli.mock_mode = args[i];
        } else if (std.mem.eql(u8, arg, "--profiles-dir") and i + 1 < args.len) {
            i += 1;
            cli.profiles_dir = args[i];
        } else if (std.mem.eql(u8, arg, "--model") and i + 1 < args.len) {
            i += 1;
            cli.model_path = args[i];
        } else if (std.mem.eql(u8, arg, "--sessions") and i + 1 < args.len) {
            i += 1;
            cli.num_sessions = std.fmt.parseInt(u16, args[i], 10) catch return error.InvalidSessions;
        } else if (std.mem.eql(u8, arg, "--scheduler")) {
            cli.use_scheduler = true;
        } else if (std.mem.eql(u8, arg, "--cache")) {
            cli.cache_enabled = true;
        } else {
            std.log.warn("unknown arg: {s}", .{arg});
        }
    }
    return cli;
}
```

并把 `main` 里 `var bind`/`var port` 等局部变量换成 `const cli = parseCli(args) catch |err| { std.log.err("invalid arguments: {s}", .{@errorName(err)}); return err; };`，`main` 其余引用改 `cli.bind` / `cli.port` / `cli.mock_mode` / `cli.profiles_dir` / `cli.use_scheduler` / `cli.cache_enabled`。

- [ ] **Step 5: 运行确认绿**

```bash
zig build test > /tmp/zjev-t1-green.log 2>&1; echo "exit=$?"
```
Expected: exit=0，原 109 例 + 新增 4 例全过。

- [ ] **Step 6: Commit**

```bash
git add src/main.zig build.zig
git commit -m "feat(cli): parseCli pure fn with --model/--sessions + main tests wired"
```

---

### Task 2: factory.open 接线 + modelNameFromPath + 启动报错

**Files:**
- Modify: `src/main.zig:38-68`（mode 计算后至 shared 构造前）
- Test: `src/main.zig` 内嵌 test 块

**Interfaces:**
- Consumes: Task 1 的 `Cli`（`model_path`、`num_sessions`）；`zjev.factory.Config{ kind, mock_mode, model_path, num_sessions }`（`src/model/factory.zig:22-27`）；`zjev.factory.open(a, io, cfg) !Model`（factory.zig:29）
- Produces:
  - `pub fn modelNameFromPath(path: []const u8) []const u8` — 取最后 `/` 后基名，去掉最后一个 `.` 及扩展名；无 `/`、无扩展名、空串均安全
  - main 启动路径：onnx=false 时 `--model` → log 清晰报错 + `error.Unsupported`；open 失败 → log 路径与错误名后退出
  - `shared.model_name` = 文件名去扩展名（onnx）或 `"mock"`

- [ ] **Step 1: 写失败测试（src/main.zig test 块追加）**

```zig
test "modelNameFromPath" {
    try std.testing.expectEqualStrings("zjev-v1", modelNameFromPath("/models/zjev-v1.onnx"));
    try std.testing.expectEqualStrings("laya", modelNameFromPath("laya.onnx"));
    try std.testing.expectEqualStrings("c.tar", modelNameFromPath("/a/b/c.tar.onnx"));
    try std.testing.expectEqualStrings("noext", modelNameFromPath("/x/noext"));
    try std.testing.expectEqualStrings("", modelNameFromPath(""));
}
```

- [ ] **Step 2: 运行确认红**

```bash
zig build test > /tmp/zjev-t2-red.log 2>&1; echo "exit=$?"
```
Expected: exit≠0，`undeclared identifier 'modelNameFromPath'`。

- [ ] **Step 3: 实现接线**

`src/main.zig` 的 `parseCli` 后追加：

```zig
pub fn modelNameFromPath(path: []const u8) []const u8 {
    const base = if (std.mem.lastIndexOfScalar(u8, path, '/')) |idx| path[idx + 1 ..] else path;
    if (std.mem.lastIndexOfScalar(u8, base, '.')) |dot| return base[0..dot];
    return base;
}
```

把 `src/main.zig:45-50`（threaded 初始化到 `var model = try zjev.mock.model(mode, gpa);`）替换为：

```zig
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const tio = threaded.io();

    var model = zjev.factory.open(gpa, tio, .{
        .kind = if (cli.model_path != null) .onnx else .mock,
        .mock_mode = mode,
        .model_path = cli.model_path,
        .num_sessions = cli.num_sessions,
    }) catch |err| {
        if (err == error.Unsupported) {
            std.log.err("--model requires onnx support; rebuild with: zig build -Donnx=true", .{});
        } else {
            std.log.err("failed to open model '{s}': {s}", .{ cli.model_path orelse "mock", @errorName(err) });
        }
        return err;
    };
    defer model.deinit(gpa);
```

`shared` 构造处（原 `:68`）改 `.model_name = if (cli.model_path) |p| modelNameFromPath(p) else "mock"`。

注意：main.zig **不**直接 `@import("build_options")`（0.17 模块文件唯一性约束，lib_module 已持有）；onnx=false 的提示改在 catch 里识别 `error.Unsupported` 输出（factory.open 的 onnx 分支在未启用时返回它）。

- [ ] **Step 4: 运行确认绿**

```bash
zig build test > /tmp/zjev-t2-test.log 2>&1; echo "exit=$?"
```
Expected: exit=0。

- [ ] **Step 5: 冒烟验证**

```bash
zig build > /tmp/zjev-t2-build.log 2>&1; echo "build=$?"
(./zig-out/bin/zjev-serve --port 18080 > /tmp/zjev-t2-serve.log 2>&1 &) ; sleep 0.8
curl -s -X POST http://127.0.0.1:18080/v1/execute -H 'Content-Type: application/json' -d '{"state":"s","questions":{"q":{"type":"noul"}}}' | head -c 200; echo
pkill -f "zjev-serve --port 18080"
./zig-out/bin/zjev-serve --model /tmp/none.onnx --port 18099 > /tmp/zjev-t2-err.log 2>&1; echo "model_exit=$?"
grep -o "requires onnx support" /tmp/zjev-t2-err.log
./zig-out/bin/zjev-serve --port abc > /tmp/zjev-t2-badport.log 2>&1; echo "badport_exit=$?"
```
Expected: build=0；mock 冒烟正常返回 JSON；model_exit≠0 且 grep 命中；badport_exit≠0。

- [ ] **Step 6: ReleaseSafe 编译 + commit**

```bash
zig build -Doptimize=ReleaseSafe > /tmp/zjev-t2-rel.log 2>&1; echo "rel=$?"
git add src/main.zig
git commit -m "feat(cli): wire --model/--sessions to factory.open with startup errors"
```
Expected: rel=0。

---

### Task 3: README + 全量验证 + 合并

**Files:**
- Modify: `README.md:25`（CLI 选项行）、`README.md:10` 附近（构建说明）

**Interfaces:**
- Consumes: Task 1/2 的行为（flag 名 `--model` / `--sessions`、报错文案）

- [ ] **Step 1: README 更新**

`README.md:25` 改为：

```markdown
CLI 选项：`--bind` `--port` `--mock-mode uniform|peaked|sequence` `--profiles-dir <dir>` `--scheduler` `--cache` `--model <path.onnx>` `--sessions <n>`。不带 `--model` 使用 mock；带 `--model` 走 ONNX 后端（需 `-Donnx=true` 构建且本机装 onnxruntime），`model_name` 取文件名去扩展名，`--sessions` 为 onnxruntime 会话数（0=默认）。
```

`README.md:10` 构建说明行后追加一段：

```markdown
真模型：`zig build -Donnx=true`（需 onnxruntime 动态库）后 `zjev-out/bin/zjev-serve --model model/zjev-v1.onnx --sessions 4`。导出约定见下文「模型导出约定」。
```

（若行号有轻微漂移，以内容定位为准。）

- [ ] **Step 2: 全量验证**

```bash
zig build test > /tmp/zjev-t3-test.log 2>&1; echo "test=$?"
zig build test-conformance > /tmp/zjev-t3-conf.log 2>&1; echo "conf=$?"
tail -3 /tmp/zjev-t3-conf.log
```
Expected: test=0；conf=0 且输出 10 pass 0 fail。

- [ ] **Step 3: Commit + 合并**

> 执行约定：进入 Task 1 前先 `git checkout -b feat/cli-model`，三个 Task 的 commit 都落在该分支，此处只做合并。

```bash
git add README.md
git commit -m "docs: document --model/--sessions CLI"
git checkout main
git merge --no-ff feat/cli-model -m "merge: CLI --model/--sessions wiring"
git branch -d feat/cli-model
```
Expected: 合并后 `git log --oneline -5` 显示 merge commit。

- [ ] **Step 4: 合并后复验**

```bash
zig build test > /tmp/zjev-t3-final.log 2>&1; echo "final=$?"
```
Expected: final=0。

---

## Self-Review

**1. Spec coverage:** 用户确认的设计四点——`--model` 接线（Task 2）、`--sessions`（Task 1+2）、纯函数可单测（Task 1）、model_name 文件名去扩展名（Task 2）、启动报错（Task 2 Step 5 验证）、工具不加 flag（Global Constraints）。全覆盖。

**2. Placeholder scan:** 所有代码块均为完整可贴代码；README 行号标注「以内容定位为准」防漂移。无 TBD。

**3. Type consistency:** `Cli` 字段名在 Task 1 定义、Task 2/3 引用一致（`model_path`/`num_sessions`）；`zjev.factory.Config` 字段名与 `src/model/factory.zig:22-27` 核对一致（`kind/mock_mode/model_path/num_sessions`）；`modelNameFromPath` 签名两处一致。`error{ InvalidPort, InvalidSessions }` 与测试断言一致。
