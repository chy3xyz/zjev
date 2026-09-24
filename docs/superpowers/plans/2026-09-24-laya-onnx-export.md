# Laya 导出 + 本机 onnxruntime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 本机装好 onnxruntime，Laya 英文 checkpoint 导出为符合 ZJEV 契约的 `laya.onnx`（图内 tokenizer + 固定 8-logit 决策束 + 随机 head），`zjev-serve --model` 端到端跑通 `/v1/execute`。

**Architecture:** 三层——(1) brew onnxruntime 1.30 + `build.zig` 库路径选项让 `-Donnx=true` 链接通过；(2) Zig 运行时加 `--ort-extensions`，`createSession` 前调 `RegisterCustomOpsLibrary`（extern 已在 `onnx_api.zig:100`）；(3) `export/laya/` Python 脚本：HF tokenizer → extensions 处理图，ModernBERT encoder（torch.onnx 导出）→ compose → 拼 [CLS]+随机 Linear head → 静态 `logits[1,8]`。

**Tech Stack:** Zig 0.17.0-dev.2151；onnxruntime 1.30（brew）；Python 3.12 venv（torch / transformers≥4.48 / onnx / onnxscript / onnxruntime-extensions / huggingface_hub）；onnxruntime-extensions 预编译 osx-arm64 dylib。

## Global Constraints

- 工具链：`zig 0.17.0-dev.2151+2ec5523d5`，**只允许 0.17 API**。
- 导出约定不变（spec）：输入 string tensor `text` [1]；输出 float tensor `logits` **静态** shape [1,8]；图内包含 tokenizer。
- 固定决策束（schema 主序）：escalate-noul(2) + topic-choice[billing,bug,other](3) + urgency-score[lo,mid,hi](3) = 8 logits。
- v0 = 纯 plumbing：head 随机初始化（numpy seed=42，scale 0.02），概率质量无意义，汇报明示。
- 测试惯例：命令**重定向到文件再查 `$?`**；TDD 红绿；分支 `feat/laya-onnx` → `--no-ff` 合并 → 删分支；每 Task 一 commit。
- git 只 add 明确路径（`zig-out/` 产物被跟踪是惯例，不提交它们的本地变动）。
- 执行进入 Task 1 前先 `git checkout -b feat/laya-onnx`。
- Python 一律用 `export/laya/.venv`，不污染 pyenv 全局。

---

### Task 1: brew onnxruntime + build.zig `-Donnx-lib-dir`

**Files:**
- Modify: `build.zig:7`（option 声明）、`build.zig:20-21`（onnx 块）

**Interfaces:**
- Produces: build 选项 `-Donnx-lib-dir <path>`（默认 `/opt/homebrew/lib`）；`-Donnx=true` 时库路径与 rpath 生效；`zig build -Donnx=true test` 全绿（含真实执行 `test "ort api base"`，onnx.zig:244）

- [ ] **Step 1: 创建分支**

```bash
git checkout -b feat/laya-onnx
```

- [ ] **Step 2: brew 安装**

```bash
brew install onnxruntime > /tmp/zjev-laya-brew.log 2>&1; echo "brew=$?"
ls /opt/homebrew/lib/libonnxruntime* >> /tmp/zjev-laya-brew.log 2>&1
```
Expected: brew=0，存在 `/opt/homebrew/lib/libonnxruntime.1.30*.dylib`。

- [ ] **Step 3: build.zig 加选项与库路径**

`build.zig:7` 后加（紧挨 `onnx` option 声明）：

```zig
    const onnx_lib_dir = b.option([]const u8, "onnx_lib_dir", "Directory containing onnxruntime lib") orelse "/opt/homebrew/lib";
```

`build.zig:20-21` 的 `if (onnx) lib_module.linkSystemLibrary(...)` 块改为：

```zig
    if (onnx) {
        lib_module.linkSystemLibrary("onnxruntime", .{});
        lib_module.addLibraryPath(.{ .cwd_relative = onnx_lib_dir });
        lib_module.addRPathOnce(.{ .cwd_relative = onnx_lib_dir });
    }
```

- [ ] **Step 4: 编译 + 链接 + 单测（此即本 Task 的测试，无可红阶段——成功=编译链接通过且 ORT API 真实调用成功）**

```bash
zig build -Donnx=true > /tmp/zjev-laya-t1.log 2>&1; echo "build=$?"
zig build -Donnx=true test > /tmp/zjev-laya-t1-test.log 2>&1; echo "test=$?"
```
Expected: 均 0。`test "ort api base"`（onnx.zig:244-247）在 onnx=true 时真实调 `OrtGetApiBase().GetApi(22)`。
若 addLibraryPath/addRPathOnce 编译报错（0.17 API 变动），fallback：去掉 rpath 行，运行时 `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib zig build -Donnx=true test`，并把该 env 要求写进 README。

- [ ] **Step 5: Commit**

```bash
git add build.zig
git commit -m "build: -Donnx-lib-dir option for local onnxruntime linking"
```

---

### Task 2: `--ort-extensions` 运行时接线

**Files:**
- Modify: `src/main.zig`（Cli、parseCli、open 调用处）、`src/model/factory.zig:22-37`、`src/model/onnx.zig:61-70,182-217`

**Interfaces:**
- Consumes: Task 1 的链接环境
- Produces:
  - `Cli.ort_extensions: ?[]const u8 = null`（默认 null）；`parseCli` 支持 `--ort-extensions <path>`
  - `factory.Config.ort_extensions: ?[]const u8 = null`；`open()` 透传给 `openOnnx`
  - `openOnnx(a, io, model_path, num_sessions, extensions_path: ?[]const u8) Error!factory.Model`
  - `createSession(ort, env, path, a, extensions_path)`：options 创建后、`CreateSession` 前注册扩展库；失败 `error.OrtInitFailed` + 日志（含 onnxruntime 错误信息 + 版本匹配提示）

- [ ] **Step 1: 写失败测试（src/main.zig test 块追加）**

```zig
test "parseCli ort extensions" {
    const cli = try parseCli(&.{ "zjev-serve", "--ort-extensions", "/tmp/liboe.dylib" });
    try std.testing.expectEqualStrings("/tmp/liboe.dylib", cli.ort_extensions.?);
    const d = try parseCli(&.{"zjev-serve"});
    try std.testing.expectEqual(@as(?[]const u8, null), d.ort_extensions);
}
```

- [ ] **Step 2: 运行确认红**

```bash
zig build test > /tmp/zjev-laya-t2-red.log 2>&1; echo "exit=$?"
```
Expected: exit≠0，`no field named 'ort_extensions' in struct 'Cli'` 或 `use of undeclared identifier`（红）。

- [ ] **Step 3: 实现三处接线**

`src/main.zig` `Cli` struct 加字段（`num_sessions` 后）：

```zig
    ort_extensions: ?[]const u8 = null,
```

`parseCli` 的 `--sessions` 分支后加：

```zig
        } else if (std.mem.eql(u8, arg, "--ort-extensions") and i + 1 < args.len) {
            i += 1;
            cli.ort_extensions = args[i];
```

main 里 `zjev.factory.open` 调用的 cfg 字面量加一行（`num_sessions` 后）：

```zig
        .ort_extensions = cli.ort_extensions,
```

`src/model/factory.zig` `Config` 加字段（`num_sessions` 后）：

```zig
    ort_extensions: ?[]const u8 = null,
```

`factory.zig` open 的 onnx 分支改为：

```zig
        .onnx => blk: {
            if (!@import("build_options").onnx) return error.Unsupported;
            break :blk @import("onnx.zig").openOnnx(a, io, cfg.model_path orelse return error.MissingModelPath, cfg.num_sessions, cfg.ort_extensions);
        },
```

`src/model/onnx.zig` `createSession` 改为（注册必须在 `CreateSession` 之前）：

```zig
fn createSession(ort: *const api.OrtApi, a: alloc.Allocator, env: *api.OrtEnv, path: [:0]const u8, extensions_path: ?[]const u8) Error!*api.OrtSession {
    var opts: ?*api.OrtSessionOptions = null;
    try checkStatus(ort, ort.CreateSessionOptions(&opts), error.SessionCreateFailed);
    defer ort.ReleaseSessionOptions(opts);
    try checkStatus(ort, ort.SetIntraOpNumThreads(opts, 1), error.SessionCreateFailed);
    try checkStatus(ort, ort.SetSessionLogSeverityLevel(opts, 3), error.SessionCreateFailed);
    if (extensions_path) |ext| {
        const zext = try a.dupeZ(u8, ext);
        defer a.free(zext);
        checkStatus(ort, ort.RegisterCustomOpsLibrary(opts, zext.ptr, null), error.OrtInitFailed) catch |err| {
            std.log.err("RegisterCustomOpsLibrary failed for '{s}': check version match with onnxruntime", .{ext});
            return err;
        };
    }
    var sess: ?*api.OrtSession = null;
    try checkStatus(ort, ort.CreateSession(env, path, opts, &sess), error.SessionCreateFailed);
    return sess.?;
}
```

`openOnnx` 签名改：

```zig
pub fn openOnnx(a: alloc.Allocator, io: std.Io, model_path: []const u8, num_sessions: u16, extensions_path: ?[]const u8) Error!factory.Model {
```

`openOnnx` 内 `createSession(ort, env, zpath)` 调用改 `createSession(ort, a, env, zpath, extensions_path)`。

- [ ] **Step 4: 运行确认绿（onnx=false 与 onnx=true 各一遍）**

```bash
zig build test > /tmp/zjev-laya-t2-test.log 2>&1; echo "t0=$?"
zig build -Donnx=true test > /tmp/zjev-laya-t2-test-onnx.log 2>&1; echo "t1=$?"
```
Expected: 均 0。

- [ ] **Step 5: Commit**

```bash
git add src/main.zig src/model/factory.zig src/model/onnx.zig
git commit -m "feat(onnx): register custom ops library via --ort-extensions"
```

---

### Task 3: export/laya 环境 + libortextensions 下载 + HF 侦察

**Files:**
- Create: `export/laya/requirements.txt`、`.gitignore`（追加）
- 环境：`export/laya/.venv/`、`export/laya/lib/`、`export/laya/out/`（gitignore）

**Interfaces:**
- Produces: 可用的 `.venv`（torch/transformers/onnx 全家桶）；`export/laya/lib/libortextensions*.dylib`；HF snapshot 在 `~/.cache/huggingface/`（含英文 checkpoint 路径与 config 结构侦察结论，供 Task 4 使用）

- [ ] **Step 1: 脚手架 + gitignore + requirements**

`export/laya/requirements.txt`：

```text
torch>=2.6
transformers>=4.48
onnx>=1.17
onnxscript>=0.2
onnxruntime>=1.30
onnxruntime-extensions>=0.14
huggingface_hub>=0.30
numpy>=2.0
safetensors>=0.5
```

`.gitignore` 追加：

```gitignore
export/laya/.venv/
export/laya/lib/
export/laya/out/
```

- [ ] **Step 2: 建 venv 并后台安装（重，约 2-3GB）**

```bash
python3 -m venv export/laya/.venv
export/laya/.venv/bin/pip install -U pip > /tmp/zjev-laya-pip.log 2>&1
export/laya/.venv/bin/pip install -r export/laya/requirements.txt >> /tmp/zjev-laya-pip.log 2>&1; echo "pip=$?"
```
Expected: pip=0（torch arm64 wheel 较大，耐心；可 `tail -f /tmp/zjev-laya-pip.log` 观察）。

- [ ] **Step 3: 下载 libortextensions（后台 pip 期间做）**

```bash
mkdir -p export/laya/lib
curl -s https://api.github.com/repos/microsoft/onnxruntime-extensions/releases/latest | grep browser_download_url | grep -iE 'osx|darwin|mac' | grep -i arm64
```
按列出的 osx-arm64 资产 URL 下载到 `export/laya/lib/`，重命名为 `libortextensions.dylib`：

```bash
curl -sL <上一步的 URL> -o export/laya/lib/libortextensions.dylib
ls -la export/laya/lib/
```
Expected: dylib 存在且 >1MB。若 latest 与 ORT 1.30 不兼容（Task 5 e2e 才暴露），fallback：从 releases 列表选与 1.30 同期的版本重下。

- [ ] **Step 4: HF snapshot 下载 + 侦察 checkpoint 结构**

```bash
export/laya/.venv/bin/python - <<'EOF' > /tmp/zjev-laya-recon.log 2>&1
from huggingface_hub import snapshot_download
p = snapshot_download("convaiinnovations/laya",
    allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.md"])
print("SNAPSHOT:", p)
import os, json
for root, _, files in os.walk(p):
    for f in files:
        if f in ("config.json",):
            cfg = json.load(open(os.path.join(root, f)))
            print(os.path.join(root, f), "->",
                  cfg.get("architectures"), cfg.get("model_type"),
                  cfg.get("hidden_size"), cfg.get("num_hidden_layers"))
EOF
echo "recon=$?"; cat /tmp/zjev-laya-recon.log
```
Expected: recon=0。从输出找出 `hidden_size=1024` 且 architectures 含 ModernBERT 的英文 checkpoint 目录（记为 `<EN_DIR>`，供 Task 4）。若 architectures 是自定义类名（非 ModernBERT），记录 `auto_map`/`model_type`，Task 4 走 `trust_remote_code=True` 路径。

- [ ] **Step 5: Commit（脚手架文件）**

```bash
git add export/laya/requirements.txt .gitignore
git commit -m "feat(export): laya export scaffolding + gitignore"
```

---

### Task 4: export_laya.py + smoke_check.py 迭代到绿

**Files:**
- Create: `export/laya/export_laya.py`、`export/laya/smoke_check.py`

**Interfaces:**
- Consumes: Task 3 的 venv、libortextensions、`<EN_DIR>`；决策束常量（本 Task 内定义）
- Produces: `export/laya/out/laya.onnx`（输入 `text` string [1]，输出 `logits` float **静态** [1,8]，图内含 ai.onnx.contrib tokenizer op）；`smoke_check.py` 退出码 0

- [ ] **Step 1: 写 export_laya.py（完整脚本）**

```python
#!/usr/bin/env python3
"""Export Laya English encoder to a ZJEV-contract ONNX graph (v0 plumbing).

Data flow:
  text:string[1] -> BertTokenizer (ai.onnx.contrib) -> input_ids/attention_mask
  -> ModernBERT encoder (Laya English weights) -> [CLS] -> Linear(1024->8, random)
  -> logits:float[1,8] (static)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.compose import merge_models
from transformers import AutoModel, AutoTokenizer
import torch

NUM_LOGITS = 8          # escalate(2) + topic(3) + urgency(3)
MAX_LEN = 512
SEED = 42
HEAD_SCALE = 0.02


def log(*a):
    print("[export_laya]", *a, flush=True)


def load_encoder(en_dir: str):
    """Load the ModernBERT encoder from the Laya English checkpoint."""
    try:
        m = AutoModel.from_pretrained(en_dir)
        log("AutoModel loaded; unused keys are decision-head weights (expected)")
    except Exception as e:  # custom config -> trust_remote_code path
        log("AutoModel failed:", e, "-- trying trust_remote_code")
        m = AutoModel.from_pretrained(en_dir, trust_remote_code=True)
    m.eval()
    return m


def export_encoder(model, out_path: str):
    ids = torch.ones(1, 8, dtype=torch.long)
    mask = torch.ones(1, 8, dtype=torch.long)
    torch.onnx.export(
        model, (ids, mask), out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {1: "seq"},
            "attention_mask": {1: "seq"},
            "last_hidden_state": {1: "seq"},
        },
        opset_version=18,
        dynamo=True,
    )
    log("encoder exported:", out_path)


def tokenizer_graph(tok) -> onnx.ModelProto:
    from onnxruntime_extensions import gen_processing_models
    m = gen_processing_models(tok)
    if isinstance(m, (list, tuple)):
        m = m[0]
    m = onnx.shape_inference.infer_shapes(m)
    assert len(m.graph.input) == 1, f"tokenizer graph inputs: {m.graph.input}"
    old = m.graph.input[0].name
    m.graph.input[0].name = "text"
    for n in m.graph.node:
        for i, s in enumerate(n.input):
            if s == old:
                n.input[i] = "text"
    outs = {o.name for o in m.graph.output}
    assert {"input_ids", "attention_mask"} <= outs, f"tokenizer outputs: {outs}"
    return m


def add_head(merged: onnx.ModelProto, hidden_size: int) -> onnx.ModelProto:
    """[CLS] -> Linear(hidden->NUM_LOGITS) -> logits[1,NUM_LOGITS] (static)."""
    g = merged.graph
    lhs = g.output[0].name            # last_hidden_state [1,seq,H]
    g.output.pop()

    rng = np.random.default_rng(SEED)
    W = (rng.standard_normal((hidden_size, NUM_LOGITS)) * HEAD_SCALE).astype(np.float32)
    b = np.zeros(NUM_LOGITS, dtype=np.float32)

    init = lambda name, arr: g.initializer.append(numpy_helper.from_array(arr, name))
    init("head_W", W)
    init("head_b", b)
    init("cls_idx", np.array([0], dtype=np.int64))
    init("cls_axes", np.array([1], dtype=np.int64))

    g.node.append(helper.make_node("Gather", [lhs, "cls_idx"], ["cls_gather"], axis=1))
    g.node.append(helper.make_node("Squeeze", ["cls_gather", "cls_axes"], ["cls_vec"]))
    g.node.append(helper.make_node("MatMul", ["cls_vec", "head_W"], ["head_mm"]))
    g.node.append(helper.make_node("Add", ["head_mm", "head_b"], ["logits"]))
    g.output.append(helper.make_tensor_value_info(
        "logits", onnx.TensorProto.FLOAT, [1, NUM_LOGITS]))
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en-dir", required=True, help="Laya English checkpoint dir")
    ap.add_argument("--out", default=str(Path(__file__).parent / "out" / "laya.onnx"))
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.en_dir)
    model = load_encoder(a.en_dir)
    hidden = model.config.hidden_size
    assert hidden in (768, 1024), hidden
    log("hidden_size:", hidden)

    tmp_enc = Path(a.out).parent / "_encoder_tmp.onnx"
    export_encoder(model, str(tmp_enc))
    enc = onnx.load(str(tmp_enc))
    tok_g = tokenizer_graph(tok)

    merged = merge_models(tok_g, enc, io_map=[
        ("input_ids", "input_ids"),
        ("attention_mask", "attention_mask"),
    ])
    final = add_head(merged, hidden)

    onnx.checker.check_model(final)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(final, a.out)
    tmp_enc.unlink()
    ins = [i.name for i in final.graph.input]
    outs = [(o.name, list(o.type.tensor_type.shape.dim)) for o in final.graph.output]
    log("saved:", a.out, "inputs:", ins, "outputs:", outs)
    assert ins == ["text"], ins


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 写 smoke_check.py（完整脚本）**

```python
#!/usr/bin/env python3
"""Self-check the exported laya.onnx under onnxruntime + extensions."""
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime_extensions  # noqa: F401  (registers ai.onnx.contrib kernels)

MODEL = Path(__file__).parent / "out" / "laya.onnx"
TEXT = ["We were billed twice for March. Refund the duplicate today or we cancel."]


def main():
    m = onnx.load(str(MODEL))
    ins = [i.name for i in m.graph.input]
    assert ins == ["text"], f"inputs={ins}"
    out = m.graph.output[0]
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    assert out.name == "logits" and dims == [1, 8], f"output={out.name} dims={dims}"

    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(MODEL), so, providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"text": TEXT})[0]
    assert got.shape == (1, 8), got.shape
    assert np.isfinite(got).all(), "non-finite logits"
    got2 = sess.run(["logits"], {"text": TEXT})[0]
    assert np.allclose(got, got2, atol=1e-5), "non-deterministic"
    print("OK logits[0]:", got[0].tolist())


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 跑导出 + 自检，迭代到绿**

```bash
export/laya/.venv/bin/python export/laya/export_laya.py --en-dir <EN_DIR> > /tmp/zjev-laya-export.log 2>&1; echo "export=$?"
export/laya/.venv/bin/python export/laya/smoke_check.py > /tmp/zjev-laya-smoke.log 2>&1; echo "smoke=$?"
tail -20 /tmp/zjev-laya-smoke.log
```
Expected: 均 0，输出 `OK logits[0]: [8 个有限值]`。

已知易碎点与对策（spec 风险 2/4，按日志对症修，每处改动重跑本步）：
- `gen_processing_models` 返回/输入名不符 → 按 assert 报错打印的实际 input/output 名改 `tokenizer_graph` 的重命名与 io_map
- `merge_models` opset 冲突 → encoder 导出的 `opset_version` 提到与 tokenizer 图一致（报错信息会给出两边版本）
- ModernBERT dynamo 导出失败 → `dynamo=False` 回退 legacy 导出
- checkpoint 自定义类（AutoModel 两次失败）→ `--en-dir` 传含 `modeling_*.py` 的 snapshot 目录 + `trust_remote_code=True`
- Squeeze/Gather 与 opset 版本形态不符（18 为 axes 输入）→ 若 onnx.checker 报 op 形态错，把 `add_head` 里 Squeeze 改为 attribute 形态并同步降 opset

- [ ] **Step 4: Commit（产出文件 + 脚本）**

```bash
git add export/laya/export_laya.py export/laya/smoke_check.py
git commit -m "feat(export): laya -> zjev-contract onnx graph + smoke check"
```

---

### Task 5: 端到端验收 + README + 合并

**Files:**
- Modify: `README.md`（CLI 行、ONNX 后端节）

**Interfaces:**
- Consumes: Task 1-4 全部产出（链接环境、`--ort-extensions`、`out/laya.onnx`、libortextensions）

- [ ] **Step 1: 端到端 serve + 正确束 200**

```bash
zig build -Donnx=true > /tmp/zjev-laya-t5.log 2>&1; echo "build=$?"
(./zig-out/bin/zjev-serve --model export/laya/out/laya.onnx \
    --ort-extensions export/laya/lib/libortextensions.dylib --port 18080 \
    > /tmp/zjev-laya-serve.log 2>&1 &); sleep 1.0
curl -s -o /tmp/zjev-laya-e2e-ok.json -w "%{http_code}" -X POST http://127.0.0.1:18080/v1/execute \
  -H 'Content-Type: application/json' \
  -d '{"state":{"text":"We were billed twice, refund now."},
       "decisions":[
         {"id":"escalate","type":"noul"},
         {"id":"topic","type":"choice","options":["billing","bug","other"]},
         {"id":"urgency","type":"score","scale":["lo","mid","hi"]}],
       "graph":{"nodes":[
         {"id":"n1","decision":"escalate"},
         {"id":"n2","decision":"topic"},
         {"id":"n3","decision":"urgency"}],
         "edges":[{"from":"n1","to":"n2"},{"from":"n2","to":"n3"}]}}'
echo; head -c 400 /tmp/zjev-laya-e2e-ok.json
```
Expected: HTTP 200，JSON 含 `trajectory`（3 步）与概率。**v0 概率无意义（随机 head），只验链路**。

- [ ] **Step 2: 错误束 4xx**

```bash
curl -s -o /tmp/zjev-laya-e2e-bad.json -w "%{http_code}" -X POST http://127.0.0.1:18080/v1/execute \
  -H 'Content-Type: application/json' \
  -d '{"state":{"text":"x"},"decisions":[{"id":"q","type":"noul"}],
       "graph":{"nodes":[{"id":"n1","decision":"q"}],"edges":[]}}'
echo; cat /tmp/zjev-laya-e2e-bad.json
pkill -f "zjev-serve --model export/laya/out/laya.onnx"; sleep 0.2
```
Expected: HTTP 400，body 含 `BadModelIO`（Σ logitCount 2 ≠ 8）。

- [ ] **Step 3: README 更新**

CLI 行（`--sessions <n>` 后）加 `` `--ort-extensions <path>` ``，句尾补「接含 extensions 自定义 op 的图时传扩展库路径」。

ONNX 后端节（导出约定段后）追加：

```markdown
真模型导出：`export/laya/`（Python venv）产出 `laya.onnx`（图内 tokenizer +
固定 8-logit 决策束 escalate-noul/topic-choice3/urgency-score3 + 随机 head，v0 纯 plumbing）。
```bash
export/laya/.venv/bin/pip install -r export/laya/requirements.txt
export/laya/.venv/bin/python export/laya/export_laya.py --en-dir <Laya英文ckpt>
export/laya/.venv/bin/python export/laya/smoke_check.py
./zig-out/bin/zjev-serve --model export/laya/out/laya.onnx \
    --ort-extensions export/laya/lib/libortextensions.dylib
```
libortextensions 从 microsoft/onnxruntime-extensions releases 下 osx-arm64。
```

- [ ] **Step 4: 全量验证 + commit**

```bash
zig build test > /tmp/zjev-laya-t5-test.log 2>&1; echo "t0=$?"
zig build -Donnx=true test > /tmp/zjev-laya-t5-test1.log 2>&1; echo "t1=$?"
zig build test-conformance > /tmp/zjev-laya-t5-conf.log 2>&1; echo "conf=$?"
tail -1 /tmp/zjev-laya-t5-conf.log
git add README.md
git commit -m "docs: e2e laya.onnx run instructions"
```
Expected: t0/t1/conf 均 0，conformance 10 pass 0 fail。

- [ ] **Step 5: 合并 + 复验**

```bash
git checkout main
git merge --no-ff feat/laya-onnx -m "merge: Laya ONNX export + local onnxruntime (v0 plumbing)"
git branch -d feat/laya-onnx
zig build -Donnx=true test > /tmp/zjev-laya-final.log 2>&1; echo "final=$?"
git log --oneline -6
```
Expected: final=0，merge commit 在顶。

---

## Self-Review

**1. Spec coverage：** 环境（Task 1）/运行时扩展（Task 2）/导出脚本+自检（Task 3-4）/端到端+文档（Task 5）；固定决策束与静态 [1,8] 在 Task 4 常量与 assert 落地；错误处理表（版本不匹配→OrtInitFailed 日志 Task 2 Step 3；缺 op→RunFailed 由 smoke/e2e 暴露；Σ≠8→4xx Task 5 Step 2）；风险 1→Task 3 Step 3 fallback；风险 4→Task 3 Step 4 侦察 + Task 4 回退路径。无缺口。

**2. Placeholder scan：** `<EN_DIR>` 是 Task 3 Step 4 产出并打印的真实路径，非占位；易碎点对策均为具体代码级操作。无 TBD。

**3. Type consistency：** `ort_extensions` 字段贯穿 Cli/parseCli/factory.Config/openOnnx 命名一致；`createSession` 新签名与 openOnnx 调用点一致；`--ort-extensions` flag 名在 Task 2/5 一致；NUM_LOGITS=8 与 spec 决策束表一致。

---

## 执行偏差记录（2026-09-24 实际执行）

计划按 Task 1-5 全部完成，以下为实际执行与计划文本的偏差，供后续参考：

1. **Task 1**：build option 实际拼写 `-Donnx_lib_dir`（下划线原样，计划的 `-Donnx-lib-dir` 无效）；`addRPathOnce` → 0.17 为 `addRPath`。onnx=true 路径暴露一批潜伏编译错误（onnx=false 时 comptime-dead 从未分析）：`dupeZ`→`dupeSentinel(u8,x,0)`；`Run` extern 的 inputs 参数应为 `[*c]const ?*const OrtValue`；若干 `[*c]` 数组需 `var`；`EngineError` 与 `head.Error` 需含 `BadModelIO`（routes else 分支 → 400）。
2. **Task 1 运行时坑**：`CastTypeInfoToTensorInfo` 返回**借出**指针，释放 `type_info` 即可，释放 tensor_info 是 double-free（abort，macOS 崩溃报告定位）。
3. **Task 2**：`RegisterCustomOpsLibrary` 第三参传 null 会在 ORT 内部 `*handle=dlopen(...)` 处段错误——必须传有效指针。扩展库缺失先做 `std.Io.Dir.cwd().access` 预检（0.17 无 `std.fs.cwd`）。brew onnxruntime 1.30 在 RegisterCustomOpsLibrary 路径**自身段错误**（缺文件也崩），弃用；改用版本自洽的一对：pip 轮 `libonnxruntime.1.30.0.dylib` + NuGet `Microsoft.ML.OnnxRuntime.Extensions` 0.15.2-dev 的 `libortextensions.dylib`（GitHub releases 无二进制资产；pip 的 `_extensions_pydll.so` 依赖 libpython 不可用）。库放 `export/laya/lib/`，链接用 `addObjectFile` 直接链 dylib（`linkSystemLibrary` 会被 zig 的系统解析抢去 brew 路径）。
4. **Task 3**：venv 依赖 63s 装完；英文 checkpoint = 仓库根 `model.safetensors`(803MB bf16) + `encoder/config.json` + `tokenizer/`（ModernBertForMaskedLM 1024/28，AutoModel 直接加载）；HF 快照阶段即完成大文件下载。
5. **Task 4**：tokenizer 图实际用 `gen_processing_models(stage_dir, pre_kwargs={...}, schema_v2=True)` 的 `HfJsonTokenizer`（输入名 `str` 改名为 `text`；**只输出 ids**——attention_mask 在图内用 Equal+Not+Cast 从 ids 派生）；transformers 5.x 的 `TokenizersBackend` 不被 extensions 识别（需传目录路径字符串）。onnx 1.23 的 schema 注册表缺 `NotEqual`（用 Equal+Not 绕开），且注册表只认 `""` domain（extensions 生成的 `ai.onnx` 字面 domain 需归一）。IR 版本不齐（8 vs 10）需对齐。
6. **Task 5 e2e**：请求三坑——score 的 `scale` 是 `{labels:[...]}` 对象；graph edge 的 `when` 必填；noul 默认 `abstain=true`（显式 false 才匹配 8-logit 图）。**executor 按 frontier 分批、engine 按类型分组各调一次 head**，与"一次前向出全部 logits"的导出图冲突：engine 增加 `Head.bundled` 路径（onnx=true 单调用全量 schema，mock 不变，conformance 全绿）。端到端（平图 edges 空）200 + 错误束 400 BadModelIO 验证通过。**多 wave（条件边）+ ONNX 头为已知限制**，如需支持应做 executor 全量预取改造（独立里程碑决策）。
