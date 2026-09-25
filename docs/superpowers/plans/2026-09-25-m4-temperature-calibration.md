# M4 温度标定 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or superpowers:executing-plans.

**Goal:** zjev-fit 支持 ONNX bundle（--model/--bundle/state 缓存），在留出集上拟合 3 决策温度，traj --profiles 复测对比。

**Architecture:** fit.zig 每记录以全 bundle 跑一次 decideRaw、按 decision id 切 logits 段进拟合组；同 state 连续行命中缓存。build_calib.py 把 eval_flat 展开为 fit 格式。

**Tech Stack:** Zig 0.17.0-dev.2151；python3.12 venv。

## Global Constraints

- spec：`docs/superpowers/specs/2026-09-25-m4-temperature-calibration-design.md`
- 验收：test/onnx test/conformance 全绿；mock 回归；bundle 宽度不匹配 skip 计数打印；escalate T 预期 >2。
- 每 Task 一 commit；`--no-ff` 合 main；不 add `zig-out/`、`export/laya/out/`。

---

### Task 1: fit.zig --model/--bundle + state 缓存

**Files:**
- Modify: `tools/fit.zig`

**Interfaces:**
- Consumes: `zjev.factory.open`（同 traj）；`zjev.engine.decideRaw(a, model, state, schemas) !RunOutcome`（`outcome.logits` = flat，schema 主序）；`zjev.logits.logitCount(schema)`；`dataset.load/labelIndex` 不变。
- Produces: `zjev-fit --dataset d [--mock-mode m] [--model p.onnx [--sessions n] [--ort-extensions lib] --bundle '<json>']`；bundle 模式打印 `skipped: N`（宽度不匹配/未知 id 的计数）。

- [ ] **Step 1: 参数与模型打开**

参数块（`--mock-mode` 分支后）加 `--model`/`--sessions`/`--ort-extensions`/`--bundle`。模型打开替换 `var model = try zjev.mock.model(mode, a);` 为 traj 同款双分支（Unsupported → 打印重建命令 exit 2）。注意 `mode` 变量在 onnx 分支未用——保持解析（mock 回归用），加 `_ = mode;` 或仅 mock 分支使用。

- [ ] **Step 2: bundle 解析与段表**

```zig
    var bundle_schemas: ?[]zjev.schema.DecisionSchema = null;
    var seg_starts: []usize = &.{};
    if (bundle_json) |bj| {
        const raw = std.json.parseFromSliceLeaky([]zjev.api_json.RawDecision, a, bj, .{}) catch {
            std.debug.print("invalid --bundle json\n", .{});
            std.process.exit(2);
        };
        const parsed = zjev.api_json.fromRaw(a, .{ .state = .{ .text = "x" }, .decisions = raw }) catch {
            std.debug.print("invalid --bundle schemas\n", .{});
            std.process.exit(2);
        };
        bundle_schemas = parsed.schemas;
        var starts = try std.ArrayList(usize).initCapacity(a, parsed.schemas.len);
        var off: usize = 0;
        for (parsed.schemas) |sc| {
            try starts.append(a, off);
            off += zjev.logits.logitCount(sc);
        }
        seg_starts = starts.items;
    }
```

- [ ] **Step 3: 记录循环（bundle 分支 + state 缓存）**

把现有循环体改为：

```zig
    const total_recs = records.len;
    var skipped: usize = 0;
    var cached_text: ?[]const u8 = null;
    var cached_logits: []const f32 = &.{};

    for (records) |rec| {
        const rd = std.json.parseFromValueLeaky(zjev.api_json.RawDecision, a, rec.decision, .{}) catch {
            skipped += 1;
            continue;
        };
        var raws = [1]zjev.api_json.RawDecision{rd};
        const raw_req = zjev.api_json.RawRequest{
            .state = .{ .id = rec.state_id, .text = rec.state_text },
            .decisions = &raws,
        };
        const parsed = zjev.api_json.fromRaw(a, raw_req) catch {
            skipped += 1;
            continue;
        };
        const s = parsed.schemas[0];
        const li = dataset.labelIndex(s, rec.label) catch {
            skipped += 1;
            continue;
        };

        var seg: []const f32 = undefined;
        if (bundle_schemas) |bs| {
            const key = rec.state_text orelse rec.state_id orelse "";
            if (cached_text == null or !std.mem.eql(u8, cached_text.?, key)) {
                const full = zjev.engine.decideRaw(a, &model, &parsed.state, bs) catch {
                    skipped += 1;
                    continue;
                };
                cached_text = try a.dupe(u8, key);
                cached_logits = full.logits;
            }
            const si = blk: {
                for (bs, 0..) |bsc, i| {
                    if (std.mem.eql(u8, bsc.id(), s.id())) break :blk i;
                }
                skipped += 1;
                continue;
            };
            const st = seg_starts[si];
            const n = zjev.logits.logitCount(s);
            if (st + n > cached_logits.len) {
                skipped += 1;
                continue;
            }
            seg = cached_logits[st .. st + n];
        } else {
            const outcome = zjev.engine.decideRaw(a, &model, &parsed.state, parsed.schemas) catch {
                skipped += 1;
                continue;
            };
            seg = outcome.logits;
        }

        const num: u16 = switch (s) {
            .choice => |c| @intCast(c.options.len),
            .noul => 2,
            .score => |sc| @intCast(sc.bucketCount()),
            .rank => |r| @intCast(r.items.len),
        };
        const gop = findOrAdd(&groups, a, std.meta.activeTag(s), num) catch continue;
        try gop.zs.append(a, seg);
        try gop.labels.append(a, li);
    }
    std.debug.print("fitted: {d} skipped: {d}\n", .{ total_recs - skipped, skipped });
```

（records 总数在循环前已存 `const total_recs = records.len;`，打印 `fitted: {d} skipped: {d}`。）

- [ ] **Step 4: 构建 + mock 回归 + 错误路径**

```bash
zig build test > /tmp/m4-t1-test.log 2>&1; echo "test exit=$?"
zig build > /dev/null 2>&1
./zig-out/bin/zjev-fit --dataset datasets/calibration_sample.jsonl --mock-mode sequence --out /tmp/m4-mock-prof
ls /tmp/m4-mock-prof/ && ./zig-out/bin/zjev-fit --dataset datasets/calibration_sample.jsonl --model export/laya/out/laya.onnx 2>&1 | tail -2; echo "exit=$?"
```

Expected: test exit 0；mock profile 产出（与 model/calibration 样例同结构）；无 onnx 构建时 --model 打印重建命令 exit 2。

- [ ] **Step 5: Commit**

```bash
git add tools/fit.zig
git commit -m "feat(fit): --model/--bundle/--sessions/--ort-extensions with state cache for ONNX bundles"
```

---

### Task 2: build_calib.py + 拟合

**Files:**
- Create: `export/laya/build_calib.py`
- Create: `datasets/support_bundle_calib.jsonl`

- [ ] **Step 1: 写 build_calib.py**

```python
#!/usr/bin/env python3
"""Expand support_bundle_eval_flat.jsonl to zjev-fit format (3 lines/record)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOPIC_ORDER = ("billing", "bug", "other")
URGENCY_ORDER = ("low", "medium", "high")

DECISIONS = [
    {"id": "escalate", "type": "noul", "abstain": False},
    {"id": "topic", "type": "choice", "options": list(TOPIC_ORDER)},
    {"id": "urgency", "type": "score", "scale": {"labels": list(URGENCY_ORDER)}},
]

def main():
    out = open(ROOT / "datasets/support_bundle_calib.jsonl", "w")
    n = 0
    for line in open(ROOT / "datasets/support_bundle_eval_flat.jsonl"):
        if not line.strip():
            continue
        r = json.loads(line)
        labels = [r["escalate"], r["topic"], URGENCY_ORDER.index(r["urgency"])]
        for dec, lab in zip(DECISIONS, labels):
            out.write(json.dumps({"state": {"text": r["text"]}, "decision": dec, "label": lab}) + "\n")
            n += 1
    out.close()
    print(f"[build_calib] wrote {n} lines")

if __name__ == "__main__":
    main()
```

运行：`export/laya/.venv/bin/python export/laya/build_calib.py && head -3 datasets/support_bundle_calib.jsonl`。

- [ ] **Step 2: 拟合（后台，约 108 min CPU）**

```bash
BUNDLE='[{"id":"escalate","type":"noul","abstain":false},{"id":"topic","type":"choice","options":["billing","bug","other"]},{"id":"urgency","type":"score","scale":{"labels":["low","medium","high"]}}]'
./zig-out/bin/zjev-fit --dataset datasets/support_bundle_calib.jsonl \
    --model export/laya/out/laya.onnx \
    --ort-extensions export/laya/lib/libortextensions.dylib \
    --bundle "$BUNDLE" --model-name laya --out model/calibration
```

Expected: `fitted: ~16956 skipped: 0`（或少量 skip 有因）；3 个 `fitted laya_*.json: T=...`；
escalate T 预期 >2。挂无超时后台。

- [ ] **Step 3: Commit**

```bash
git add export/laya/build_calib.py datasets/support_bundle_calib.jsonl model/calibration/laya_*.json
git commit -m "feat(calibration): fitted laya temperatures on held-out support bundle (noul/choice/score)"
```

---

### Task 3: traj --profiles 对比 + benchmark + 合并

- [ ] **Step 1: 复测（后台，约 108 min）**

```bash
./zig-out/bin/zjev-traj --dataset datasets/support_bundle_eval.jsonl \
    --model export/laya/out/laya.onnx \
    --ort-extensions export/laya/lib/libortextensions.dylib \
    --profiles-dir model/calibration --model-name laya --domain general \
    2> /tmp/m4-traj.json
```

- [ ] **Step 2: benchmark**

`benchmarks/temp_laya_2026-09-25.md`：无标定（引 traj_laya_ft）vs 标定后并排
（trajectory_accuracy、traj_ece、by_node acc/ece、selective_risk）；escalate
ece 预期 0.586 → ~0.3 以下；trajectory_accuracy 变化 ±2pt 内。

- [ ] **Step 3: README 标定命令段 + 全量验证 + 合并**

```bash
zig build test > /tmp/m4-final-test.log 2>&1; echo "test exit=$?"
zig build -Donnx=true -Donnx_lib_dir=$PWD/export/laya/lib test > /tmp/m4-final-onnx.log 2>&1; echo "onnx exit=$?"
zig build test-conformance > /tmp/m4-final-conf.log 2>&1; echo "conf exit=$?"; tail -1 /tmp/m4-final-conf.log
git add benchmarks/temp_laya_2026-09-25.md README.md
git commit -m "docs(bench): temperature calibration before/after on laya bundle"
git checkout main && git merge --no-ff feat/m4-temperature-calibration -m "merge: M4 temperature calibration for the laya decision bundle" && git branch -d feat/m4-temperature-calibration
```
