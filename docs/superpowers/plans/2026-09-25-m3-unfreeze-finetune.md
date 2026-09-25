# M3 解冻末层微调 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** train_head.py 支持解冻最后 N 层（判别学习率），微调后的 encoder 尾部随导出嵌入 ONNX，traj 实测与 M2 并排。

**Architecture:** 训练侧 param groups + 保存 thawed 参数；导出侧 `--encoder-tail` 覆盖加载 + key 命中断言。

**Tech Stack:** Zig 0.17.0-dev.2151；python3.12 venv（torch 2.14 MPS）。

## Global Constraints

- spec：`docs/superpowers/specs/2026-09-25-m3-unfreeze-finetune-design.md`。
- M3 运行参数：N=2，head lr 1e-3 / encoder lr 2e-5，AdamW wd 0.01，epochs ≤10、patience 3。
- 验收：`zig build test`、`-Donnx=true -Donnx_lib_dir=$PWD/export/laya/lib test`、conformance 10 pass。
- 每 Task 一 commit；`--no-ff` 合 main；不 add `zig-out/` 与 `export/laya/out/`。
- encoder 权重教训：`--encoder-tail` 必须断言所有 key 命中（复用 M2 前缀 bug 模式）。

---

### Task 1: train_head.py + export_laya.py 改造

**Files:**
- Modify: `export/laya/train_head.py`
- Modify: `export/laya/export_laya.py`

**Interfaces:**
- Consumes: M2 全部既有接口。
- Produces: `train_head.py --unfreeze-last N --head-lr 1e-3 --encoder-lr 2e-5 --encoder-out export/laya/out/encoder_tail.pt`；`export_laya.py --encoder-tail <pt>`；pt 格式 = `{param_name: tensor}`（name 与 `model.state_dict()` 一致，无前缀）。

- [ ] **Step 1: train_head.py**

参数块加：

```python
    ap.add_argument("--unfreeze-last", type=int, default=0)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--encoder-lr", type=float, default=2e-5)
    ap.add_argument("--encoder-out", default="export/laya/out/encoder_tail.pt")
```

`model.eval()` + 冻结循环之后加：

```python
    thaw_names: list[str] = []
    if a.unfreeze_last > 0:
        n_layers = model.config.num_hidden_layers
        thaw = tuple(f"layers.{i}." for i in range(n_layers - a.unfreeze_last, n_layers))
        for name, p in model.named_parameters():
            if name.startswith(thaw) or name.startswith("final_norm."):
                p.requires_grad_(True)
                thaw_names.append(name)
        print(f"[train] thawed last {a.unfreeze_last} layers + final_norm: "
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M params", flush=True)
```

优化器改为 param groups：

```python
    groups = [{"params": list(head.parameters()), "lr": a.head_lr}]
    enc_trainable = [p for p in model.parameters() if p.requires_grad]
    if enc_trainable:
        groups.append({"params": enc_trainable, "lr": a.encoder_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
```

best 分支里在 `torch.save({"weight": ..., "bias": ...}, a.out + ".best")` 后加：

```python
            if thaw_names:
                msd = model.state_dict()
                torch.save({k: msd[k].cpu() for k in thaw_names}, a.encoder_out + ".best")
```

最终落盘段（`sd = torch.load(a.out + ".best", ...)` 之后）加：

```python
    if thaw_names:
        tail = torch.load(a.encoder_out + ".best", map_location="cpu")
        torch.save(tail, a.encoder_out)
        print("[train] saved encoder tail:", a.encoder_out, f"({len(tail)} tensors)")
```

- [ ] **Step 2: export_laya.py --encoder-tail**

argparse 加 `ap.add_argument("--encoder-tail", default=None, help="fine-tuned encoder tail .pt (state_dict of thawed params)")`。`model = load_laya_encoder(str(stage))` 之后加：

```python
    if a.encoder_tail:
        tail = torch.load(a.encoder_tail, map_location="cpu")
        msd = model.state_dict()
        missing = [k for k in tail if k not in msd]
        assert not missing, f"encoder-tail keys not in model: {missing[:5]}"
        msd.update(tail)
        model.load_state_dict(msd)
        log("encoder tail overridden:", a.encoder_tail, f"({len(tail)} tensors)")
```

- [ ] **Step 3: 小样例验证新路径**

```bash
export/laya/.venv/bin/python export/laya/train_head.py --train /tmp/train_subset.jsonl --eval /tmp/eval_subset.jsonl \
  --out /tmp/head_ft.pt --encoder-out /tmp/enc_tail.pt --unfreeze-last 2 --epochs 1 2>&1 \
  | grep -E "^\[train\]"
export/laya/.venv/bin/python export/laya/export_laya.py --head /tmp/head_ft.pt --encoder-tail /tmp/enc_tail.pt --out /tmp/laya_ft.onnx > /tmp/m3-mini-export.log 2>&1 \
  && grep -E "thawed|encoder tail|saved" /tmp/m3-mini-export.log
```

Expected: 打印 thawed 参数量（约 130M 级）；导出打印 `encoder tail overridden: ... (N tensors)`；保存成功。
默认路径回归：`export/laya/.venv/bin/python export/laya/train_head.py --train /tmp/train_subset.jsonl --eval /tmp/eval_subset.jsonl --out /tmp/head_def.pt --epochs 1` 无 thaw 打印、正常保存。

- [ ] **Step 4: Commit**

```bash
git add export/laya/train_head.py export/laya/export_laya.py
git commit -m "feat(export): --unfreeze-last fine-tune with encoder-tail export override"
```

---

### Task 2: M3 全量训练（N=2）

- [ ] **Step 1: 启动后台训练**

```bash
rm -f export/laya/out/head.pt export/laya/out/head.pt.best export/laya/out/encoder_tail.pt export/laya/out/encoder_tail.pt.best
export/laya/.venv/bin/python export/laya/train_head.py --unfreeze-last 2 > /tmp/m3-train.log 2>&1
```

挂无超时后台任务。预期 epoch ≈60min，early stop 通常 4-7 epoch（约 4-7h）。
判定：日志出现 `thawed last 2 layers` 行；eval escalate 相比 M2 的 0.659 有提升即方向正确。

- [ ] **Step 2: 训练完成后 commit 标记**

```bash
git add export/laya/out  # 若被 gitignore 则跳过——out/ 不入库
# 产物不入库；无需 commit。训练完成即 Task 完成。
```

（head.pt / encoder_tail.pt 属 gitignored 产物，同 M2。）

---

### Task 3: 导出 + 冒烟 + traj 实测 + benchmark + 合并

- [ ] **Step 1: 导出 + smoke + serve 冒烟**

```bash
export/laya/.venv/bin/python export/laya/export_laya.py --head export/laya/out/head.pt --encoder-tail export/laya/out/encoder_tail.pt
export/laya/.venv/bin/python export/laya/smoke_check.py && echo smoke-ok
zig build -Donnx=true -Donnx_lib_dir=$PWD/export/laya/lib > /tmp/m3-build.log 2>&1 && echo build-ok
./zig-out/bin/zjev-serve --model export/laya/out/laya.onnx --ort-extensions export/laya/lib/libortextensions.dylib --sessions 1 --port 18082 > /tmp/m3-serve.log 2>&1 &
sleep 3
curl -s --max-time 90 -o /dev/null -w "flat=%{http_code}\n" -X POST localhost:18082/v1/execute -H 'content-type: application/json' -d '{"state":{"text":"my bill looks wrong"},"decisions":[{"id":"escalate","type":"noul","abstain":false},{"id":"topic","type":"choice","options":["billing","bug","other"]},{"id":"urgency","type":"score","scale":{"labels":["low","medium","high"]}}],"graph":{"nodes":[{"id":"esc","decision":"escalate"},{"id":"topic","decision":"topic"},{"id":"urg","decision":"urgency"}],"edges":[]}}'
kill %1 2>/dev/null; pkill -f "port 18082" 2>/dev/null
```

Expected: smoke-ok、flat=200。

- [ ] **Step 2: traj 实测（后台，约 90min）**

```bash
./zig-out/bin/zjev-traj --dataset datasets/support_bundle_eval.jsonl \
    --model export/laya/out/laya.onnx \
    --ort-extensions export/laya/lib/libortextensions.dylib 2> /tmp/m3-traj.json
```

（记住：读数在 stderr。）

- [ ] **Step 3: benchmark 并排 + README**

`benchmarks/traj_laya_ft_2026-09-25.md`：M2（冻结）vs M3（解冻 2 层）并排表
（训练指标 + traj 读数），解读（触发率变化、escalate ece、负结果也如实写）。
README M2 段落补 `--encoder-tail` 命令与 M3 读数一句。

- [ ] **Step 4: 全量验证 + commit + 合并**

```bash
zig build test > /tmp/m3-final-test.log 2>&1; echo "test exit=$?"
zig build -Donnx=true -Donnx_lib_dir=$PWD/export/laya/lib test > /tmp/m3-final-onnx.log 2>&1; echo "onnx exit=$?"
zig build test-conformance > /tmp/m3-final-conf.log 2>&1; echo "conf exit=$?"; tail -1 /tmp/m3-final-conf.log
git add benchmarks/traj_laya_ft_2026-09-25.md README.md
git commit -m "docs(bench): M3 fine-tuned trajectory readings vs M2 frozen-head baseline"
git checkout main && git merge --no-ff feat/m3-unfreeze-finetune -m "merge: M3 unfreeze-last-2 fine-tune + encoder-tail export" && git branch -d feat/m3-unfreeze-finetune
```
