# M2 自训 head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用公开工单数据训练 8-logit 决策 head（冻结 Laya encoder），按同一契约导出 ONNX，并用 zjev-traj 在留出集实测轨迹校准。

**Architecture:** `build_dataset.py`（HF 数据集 → 训练/评估 jsonl）→ `train_head.py`（单 Linear(1024,8) 三任务 CE）→ `export_laya.py --head`（替换随机 head）→ `zjev-traj --model`（ONNX 实测）。Zig 侧只改 `tools/traj.zig` CLI。

**Tech Stack:** Zig 0.17.0-dev.2151+2ec5523d5；python3.12 venv `export/laya/.venv`（torch 2.14 MPS / transformers 5.17 / onnx 1.23 / ort 1.30 / extensions 0.15.2，本计划新增 `datasets`）。

## Global Constraints

- 标签-契约对照（唯一事实源，spec §3）：
  - logits[0:2] = escalate noul（abstain=false）：`[yes, no]`，yes=0 / no=1
  - logits[2:5] = topic choice：`[billing, bug, other]`
  - logits[5:8] = urgency score：`[low, medium, high]` = bucket 下标 0/1/2
- 测试：`zig build test`、`-Donnx=true -Donnx_lib_dir=$PWD/export/laya/lib test` exit 0；`test-conformance` 10 pass 0 fail。
- 训练设备自动选 MPS（本机 `torch.backends.mps.is_available()=True` 已验证），fallback CPU。
- 分支流：每 Task 一 commit；完成后 `--no-ff` 合 main、删分支；不 add `zig-out/`。
- spec 全文：`docs/superpowers/specs/2026-09-24-m2-trained-head-design.md`。

---

### Task 1: zjev-traj 增加 --model/--sessions/--ort-extensions

**Files:**
- Modify: `tools/traj.zig:11-62`（main 开头：参数解析 + 模型打开）

**Interfaces:**
- Consumes: `zjev.factory.open(a, io, cfg) !Model`（`src/model/factory.zig:30`；`cfg.kind=.onnx` 且未用 `-Donnx=true` 构建时返回 `error.Unsupported`）；`Model.deinit(a)`。
- Produces: `zjev-traj --dataset <jsonl> [--mock-mode m] [--model p.onnx [--sessions n] [--ort-extensions lib]]`；`--model` 在不支持 onnx 的构建上 exit 2 并打印重建命令。

- [ ] **Step 1: 修改参数解析与模型打开**

`tools/traj.zig` 中，在 `var mock_mode: []const u8 = "peaked";` 之后加：

```zig
    var model_path: ?[]const u8 = null;
    var num_sessions: u16 = 0;
    var ort_extensions: ?[]const u8 = null;
```

在参数 while 循环里 `--mock-mode` 分支后加：

```zig
        } else if (std.mem.eql(u8, arg, "--model") and i + 1 < args.len) {
            i += 1;
            model_path = args[i];
        } else if (std.mem.eql(u8, arg, "--sessions") and i + 1 < args.len) {
            i += 1;
            num_sessions = try std.fmt.parseInt(u16, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--ort-extensions") and i + 1 < args.len) {
            i += 1;
            ort_extensions = args[i];
        }
```

把 usage 行（第 40 行）改为：

```zig
        std.debug.print("usage: zjev-traj --dataset <jsonl> [--mock-mode m] [--profiles-dir d] [--model p.onnx [--sessions n] [--ort-extensions lib]]\n", .{});
```

把 `var model = try zjev.mock.model(mode, a);`（第 61 行）替换为：

```zig
    var model = if (model_path) |mp| blk: {
        const m = zjev.factory.open(a, io, .{
            .kind = .onnx,
            .model_path = mp,
            .num_sessions = num_sessions,
            .ort_extensions = ort_extensions,
        }) catch |e| {
            if (e == error.Unsupported) {
                std.debug.print("--model requires an onnx build: zig build -Donnx=true -Donnx_lib_dir=<dir>\n", .{});
                std.process.exit(2);
            }
            std.debug.print("failed to open model '{s}': {s}\n", .{ mp, @errorName(e) });
            std.process.exit(1);
        };
        break :blk m;
    } else try zjev.mock.model(mode, a);
```

- [ ] **Step 2: 验证（mock 回归 + 错误路径）**

```bash
zig build test > /tmp/m2-t1-test.log 2>&1; echo "test exit=$?"
./zig-out/bin/zjev-traj --dataset datasets/traj_sample.jsonl --mock-mode sequence | head -c 300; echo
zig build > /dev/null 2>&1   # 无 onnx 构建
./zig-out/bin/zjev-traj --dataset datasets/traj_sample.jsonl --model export/laya/out/laya.onnx; echo "exit=$?"
```

Expected: test exit 0；mock 基线输出与 benchmarks/traj_2026-09-24.md 一致（trajectory_accuracy 0.6）；无 onnx 构建时打印 `--model requires an onnx build: ...` 且 exit=2。

- [ ] **Step 3: onnx 构建冒烟**

```bash
zig build -Donnx=true -Donnx_lib_dir=$PWD/export/laya/lib > /tmp/m2-t1-build.log 2>&1 && echo build-ok
./zig-out/bin/zjev-traj --dataset datasets/traj_sample.jsonl \
    --model export/laya/out/laya.onnx \
    --ort-extensions export/laya/lib/libortextensions.dylib | head -c 300; echo
```

Expected: 能运行（现有 traj_sample 是 5-logit 束，宽度不匹配 → 记录全 skipped 或 error，重点是不崩溃、走通 --model 路径；输出 `skipped_records` 后正常退出）。

- [ ] **Step 4: Commit**

```bash
git add tools/traj.zig
git commit -m "feat(traj): --model/--sessions/--ort-extensions to measure trajectory calibration on ONNX models"
```

---

### Task 2: build_dataset.py（HF 工单 → 训练/评估 jsonl）

**Files:**
- Create: `export/laya/build_dataset.py`
- Modify: `export/laya/requirements.txt`（追加 `datasets`）

**Interfaces:**
- Consumes: HF `Tobi-Bueck/customer-support-tickets`（经 `datasets.load_dataset`）。
- Produces: `datasets/support_bundle_train.jsonl`（`{"text","escalate":bool,"topic":str,"urgency":str}`）、`datasets/support_bundle_eval.jsonl`（zjev-traj 记录格式，见 Step 4）；stdout 打印映射统计。

- [ ] **Step 1: 侦察数据集字段**

```bash
export/laya/.venv/bin/pip install datasets > /tmp/m2-pip.log 2>&1; echo "pip exit=$?"
export/laya/.venv/bin/python - <<'EOF'
from datasets import load_dataset
ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
print(ds.column_names)
print(ds[0])
print("langs:", sorted(set(ds["language"]))[:10] if "language" in ds.column_names else "no lang field")
EOF
```

Expected: 打印列名 + 首条样本。把语言字段、正文字段、priority/type/tags 字段的**确切列名**记下来，用于 Step 2 的常量（侦察结果若与下方代码假设的 `issue`/`response`/`ticket_type`/`priority`/`language`/`tags` 不同，以侦察为准改常量，并在 commit message 里注明）。

- [ ] **Step 2: 写 build_dataset.py**

```python
#!/usr/bin/env python3
"""Map Tobi-Bueck/customer-support-tickets (en) to the ZJEV 8-logit bundle.

Outputs:
  datasets/support_bundle_train.jsonl  {"text","escalate","topic","urgency"}
  datasets/support_bundle_eval.jsonl   zjev-traj records (state/decisions/graph/expected)
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

TOPIC_ORDER = ("billing", "bug", "other")
URGENCY_ORDER = ("low", "medium", "high")
SEED = 42
ROOT = Path(__file__).resolve().parents[2]

BILLING = re.compile(r"bill|refund|invoice|payment|charge|pricing|subscription|price", re.I)
BUG = re.compile(r"bug|error|crash|broken|outage|fail|not working|doesn'?t work|incident|disruption", re.I)


def topic_of(type_and_tags: str, head: str) -> str:
    text = f"{type_and_tags} {head}"
    if BILLING.search(text):
        return "billing"
    if BUG.search(text):
        return "bug"
    return "other"


def main():
    ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    rows = [r for r in ds if (r.get("language") or "").lower().startswith("en")]
    print(f"[build_dataset] en rows: {len(rows)} / {len(ds)}")

    stats = Counter()
    recs = []
    for r in rows:
        text = (r["issue"] or "").strip()
        if len(text) < 20:
            continue
        esc = (r.get("priority") or "").lower() == "high"
        topic = topic_of(f"{r.get('ticket_type') or ''} {' '.join(r.get('tags') or [])}", text[:200])
        urg = (r.get("priority") or "").lower()
        if urg not in URGENCY_ORDER:
            urg = "medium"
        stats[topic] += 1
        stats[f"urgency:{urg}"] += 1
        stats[f"escalate:{esc}"] += 1
        recs.append({"text": text, "escalate": esc, "topic": topic, "urgency": urg})

    print("[build_dataset] mapping stats:", dict(stats))
    assert len(recs) >= 800, f"only {len(recs)} usable en rows; relax filters or pick another dataset"

    rng = random.Random(SEED)
    rng.shuffle(recs)
    # 按 urgency 分层 80/20
    by_urg = {u: [r for r in recs if r["urgency"] == u] for u in URGENCY_ORDER}
    train, evals = [], []
    for u, group in by_urg.items():
        cut = max(1, int(len(group) * 0.8))
        train += group[:cut]
        evals += group[cut:]
    rng.shuffle(train)
    rng.shuffle(evals)

    (ROOT / "datasets").mkdir(exist_ok=True)
    with open(ROOT / "datasets/support_bundle_train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open(ROOT / "datasets/support_bundle_eval_flat.jsonl", "w") as f:
        for r in evals:
            f.write(json.dumps(r) + "\n")

    with open(ROOT / "datasets/support_bundle_eval.jsonl", "w") as f:
        for r in evals:
            expected = {"escalate": r["escalate"]}
            if r["escalate"]:
                expected["topic"] = r["topic"]
                expected["urgency"] = URGENCY_ORDER.index(r["urgency"])
            rec = {
                "state": {"text": r["text"]},
                "decisions": [
                    {"id": "escalate", "type": "noul", "abstain": False},
                    {"id": "topic", "type": "choice", "options": list(TOPIC_ORDER)},
                    {"id": "urgency", "type": "score", "scale": {"labels": list(URGENCY_ORDER)}},
                ],
                "graph": {
                    "nodes": [
                        {"id": "esc", "decision": "escalate"},
                        {"id": "topic", "decision": "topic"},
                        {"id": "urg", "decision": "urgency"},
                    ],
                    "edges": [
                        {"from": "esc", "to": "topic", "when": "escalate == true"},
                        {"from": "esc", "to": "urg", "when": "escalate == true"},
                    ],
                },
                "expected": expected,
            }
            f.write(json.dumps(rec) + "\n")
    print(f"[build_dataset] train={len(train)} eval={len(evals)}")


if __name__ == "__main__":
    main()
```

同时 `export/laya/requirements.txt` 追加一行 `datasets`。

- [ ] **Step 3: 运行并核对统计**

```bash
export/laya/.venv/bin/python export/laya/build_dataset.py
head -1 datasets/support_bundle_train.jsonl | head -c 400; echo
head -1 datasets/support_bundle_eval.jsonl | python3 -m json.tool | head -30
wc -l datasets/support_bundle_train.jsonl datasets/support_bundle_eval.jsonl
```

Expected: ≥800 训练条（不足则按 Step 1 偏差处理放宽过滤并在 commit 注明）；三个 topic 都有覆盖；eval 条目的 `expected` 在 escalate=false 时只有 `escalate` 键。

- [ ] **Step 4: Commit**

```bash
git add export/laya/build_dataset.py export/laya/requirements.txt datasets/support_bundle_train.jsonl datasets/support_bundle_eval.jsonl datasets/support_bundle_eval_flat.jsonl
git commit -m "feat(export): map Tobi-Bueck support tickets to the 8-logit bundle (train + eval splits)"
```

---

### Task 3: train_head.py（冻结 encoder 训练 Linear head）

**Files:**
- Create: `export/laya/train_head.py`

**Interfaces:**
- Consumes: `datasets/support_bundle_train.jsonl`（Task 2 格式）；`export_laya.stage_files` / `export_laya.NUM_LOGITS`（`from export_laya import stage_files, NUM_LOGITS`）。
- Produces: `export/laya/out/head.pt` = `torch.save({"weight": Tensor[8,1024], "bias": Tensor[8], "metrics": {...}})`；stdout 逐 epoch 打印三任务 train/eval accuracy。

- [ ] **Step 1: 写 train_head.py**

```python
#!/usr/bin/env python3
"""Train the 8-logit decision head on a frozen Laya encoder (MPS if available).

Loss: 3 x CE over logits segments [0:2] escalate / [2:5] topic / [5:8] urgency
(label order matches the export contract, see spec §3).
"""
import argparse
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from export_laya import NUM_LOGITS, REPO_ID, stage_files

TOPIC_ORDER = ("billing", "bug", "other")
URGENCY_ORDER = ("low", "medium", "high")
SEGMENTS = ((0, 2), (2, 5), (5, 8))


class Jsonl(Dataset):
    def __init__(self, path, tok, max_len=128):
        self.rows = [json.loads(l) for l in open(path) if l.strip()]
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["text"], truncation=True, max_length=self.max_len, return_tensors="pt")
        y = torch.tensor([
            0 if r["escalate"] else 1,
            TOPIC_ORDER.index(r["topic"]),
            URGENCY_ORDER.index(r["urgency"]),
        ])
        return enc["input_ids"][0], enc["attention_mask"][0], y


def collate(batch):
    n = max(ids.shape[0] for ids, _, _ in batch)
    ids = torch.zeros(len(batch), n, dtype=torch.long)
    mask = torch.zeros(len(batch), n, dtype=torch.long)
    y = torch.stack([t for _, _, t in batch])
    for i, (b_ids, b_mask, _) in enumerate(batch):
        ids[i, : b_ids.shape[0]] = b_ids
        mask[i, : b_mask.shape[0]] = b_mask
    return ids, mask, y


def run(model, head, loader, device, opt=None):
    train = opt is not None
    model.train(False)
    head.train(train)
    lossf = nn.CrossEntropyLoss()
    tot = [0, 0, 0]
    correct = [0, 0, 0]
    loss_sum = 0.0
    for ids, mask, y in loader:
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            hs = model(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
            logits = head(hs)
            loss = sum(lossf(logits[:, s:e], y[:, k]) for k, (s, e) in enumerate(SEGMENTS))
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        loss_sum += loss.item()
        for k, (s, e) in enumerate(SEGMENTS):
            tot[k] += y.shape[0]
            correct[k] += (logits[:, s:e].argmax(1) == y[:, k]).sum().item()
    acc = [c / t for c, t in zip(correct, tot)]
    return loss_sum / max(1, len(loader)), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="datasets/support_bundle_train.jsonl")
    ap.add_argument("--eval", default="datasets/support_bundle_eval.jsonl")
    ap.add_argument("--out", default="export/laya/out/head.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    a = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("[train] device:", device)
    stage = stage_files(REPO_ID)
    tok = AutoTokenizer.from_pretrained(str(stage))
    model = AutoModel.from_pretrained(str(stage))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    head = nn.Linear(model.config.hidden_size, NUM_LOGITS).to(device)

    train_dl = DataLoader(Jsonl(a.train, tok), batch_size=32, shuffle=True, collate_fn=collate)
    # eval 用 train_head 格式的 eval jsonl？不需要——直接用 bundle 字符串标签重排一份：
    eval_dl = DataLoader(Jsonl(a.eval, tok), batch_size=64, shuffle=False, collate_fn=collate)

    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    best = -1.0
    bad = 0
    metrics = {}
    for epoch in range(a.epochs):
        tl, ta = run(model, head, train_dl, device, opt)
        el, ea = run(model, head, eval_dl, device)
        score = sum(ea) / 3
        print(f"[train] epoch {epoch}: train_loss={tl:.4f} train_acc={ta} eval_acc={ea} eval_avg={score:.4f}")
        if score > best:
            best = score
            bad = 0
            metrics = {"epoch": epoch, "eval_acc": ea, "train_acc": ta}
            torch.save({"weight": head.weight.detach().cpu(), "bias": head.bias.detach().cpu()}, a.out + ".best")
        else:
            bad += 1
            if bad >= a.patience:
                break
    # 用 best checkpoint 落最终文件
    sd = torch.load(a.out + ".best", map_location="cpu")
    sd["metrics"] = metrics
    torch.save(sd, a.out)
    print("[train] saved:", a.out, "metrics:", metrics)


if __name__ == "__main__":
    main()
```

注意：Task 2 的 eval jsonl 是 zjev-traj 记录格式（`decisions`/`graph`/`expected`），不是训练格式——脚本里 `eval_dl` 直接读它会失败。修正：训练用 eval 集从训练 jsonl 切不出（train_head 的 --eval 应该读训练格式文件）。**做法：`build_dataset.py` 额外落第三份文件 `datasets/support_bundle_eval_flat.jsonl`**（与 train 同格式，即 80/20 中的 eval 部分），train_head 默认 `--eval datasets/support_bundle_eval_flat.jsonl`。Task 2 Step 2 的脚本在写 eval jsonl 的同一个循环里加写 flat 文件（每行 `{"text","escalate","topic","urgency"}`，用 evals 列表）；Task 2 Step 4 的 git add 加该文件。

- [ ] **Step 2: 训练**

```bash
export/laya/.venv/bin/python export/laya/train_head.py 2>&1 | tee /tmp/m2-train.log
```

（`--eval` 默认读 `datasets/support_bundle_eval_flat.jsonl`——与 train 同格式、80/20 切分里的 eval 部分，由 Task 2 生成。）
Expected: device 打印 mps；逐 epoch 三任务 eval_acc 上升；早停后保存 head.pt。
验收线（spec §5.3）：eval 三任务平均 accuracy ≥ 0.70。模板数据期望 0.85+；达不到先查
topic 映射（打印各任务 acc 定位）。

- [ ] **Step 3: Commit**

```bash
git add export/laya/train_head.py
git commit -m "feat(export): train 8-logit decision head on frozen Laya encoder (3-task CE, MPS)"
```

---

### Task 4: export_laya.py --head + smoke + serve 冒烟

**Files:**
- Modify: `export/laya/export_laya.py:123-145`（`add_head`）、`148-152`（参数）

**Interfaces:**
- Consumes: `export/laya/out/head.pt`（Task 3：`weight[8,1024]`/`bias[8]`）。
- Produces: `add_head(merged, hidden_size, head_path=None)`；`--head <path>` CLI 参数。

- [ ] **Step 1: 改 add_head 与参数**

`add_head` 签名改为 `def add_head(merged: onnx.ModelProto, hidden_size: int, head_path: str | None = None) -> onnx.ModelProto:`，其中随机初始化块替换为：

```python
    if head_path:
        sd = torch.load(head_path, map_location="cpu")
        W = sd["weight"].numpy().T.astype(np.float32)  # [8,H] -> [H,8]
        b = sd["bias"].numpy().astype(np.float32)
        log("loaded trained head:", head_path)
    else:
        rng = np.random.default_rng(SEED)
        W = (rng.standard_normal((hidden_size, NUM_LOGITS)) * HEAD_SCALE).astype(np.float32)
        b = np.zeros(NUM_LOGITS, dtype=np.float32)
    assert W.shape == (hidden_size, NUM_LOGITS), W.shape
```

`main()` 的 argparse 加 `ap.add_argument("--head", default=None, help="trained head .pt (weight[8,H]/bias[8]); default random")`；`final = add_head(merged, hidden, a.head)`。

- [ ] **Step 2: 导出 + smoke_check**

```bash
export/laya/.venv/bin/python export/laya/export_laya.py --head export/laya/out/head.pt
export/laya/.venv/bin/python export/laya/smoke_check.py && echo smoke-ok
ls -la export/laya/out/
```

Expected: 导出成功打印 `loaded trained head` 且保存到默认 `out/laya.onnx`
（smoke_check 固定读该路径，故不另起文件名）；smoke_check 绿。

- [ ] **Step 3: serve 冒烟（平图 + 条件图）**

```bash
zig build -Donnx=true -Donnx_lib_dir=$PWD/export/laya/lib > /tmp/m2-t4-build.log 2>&1 && echo build-ok
./zig-out/bin/zjev-serve --model export/laya/out/laya.onnx \
    --ort-extensions export/laya/lib/libortextensions.dylib --sessions 2 --port 18080 \
    > /tmp/m2-serve.log 2>&1 &
sleep 2
# 平图
curl -s -o /dev/null -w "flat=%{http_code}\n" -X POST localhost:18080/v1/execute -H 'content-type: application/json' -d '{"state":{"text":"my bill looks wrong"},"decisions":[{"id":"escalate","type":"noul","abstain":false},{"id":"topic","type":"choice","options":["billing","bug","other"]},{"id":"urgency","type":"score","scale":{"labels":["low","medium","high"]}}],"graph":{"nodes":[{"id":"esc","decision":"escalate"},{"id":"topic","decision":"topic"},{"id":"urg","decision":"urgency"}],"edges":[]}}'
# 条件图（escalate=true 才激活 topic/urgency）
curl -s -X POST localhost:18080/v1/execute -H 'content-type: application/json' -d '{"state":{"text":"my bill looks wrong"},"decisions":[{"id":"escalate","type":"noul","abstain":false},{"id":"topic","type":"choice","options":["billing","bug","other"]},{"id":"urgency","type":"score","scale":{"labels":["low","medium","high"]}}],"graph":{"nodes":[{"id":"esc","decision":"escalate"},{"id":"topic","decision":"topic"},{"id":"urg","decision":"urgency"}],"edges":[{"from":"esc","to":"topic","when":"escalate == true"},{"from":"esc","to":"urg","when":"escalate == true"}]}}' | head -c 400; echo
kill %1
```

Expected: flat=200；条件图 200 且 escalate=true 时 trajectory 3 步、false 时 1 步 + skipped 含 topic/urg。且概率应明显偏离随机 head 的 0.5 平分布（质量冒烟）。

- [ ] **Step 4: Commit**

```bash
git add export/laya/export_laya.py
git commit -m "feat(export): --head flag to export a trained decision head instead of random init"
```

---

### Task 5: zjev-traj 实测 + benchmark + README + 合并

**Files:**
- Create: `benchmarks/traj_laya_2026-09-24.md`
- Modify: `README.md`（ONNX 段「真模型已端到端打通」小节更新为 M2 读数）

- [ ] **Step 1: 实测**

```bash
zig build -Donnx=true -Donnx_lib_dir=$PWD/export/laya/lib > /dev/null 2>&1
./zig-out/bin/zjev-traj --dataset datasets/support_bundle_eval.jsonl \
    --model export/laya/out/laya.onnx \
    --ort-extensions export/laya/lib/libortextensions.dylib | tee /tmp/m2-traj.json
```

Expected: `n` = eval 条数（minus skipped_records=0）；trajectory_accuracy ≥ 0.70 显著高于 mock 基线 0.6；by_node 三个决策各有余量读数（topic 在 escalate=false 的记录下不进 trajectory，n 会小于总条数——预期行为）。

- [ ] **Step 2: 写 benchmark 记录**

`benchmarks/traj_laya_2026-09-24.md`：命令、模型（laya.onnx + trained head）、数据集来源与映射、完整 JSON 读数、与 mock 基线（benchmarks/traj_2026-09-24.md）的对照、已知偏差（合成模板数据、escalate 与 urgency 共线）。

- [ ] **Step 3: README 更新**

「真模型已端到端打通（v0 plumbing）」段落后补 M2 小节：训练命令（build_dataset / train_head / export --head）、实测命令（zjev-traj --model）、benchmark 链接、读数一句话。

- [ ] **Step 4: 全量验证 + commit + 合并**

```bash
zig build test > /tmp/m2-final-test.log 2>&1; echo "test exit=$?"
zig build -Donnx=true -Donnx_lib_dir=$PWD/export/laya/lib test > /tmp/m2-final-onnx.log 2>&1; echo "onnx exit=$?"
zig build test-conformance > /tmp/m2-final-conf.log 2>&1; echo "conf exit=$?"; tail -1 /tmp/m2-final-conf.log
git add benchmarks/traj_laya_2026-09-24.md README.md
git commit -m "docs(bench): trajectory calibration readings for the trained Laya head (M2)"
git checkout main && git merge --no-ff feat/m2-trained-head -m "merge: M2 trained decision head — dataset, training, export, traj calibration" && git branch -d feat/m2-trained-head
```

Expected: 三个 exit 均为 0；merge commit 在 main。
