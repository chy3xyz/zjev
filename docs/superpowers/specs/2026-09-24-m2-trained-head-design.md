# M2 自训 head：训练 → 导出 → 轨迹校准实测 设计

日期：2026-09-24
状态：已批准（auto 模式 inline 决策）
前置：`docs/superpowers/specs/2026-09-24-laya-onnx-export-design.md`（导出契约）、
`docs/superpowers/specs/2026-09-24-executor-prefetch-design.md`（多 wave 执行）

## 1. 问题

当前 Laya ONNX 图内 Linear head 是随机初始化（seed 42）——概率无意义，
benchmarks/traj_2026-09-24.md 的 mock 读数只是管线演示。quest1.md §12 的命题
（单跳 calibrated ⇏ 轨迹 calibrated）需要一个**有真实质量、真实校准误差**的
模型才能复核。M2 目标：不换 encoder、不改动 Zig 运行时，只训练决策 head，
按同一契约导出，并用 zjev-traj 在留出集上实测 node-level vs trajectory-level
校准。

## 2. 方案总览

数据 → 训练 → 导出 → 实测 四段，全部脚本入 `export/laya/`，延续既有惯例。

### 2.1 数据集

来源：HF `Tobi-Bueck/customer-support-tickets`（合成工单，含 priority /
type / tags / 语言字段，德英双语）。脚本 `export/laya/build_dataset.py`：

- 只取 `lang == "en"`（Laya 是英文 encoder）。
- 字段映射到 8-logit 束（schema 主序，与导出契约一致）：
  - `escalate`（noul, abstain=false）：`priority == "high"` → yes，否则 no。
    logits[0]=yes, [1]=no（finishOne 取 probs[0] 为 yes）。
  - `topic`（choice）：type/tags 命中 payment/refund/invoice/billing → `billing`；
    命中 incident/outage/error/bug/crash → `bug`；其余 → `other`。
  - `urgency`（score, labels [low, medium, high]）：priority 直接映射
    low/medium/high → bucket 下标 0/1/2。
- 输出两个文件：
  - `datasets/support_bundle_train.jsonl`：`{"text", "escalate": bool,
    "topic": str, "urgency": str}`（训练用）。
  - `datasets/support_bundle_eval.jsonl`：zjev-traj 记录格式
    （state/decisions/graph/expected），图用两 wave 条件图：
    `escalate == true` 才激活 topic+urgency，验证剪枝路径上的 trajectory 读数。
- 确定性 80/20 分层划分（按 priority 分层，seed 固定）。
- 映射命中统计打印（billing/bug/other 各多少条，低资源类报警）。

偏差风险：该数据集是模板合成数据，标签有模板泄漏（如 "billing" 词频高）。
M2 v1 接受——目标是打通「真质量 head → 校准读数」全链路；数据真实性作为
后续改进项记录在 §6。

### 2.2 训练

脚本 `export/laya/train_head.py`：

- 加载 staging 后的 Laya encoder（复用 `export_laya.stage_files`），**冻结
  encoder**，只训 `torch.nn.Linear(1024, 8)`（与导出 head 同形）。
- 三条 CE 损失对应三段 logits：`[0:2]` escalate（标签 yes→0 / no→1）、
  `[2:5]` topic（billing/bug/other 按序）、`[5:8]` urgency（low/medium/high
  bucket 下标）。总 loss = 三段之和。
- 设备自动：MPS 可用则用（本机已验证 `torch.backends.mps.is_available()=True`），
  否则 CPU。ModernBERT-large 前向-only + 小数据，冻结训练在 MPS 上可行。
- batch 32，AdamW lr 1e-3，最多 20 epoch，eval split 早停（patience 3，
  指标 = 三任务平均 accuracy）。
- tokenizer 复用 staging 的 tokenizer.json（transformers AutoTokenizer）。
- 产出：
  - `export/laya/out/head.pt`：`{"weight": [8,1024], "bias": [8],
    "metrics": {...}}`（torch.save）。
  - stdout 打印各任务 train/eval accuracy。

### 2.3 导出

`export_laya.py` 增加 `--head <path.pt>`（默认 None = 随机 seed 42，保持
向后兼容）：加载 head.pt，取 `weight.numpy().T` 为 W[1024,8]、`bias.numpy()`
为 b，替换 `add_head` 里的随机初始化。图结构不变，smoke_check 不变。

### 2.4 实测（zjev-traj 接 ONNX）

`tools/traj.zig` 增加 `--model <path.onnx> [--sessions n]
[--ort-extensions <path>]`：设置后经 `zjev.factory.open(.{.kind=.onnx, ...})`
开模型；未用 `-Donnx=true` 构建时 factory.open 返回 `error.Unsupported`，
traj 捕获后打印「rebuild with: zig build -Donnx=true -Donnx_lib_dir=<dir>」
并以 exit 2 退出。不带 `--model` 时行为不变（mock）。

实测流程：

```bash
zig build -Donnx=true -Donnx_lib_dir=$PWD/export/laya/lib
./zig-out/bin/zjev-traj --dataset datasets/support_bundle_eval.jsonl \
    --model export/laya/out/laya.onnx \
    --ort-extensions export/laya/lib/libortextensions.dylib
```

读数写入 `benchmarks/traj_laya_2026-09-24.md`（n、trajectory_accuracy、
traj_brier、traj_ece/mce、selective_risk、by_node accuracy/ECE），并附与
mock 读数的对比说明。

## 3. 标签-契约对照（唯一事实源）

| 段 | schema | logits | 标签 → 下标 |
|---|---|---|---|
| [0:2] | escalate (noul, abstain=false) | [yes, no] | yes=0, no=1 |
| [2:5] | topic (choice) | [billing, bug, other] | 按 options 序 |
| [5:8] | urgency (score, labels) | [low, medium, high] | low=0, medium=1, high=2 |

训练标签、eval expected、export 图序三处必须与此表一致；计划中以代码常量
`TOPIC_ORDER = ("billing", "bug", "other")` / `URGENCY_ORDER = ("low","medium","high")`
为准，build_dataset 与 train_head 共用（train_head 从 jsonl 读字符串标签，
按同序转下标）。

## 4. 改动面

- 新增 `export/laya/build_dataset.py`、`export/laya/train_head.py`
- 修改 `export/laya/export_laya.py`（`--head` 参数）
- 修改 `tools/traj.zig`（`--model`/`--sessions`/`--ort-extensions`）
- 新增数据集 `datasets/support_bundle_{train,eval}.jsonl`
- 新增 `benchmarks/traj_laya_2026-09-24.md`
- README：M2 段更新（真 head + 实测读数 + 命令）

## 5. 验收

1. `zig build test`、`-Donnx=true ... test` exit 0；conformance 10 pass。
2. `build_dataset.py` 产出两 jsonl，映射统计打印，英文条数 ≥ 800
   （不足则放宽 lang 过滤并在偏差记录里写明）。
3. `train_head.py` 跑完，eval 三任务平均 accuracy ≥ 0.70（合成模板数据，
   期望值 0.85+；低于 0.70 说明映射或训练有 bug）。
4. `export_laya.py --head out/head.pt` 产出新 laya.onnx；`smoke_check.py` 绿；
   `zjev-serve --model ...` 起服务，平图 + 条件图各 curl 一次 200。
5. `zjev-traj --model ...` 在 eval 集上输出读数写入 benchmark 文件；
   trajectory_accuracy 显著高于 mock 基线 0.6。
6. README 与实测命令一致。

## 6. 已知限制 / 后续

- 合成模板数据：lexical 捷径明显，校准读数不能外推到真实分布。
  后续：换真实工单分布（如真实客服日志）或加抗泄漏清洗。
- 冻结 encoder：质量上限受限；后续可解冻最后 N 层或 LoRA。
- escalate 标签 = priority=high，与 urgency 标签相关（共线），
  trajectory 读数解释时需注意 noul 与 score 不独立。

## 7. 执行偏差记录

1. **encoder 权重零加载（严重，已修）**：M2 训练首跑发现 transformers 5.17
   不自动剥离 checkpoint 的 `encoder.` 前缀——`AutoModel.from_pretrained`
   的 LOAD REPORT 显示 0 个直接命中、全部 encoder 参数随机初始化。
   此前所有 `laya.onnx`（含上一里程碑 e2e）的 encoder 均为随机权重，
   plumbing 结论不受影响，但「真 Laya encoder」不成立。修复：
   `export_laya.load_laya_encoder()` 手动 `load_file` + 剥前缀 +
   `load_state_dict(strict=False)`，并用 `torch.equal` 断言与 checkpoint
   逐位一致。训练随之在真 encoder 上重跑。
2. **数据集列名与计划假设不同**（侦察步骤消化）：正文列是 `body`/`subject`
   （非 `issue`），type 列是 `type`（非 `ticket_type`），tags 是 `tag_1..8`
   标量列（非列表），priority 有 5 档（very_low/low/medium/high/critical），
   escalate=high|critical，very_low→low、critical→high。已提交注明。

