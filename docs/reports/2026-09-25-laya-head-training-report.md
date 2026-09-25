# Laya 决策 head 训练报告

日期：2026-09-25
状态：M2（冻结 encoder）与 M3（解冻末 2 层）均已完成并实测（修复图）。
轨迹级读数以 `benchmarks/traj_laya_ft_2026-09-25.md` 为准。
关联：`docs/superpowers/specs/2026-09-24-m2-trained-head-design.md`、`docs/superpowers/specs/2026-09-25-m3-unfreeze-finetune-design.md`、`benchmarks/traj_laya_2026-09-25.md`

## 1. 概述

目标：给 ZJEV 决策运行时提供一个**有真实质量、真实校准误差**的 ONNX 模型，
替代 v0 的随机 head，使 quest1.md §12（单跳 calibrated ⇏ 轨迹 calibrated）
可以用真实读数复核。

链路（全部脚本在 `export/laya/`）：

```
HF 工单数据 --build_dataset.py--> 训练/评估 jsonl --train_head.py--> head.pt [+ encoder_tail.pt]
--export_laya.py--> laya.onnx（图内 tokenizer + encoder + head）--zjev-traj--> 轨迹校准读数
```

## 2. 数据

来源：HuggingFace `Tobi-Bueck/customer-support-tickets`（合成客服工单，61765 条，
德英双语；取英文 28261 条）。

字段映射（8-logit 束，schema 主序）：

| 束段 | schema | 派生规则 |
|---|---|---|
| escalate (noul) | `priority ∈ {high, critical}` → true | 下标：yes=0 / no=1 |
| topic (choice) | type+tags+正文前 200 字符正则 | billing 10268 / bug 12463 / other 5525 |
| urgency (score) | priority 归三档 | very_low/low→low，medium→medium，high/critical→high |

切分：按 urgency 分层 80/20，seed 42 → train 22604 / eval 5652。

已知数据偏差：合成模板数据（lexical 捷径明显）；escalate 与 urgency 同源自
priority 字段（共线）。

## 3. 训练契约

单一 `Linear(1024, 8)`，三段 CE 损失（标签序 = logits 序）：

| logits 段 | 任务 | 类别序 |
|---|---|---|
| [0:2] | escalate | yes, no |
| [2:5] | topic | billing, bug, other |
| [5:8] | urgency | low, medium, high |

encoder：Laya 英文 ModernBERT-large（HF `convaiinnovations/laya`，28 层，
hidden 1024）。**权重加载必须走 `load_laya_encoder()`**（手动剥 `encoder.`
前缀）；naive `from_pretrained` 会得到全随机 encoder（见 §7 事故 1）。

## 4. 实验配置

| | M2（冻结） | M3（解冻末 2 层） |
|---|---|---|
| 可训参数 | head ≈ 8.2K | head + layers.26/27 + final_norm = 24.5M |
| 学习率 | AdamW 1e-3 | head 1e-3 / encoder 2e-5（判别），wd 0.01 |
| batch / epochs / patience | 32 / 20 / 3 | 同左 |
| dropout | 关闭（model.eval()） | 训练时开启 |
| 设备 | MPS（Apple Silicon） | 同左 |
| 实测吞吐 | ~23 min/epoch | ~36 min/epoch |

## 5. 结果

### 5.1 M2 冻结（已完成，early stop epoch 12，best epoch 9）

| epoch | train loss | eval escalate | eval topic | eval urgency | avg |
|---|---|---|---|---|---|
| 0 | 2.351 | 0.653 | 0.772 | 0.471 | 0.632 |
| 3 | 2.167 | 0.646 | 0.790 | 0.468 | 0.635 |
| 6 | 2.133 | 0.646 | 0.794 | 0.380 | 0.607 |
| 9 ★ | 2.103 | 0.659 | 0.792 | 0.487 | **0.646** |
| 12 | 2.091 | 0.650 | 0.796 | 0.461 | 0.635（触发早停） |

### 5.2 M3 解冻（early stop epoch 18，best epoch 15）

| epoch | train loss | eval escalate | eval topic | eval urgency | avg |
|---|---|---|---|---|---|
| 0 | 2.249 | 0.657 | 0.802 | 0.479 | 0.646 |
| 2 | 1.934 | 0.692 | 0.825 | 0.526 | 0.681 |
| 4 | 1.473 | 0.730 | 0.826 | 0.593 | 0.716 |
| 6 | 0.856 | 0.776 | 0.835 | 0.638 | 0.750 |
| 8 | 0.379 | 0.777 | 0.837 | 0.656 | 0.757 |
| 10 | 0.143 | 0.787 | 0.838 | 0.674 | 0.767 |
| 12 | 0.085 | 0.785 | 0.844 | 0.677 | 0.769 |
| 14 | 0.094 | 0.790 | 0.840 | 0.688 | 0.773 |
| 15 ★ | 0.071 | 0.786 | **0.847** | **0.686** | **0.7731** |
| 16 | 0.051 | 0.786 | 0.847 | 0.683 | 0.772 |
| 18 | 0.061 | 0.792 | 0.841 | 0.683 | 0.772（触发早停） |

后期 train acc 达 0.99（强过拟合），eval 自 epoch 10 起进入平台期在 0.77 附近
波动，best 落在 epoch 15。全程 19 epoch × ~36 min ≈ 11.4 h（MPS）。

### 5.3 并排对比（M3 取 best epoch 15）

| 指标 | 多数类基线 | M2 best | M3 best | Δ(M3−M2) |
|---|---|---|---|---|
| escalate | 0.614 | 0.659 | 0.786 | **+12.7pt** |
| topic | 0.441 | 0.792 | 0.847 | +5.5pt |
| urgency | 0.409 | 0.487 | 0.686 | **+19.9pt** |
| 平均 | 0.488 | 0.646 | 0.773 | **+12.7pt** |

结论：escalate 提升远超 M3 验收线（+3pt），路线判定**有效**；线性探测（M2）
确认卡在容量上限。M3 代价：训练时长 ~11.4h（vs M2 ~5.7h），过拟合加剧
（train-eval 差距 0.20 vs M2 的 0.03），eval 已接近该标签口径的天花板
（priority 字段派生标签的噪声约束）。

### 5.4 轨迹级实测（M2 模型，5652 条 eval，约 90 min CPU）

`{"trajectory_accuracy": 0.548, "traj_ece": 0.173}`；
by_node：escalate acc 0.593 / ece 0.381；topic acc 0.571（n=592）；
urgency acc 1.0（n=592，退化读数——升级路径上金标恒 high）。详见
`benchmarks/traj_laya_2026-09-25.md`（**已勘误：坏图读数**）。
要点：escalate 节点严重欠校准且触发率（10.5%）远低于金标（38.6%），是
trajectory 误差主源。**M3 修复图读数**：trajectory_accuracy 0.754、
escalate acc 0.785（与训练 eval 0.786 对齐）/ ece 0.586（过拟合饱和）、
topic acc 0.888 / ece 0.093——见 `benchmarks/traj_laya_ft_2026-09-25.md`。

**M4 标定后读数**（2026-09-25 追加，同 eval 集带 temperature profiles 复测）：
trajectory_accuracy 0.754（不变，T 保序）/ **traj_ece 0.106**（-47%）/
traj_brier 0.181（-14%）；escalate ece 0.393（-33%，P(yes) 口径，结构性
上限见 benchmark 口径说明）、topic ece 0.045。温度 T=9.61/4.82/7.97，
拟合集 max-prob ece 0.03/0.03/0.04。详见
`benchmarks/temp_laya_2026-09-25.md`。

## 6. 分析

1. **容量**：M2 的 escalate/urgency 仅略高于多数类，M3 解冻后显著上升——
   「紧急程度」是分布式语义，线性探测不够。
2. **标签噪声**：escalate/urgency 由 priority 单一字段派生，合成文本里
   medium/high 边界模糊，eval 上限受此约束（M3 的 0.77 可能接近该标签
   口径的天花板）。
3. **共线**：escalate=true ⇒ 金标 urgency=high，升级路径上 urgency 全对
   是退化读数，解释 trajectory 结果时须剔除这一分量。
4. **校准**：M2 escalate 节点 ECE 0.381——置信度与正确率严重脱节
   （模型倾向不升级）。温度标定（`zjev-fit` / profiles）是不重训的修校准
   手段，M4 已端到端验证（traj_ece 0.200→0.106，零精度代价）。

## 7. 事故与偏差记录（按严重度）

1. **[严重] encoder 权重零加载**：transformers 5.x 不剥 `encoder.` 前缀，
   naive 加载 0 命中全随机——M2 之前所有 laya.onnx 的 encoder 都是随机的。
   修复 `load_laya_encoder()` + `torch.equal` 断言（`75eac39`）。
2. **[中] serve 退出必 abort**：sentinel 分配按错长度 free（SafeAllocator），
   已修（`64a4e7e`）。
5. **[严重] 图内 tokenizer 丢特殊 token**：HfJsonTokenizer 输出 raw ids，无
   [CLS]/[SEP]（训练走 AutoTokenizer 有）→ ONNX 图在 CLS-less 输入上跑，
   而 head 按位置 0 = CLS 训练——500 条探针 escalate acc **0.61 vs torch 0.79**
   （torch-cpu 与 torch-mps 完全一致，排除设备因素）。修复：图内
   Concat([cls], ids, [sep]) + 全长 mask（smoke_check 断言常量存在），
   修复后 ORT 0.78 vs torch 0.79、一致率 98.4%（残差 = >128 token 的
   截断差异）。**M2 的 traj 读数是在坏图上测的，已加勘误**；M3 traj 复测后补齐。
6. **[低] MPS「假死」误判**：训练日志只在 epoch 末打印，中途采样主线程
   阻塞在 `.item()` 同步属正常；两次误杀后定位（吞吐实测 1.6s/batch）。
7. **[低] 数据集列名与计划假设不符**：侦察步骤消化（body/subject、type、
   tag_1..8、priority 五档），映射规则以侦察为准。
8. **[低] zjev-traj 读数在 stderr**（`std.debug.print`），重定向时易丢。

## 8. 复现

```bash
# 数据
export/laya/.venv/bin/python export/laya/build_dataset.py
# 训练（M2：去 --unfreeze-last；M3：如下）
export/laya/.venv/bin/python export/laya/train_head.py --unfreeze-last 2
# 导出（M3 加 --encoder-tail）
export/laya/.venv/bin/python export/laya/export_laya.py \
    --head export/laya/out/head.pt [--encoder-tail export/laya/out/encoder_tail.pt]
export/laya/.venv/bin/python export/laya/smoke_check.py
# 实测（读数在 stderr）
zig build -Donnx=true -Donnx_lib_dir=$PWD/export/laya/lib
./zig-out/bin/zjev-traj --dataset datasets/support_bundle_eval.jsonl \
    --model export/laya/out/laya.onnx \
    --ort-extensions export/laya/lib/libortextensions.dylib 2>readings.json
```

## 9. 后续方向

1. M3 完成后补 traj 实测，与 M2 并排（benchmark 已留位）。
2. 温度标定 / Gate 阈值调参——不重训改善 escalate 欠校准与触发率。
3. 真实分布数据替换合成数据（lexical 捷径 + 标签噪声的根因）。
4. 训练吞吐：bf16 autocast 或 `--sessions` 并行实测（M2 traj 实测 90 min
   是下一个交互瓶颈）。
