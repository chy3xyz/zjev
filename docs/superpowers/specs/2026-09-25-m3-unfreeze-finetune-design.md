# M3 解冻末层微调 设计

日期：2026-09-25
状态：已批准（auto 模式 inline 决策）
前置：`docs/superpowers/specs/2026-09-24-m2-trained-head-design.md`（数据/契约/偏差记录）

## 1. 问题

M2 读数（benchmarks/traj_laya_2026-09-25.md）：冻结 encoder + 线性 head 下
escalate eval acc 0.659、urgency 0.487，仅略高于多数类基线；模型升级触发率
10.5% ≪ 金标 38.6%，trajectory_accuracy 0.548。瓶颈在 head 容量与标签噪声
的叠加——escalate/urgency 需要 encoder 对「紧急程度」的分布语义，线性探测
（linear probe）不够用。

## 2. 方案

`train_head.py --unfreeze-last N`（默认 0 = M2 行为）：解冻最后 N 个
transformer 层 + `final_norm`，判别学习率（head 1e-3 / encoder 2e-5，
AdamW，weight decay 0.01），其余不变（同数据/批次/早停）。

**M3 运行参数**：N=2（28 层中的 layers.26/27 + final_norm）。
预期：escalate/urgency +5~10pt；epoch 成本升至 ~2.5×（约 60min），
epochs 上限 10、patience 3 不变。

**导出契约变化（关键）**：微调后 ONNX 图必须嵌入**调过的** encoder 权重，
否则 head 与 encoder 失配。做法：
- `train_head.py` 额外落 `export/laya/out/encoder_tail.pt`
  （被解冻参数的 state_dict：last-N 层 + final_norm，约 0.5GB fp32）。
- `export_laya.py` 增加 `--encoder-tail <pt>`：`load_laya_encoder` 之后
  用 pt 里的 key 覆盖对应参数（key 无前缀，与 model.state_dict() 一致），
  并断言所有 key 命中（防 silently 丢权重——同 encoder 前缀 bug 的教训）。

## 3. 否决的替代

- **全量微调**：MPS 上 epoch 成本 ~10×，合成数据上过拟合风险高，交互迭代慢。
- **LoRA/peft**：多一个依赖与适配层抽象，v1 收益/复杂度比不如直接解冻末层；
  后续如需再引入。

## 4. 改动面

- `export/laya/train_head.py`：`--unfreeze-last`/`--encoder-lr` 参数、param
  groups、保存 encoder_tail.pt、日志标注可训参数量。
- `export/laya/export_laya.py`：`--encoder-tail` 覆盖加载 + 命中断言。
- 新增 `benchmarks/traj_laya_ft_2026-09-25.md`（M2 vs M3 并排）。
- README：M2 链路命令补 `--encoder-tail` 说明 + M3 读数一句。

## 5. 验收

1. 默认（N=0）行为与 M2 逐比特兼容：`train_head.py` 不带新参数时产出与
   M2 相同结构 head.pt（不强制同数值——随机种子未固定）。
2. N=2 训练跑完，日志打印可训参数量 ≈ 2 层 + head（约 1.3 亿）。
3. `--encoder-tail` 导出 + smoke_check 绿；serve 冒烟 200。
4. zjev-traj 实测读数与 M2 并排写入 benchmark；若 escalate eval acc 提升
   <3pt，如实记录为负结果并停止该路线（改数据路线）。
5. `zig build test`（默认与 onnx）+ conformance 全绿。
