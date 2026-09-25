# zjev-traj 实测：M3 解冻微调 head，修复图（2026-09-25）

模型：`export/laya/out/laya.onnx`（真 Laya encoder + 解冻末 2 层微调，
best epoch 15，训练 eval escalate 0.786 / topic 0.847 / urgency 0.686）
图：**已修复**——图内 Concat([CLS], ids, [SEP])（见训练报告 §7 事故 5；
M2 读数在坏图上测的，已勘误存档）
数据：`datasets/support_bundle_eval.jsonl`（5652 条），两 wave 条件图

命令：

```bash
./zig-out/bin/zjev-traj --dataset datasets/support_bundle_eval.jsonl \
    --model export/laya/out/laya.onnx \
    --ort-extensions export/laya/lib/libortextensions.dylib 2>readings.json
```

读数（CPU onnxruntime，约 108 min）：

```json
{"n":5652,"trajectory_accuracy":0.753539,"traj_brier":0.210130,"traj_ece":0.199872,"traj_mce":0.436636,"skipped_records":0,
 "by_node":[
   {"id":"escalate","n":5652,"accuracy":0.784501,"ece":0.585532},
   {"id":"topic","n":1556,"accuracy":0.887532,"ece":0.092520},
   {"id":"urgency","n":1556,"accuracy":1.0,"ece":0.021754}]}
```

selective_risk：coverage 0.5 → risk 0.120（threshold 0.99997）；0.9 → risk 0.211。

## 有效性交叉验证

500 条探针：ORT escalate acc 0.780 vs torch 0.788（一致率 98.4%，残差为
>128 token 记录的截断差异）——图修复后跨引擎一致性闭合，traj 读数可信。

## 与历史读数对比（注意基准差异）

| 指标 | M2 冻结（坏图） | M3 解冻（坏图） | M3 解冻（修复图） |
|---|---|---|---|
| trajectory_accuracy | 0.548 | 0.541 | **0.754** |
| traj_ece | 0.173 | 0.354 | 0.200 |
| escalate acc / ece | 0.593 / 0.381 | 0.610 / 0.555 | **0.785 / 0.586** |
| topic acc / ece（n） | 0.571（592） | 0.585（937） | **0.888（1556）** / 0.093 |

## 解读

1. **acc 一致性**：轨迹级 escalate acc 0.7845 ≈ 训练 eval 0.7863——训练侧
   指标与部署侧读数首次对齐。
2. **topic 全面健康**：acc 0.888 且 ece 0.093——准且校准良好。
3. **escalate 准而狂**：acc 0.785 但 ece 0.586——过拟合（train 0.99）导致
   softmax 饱和，置信度趋近 0/1 而与对错脱节；selective_risk 在
   coverage 0.5 仍有 risk 0.12，说明顶部置信度仍含信息，但整体
   温度明显过高。**温度标定（zjev-fit / profiles）是下一步最高性价比
   改进**。
4. **urgency acc 1.0 仍是退化读数**（升级路径金标恒 high，ece 0.022 仅
   说明饱和置信恰好全对）。
5. 触发率 27.5%（1556/5652），较 M2 坏图读数（10.5%）接近金标 38.6%，
   但仍有差距——阈值/温度调整后应进一步改善。

## 对 quest1 §12 的数据点

单跳节点校准 ⇏ 轨迹校准：**escalate 节点 ece 0.586（极差） vs trajectory
ece 0.200（中等）**——path_prob 连乘把饱和置信重新摊开，多步组合对单跳
失准有一次「平均化」效应；但 trajectory_accuracy 0.754 仍显著低于单跳最好
的 topic 0.888，组合误差主要由 escalate 单跳错误传导（升级判错 ⇒ topic
全步判错或整支剪错）。
