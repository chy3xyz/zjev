# 速度对比：ZJEV（ORT 图） vs 原版 Laya encoder（torch）（2026-09-30）

同机（10 核 M 系，测试时多会话共用有争抢，绝对值看量级、看倍数），同数据
（`datasets/support_bundle_eval.jsonl` 前 200 条，平均 58 词 ≈ 80 token，
截断 ≤128），同一 ModernBERT-large 编码器，fp32。复跑：

```bash
export/laya/.venv/bin/python export/laya/speed_bench.py
```

## batch=1（在线服务形态）

| 引擎 | 线程 | median ms/条 | 条/s |
|---|---|---|---|
| torch CPU（原版 encoder） | 1 | 140 | 7.1 |
| torch CPU | 8（默认） | 231 | 4.3（争抢下多线程反而波动） |
| **zjev ORT 图（图内 tokenizer）** | 1（当前 serve 固定） | **622** | **1.6** |
| **zjev ORT 图** | 默认 | **269** | **3.7** |
| torch MPS | 1 | 42.8 | 23.4 |

## 批处理形态（吞吐参考）

| 引擎 | 配置 | ms/条 | 条/s |
|---|---|---|---|
| torch CPU | 8 线程 batch=8 | 90.5 | 11.1 |
| torch MPS | batch=8 | 27.7 | 36.1 |
| zjev ORT 图 | **batch>1 不支持**（图内 Concat 限 batch=1） | — | — |

## 结论

1. **单条延迟**：zjev ORT CPU 约为 torch CPU 同线程的 **2–4.5×**
   （默认线程 269 vs 140 ms；单线程 622 vs 140 ms）。per-token 约
   7.8 ms（ORT）vs 1.75 ms（torch）。延迟随 token 数**线性**增长
   （10→400 词：123→3513 ms），图是动态的，无固定 padding 浪费。
2. **吞吐**：serve 靠 `--sessions N` 进程内并行（每会话 1 线程 intra-op），
   吞吐 ≈ N × 1.6 条/s；sessions=8 约 13 条/s，打平 torch CPU 批处理。
   单进程内存 ~7.9 GB（含 ORT arena）。
3. **差距来源**：ORT CPU 对 ModernBERT 的 attention/MLP kernel 不如
   torch SDPA 优化；我们图内还包含 tokenizer。MPS/GPU 路径当前不可用
   （无 CoreML/MPS EP）。
4. **质量维度不可直接比**：原版 `convaiinnovations/laya` 的
   typed-decisions 是 **RL 训练的策略模型**（rl_agent_config：act_costs、
   cost_wrong_act、7313 updates，且自带 temperature_by_options——与 M4
   标定同一思路但训练信号/数据不同）；本项目 head 是**监督训练**在客服
   工单 escalate/topic/urgency 上（acc 0.785/0.888，见 traj benchmark）。
   两者只有编码器同源，acc 数字不能互比。

## 后续提速方向（按性价比）

1. ORT 图优化：transformers 工具链（optimizer / 融合 attention），预期最大头
2. `--sessions` 默认值调高 / 按核数自适应
3. 导出动态 batch 图（修图内 Concat），吃批处理红利
4. CoreML EP 接 MPS/ANE（27.7 ms/条的参照系）
