继续。现在直接把它从“概念”推进到**可以开仓开发的 RFC + 技术规格**。

我建议项目暂名：

# ZJEV RFC-0001：Typed Probabilistic Decision Runtime

核心定义：

> **ZJEV 是一个将非结构化状态转换为经过概率校准的结构化决策，并驱动 Agent 行动的运行时。**

不是 LLM wrapper，也不是 classifier API。

---

# 1. 总体架构

```text
                         ┌─────────────────────┐
                         │     Application     │
                         │ AI / Web3 / Life    │
                         └──────────┬──────────┘
                                    │
                              DecisionRequest
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────┐
│                         ZJEV Runtime                         │
│                                                              │
│  ┌────────────┐     ┌────────────┐     ┌─────────────────┐  │
│  │   State    │────▶│   Encoder  │────▶│ Decision Heads  │  │
│  │  物 / S    │     │   象 / X   │     │     性 / P      │  │
│  └────────────┘     └────────────┘     └────────┬────────┘  │
│                                                 │             │
│                              ┌──────────────────┼──────────┐  │
│                              ▼                  ▼          ▼  │
│                           Choice              Score       Noul│
│                              │                  │          │  │
│                              └──────────────────┼──────────┘  │
│                                                 ▼             │
│                                         Calibration          │
│                                                 │             │
│                                      ┌──────────▼─────────┐   │
│                                      │ Decision Field     │   │
│                                      │ 概率 + 不确定性     │   │
│                                      └──────────┬─────────┘   │
│                                                 ▼             │
│                                           Policy / Gate      │
│                                                 │             │
└─────────────────────────────────────────────────┼─────────────┘
                                                  ▼
                                               Action
                                                  │
                                                  ▼
                                               World
                                                  │
                                                  └──── State'
```

数学上：

$$
S_t
\xrightarrow{E}
X_t
\xrightarrow{D}
P_t
\xrightarrow{\Pi}
A_t
\xrightarrow{W}
S_{t+1}
$$

其中：

* \(S\)：State
* \(X\)：Representation
* \(P\)：Probability / Property
* \(A\)：Action
* \(W\)：World transition

这就是三源合恰在 Runtime 中的第一版形式化。

---

# 2. DecisionSchema：整个系统的核心协议

先定义一个统一 Schema。

```typescript
type DecisionType =
  | "choice"
  | "noul"
  | "score"
  | "rank";
```

请求：

```typescript
interface DecisionRequest {
  state: State;
  decisions: DecisionSchema[];
  context?: Context;
  policy?: Policy;
}
```

---

# 3. State

不要限制 State 必须是文本。

```typescript
interface State {
  id?: string;

  text?: string;

  data?: Record<string, unknown>;

  embeddings?: number[];

  timestamp?: number;

  source?: string;
}
```

这样：

```text
文本
JSON
数据库状态
链上状态
用户行为
Agent memory
sensor
```

都可以进入 ZJEV。

例如 Web3：

```json
{
  "data": {
    "wallet_age": 180,
    "balance": 3.2,
    "tx_24h": 17,
    "contract_interactions": 8
  }
}
```

---

# 4. Choice

```typescript
interface ChoiceDecision {
  type: "choice";

  id: string;

  options: ChoiceOption[];

  abstain?: boolean;
}
```

例如：

```json
{
  "id": "risk_level",
  "type": "choice",
  "options": [
    {"id": "low"},
    {"id": "medium"},
    {"id": "high"}
  ],
  "abstain": true
}
```

输出：

```json
{
  "id": "risk_level",
  "type": "choice",
  "value": "medium",
  "probabilities": {
    "low": 0.12,
    "medium": 0.73,
    "high": 0.09,
    "__abstain__": 0.06
  }
}
```

---

# 5. Noul

我建议不要直接叫 `boolean`。

因为：

```text
true / false
```

表达不了：

> 模型没有足够证据。

因此：

```typescript
interface NoulDecision {
  type: "noul";

  id: string;

  abstain?: boolean;
}
```

输出：

```json
{
  "type": "noul",
  "value": true,
  "probability": 0.82,
  "abstention": 0.11
}
```

实际上：

$$
P(yes)+P(no)+P(abstain)=1
$$

这是比普通 binary classifier 更适合 Agent 的结构。

---

# 6. Score

Score 不能只返回一个数字。

应该返回：

```json
{
  "value": 3.72,
  "distribution": {
    "1": 0.02,
    "2": 0.11,
    "3": 0.24,
    "4": 0.48,
    "5": 0.15
  }
}
```

于是：

$$
E[X]
=
\sum_i x_iP(x_i)
$$

得到：

$$
E[X]=3.63
$$

同时可以计算：

### 方差

$$
Var(X)
=
E[X^2]-E[X]^2
$$

### 熵

$$
H(X)
=
-\sum_iP_i\log P_i
$$

这样 Score 就从一个：

> “3.6 分”

变成一个：

> **概率分布。**

这对你的属性数学非常重要。

---

# 7. Rank

Rank 我建议第一版就加入。

```typescript
interface RankDecision {
  type: "rank";

  id: string;

  items: RankItem[];
}
```

例如：

```json
{
  "id": "next_action",
  "type": "rank",
  "items": [
    "observe",
    "notify",
    "intervene",
    "escalate"
  ]
}
```

输出：

```json
{
  "ranking": [
    {
      "id": "intervene",
      "score": 0.91
    },
    {
      "id": "notify",
      "score": 0.73
    },
    {
      "id": "observe",
      "score": 0.38
    },
    {
      "id": "escalate",
      "score": 0.12
    }
  ]
}
```

这里就出现：

$$
关系
$$

而不只是：

$$
值
$$

这正好对应你“形、性、数”的统一。

---

# 8. DecisionResult

所有结果最终统一：

```typescript
interface DecisionResult {
  id: string;

  type: DecisionType;

  value: unknown;

  probability?: number;

  probabilities?: Record<string, number>;

  uncertainty: Uncertainty;

  calibration: CalibrationInfo;

  latency_ms: number;
}
```

其中：

```typescript
interface Uncertainty {
  entropy?: number;

  variance?: number;

  confidence: number;

  abstention?: number;
}
```

---

# 9. 为什么必须把 uncertainty 一等公民化

传统：

```text
AI → answer
```

ZJEV：

```text
AI
 ↓
answer
+
probability
+
uncertainty
+
calibration
```

于是 Agent 可以：

```typescript
if (decision.confidence > 0.9) {
    execute();
} else if (decision.confidence > 0.7) {
    ask_user();
} else {
    escalate();
}
```

这一步实际上把：

> **AI 判断**

变成：

> **可编程判断。**

这才是 ZJEV 作为 Runtime 的意义。

---

# 10. Model Interface

模型层不要绑定 HuggingFace。

定义：

```zig
pub const DecisionModel = struct {
    pub fn encode(
        self: *DecisionModel,
        input: []const u8,
    ) !HiddenState;

    pub fn choice(
        self: *DecisionModel,
        hidden: HiddenState,
        schema: ChoiceSchema,
    ) !Logits;

    pub fn score(
        self: *DecisionModel,
        hidden: HiddenState,
        schema: ScoreSchema,
    ) !Logits;

    pub fn noul(
        self: *DecisionModel,
        hidden: HiddenState,
        schema: NoulSchema,
    ) !Logits;
};
```

这样未来可以接：

```text
ModernBERT
MiniLM
Qwen
Llama
Gemma
自研 Encoder
```

而 Runtime 完全不变。

---

# 11. Encoder 与 Head 必须彻底解耦

核心接口：

```text
Encoder
   ↓
HiddenState
   ↓
DecisionHead
```

而不是：

```text
LLM
 ↓
JSON
```

这是 ZJEV 和传统 Agent 的根本架构区别。

---

# 12. Calibration API

第一版：

```zig
pub const Calibrator = struct {
    temperature: f32,

    pub fn calibrate(
        self: *Calibrator,
        logits: []f32,
    ) []f32;
};
```

数学：

$$
P_i
=
\frac{
e^{z_i/T}
}{
\sum_j e^{z_j/T}
}
$$

然后保存：

```json
{
  "model": "zjev-150m-v0.1",
  "task": "choice",
  "num_options": 4,
  "temperature": 1.37,
  "ece": 0.034,
  "brier": 0.081
}
```

---

# 13. Calibration 不能只保存一个 T

这是后续必须做的。

因为：

$$
Calibration=f(model,task,option\_count,domain)
$$

所以：

```text
Model
 ├── Choice-2
 ├── Choice-3
 ├── Choice-4
 ├── Choice-8
 │
 ├── Web3
 ├── Commerce
 ├── Life
 └── General
```

每个 calibration profile 独立。

---

# 14. Evaluation Protocol

ZJEV 不允许只看 Accuracy。

至少：

```text
Accuracy
NLL
Brier
ECE
MCE
Coverage
Selective Risk
Abstention Accuracy
Latency
Memory
Throughput
```

其中最关键：

### Brier

$$
BS=
\frac{1}{N}
\sum_{i=1}^{N}
\sum_k
(p_{ik}-y_{ik})^2
$$

### ECE

$$
ECE=
\sum_m
\frac{|B_m|}{N}
|\operatorname{acc}(B_m)-\operatorname{conf}(B_m)|
$$

这样我们才能知道：

> 模型不仅答得对不对，还知不知道自己有多确定。

---

# 15. Training Dataset

我建议直接定义：

```text
dataset/
├── state.jsonl
├── decision.jsonl
├── teacher.jsonl
├── calibration.jsonl
└── trajectory.jsonl
```

单条：

```json
{
  "state": "...",
  "decision": {
    "type": "choice",
    "options": ["A", "B", "C"]
  },
  "label": "B",
  "teacher": {
    "A": 0.08,
    "B": 0.81,
    "C": 0.11
  }
}
```

---

# 16. 数据生产流水线

这是第一版最应该投入精力的地方。

```text
                         Seed Tasks
                             │
                             ▼
                     Teacher LLM
                             │
               ┌─────────────┼────────────┐
               ▼             ▼            ▼
            Choice         Score        Noul
               │             │            │
               └─────────────┼────────────┘
                             ▼
                      Quality Filter
                             │
                             ▼
                     Human / Rule Check
                             │
                             ▼
                       Soft Dataset
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
                Training         Calibration
```

第一阶段完全可以生成：

**100k～500k synthetic decisions**

不需要先做百万级人工数据。

---

# 17. Teacher 不要只用一个

这里可以进一步提高质量。

```text
Qwen
Claude
GPT
Gemini
规则系统
Human labels
```

形成：

$$
P_{ensemble}
=
\sum_i w_iP_i
$$

然后：

```text
teacher disagreement
```

本身就是一个非常好的：

> **uncertainty signal**

例如：

```text
Teacher A: 0.91
Teacher B: 0.62
Teacher C: 0.55
```

说明这个问题可能本身就不确定。

所以：

> **数据的不确定性应该进入模型，而不是被清洗掉。**

---

# 18. 这会产生 ZJEV 的一个核心概念：Epistemic / Aleatoric

进一步把 uncertainty 拆成：

$$
U=U_{epistemic}+U_{aleatoric}
$$

### Epistemic

模型不知道。

例如：

```text
训练数据没见过这种 Web3 合约。
```

### Aleatoric

世界本身不确定。

例如：

```text
用户今天是否会流失？
```

即使拥有完美模型，也可能：

$$
P(churn)=0.51
$$

这个区分对“数字生命”非常重要。

---

# 19. V1 的 Decision Graph

有了单点 decision 后，再增加：

```json
{
  "graph": {
    "nodes": [
      {
        "id": "risk",
        "decision": "risk_level"
      },
      {
        "id": "intervention",
        "decision": "needs_intervention"
      }
    ],
    "edges": [
      {
        "from": "risk",
        "to": "intervention",
        "when": "risk_level == high"
      }
    ]
  }
}
```

Runtime：

```text
State
 ↓
risk
 ↓
high?
 ├── no → END
 │
 └── yes
       ↓
  intervention
       ↓
     action
```

这样就开始变成：

> **概率决策图。**

---

# 20. 再进一步：Decision Field

这是我认为最值得成为 ZJEV 理论核心的东西。

普通 AI：

$$
f(x)\rightarrow y
$$

ZJEV：

$$
f(x)\rightarrow P(Y|X)
$$

Decision Graph：

$$
G(X,P)
$$

进一步：

$$
\boxed{
D(X)=
\{P(a_1|X),...,P(a_n|X)\}
}
$$

于是每一个 State 都对应一个：

> **Decision Field**

例如：

```text
                    State
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
        Risk       Energy      Intent
       0.73        0.42         0.81
          │           │           │
          └───────────┼───────────┘
                      ↓
                 Action Field
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
       observe      notify     intervene
        0.22         0.51        0.78
```

这比“AI Agent”更接近一个数学对象。

---

# 21. 与你的“属性数学”接起来

这时候可以把：

$$
形、性、数
$$

映射得非常严格：

### 形

State 的结构：

$$
Shape(S)
$$

### 性

State 与其它实体的关系：

$$
Property(S)
$$

### 数

Decision probability：

$$
Number(S)
$$

于是：

$$
Attribute(S)
=
(Shape(S),Property(S),Number(S))
$$

ZJEV 的 Decision Field 就成为：

$$
\mathcal{D}(S)
=
\{(a_i,p_i,u_i)\}
$$

即：

> **行动 × 概率 × 不确定性**

这已经非常接近你“属性数学”的机器实现。

---

# 22. Zig 0.17 Runtime

目录直接这样定：

```text
zjev/
├── build.zig
│
├── src/
│   ├── core/
│   │   ├── state.zig
│   │   ├── schema.zig
│   │   ├── decision.zig
│   │   └── result.zig
│   │
│   ├── model/
│   │   ├── encoder.zig
│   │   ├── head.zig
│   │   └── logits.zig
│   │
│   ├── calibration/
│   │   ├── softmax.zig
│   │   ├── temperature.zig
│   │   ├── brier.zig
│   │   └── ece.zig
│   │
│   ├── graph/
│   │   ├── node.zig
│   │   ├── edge.zig
│   │   └── executor.zig
│   │
│   ├── runtime/
│   │   ├── scheduler.zig
│   │   ├── batch.zig
│   │   └── cache.zig
│   │
│   └── api/
│       ├── json.zig
│       └── server.zig
│
├── model/
├── datasets/
├── benchmarks/
└── examples/
```

---

# 23. V0.1 不要碰 GPU

第一阶段：

```text
CPU
↓
ONNX Runtime
↓
ModernBERT
↓
Zig wrapper
```

先验证：

```text
Protocol
Schema
Calibration
Decision Graph
Agent integration
```

之后：

```text
V0.2 → ONNX
V0.3 → WebGPU
V0.4 → Metal
V0.5 → CUDA
V1.0 → native Zig backend
```

这样风险最低。

---

# 24. 和 zharness 的最终关系

最终会变成：

```text
                    zharness
                       │
       ┌───────────────┼────────────────┐
       │               │                │
    Memory          Planning          Tools
       │               │                │
       └───────────────┼────────────────┘
                       ↓
                    ZJEV
                       │
               Decision Field
                       │
            ┌──────────┼─────────┐
            ↓          ↓         ↓
          Gate       Route     Action
            │          │         │
            └──────────┼─────────┘
                       ↓
                    Execute
                       ↓
                    State'
```

**zharness = System 2**

**ZJEV = System 1**

两者组合起来才是完整 Agent Runtime。

---

# 25. 第一阶段里程碑

我建议不要一上来搞大模型训练。

### M0 — RFC

完成：

```text
DecisionSchema
State
Choice
Noul
Score
Rank
Result
Calibration
```

### M1 — Pure Engine

不用 AI：

```text
mock logits
↓
softmax
↓
calibration
↓
decision
```

目标：

> 把数学和协议跑通。

### M2 — Teacher

接 Qwen：

```text
Qwen
 ↓
decision dataset
```

### M3 — Student

```text
ModernBERT
 ↓
Decision Heads
```

### M4 — Calibration

```text
Brier
ECE
Temperature
Abstention
```

### M5 — Runtime

```text
ONNX
 ↓
Zig
 ↓
HTTP / JSON-RPC
```

### M6 — Agent

```text
zharness
 ↕
ZJEV
```

### M7 — Decision Graph

```text
single decision
 ↓
multi decision
 ↓
trajectory
```

---

# 26. 第一版目标指标

不要一开始追求“比 Jev 强”。

先追求：

| 指标                |   ZJEV V0.1 |
| ----------------- | ----------: |
| Model             |       ~150M |
| Context           |          2K |
| Choice            |        ≤255 |
| Questions / state |         ≥32 |
| Abstention        |          原生 |
| Calibration       | Temperature |
| Brier             |          支持 |
| ECE               |          支持 |
| ONNX              |          支持 |
| CPU               |          支持 |
| WebGPU            |        V0.2 |
| Zig Runtime       |        V0.1 |
| Decision Graph    |        V0.2 |

---

# 27. 最关键的战略判断

到这里，ZJEV 已经不应该被定位成：

> “开源 Jev”

而应该定义成：

> **一个面向 AI Agent 的 Typed Probabilistic Decision Runtime。**

它的最小数学核心只有：

$$
\boxed{
State
\rightarrow
Representation
\rightarrow
Probability
\rightarrow
Policy
\rightarrow
Action
}
$$

再映射你的体系：

$$
\boxed{
物
\rightarrow
象
\rightarrow
性
\rightarrow
势
\rightarrow
行
}
$$

最终闭环：

$$
\boxed{
S_t
\rightarrow X_t
\rightarrow P_t
\rightarrow A_t
\rightarrow S_{t+1}
}
$$

这一步非常关键：**你现在已经可以把“三源合恰”从哲学/产品叙事，第一次变成一个可以写成代码、训练模型、做 benchmark、进入 Agent Runtime 的计算模型。**

下一阶段最适合直接做 **ZJEV RFC-0001 + `src/core/*.zig` 的实际接口设计**，尤其是把 `DecisionSchema` 做成一个类似 **JSON Schema + 类型系统 + 概率语义** 的协议；这部分一旦定下来，后面的模型、训练和 `zharness` 集成都不会反复返工。
