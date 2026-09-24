进一步往下拆，已经可以把 **Jev / RLCD 的“可复现核心”基本还原出来**。而且有一个重要更新：我上一条把 RLCD 说得偏黑盒了——截至 2026-09-24，TypeSafe 已经公开了 RLCD 的基本目标，但**完整训练配方、模型架构和权重仍未公开**。公开社区已经出现了几套相当接近的复现。([Jev][1])

我建议把它理解成：

> **Jev = LLM 的“语言生成范式”被替换成“概率决策范式”。**

---

# 一、先把真正的核心公式建立起来

普通 LLM 优化的是：

$$
P(token_t|token_{<t},x)
$$

即：

> 下一句话应该说什么？

Jev 优化的是：

$$
P(y|x)
$$

即：

> 在给定状态 \(x\) 下，有限候选结果 \(y\) 的概率是多少？

因此：

```text
ChatGPT

x
↓
Transformer
↓
P(token₁)
↓
P(token₂)
↓
P(token₃)
↓
...
↓
text
```

而 Jev：

```text
x
↓
Transformer
↓
Decision Head
↓
P(y₁), P(y₂), ... P(yₙ)
↓
typed decision
```

这就是 **System One** 的核心抽象。Jev 官方把 Choice、Score、Noul 定义为三类基本决策输出。([Jev][1])

---

# 二、真正关键的是：为什么“校准”比“分类准确率”重要？

这是 RLCD 最值得研究的地方。

假设一个模型判断：

```text
用户是否应该退款？

YES = 0.9
NO  = 0.1
```

传统分类模型只关心：

```text
argmax = YES
```

但 Jev 关心：

$$
P(Y=YES)=0.9
$$

如果模型说 0.9 的样本，长期只有 65% 真的是 YES，那么这个模型：

> **准确率可能不错，但决策概率是不可信的。**

反过来：

```text
模型输出：

0.51
0.52
0.55
0.60
0.70
0.80
```

如果这些概率和真实发生频率高度吻合，那么它非常适合：

```text
if p > 0.8:
    自动执行

elif p > 0.5:
    人工确认

else:
    拒绝/升级
```

这就是 Jev 真正的产品价值：

> **它不是告诉软件“答案是什么”，而是告诉软件“这个答案有多大概率成立”。**

---

# 三、RLCD 的核心：Proper Scoring Rule

这里就进入数学核心。

假设真实结果：

$$
y\in\{0,1\}
$$

模型预测：

$$
p=P(y=1)
$$

最简单的 Brier Score：

$$
L_{Brier}=(p-y)^2
$$

例如真实答案：

$$
y=1
$$

模型：

```text
p = 0.9
```

损失：

$$
(0.9-1)^2=0.01
$$

模型：

```text
p=0.5
```

损失：

$$
(0.5-1)^2=0.25
$$

模型：

```text
p=0.1
```

损失：

$$
(0.1-1)^2=0.81
$$

所以模型会自然学会：

> **越准确，同时越诚实地表达自己的不确定性。**

这就是 RLCD 与普通 RLHF / RLVR 的根本区别。

TypeSafe 对 RLCD 的公开定义就是让模型的概率与实际结果频率对齐，而不是优化语言流畅性。([Jev][1])

---

# 四、为什么不能只用 Accuracy Reward？

假设：

```text
真实答案 = YES

模型 A：
YES 0.51

模型 B：
YES 0.99
```

Accuracy：

```text
A = 1
B = 1
```

传统 reward：

$$
R=1
$$

二者没有区别。

但是如果模型长期遇到类似数据：

```text
真实 YES 约占 60%
```

那么理想模型应该逐渐学到：

$$
p\approx0.6
$$

而不是：

$$
p\approx1
$$

因此 RLCD 必须让：

$$
R(p,y)
$$

对概率本身敏感。

这就是 **proper scoring rule** 的价值。

---

# 五、社区已经做出了一个非常关键的实验

`rlcd-lite` 已经公开了一个非常简化的复现：

```text
Causal LM
   ↓
Choice + Confidence
   ↓
GRPO
   ↓
Brier / proper scoring reward
   ↓
Calibration
```

而且它专门做了：

```text
binary reward
vs
Brier reward
```

的 ablation。

这实际上就是在验证：

> **“只训练答对”与“训练答对 + 概率诚实”之间有什么差异。**

([GitHub][2])

---

# 六、进一步推导：为什么 RLCD 可以用 GRPO？

假设一次任务：

```text
State = x

Options:
A
B
C
```

模型产生：

```text
rollout 1 → A, 0.72
rollout 2 → A, 0.61
rollout 3 → B, 0.85
rollout 4 → A, 0.55
```

真实：

```text
A
```

那么 reward：

$$
R_i
=
-(p_i-y_i)^2
$$

然后做 group normalization：

$$
A_i=
\frac{R_i-\mu_R}
{\sigma_R+\epsilon}
$$

最后：

$$
\nabla_\theta J
\approx
\sum_i
A_i
\nabla_\theta
\log \pi_\theta(a_i|x)
$$

这就变成：

```text
同一个 state
      ↓
多个 decision samples
      ↓
proper scoring reward
      ↓
group relative advantage
      ↓
更新模型
```

这就是为什么 Jev-style RL 可以比传统 reasoning RL 简洁很多。

因为：

> **它不是训练“长链推理”，而是在训练“单步决策”。**

---

# 七、但是这里有一个非常有意思的问题

如果：

```text
Choice
Score
Noul
```

本质都是有限输出，那么甚至不一定需要 autoregressive generation。

真正理想的结构可能是：

```text
                    ┌── Choice Head
                    │
Transformer ────────┼── Score Head
                    │
                    └── Noul Head
```

即：

$$
h=f_\theta(x)
$$

然后：

$$
P_{choice}=softmax(W_c h)
$$

$$
P_{noul}=sigmoid(W_n h)
$$

$$
score=W_s h
$$

这比：

```text
LLM → generate JSON
```

更加干净。

---

# 八、这已经出现了更激进的开源实现

目前已经有人直接采用：

> **ModernBERT + decision head**

而不是完整 causal LLM。

例如 OpenJev / Verdict 使用一个约 **151M 参数**的非自回归决策模型，并公开描述了：

```text
ModernBERT
+
Decision Head
+
RLCD
+
Brier
+
Temperature Scaling
+
Abstention
```

([GitHub][3])

这其实揭示了一个非常重要的事实：

## Jev 的真正终局可能根本不是 LLM。

而是：

> **Decision Model**

也就是说：

```text
LLM
=
Language Model

Jev
=
Decision Model
```

两者只是共享 Transformer 表征能力。

---

# 九、这时候就可以理解 Jev 为什么会特别快

普通 LLM：

```text
prefill
   ↓
KV cache
   ↓
autoregressive decode
   ↓
token
   ↓
token
   ↓
token
   ↓
...
```

Decision Model：

```text
input
 ↓
encoder
 ↓
hidden representation
 ↓
multiple heads
 ↓
probabilities
```

因此可以：

```text
             ┌─ question 1
             ├─ question 2
state ───────┼─ question 3
             ├─ question 4
             └─ question 5
                  ↓
             parallel heads
```

多个问题共享同一次：

$$
f_\theta(x)
$$

所以：

$$
Cost(n\ questions)
\approx
Cost(encoder)+Cost(heads)
$$

而不是：

$$
n\times Cost(LLM)
$$

这就是 **“one state → many decisions”** 的真正工程价值。

---

# 十、再往前一步：把它变成你的“三源决策引擎”

这里我认为就非常值得你自己做了。

你现有理论：

$$
三源=(物,象,性)
$$

可以直接对应：

| 三源 | AI 系统          |
| -- | -------------- |
| 物  | State          |
| 象  | Representation |
| 性  | Decision       |
| 行  | Action         |

形成：

```text
┌─────────────────────────────┐
│            物               │
│        World State          │
│                             │
│ 用户 / 产品 / 链 / Agent     │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│            象               │
│       Semantic State        │
│                             │
│ text / JSON / embedding     │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│            性               │
│      Decision Engine        │
│                             │
│ Choice / Score / Noul       │
│ Rank / Gate / Route         │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│            行               │
│          Action             │
└──────────────┬──────────────┘
               ↓
             新物
```

于是：

$$
物\rightarrow象\rightarrow性\rightarrow行\rightarrow物'
$$

这实际上已经形成一个**生命/Agent 状态演化环**。

---

# 十一、而且可以比 Jev 多一个东西：`State`

这是我认为你真正可以创新的地方。

Jev：

```text
State
 ↓
Decision
```

你的三源模型可以：

```text
Stateₜ
 ↓
Representationₜ
 ↓
Decisionₜ
 ↓
Actionₜ
 ↓
Stateₜ₊₁
```

也就是：

$$
S_{t+1}=F(S_t,a_t,e_t)
$$

于是你的系统不再只是 classifier，而是：

> **Stateful Decision Engine**

进一步：

```text
                 ┌──────────────┐
                 │ World State  │
                 └──────┬───────┘
                        ↓
                  ┌───────────┐
                  │  Encoder  │
                  └─────┬─────┘
                        ↓
               ┌─────────────────┐
               │ Decision Model  │
               └───────┬─────────┘
                       ↓
              ┌──────────────────┐
              │ Probability Field│
              └───────┬──────────┘
                      ↓
               Policy / Action
                      ↓
               Environment
                      ↓
                  New State
```

这就开始接近：

**AI Cybernetics / Digital Life Runtime**

---

# 十二、再解决 Jev 一个明显的理论缺陷：概率不能简单串联

社区已经有人专门指出这个问题：

> 单跳 calibrated，不代表整个 decision trajectory calibrated。

例如：

```text
Decision A
P=0.9
   ↓
Decision B
P=0.9
   ↓
Decision C
P=0.9
```

你如果简单相乘：

$$
0.9^3=0.729
$$

但如果 A/B/C 之间存在条件依赖：

$$
P(A,B,C)
\neq
P(A)P(B)P(C)
$$

所以：

> **Calibration 并不会自动 compositional。**

这个问题已经被公开研究明确指出。([GitHub][4])

这对于你做 Agent Runtime 非常重要。

---

# 十三、所以我会给“三源 JEV”增加第四层

不是：

```text
物 → 象 → 性
```

而是：

```text
物
 ↓
象
 ↓
性
 ↓
势
 ↓
行
```

其中：

### 物

现实状态。

### 象

模型对现实的表征。

### 性

单次决策概率：

$$
P(a|s)
$$

### 势

多个决策组成的：

$$
P(a_1,a_2,...,a_n|s)
$$

也就是**轨迹概率 / 状态势场**。

### 行

真正执行。

于是：

$$
物\rightarrow象\rightarrow性\rightarrow势\rightarrow行
$$

这个扩展我认为和你现有的“形—性—数”框架也能很好结合。

---

# 十四、工程上我会这样做

如果现在开始做，不需要训练大模型。

### V0.1

```text
Qwen3-0.6B / 1.7B
        ↓
shared encoder
        ↓
Choice / Score / Noul
        ↓
softmax
```

### V0.2

加入：

```text
Brier Loss
+
Cross Entropy
+
Temperature Scaling
```

即：

$$
L=L_{CE}+\lambda L_{Brier}
$$

公开的 Verdict 实现已经采用类似的组合：

$$
L_{total}=L_{CE}+L_{Brier}
$$

并再做 temperature scaling。([GitHub][5])

### V0.3

加入：

```text
Abstention
```

例如：

```text
YES       0.42
NO        0.38
ABSTAIN   0.20
```

而不是强迫：

```text
YES 100%
```

### V0.4

加入：

```text
Decision Graph
```

例如：

```text
          ┌─ refund?
          │
State ────┼─ fraud?
          │
          ├─ urgency?
          │
          └─ routing?
```

### V1.0

再进入：

```text
RLCD
 ↓
GRPO
 ↓
proper scoring reward
 ↓
trajectory calibration
```

---

# 十五、如果用 Zig 做，反而非常漂亮

你现在已经在做 `Cordis → Zig`、`DeepSeek Harness → zharness` 这些事情。

我会把这个东西独立成：

```text
zdecision
```

或者：

```text
zjev
```

架构：

```text
zjev/
├── core/
│   ├── state.zig
│   ├── question.zig
│   ├── choice.zig
│   ├── score.zig
│   └── noul.zig
│
├── model/
│   ├── encoder.zig
│   ├── head.zig
│   └── logits.zig
│
├── calibration/
│   ├── brier.zig
│   ├── ece.zig
│   └── temperature.zig
│
├── policy/
│   ├── gate.zig
│   ├── route.zig
│   └── abstain.zig
│
├── runtime/
│   ├── batch.zig
│   ├── cache.zig
│   └── scheduler.zig
│
└── graph/
    ├── node.zig
    ├── edge.zig
    └── trajectory.zig
```

然后：

```text
zharness
    ↓
zjev
    ↓
Agent Runtime
    ↓
Life++ / 过日子 / 三源数字生命
```

---

## 最后一个非常关键的判断

我现在反而不建议把研究重点放在：

> **“如何复刻 TypeSafe 的 Jev”**

而应该升级成：

> **“Jev 揭示了一个新的 AI 软件原语：Decision-as-a-Primitive。”**

LLM 的原语是：

$$
Generate
$$

Jev 的原语是：

$$
Decide
$$

而你可以继续往前：

$$
Generate
\rightarrow
Decide
\rightarrow
Act
\rightarrow
Evolve
$$

对应你的三源：

$$
象
\rightarrow
性
\rightarrow
行
\rightarrow
物'
$$

这样就不是做一个 Jev clone，而是在构造一个 **“三源合恰 AI Runtime”**。

而且目前开源复现已经证明：**151M 级别的非自回归 Decision Model 就可以做这件事**，这意味着它非常适合进一步用 **Zig + ONNX/llama.cpp/MLX/CoreML/WebGPU** 做成边缘侧决策运行时，而不必依赖大模型 API。([GitHub][3])

如果按你现在的技术路线继续，我下一步会直接进入 **《ZJEV：基于 Zig 0.17 的三源决策引擎 V0.1 技术架构》**，把 **模型结构、训练数据生成、Brier/RLCD 数学、GRPO、推理协议、Zig API、WebGPU/Metal 后端以及和 Cordis/zharness 的 Agent Loop** 一起定下来。

[1]: https://www.jevtypesafeai.com/jev/rlcd?utm_source=chatgpt.com "RLCD — the training method behind Jev"
[2]: https://github.com/arnabgho/rlcd-lite?utm_source=chatgpt.com "GitHub - arnabgho/rlcd-lite: Simplified RL for Calibrated Decisions: parallel constrained JSON decoding + GRPO with proper-scoring-rule rewards + calibration eval (Jev/RLCD reconstruction) · GitHub"
[3]: https://github.com/heman10x-ngu/verdict-open-jev?utm_source=chatgpt.com "GitHub - Heman10x-NGU/Verdict-open-jev: Non-autoregressive decision engine on ModernBERT (151M) with calibrated uncertainty (RLCD), TypeSafe AI Jev benchmark audit, and in-browser WebGPU playground · GitHub"
[4]: https://github.com/dnakhoa/jev-deferred-crispification?utm_source=chatgpt.com "GitHub - dnakhoa/jev-deferred-crispification: Position paper: the Hidden-Markov and fuzzy primitives missing from TypeSafe AI's Jev and System-One decision models. Two lemmas, one principle (Deferred Crispification), one architecture (BSF-S1). · GitHub"
[5]: https://github.com/DINHCHUNG93/verdict-open-jev?utm_source=chatgpt.com "GitHub - DINHCHUNG93/verdict-open-jev: Non-autoregressive decision engine on ModernBERT (151M) with calibrated uncertainty (RLCD), TypeSafe AI Jev benchmark audit, and in-browser WebGPU playground · GitHub"
