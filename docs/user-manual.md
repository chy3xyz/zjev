# ZJEV User Manual

Typed Probabilistic Decision Runtime — turn unstructured state (text) into
**probability-calibrated structured decisions**. Implemented in Zig 0.17,
with an optional ONNX backend for real models.

- Protocol & theory: `docs/rfc-0001-zjev-decision-runtime.md`, `docs/prd.md`, `docs/quest1.md`
- Design spec: `docs/specs/2026-09-24-zjev-v0.2-decision-graph-design.md`
- Training report: `docs/reports/2026-09-25-laya-head-training-report.md`
- Measured benchmarks: `benchmarks/`

---

## 1. Installation & build

### Requirements

| Dependency | Version | Notes |
|---|---|---|
| Zig | 0.17.0-dev (via zigup) | required |
| onnxruntime + onnxruntime-extensions shared libs | 1.30 / 0.15.2 | only for the real-model backend; a version-matched pair ships in `export/laya/lib/` — do not mix with other copies |
| Python venv + torch | see `export/laya/requirements.txt` | only for training/exporting models |

### Build

```bash
zig build                              # default build: zjev-serve / zjev-fit / zjev-bench / zjev-traj / zjev-conformance
zig build test                         # unit tests
zig build test-conformance             # protocol conformance fixtures (expect 10 pass, 0 fail)
zig build -Donnx=true -Donnx_lib_dir=$PWD/export/laya/lib   # ONNX backend build
```

> **Gotcha**: a plain `zig build` (without the onnx flags) overwrites the
> ONNX-enabled binaries. Always rebuild with `-Donnx=true` before running any
> `--model` tool or server, or you get "--model requires an onnx build" with
> the rebuild hint.

---

## 2. Five-minute tour (mock mode, no model)

```bash
zig build
zig-out/bin/zjev-serve --port 9377
curl -s -X POST localhost:9377/v1/decide -d @examples/request.json
```

`examples/request.json`:

```json
{
  "state": { "text": "17 on-chain transactions in 24h, balance 3.2 SOL, 8 contract interactions" },
  "decisions": [
    { "id": "risk_level", "type": "choice", "options": ["low", "medium", "high"], "abstain": true },
    { "id": "churn_7d", "type": "score", "scale": { "min": 1, "max": 5 } }
  ]
}
```

Mock distributions: `--mock-mode uniform|peaked|sequence`
(default `peaked`). `--scheduler --cache` enables queue batching and
single-flight caching.

---

## 3. Core concepts

**State** — the unstructured input. The `text` field feeds the model
(in-graph tokenizer); `id/data/embeddings/timestamp/source` are metadata.
The bundled real model uses `text` only.

**Decision schema** — declares the decision type and its legal output space;
the contract shared by requests, graphs, and model bundles. Types: §5.1.

**Uncertainty** — every result carries:
- `confidence`: choice/score/rank → max class probability; **noul → P(yes)**
  (positive-class detector score — gate thresholds depend on this)
- `entropy` / `variance` (score) — distribution spread
- `abstention` — present only with `"abstain": true`; probability mass of
  the model's "decline to answer" column

**Temperature / profiles** — fix overconfident softmax without retraining.
With `--profiles-dir`, the serve looks up a temperature per
(model_name, task, num_options, domain) from `model/calibration/*.json`
and applies `softmax(logits / T)`. T>1 softens (fixes overconfidence),
T<1 sharpens. Argmax decisions never change.

**Decision graph (V0.2)** — decisions wired in a DAG; edges carry `when`
condition expressions; branches whose conditions fail are skipped
(`skipped[]`). Nodes may carry policy gates. One encoder forward serves the
whole graph (full prefetch + activation-wave simulation).

**Gate**:

```
confidence >= threshold        → action_above
confidence <  threshold        → action_below
abstention >= threshold (if set) → action_abstain (takes priority)
```

The step's `action` field in the response is the gate output. Threshold
tuning reference: the selective_risk thresholds printed by `zjev-fit` /
`zjev-traj`.

---

## 4. Pick your path

| I want to… | Go to |
|---|---|
| try the API and see response shapes | §2 mock tour |
| run the bundled real model | §7 Laya pipeline + §5 API |
| branch on conditions / attach policy gates | §5.3 decision graph |
| fix over-/under-confident scores | §9 temperature calibration |
| evaluate model + calibration | §8 eval tools |
| plug in my own model | §6 model contract |
| debug an error | §10 FAQ |

---

## 5. HTTP API reference

Start: `./zig-out/bin/zjev-serve --port 9377 [options]` (options: §8.1).

### 5.1 Decision types

| type | value | probabilities | logitCount (model bundle) | notes |
|---|---|---|---|---|
| `noul` | `true/false` | `probability` = P(yes) | 2 (3 with `abstain:true`) | yes/no. **Defaults to `abstain:true`** |
| `choice` | option string (or `"__abstain__"`) | `probabilities{option: p}` | \|options\| (+1 if abstain) | pick one |
| `score` | expected value (float) | `probabilities{bucket: p}` | #buckets (+1 if abstain) | `scale.labels` or `scale.{min,max}` |
| `rank` | `[{id, score}]` descending | `probabilities{item: p}` | \|items\| | ordering |

### 5.2 `POST /v1/decide` — flat decisions

One forward pass computes all decisions; no graph, no branching.
Request = `state` + `decisions[]` (+ optional `domain`, default `"general"`;
`policy` is not yet supported).

Response:

```json
{
  "results": [
    {
      "id": "escalate", "type": "noul", "value": false,
      "probability": 0.027567,
      "uncertainty": { "confidence": 0.027567 },
      "latency_us": 12834059
    },
    {
      "id": "topic", "type": "choice", "value": "bug",
      "probabilities": { "billing": 0.0, "bug": 0.999914, "other": 0.000086 },
      "uncertainty": { "entropy": 0.000892, "confidence": 0.999914 },
      "latency_us": 12834059
    }
  ],
  "calibration": "matched"
}
```

- `calibration`: `"matched"` = a temperature profile was applied;
  `"default"` = none.
- With `"abstain": true`, `probabilities` gains a `"__abstain__"` key and
  `uncertainty` gains `abstention`.

**`POST /v1/decide/batch`** — body `{"requests": [<decide request>, ...]}`,
returns an array. Pairs with `--scheduler`.

### 5.3 `POST /v1/execute` — decision graph

Same as decide plus a `graph`:

```json
{
  "state": { "text": "..." },
  "decisions": [
    {"id":"escalate","type":"noul","abstain":false},
    {"id":"topic","type":"choice","options":["billing","bug","other"]},
    {"id":"urgency","type":"score","scale":{"labels":["low","medium","high"]}}
  ],
  "graph": {
    "nodes": [
      {"id": "esc", "decision": "escalate"},
      {"id": "topic", "decision": "topic",
       "gate": {"threshold": 0.6, "action_above": "route_human",
                "action_below": "auto_reply", "action_abstain": "review"}},
      {"id": "urg", "decision": "urgency"}
    ],
    "edges": [
      {"from": "esc", "to": "topic", "when": "escalate == true"},
      {"from": "esc", "to": "urg", "when": "escalate == true"}
    ]
  }
}
```

- A node's `decision` must match an id in `decisions[]`; `gate` is optional.
- Edge `when` is a condition expression, compiled to an AST and
  type-checked at parse time. Syntax: `ident == value`, `and/or/not`,
  comparisons `> >= < <= == !=`, field refs `decision_id.confidence` /
  `.value` / `.abstention`
  (e.g. `risk.confidence > 0.5 and risk_level == high`).
- Execution: one forward pass computes all nodes, then activation waves walk
  the frontier; nodes whose conditions fail are not executed and land in
  `skipped[]`.

Response:

```json
{
  "trajectory": [
    {"node_id": "esc", "decision_id": "escalate", "result": { "...": "..." }, "action": "route_human"},
    {"node_id": "topic", "decision_id": "topic", "result": { "...": "..." }}
  ],
  "skipped": ["urg"],
  "path_prob": 0.6123,
  "calibration": "matched"
}
```

`path_prob` = product of per-step condition probabilities along the actual
path (trajectory-level confidence).

### 5.4 Error codes

| HTTP | code | meaning |
|---|---|---|
| 400 | `invalid_request` / `InvalidJson` / `MissingField` | malformed body / missing fields |
| 400 | `BadModelIO` | ΣlogitCount ≠ graph width (most common: noul missing `"abstain": false`; it defaults to true = one extra column) |
| 400 | `invalid_graph` / `InvalidGraph` | graph structure / condition expression / gate threshold illegal |
| 400 | `unsupported` | unsupported fields such as `policy` |
| 503 | `overloaded` | queue full |

Error body: `{"error":{"code":"...","message":"..."}}`.

---

## 6. ONNX backend & model contract

### 6.1 Model contract (read before plugging in your own)

- The graph embeds the tokenizer: input string tensor `text`, output float
  tensor `logits`.
- **Output shape must be static**; `logits` width = Σ logitCount of the
  fixed decision bundle.
- The request's Σ logitCount must equal the graph width, or 400 BadModelIO.
  logitCount: noul=2 (3 if abstain:true), choice=|options| (+1),
  score=#buckets (+1), rank=|items|.
- Logits are segmented in the bundle's declaration order.
- If the graph uses ai.onnx.contrib custom ops (HfJsonTokenizer), pass
  `--ort-extensions` pointing at the extensions library.

### 6.2 Shared-library pairing

`export/laya/lib/` is a version-matched pair (onnxruntime 1.30 from the pip
wheel, libortextensions 0.15.2 from the NuGet package), no Python deps.
**Homebrew's onnxruntime 1.30 segfaults on the
`RegisterCustomOpsLibrary` path — do not mix.**

```bash
zig build -Donnx=true -Donnx_lib_dir=$PWD/export/laya/lib
./zig-out/bin/zjev-serve --model <m.onnx> --ort-extensions export/laya/lib/libortextensions.dylib
```

`--model`'s file name (sans extension) becomes `model_name`, matched by
profiles.

---

## 7. The Laya decision bundle (bundled real model)

Model: ModernBERT-large (HuggingFace `convaiinnovations/laya`) + a trained
Linear(1024,8) head. Bundle = escalate-noul (no abstain) / topic-choice3 /
urgency-score3 (ΣlogitCount=8). English tickets.

### 7.1 Run the server

```bash
zig build -Donnx=true -Donnx_lib_dir=$PWD/export/laya/lib
./zig-out/bin/zjev-serve --model export/laya/out/laya.onnx \
    --ort-extensions export/laya/lib/libortextensions.dylib \
    --profiles-dir model/calibration --port 9377
```

Note: the request bundle must be exactly 8 logits — noul needs an explicit
`"abstain": false`.

### 7.2 Train from scratch (optional; readings in the training report)

```bash
export/laya/.venv/bin/pip install -r export/laya/requirements.txt
export/laya/.venv/bin/python export/laya/build_dataset.py   # HF tickets → 8-logit bundle (train/eval)
export/laya/.venv/bin/python export/laya/train_head.py      # frozen-encoder head training (MPS, ~hours)
export/laya/.venv/bin/python export/laya/export_laya.py --head export/laya/out/head.pt
export/laya/.venv/bin/python export/laya/smoke_check.py
```

### 7.3 Current readings (5,652-record eval, fixed graph, M4 calibration on)

| metric | uncalibrated | calibrated |
|---|---|---|
| trajectory_accuracy | 0.754 | 0.754 (unchanged) |
| traj_ece | 0.200 | **0.106** |
| escalate acc / ece | 0.785 / 0.586 | 0.785 / 0.393 |
| topic acc / ece | 0.888 / 0.093 | 0.888 / 0.045 |

Confidence conventions: traj's noul node ECE uses P(yes) (gate semantics),
which is not comparable to fit's max-prob ECE; on the escalation path the
urgency gold label is constantly "high", making its ECE a degenerate
reading. See `benchmarks/temp_laya_2026-09-25.md` for the full discussion.

### 7.4 See calibration in action

```bash
# instance A (:8790, with --profiles-dir) vs instance B (:8791, without)
python3 export/laya/demo_profiles.py
# same tickets on both; watch confidences leave the 0.000x/0.999x
# saturation band and spread into a usable range
```

---

## 8. CLI tools

### 8.1 `zjev-serve`

| option | default | notes |
|---|---|---|
| `--bind` / `--port` | 127.0.0.1 / 9377 | listen address |
| `--mock-mode` | peaked | uniform / peaked / sequence |
| `--profiles-dir` | off | temperature calibration directory |
| `--scheduler` / `--cache` | off | queue batching / single-flight cache |
| `--model` | none | ONNX model path (requires onnx build) |
| `--sessions` | 0=default | onnxruntime session count (intra-op threads pinned to 1; parallelism via the scheduler) |
| `--ort-extensions` | none | required when the graph has custom ops |

### 8.2 `zjev-fit` — temperature fitting

```bash
# mock, for learning the ropes:
zig-out/bin/zjev-fit --dataset datasets/calibration_sample.jsonl [--out model/calibration]
# real model bundle (groups by (task, num_options), minimizes NLL):
zig-out/bin/zjev-fit --dataset <jsonl> --model m.onnx \
    [--sessions n] [--ort-extensions lib] --bundle '<full schema JSON>' \
    [--model-name n] [--domain d] [--out dir]
```

- `--bundle`: bundle schema JSON (same shape as a serve request) providing
  the segment table; ΣlogitCount must equal the graph width — a single
  decision will 400 BadModelIO.
- Dataset line: `{"state": {...}, "decision": {...}, "label": <bool|string>}`.
  For score decisions use the string bucket name (`"low"/"medium"/"high"`);
  integer bucket indices are rejected.
- Consecutive records with the same state reuse one forward (state cache);
  prints `fitted: N skipped: M` at the end.
- Output `model/calibration/<model>_<task>_<num>_<domain>.json`:
  temperature + ece/brier/nll + selective_risk@0.5/0.7/0.9/0.95.

### 8.3 `zjev-traj` — trajectory-level eval (the quest1 §12 testbed)

```bash
zig-out/bin/zjev-traj --dataset <jsonl> [--mock-mode m] [--profiles-dir d] \
    [--model m.onnx [--sessions n] [--ort-extensions lib]]
```

Dataset line:
`{"state":..., "decisions":[...], "graph":{...}, "expected":{decision_id: gold}}`.
Readings go to **stderr** (`2>readings.json`). Output: trajectory_accuracy /
traj_brier / traj_ece / traj_mce / selective_risk (four coverage tiers) /
by_node (per-node acc + ece). CPU + large model ≈ 2 h per 5,652 records.

### 8.4 `zjev-bench`

```bash
zig-out/bin/zjev-bench --dataset <jsonl> [--profiles-dir dir]
```

Flat calibration report: accuracy / brier / ece / mce + selective_risk.

### 8.5 `zjev-conformance`

Protocol fixture validation — `zig build test-conformance` (expect
10 pass, 0 fail).

---

## 9. Temperature calibration workflow (M4)

**When you need it**: single-hop ECE is high (confidence piled at 0/1),
selective_risk thresholds land in the ≈1.0 saturation band, gate thresholds
are effectively untunable.

**Steps**:

1. Build a calibration set: cover all decisions, with gold labels
   (same distribution as production; disjoint from the eval set is best —
   M4 expanded the eval split, see `export/laya/build_calib.py`).
2. Fit: `zjev-fit --model ... --bundle ...` (real model).
3. Serve with `--profiles-dir`, or re-run `zjev-traj` for measurement.
4. Compare: accuracy should not move (T preserves argmax); ece/brier should
   drop; selective_risk thresholds should leave the saturation band.

Bundled example: escalate T=9.61 / topic T=4.82 / urgency T=7.97,
traj_ece 0.200→0.106 — see `benchmarks/temp_laya_2026-09-25.md`.

---

## 10. Troubleshooting FAQ

| symptom | cause & fix |
|---|---|
| `--model requires an onnx build` | binary overwritten by plain `zig build`; rebuild with `zig build -Donnx=true -Donnx_lib_dir=...` |
| 400 BadModelIO | ΣlogitCount ≠ graph width. Most common: noul missing `"abstain": false` (defaults true = extra column) |
| segfault at startup near `RegisterCustomOpsLibrary` | onnxruntime/ortextensions version mismatch (often Homebrew's); use the matched pair in `export/laya/lib/` |
| 400 InvalidJson | malformed body or wrong field type (e.g. integer bucket index as a score label) |
| 400 InvalidGraph | bad condition syntax, unknown decision reference, gate threshold outside [0,1] |
| traj/fit silent for a long time | not stuck: logs print only at batch end; check `ps cputime` is growing |
| topic/urgency look like guesses | known model limitation (synthetic template training data), see §11 |
| urgency acc constantly 1.0 | degenerate reading: gold is constantly "high" on the escalation path, not model magic |

---

## 11. Known limitations

1. **Synthetic template training data**: the Laya head is trained on
   templated tickets and will mislabel natural text (calibration makes
   confidence honest, not correct). Real-distribution data is the top
   follow-up.
2. **Collinear labels**: escalate=true ⇒ gold urgency=high by construction.
3. **noul confidence is P(yes)**: traj by_node ECE and fit ECE use different
   confidence conventions — read the notes in
   `benchmarks/temp_laya_2026-09-25.md` before comparing.
4. `zjev-traj`/`zjev-fit` are CPU-serial: ~2 h per 5,652 records with the
   large model — the bottleneck for interactive evaluation.

---

## 12. Milestones

| milestone | content |
|---|---|
| V0.1 | core runtime + mock/ONNX backends + calibration protocol |
| V0.2 | decision graph (conditions + gates + multi-wave prefetch) |
| M2 | Laya frozen-head training + export + real readings (broken-graph readings since corrected) |
| M3 | unfreeze-last-2 fine-tune + tokenizer `[CLS]/[SEP]` in-graph fix |
| M4 | end-to-end temperature calibration (this manual §9) |
