# ZJEV

**Typed Probabilistic Decision Runtime** — turn unstructured state into
**probability-calibrated structured decisions**. Implemented in Zig 0.17,
with an optional ONNX backend for real models.

[![Zig](https://img.shields.io/badge/zig-0.17-orange)](https://ziglang.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

English manual: [`docs/user-manual.md`](docs/user-manual.md) ·
Protocol & theory: [`docs/rfc-0001-zjev-decision-runtime.md`](docs/rfc-0001-zjev-decision-runtime.md)

---

## Why

Most decision APIs return a label and call it a day. ZJEV treats a decision as
a **typed random variable**: every answer ships with its full distribution,
an uncertainty summary (confidence / entropy / abstention), and an optional
temperature-calibrated softmax so the confidence actually means what it says.
Decisions can be composed into **conditional decision graphs** with policy
gates — and the runtime reports calibration at both the single-decision and
the whole-trajectory level (the thesis explored in `docs/quest1.md` §12:
*single-hop calibrated ≠ trajectory calibrated*).

## Features

- **Typed decisions** — `noul` (yes/no), `choice` (pick one), `score`
  (bucketed expectation), `rank` (ordering), each with optional abstention.
- **Uncertainty out of the box** — per-decision probabilities, entropy,
  variance, abstention mass; single forward pass computes the whole bundle.
- **Decision Graph (V0.2)** — DAG with compiled `when` conditions, activation
  waves, policy `Gate`s (`action_above` / `action_below` / `action_abstain`).
- **ONNX backend** — in-graph tokenizer (`text` string in → static `logits`
  out); multi-wave conditional graphs share one encoder forward.
- **Temperature calibration** — fit per-decision temperatures without
  retraining; serve applies them via profiles.
- **Eval tooling** — `zjev-traj` (node vs trajectory calibration),
  `zjev-bench`, `zjev-fit`, protocol conformance suite.

## Quickstart

```bash
zig build                     # builds zjev-serve / zjev-fit / zjev-bench / zjev-traj / zjev-conformance
zig build test                # unit tests
zig build test-conformance    # protocol fixtures (expect 10 pass, 0 fail)

# mock mode — no model needed
zig-out/bin/zjev-serve --port 9377
curl -s -X POST localhost:9377/v1/decide -d @examples/request.json
```

Real-model mode (the bundled Laya bundle: ModernBERT-large + a trained
8-logit head for escalate/topic/urgency on English tickets):

```bash
zig build -Donnx=true -Donnx_lib_dir=$PWD/export/laya/lib
./zig-out/bin/zjev-serve --model export/laya/out/laya.onnx \
    --ort-extensions export/laya/lib/libortextensions.dylib \
    --profiles-dir model/calibration --port 9377
```

> **Gotcha**: a plain `zig build` overwrites the ONNX-enabled binaries.
> Rebuild with `-Donnx=true` before using `--model` tools.

See the difference calibration makes:

```bash
# instance A (:8790, with --profiles-dir) vs instance B (:8791, without)
python3 export/laya/demo_profiles.py
```

## HTTP API at a glance

| Endpoint | Purpose |
|---|---|
| `POST /v1/decide` | flat decision bundle → per-decision values + probabilities + uncertainty |
| `POST /v1/decide/batch` | batched decide (`{"requests":[...]}`) |
| `POST /v1/execute` | decision graph → `trajectory[]` + `skipped[]` + `path_prob` |

Full request/response schemas, decision types, error codes and condition
expression syntax: **[`docs/user-manual.md`](docs/user-manual.md) §5**.

## CLI tools

| Tool | What it does |
|---|---|
| `zjev-serve` | HTTP runtime; `--model/--sessions/--ort-extensions/--profiles-dir/--scheduler/--cache` |
| `zjev-fit` | temperature fitting (mock dataset, or real model with `--bundle`) |
| `zjev-traj` | trajectory-level calibration report (node vs trajectory, selective risk) |
| `zjev-bench` | flat calibration report (accuracy/brier/ece/mce + selective risk) |
| `zjev-conformance` | protocol fixture validation |

## Status & milestones

| Milestone | Content | Headline numbers (5,652-ticket eval) |
|---|---|---|
| V0.1 | core runtime, mock/ONNX backends, calibration protocol | — |
| V0.2 | decision graph: conditions, gates, wave-prefetch execution | — |
| M2 | Laya frozen-head training, export, real readings | traj acc 0.548 (broken graph, since corrected) |
| M3 | unfreeze-last-2 fine-tune + in-graph `[CLS]/[SEP]` fix | traj acc **0.754**, escalate acc 0.785 / ece 0.586 |
| M4 | end-to-end temperature calibration | traj_ece **0.200 → 0.106**, brier −14%, accuracy unchanged |

Benchmarks: [`benchmarks/`](benchmarks) · training report (with the full
accident log): [`docs/reports/2026-09-25-laya-head-training-report.md`](docs/reports/2026-09-25-laya-head-training-report.md)

## Project layout

```
src/            Zig runtime (api / core / graph / runtime / calib / model)
tools/          CLI entry points (serve/fit/bench/traj/conformance)
examples/       sample requests (flat / batch / graph)
datasets/       eval & calibration datasets (JSONL)
model/calibration/  fitted temperature profiles
export/laya/    Laya model pipeline (dataset build, training, ONNX export, demo)
benchmarks/     measured readings, before/after comparisons
docs/           RFC, specs, plans, user manual, reports
```

## Documentation

| Doc | Language | Content |
|---|---|---|
| [`docs/user-manual.md`](docs/user-manual.md) | EN | install, API reference, calibration workflow, troubleshooting |
| [`docs/rfc-0001-zjev-decision-runtime.md`](docs/rfc-0001-zjev-decision-runtime.md) | ZH | protocol & theory |
| [`docs/prd.md`](docs/prd.md), [`docs/quest1.md`](docs/quest1.md) | ZH | product requirements; calibration thesis (§12) |
| [`docs/specs/`](docs/specs), [`docs/plans/`](docs/plans) | ZH | design specs & implementation plans |
| [`docs/reports/`](docs/reports) | ZH | training reports with incident log |
| [`benchmarks/`](benchmarks) | ZH | measured readings |

## Known limitations

- The bundled head is trained on **synthetic template tickets** — expect
  wrong labels on natural text (calibration makes confidence honest, not
  correct). Real-distribution data is the top follow-up.
- `escalate` and `urgency` labels are collinear by construction
  (escalate ⇒ gold urgency = high).
- For `noul` decisions, `uncertainty.confidence` is **P(yes)** (detector
  semantics, required by gates) — node ECE in `zjev-traj` and `zjev-fit` use
  different confidence conventions; see the benchmark notes.
- `zjev-traj`/`zjev-fit` on CPU run ~2 h per 5,652 records with the large
  model.

## Contributing

Build, test, and conventions: [`CONTRIBUTING.md`](CONTRIBUTING.md).
History: [`CHANGELOG.md`](CHANGELOG.md).

## License

[MIT](LICENSE)
