# Changelog

## [Unreleased]

- Open-source docs: English README + user manual, LICENSE (MIT),
  CONTRIBUTING, CHANGELOG.

## M4 — 2026-09-25 — Temperature calibration

End-to-end temperature calibration for the Laya decision bundle, no
retraining.

- `zjev-fit`: `--model/--bundle/--sessions/--ort-extensions` for fitting on
  real ONNX bundles (segment table from the bundle schema, per-state forward
  caching, `fitted/skipped` reporting).
- `export/laya/build_calib.py`: expands the eval split to fit format
  (3 lines/record).
- Fitted profiles (`model/calibration/laya_*.json`): escalate T=9.61,
  topic T=4.82, urgency T=7.97.
- `zjev-traj --profiles-dir` re-measurement, 5,652-ticket eval:
  traj_ece 0.200 → 0.106, traj_brier −14%, trajectory_accuracy unchanged
  at 0.754. Benchmark: `benchmarks/temp_laya_2026-09-25.md`.
- `export/laya/demo_profiles.py`: side-by-side serve demo (with/without
  profiles).

## M3 — 2026-09-25 — Unfreeze-last-2 fine-tune + tokenizer fix

- `--unfreeze-last` fine-tune with encoder-tail export override; best
  epoch 15 (early stop 18), ~11.4 h on MPS. Eval: escalate 0.786 /
  topic 0.847 / urgency 0.686.
- In-graph `[CLS]/[SEP]` fix: `HfJsonTokenizer` drops post-processing, so
  the export now concatenates special tokens inside the graph. Before the
  fix, ORT escalate acc was 0.61 vs torch 0.79; after: 0.78, cross-engine
  agreement 98.4%. Earlier M2 readings (broken graph) corrected.
- Fixed-graph trajectory readings: trajectory_accuracy **0.754**,
  escalate acc 0.785 / ece 0.586, topic acc 0.888 / ece 0.093.

## M2 — 2026-09-24/25 — Laya frozen head

- `export/laya/` pipeline: dataset build → frozen-head training
  (Linear(1024,8), MPS) → ONNX export with in-graph tokenizer → smoke check.
- End-to-end real-model serving through the ONNX backend.
- Incident fixed: the Laya checkpoint's `encoder.` prefix is not auto-stripped
  by transformers 5.x — `load_laya_encoder()` loads it manually and asserts
  bit-exact equality (earlier exports had random encoders).

## V0.2 — 2026-09-24 — Decision graph

- `POST /v1/execute`: nodes/edges/gates, `when` conditions compiled to AST
  with static type checking, activation-wave execution with full prefetch
  (one encoder forward for the whole graph, bundled ONNX heads included).
- `zjev-traj`: node-vs-trajectory calibration reporting (quest1.md §12).

## V0.1 — 2026-09-23/24 — Core runtime

- Typed probabilistic decisions: noul / choice / score / rank with optional
  abstention.
- Uncertainty reporting: confidence / entropy / variance / abstention.
- `zjev-serve` (mock + ONNX backends), `zjev-fit`, `zjev-bench`,
  `zjev-conformance`.
- Temperature fitting + profiles; selective-risk reporting.
- Calibration protocol & RFC.
