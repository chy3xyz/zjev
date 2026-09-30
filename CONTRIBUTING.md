# Contributing to ZJEV

Thanks for your interest in contributing. This document covers how to build,
test, and land changes.

## Build & test

```bash
zig build                          # default binaries
zig build test                     # unit tests — must pass
zig build test-conformance         # protocol fixtures — expect 10 pass, 0 fail
zig build -Donnx=true -Donnx_lib_dir=$PWD/export/laya/lib test   # ONNX-backend tests
```

Before merging any branch, all three suites above must be green.

> **Gotcha**: a plain `zig build` overwrites the ONNX-enabled binaries in
> `zig-out/`. If you intend to run `--model` tools afterwards, rebuild with
> `-Donnx=true` first.

## Repository conventions

- **One task, one branch, one merge**: branch off `main` as
  `feat/<topic>` / `fix/<topic>` / `docs(<scope>)`, merge back with
  `--no-ff`, delete the branch.
- **Artifacts stay out of git**: `zig-out/`, `export/laya/out/`,
  `export/laya/.venv/`, `export/laya/lib/`, `__pycache__/` are ignored.
  Only source, datasets, docs, and fitted calibration profiles
  (`model/calibration/*.json`) are tracked.
- **Docs**: user-facing docs in English (`README.md`,
  `docs/user-manual.md`); historical design docs and reports may stay in
  Chinese — don't churn them without reason.
- **Measured claims**: benchmark numbers belong in `benchmarks/` with the
  exact command, dataset, and model revision used to produce them.

## Code style

- Match the file's existing conventions; Zig fmt conventions apply.
- No decorative comments — let the code and its history speak.
- New tests only where the project already has tests for that area.

## Reporting issues

Include: the command, the request/response bodies (or dataset sample), the
error code, and the output of `zig build test` + `zig build test-conformance`.
Check §10 (FAQ) of the user manual first — most common failures are listed
there.
