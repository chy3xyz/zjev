#!/usr/bin/env python3
"""Self-check the exported laya.onnx under onnxruntime + extensions."""
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime_extensions  # noqa: F401  (registers ai.onnx.contrib kernels)

MODEL = Path(__file__).parent / "out" / "laya.onnx"
TEXT = ["We were billed twice for March. Refund the duplicate today or we cancel."]


def main():
    m = onnx.load(str(MODEL))
    ins = [i.name for i in m.graph.input]
    assert ins == ["text"], f"inputs={ins}"
    init_names = {i.name for i in m.graph.initializer}
    assert {"cls_id", "sep_id"} <= init_names, "special-token constants missing in graph"
    out = m.graph.output[0]
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    assert out.name == "logits" and dims == [1, 8], f"output={out.name} dims={dims}"

    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    so.register_custom_ops_library(onnxruntime_extensions.get_library_path())
    sess = ort.InferenceSession(str(MODEL), so, providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"text": TEXT})[0]
    assert got.shape == (1, 8), got.shape
    assert np.isfinite(got).all(), "non-finite logits"
    got2 = sess.run(["logits"], {"text": TEXT})[0]
    assert np.allclose(got, got2, atol=1e-5), "non-deterministic"
    print("OK logits[0]:", got[0].tolist())


if __name__ == "__main__":
    main()
