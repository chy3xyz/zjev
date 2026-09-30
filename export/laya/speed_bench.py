#!/usr/bin/env python3
"""Speed benchmark: original Laya encoder (torch) vs ZJEV ONNX graph (ORT), same texts."""
import json, time, statistics
import numpy as np

N = 200
texts = []
for line in open("datasets/support_bundle_eval.jsonl"):
    texts.append(json.loads(line)["state"]["text"])
    if len(texts) >= N:
        break
lens = [len(t.split()) for t in texts]
print(f"sample: {len(texts)} texts, words avg {np.mean(lens):.0f} max {max(lens)}")

# --- torch side: original laya encoder (ModernBERT-large) ---
import torch
from transformers import AutoModel, AutoTokenizer

SNAP = __import__("pathlib").Path.home() / ".cache/huggingface/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
from transformers import ModernBertModel, AutoConfig
from safetensors.torch import load_file
tok = AutoTokenizer.from_pretrained(SNAP / "tokenizer")
enc = ModernBertModel(AutoConfig.from_pretrained(SNAP / "encoder"))
sd = load_file(str(SNAP / "model.safetensors"))
base = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
missing, unexpected = enc.load_state_dict(base, strict=False)
assert not missing, missing[:5]
enc.eval()

def bench_torch(device, dtype):
    m = enc.to(device=device, dtype=dtype)
    batch = [tok(t, truncation=True, max_length=128, return_tensors="pt") for t in texts]
    with torch.no_grad():
        # warmup
        for i in range(5):
            m(**{k: v.to(device) for k, v in batch[i % 10].items()})
        if device == "mps":
            torch.mps.synchronize()
        ts = []
        for b in batch:
            x = {k: v.to(device) for k, v in b.items()}
            torch.cuda.synchronize if False else None
            t0 = time.perf_counter()
            m(**x)
            if device == "mps":
                torch.mps.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), statistics.mean(ts)

# --- ORT side: our laya.onnx (tokenizer in-graph) ---
import onnxruntime as ort
so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.register_custom_ops_library("export/laya/lib/libortextensions.dylib")
sess = ort.InferenceSession("export/laya/out/laya.onnx",
                            sess_options=so,
                            providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name

def bench_ort():
    arr = np.array(texts, dtype=object)
    for i in range(5):  # warmup
        sess.run(None, {in_name: arr[i:i+1].reshape(1)})
    ts = []
    for t in texts:
        t0 = time.perf_counter()
        sess.run(None, {in_name: np.array([t], dtype=object)})
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), statistics.mean(ts)

med_cpu, avg_cpu = bench_torch("cpu", torch.float32)
print(f"torch  CPU fp32 : median {med_cpu:7.1f} ms  mean {avg_cpu:7.1f} ms  ({1000/med_cpu:.1f} fwd/s)")
if torch.backends.mps.is_available():
    med_mps, avg_mps = bench_torch("mps", torch.float32)
    print(f"torch  MPS fp32 : median {med_mps:7.1f} ms  mean {avg_mps:7.1f} ms  ({1000/med_mps:.1f} fwd/s)")
med_o, avg_o = bench_ort()
print(f"ort    CPU graph: median {med_o:7.1f} ms  mean {avg_o:7.1f} ms  ({1000/med_o:.1f} fwd/s)  <- zjev")
