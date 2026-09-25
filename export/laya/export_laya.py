#!/usr/bin/env python3
"""Export the Laya English encoder to a ZJEV-contract ONNX graph (v0 plumbing).

Data flow:
  text:string[1] -> BertTokenizer (ai.onnx.contrib) -> input_ids/attention_mask
  -> ModernBERT encoder (Laya English weights) -> [CLS] -> Linear(1024->8, random)
  -> logits:float[1,8] (static)

The HF repo layout keeps config/weights/tokenizer in separate subdirs, so we
stage symlinks into export/laya/.stage before loading with transformers.
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.compose import merge_models
from transformers import AutoModel
import torch

NUM_LOGITS = 8          # escalate(2) + topic(3) + urgency(3)
SEED = 42
HEAD_SCALE = 0.02
REPO_ID = "convaiinnovations/laya"
STAGE = Path(__file__).parent / ".stage"


def log(*a):
    print("[export_laya]", *a, flush=True)


def stage_files(repo_id: str) -> Path:
    from huggingface_hub import hf_hub_download
    files = {
        "config.json": "encoder/config.json",
        "model.safetensors": "model.safetensors",
        "tokenizer.json": "tokenizer/tokenizer.json",
        "tokenizer_config.json": "tokenizer/tokenizer_config.json",
    }
    STAGE.mkdir(parents=True, exist_ok=True)
    for dst, src in files.items():
        cached = Path(hf_hub_download(repo_id, src))
        link = STAGE / dst
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(cached)
        log("staged", dst, "->", cached)
    return STAGE


def load_laya_encoder(stage_dir: str):
    """Load the Laya ModernBERT with REAL weights.

    The HF repo's checkpoint prefixes the base encoder with "encoder."
    (plus Laya's own act_head/scorer/head extras). transformers 5.x does NOT
    auto-strip the prefix: naive from_pretrained silently leaves the whole
    encoder randomly initialized (LOAD REPORT: 0 direct hits, all MISSING).
    We strip the prefix and load manually; Laya's downstream heads are dropped.
    """
    from safetensors.torch import load_file
    model = AutoModel.from_pretrained(stage_dir)
    sd = load_file(str(Path(stage_dir) / "model.safetensors"))
    base = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = model.load_state_dict(base, strict=False)
    assert not missing, f"missing keys after prefix strip: {missing[:5]}"
    log(f"encoder weights loaded: {len(base)} tensors "
        f"(dropped Laya heads: {len(unexpected)})")
    return model


def export_encoder(model, out_path: str):
    ids = torch.ones(1, 8, dtype=torch.long)
    mask = torch.ones(1, 8, dtype=torch.long)
    torch.onnx.export(
        model, (ids, mask), out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {1: "seq"},
            "attention_mask": {1: "seq"},
            "last_hidden_state": {1: "seq"},
        },
        opset_version=18,
        dynamo=True,
    )
    log("encoder exported:", out_path)


def tokenizer_graph(stage_dir: str, pad_id: int):
    """HfJsonTokenizer graph (schema v2) + in-graph attention_mask derivation.

    HfJsonTokenizer emits only `ids`; ModernBERT also needs `attention_mask`,
    derived as ids != pad_id (the tokenizer never pads, so this is all-ones
    in practice, but shape-correct for any input).
    Returns (model, ids_output_name).
    """
    from onnxruntime_extensions import gen_processing_models
    pre, _post = gen_processing_models(
        stage_dir, pre_kwargs={"WITH_DEFAULT_INPUTS": True}, schema_v2=True)
    m = onnx.shape_inference.infer_shapes(pre)
    # the extensions builder tags standard ops with the literal domain
    # "ai.onnx"; the onnx schema registry only knows "", so normalize.
    for n in m.graph.node:
        if n.domain == "ai.onnx":
            n.domain = ""
    for o in m.opset_import:
        if o.domain == "ai.onnx":
            o.domain = ""
    g = m.graph
    assert len(g.input) == 1, f"tokenizer graph inputs: {[i.name for i in g.input]}"
    old = g.input[0].name
    g.input[0].name = "text"
    for n in g.node:
        for i, s in enumerate(n.input):
            if s == old:
                n.input[i] = "text"
    ids_out = g.output[0].name
    assert ids_out == "ids", ids_out
    ids_type = g.output[0].type.tensor_type.elem_type
    g.output.pop()

    g.initializer.append(numpy_helper.from_array(
        np.array(pad_id, dtype=np.int64), "pad_id"))
    ids_i64 = ids_out
    if ids_type != onnx.TensorProto.INT64:
        g.node.append(helper.make_node(
            "Cast", [ids_out], ["ids_i64"], to=onnx.TensorProto.INT64))
        ids_i64 = "ids_i64"
    g.node.append(helper.make_node(
        "Equal", [ids_i64, "pad_id"], ["mask_eq"]))
    g.node.append(helper.make_node(
        "Not", ["mask_eq"], ["mask_ne"]))
    g.node.append(helper.make_node(
        "Cast", ["mask_ne"], ["attention_mask"], to=onnx.TensorProto.INT64))
    g.output.append(helper.make_tensor_value_info(
        ids_i64, onnx.TensorProto.INT64, ["N", None]))
    g.output.append(helper.make_tensor_value_info(
        "attention_mask", onnx.TensorProto.INT64, ["N", None]))
    return m, ids_i64


def add_head(merged: onnx.ModelProto, hidden_size: int, head_path: str | None = None) -> onnx.ModelProto:
    """[CLS] -> Linear(hidden->NUM_LOGITS) -> logits[1,NUM_LOGITS] (static).

    head_path: trained checkpoint {"weight"[8,H], "bias"[8]} (torch .pt);
    default random init (seed 42) for v0 plumbing.
    """
    g = merged.graph
    lhs = g.output[0].name            # last_hidden_state [1,seq,H]
    g.output.pop()

    if head_path:
        sd = torch.load(head_path, map_location="cpu")
        W = sd["weight"].numpy().T.astype(np.float32)  # [8,H] -> [H,8]
        b = sd["bias"].numpy().astype(np.float32)
        log("loaded trained head:", head_path)
    else:
        rng = np.random.default_rng(SEED)
        W = (rng.standard_normal((hidden_size, NUM_LOGITS)) * HEAD_SCALE).astype(np.float32)
        b = np.zeros(NUM_LOGITS, dtype=np.float32)
    assert W.shape == (hidden_size, NUM_LOGITS), W.shape

    init = lambda name, arr: g.initializer.append(numpy_helper.from_array(arr, name))
    init("head_W", W)
    init("head_b", b)
    init("cls_idx", np.array([0], dtype=np.int64))
    init("cls_axes", np.array([1], dtype=np.int64))

    g.node.append(helper.make_node("Gather", [lhs, "cls_idx"], ["cls_gather"], axis=1))
    g.node.append(helper.make_node("Squeeze", ["cls_gather", "cls_axes"], ["cls_vec"]))
    g.node.append(helper.make_node("MatMul", ["cls_vec", "head_W"], ["head_mm"]))
    g.node.append(helper.make_node("Add", ["head_mm", "head_b"], ["logits"]))
    g.output.append(helper.make_tensor_value_info(
        "logits", onnx.TensorProto.FLOAT, [1, NUM_LOGITS]))
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--out", default=str(Path(__file__).parent / "out" / "laya.onnx"))
    ap.add_argument("--head", default=None,
                    help="trained head .pt (weight[8,H]/bias[8]); default random seed 42")
    ap.add_argument("--encoder-tail", default=None,
                    help="fine-tuned encoder tail .pt (state_dict of thawed params)")
    a = ap.parse_args()

    stage = stage_files(a.repo_id)
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(stage / "tokenizer.json"))
    pad_id = tk.token_to_id("[PAD]")
    assert pad_id is not None, "no [PAD] token in tokenizer"
    log("pad_id:", pad_id)
    model = load_laya_encoder(str(stage))
    if a.encoder_tail:
        tail = torch.load(a.encoder_tail, map_location="cpu")
        msd = model.state_dict()
        missing = [k for k in tail if k not in msd]
        assert not missing, f"encoder-tail keys not in model: {missing[:5]}"
        msd.update(tail)
        model.load_state_dict(msd)
        log(f"encoder tail overridden: {a.encoder_tail} ({len(tail)} tensors)")
    model.eval()
    hidden = model.config.hidden_size
    assert hidden == 1024, hidden
    log("hidden_size:", hidden)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_enc = out.parent / "_encoder_tmp.onnx"
    export_encoder(model, str(tmp_enc))
    enc = onnx.load(str(tmp_enc))
    enc_ins = [i.name for i in enc.graph.input]
    log("encoder inputs:", enc_ins)
    enc_ids = next(n for n in enc_ins if "input_ids" in n)
    enc_mask = next(n for n in enc_ins if "attention_mask" in n)

    tok_g, ids_out_name = tokenizer_graph(str(stage), pad_id)
    tok_g.ir_version = enc.ir_version

    merged = merge_models(tok_g, enc, io_map=[
        (ids_out_name, enc_ids),
        ("attention_mask", enc_mask),
    ])
    del merged.opset_import[:]
    merged.opset_import.append(onnx.helper.make_opsetid("", 18))
    merged.opset_import.append(onnx.helper.make_opsetid("ai.onnx.contrib", 1))
    final = add_head(merged, hidden, a.head)

    onnx.checker.check_model(final)
    onnx.save(final, a.out)
    tmp_enc.unlink()
    ins = [i.name for i in final.graph.input]
    outs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in final.graph.output]
    log("saved:", a.out, "inputs:", ins, "outputs:", outs)
    assert ins == ["text"], ins


if __name__ == "__main__":
    main()
