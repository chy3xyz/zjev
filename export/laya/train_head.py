#!/usr/bin/env python3
"""Train the 8-logit decision head on a frozen Laya encoder (MPS if available).

Loss: 3 x CE over logits segments [0:2] escalate / [2:5] topic / [5:8] urgency
(label order matches the export contract, see spec §3 of
docs/superpowers/specs/2026-09-24-m2-trained-head-design.md).
"""
import argparse
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from export_laya import NUM_LOGITS, REPO_ID, load_laya_encoder, stage_files

TOPIC_ORDER = ("billing", "bug", "other")
URGENCY_ORDER = ("low", "medium", "high")
SEGMENTS = ((0, 2), (2, 5), (5, 8))
NAMES = ("escalate", "topic", "urgency")


class Jsonl(Dataset):
    def __init__(self, path, tok, max_len=128):
        self.rows = [json.loads(l) for l in open(path) if l.strip()]
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["text"], truncation=True, max_length=self.max_len, return_tensors="pt")
        y = torch.tensor([
            0 if r["escalate"] else 1,
            TOPIC_ORDER.index(r["topic"]),
            URGENCY_ORDER.index(r["urgency"]),
        ])
        return enc["input_ids"][0], enc["attention_mask"][0], y


def collate(batch):
    n = max(ids.shape[0] for ids, _, _ in batch)
    ids = torch.zeros(len(batch), n, dtype=torch.long)
    mask = torch.zeros(len(batch), n, dtype=torch.long)
    y = torch.stack([t for _, _, t in batch])
    for i, (b_ids, b_mask, _) in enumerate(batch):
        ids[i, : b_ids.shape[0]] = b_ids
        mask[i, : b_mask.shape[0]] = b_mask
    return ids, mask, y


def run(model, head, loader, device, opt=None):
    train = opt is not None
    model.eval()
    head.train(train)
    lossf = nn.CrossEntropyLoss()
    tot = [0, 0, 0]
    correct = [0, 0, 0]
    loss_sum = 0.0
    for ids, mask, y in loader:
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            hs = model(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
            logits = head(hs)
            loss = sum(lossf(logits[:, s:e], y[:, k]) for k, (s, e) in enumerate(SEGMENTS))
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        loss_sum += loss.item()
        for k, (s, e) in enumerate(SEGMENTS):
            tot[k] += y.shape[0]
            correct[k] += (logits[:, s:e].argmax(1) == y[:, k]).sum().item()
    acc = [c / t for c, t in zip(correct, tot)]
    return loss_sum / max(1, len(loader)), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="datasets/support_bundle_train.jsonl")
    ap.add_argument("--eval", default="datasets/support_bundle_eval_flat.jsonl")
    ap.add_argument("--out", default="export/laya/out/head.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    a = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("[train] device:", device)
    stage = stage_files(REPO_ID)
    tok = AutoTokenizer.from_pretrained(str(stage))
    model = load_laya_encoder(str(stage))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    head = nn.Linear(model.config.hidden_size, NUM_LOGITS).to(device)

    train_dl = DataLoader(Jsonl(a.train, tok), batch_size=32, shuffle=True, collate_fn=collate)
    eval_dl = DataLoader(Jsonl(a.eval, tok), batch_size=64, shuffle=False, collate_fn=collate)

    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    best = -1.0
    bad = 0
    metrics = {}
    for epoch in range(a.epochs):
        tl, ta = run(model, head, train_dl, device, opt)
        el, ea = run(model, head, eval_dl, device)
        score = sum(ea) / 3
        ta_s = " ".join(f"{NAMES[k]}={ta[k]:.4f}" for k in range(3))
        ea_s = " ".join(f"{NAMES[k]}={ea[k]:.4f}" for k in range(3))
        print(f"[train] epoch {epoch}: train_loss={tl:.4f} | train {ta_s} | eval {ea_s} | avg={score:.4f}", flush=True)
        if score > best:
            best = score
            bad = 0
            metrics = {"epoch": epoch, "eval_acc": dict(zip(NAMES, ea)), "train_acc": dict(zip(NAMES, ta))}
            torch.save({"weight": head.weight.detach().cpu(), "bias": head.bias.detach().cpu()}, a.out + ".best")
        else:
            bad += 1
            if bad >= a.patience:
                print(f"[train] early stop at epoch {epoch} (best avg={best:.4f})")
                break
    sd = torch.load(a.out + ".best", map_location="cpu")
    sd["metrics"] = metrics
    torch.save(sd, a.out)
    print("[train] saved:", a.out, "metrics:", metrics)


if __name__ == "__main__":
    main()
