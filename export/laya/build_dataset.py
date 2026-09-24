#!/usr/bin/env python3
"""Map Tobi-Bueck/customer-support-tickets (en) to the ZJEV 8-logit bundle.

Dataset columns (recon 2026-09-24): subject/body/answer/type/queue/priority/
language/version/tag_1..tag_8. Priority values: very_low/low/medium/high/critical.

Outputs (under <repo>/datasets/):
  support_bundle_train.jsonl       {"text","escalate","topic","urgency"}  train split
  support_bundle_eval_flat.jsonl   same format, eval split (for train_head.py)
  support_bundle_eval.jsonl        zjev-traj records (state/decisions/graph/expected)
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

TOPIC_ORDER = ("billing", "bug", "other")
URGENCY_ORDER = ("low", "medium", "high")
SEED = 42
ROOT = Path(__file__).resolve().parents[2]

URGENCY_MAP = {"very_low": "low", "low": "low", "medium": "medium",
               "high": "high", "critical": "high"}
ESCALATE_PRIORITIES = {"high", "critical"}

BILLING = re.compile(r"bill|refund|invoice|payment|charge|pricing|price|subscription|fee", re.I)
BUG = re.compile(r"bug|error|crash|broken|outage|fail|not working|doesn'?t work|incident|disruption|defect", re.I)


def topic_of(type_str: str, tags: str, head: str) -> str:
    text = f"{type_str} {tags} {head}"
    if BILLING.search(text):
        return "billing"
    if BUG.search(text):
        return "bug"
    return "other"


def main():
    ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    rows = [r for r in ds if (r.get("language") or "").lower() == "en"]
    print(f"[build_dataset] en rows: {len(rows)} / {len(ds)}")

    stats = Counter()
    recs = []
    for r in rows:
        body = (r["body"] or "").strip()
        subj = (r["subject"] or "").strip()
        text = f"{subj}\n{body}" if subj else body
        if len(text) < 20:
            continue
        pri = (r.get("priority") or "").lower()
        esc = pri in ESCALATE_PRIORITIES
        urg = URGENCY_MAP.get(pri, "medium")
        tags = " ".join(str(r.get(f"tag_{i}") or "") for i in range(1, 9))
        topic = topic_of(r.get("type") or "", tags, f"{subj} {body[:200]}")
        stats[topic] += 1
        stats[f"urgency:{urg}"] += 1
        stats[f"escalate:{esc}"] += 1
        stats[f"type:{r.get('type')}"] += 1
        recs.append({"text": text, "escalate": esc, "topic": topic, "urgency": urg})

    print("[build_dataset] mapping stats:", dict(stats))
    assert len(recs) >= 800, f"only {len(recs)} usable en rows; relax filters or pick another dataset"

    rng = random.Random(SEED)
    rng.shuffle(recs)
    # 按 urgency 分层 80/20
    by_urg = {u: [r for r in recs if r["urgency"] == u] for u in URGENCY_ORDER}
    train, evals = [], []
    for group in by_urg.values():
        cut = max(1, int(len(group) * 0.8))
        train += group[:cut]
        evals += group[cut:]
    rng.shuffle(train)
    rng.shuffle(evals)

    (ROOT / "datasets").mkdir(exist_ok=True)
    with open(ROOT / "datasets/support_bundle_train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open(ROOT / "datasets/support_bundle_eval_flat.jsonl", "w") as f:
        for r in evals:
            f.write(json.dumps(r) + "\n")

    with open(ROOT / "datasets/support_bundle_eval.jsonl", "w") as f:
        for r in evals:
            expected = {"escalate": r["escalate"]}
            if r["escalate"]:
                expected["topic"] = r["topic"]
                expected["urgency"] = URGENCY_ORDER.index(r["urgency"])
            rec = {
                "state": {"text": r["text"]},
                "decisions": [
                    {"id": "escalate", "type": "noul", "abstain": False},
                    {"id": "topic", "type": "choice", "options": list(TOPIC_ORDER)},
                    {"id": "urgency", "type": "score", "scale": {"labels": list(URGENCY_ORDER)}},
                ],
                "graph": {
                    "nodes": [
                        {"id": "esc", "decision": "escalate"},
                        {"id": "topic", "decision": "topic"},
                        {"id": "urg", "decision": "urgency"},
                    ],
                    "edges": [
                        {"from": "esc", "to": "topic", "when": "escalate == true"},
                        {"from": "esc", "to": "urg", "when": "escalate == true"},
                    ],
                },
                "expected": expected,
            }
            f.write(json.dumps(rec) + "\n")
    print(f"[build_dataset] train={len(train)} eval={len(evals)}")


if __name__ == "__main__":
    main()
