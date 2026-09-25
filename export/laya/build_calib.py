#!/usr/bin/env python3
"""Expand support_bundle_eval_flat.jsonl to zjev-fit format (3 lines/record)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOPIC_ORDER = ("billing", "bug", "other")
URGENCY_ORDER = ("low", "medium", "high")

DECISIONS = [
    {"id": "escalate", "type": "noul", "abstain": False},
    {"id": "topic", "type": "choice", "options": list(TOPIC_ORDER)},
    {"id": "urgency", "type": "score", "scale": {"labels": list(URGENCY_ORDER)}},
]


def main():
    out = open(ROOT / "datasets/support_bundle_calib.jsonl", "w")
    n = 0
    for line in open(ROOT / "datasets/support_bundle_eval_flat.jsonl"):
        if not line.strip():
            continue
        r = json.loads(line)
        labels = [r["escalate"], r["topic"], r["urgency"]]  # score labels scale: label 必须是字符串
        for dec, lab in zip(DECISIONS, labels):
            out.write(json.dumps({"state": {"text": r["text"]}, "decision": dec, "label": lab}) + "\n")
            n += 1
    out.close()
    print(f"[build_calib] wrote {n} lines")


if __name__ == "__main__":
    main()
