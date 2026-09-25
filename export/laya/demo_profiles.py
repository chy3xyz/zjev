#!/usr/bin/env python3
"""M4 demo: same tickets -> serve with/without temperature profiles, side by side.

起两个实例后运行本脚本：
  ./zig-out/bin/zjev-serve --model export/laya/out/laya.onnx \
      --ort-extensions export/laya/lib/libortextensions.dylib \
      --profiles-dir model/calibration --port 8790          # 带标定
  ./zig-out/bin/zjev-serve --model export/laya/out/laya.onnx \
      --ort-extensions export/laya/lib/libortextensions.dylib --port 8791   # 无标定
  python3 export/laya/demo_profiles.py
"""
import json
import urllib.request

DECISIONS = [
    {"id": "escalate", "type": "noul", "abstain": False},
    {"id": "topic", "type": "choice", "options": ["billing", "bug", "other"]},
    {"id": "urgency", "type": "score", "scale": {"labels": ["low", "medium", "high"]}},
]

TICKETS = [
    ("该升级·billing", "URGENT: you double-charged my card $2,400 this month. I have emailed three times with no reply. I want a manager to call me back today or I am disputing every charge with my bank."),
    ("该升级·bug", "CRITICAL: since yesterday's update the app crashes on startup for our entire team of 40. We are completely blocked on a client deliverable. This needs immediate engineering escalation."),
    ("边界·billing", "I was charged twice for the same invoice in January and February. It is not a huge amount but I would like the duplicate refunded before my next billing cycle please."),
    ("不该升级·howto", "How do I export my data to CSV? I looked in the settings but could not find the option. Thanks!"),
    ("不该升级·other", "Do you offer a student discount on the annual plan? A friend of mine said there was one but I cannot find it on the pricing page."),
    ("模糊·bug", "Sometimes the search returns no results even though I know the document exists. Refreshing usually fixes it. Not sure if it is just me."),
]

def ask(port, text):
    body = json.dumps({"state": {"text": text}, "decisions": DECISIONS}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/decide", data=body,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

def pick(resp, did):
    for d in resp["results"]:
        if d["id"] == did:
            return d
    return None

rows = []
for name, text in TICKETS:
    raw = ask(8791, text)
    cal = ask(8790, text)
    r_esc, c_esc = pick(raw, "escalate"), pick(cal, "escalate")
    r_top, c_top = pick(raw, "topic"), pick(cal, "topic")
    rows.append({
        "case": name,
        "esc_raw": r_esc["probability"], "esc_cal": c_esc["probability"],
        "esc_raw_conf": r_esc["uncertainty"]["confidence"], "esc_cal_conf": c_esc["uncertainty"]["confidence"],
        "topic_raw": r_top["value"], "topic_raw_conf": r_top["uncertainty"]["confidence"],
        "topic_cal": c_top["value"], "topic_cal_conf": c_top["uncertainty"]["confidence"],
    })

print(f"{'case':<18}{'esc P(yes) 无标定':>18}{'esc P(yes) 标定':>16}{'判定':>6}   topic（无标定 conf → 标定 conf）")
for r in rows:
    esc = "升级" if r["esc_cal"] >= 0.5 else "不升级"
    same = " 一致" if r["topic_raw"] == r["topic_cal"] else " 翻转!"
    print(f"{r['case']:<18}{r['esc_raw']:>18.4f}{r['esc_cal']:>16.4f}{esc:>6}   "
          f"{r['topic_raw']} {r['topic_raw_conf']:.3f} → {r['topic_cal_conf']:.3f}{same}")

print("\n说明：P(yes)>=0.5 判定升级（温度保序，两实例判定必然一致）；"
      "conf = 预测类置信度（noul 的 uncertainty.confidence 即 P(yes)）。")
