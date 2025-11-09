import argparse, json
from pathlib import Path
from statistics import mean

ap = argparse.ArgumentParser()
ap.add_argument("--val_log", required=True)
ap.add_argument("--downlink_log", required=True)
ap.add_argument("--val_dir", required=True)
ap.add_argument("--calib", required=True)
ap.add_argument("--out_dir", required=True)
a = ap.parse_args()

out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

def read_jsonl(p):
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def is_kept(rec, thr=None):
    if "kept" in rec:
        return bool(rec["kept"])
    if "decision" in rec:
        return str(rec["decision"]).lower() in ("keep", "kept", "true", "1")
    if "prob" in rec:
        t = rec.get("threshold", thr)
        return (t is not None) and float(rec["prob"]) >= float(t)
    # default to NOT kept if no signal is present
    return False

val  = list(read_jsonl(a.val_log))
down = list(read_jsonl(a.downlink_log))
cal  = json.load(open(a.calib))

tiles_total = len(val)
tiles_kept  = sum(1 for r in down if is_kept(r, cal.get("threshold")))
saved_pct   = 100.0 * (1.0 - tiles_kept / max(1, tiles_total))
avg_latency_ms = mean([r.get("latency_ms", 0.0) for r in val]) if val else 0.0

precision = cal.get("precision_at_threshold")
recall    = cal.get("achieved_recall")
f1 = (2*precision*recall/(precision+recall)) if precision and recall and (precision+recall)>0 else 0.0

metrics = {
    "threshold": cal.get("threshold"),
    "target_recall": cal.get("target_recall"),
    "achieved_recall": recall,
    "precision": precision,
    "f1": f1,
    "auc_roc": cal.get("auc_roc"),
    "avg_latency_ms": avg_latency_ms,
    "tiles_total": tiles_total,
    "tiles_kept": tiles_kept,
    "bandwidth_saved_pct": round(saved_pct, 1),
}

json.dump(metrics, open(out/"metrics.json", "w"), indent=2)
with open(out/"summary.md", "w") as f:
    f.write("# Run summary\n\n")
    for k, v in metrics.items():
        f.write(f"- **{k}**: {v}\n")
print(f"wrote {out/'metrics.json'} and {out/'summary.md'}")
