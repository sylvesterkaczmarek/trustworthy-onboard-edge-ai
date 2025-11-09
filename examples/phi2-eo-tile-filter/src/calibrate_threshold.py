import argparse, json, time
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort
from sklearn.metrics import precision_recall_curve, roc_auc_score

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def load_set(root: Path):
    files = []
    for cls, name in enumerate(["background", "event"]):
        for f in sorted((root / name).glob("*.png")):
            files.append((f, cls))
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--data", required=True, help="path to val split (…/tiles/val)")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--target_recall", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="calibration.json")
    a = ap.parse_args()

    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    xs, ys = [], []
    t0 = time.time()
    for f, cls in load_set(Path(a.data)):
        img = Image.open(f).convert("RGB").resize((a.size, a.size))
        x = np.transpose(np.asarray(img, dtype=np.float32) / 255.0, (2, 0, 1))[None, ...]
        logits = sess.run(None, {"input": x})[0]
        if a.temperature != 1.0:
            logits = logits / a.temperature
        prob = softmax(logits)[0]
        xs.append(prob[1]); ys.append(cls)
    dur = time.time() - t0
    xs = np.array(xs); ys = np.array(ys)

    precision, recall, thresholds = precision_recall_curve(ys, xs)
    # pick the smallest threshold that achieves target recall
    meet = np.where(recall[:-1] >= a.target_recall)[0]
    if len(meet) == 0:
        idx = np.argmax(recall[:-1])  # fall back to best possible recall
    else:
        idx = meet[-1]
    thr = float(thresholds[idx])
    auc = float(roc_auc_score(ys, xs))
    out = {
        "threshold": thr,
        "target_recall": a.target_recall,
        "achieved_recall": float(recall[idx]),
        "precision_at_threshold": float(precision[idx]),
        "auc_roc": auc,
        "temperature": a.temperature,
        "val_samples": int(len(ys)),
        "duration_s": float(dur)
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", a.out, out)

if __name__ == "__main__":
    main()
