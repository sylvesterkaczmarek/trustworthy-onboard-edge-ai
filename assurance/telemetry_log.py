import argparse, json, time, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--out", type=str, default="logs/val.jsonl")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--temperature", type=float, default=1.0)
    a = ap.parse_args()

    root = Path(a.data)
    files = []
    for cls, name in enumerate(["background", "event"]):
        files += [(f, cls) for f in sorted((root / name).glob("*.png"))]

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    f = open(out, "w")

    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    model_sha = sha256(Path(a.onnx))

    for pth, cls in files:
        img = Image.open(pth).convert("RGB").resize((a.size, a.size))
        x = np.transpose(np.asarray(img, dtype=np.float32) / 255.0, (2, 0, 1))[None, ...]
        t0 = time.time()
        logits = sess.run(None, {"input": x})[0]
        if a.temperature != 1.0:
            logits = logits / a.temperature
        prob = softmax(logits)[0]
        latency_ms = (time.time() - t0) * 1000
        rec = {
            "timestamp": time.time(),
            "file": str(pth),
            "true_class": int(cls),
            "pred_class": int(prob.argmax()),
            "max_prob": float(prob.max()),
            "prob_event": float(prob[1]),
            "threshold": float(a.threshold),
            "ok_flag": bool(prob.max() >= a.threshold),
            "latency_ms": float(latency_ms),
            "model_sha256": model_sha,
        }
        f.write(json.dumps(rec) + "\n")
    f.close()
    print("wrote", str(out))

if __name__ == "__main__":
    main()
