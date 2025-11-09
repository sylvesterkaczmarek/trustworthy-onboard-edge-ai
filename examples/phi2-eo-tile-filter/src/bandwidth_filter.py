import argparse, shutil, json, time, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--calibration", type=str, default=None, help="JSON from calibrate_threshold.py")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--downlink_out", type=str, default="downlink")
    p.add_argument("--log", type=str, default=None, help="optional JSONL log path")
    a = p.parse_args()

    # load calibration if provided
    if a.calibration:
        with open(a.calibration) as f:
            cfg = json.load(f)
        a.threshold = float(cfg.get("threshold", a.threshold))
        if "temperature" in cfg:
            a.temperature = float(cfg["temperature"])

    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    files = list(Path(a.data).glob("**/*.png"))
    total = sum(p.stat().st_size for p in files)
    model_hash = file_sha256(Path(a.onnx))

    # ensure downlink folder
    out = Path(a.downlink_out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    sent = 0
    kept = 0

    # >>> changed block: auto-create log folder
    log_f = None
    if a.log:
        log_path = Path(a.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = log_path.open("w")
    # <<<

    t_start = time.time()

    for pth in files:
        img = Image.open(pth).convert("RGB").resize((a.size, a.size))
        x = np.transpose(np.asarray(img, dtype=np.float32) / 255.0, (2, 0, 1))[None, ...]
        t1 = time.time()
        logits = sess.run(None, {"input": x})[0]
        if a.temperature != 1.0:
            logits = logits / a.temperature
        prob = softmax(logits)[0]
        elapsed_ms = (time.time() - t1) * 1000

        if prob[1] >= a.threshold:
            kept += 1
            dest = out / pth.name
            dest.write_bytes(pth.read_bytes())
            sent += pth.stat().st_size
            ok = True
        else:
            ok = False

        if log_f:
            rec = {
                "file": str(pth),
                "model_sha256": model_hash,
                "size": pth.stat().st_size,
                "prob_event": float(prob[1]),
                "pred_class": int(prob.argmax()),
                "ok": bool(ok),
                "latency_ms": float(elapsed_ms),
                "threshold": float(a.threshold),
                "temperature": float(a.temperature)
            }
            log_f.write(json.dumps(rec) + "\n")

    if log_f:
        log_f.close()
    saved = 1 - sent / total if total > 0 else 0
    print(f"tiles {len(files)} kept {kept} saved_bandwidth {saved*100:.1f}% elapsed_s {time.time()-t_start:.2f}")

if __name__ == "__main__":
    main()
