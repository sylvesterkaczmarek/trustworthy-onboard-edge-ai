import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort
from sklearn.metrics import accuracy_score, confusion_matrix

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--bands", type=int, default=3)
    p.add_argument("--size", type=int, default=64)
    a = p.parse_args()

    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    ys, xs = [], []
    files = []
    for cls, name in enumerate(["background", "event"]):
        files += [(f, cls) for f in sorted((Path(a.data) / name).glob("*.png"))]
    for f, cls in files:
        img = Image.open(f).convert("RGB").resize((a.size, a.size))
        x = np.transpose(np.asarray(img, dtype=np.float32) / 255.0, (2, 0, 1))[None, ...]
        prob = softmax(sess.run(None, {"input": x})[0])
        xs.append(prob.argmax(1)[0]); ys.append(cls)
    acc = accuracy_score(ys, xs)
    cm = confusion_matrix(ys, xs)
    print("accuracy", float(acc))
    print("confusion\n", cm)

if __name__ == "__main__":
    main()
