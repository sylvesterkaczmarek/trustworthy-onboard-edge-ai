import argparse
from pathlib import Path
import numpy as np
from PIL import Image

rng = np.random.default_rng(0)

def make_tile(size: int, bands: int, cls: int) -> np.ndarray:
    img = rng.integers(0, 255, size=(size, size, bands), dtype=np.uint8)
    if cls == 1:
        s = size // 3
        x0 = rng.integers(0, size - s)
        y0 = rng.integers(0, size - s)
        img[y0:y0+s, x0:x0+s, :] = 255
    return img

def write_split(root: Path, split: str, n: int, bands: int, size: int):
    for i in range(n):
        cls = int(i % 2 == 0)     # 50/50 background/event
        arr = make_tile(size, bands, cls)
        d = root / split / ("event" if cls else "background")
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr[:, :, :3]).save(d / f"{i:05d}.png")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--bands", type=int, default=3)
    p.add_argument("--size", type=int, default=64)
    a = p.parse_args()
    out = Path(a.out)
    write_split(out, "train", int(a.n * 0.8), a.bands, a.size)
    write_split(out, "val", a.n - int(a.n * 0.8), a.bands, a.size)
    print("Wrote", out)

if __name__ == "__main__":
    main()
