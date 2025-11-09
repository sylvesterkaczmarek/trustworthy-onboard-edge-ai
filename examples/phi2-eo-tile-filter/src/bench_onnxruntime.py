import argparse, time, psutil, numpy as np, onnxruntime as ort

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--bands", type=int, default=3)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--iters", type=int, default=200)
    a = p.parse_args()

    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    x = np.random.rand(1, a.bands, a.size, a.size).astype(np.float32)

    for _ in range(10): sess.run(None, {"input": x})
    lat = []; mem0 = psutil.Process().memory_info().rss; t0 = time.time()
    for _ in range(a.iters):
        t1 = time.time()
        sess.run(None, {"input": x})
        lat.append((time.time()-t1)*1000)
    t = time.time() - t0; mem1 = psutil.Process().memory_info().rss

    print(f"avg_ms {np.mean(lat):.3f} p50 {np.percentile(lat,50):.3f} p90 {np.percentile(lat,90):.3f} p99 {np.percentile(lat,99):.3f}")
    print(f"throughput_fps {a.iters/t:.1f} mem_delta_mb {(mem1-mem0)/1e6:.2f}")

if __name__ == "__main__":
    main()
