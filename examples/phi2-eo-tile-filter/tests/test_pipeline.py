from pathlib import Path
import subprocess

def run(cmd): subprocess.check_call(cmd, shell=True)

def test_end_to_end(tmp_path):
    tiles = tmp_path / "tiles"
    run(f"python -m data.synth --out {tiles} --n 64 --bands 3 --size 32")
    run(f"python -m src.train --data {tiles} --epochs 1 --bands 3 --size 32")
    Path("models").mkdir(exist_ok=True)
    run("python -m src.export_onnx --weights runs/tinycnn.pt --out models/tinycnn_fp32.onnx --bands 3 --size 32")
    run(f"python -m src.quantize_ptq --onnx models/tinycnn_fp32.onnx --calib {tiles}/val --out models/tinycnn_int8.onnx --size 32")
    run(f"python -m src.infer_onnx --onnx models/tinycnn_int8.onnx --data {tiles}/val --bands 3 --size 32")
