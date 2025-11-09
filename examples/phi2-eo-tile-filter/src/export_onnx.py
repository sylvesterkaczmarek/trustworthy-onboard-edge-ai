import argparse
from pathlib import Path
import torch
from .models.tiny_cnn import TinyCNN

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bands", type=int, default=3)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--base", type=int, default=16)
    a = p.parse_args()

    model = TinyCNN(in_ch=a.bands, base=a.base)
    model.load_state_dict(torch.load(a.weights, map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, a.bands, a.size, a.size)
    Path(Path(a.out).parent).mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (dummy,), a.out,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}},
        opset_version=18, dynamo=False, do_constant_folding=True,
    )
    print("Exported", a.out)

if __name__ == "__main__":
    main()
