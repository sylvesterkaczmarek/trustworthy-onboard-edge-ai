import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantType, QuantFormat

class TileReader(CalibrationDataReader):
    def __init__(self, folder, size=64, bands=3, batch=8):
        self.files = sorted(list(Path(folder).glob("**/*.png")))
        self.size, self.bands, self.batch = size, bands, batch
        self._it = None

    def get_next(self):
        if self._it is None:
            def gen():
                for i in range(0, len(self.files), self.batch):
                    xs = []
                    for p in self.files[i:i+self.batch]:
                        img = Image.open(p).convert("RGB").resize((self.size, self.size))
                        x = np.asarray(img, dtype=np.float32) / 255.0
                        xs.append(np.transpose(x, (2, 0, 1))[None, ...])
                    yield {"input": np.concatenate(xs, axis=0)}
            self._it = gen()
        return next(self._it, None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--bands", type=int, default=3)
    a = ap.parse_args()

    Path(Path(a.out).parent).mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=a.onnx,
        model_output=a.out,
        calibration_data_reader=TileReader(a.calib, a.size, a.bands),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        extra_options={"ActivationSymmetric": False},
    )
    print("Wrote", a.out)

if __name__ == "__main__":
    main()
