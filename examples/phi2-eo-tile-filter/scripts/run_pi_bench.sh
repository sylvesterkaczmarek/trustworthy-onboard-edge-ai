#!/usr/bin/env bash
set -e
source .venv/bin/activate
python -m src.bench_onnxruntime --onnx models/tinycnn_int8.onnx --bands 3 --size 64 --iters 500
