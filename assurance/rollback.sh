#!/usr/bin/env bash
set -e
prev="models/tinycnn_fp32.onnx.bak"
curr="models/tinycnn_fp32.onnx"
if [ -f "$prev" ]; then
  mv "$curr" "${curr}.bad.$(date +%s)" || true
  mv "$prev" "$curr"
  echo "Rolled back to previous FP32."
else
  echo "No backup found."
fi
