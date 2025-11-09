# Assurance

Guardrails and operational hooks for running models in constrained environments.

## What this covers
- Watchdog that restarts a failing command a few times.
- Calibrated confidence gate with fallback to downlink.
- JSONL telemetry for latency, hashes and decisions.
- Rollback script for the last good FP32 model.
- Simple summarizer that turns logs into a short report.

## Folder contents
- `watchdog.py` process restarter
- `rollback.sh` swap back to previous FP32
- `telemetry_log.py` emit per tile JSON lines
- `summarize.py` build `reports/summary.md` and `reports/metrics.json`
- `telemetry.md` log description
- `safety.md` notes and assumptions

## Quick run
From `examples/phi2-eo-tile-filter`:
```bash
# 1) run bandwidth filter and write decision log
python -m src.bandwidth_filter --onnx models/tinycnn_int8.onnx   --data ./tiles/val --calibration calibration.json   --downlink_out downlink --log logs/downlink.jsonl

# 2) emit detailed telemetry per tile
python ../../assurance/telemetry_log.py   --onnx models/tinycnn_int8.onnx   --data ./tiles/val   --out logs/val.jsonl   --threshold "$(python -c 'import json;print(json.load(open("calibration.json"))["threshold"])')"

# 3) build a short report
python ../../assurance/summarize.py   --val_log logs/val.jsonl   --downlink_log logs/downlink.jsonl   --val_dir tiles/val   --calib calibration.json   --out_dir reports
```

Watchdog and rollback, from the repo root:
```bash
INFER_CMD="python -m src.infer_onnx --onnx examples/phi2-eo-tile-filter/models/tinycnn_int8.onnx --data examples/phi2-eo-tile-filter/tiles/val" python assurance/watchdog.py

bash assurance/rollback.sh
```

## Policy knobs
- Threshold comes from `calibration.json` written by `src.calibrate_threshold`.
- You can override with `--threshold` when calling `src.bandwidth_filter`.
- For different confidence behaviour use `--temperature` during calibration and filtering.

## Telemetry schema (JSON line per tile)
What `assurance/telemetry_log.py` writes:
```json
{
  "timestamp": 1731174800.12,
  "file": "tiles/val/event/00012.png",
  "true_class": 1,
  "pred_class": 1,
  "max_prob": 0.982,
  "prob_event": 0.982,
  "threshold": 0.678,
  "ok_flag": true,
  "latency_ms": 0.46,
  "model_sha256": "2d4c..."
}
```
One JSON object per line, suitable for grep and jq.

## Failure and fallback
- Low confidence or exception: mark for downlink and log the reason.
- Latency over budget: still emit a record so ground review can decide.
- Crash: watchdog retries a few times.
- Bad model: run `rollback.sh` then re-run export and quantize.

## Checklist before you claim results
- [ ] Record SHA256 for `models/tinycnn_fp32.onnx` and `models/tinycnn_int8.onnx`.
- [ ] Save accuracy, confusion matrix and latency stats that match the logs.
- [ ] Keep `calibration.json` with the run.
- [ ] Verify watchdog and rollback once.
- [ ] Commit `logs/` and `reports/` or attach them to the concept note.

## Extending
- Add a memory guard that stops inference if RSS exceeds a limit.
- Emit CSV next to JSON if ground tools expect CSV.
- Swap to a stronger model and recalibrate to keep the recall target.

## License
MIT. See [LICENSE](LICENSE).
© Sylvester Kaczmarek · https://www.sylvesterkaczmarek.com