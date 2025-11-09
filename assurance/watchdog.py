import time, subprocess, sys, os

CMD = os.environ.get("INFER_CMD", "python -m src.infer_onnx --onnx models/tinycnn_int8.onnx --data ./tiles/val")
RESTARTS = 3
SLEEP = 2

for i in range(RESTARTS):
    print(f"watchdog start {i+1}")
    rc = subprocess.call(CMD, shell=True)
    if rc == 0:
        sys.exit(0)
    time.sleep(SLEEP)
print("watchdog: max restarts reached")
sys.exit(1)
