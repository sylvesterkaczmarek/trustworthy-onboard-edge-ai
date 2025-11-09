Goals: watchdog process, confidence threshold + fallback, telemetry log, rollback.
Run: inference → check max softmax vs threshold → if low, flag for downlink anyway → log timing, version, checksum.
