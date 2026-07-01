from pathlib import Path
import csv
import json
import time

LOG_PATH = Path(__file__).resolve().parent / "nn_log.csv"

def log_nn_event(stage, status, execution_time=None, parameters=None, message=""):
    # Log folder
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # New file check
    new_file = not LOG_PATH.exists()

    # Log row
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "status": status,
        "execution_time_sec": "" if execution_time is None else round(float(execution_time), 4),
        "parameters": json.dumps(parameters or {}, ensure_ascii=False),
        "message": message or "",
    }

    with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        # To be readable by Excel
        if new_file:
            f.write("sep=,\n")

        writer = csv.DictWriter(f, fieldnames=row.keys())

        if new_file:
            writer.writeheader()

        writer.writerow(row)