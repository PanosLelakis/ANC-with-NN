from pathlib import Path
import csv
import json
import time
import threading
from utils.logger import get_logs_dir

_HISTORY_LOCK = threading.Lock()

def get_nn_log_path():
    # Build Neural Network log path
    return Path(get_logs_dir()) / "nn_log.csv"

def get_training_history_path():
    # Build training history path
    return Path(get_logs_dir()) / "training_history.csv"

def log_nn_event(stage, status, execution_time=None, parameters=None, message=""):
    # Read log path
    log_path = get_nn_log_path()

    # Create log folder
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # New file check
    new_file = not log_path.exists()

    # Log row
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "status": status,
        "execution_time_sec": (
            ""
            if execution_time is None
            else f"{float(execution_time):.2f}"
        ),
        "parameters": json.dumps(parameters or {}, ensure_ascii=False),
        "message": message or "",
    }

    with open(log_path, "a", newline="", encoding="utf-8-sig") as file:
        # To be readable by Excel
        if new_file:
            file.write("sep=,\n")

        writer = csv.DictWriter(file, fieldnames=row.keys())

        if new_file:
            writer.writeheader()

        writer.writerow(row)

def initialize_training_history():
    # Read history path
    history_path = get_training_history_path()

    # Create logs folder
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # Create new training history
    with _HISTORY_LOCK:
        with open(history_path, "w", newline="", encoding="utf-8-sig") as file:
            file.write("sep=,\n")

            writer = csv.writer(file)
            writer.writerow([
                "epoch",
                "training_loss",
                "training_anc_loss",
                "training_silence_loss",
                "training_weighted_silence_loss",
                "training_frequency_loss",
                "training_weighted_frequency_loss",
                "validation_loss",
                "validation_anc_off_loss",
                "validation_nmse_db"
            ])

    # Return history path
    return str(history_path)

def append_training_history(
    epoch,
    training_loss,
    training_anc_loss,
    training_silence_loss,
    training_weighted_silence_loss,
    training_frequency_loss,
    training_weighted_frequency_loss,
    validation_loss,
    validation_anc_off_loss,
    validation_nmse_db
):
    # Read history path
    history_path = get_training_history_path()

    # Create history file when missing
    if not history_path.exists():
        initialize_training_history()

    # Build history row
    row = [
        int(epoch),
        f"{float(training_loss):.8f}",
        f"{float(training_anc_loss):.8f}",
        f"{float(training_silence_loss):.8f}",
        f"{float(training_weighted_silence_loss):.8f}",
        f"{float(training_frequency_loss):.8f}",
        f"{float(training_weighted_frequency_loss):.8f}",
        f"{float(validation_loss):.8f}",
        f"{float(validation_anc_off_loss):.8f}",
        f"{float(validation_nmse_db):.2f}"
    ]

    # Append history row
    with _HISTORY_LOCK:
        with open(
            history_path,
            "a",
            newline="",
            encoding="utf-8-sig"
        ) as file:
            csv.writer(
                file
            ).writerow(
                row
            )

    # Return history path
    return str(
        history_path
    )