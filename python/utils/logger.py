import os
import csv
import threading
import time

_LOG_PATH = None
_LOCK = threading.Lock()

def init_log(run_kind: str, clear: bool = True, log_dir: str = "."):
    """
    Prepare a CSV log. clear=True overwrites any previous file.
    run_kind: "single" or "multi" (free text).
    """
    global _LOG_PATH
    os.makedirs(log_dir, exist_ok=True)
    _LOG_PATH = os.path.join(log_dir, "anc_run_log.csv")
    if clear or (not os.path.exists(_LOG_PATH)):
        with open(_LOG_PATH, "w", newline="") as f:
            f.write("sep=,\n")
            w = csv.writer(f)
            w.writerow([
                "ts", "run_kind", "stage", "status",
                "algorithm", "source", "noise_label",
                "L", "mu",
                "conv_ms", "sse_db", "exec_time_s",
                "power_anc_off", "power_anc_on",
                "save_path", "message"
            ])
    # header written; subsequent rows will append

def log_case(stage, status, algorithm, source, noise_label,
             L, mu, conv_ms, sse_db, exec_time, in_power, out_power,
             save_path, message, run_kind=None):
    """
    Append one line. All numeric fields may be None.
    """
    global _LOG_PATH
    if _LOG_PATH is None:
        # default to cwd file if not initialized
        _LOG_PATH = os.path.join(".", "anc_run_log.csv")

    row = [
        time.strftime("%Y-%m-%d %H:%M:%S"),
        (run_kind or ""), stage, status,
        algorithm, source, noise_label,
        ("" if L is None else int(L)),
        ("" if mu is None else f"{float(mu):.4f}"),
        ("" if conv_ms is None else f"{float(conv_ms):.2f}"),
        ("" if sse_db  is None else f"{float(sse_db):.2f}"),
        ("" if exec_time is None else f"{float(exec_time):.2f}"),
        ("" if in_power is None else f"{float(in_power):.3f}"),
        ("" if out_power is None else f"{float(out_power):.3f}"),
        (save_path or ""), (message or "")
    ]
    try:
        with _LOCK:
            with open(_LOG_PATH, "a", newline="") as f:
                csv.writer(f).writerow(row)
    except Exception:
        pass