import os
import csv
import threading
import time

_LOG_PATH = None
_RUN_KIND = ""
_LOCK = threading.Lock()

def get_logs_dir():
    # Build common logs folder
    log_dir = os.path.join(os.getcwd(), "logs")

    # Create logs folder
    os.makedirs(log_dir, exist_ok=True)

    # Return logs folder
    return log_dir

def init_log(run_kind: str, clear: bool = True, log_dir=None):
    """
    Prepare a CSV log. clear=True overwrites any previous file.
    run_kind: "single" or "multi" (free text).
    """
    global _LOG_PATH, _RUN_KIND

    # Store run type
    _RUN_KIND = str(run_kind or "")

    # Use common logs folder
    if log_dir is None:
        log_dir = get_logs_dir()

    # Create log folder
    os.makedirs(log_dir, exist_ok=True)

    # Build ANC log path
    _LOG_PATH = os.path.join(log_dir, "anc_run_log.csv")

    # Create or clear log
    if clear or (not os.path.exists(_LOG_PATH)):
        with open(
            _LOG_PATH,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:
            file.write("sep=,\n")

            writer = csv.writer(file)
            writer.writerow([
                "ts",
                "run_kind",
                "stage",
                "status",
                "divergence",
                "algorithm",
                "source",
                "noise_label",
                "L",
                "mu",
                "conv_ms",
                "sse_db",
                "exec_time_s",
                "power_anc_off",
                "power_anc_on",
                "save_path",
                "message"
            ])

    # Return created log path
    return _LOG_PATH

def log_case(stage, status, algorithm, source, noise_label,
             L, mu, conv_ms, sse_db, exec_time, in_power, out_power,
             save_path, message, run_kind=None, divergence=False):
    """
    Append one line. All numeric fields may be None.
    """
    global _LOG_PATH
    # Use common log path when not initialized
    if _LOG_PATH is None:
        _LOG_PATH = os.path.join(
            get_logs_dir(),
            "anc_run_log.csv"
        )

        # Create header when file is missing
        if not os.path.exists(_LOG_PATH):
            init_log(
                run_kind=run_kind or "",
                clear=False
            )

    row = [
        time.strftime("%Y-%m-%d %H:%M:%S"),
        (run_kind or _RUN_KIND or ""), stage, status, str(bool(divergence)).lower(),
        algorithm, source, noise_label,
        ("" if L is None else int(L)),
        ("" if mu is None else f"{float(mu):.6g}"),
        ("" if conv_ms is None else f"{float(conv_ms):.2f}"),
        ("" if sse_db  is None else f"{float(sse_db):.2f}"),
        ("" if exec_time is None else f"{float(exec_time):.2f}"),
        ("" if in_power is None else f"{float(in_power):.3f}"),
        ("" if out_power is None else f"{float(out_power):.3f}"),
        (save_path or ""), (message or "")
    ]
    try:
        with _LOCK:
            with open(_LOG_PATH, "a", newline="", encoding="utf-8-sig") as file:
                csv.writer(file).writerow(row)
    except Exception:
        pass