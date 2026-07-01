import time

def format_eta(seconds):
    # Format ETA
    if seconds is None or seconds < 0:
        return "ETA --:--"

    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"ETA {h:02d}:{m:02d}:{s:02d}"

    return f"ETA {m:02d}:{s:02d}"

def estimate_eta(start_time, done, total):
    # Estimate remaining time
    done = int(done)
    total = int(total)

    if done <= 0 or total <= 0:
        return "ETA --:--"

    elapsed = time.time() - float(start_time)
    remaining = (elapsed / done) * max(0, total - done)

    return format_eta(remaining)

def format_execution_time(execution_time):
    # Format execution time
    if execution_time is None:
        return "Execution time: --:--"

    seconds = int(round(float(execution_time) * 60.0))

    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"Execution time: {h:02d}:{m:02d}:{s:02d}"

    return f"Execution time: {m:02d}:{s:02d}"