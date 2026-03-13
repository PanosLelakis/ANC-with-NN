import numpy as np

def _finite_view(x):
    x = np.asarray(x)
    if x.ndim == 0:
        return np.array([0.0], dtype=float) if not np.isfinite(x) else x.astype(float)
    mask = np.isfinite(x)
    if not np.any(mask):
        return np.zeros(1, dtype=float)
    return x[mask].astype(float)

def compute_convergence_time(error, fs, sse_db, improvement_ratio=0.80,
                             start_ms=5.0, stable_ms=0, min_start_ms=20.0, tail_guard_ms=10.0):
    """
    Convergence time based on % improvement from start level to steady-state level.

    - error: error curve in dB (smoothed)
    - sse_db: steady-state error in dB (already computed correctly)
    - improvement_ratio: e.g. 0.90 means 90% of the drop achieved
    - start_ms: how many ms from the beginning define "start level"
    - stable_ms: must stay below threshold for this long
    """
    err = _finite_view(error)
    if err.size == 0:
        return None

    fs = float(fs)

    # start level = mean of first start_ms
    start_len = int(round((start_ms / 1000.0) * fs))
    start_len = max(1, min(start_len, err.size))
    start_db = float(np.mean(err[:start_len]))

    sse_db = float(sse_db)

    # if there is no improvement (or it gets worse), convergence is undefined
    improvement = np.abs(start_db - sse_db)
    if not np.isfinite(improvement) or improvement <= 0:
        return None

    # 90% of improvement achieved => threshold in dB
    thr_db = start_db - float(improvement_ratio) * improvement

    # stability requirement (must stay below threshold)
    stable_len = max(1, int(round((stable_ms / 1000.0) * fs)))

    # convergence in the first min_start_ms would be unrealistic
    start_idx = max(0, int(round((min_start_ms / 1000.0) * fs)))

    run = 0
    for i in range(start_idx, err.size):
        if err[i] <= thr_db:
            run += 1
            if run >= stable_len:
                first_idx = i - stable_len + 1
                #guard = int(round((tail_guard_ms / 1000.0) * fs))
                #idx = min(first_idx + guard, err.size - 1)
                return 1000.0 * (first_idx / fs)
        else:
            run = 0

    return None

def compute_steady_state_error(error_signal, percentage=0.2):
    err = _finite_view(error_signal)
    last_samples = max(1, int(percentage * len(err)))
    seg = err[-last_samples:]
    return float(np.mean(seg))