import numpy as np

def _finite_view(x):
    x = np.asarray(x)
    if x.ndim == 0:
        return np.array([0.0], dtype=float) if not np.isfinite(x) else x.astype(float)
    mask = np.isfinite(x)
    if not np.any(mask):
        return np.zeros(1, dtype=float)
    return x[mask].astype(float)

def compute_convergence_time(error_signal, fs, sse,
                             threshold_factor=1.05, window_ms=20.0,
                             min_stable_duration=0.02, acceptance_ratio=0.90,
                             min_start_ms=100.0):
    """
    First time where a sliding-RMS (window_ms) stays below threshold_factor * sse
    for at least min_stable_duration with >= acceptance_ratio of samples,
    AFTER min_start_ms has elapsed, and preceded by a violation segment.
    """
    err = _finite_view(error_signal)
    N = len(err)
    if N == 0:
        return None

    w = max(1, int(round((window_ms/1000.0) * float(fs))))
    x2 = err * err
    csum = np.concatenate(([0.0], np.cumsum(x2)))
    rms = np.sqrt((csum[w:] - csum[:-w]) / float(w))
    centers = np.arange(w//2, w//2 + len(rms))  # center indices of windows

    # skip early part
    start_idx = int(max(0, np.round((min_start_ms/1000.0) * float(fs))))
    valid = centers >= start_idx
    if not np.any(valid):
        return None
    rms = rms[valid]; centers = centers[valid]

    thr = max(1e-12, threshold_factor * float(sse))
    below = rms <= thr

    need_len = max(1, int(round(min_stable_duration * float(fs))))
    kernel = np.ones(need_len, dtype=int)
    roll = np.convolve(below.astype(int), kernel, mode="valid")
    need = int(np.ceil(acceptance_ratio * need_len))
    idxs = np.where(roll >= need)[0]
    if idxs.size == 0:
        return None

    # require that immediately before the detected block there was a violation
    k0 = idxs[0]
    guard_start = max(0, k0 - need)
    if np.all(below[max(0, guard_start):k0]):  # no violation before -> keep searching
        idxs = idxs[1:]
        if idxs.size == 0:
            return None
        k0 = idxs[0]

    k_center = k0 + need_len//2
    k_center = min(k_center, len(centers)-1)
    return 1000.0 * (centers[k_center] / float(fs))  # msec

def compute_steady_state_error(error_signal, percentage=0.2):
    err = _finite_view(error_signal)
    last_samples = max(1, int(percentage * len(err)))
    seg = err[-last_samples:]
    # robust RMSE with clipping to avoid outliers dominating
    seg = np.clip(seg, -10.0, 10.0)
    return float(np.sqrt(np.mean(seg * seg)))