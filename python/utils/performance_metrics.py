import numpy as np

DEFAULT_BANDS = [
    (0, 500),
    (500, 1000),
    (1000, 3000),
    (3000, 5000),
    (5000, 10000),
    (10000, 20000),
]

def _finite_view(x):
    x = np.asarray(x)
    if x.ndim == 0:
        return np.array([0.0], dtype=float) if not np.isfinite(x) else x.astype(float)
    mask = np.isfinite(x)
    if not np.any(mask):
        return np.zeros(1, dtype=float)
    return x[mask].astype(float)

def compute_convergence_time(error, fs, sse_db, improvement_ratio=0.80,
                             start_ms=5.0, stable_ms=0, min_start_ms=20.0):
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
                return 1000.0 * (first_idx / fs)
        else:
            run = 0

    return None

def compute_steady_state_error(error_signal, percentage=0.2):
    err = _finite_view(error_signal)
    last_samples = max(1, int(percentage * len(err)))
    seg = err[-last_samples:]
    return float(np.mean(seg))

def compute_avg_pnc_dbr(passive_signal, noisy_signal, percentage=0.2):
    # Convert inputs to float arrays.
    passive_signal = np.asarray(passive_signal, dtype=float)
    noisy_signal = np.asarray(noisy_signal, dtype=float)

    # Use the last part of the signal, as with SSE.
    N = min(len(passive_signal), len(noisy_signal))
    start = int((1.0 - float(percentage)) * N)

    passive_tail = passive_signal[start:N]
    noisy_tail = noisy_signal[start:N]

    # Compute RMS values.
    passive_rms = np.sqrt(np.mean(passive_tail ** 2) + 1e-12)
    noisy_rms = np.sqrt(np.mean(noisy_tail ** 2) + 1e-12)

    # Return passive noise cancelling level relative to noisy signal.
    return float(20.0 * np.log10((passive_rms + 1e-12) / (noisy_rms + 1e-12)))

def compute_band_attenuation(d_signal, e_signal, fs, bands=None, percentage=0.2):
    # Use default frequency bands if none are given.
    if bands is None:
        bands = DEFAULT_BANDS

    # Convert inputs to float arrays.
    d_signal = np.asarray(d_signal, dtype=float)
    e_signal = np.asarray(e_signal, dtype=float)

    # Use the last part of the signal, as with SSE.
    N = min(len(d_signal), len(e_signal))
    start = int((1.0 - float(percentage)) * N)

    d_tail = d_signal[start:N]
    e_tail = e_signal[start:N]

    # Apply Hann window to reduce spectral leakage.
    win = np.hanning(len(d_tail))

    d_tail = d_tail * win
    e_tail = e_tail * win

    # Compute spectra.
    D = np.fft.rfft(d_tail)
    E = np.fft.rfft(e_tail)

    freqs = np.fft.rfftfreq(len(d_tail), 1.0 / float(fs))

    # Compute proportional power spectra.
    denom = np.sum(win ** 2) + 1e-12
    P_d = (np.abs(D) ** 2) / denom
    P_e = (np.abs(E) ** 2) / denom

    # Compute one attenuation value per band.
    out = {}

    for f1, f2 in bands:
        idx = np.where((freqs >= f1) & (freqs < f2))[0]

        Pd = np.sum(P_d[idx]) + 1e-12
        Pe = np.sum(P_e[idx]) + 1e-12

        label = f"{int(f1)}-{int(f2)}"

        # Negative value means improvement, consistent with the current band attenuation plot.
        out[label] = float(10.0 * np.log10(Pe / Pd))

    return out