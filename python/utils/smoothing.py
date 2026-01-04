import numpy as np
from scipy.signal import medfilt

def smooth_signal(signal, kernel_size):
    if kernel_size % 2 == 0:
        kernel_size += 1
    return medfilt(signal, kernel_size)

def smooth_fractional_octave(freqs_hz, values_db, fraction=6, min_bins=5, max_bins=199, scale=2.0):
    """
    Median smoothing with window size ~ fractional-octave bandwidth.
    Window (in bins) grows with frequency (constant-Q idea).
    'fraction' = 6  -> 1/6-octave base window.
    'scale'    > 1  -> multiplies the window size across ALL frequencies.

    Larger 'scale' -> more smoothing at all freqs; window remains frequency-dependent.
    """
    freqs = np.asarray(freqs_hz, dtype=float)
    vals  = np.asarray(values_db, dtype=float)
    out   = np.empty_like(vals)

    # helper: clamp & ensure odd length
    def _odd(n):
        n = int(max(min_bins, min(max_bins, n)))
        return n if (n % 2) == 1 else n + 1

    # log-frequency spacing; bins-per-octave ~ 1 / Δlog2(f)
    logf = np.log2(np.maximum(freqs, 1e-3))
    dlog = np.gradient(logf)

    # base width (in bins) for 1/fraction octave, then scale up globally
    width_bins = (1.0 / float(fraction)) / np.maximum(dlog, 1e-6)
    width_bins = np.round(width_bins * float(scale)).astype(int)

    # median with varying window
    for i in range(len(vals)):
        w  = _odd(width_bins[i])
        lo = max(0, i - w // 2)
        hi = min(len(vals), i + w // 2 + 1)
        out[i] = np.median(vals[lo:hi])
    return out

def smooth_time_median(values_db, fs, window_ms=100):
    """
    Median smoothing in time domain using a window specified in milliseconds.
    Ensures odd window length; clamps to at least 3 samples.
    """
    vals = np.asarray(values_db, dtype=float)
    # samples for requested ms
    w = int(round((window_ms / 1000.0) * float(fs)))
    if w < 3: w = 3
    if (w % 2) == 0: w += 1  # odd
    return medfilt(vals, kernel_size=w)