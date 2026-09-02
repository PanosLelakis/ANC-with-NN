def clamp_odd_window(desired, n):
    """
    Returns an odd window length <= n and >= 3 (or 1 if n<3).
    """
    w = int(desired)
    if n < 3:
        return 1
    if w < 3:
        w = 3
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if (n % 2 == 1) else (n - 1)
    if w < 3:
        return 1
    return w

def sg_smooth(y, window, order=3):
    """
    Savitzky-Golay smoothing with safe window clamping.
    """
    #y = np.asarray(y, dtype=float)
    #w = clamp_odd_window(window, len(y))
    #if w < 3:
        #return y
    w = window
    return savitzky_golay(y, w, order=order)
    
def savitzky_golay(y, window_size, order):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    - window_size: must be odd
    - order: polynomial order
    - rate: sampling rate-like factor (kept for compatibility with your old function)
    """
    import numpy as np
    from scipy.signal import savgol_filter

    y = np.asarray(y, dtype=float)

    window_size = abs(int(window_size))
    order = abs(int(order))

    if window_size < 3:
        return y

    # must be odd
    if window_size % 2 == 0:
        window_size += 1

    # must satisfy window_size >= order + 2
    if window_size < order + 2:
        window_size = order + 2
        if window_size % 2 == 0:
            window_size += 1

    # clamp window to signal length (must be <= len(y) and odd)
    if len(y) < window_size:
        window_size = len(y) if (len(y) % 2 == 1) else max(1, len(y) - 1)
        if window_size < 3:
            return y

    return savgol_filter(y, window_length=window_size, polyorder=order)
                         #mode="mirror")

def whittaker_eilers_smooth(y, lmbd=1e8, order=2, weights=None, x_input=None):
    """
    Whittaker-Eilers penalized least-squares smoothing.

    Parameters
    ----------
    y : array-like
        Input data (1D).
    lmbd : float
        Smoothing strength (bigger => smoother). Typical range: 1e2 ... 1e7.
    d : int
        Difference order (usually 2).
    w : array-like or None
        Optional weights (same length as y). If None, all ones.

    Returns
    -------
    z : np.ndarray
        Smoothed signal.
    """
    from whittaker_eilers import WhittakerSmoother
    import numpy as np

    ws = None if weights is None else np.asarray(weights, dtype=float).tolist()
    xi = None if x_input is None else np.asarray(x_input, dtype=float).tolist()

    smoother = WhittakerSmoother(
        lmbda=float(lmbd),
        order=int(order),
        data_length=int(len(y)),
        x_input=xi,
        weights=ws
    )
    
    return np.asarray(smoother.smooth(y.tolist()), dtype=float)

def smooth_fractional_octave_db(freqs, values_db, num_fractions=12,
                                db_kind="amplitude", window="boxcar",
                                sampling_rate=None, n_samples=None):
    """
    Fractional-octave smoothing for frequency-domain dB curves using pyfar.

    freqs : array-like
        Frequency axis in Hz.
    values_db : array-like
        Spectrum values in dB.
    num_fractions : int
        1  -> 1 octave smoothing
        3  -> 1/3 octave smoothing
        6  -> 1/6 octave smoothing
        12 -> 1/12 octave smoothing
    db_kind : {"amplitude", "power"}
        "amplitude": values_db came from 20*log10(...)
        "power":     values_db came from 10*log10(...)
    sampling_rate : float or None
        Needed to construct a pyfar.Signal. If None, estimated from freqs.
    n_samples : int or None
        FFT length / original segment length. For Welch spectra, pass nperseg.
    """
    import numpy as np
    import pyfar as pf

    freqs = np.asarray(freqs, dtype=float)
    values_db = np.asarray(values_db, dtype=float)

    out = values_db.copy()

    mask = np.isfinite(freqs) & np.isfinite(values_db) & (freqs > 0.0)
    if np.count_nonzero(mask) < 5:
        return out

    f = freqs[mask]
    y_db = values_db[mask]

    # Sort frequencies
    order = np.argsort(f)
    f_sorted = f[order]
    y_sorted = y_db[order]

    # Remove duplicate frequencies
    f_unique, unique_idx = np.unique(f_sorted, return_index=True)
    y_unique = y_sorted[unique_idx]

    # Infer sampling rate and FFT length if not provided
    if sampling_rate is None:
        sampling_rate = 2.0 * float(np.max(f_unique))

    if n_samples is None:
        df = float(np.median(np.diff(f_unique)))
        n_samples = int(round(float(sampling_rate) / df))

    sampling_rate = float(sampling_rate)
    n_samples = int(n_samples)

    # Create complete rFFT frequency grid expected by pyfar.Signal
    full_freqs = np.fft.rfftfreq(n_samples, d=1.0 / sampling_rate)

    # Interpolate your dB curve onto the complete FFT grid, including DC
    y_full_db = np.interp(
        full_freqs,
        f_unique,
        y_unique,
        left=y_unique[0],
        right=y_unique[-1]
    )

    # Convert dB to linear magnitude-like data
    if db_kind == "power":
        y_full_lin = 10.0 ** (y_full_db / 10.0)
        log_factor = 10.0
    elif db_kind == "amplitude":
        y_full_lin = 10.0 ** (y_full_db / 20.0)
        log_factor = 20.0
    else:
        raise ValueError("db_kind must be 'amplitude' or 'power'")

    # pyfar.smooth_fractional_octave requires pyfar.Signal, not FrequencyData
    sig = pf.Signal(
        y_full_lin.astype(np.complex128),
        sampling_rate=sampling_rate,
        n_samples=n_samples,
        domain="freq",
        fft_norm="none"
    )

    sig_smooth, _ = pf.dsp.smooth_fractional_octave(
        sig,
        num_fractions=num_fractions,
        mode="magnitude_zerophase",
        window=window
    )

    y_smooth_lin = np.maximum(np.abs(np.asarray(sig_smooth.freq).squeeze()), 1e-24)
    y_smooth_db_full = log_factor * np.log10(y_smooth_lin)

    # Interpolate back to the original frequency points
    y_back = np.interp(f, full_freqs, y_smooth_db_full)

    # Undo sorting and write into output
    y_unsorted = np.empty_like(y_back)
    y_unsorted[order] = y_back

    out[mask] = y_unsorted
    return out