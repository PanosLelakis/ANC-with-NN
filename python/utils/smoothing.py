import numpy as np

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
    
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    - window_size: must be odd
    - order: polynomial order
    - deriv: derivative order (0 = smoothing)
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

    # delta corresponds to sample spacing; keep compatibility with "rate"
    delta = 1.0 / float(rate) if rate else 1.0

    return savgol_filter(y, window_length=window_size, polyorder=order,
                         deriv=deriv, delta=delta, mode="interp")

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