import numpy as np
from scipy.signal import medfilt

#def smooth_signal(signal, kernel_size):
    #if kernel_size % 2 == 0:
        #kernel_size += 1
    #return medfilt(signal, kernel_size)

def smooth_signal(signal, kernel_size):
    x = np.asarray(signal, dtype=float)
    k = int(kernel_size)
    if k <= 1:
        return x
    if (k % 2) == 0:
        k += 1
    w = np.ones(k, dtype=float) / float(k)
    return np.convolve(x, w, mode="same")

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

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
    the values of the time history of the signal.
    window_size : int
    the length of the window. Must be an odd integer number.
    order : int
    the order of the polynomial used in the filtering.
    Must be less then `window_size` - 1.
    deriv: int
    the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
    the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    References
    ----------
    [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
    Data by Simplified Least Squares Procedures. Analytical
    Chemistry, 1964, 36 (8), pp 1627-1639.
    [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
    W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
    Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = abs(int(window_size))
        order = abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    
    # pad the signal at the extremes with values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    
    return np.convolve( m[::-1], y, mode='valid')