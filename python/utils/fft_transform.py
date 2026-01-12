import numpy as np

def compute_fft(signal, fs, n_fft=None):
    """
    Computes the FFT of a signal and returns the frequency spectrum in dBFS.

    Parameters:
    - signal (numpy array): Input time-domain signal.
    - fs (int): Sampling frequency in Hz.
    - n_fft (int or None): FFT size. If None, defaults to len(signal).
                           Larger n_fft (e.g. 2x, 4x signal length) gives
                           better frequency resolution in the plot.

    Returns:
    - freqs (numpy array): Frequency bins (Hz).
    - signal_db (numpy array): Magnitude spectrum in dB.
    """
    x = np.asarray(signal, dtype=float)
    N = len(x)
    
    if N == 0:
        return np.array([0.0]), np.array([0.0])
    if n_fft is None or n_fft < N:
        n_fft = N

    X = np.fft.rfft(x, n=n_fft)
    mag = np.abs(X)

    # amplitude normalization: full-scale sine (amp=1) ~ 0 dB
    #mag = mag / (n_fft / 2.0)

    freqs = np.fft.rfftfreq(n_fft, 1.0/fs)
    mag_db = 20.0 * np.log10(mag + 1e-12)
    return freqs, mag_db