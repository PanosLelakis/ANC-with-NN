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
    - signal_dbfs (numpy array): Magnitude spectrum in dBFS.
    """
    N = len(signal)
    if n_fft is None or n_fft < N:
        n_fft = N  # Default to signal length if not specified or too small

    # Apply zero-padding if n_fft > N
    fft_result = np.fft.fft(signal, n=n_fft)
    fft_magnitude = np.abs(fft_result[:n_fft // 2])
    freqs = np.fft.fftfreq(n_fft, 1/fs)[:n_fft // 2]

    # Convert to dBFS (avoid log(0))
    fft_magnitude_dbfs = 20 * np.log10(fft_magnitude + 1e-10)

    return freqs, fft_magnitude_dbfs