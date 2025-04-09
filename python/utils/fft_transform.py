import numpy as np

def compute_fft(signal, fs):
    """
    Computes the FFT of a signal and returns the frequency spectrum in dBFS.

    Parameters:
    - signal (numpy array): Input time-domain signal.
    - fs (int): Sampling frequency in Hz.

    Returns:
    - freqs (numpy array): Frequency bins (Hz).
    - signal_dbfs (numpy array): Magnitude spectrum in dBFS.
    """
    N = len(signal)  # Number of samples
    fft_result = np.fft.fft(signal)  # Compute FFT
    fft_magnitude = np.abs(fft_result[:N // 2])  # Take magnitude of positive frequencies
    freqs = np.fft.fftfreq(N, 1/fs)[:N // 2]  # Compute frequency bins

    # Convert magnitude to dBFS (avoid log(0) errors)
    fft_magnitude_dbfs = 20 * np.log10(fft_magnitude + 1e-10)

    return freqs, fft_magnitude_dbfs