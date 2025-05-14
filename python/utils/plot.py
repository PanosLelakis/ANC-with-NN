import numpy as np
import matplotlib.pyplot as plt
from utils.smoothing import smooth_signal
from utils.convert_to_dbfs import convert_to_dbfs
from utils.fft_transform import compute_fft
from matplotlib.ticker import ScalarFormatter

def plot_results(reference_signal, noisy_signal, filtered_signal, error_signal, t, fs):
    # Convert signals to dBFS
    max_val = np.max(np.abs(reference_signal))
    reference_signal_dbfs = convert_to_dbfs(reference_signal, max_val)
    noisy_signal_dbfs = convert_to_dbfs(noisy_signal, max_val)
    filtered_signal_dbfs = convert_to_dbfs(filtered_signal, max_val)
    error_signal_dbfs = convert_to_dbfs(error_signal, max_val)

    # Smoothed signals for visualization
    reference_signal_dbfs_median = smooth_signal(reference_signal_dbfs, 401)
    noisy_signal_dbfs_median = smooth_signal(noisy_signal_dbfs, 401)
    filtered_signal_dbfs_median = smooth_signal(filtered_signal_dbfs, 401)
    error_signal_dbfs_median = smooth_signal(error_signal_dbfs, 401)
    error_signal_db_median = smooth_signal(20 * np.log10(abs(error_signal + 1e-10)), 401)

    # Compute FFT of the error signal
    freqs, error_signal_fft = compute_fft(error_signal, fs)

    # Plot results
    #plt.figure(figsize=(10, 6))

    # Plot signals in time domain
    plt.subplot(2, 2, 1)
    #plt.plot(t, reference_signal, label="Original Signal", alpha=0.5)
    #plt.plot(t, noisy_signal, label="Noisy Signal", alpha=0.5)
    #plt.plot(t, filtered_signal, label="Filtered Signal", alpha=0.5)
    plt.plot(t, error_signal, label="Error Signal", alpha=0.5)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    #plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    #plt.gca().set_yticks([-3, -2, -1, 0, 1, 2, 3])
    #plt.gca().minorticks_on()
    #plt.ylim([-3, 3])  # Set y-axis limit
    #plt.xlim([0, 2])
    plt.grid()

    # Plot signals in dBFS
    plt.subplot(2, 2, 2)
    #plt.plot(t, reference_signal_dbfs, label="Original Signal", alpha=0.5)
    #plt.plot(t, noisy_signal_dbfs, label="Noisy Signal", alpha=0.5)
    #plt.plot(t, filtered_signal_dbfs, label="Filtered Signal", alpha=0.5)
    #plt.plot(t, error_signal_dbfs, label="Error Signal", alpha=0.5)
    plt.plot(t, 20 * np.log10(abs(error_signal + 1e-10)), label="Error Signal", alpha=0.5)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (dB)")
    #plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    #plt.gca().set_yticks([-60, -40, -20, 0, 20])
    #plt.gca().minorticks_on()
    #plt.ylim([-60, 20])  # Set y-axis limit
    #plt.xlim([0, 2])
    plt.grid()

    # Plot in frequency domain
    plt.subplot(2, 2, 3)
    plt.plot(freqs, error_signal_fft, label="Error Signal FFT", alpha=0.7)
    #plt.plot(t, reference_signal_dbfs_median, label="Original Signal", alpha=0.5)
    #plt.plot(t, noisy_signal_dbfs_median, label="Noisy Signal", alpha=0.5)
    #plt.plot(t, filtered_signal_dbfs_median, label="Filtered Signal", alpha=0.5)
    #plt.plot(t, error_signal_dbfs_median, label="Error Signal", alpha=0.5)
    plt.legend()
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Amplitude (dB)")
    plt.gca().set_xscale('log')
    #plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    #plt.gca().set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
    #plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    #plt.gca().set_yticks([-60, -40, -20, 0, 20])
    #plt.gca().minorticks_on()
    #plt.xlim([20, 10000])
    #plt.ylim([-20, 60])
    plt.grid()

    # Plot smoothed signals
    plt.subplot(2, 2, 4)
    #plt.plot(t, reference_signal_dbfs_median, label="Original Signal", alpha=0.5)
    #plt.plot(t, noisy_signal_dbfs_median, label="Noisy Signal", alpha=0.5)
    #plt.plot(t, filtered_signal_dbfs_median, label="Filtered Signal", alpha=0.5)
    #plt.plot(t, error_signal_dbfs_median, label="Error Signal", alpha=0.5)
    plt.plot(t, error_signal_db_median, label="Error Signal", alpha=0.5)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (dBFS) - Smoothed")
    #plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    #plt.gca().set_yticks([-3, -2, -1, 0, 1, 2, 3])
    #plt.gca().minorticks_on()
    #plt.ylim([-3, 3])  # Set y-axis limit
    #plt.xlim([0, 2])
    plt.grid()

    #plt.tight_layout()
    plt.show()