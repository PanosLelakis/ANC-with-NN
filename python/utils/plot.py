import numpy as np
import matplotlib.pyplot as plt
from utils.smoothing import smooth_signal
from utils.convert_to_dbfs import convert_to_dbfs

def plot_results(reference_signal, noisy_signal, filtered_signal, error_signal, t):
    """
    Plots the original, noisy, filtered, and error signals in both amplitude and dBFS scales.
    """
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

    # Plot results
    plt.figure(figsize=(10, 6))

    # Plot signals in time domain
    plt.subplot(3, 1, 1)
    plt.plot(t, reference_signal, label="Original Signal", alpha=0.5)
    plt.plot(t, noisy_signal, label="Noisy Signal", alpha=0.5)
    plt.plot(t, filtered_signal, label="Filtered Signal", alpha=0.5)
    plt.plot(t, error_signal, label="Error Signal", alpha=0.5)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (Volt)")
    plt.grid()

    # Plot signals in dBFS (with y-axis limit)
    plt.subplot(3, 1, 2)
    plt.plot(t, reference_signal_dbfs, label="Original Signal", alpha=0.5)
    plt.plot(t, noisy_signal_dbfs, label="Noisy Signal", alpha=0.5)
    plt.plot(t, filtered_signal_dbfs, label="Filtered Signal", alpha=0.5)
    plt.plot(t, error_signal_dbfs, label="Error Signal", alpha=0.5)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (dBFS)")
    plt.ylim([-60, 50])  # Set y-axis limit
    plt.grid()

    # Plot smoothed signals in dBFS
    plt.subplot(3, 1, 3)
    plt.plot(t, reference_signal_dbfs_median, label="Original Signal", alpha=0.5)
    plt.plot(t, noisy_signal_dbfs_median, label="Noisy Signal", alpha=0.5)
    plt.plot(t, filtered_signal_dbfs_median, label="Filtered Signal", alpha=0.5)
    plt.plot(t, error_signal_dbfs_median, label="Error Signal", alpha=0.5)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (dBFS) - Smoothed")
    plt.ylim([-60, 50])  # Set y-axis limit
    plt.grid()

    plt.tight_layout()
    plt.show()