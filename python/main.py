import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
import sys
import time
from algorithms.lms import LMS
from algorithms.nlms import NLMS
from algorithms.fxlms import FxLMS
from algorithms.fxnlms import FxNLMS
from utils.noise import add_noise
from utils.smoothing import smooth_signal
from utils.convert_to_dbfs import convert_to_dbfs

def run_anc(algorithm_name, L, mu, snr, progress_callback, completion_callback):
    """Run ANC process and update progress in real time."""
    
    # Start measuring execution time
    start_time = time.time()

    # Load system impulse responses
    primary_path = loadmat("primary_path.mat")['sim_imp'].flatten()[:4000]
    secondary_path = loadmat("secondary_path.mat")['sim_imp'].flatten()[:2000]

    # Generate input signal
    fs = 44100
    duration = 10
    t = np.arange(0, duration, 1/fs)
    reference_signal = np.sin(2 * np.pi * 500 * t)

    # Add noise with user-defined SNR
    noisy_signal = add_noise(reference_signal, snr)

    # Select the algorithm
    if algorithm_name == "LMS":
        algorithm = LMS(L, mu)
    elif algorithm_name == "NLMS":
        algorithm = NLMS(L, mu)
    elif algorithm_name == "FxLMS":
        algorithm = FxLMS(L, mu, secondary_path)
    elif algorithm_name == "FxNLMS":
        algorithm = FxNLMS(L, mu, secondary_path)
    else:
        print(f"Error: Unknown algorithm '{algorithm_name}'")
        sys.exit(1)

    # Apply ANC with the primary path
    filtered_signal = np.zeros(len(noisy_signal))
    error_signal = np.zeros(len(noisy_signal))
    primary_output = np.convolve(noisy_signal, primary_path, mode='full')[:len(noisy_signal)]

    for n in tqdm(range(len(noisy_signal))):
        error_signal[n], filtered_signal[n] = algorithm.estimate(noisy_signal[n], primary_output[n])

        # Update progress
        if n % (len(noisy_signal) // 100) == 0:
            progress = int((n / len(noisy_signal)) * 100)
            progress_callback(progress)

    # Save signals for playback
    #np.save("noisy_signal.npy", noisy_signal)
    #np.save("filtered_signal.npy", filtered_signal)

    # Compute total execution time
    end_time = time.time()
    total_execution_time = end_time - start_time

    # Compute convergence speed
    threshold = 0.01 * np.max(np.abs(error_signal))  # Define convergence threshold (1% of max error)
    for conv_idx in range(len(error_signal)):
        if np.abs(error_signal[conv_idx]) < threshold:
            convergence_time = conv_idx / fs
            break
    else:
        convergence_time = total_execution_time  # If it never converges, set it to total time

    # Compute steady-state error (last 20% of the time)
    last_20_percent_samples = int(0.2 * len(error_signal))
    steady_state_error = 10 * np.log10(np.mean(error_signal[-last_20_percent_samples:] ** 2) + 1e-10)  # dB

    # Send results to GUI callback
    completion_callback(reference_signal, noisy_signal, filtered_signal, error_signal, t, total_execution_time, convergence_time, steady_state_error)

def plot_results(reference_signal, noisy_signal, filtered_signal, error_signal, t):
    """Displays the results in a matplotlib figure using precomputed signals."""
    
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
    plt.figure() #figsize=(10, 6)

    plt.subplot(3,1,1)
    plt.plot(t, reference_signal, label="Original Signal", alpha=0.5)
    plt.plot(t, noisy_signal, label="Noisy Signal", alpha=0.5)
    plt.plot(t, filtered_signal, label="Filtered Signal", alpha=0.5)
    plt.plot(t, error_signal, label="Error Signal", alpha=0.5)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (Volt)")
    plt.grid()

    plt.subplot(3,1,2)
    plt.plot(t, reference_signal_dbfs, label="Original Signal", alpha=0.5)
    plt.plot(t, noisy_signal_dbfs, label="Noisy Signal", alpha=0.5)
    plt.plot(t, filtered_signal_dbfs, label="Filtered Signal", alpha=0.5)
    plt.plot(t, error_signal_dbfs, label="Error Signal", alpha=0.5)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (dBFS)")
    plt.grid()

    plt.subplot(3,1,3)
    plt.plot(t, reference_signal_dbfs_median, label="Original Signal", alpha=0.5)
    plt.plot(t, noisy_signal_dbfs_median, label="Noisy Signal", alpha=0.5)
    plt.plot(t, filtered_signal_dbfs_median, label="Filtered Signal", alpha=0.5)
    plt.plot(t, error_signal_dbfs_median, label="Error Signal", alpha=0.5)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (dBFS) - median")
    plt.grid()

    plt.tight_layout()
    plt.show()