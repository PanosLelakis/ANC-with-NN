import numpy as np
import matplotlib.pyplot as plt
from utils.smoothing import smooth_signal
from utils.convert_to_dbfs import convert_to_dbfs
from utils.fft_transform import compute_fft
from matplotlib.ticker import ScalarFormatter

def figure_title_metadata(algorithm_name="", mu=None, L=None, noise_type="", snr=None,
                 convergence_time=None, steady_state_error=None, title_line3=""):
    # Create the simulation metadata title
    title_line1 = f"Algorithm: {algorithm_name} | μ: {mu} | L: {L} | Noise: {noise_type} | SNR: {snr} dB"

    # Format the convergence time
    if convergence_time is None:
        convergence_str = "Convergence Speed: N/A"
    else:
        convergence_str = f"Convergence Speed: {convergence_time:.2f} ms"

    # Format steady-state error
    if steady_state_error is None:
        error_str = "Steady-State Error: N/A"
    else:
        error_str = f"Steady-State Error: {steady_state_error:.2f} dB"

    title_line2 = f"{convergence_str} | {error_str}"
    if title_line3 == "":
        plt.suptitle(f"{title_line1}\n{title_line2}", fontsize=11, fontweight='bold')
    else:
        # Combine both lines into the suptitle
        plt.suptitle(f"{title_line1}\n{title_line2}\n{title_line3}", fontsize=11, fontweight='bold')

def plot_filter_weights(fs, w_final, #w_initial, 
                 algorithm_name="", mu=None, L=None, noise_type="", snr=None,
                 convergence_time=None, steady_state_error=None):
    
    freqs, w_fft = compute_fft(w_final, fs, n_fft=2**14)
    
    plt.figure()
    figure_title_metadata(algorithm_name, mu, L, noise_type, snr,
                 convergence_time, steady_state_error, "Filter Weights")
    plt.plot(w_final, label="Final Weights")
    plt.title("Filter Weights (Time Domain)")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Weight Value")
    plt.xlim([0, L])
    plt.grid()
    plt.legend()

    plt.figure()
    figure_title_metadata(algorithm_name, mu, L, noise_type, snr,
                 convergence_time, steady_state_error, "Filter Weights")
    plt.plot(freqs, w_fft, label="FFT of Weights")
    plt.title("Filter Weights (Frequency Domain)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xscale("log")
    plt.xlim([10, 10000])
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_path_analysis(path_ir, signal_before, signal_after, t, fs, title_prefix,
                 algorithm_name="", mu=None, L=None, noise_type="", snr=None,
                 convergence_time=None, steady_state_error=None):

    freqs_path, path_fft = compute_fft(path_ir, fs, n_fft=2**14)
    freqs_before, before_fft = compute_fft(signal_before, fs, n_fft=2**14)
    freqs_after, after_fft = compute_fft(signal_after, fs, n_fft=2**14)

    plt.figure()
    figure_title_metadata(algorithm_name, mu, L, noise_type, snr,
                          convergence_time, steady_state_error,
                          f"{title_prefix} Path Frequency Domain")
    plt.plot(freqs_path, path_fft, label="Impulse Response FFT")
    plt.title(f"{title_prefix} Path Frequency Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xscale("log")
    plt.xlim([10, 10000])
    plt.grid()
    plt.legend()

    plt.figure()
    figure_title_metadata(algorithm_name, mu, L, noise_type, snr,
                          convergence_time, steady_state_error,
                          f"{title_prefix} Path Frequency Domain")
    plt.plot(freqs_before, before_fft, label="Input Signal", alpha=0.7)
    plt.plot(freqs_after, after_fft, label="After Convolution", alpha=0.7)
    plt.title(f"Signal Before & After {title_prefix} Path (FFT)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xscale("log")
    plt.xlim([10, 10000])
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_error_analysis(error_signal, t, fs, passive_cancelling=None,
                 algorithm_name="", mu=None, L=None, noise_type="", snr=None,
                 convergence_time=None, steady_state_error=None):

    # Use last 20% of samples
    start_idx = int(0.8 * len(error_signal))
    active_segment = error_signal[start_idx:]
    freqs, error_fft = compute_fft(active_segment, fs, n_fft=2**14)

    max_value = np.max(abs(error_signal))
    error_dbfs = convert_to_dbfs(error_signal, max_value)
    error_dbfs_smooth = smooth_signal(error_dbfs, 401)

    if passive_cancelling is not None:
        passive_dbfs = convert_to_dbfs(passive_cancelling, max_value)
        passive_dbfs_smooth = smooth_signal(passive_dbfs, 401)
        passive_segment = passive_cancelling[start_idx:]
        _, passive_fft = compute_fft(passive_segment, fs, n_fft=2**14)

    plt.figure()
    figure_title_metadata(algorithm_name, mu, L, noise_type, snr,
                 convergence_time, steady_state_error, "Error Signal Analysis")
    if passive_cancelling is not None:
        plt.plot(t, passive_dbfs_smooth, label="Passive Cancelling", linestyle="--", alpha=0.7)
        plt.plot(t, error_dbfs_smooth, label="Active Cancelling", linestyle="-", alpha=0.7)
    else:
        plt.plot(t, error_dbfs_smooth, label="Active Cancelling", linestyle="-")
    plt.title("Active - Passive Cancelling (Time Domain)")
    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude (dBFS)")
    plt.xlim([0, 0.5])
    plt.legend()
    plt.grid()

    plt.figure()
    figure_title_metadata(algorithm_name, mu, L, noise_type, snr,
                 convergence_time, steady_state_error, "Error Signal Analysis")
    if passive_cancelling is not None:
        plt.plot(freqs, passive_fft, label="Passive Cancelling FFT", alpha=0.7)
        plt.plot(freqs, error_fft, label="Active Cancelling FFT", alpha=0.7)
    else:
        plt.plot(freqs, error_fft, label="Active Cancelling FFT")
    plt.title("Active - Passive Cancelling (Frequency Domain)")
    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xscale("log")
    plt.xlim([10, 10000])
    plt.ylim([-30, 60])
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_signal_flow(reference, noisy, filtered, t,
                 algorithm_name="", mu=None, L=None, noise_type="", snr=None,
                 convergence_time=None, steady_state_error=None):
    
    max_value = np.max(abs(noisy))
    reference_dbfs = convert_to_dbfs(reference, max_value)
    noisy_dbfs = convert_to_dbfs(noisy, max_value)
    filtered_dbfs = convert_to_dbfs(filtered, max_value)

    plt.figure()
    figure_title_metadata(algorithm_name, mu, L, noise_type, snr,
                 convergence_time, steady_state_error, "Signal Flow Comparison")

    plt.subplot(2, 2, 1)
    plt.plot(t, reference, label="Reference", alpha=0.7)
    plt.plot(t, noisy, label="Noisy", alpha=0.7)
    plt.title("Reference vs Noisy (Time Domain)")
    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude")
    plt.xlim([0, 0.5])
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(t, noisy, label="Noisy", alpha=0.7)
    plt.plot(t, filtered, label="Filtered", alpha=0.7)
    plt.title("Noisy vs Filtered (Time Domain)")
    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude")
    plt.xlim([0, 0.5])
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(t, reference_dbfs, label="Reference", alpha=0.7)
    plt.plot(t, noisy_dbfs, label="Noisy", alpha=0.7)
    plt.title("Reference vs Noisy (Time Domain)")
    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude (dBFS)")
    plt.xlim([0, 0.5])
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(t, noisy_dbfs, label="Noisy", alpha=0.7)
    plt.plot(t, filtered_dbfs, label="Filtered", alpha=0.7)
    plt.title("Noisy vs Filtered (Time Domain)")
    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude (dBFS)")
    plt.xlim([0, 0.5])
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()