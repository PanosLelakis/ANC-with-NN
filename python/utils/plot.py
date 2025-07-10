import numpy as np
import matplotlib.pyplot as plt
from utils.smoothing import smooth_signal
from utils.convert_to_dbfs import convert_to_dbfs
from utils.fft_transform import compute_fft
from matplotlib.ticker import ScalarFormatter
from matplotlib.widgets import CheckButtons

def plot_results(reference_signal, noisy_signal, filtered_signal, error_signal, t, fs,
                 algorithm_name="", mu=None, L=None, noise_type="", snr=None,
                 convergence_time=None, steady_state_error=None):

    # Convert signals to dBFS
    #max_val = np.max(np.abs(reference_signal))
    #reference_signal_dbfs = convert_to_dbfs(reference_signal, max_val)
    #noisy_signal_dbfs = convert_to_dbfs(noisy_signal, max_val)
    #filtered_signal_dbfs = convert_to_dbfs(filtered_signal, max_val)
    #error_signal_dbfs = convert_to_dbfs(error_signal, max_val)

    # Smoothed signals for visualization
    #reference_signal_dbfs_median = smooth_signal(reference_signal_dbfs, 401)
    #noisy_signal_dbfs_median = smooth_signal(noisy_signal_dbfs, 401)
    #filtered_signal_dbfs_median = smooth_signal(filtered_signal_dbfs, 401)
    #error_signal_dbfs_median = smooth_signal(error_signal_dbfs, 401)
    error_signal_db_median = smooth_signal(20 * np.log10(abs(error_signal + 1e-10)), 401)

    # Compute FFT of the error signal
    freqs, error_signal_fft = compute_fft(error_signal, fs)

    # Plot results
    plt.figure()

    # Create the simulation metadata title
    title_line1 = f"Algorithm: {algorithm_name} | μ: {mu} | L: {L} | Noise: {noise_type} | SNR: {snr} dB"

    # Format the convergence time
    if convergence_time is None:
        convergence_str = "Convergence Speed: N/A"
    else:
        convergence_str = f"Convergence Speed: {convergence_time:.2f} sec"

    # Format steady-state error
    if steady_state_error is None:
        error_str = "Steady-State Error: N/A"
    else:
        error_str = f"Steady-State Error: {steady_state_error:.2f} dB"

    title_line2 = f"{convergence_str} | {error_str}"

    # Combine both lines into the suptitle
    plt.suptitle(f"{title_line1}\n{title_line2}", fontsize=11, fontweight='bold')

    # Plot signals in time domain
    plt.subplot(2, 2, 1)
    #plt.plot(t, reference_signal, label="Original Signal", alpha=0.5)
    #plt.plot(t, noisy_signal, label="Noisy Signal", alpha=0.5)
    #plt.plot(t, filtered_signal, label="Filtered Signal", alpha=0.5)
    plt.plot(t, error_signal, label="Error Signal")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    #plt.gca().set_yticks([-3, -2, -1, 0, 1, 2, 3])
    plt.gca().minorticks_on()
    #plt.ylim([-3, 3])  # Set y-axis limit
    plt.xlim([0, 2])
    plt.grid()

    # Plot signals in dBFS
    plt.subplot(2, 2, 2)
    #plt.plot(t, reference_signal_dbfs, label="Original Signal", alpha=0.5)
    #plt.plot(t, noisy_signal_dbfs, label="Noisy Signal", alpha=0.5)
    #plt.plot(t, filtered_signal_dbfs, label="Filtered Signal", alpha=0.5)
    #plt.plot(t, error_signal_dbfs, label="Error Signal", alpha=0.5)
    plt.plot(t, 20 * np.log10(abs(error_signal + 1e-10)), label="Error Signal")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (dB)")
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    #plt.gca().set_yticks([-60, -40, -20, 0, 20])
    plt.gca().minorticks_on()
    #plt.ylim([-60, 20])  # Set y-axis limit
    plt.xlim([0, 2])
    plt.grid()

    # Plot in frequency domain
    plt.subplot(2, 2, 3)
    plt.plot(freqs, error_signal_fft, label="Error Signal FFT")
    #plt.plot(t, reference_signal_dbfs_median, label="Original Signal", alpha=0.5)
    #plt.plot(t, noisy_signal_dbfs_median, label="Noisy Signal", alpha=0.5)
    #plt.plot(t, filtered_signal_dbfs_median, label="Filtered Signal", alpha=0.5)
    #plt.plot(t, error_signal_dbfs_median, label="Error Signal", alpha=0.5)
    plt.legend()
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Amplitude (dB)")
    plt.gca().set_xscale('log')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    #plt.gca().set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    #plt.gca().set_yticks([-60, -40, -20, 0, 20])
    plt.gca().minorticks_on()
    plt.xlim([20, 10000])
    #plt.ylim([-20, 60])
    plt.grid()

    # Plot smoothed signals
    plt.subplot(2, 2, 4)
    #plt.plot(t, reference_signal_dbfs_median, label="Original Signal", alpha=0.5)
    #plt.plot(t, noisy_signal_dbfs_median, label="Noisy Signal", alpha=0.5)
    #plt.plot(t, filtered_signal_dbfs_median, label="Filtered Signal", alpha=0.5)
    #plt.plot(t, error_signal_dbfs_median, label="Error Signal", alpha=0.5)
    plt.plot(t, error_signal_db_median, label="Error Signal")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (dBFS) - Smoothed")
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    #plt.gca().set_yticks([-3, -2, -1, 0, 1, 2, 3])
    plt.gca().minorticks_on()
    #plt.ylim([-3, 3])  # Set y-axis limit
    plt.xlim([0, 2])
    plt.grid()

    plt.tight_layout()
    plt.show()

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

def add_toggle_panel(fig, raw_lines, smooth_lines):
    # Add checkbox axes to the right of the plot (x, y, width, height)
    toggle_ax = fig.add_axes([0.85, 0.4, 0.13, 0.2])
    labels = [f"Line {i+1}" for i in range(len(raw_lines))]

    # Initial checkbox states (True = show raw)
    visibility = [True] * len(raw_lines)

    check = CheckButtons(toggle_ax, labels, visibility)

    def toggle(label):
        index = labels.index(label)
        raw_lines[index].set_visible(not raw_lines[index].get_visible())
        smooth_lines[index].set_visible(not smooth_lines[index].get_visible())
        plt.draw()

    check.on_clicked(toggle)

def plot_filter_weights(fs, w_final, #w_initial, 
                 algorithm_name="", mu=None, L=None, noise_type="", snr=None,
                 convergence_time=None, steady_state_error=None):
    
    freqs, w_fft = compute_fft(w_final, fs)
    
    plt.figure()
    figure_title_metadata(algorithm_name, mu, L, noise_type, snr,
                 convergence_time, steady_state_error, "Filter Weights (Initial vs Final)")

    plt.subplot(2, 1, 1)
    #plt.plot(w_initial, label="Initial")
    plt.plot(w_final, label="Final Weights")
    plt.title("Filter Weights (Time Domain)")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Weight Value")
    plt.xlim([0, L])
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
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

    freqs_path, path_fft = compute_fft(path_ir, fs)
    freqs_before, before_fft = compute_fft(signal_before, fs)
    freqs_after, after_fft = compute_fft(signal_after, fs)

    plt.figure()
    figure_title_metadata(algorithm_name, mu, L, noise_type, snr,
                          convergence_time, steady_state_error,
                          f"{title_prefix} Path Frequency Domain")

    plt.subplot(2, 1, 1)
    plt.plot(freqs_path, path_fft, label="Impulse Response FFT")
    plt.title(f"{title_prefix} Path Frequency Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xscale("log")
    plt.xlim([10, 10000])
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
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
    
    #error_smooth = smooth_signal(error_signal, 401)
    #error_db = 20 * np.log10(np.abs(error_signal + 1e-10))
    #error_db_smooth = smooth_signal(error_db, 401)

    # Use last 20% of samples
    start_idx = int(0.8 * len(error_signal))
    active_segment = error_signal[start_idx:]
    freqs, error_fft = compute_fft(active_segment, fs)

    max_value = np.max(abs(error_signal))
    error_dbfs = convert_to_dbfs(error_signal, max_value)
    error_dbfs_smooth = smooth_signal(error_dbfs, 401)

    if passive_cancelling is not None:
        passive_dbfs = convert_to_dbfs(passive_cancelling, max_value)
        passive_dbfs_smooth = smooth_signal(passive_dbfs, 401)
        passive_segment = passive_cancelling[start_idx:]
        _, passive_fft = compute_fft(passive_segment, fs)

    plt.figure()
    figure_title_metadata(algorithm_name, mu, L, noise_type, snr,
                 convergence_time, steady_state_error, "Error Signal Analysis")

    plt.subplot(2, 1, 1)
    if passive_cancelling is not None:
        plt.plot(t, passive_dbfs_smooth, label="Passive Cancelling", linestyle="--", alpha=0.7)
        plt.plot(t, error_dbfs_smooth, label="Active Cancelling", linestyle="-", alpha=0.7)
    else:
        plt.plot(t, error_dbfs_smooth, label="Active Cancelling", linestyle="-")
    plt.title("Active - Passive Cancelling (Time Domain)")
    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude (dBFS)")
    plt.xlim([0, 1])
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
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
    plt.ylim([50, 130])
    plt.legend()
    plt.grid()

    # Add toggle after plotting
    #add_toggle_panel(plt.gcf(), [raw1, raw2], [smooth1, smooth2])

    #plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for the toggle panel
    plt.tight_layout()
    plt.show()

def plot_signal_flow(reference, noisy, filtered, t,
                 algorithm_name="", mu=None, L=None, noise_type="", snr=None,
                 convergence_time=None, steady_state_error=None):
    
    plt.figure()
    figure_title_metadata(algorithm_name, mu, L, noise_type, snr,
                 convergence_time, steady_state_error, "Signal Flow Comparison")

    plt.subplot(2, 1, 1)
    plt.plot(t, reference, label="Reference", alpha=0.7)
    plt.plot(t, noisy, label="Noisy", alpha=0.7)
    plt.title("Reference vs Noisy")
    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude")
    plt.xlim([0, 1])
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(t, noisy, label="Noisy", alpha=0.7)
    plt.plot(t, filtered, label="Filtered", alpha=0.7)
    plt.title("Noisy vs Filtered")
    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude")
    plt.xlim([0, 1])
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()