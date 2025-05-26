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
        convergence_str = f"Convergence Speed: {convergence_time:.2f} sec"

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

def toggle_lines(checkbox, raw_lines, smooth_lines):
    for i in range(len(raw_lines)):
        show_raw = checkbox.get_status()[i]
        raw_lines[i].set_visible(show_raw)
        smooth_lines[i].set_visible(not show_raw)
    plt.draw()

def create_toggle(raw_lines, smooth_lines):
    ax_check = plt.add_axes([0.01, 0.5, 0.15, 0.15])
    check = CheckButtons(ax_check, [f"Show Raw {i+1}" for i in range(len(raw_lines))], [True]*len(raw_lines))
    check.on_clicked(lambda label: toggle_lines(check, raw_lines, smooth_lines))

def plot_filter_weights(w_initial, w_final,
                 algorithm_name="", mu=None, L=None, noise_type="", snr=None,
                 convergence_time=None, steady_state_error=None):
    
    plt.figure()
    figure_title_metadata(algorithm_name, mu, L, noise_type, snr,
                 convergence_time, steady_state_error, "Filter Weights (Initial vs Final)")
    
    plt.plot(w_initial, label="Initial")
    plt.plot(w_final, label="Final")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Value")
    plt.xlim([0, L])
    plt.grid()
    plt.legend()
    plt.show()

def plot_path_analysis(path_ir, signal_before, signal_after, t, fs, title_prefix,
                 algorithm_name="", mu=None, L=None, noise_type="", snr=None,
                 convergence_time=None, steady_state_error=None):

    plt.figure()
    figure_title_metadata(algorithm_name, mu, L, noise_type, snr,
                 convergence_time, steady_state_error, f"{title_prefix} Path Effect")

    plt.subplot(2, 2, 1)
    plt.plot(path_ir, label="Impulse Response")
    plt.title("Impulse Response")
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(t, signal_before, label="Input")
    plt.title("Signal Before Convolution")
    plt.xlim([0, 2])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(t, signal_after, label="Output")
    plt.title("Signal After Convolution")
    plt.xlim([0, 2])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_error_analysis(error_signal, t, fs,
                 algorithm_name="", mu=None, L=None, noise_type="", snr=None,
                 convergence_time=None, steady_state_error=None):
    
    error_smooth = smooth_signal(error_signal, 401)
    error_db = 20 * np.log10(np.abs(error_signal + 1e-10))
    error_db_smooth = smooth_signal(error_db, 401)
    freqs, error_fft = compute_fft(error_signal, fs)

    plt.figure()
    figure_title_metadata(algorithm_name, mu, L, noise_type, snr,
                 convergence_time, steady_state_error, "Error Signal Analysis")

    plt.subplot(2, 2, 1)
    raw1, = plt.plot(t, error_signal, label="Raw")
    smooth1, = plt.plot(t, error_smooth, label="Smoothed", linestyle="--")
    plt.title("Error Signal (Amplitude)")
    plt.xlim([0, 2])
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)
    raw2, = plt.plot(t, error_db, label="Raw")
    smooth2, = plt.plot(t, error_db_smooth, label="Smoothed", linestyle="--")
    plt.title("Error Signal (dB)")
    plt.xlim([0, 2])
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(freqs, error_fft, label="Error FFT", color="green")
    plt.title("Error Signal FFT")
    plt.set_xscale("log")
    plt.xlim([10, 10000])
    plt.legend()
    plt.grid()

    create_toggle([raw1, raw2], [smooth1, smooth2])
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
    plt.xlim([0, 2])
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(t, noisy, label="Noisy", alpha=0.7)
    plt.plot(t, filtered, label="Filtered", alpha=0.7)
    plt.title("Noisy vs Filtered")
    plt.xlim([0, 2])
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()