import numpy as np
from scipy.io import loadmat
import sys
import time
from algorithms.lms import LMS
from algorithms.nlms import NLMS
from algorithms.fxlms import FxLMS
from algorithms.fxnlms import FxNLMS
from utils.noise import generate_white_noise, generate_pink_noise, generate_brownian_noise, generate_violet_noise, generate_grey_noise, generate_blue_noise
from utils.performance_metrics import compute_convergence_time, compute_steady_state_error

def run_anc(algorithm_name, L, mu, snr, noise_type, progress_callback, completion_callback):
    
    # Start measuring execution time
    start_time = time.time()

    # Load system impulse responses
    primary_path = loadmat("python/primary_path.mat")['sim_imp'].flatten()[:4000]
    secondary_path = loadmat("python/secondary_path.mat")['sim_imp'].flatten()[:2000]

    # Generate input signal
    fs = 44100
    duration = 10
    t = np.arange(0, duration, 1/fs)
    reference_signal = np.sin(2 * np.pi * 500 * t)

    # Select noise type
    noise_generators = {
        "White": generate_white_noise,
        "Pink": generate_pink_noise,
        "Brownian": generate_brownian_noise,
        "Violet": generate_violet_noise,
        "Grey": generate_grey_noise,
        "Blue": generate_blue_noise,
    }

    if noise_type not in noise_generators:
        print(f"Error: Unknown noise type '{noise_type}'")
        sys.exit(1)

    noisy_signal = noise_generators[noise_type](reference_signal, snr)

    # Select algorithm
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

    if algorithm_name == "FxLMS" or algorithm_name == "FxNLMS":
        secondary_output = np.convolve(primary_output, secondary_path, mode='full')[:len(noisy_signal)]
    else:
        secondary_output = primary_output

    for n in range(len(noisy_signal)):
        error_signal[n], filtered_signal[n] = algorithm.estimate(secondary_output[n], noisy_signal[n])

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

    # Compute performance metrics
    convergence_time = compute_convergence_time(error_signal, fs)
    steady_state_error = compute_steady_state_error(error_signal, fs)

    # Send results to GUI callback
    completion_callback(reference_signal, noisy_signal, filtered_signal, error_signal, t, total_execution_time, convergence_time, steady_state_error)