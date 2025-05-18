import numpy as np
from scipy.io import loadmat
from scipy.io import wavfile
import sys
import time
from algorithms.lms import LMS
from algorithms.nlms import NLMS
from algorithms.fxlms import FxLMS
from algorithms.fxnlms import FxNLMS
from utils.noise import generate_white_noise, generate_pink_noise, generate_brownian_noise, generate_violet_noise, generate_grey_noise, generate_blue_noise
from utils.performance_metrics import compute_convergence_time, compute_steady_state_error
from utils.smoothing import smooth_signal
from utils.convert_to_dbfs import convert_to_dbfs
import mat73

def run_anc(algorithm_name, L, mu, snr, noise_type, progress_callback, completion_callback):
    
    # Start measuring execution time
    start_time = time.time()

    # Load system impulse responses
    #primary_path = loadmat("python/primary_path.mat")['sim_imp'].flatten()[:4000]
    #secondary_path = loadmat("python/secondary_path.mat")['sim_imp'].flatten()[:2000]
    #primary_path = mat73.loadmat("python/primary_path_new.mat")['sim_imp'].flatten()[:4000]
    #secondary_path = loadmat("python/secondary_path_new.mat")['sim_imp'].flatten()[:2000]
    fs, primary_path = wavfile.read("python/primary_anechoic.wav")
    _, secondary_path = wavfile.read("python/secondary_anechoic.wav")
    
    primary_path = primary_path.astype(np.float32)
    secondary_path = secondary_path.astype(np.float32)

    # Generate input signal
    #fs = 44100
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
        algorithm = FxLMS(L, mu)
    elif algorithm_name == "FxNLMS":
        algorithm = FxNLMS(L, mu)
    else:
        print(f"Error: Unknown algorithm '{algorithm_name}'")
        sys.exit(1)

    # Apply ANC with the primary path
    filtered_signal = np.zeros(len(noisy_signal))
    error_signal = np.zeros(len(noisy_signal))
    primary_output = np.convolve(noisy_signal, primary_path, mode='full')[:len(noisy_signal)]

    if algorithm_name == "FxLMS" or algorithm_name == "FxNLMS":
        secondary_output = np.convolve(noisy_signal, secondary_path, mode='full')[:len(noisy_signal)]
        for n in range(len(noisy_signal)):
            #error_signal[n], filtered_signal[n] = algorithm.estimate(secondary_output[n], noisy_signal[n])
            error_signal[n], filtered_signal[n] = algorithm.estimate(secondary_output[n], primary_output[n])

            # Update progress
            if n % (len(noisy_signal) // 100) == 0:
                progress = int((n / len(noisy_signal)) * 100)
                progress_callback(progress)
        
    else:
        for n in range(len(noisy_signal)):
            #error_signal[n], filtered_signal[n] = algorithm.estimate(secondary_output[n], noisy_signal[n])
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

    max_val = np.max(np.abs(reference_signal))
    #error_signal_dbfs = convert_to_dbfs(error_signal, max_val)
    error_signal_median = smooth_signal(error_signal, 101)

    # Compute performance metrics
    #convergence_time = compute_convergence_time(error_signal, fs)
    #steady_state_error = compute_steady_state_error(error_signal)
    convergence_time = compute_convergence_time(error_signal_median, fs)
    steady_state_error = compute_steady_state_error(error_signal_median)

    # Send results to GUI callback
    completion_callback(reference_signal, noisy_signal, filtered_signal, error_signal, t, fs, total_execution_time, convergence_time, steady_state_error)