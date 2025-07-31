import numpy as np
from scipy.io import loadmat
from scipy.io import wavfile
import sys
import time
from algorithms.lms import LMS
from algorithms.nlms import NLMS
from algorithms.fxlms import FxLMS
from algorithms.fxnlms import FxNLMS
from utils.noise import *
from utils.performance_metrics import compute_convergence_time, compute_steady_state_error
from utils.smoothing import smooth_signal
import mat73

def run_anc(algorithm_name, L, mu, snr, noise_type,
            input_type, freq, amp, wav_path, duration,
            progress_callback, completion_callback):
    
    # Start measuring execution time
    start_time = time.time()

    # Load system impulse responses
    #primary_path = loadmat("python/primary_path.mat")['sim_imp'].flatten()[:4000]
    #secondary_path = loadmat("python/secondary_path.mat")['sim_imp'].flatten()[:2000]
    #primary_path = mat73.loadmat("python/primary_path_new.mat")['sim_imp'].flatten()[:4000]
    #secondary_path = loadmat("python/secondary_path_new.mat")['sim_imp'].flatten()[:2000]
    fs, primary_path = wavfile.read("python/primary_paths/primary_anechoic.wav")
    _, secondary_path = wavfile.read("python/secondary_paths/secondary_anechoic.wav")

    # Normalize path impulse responses to have peak at 0 dBFS
    primary_path = primary_path / np.max(np.abs(primary_path))
    secondary_path = secondary_path / np.max(np.abs(secondary_path))
    
    primary_path = primary_path.astype(np.float32)
    secondary_path = secondary_path.astype(np.float32)

    # Generate input signal
    duration = float(duration)
    if input_type == "Sinusoid":
        freq = float(freq)
        amp = float(amp)
        t = np.arange(0, duration, 1/fs)
        reference_signal = amp * np.sin(2 * np.pi * freq * t)

    elif input_type == "WAV":
        fs, reference_signal = wavfile.read(wav_path)
        #reference_signal = reference_signal.astype(np.float32)
        total_dur = len(reference_signal) / fs
        if total_dur < duration:
            raise ValueError(f"Error: WAV duration {total_dur:.2f}s < selected duration {duration}s")
        reference_signal = reference_signal[:int(duration * fs)]
        t = np.arange(0, len(reference_signal)) / fs
    
    reference_signal = reference_signal / np.max(np.abs(reference_signal))

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

    noise = noise_generators[noise_type](reference_signal, snr)
    noisy_signal = generate_noisy_signal(reference_signal, noise)

    initial_weights = np.zeros(L)
    init_weights = initial_weights.copy()

    # Select algorithm
    if algorithm_name == "LMS":
        algorithm = LMS(L, mu, initial_weights)
    elif algorithm_name == "NLMS":
        algorithm = NLMS(L, mu, initial_weights)
    elif algorithm_name == "FxLMS":
        algorithm = FxLMS(L, mu, initial_weights)
    elif algorithm_name == "FxNLMS":
        algorithm = FxNLMS(L, mu, initial_weights)
    else:
        print(f"Error: Unknown algorithm '{algorithm_name}'")
        sys.exit(1)

    # Apply ANC with the primary path
    filtered_signal = np.zeros(len(noisy_signal))
    error_signal = np.zeros(len(noisy_signal))
    primary_output = np.convolve(noisy_signal, primary_path, mode='full')[:len(noisy_signal)]
    secondary_output = None

    if algorithm_name == "FxLMS" or algorithm_name == "FxNLMS":
        secondary_output = np.convolve(noisy_signal, secondary_path, mode='full')[:len(noisy_signal)]
        for n in range(len(noisy_signal)):
            error_signal[n], filtered_signal[n] = algorithm.estimate(secondary_output[n], primary_output[n])

            # Update progress
            if n % (len(noisy_signal) // 100) == 0:
                progress = int((n / len(noisy_signal)) * 100)
                progress_callback(progress)
        
    else:
        for n in range(len(noisy_signal)):
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

    error_signal_median = smooth_signal(error_signal, 401)

    # Compute performance metrics
    #convergence_time = compute_convergence_time(error_signal, fs)
    #steady_state_error = compute_steady_state_error(error_signal)
    steady_state_error = compute_steady_state_error(error_signal_median)
    steady_state_error_db = 20 * np.log10(steady_state_error + 1e-10)
    convergence_time = compute_convergence_time(error_signal_median, fs, steady_state_error)

    # Compute noise power
    #noise_power = compute_noise_power(noise)

    # Send results to GUI callback
    completion_callback(
        reference_signal, noisy_signal, filtered_signal, error_signal, 
        t, fs, total_execution_time, convergence_time, steady_state_error_db,
        init_weights, algorithm.w, primary_path, secondary_path,
        primary_output, secondary_output#, noise_power
    )