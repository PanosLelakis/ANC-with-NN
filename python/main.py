import numpy as np
from scipy.io import loadmat
from scipy.io import wavfile
from scipy.signal import resample
import sys
import time
from algorithms.lms import LMS
from algorithms.nlms import NLMS
from algorithms.fxlms import FxLMS
from algorithms.fxnlms import FxNLMS
from utils.noise import *
from utils.performance_metrics import compute_convergence_time, compute_steady_state_error
from utils.smoothing import smooth_signal
import warnings
from scipy.io.wavfile import WavFileWarning
import faulthandler

faulthandler.enable()

#import mat73

warnings.simplefilter("ignore", WavFileWarning)

def run_anc(algorithm_name, L, mu, noise_source, noise_type, noise_wav_path, duration,
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
    primary_path = (primary_path / np.max(np.abs(primary_path))).astype(np.float32)
    secondary_path = (secondary_path / np.max(np.abs(secondary_path))).astype(np.float32)
    
    primary_path = primary_path.astype(np.float32)
    secondary_path = secondary_path.astype(np.float32)
    #def _l2(x): return np.sqrt(np.sum(x.astype(np.float64)**2) + 1e-12)
    #primary_path  = primary_path  / _l2(primary_path)
    #secondary_path= secondary_path/ _l2(secondary_path)

    # Generate input signal
    # --- Noise generation (stationary or WAV) ---
    duration = float(duration)
    N = int(duration * fs)
    t = np.arange(N) / fs
    reference_signal = np.zeros(N, dtype=np.float32)

    if noise_source == "WAV" and noise_wav_path:
        wav_fs, wav_data = wavfile.read(noise_wav_path)
        # mono
        if wav_data.ndim > 1:
            wav_data = wav_data.mean(axis=1)
        # scale ints to [-1,1]
        if np.issubdtype(wav_data.dtype, np.integer):
            max_i = max(1.0, float(np.iinfo(wav_data.dtype).max))
            wav = wav_data.astype(np.float32) / max_i
        else:
            wav = wav_data.astype(np.float32)
        # resample
        if wav_fs != fs:
            new_len = int(len(wav) * fs / wav_fs)
            wav = resample(wav, new_len).astype(np.float32)
        # loop/trim to N
        if len(wav) >= N:
            noise = wav[:N]
        else:
            reps = int(np.ceil(N / len(wav))); noise = np.tile(wav, reps)[:N]
        noise = noise.astype(np.float32)
        noise /= np.sqrt(np.mean(noise**2) + 1e-12)

    else:
        # Stationary noise (existing types)
        noise_generators = {
            "White": generate_white_noise_len,
            "Pink": generate_pink_noise_len,
            "Brownian": generate_brownian_noise_len,
            "Violet": generate_violet_noise_len,
            "Grey": generate_grey_noise_len,
            "Blue": generate_blue_noise_len,
        }
        if noise_type not in noise_generators:
            print(f"Error: Unknown noise type '{noise_type}'"); sys.exit(1)
        noise = noise_generators[noise_type](N).astype(np.float32)

    noisy_signal = generate_noisy_signal(reference_signal, noise)

    L = int(L)
    mu = float(mu)
    initial_weights = np.zeros(L, dtype=np.float32)
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

    # --- Convolve through plant paths (mic-domain raw) ---
    primary_output_raw = np.convolve(noisy_signal, primary_path,   mode='full')[:N]
    secondary_output_raw = np.convolve(noisy_signal, secondary_path, mode='full')[:N]

    # --- Form streams for the algorithm (PEAK normalization, independently) ---
    def _robust_peak(x, q=99.5):
        a = np.abs(x)
        return float(np.percentile(a, q)) + 1e-12

    # --- Form streams for the algorithm (robust PEAK normalization, independently) ---
    pmax = _robust_peak(primary_output_raw,   q=99.5)
    smax = _robust_peak(secondary_output_raw, q=99.5)
    primary_stream   = primary_output_raw   / pmax   # d[n]
    secondary_stream = secondary_output_raw / smax   # z[n]
    
    # --- Sample loop (finite guards) ---
    antinoise_signal = np.zeros(N, dtype=np.float32)
    error_signal = np.zeros(N, dtype=np.float32)
    progress_step = max(1, N // 100)

    MAX_ABS = 10.0          # hard safety clamp for en/yn to avoid blow-ups
    W_NORM_CAP = 1e4       # if ||w|| exceeds this, treat as divergent

    if algorithm_name == "LMS" or algorithm_name == "NLMS":
        secondary_stream = noisy_signal

    for n in range(N):
        en, yn = algorithm.estimate(secondary_stream[n], primary_stream[n])
        if not np.isfinite(en): en = 0.0
        if not np.isfinite(yn): yn = 0.0
        en = float(np.clip(en, -MAX_ABS, MAX_ABS))
        yn = float(np.clip(yn, -MAX_ABS, MAX_ABS))
        error_signal[n] = en
        antinoise_signal[n] = yn

        # divergence guard on weights
        try:
            if (not np.all(np.isfinite(algorithm.w))) or (np.linalg.norm(algorithm.w) > W_NORM_CAP):
                # zero-out the rest and stop early
                error_signal[n+1:] = 0.0
                antinoise_signal[n+1:] = 0.0
                break
        except Exception:
            pass

        # Update progress
        if (n % progress_step) == 0:
            progress_callback(int((n / N) * 100))

    # Save signals for playback
    #np.save("noisy_signal.npy", noisy_signal)
    #np.save("antinoise_signal.npy", antinoise_signal)

    # Compute total execution time
    end_time = time.time()
    total_execution_time = end_time - start_time

    error_signal_median = smooth_signal(error_signal, 401)

    # Compute performance metrics
    steady_state_error = compute_steady_state_error(error_signal_median)
    steady_state_error_db = 20 * np.log10(steady_state_error + 1e-10)
    convergence_time = compute_convergence_time(error_signal_median, fs, steady_state_error)

    error_signal = np.nan_to_num(error_signal, nan=0.0, posinf=0.0, neginf=0.0)
    antinoise_signal = np.nan_to_num(antinoise_signal, nan=0.0, posinf=0.0, neginf=0.0)

    after_signal_raw  = np.clip(pmax * error_signal, -1e3, 1e3)
    before_signal_raw = primary_output_raw

    # Compute noise power
    tail = slice(int(0.8 * N), N)
    in_power  = compute_noise_power(before_signal_raw[tail])
    out_power = compute_noise_power(after_signal_raw[tail])

    # Send results to GUI callback
    completion_callback(
        reference_signal, noisy_signal, error_signal, 
        t, fs, total_execution_time, convergence_time, steady_state_error_db,
        init_weights, algorithm.w, primary_path, secondary_path,
        primary_stream, secondary_stream, in_power, out_power,
        before_signal_raw, after_signal_raw
    )

def run_anc_headless(algorithm_name, L, mu,
                     noise_source, noise_type, noise_wav_path,
                     duration, rng_seed=None):
    """
    Like run_anc, but returns a dict of metrics (no GUI callbacks).
    Used for multi-run so metrics match single-run.
    """
    results = {}

    def dummy_progress(pct): pass
    def dummy_completion(reference_signal, noisy_signal, error_signal,
                         t, fs, exec_time, conv_time, steady_error_db,
                         initial_weights, final_weights, primary_ir, secondary_ir,
                         signal_after_primary, signal_after_secondary,
                         in_power, out_power,
                         before_signal_raw, after_signal_raw):
        results.update({
            "mu": float(mu), "L": int(L),
            "conv_ms": 0.0 if conv_time is None else float(conv_time),
            "sse_db": float(steady_error_db),
            "in_power": float(in_power),
            "out_power": float(out_power),
            "fs": int(fs),
            # keep small tails for later plotting/playback if needed
            "d_mic_raw_tail": before_signal_raw,
            "e_mic_raw_tail": after_signal_raw,
        })

    run_anc(algorithm_name, L, mu,
            noise_source, noise_type, noise_wav_path,
            duration, dummy_progress, dummy_completion)

    return results