import numpy as np
from scipy.io import wavfile
from scipy.signal import resample, lfilter
import time
from engine.engine_common import load_paths
from algorithms.lms import LMS
from algorithms.nlms import NLMS
from algorithms.fxlms import FxLMS
from algorithms.fxnlms import FxNLMS
from utils.noise import *
from utils.performance_metrics import compute_convergence_time, compute_steady_state_error
from utils.smoothing import whittaker_eilers_smooth, moving_rms
from utils.convert_to_db import val_to_db, val_to_dbr
#from utils.fft_transform import compute_fft
import warnings
from scipy.io.wavfile import WavFileWarning
import faulthandler

# Enable faulthandler for better debugging of crashes
faulthandler.enable()

# Ignore WAV file warnings because they mean nothing
warnings.simplefilter("ignore", WavFileWarning)

def _make_noise(N, fs, noise_source, noise_type, noise_wav_path):
    """Generate stationary noise or load/loop a WAV, then (for WAV) RMS-normalize."""
    if noise_source == "WAV" and noise_wav_path:
        wav_fs, wav_data = wavfile.read(noise_wav_path)

        # If stereo, convert to mono
        if wav_data.ndim > 1:
            wav_data = wav_data.mean(axis=1)

        # Convert to float32
        wav = wav_data.astype(np.float32)

        # Scale to [-1, 1]
        if np.issubdtype(wav_data.dtype, np.integer):
            wav /= max(1.0, float(np.iinfo(wav_data.dtype).max))

        # Resample if needed
        if wav_fs != fs:
            new_len = int(len(wav) * fs / wav_fs)
            wav = resample(wav, new_len).astype(np.float32)

        # Loop/trim to N if needed
        if len(wav) >= N:
            noise = wav[:N] # Trim
        else:
            reps = int(np.ceil(N / max(1, len(wav))))
            noise = np.tile(wav, reps)[:N] # Loop

        # RMS normalize so different WAVs have comparable energy
        noise /= np.sqrt(np.mean(noise**2) + 1e-12)
        
        # Return wav-loaded noise
        return noise

    else:
        # Known noise colors
        gens = {
            "White": generate_white_noise_len,
            "Pink": generate_pink_noise_len,
            "Brownian": generate_brownian_noise_len,
            "Violet": generate_violet_noise_len,
            "Grey": generate_grey_noise_len,
            "Blue": generate_blue_noise_len,
        }

        # If unknown noise type, raise error
        if noise_type not in gens:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Return generated colored noise
        return gens[noise_type](N).astype(np.float32)

def compute_metrics(start_time, error_signal, noisy_signal, fs, N, anc_off_signal):
    # Compute total execution time
    exec_time = time.time() - start_time

    # Convert error signal to dBr
    win = max(32, int(0.02 * fs))
    ref = np.sqrt(np.mean(noisy_signal ** 2) + 1e-12)
    error_dbr = val_to_dbr(moving_rms(error_signal, win), ref)

    # Smooth error signal dB curve
    error_dbr_smoothed = whittaker_eilers_smooth(error_dbr, lmbd=1e12)

    # Compute SSE from smoothed error signal
    sse_dbr = compute_steady_state_error(error_dbr_smoothed)

    # Compute convergence time from smoothed error signal
    conv_ms = compute_convergence_time(error_dbr_smoothed, fs, sse_dbr)
    
    # Compute input/output noise power from the tail of the signals
    tail = slice(int(0.8 * N), N)
    in_power = compute_noise_power(anc_off_signal[tail])
    out_power = compute_noise_power(error_signal[tail])
    
    # Return all metrics
    return exec_time, conv_ms, sse_dbr, in_power, out_power

def run_anc(algorithm_name, L, mu, noise_source, noise_type,
            noise_wav_path, duration, progress_callback,
            completion_callback=None, metrics_callback=None):
    """Single simulation engine"""

    # Start measuring execution time
    start_time = time.time()

    # Load paths
    fs, primary_path, secondary_path = load_paths()

    # Load constants and time vector
    duration = float(duration)
    N = int(duration * fs)
    t = np.arange(N) / fs
    L = int(L)
    mu = float(mu)
    initial_weights = np.zeros(L) # Zero inital weights (for simplicity)
    init_weights = initial_weights.copy()

    # Generate reference signal (zero)
    reference_signal = np.zeros(N, dtype=np.float32)

    # Load noise
    noise = _make_noise(N, fs, noise_source, noise_type, noise_wav_path)
    noisy_signal = generate_noisy_signal(reference_signal, noise)

    # Call selected algorithm constructor
    if algorithm_name == "LMS":
        algorithm = LMS(L, mu, initial_weights)
    elif algorithm_name == "NLMS":
        algorithm = NLMS(L, mu, initial_weights)
    elif algorithm_name == "FxLMS":
        algorithm = FxLMS(L, mu, initial_weights)
    elif algorithm_name == "FxNLMS":
        algorithm = FxNLMS(L, mu, initial_weights)
    else:
        # If uknown algorithm, raise error
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    # Convolve input with paths
    primary_output_raw = np.convolve(noisy_signal, primary_path, mode="full")[:N].astype(np.float32, copy=False)
    secondary_output_raw = np.convolve(noisy_signal, secondary_path, mode="full")[:N].astype(np.float32, copy=False)

    anc_off_rms_dbr = 20.0 * np.log10(
        (np.sqrt(np.mean(primary_output_raw ** 2)) + 1e-12) /
        (np.sqrt(np.mean(noisy_signal ** 2)) + 1e-12)
    )
    print(f"{noise_type} ANC OFF RMS relative to noisy RMS: {anc_off_rms_dbr:.2f} dBr")

    # Streams used by adaptive algorithm
    primary_stream = primary_output_raw # d[n]

    # LMS/NLMS use the raw noise as y[n]
    if algorithm_name in ("LMS", "NLMS"):
        secondary_stream = noisy_signal
    else:
        secondary_stream = secondary_output_raw

    # Initialize produced signals
    error_signal = np.zeros(N, dtype=np.float32)
    #antinoise_signal = np.zeros(N, dtype=np.float32)

    progress_step = max(1, N // 100)
    #MAX_ABS = 10.0 # Safety clamp for en/yn to avoid crazy blow-ups
    #W_NORM_CAP = 1e4 # If ||w|| exceeds this, treat as divergent (garbage)

    # Use a clamp relative to baseline peak (avoid hardcoded 10.0)
    #baseline_peak = float(np.max(np.abs(primary_output_raw)) + 1e-12)
    #MAX_ABS = 5.0 * baseline_peak

    zi = np.zeros(len(secondary_path) - 1, dtype=np.float64)

    for n in range(N):

        if algorithm_name in ("FxNLMS", "FxLMS"):
            # Controller output from raw x
            y = algorithm.predict(noisy_signal[n])
            if not np.isfinite(y):
                y = 0.0
            y = float(y)

            # Secondary path acts on controller output to create anti-noise at mic
            ys, zi = lfilter(secondary_path, [1.0], [y], zi=zi)

            # Physical residual at error mic (sign convention)
            e = primary_output_raw[n] - float(ys[0])

            # Update uses filtered-x
            algorithm.adapt(e, secondary_output_raw[n])

            # Finite guard
            if not np.isfinite(e):
                e = 0.0
            
            en = e

        else:
            en, _ = algorithm.estimate(noisy_signal[n], primary_output_raw[n])
        #en = float(np.clip(en, -MAX_ABS, MAX_ABS))
        error_signal[n] = float(en)
        
        # Divergence guard
        try:
            w = np.asarray(algorithm.w, dtype=float)
            if (not np.all(np.isfinite(w))) or (np.linalg.norm(w) > 1e4):
                error_signal[n:] = error_signal[n]
                break
        except Exception:
            pass
        
        if (n % progress_step) == 0:
            try:
                progress_callback(int((n / N) * 100))
            except Exception:
                pass

    # Make sure UI reaches 100%
    try:
        progress_callback(100)
    except Exception:
        pass

    after_signal_raw = np.clip(error_signal, -1e3, 1e3)
    before_signal_raw = primary_output_raw

    # Compute performance metrics
    exec_time, conv_ms, sse_db, in_power, out_power = compute_metrics(
        start_time, error_signal, noisy_signal, fs, N, before_signal_raw)

    if metrics_callback is not None:
        metrics_callback(
            fs=fs,
            conv_ms=conv_ms,
            sse_db=sse_db,
            in_power=in_power,
            out_power=out_power
        )

    if completion_callback is not None:
        completion_callback(
            reference_signal, noisy_signal, error_signal, t, fs, exec_time,
            conv_ms, sse_db, init_weights, algorithm.w, primary_path, secondary_path,
            primary_stream, secondary_stream, in_power, out_power,
            before_signal_raw, after_signal_raw
        )

def run_anc_headless(algorithm_name, L, mu, noise_source, noise_type,
                     noise_wav_path, duration):
    # Simulation results dict
    results = {}

    # Dummy progress callback for multi runner
    def _dummy_progress(_pct):
        pass

    # Metrics callback to capture results for multi runner
    def _capture_metrics(*, fs, conv_ms, sse_db, in_power, out_power):
        # Add simulation metrics to results dict
        results.update({
            "mu": float(mu),
            "L": int(L),
            "conv_ms": 0.0 if conv_ms is None else float(conv_ms),
            "sse_db": float(sse_db),
            "in_power": float(in_power),
            "out_power": float(out_power),
            "fs": int(fs)
        })

    # Run simulation for multi runner
    run_anc(
        algorithm_name, int(L), float(mu), noise_source, noise_type,
        noise_wav_path, float(duration),
        _dummy_progress,
        completion_callback=None,
        metrics_callback=_capture_metrics
    )

    # Return simulation results dict for multi runner
    return results

def simulate_once(algorithm_name, L, mu, noise_source, noise_type,
                    noise_wav_path, duration):
    """Compatibility wrapper used by engine_multi.py (joblib grid runner)."""
    # Single simulation run for multi runner
    return run_anc_headless(algorithm_name, L, mu, noise_source,
                                noise_type, noise_wav_path, duration)