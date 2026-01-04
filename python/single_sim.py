import numpy as np
from scipy.io import wavfile
from algorithms.lms import LMS
from algorithms.nlms import NLMS
from algorithms.fxlms import FxLMS
from algorithms.fxnlms import FxNLMS
from utils.noise import (
    generate_white_noise_len, generate_pink_noise_len, generate_brownian_noise_len,
    generate_violet_noise_len, generate_grey_noise_len, generate_blue_noise_len,
    generate_noisy_signal
)
from utils.smoothing import smooth_signal
from utils.performance_metrics import compute_convergence_time, compute_steady_state_error
from scipy.signal import resample
import warnings
from scipy.io.wavfile import WavFileWarning

warnings.simplefilter("ignore", WavFileWarning)

def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64)**2) + 1e-12))

def _load_ir(primary_path_file="python/primary_paths/primary_anechoic.wav",
             secondary_path_file="python/secondary_paths/secondary_anechoic.wav"):
    fs, p = wavfile.read(primary_path_file)
    _,  s = wavfile.read(secondary_path_file)
    p = np.asarray(p, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)
    # L2 normalization (energy)
    p = p / (np.sqrt(np.sum(p.astype(np.float64)**2)) + 1e-12)
    s = s / (np.sqrt(np.sum(s.astype(np.float64)**2)) + 1e-12)
    return fs, p, s

def _make_noise(N, fs, noise_source, noise_type, noise_wav_path):
    if (noise_source == "WAV") and noise_wav_path:
        wav_fs, wav_data = wavfile.read(noise_wav_path)
        if wav_data.ndim > 1:
            wav_data = wav_data.mean(axis=1)
        if np.issubdtype(wav_data.dtype, np.integer):
            max_i = max(1.0, float(np.iinfo(wav_data.dtype).max))
            wav = wav_data.astype(np.float32) / max_i
        else:
            wav = wav_data.astype(np.float32)
        if wav_fs != fs:
            wav = resample(wav, int(len(wav) * fs / wav_fs)).astype(np.float32)
        if len(wav) >= N:
            return wav[:N].astype(np.float32)
        reps = int(np.ceil(N / len(wav)))
        return np.tile(wav, reps)[:N].astype(np.float32)
    gens = {
        "White": generate_white_noise_len,
        "Pink": generate_pink_noise_len,
        "Brownian": generate_brownian_noise_len,
        "Violet": generate_violet_noise_len,
        "Grey": generate_grey_noise_len,
        "Blue": generate_blue_noise_len,
    }
    return gens[noise_type](N).astype(np.float32)

def simulate_once(algorithm_name, L, mu,
                  noise_source, noise_type, noise_wav_path,
                  duration, rng_seed=None):
    """
    Pure simulation: no GUI, no plotting, no audio.
    Returns a dict with metrics and small tails for listening preview.
    """
    if rng_seed is not None:
        np.random.seed(int(rng_seed))

    fs, primary_path, secondary_path = _load_ir()
    N = int(float(duration) * fs)

    noise = _make_noise(N, fs, noise_source, noise_type, noise_wav_path)
    reference_signal = np.zeros(N, dtype=np.float32)
    noisy_signal = generate_noisy_signal(reference_signal, noise)

    d_raw = np.convolve(noisy_signal, primary_path,   mode='full')[:N]
    z_raw = np.convolve(noisy_signal, secondary_path, mode='full')[:N]

    rp = _rms(d_raw)
    rs = _rms(z_raw)
    d = d_raw / rp
    z = z_raw / rs

    L = int(L); mu = float(mu)
    w0 = np.zeros(L, dtype=np.float64)

    if algorithm_name == "LMS" and LMS is not None:
        alg = LMS(L, mu, w0.copy())
        x_stream = noisy_signal
    elif algorithm_name == "NLMS" and NLMS is not None:
        alg = NLMS(L, mu, w0.copy())
        x_stream = noisy_signal
    elif algorithm_name == "FxLMS":
        alg = FxLMS(L, mu, w0.copy())
        x_stream = z
    elif algorithm_name == "FxNLMS":
        alg = FxNLMS(L, mu, w0.copy())
        x_stream = z
    else:
        raise ValueError("Unknown or unavailable algorithm")

    e = np.zeros(N, dtype=np.float64)
    y = np.zeros(N, dtype=np.float64)
    for n in range(N):
        en, yn = alg.estimate(x_stream[n], d[n])
        if not np.isfinite(en): en = 0.0
        if not np.isfinite(yn): yn = 0.0
        e[n] = np.clip(en, -10.0, 10.0)
        y[n] = np.clip(yn, -10.0, 10.0)

    e_med = smooth_signal(e, 401)
    sse = compute_steady_state_error(e_med)
    sse_db = 20 * np.log10(sse + 1e-12)
    conv_ms = compute_convergence_time(e_med, fs, sse)

    tail = slice(int(0.8 * N), N)
    in_power  = float(np.mean(d[tail]**2))
    out_power = float(np.mean(e[tail]**2))

    # keep at most 2 seconds of preview to reduce RAM
    tail_len = min(int(0.2 * N), int(2 * fs))
    d_tail_raw = (rp * d[-tail_len:]).astype(np.float32)
    e_tail_raw = (rp * e[-tail_len:]).astype(np.float32)

    return {
        "mu": mu, "L": L,
        "conv_ms": float(conv_ms) if conv_ms is not None else float("inf"),
        "sse_db": float(sse_db),
        "in_power": in_power, "out_power": out_power,
        "fs": int(fs),
        "d_mic_raw_tail": d_tail_raw,
        "e_mic_raw_tail": e_tail_raw,
    }