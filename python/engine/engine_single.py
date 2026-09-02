import numpy as np
from scipy.io import wavfile
from scipy.signal import (
    resample,
    lfilter
)
import time
from algorithms.lms import LMS
from algorithms.nlms import NLMS
from algorithms.fxlms import FxLMS
from algorithms.fxnlms import FxNLMS
from utils.noise import (
    compute_noise_power,
    generate_blue_noise_len,
    generate_brownian_noise_len,
    generate_grey_noise_len,
    generate_noisy_signal,
    generate_pink_noise_len,
    generate_violet_noise_len,
    generate_white_noise_len
)
from utils.performance_metrics import (
    compute_convergence_time,
    compute_steady_state_error,
    compute_avg_pnc_dbr,
    compute_band_attenuation,
)
from utils.smoothing import whittaker_eilers_smooth
from utils.convert_to_db import val_to_dbr
import warnings
from scipy.io.wavfile import WavFileWarning

# Ignore WAV file warnings because they mean nothing
warnings.simplefilter("ignore", WavFileWarning)

# Available adaptive algorithms
ALGORITHM_CLASSES = {
    "LMS": LMS,
    "NLMS": NLMS,
    "FxLMS": FxLMS,
    "FxNLMS": FxNLMS
}

def make_noise(N, fs, noise_source, noise_type, noise_wav_path):
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

        # Generate selected colored noise
        return gens[noise_type](N).astype(np.float32)

def compute_metrics(start_time, error_signal, noisy_signal, fs, N, anc_off_signal):
    # Compute total execution time
    exec_time = time.time() - start_time

    # Convert error signal to dBr
    ref = np.sqrt(np.mean(noisy_signal ** 2) + 1e-12)
    error_dbr = val_to_dbr(error_signal, ref)

    # Smooth error signal dB curve
    error_dbr_smoothed = whittaker_eilers_smooth(error_dbr, lmbd=1e13)

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

def run_anc(
    algorithm_name,
    L,
    mu,
    noise_source,
    noise_type,
    noise_wav_path,
    duration,
    progress_callback=None,
    nn_checkpoint_path=None,
    nn_backend="pytorch",
    preloaded_noise=None,
    paths=None
):
    """Single simulation engine"""

    # Start measuring execution time
    start_time = time.time()

    # Use empty progress callback when none was provided
    if progress_callback is None:
        # Ignore progress updates
        progress_callback = lambda _percentage: None

    # Run Neural Network controller
    if algorithm_name == "Neural Network":
        # Run Neural Network simulation
        return run_neural_anc_single(
            nn_checkpoint_path=nn_checkpoint_path,
            nn_backend=nn_backend,
            noise_source=noise_source,
            noise_type=noise_type,
            noise_wav_path=noise_wav_path,
            duration=duration,
            start_time=start_time,
            progress_callback=progress_callback,
            preloaded_noise=preloaded_noise,
            paths=paths
        )

    # Read provided paths
    fs, primary_path, secondary_path = paths

    # Load constants and time vector
    N = int(duration * fs)
    t = np.arange(N) / fs
    L = int(L)
    mu = float(mu)
    initial_weights = np.zeros(L) # Zero inital weights (for simplicity)
    init_weights = initial_weights.copy()

    # Generate reference signal (zero)
    reference_signal = np.zeros(N, dtype=np.float32)

    # Use preloaded noise when available
    if preloaded_noise is None:
        # Create or load noise
        noise = make_noise(
            N,
            fs,
            noise_source,
            noise_type,
            noise_wav_path
        )
    else:
        # Read provided noise
        noise = np.asarray(preloaded_noise, dtype=np.float32)

    # Build noisy signal
    noisy_signal = generate_noisy_signal(reference_signal, noise)

    # Create selected adaptive algorithm
    algorithm = ALGORITHM_CLASSES[algorithm_name](L, mu, initial_weights)

    # Convolve input with paths
    primary_output_raw = np.convolve(noisy_signal, primary_path, mode="full")[:N].astype(np.float32, copy=False)
    secondary_output_raw = np.convolve(noisy_signal, secondary_path, mode="full")[:N].astype(np.float32, copy=False)

    # Streams used by adaptive algorithm
    primary_stream = primary_output_raw # d[n]

    # LMS/NLMS use the raw noise as y[n]
    if algorithm_name in ("LMS", "NLMS"):
        secondary_stream = noisy_signal
    else:
        secondary_stream = secondary_output_raw

    # Initialize produced signals
    error_signal = np.zeros(N, dtype=np.float32)

    MAX_WEIGHT_NORM = 1e4
    MAX_INSTANT_PEAK_RATIO = 25.0
    MAX_TAIL_POWER_RATIO = 10.0
    MAX_TAIL_PEAK_RATIO = 10.0

    baseline_peak = float(np.max(np.abs(primary_output_raw)) + 1e-12)

    divergence = False
    progress_step = max(1, N // 100)
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

        if (not np.isfinite(en)) or (abs(float(en)) > MAX_INSTANT_PEAK_RATIO * baseline_peak):
            divergence = True
            en = 0.0 if not np.isfinite(en) else float(en)
            error_signal[n:] = en
            break

        error_signal[n] = float(en)
        
        # Read adaptive weights
        w = np.asarray(algorithm.w,dtype=float)

        # Check unstable weights
        if (
            not np.all(np.isfinite(w))
            or np.linalg.norm(w) > MAX_WEIGHT_NORM
        ):
            # Mark divergence
            divergence = True

            # Complete remaining signal
            error_signal[n:] = error_signal[n]

            # Stop simulation
            break
        
        # Update simulation progress
        if (n % progress_step) == 0:
            # Send percentage
            progress_callback(int((n / N) * 100))

    # Complete simulation progress
    progress_callback(100)

    after_signal_raw = np.clip(error_signal, -1e3, 1e3)
    before_signal_raw = primary_output_raw

    # Compute performance metrics
    exec_time, conv_ms, sse_db, in_power, out_power = compute_metrics(
        start_time, error_signal, noisy_signal, fs, N, before_signal_raw)
    
    # Compute passive noise cancelling level.
    avg_pnc_dbr = compute_avg_pnc_dbr(before_signal_raw, noisy_signal)

    # Compute band attenuation values.
    band_attenuation = compute_band_attenuation(before_signal_raw, after_signal_raw, fs)

    tail = slice(int(0.8 * N), N)
    tail_error_peak = float(np.max(np.abs(error_signal[tail])) + 1e-12)
    tail_before_peak = float(np.max(np.abs(before_signal_raw[tail])) + 1e-12)

    # Check tail power increase
    bad_tail_power = (out_power > MAX_TAIL_POWER_RATIO * max(float(in_power), 1e-12))

    bad_tail_peak = (tail_error_peak > MAX_TAIL_PEAK_RATIO * tail_before_peak)

    # Mark unstable final result
    if bad_tail_power or bad_tail_peak:
        divergence = True

    # Build complete simulation result
    result = {
        # Simulation settings
        "algorithm": algorithm_name,
        "L": int(L),
        "mu": float(mu),

        # Noise settings
        "source": noise_source,
        "noise_label": noise_type,
        "wav_path": (
            noise_wav_path
            if noise_source == "WAV"
            else ""
        ),

        # Input and output signals
        "reference": reference_signal,
        "noisy": noisy_signal,
        "error": error_signal,
        "t": t,

        # Sampling rate
        "fs": int(fs),

        # Performance metrics
        "exec_time": float(exec_time),
        "conv_ms": conv_ms,
        "sse_db": sse_db,
        "in_power": float(in_power),
        "out_power": float(out_power),
        "avg_pnc_dbr": float(avg_pnc_dbr),
        "band_attenuation": band_attenuation,

        # Filter weights
        "w0": init_weights,
        "wf": algorithm.w,

        # ANC paths
        "pir": primary_path,
        "sir": secondary_path,

        # Signals after paths
        "d": primary_stream,
        "z": secondary_stream,

        # Signals used for playback and saving
        "before_raw": before_signal_raw,
        "after_raw": after_signal_raw,

        # Simulation status
        "divergence": bool(divergence)
    }

    # Return complete result
    return result

def run_neural_anc_single(
    nn_checkpoint_path,
    nn_backend,
    noise_source,
    noise_type,
    noise_wav_path,
    duration,
    start_time,
    progress_callback,
    preloaded_noise=None,
    paths=None
):
    # Run trained neural network controller for Single Run

    import torch
    from neural.inference import load_inference_model, run_anc_inference
    from neural.train import load_paths_as_tensors
    from neural.preprocess import resample_if_needed

    # Load trained model
    model, config, device = load_inference_model(nn_checkpoint_path, nn_backend)

    # Read sampling rate
    fs = int(config.get("target_fs", 16000))

    frame_ms = int(  # Read checkpoint frame duration
        config.get(
            "frame_ms",
            32
        )
    )

    hop_ms = int(  # Read checkpoint frame shift
        config.get(
            "hop_ms",
            10
        )
    )

    # Build signal length
    duration = float(duration)
    N = int(duration * fs)
    t = np.arange(N) / fs

    # Reference signal is zero
    reference_signal = np.zeros(N, dtype=np.float32)

    # Generate noise for Single Run
    if preloaded_noise is None:
        noise = make_noise(
            N=N,
            fs=fs,
            noise_source=noise_source,
            noise_type=noise_type,
            noise_wav_path=noise_wav_path
        ).astype(np.float32)

    # Use common Multi Run noise
    else:
        # Read original path sampling rate
        source_fs = int(paths[0])

        # Resample common noise to NN sampling rate
        noise = resample_if_needed(
            np.asarray(
                preloaded_noise,
                dtype=np.float32
            ),
            source_fs,
            fs
        )

        # Match requested duration
        if len(noise) >= N:
            noise = noise[:N]
        else:
            repetitions = int(
                np.ceil(
                    N / max(1, len(noise))
                )
            )

            noise = np.tile(
                noise,
                repetitions
            )[:N]

        # Normalize after resampling
        noise = noise.astype(np.float32)

        noise /= np.sqrt(
            np.mean(noise ** 2)
            + 1e-12
        )

    # Input signal for neural controller
    noisy_signal = generate_noisy_signal(reference_signal, noise).astype(np.float32)

    # Convert input to torch tensor
    x = torch.from_numpy(noisy_signal).float().unsqueeze(0).to(device)

    # Convert startup paths to tensors
    primary_path, secondary_path = load_paths_as_tensors(
        paths=paths,
        device=device,
        target_fs=fs
    )

    # Run neural inference
    signals = run_anc_inference(
        model=model,
        x=x,
        primary_path=primary_path,
        secondary_path=secondary_path,
        backend=nn_backend,
        config=config
    )

    # Build zero input
    zero_x = torch.zeros_like(
        x
    )

    # Run zero-input inference
    zero_signals = run_anc_inference(
        model=model,
        x=zero_x,
        primary_path=primary_path,
        secondary_path=secondary_path,
        backend=nn_backend,
        config=config
    )

    # Read ANC OFF signal and secondary-path outputs
    d = signals["d"]
    a = signals["a"]
    a0 = zero_signals["a"]

    # Remove the zero-input component.
    # The secondary path is linear, so S(y - y0) = S(y) - S(y0).
    corrected_a = (
        a
        - a0
    )

    # Compute zero-input corrected residual
    corrected_e = (
        d
        - corrected_a
    )

    # Convert tensors to numpy
    primary_output_raw = d.squeeze(0).detach().cpu().numpy().astype(np.float32)
    # Use zero-bias corrected secondary output
    secondary_output_raw = (
        corrected_a.squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # Use zero-bias corrected residual
    error_signal = (
        corrected_e.squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # Match metadata length
    N = len(error_signal)
    t = t[:N]
    noisy_signal = noisy_signal[:N]
    reference_signal = reference_signal[:N]

    # Complete Neural Network progress
    progress_callback(100)

    # Compute metrics
    exec_time, _, sse_db, in_power, out_power = compute_metrics(
        start_time=start_time,
        error_signal=error_signal,
        noisy_signal=noisy_signal,
        fs=fs,
        N=N,
        anc_off_signal=primary_output_raw
    )

    conv_ms = None

    # Compute additional metrics
    avg_pnc_dbr = compute_avg_pnc_dbr(primary_output_raw, noisy_signal)
    band_attenuation = compute_band_attenuation(primary_output_raw, error_signal, fs)

    # NN does not have adaptive filter weights
    initial_weights = np.array([], dtype=np.float32)
    final_weights = np.array([], dtype=np.float32)

    # Convert paths to numpy
    primary_ir = primary_path.detach().cpu().numpy().astype(np.float32)
    secondary_ir = secondary_path.detach().cpu().numpy().astype(np.float32)

    # Divergence flag
    divergence = False

    # Build complete Neural Network result
    result = {
        # Simulation settings
        "algorithm": "Neural Network",
        "L": 0,
        "mu": 0.0,
        "nn_backend": str(nn_backend),

        # Noise settings
        "source": noise_source,
        "noise_label": noise_type,
        "wav_path": (
            noise_wav_path
            if noise_source == "WAV"
            else ""
        ),

        # Input and output signals
        "reference": reference_signal,
        "noisy": noisy_signal,
        "error": error_signal,
        "t": t,

        # Sampling rate
        "fs": int(fs),

        # Performance metrics
        "exec_time": float(exec_time),
        "conv_ms": conv_ms,
        "sse_db": sse_db,
        "in_power": float(in_power),
        "out_power": float(out_power),
        "avg_pnc_dbr": float(avg_pnc_dbr),
        "band_attenuation": band_attenuation,

        # Neural Network has no adaptive weights
        "w0": initial_weights,
        "wf": final_weights,

        # ANC paths
        "pir": primary_ir,
        "sir": secondary_ir,

        # Signals after paths
        "d": primary_output_raw,
        "z": secondary_output_raw,

        # Signals used for playback and saving
        "before_raw": primary_output_raw,
        "after_raw": error_signal,

        # Simulation status
        "divergence": bool(divergence)
    }

    # Return complete result
    return result