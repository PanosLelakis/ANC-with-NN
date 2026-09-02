import torch
import torch.nn.functional as F

# STFT constants
FRAME = 32 # msec (20 msec original)
HOP = 10 # = frame - overlap = 10 msec

# Default STFT window
DEFAULT_WINDOW_TYPE = "hamming"

def ms_to_samples(ms, fs):
    # Convert ms to samples
    return int(round(float(ms) * float(fs) / 1000.0))

def get_stft_params(  # Build STFT settings
    fs,  # Sampling rate
    frame_ms=FRAME,  # Frame duration
    hop_ms=HOP  # Frame shift
):
    win_length = ms_to_samples(frame_ms, fs)  # Convert frame duration to samples
    hop_length = ms_to_samples(hop_ms, fs)  # Convert frame shift to samples
    n_fft = win_length  # Use one FFT point per frame sample

    return {  # Return STFT settings
        "n_fft": n_fft,  # FFT size
        "win_length": win_length,  # Window size
        "hop_length": hop_length  # Frame shift
    }

def make_window(
    win_length,
    device,
    window_type=DEFAULT_WINDOW_TYPE
):
    # Normalize window type
    window_type = str(
        window_type
    ).lower()

    # Create rectangular window
    if window_type == "rectangular":
        return torch.ones(
            win_length,
            device=device
        )

    # Create Hamming window
    if window_type == "hamming":
        return torch.hamming_window(
            win_length,
            periodic=True,
            device=device
        )

    # Reject unknown window
    raise ValueError(
        f"Unknown STFT window: "
        f"{window_type}"
    )

def signal_to_complex_stft(  # Convert waveform to complex STFT
    x,  # Input waveform
    fs,  # Sampling rate
    window_type=DEFAULT_WINDOW_TYPE,  # STFT window
    frame_ms=FRAME,  # Frame duration
    hop_ms=HOP  # Frame shift
):
    # Convert time signal to complex STFT
    
    # Get STFT parameters
    params = get_stft_params(  # Read model-specific STFT settings
        fs,  # Sampling rate
        frame_ms=frame_ms,  # Frame duration
        hop_ms=hop_ms  # Frame shift
    )

    # Read signal length
    signal_length = x.shape[-1]

    # Read frame settings
    frame_length = params["win_length"]
    hop_length = params["hop_length"]

    # Compute required right padding
    if signal_length < frame_length:
        padding = frame_length - signal_length
    else:
        remainder = (signal_length - frame_length) % hop_length
        padding = (hop_length - remainder) % hop_length

    # Pad signal to complete the final STFT frame
    if padding > 0:
        x = F.pad(x, (0, padding))

    # Create window
    window = make_window(
        params["win_length"],
        device=x.device,
        window_type=window_type
    )

    # Compute STFT
    X = torch.stft(
        x,
        n_fft = params["n_fft"],
        window = window,
        hop_length = params["hop_length"],
        win_length = params["win_length"],
        center = False,
        normalized = False,
        return_complex = True
    )

    # Return complex STFT
    return X

def complex_stft_to_channels(X):
    # Split complex STFT to real and imaginary channels

    # Store real and imaginary parts
    real = X.real
    imag = X.imag

    # Return real and imaginary channels
    return torch.stack([real, imag], dim=1)

def channels_to_complex_stft(X_channels):
    # Merge real and imaginary channels to complex STFT
    real = X_channels[:, 0, :, :]
    imag = X_channels[:, 1, :, :]

    # Return complex STFT
    return torch.complex(real, imag)

def signal_to_stft_channels(  # Convert waveform to real-imaginary channels
    x,  # Input waveform
    fs,  # Sampling rate
    window_type=DEFAULT_WINDOW_TYPE,  # STFT window
    frame_ms=FRAME,  # Frame duration
    hop_ms=HOP  # Frame shift
):
    X = signal_to_complex_stft(  # Compute complex STFT
        x,  # Input waveform
        fs,  # Sampling rate
        window_type=window_type,  # Selected window
        frame_ms=frame_ms,  # Selected frame duration
        hop_ms=hop_ms  # Selected frame shift
    )

    return complex_stft_to_channels(X)  # Split real and imaginary parts

def stft_channels_to_signal(  # Convert real-imaginary channels to waveform
    Y_channels,  # Predicted STFT channels
    length,  # Requested waveform length
    fs,  # Sampling rate
    window_type=DEFAULT_WINDOW_TYPE,  # STFT window
    frame_ms=FRAME,  # Frame duration
    hop_ms=HOP  # Frame shift
):
    # Convert STFT channels to time signal
    
    # Merge channels to complex STFT
    Y = channels_to_complex_stft(Y_channels)
    
    # Get STFT parameters
    params = get_stft_params(  # Read model-specific STFT settings
        fs,  # Sampling rate
        frame_ms=frame_ms,  # Frame duration
        hop_ms=hop_ms  # Frame shift
    )

    # Create window
    window = make_window(
        params["win_length"],
        device=Y_channels.device,
        window_type=window_type
    )

    y = torch.istft(
        Y,
        n_fft = params["n_fft"],
        window = window,
        hop_length = params["hop_length"],
        win_length = params["win_length"],
        center = False,
        normalized = False,
        return_complex = False,
        length = int(length)
    )

    # Return time signal
    return y

def apply_frame_delay(X_channels, delay_m):
    # Shift input frames
    # Add zero frames
    # Keep same shape

    # Convert delay to int
    delay_m = int(delay_m)

    # Return original if no delay
    if delay_m <= 0:
        return X_channels

    # Get number of frames
    frames = X_channels.shape[3]

    # Return zeros if delay exceeds number of frames
    if delay_m >= frames:
        return torch.zeros_like(X_channels)
    
    # Create zeros for delay
    zeros = torch.zeros(
        X_channels.shape[0], # batch
        X_channels.shape[1], # channels
        X_channels.shape[2], # freq bins
        delay_m, # delay frames
        dtype=X_channels.dtype,
        device=X_channels.device
    )

    # Keep frames that are not delayed
    kept_frames = X_channels[..., :frames - delay_m]

    # Add zero frames
    shifted = torch.cat(
        [zeros, kept_frames],
        dim=-1
    )

    # Return shifted frames
    return shifted

def compute_frequency_balanced_residual_loss(  # Compute frequency-balanced residual loss
    desired_signal,  # Desired microphone signal
    residual_signal,  # ANC residual signal
    fs,  # Sampling rate
    valid_mask=None,  # Valid training items
    power_floor_ratio=0.01,  # Frequency normalization floor
    window_type=DEFAULT_WINDOW_TYPE,  # STFT window
    frame_ms=FRAME,  # Frame duration
    hop_ms=HOP  # Frame shift
):
    # Convert desired signal to STFT
    D = signal_to_complex_stft(
        desired_signal,
        fs,
        window_type=window_type,
        frame_ms=frame_ms,  # Use model frame duration
        hop_ms=hop_ms  # Use model frame shift
    )

    # Convert residual signal to STFT
    E = signal_to_complex_stft(
        residual_signal,
        fs,
        window_type=window_type,
        frame_ms=frame_ms,  # Use model frame duration
        hop_ms=hop_ms  # Use model frame shift
    )

    # Match frame counts
    frames = min(
        D.shape[-1],
        E.shape[-1]
    )

    D = D[
        ...,
        :frames
    ]

    E = E[
        ...,
        :frames
    ]

    # Compute desired power per frequency
    desired_power = torch.mean(
        torch.abs(D) ** 2,
        dim=-1
    ).detach()

    # Compute residual power per frequency
    residual_power = torch.mean(
        torch.abs(E) ** 2,
        dim=-1
    )

    # Find maximum desired power per item
    maximum_desired_power = torch.max(
        desired_power,
        dim=1,
        keepdim=True
    ).values

    # Compute minimum normalization power
    power_floor = (
        maximum_desired_power
        * float(power_floor_ratio)
    )

    power_floor = torch.clamp(
        power_floor,
        min=1e-12
    )

    # Apply power floor
    normalization_power = torch.maximum(
        desired_power,
        power_floor
    )

    # Compute normalized residual power
    normalized_residual_power = (
        residual_power
        / normalization_power
    )

    # Give equal weight to frequency bins
    item_loss = torch.mean(
        normalized_residual_power,
        dim=1
    )

    # Use all items when no mask is provided
    if valid_mask is None:
        return torch.mean(
            item_loss
        )

    # Keep only valid non-silence items
    valid_mask = valid_mask.bool()

    if torch.any(
        valid_mask
    ):
        return torch.mean(
            item_loss[
                valid_mask
            ]
        )

    # Return differentiable zero
    return (
        residual_power.mean()
        * 0.0
    )