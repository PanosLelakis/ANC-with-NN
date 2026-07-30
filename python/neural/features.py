import torch
import torch.nn.functional as F

# STFT constants
FRAME = 32 # msec (20 msec original)
HOP = 10 # = frame - overlap = 10 msec

def ms_to_samples(ms, fs):
    # Convert ms to samples
    return int(round(float(ms) * float(fs) / 1000.0))

def get_stft_params(fs):
    # Build STFT parameters
    win_length = ms_to_samples(FRAME, fs)
    hop_length = ms_to_samples(HOP, fs)
    n_fft = win_length
    
    # Return parameters
    return {
        "n_fft": n_fft,
        "win_length": win_length,
        "hop_length": hop_length
    }

def make_window(win_length, device):
    # Create Hann window
    #return torch.hann_window(win_length, periodic=True, device=device)
    # Create rectangular window
    return torch.ones(win_length, device=device)

def frame_signal(x, frame_length, hop_length):
    # Split signal into overlapping frames
    frame_length = int(frame_length)
    hop_length = int(hop_length)

    return x.unfold(dimension=-1, size=frame_length, step=hop_length)

def reference_to_frames(x, fs):
    # Split reference into frames
    # input: x shape = [batch, samples]
    # output: frames shape = [batch, num_frames, 512]
    
    # Get STFT parameters
    params = get_stft_params(fs)

    # Return framed signal
    return frame_signal(
        x = x,
        frame_length = params["win_length"],
        hop_length = params["hop_length"]
    )

def signal_to_complex_stft(x, fs):
    # Convert time signal to complex STFT
    
    # Get STFT parameters
    params = get_stft_params(fs)

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
    window = make_window(params["win_length"], device=x.device)

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

def signal_to_stft_channels(x, fs):
    # Convert time signal to STFT channels
    X = signal_to_complex_stft(x, fs)
    
    # Return STFT channels
    return complex_stft_to_channels(X)

def stft_channels_to_signal(Y_channels, length, fs):
    # Convert STFT channels to time signal
    
    # Merge channels to complex STFT
    Y = channels_to_complex_stft(Y_channels)
    
    # Get STFT parameters
    params = get_stft_params(fs)

    # Create window
    window = make_window(params["win_length"], device=Y_channels.device)
    
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