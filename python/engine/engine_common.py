def load_paths(target_fs=16000):
    """Load and peak-normalize the primary/secondary impulse responses."""
    import numpy as np
    from scipy.io import loadmat, wavfile

    #import h5py
    #import mat73
    
    fs = 0
    try:
        #primary_path = loadmat("python/primary_paths/primary_path.mat")["sim_imp"].flatten()#[:4000]
        #secondary_path = loadmat("python/secondary_paths/secondary_path.mat")["sim_imp"].flatten()#[:2000]
        #primary_path = h5py.File("python/primary_paths/primary_path_new.mat", "r")['sim_imp'].flatten()[:4000]
        #secondary_path = loadmat("python/secondary_paths/secondary_path_new.mat")['sim_imp'].flatten()[:2000]
        fs, primary_path = wavfile.read("python/primary_paths/primary_anechoic.wav")
        _, secondary_path = wavfile.read("python/secondary_paths/secondary_anechoic.wav")
        #primary_path = loadmat("python/primary_paths/primary_path_gh.mat")["Pz1"].flatten()
        #secondary_path = loadmat("python/secondary_paths/secondary_path_gh.mat")["S"].flatten()
    except Exception as e:
        print("Open the whole ANC-WITH-NN folder as project.")
        print(f"Error loading impulse responses: {e}")
        raise RuntimeError("Open the whole ANC-WITH-NN folder as project.")

    # Set custom fs if not loaded from impulse response
    if fs == 0:
        fs = 44100

    # Convert to float32
    primary_path = primary_path.astype(np.float32)
    secondary_path = secondary_path.astype(np.float32)

    # Peak normalization (same scale factor applied to both)
    primary_path, secondary_path = scale_paths(primary_path, secondary_path, fs)

    # Return fs and normalized paths
    return fs, primary_path, secondary_path

def scale_paths(primary_path, secondary_path, fs):
    """Scale the primary and secondary paths by the same factor (peak normalization)."""

    from utils.smoothing import smooth_fractional_octave_db
    from utils.fft_transform import compute_fft
    import numpy as np

    freqs_primary, primary_path_fft = compute_fft(primary_path, fs)
    num_fractions = 6  # 1/12-octave smoothing

    primary_path_fft_smoothed = smooth_fractional_octave_db(
        freqs_primary,
        primary_path_fft,
        num_fractions=num_fractions,
        db_kind="amplitude",
        sampling_rate=fs,
        n_samples=len(primary_path)
    )

    # Use only the meaningful frequency range for the peak
    f_min = 20
    f_max = 20000
    mask = (freqs_primary >= f_min) & (freqs_primary <= f_max)

    max_val_db = float(np.max(primary_path_fft_smoothed[mask])) + 1e-12
    
    # Convert dB peak to linear gain
    max_val_linear = 10 ** (max_val_db / 20)

    # Apply the same scale factor to both paths
    primary_path_scaled = primary_path / max_val_linear
    secondary_path_scaled = secondary_path / max_val_linear

    primary_path_scaled = primary_path_scaled.astype(np.float32, copy=False)
    secondary_path_scaled = secondary_path_scaled.astype(np.float32, copy=False)
    
    # Return scaled paths
    return primary_path_scaled, secondary_path_scaled