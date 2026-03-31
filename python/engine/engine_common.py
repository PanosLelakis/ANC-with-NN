from scipy.io import loadmat, wavfile
import numpy as np
#import h5py
#import mat73

def load_paths():
    """Load and peak-normalize the primary/secondary impulse responses."""
    
    fs = 0
    try:
        primary_path = loadmat("python/primary_paths/primary_path.mat")["sim_imp"].flatten()#[:4000]
        secondary_path = loadmat("python/secondary_paths/secondary_path.mat")["sim_imp"].flatten()#[:2000]
        #primary_path = h5py.File("python/primary_paths/primary_path_new.mat", "r")['sim_imp'].flatten()[:4000]
        #secondary_path = loadmat("python/secondary_paths/secondary_path_new.mat")['sim_imp'].flatten()[:2000]
        #fs, primary_path = wavfile.read("python/primary_paths/primary_anechoic.wav")
        #_, secondary_path = wavfile.read("python/secondary_paths/secondary_anechoic.wav")
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

    from utils.smoothing import whittaker_eilers_smooth
    from utils.fft_transform import compute_fft

    # Compute scale factor based on the maximum absolute value of the primary path
    #max_val = np.percentile(np.abs(primary_path), 99.9) + 1e-12  # Add small epsilon to avoid division by zero

    _, primary_path_fft = compute_fft(primary_path, fs)
    primary_path_fft_smoothed = whittaker_eilers_smooth(primary_path_fft, lmbd=1e5)

    #_, secondary_path_fft = compute_fft(secondary_path, fs)
    #secondary_path_fft_smoothed = whittaker_eilers_smooth(secondary_path_fft, lmbd=1e5)

    max_val = float(np.max(np.abs(primary_path_fft_smoothed))) + 1e-12
    #max_val = np.percentile(np.abs(primary_path_fft_smoothed), 99) + 1e-12
    #max_val = np.percentile(primary_path, 99) + 1e-12

    #primary_path_scaled = np.fft.irfft(primary_path_fft - max_val)
    #secondary_path_scaled = np.fft.irfft(secondary_path_fft - max_val)
    
    # Scale both paths by the same factor
    max_val = 10 ** (max_val / 20)
    primary_path_scaled = primary_path / max_val
    secondary_path_scaled = secondary_path / max_val

    primary_path_scaled = primary_path_scaled.astype(np.float32, copy=False)
    secondary_path_scaled = secondary_path_scaled.astype(np.float32, copy=False)
    
    # Return scaled paths
    return primary_path_scaled, secondary_path_scaled