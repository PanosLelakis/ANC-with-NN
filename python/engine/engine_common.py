from scipy.io import loadmat
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
    #_, pp_fft = compute_fft(primary_path, fs)
    #_, sp_fft = compute_fft(secondary_path, fs)
    #scale = float(np.max(np.abs(whittaker_eilers_smooth(pp_fft, lmbd=1e5)))) + 1e-12 # Max abs value of primary path
    #scale = 1
    #pp_fft /= scale
    #sp_fft /= scale
    #primary_path = np.real(np.fft.ifft(pp_fft)).astype(np.float32, copy=False)
    #secondary_path = np.real(np.fft.ifft(sp_fft)).astype(np.float32, copy=False)
    #primary_path /= scale
    #secondary_path /= scale

    # Return fs and normalized paths
    return fs, primary_path, secondary_path