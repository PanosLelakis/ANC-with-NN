import numpy as np

def compute_convergence_time(error_signal, fs, sse, threshold_factor=1.1, min_stable_duration=0.01, acceptance_ratio=0.99):
    threshold = threshold_factor * sse
    samples = int(min_stable_duration * fs)
    required_below = int(acceptance_ratio * samples)

    for conv_idx in range(len(error_signal) - samples):
        window = np.sqrt(error_signal[conv_idx:conv_idx + samples] ** 2)
        if np.sum(window < threshold) >= required_below:
            return 1000*(conv_idx / fs) # multiply by 1000 to convert from sec to ms

    return None

def compute_steady_state_error(error_signal, percentage=0.2):
    last_samples = int(percentage * len(error_signal))
    rmse = np.sqrt(np.mean(error_signal[-last_samples:] ** 2))  # RMSE
    return rmse