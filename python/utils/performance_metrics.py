import numpy as np

def compute_convergence_time(error_signal, fs, threshold_factor=0.1, min_stable_duration=0.05):
    """
    Computes the convergence time of the ANC system.
    
    Parameters:
    - error_signal (numpy array): The error signal produced by the algorithm.
    - fs (int): Sampling frequency in Hz.
    - threshold_factor (float): Percentage of max error used as threshold (default 10%).
    - min_stable_duration (float): Minimum time (seconds) the error must stay below threshold.
    
    Returns:
    - convergence_time (float): Time in seconds when the system is considered converged.
    """
    # Define a threshold for convergence detection
    threshold = threshold_factor * np.max(np.abs(error_signal))

    # Convert stable duration to samples
    stable_samples = int(min_stable_duration * fs)

    # Find when the error stays below threshold for the required duration
    for conv_idx in range(len(error_signal) - stable_samples):
        if np.all(np.abs(error_signal[conv_idx:conv_idx + stable_samples]) < threshold):
            return conv_idx / fs  # Convert index to time (seconds)

    return None  # If no convergence detected


def compute_steady_state_error(error_signal, percentage=0.2):
    """
    Computes the steady-state error of the ANC system.

    Parameters:
    - error_signal (numpy array): The error signal produced by the algorithm.
    - fs (int): Sampling frequency in Hz.
    - percentage (float): Percentage of final samples to use for SSE calculation (default 20%).

    Returns:
    - steady_state_error (float): The steady-state error in dB.
    """
    last_samples = int(percentage * len(error_signal))
    sse = np.mean(error_signal[-last_samples:] ** 2)  # Compute mean squared error
    return 10 * np.log10(sse + 1e-10)  # Convert to dB, avoid log(0)