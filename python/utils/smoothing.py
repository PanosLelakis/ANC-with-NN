from scipy.signal import medfilt

def smooth_signal(signal, kernel_size):
    """
    Apply median filtering to smooth a signal.
    
    Parameters:
    signal (numpy array): Input signal to be smoothed.
    kernel_size (int): Size of the median filter kernel (must be an odd number).
    
    Returns:
    numpy array: Smoothed signal.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return medfilt(signal, kernel_size)