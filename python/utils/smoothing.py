from scipy.signal import medfilt

def smooth_signal(signal, kernel_size):
    if kernel_size % 2 == 0:
        kernel_size += 1
    return medfilt(signal, kernel_size)