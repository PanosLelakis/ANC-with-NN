import numpy as np

def add_noise(signal, snr_db):
    noise = np.random.normal(0, 1, len(signal))
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * noise
    return signal + noise