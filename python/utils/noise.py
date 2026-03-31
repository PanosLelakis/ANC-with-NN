import numpy as np

def _colored_noise_len(num_samples, exponent, amplitude=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    # white Gaussian
    noise = rng.standard_normal(num_samples)

    # frequency shaping
    f = np.fft.rfftfreq(num_samples)
    f[0] = 1.0
    noise_spectrum = np.fft.rfft(noise) * (f ** (exponent / 2.0))
    colored = np.fft.irfft(noise_spectrum, n=num_samples)

    # Remove DC drift
    colored -= np.mean(colored)

    # unit power then scale
    colored /= np.sqrt(np.mean(colored ** 2) + 1e-12)
    return amplitude * colored

def generate_white_noise_len(n, amplitude=1.0):   return _colored_noise_len(n,  0, amplitude)
def generate_pink_noise_len(n, amplitude=1.0):    return _colored_noise_len(n, -1, amplitude)
def generate_brownian_noise_len(n, amplitude=1.0):return _colored_noise_len(n, -2, amplitude)
def generate_blue_noise_len(n, amplitude=1.0):    return _colored_noise_len(n,  1, amplitude)
def generate_violet_noise_len(n, amplitude=1.0):  return _colored_noise_len(n,  2, amplitude)
def generate_grey_noise_len(n, amplitude=1.0):    return _colored_noise_len(n, -0.5, amplitude)

def generate_noisy_signal(signal, noise):
    return signal + noise

def compute_noise_power(noise):
    """
    Compute the power of a noise signal.
    Power = sum(noise^2) / N
    """
    pwr = np.asarray(noise, dtype=np.float64)
    pwr = np.nan_to_num(pwr, nan=0.0, posinf=0.0, neginf=0.0)
    pwr = np.clip(pwr, -1e6, 1e6)
    pwr = float(np.sum(pwr ** 2) / len(pwr))
    return pwr