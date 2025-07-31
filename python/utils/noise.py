import numpy as np

def generate_colored_noise(signal, snr, exponent):
    """
    Generate colored noise based on power law exponent and return both
    the noisy signal and the noise power.

    Exponents:
      White noise (0)
      Pink noise (-1)
      Brownian noise (-2)
      Blue noise (1)
      Violet noise (2)
      Grey noise (custom)
    """
    num_samples = len(signal)

    # Generate random noise values
    noise = np.random.randn(num_samples)

    # Transform to frequency domain
    f = np.fft.rfftfreq(num_samples)
    f[0] = 1  # avoid division by zero

    noise_spectrum = np.fft.rfft(noise) * (f ** (exponent / 2.0))
    noise = np.fft.irfft(noise_spectrum, n=num_samples)

    # Normalize noise power to 1
    #noise /= np.sqrt(np.mean(noise ** 2))

    # Scale noise to match SNR
    scaled_noise = scale_noise(signal, noise, snr)

    return scaled_noise

def generate_white_noise(signal, snr):
    return generate_colored_noise(signal, snr, 0)

def generate_pink_noise(signal, snr):
    return generate_colored_noise(signal, snr, -1)

def generate_brownian_noise(signal, snr):
    return generate_colored_noise(signal, snr, -2)

def generate_blue_noise(signal, snr):
    return generate_colored_noise(signal, snr, 1)

def generate_violet_noise(signal, snr):
    return generate_colored_noise(signal, snr, 2)

def generate_grey_noise(signal, snr):
    return generate_colored_noise(signal, snr, -0.5)  # Custom exponent

def scale_noise(signal, noise, snr):
    """
    Scale noise to achieve the desired SNR.
    If signal is silence (power = 0), return normalized noise.
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if signal_power == 0:
        # If the signal is silent, just return normalized noise
        return noise
    else:
        scaling_factor = np.sqrt(signal_power / (10 ** (snr / 10) * noise_power))
        return scaling_factor * noise

def generate_noisy_signal(signal, noise):
    return signal + noise

def compute_noise_power(noise):
    """
    Compute the power of a noise signal.
    Power = sum(noise^2) / N
    """
    return np.sum(noise ** 2) / len(noise)