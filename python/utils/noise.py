import numpy as np

def generate_colored_noise(signal, snr, exponent):
    #Generate colored noise based on the given power law exponent.
    #Exponents:
      # White noise (0)
      # Pink noise (-1)
      # Brownian noise (-2)
      # Blue noise (1)
      # Violet noise (2)
      # Grey noise (custom)

    num_samples = len(signal)

    # Generate white noise
    noise = np.random.randn(num_samples)

    # Transform to frequency domain
    f = np.fft.rfftfreq(num_samples)
    f[0] = 1  # Avoid division by zero at f=0

    # Apply frequency weighting for the desired noise color
    noise_spectrum = np.fft.rfft(noise) * (f ** (exponent / 2.0))
    
    # Convert back to time domain
    noise = np.fft.irfft(noise_spectrum, n=num_samples)

    return scale_noise(signal, noise, snr)

def generate_white_noise(signal, snr):
    #flat spectrum
    return generate_colored_noise(signal, snr, 0)

def generate_pink_noise(signal, snr):
    #1/f spectrum
    return generate_colored_noise(signal, snr, -1)

def generate_brownian_noise(signal, snr):
    #1/f^2 spectrum
    return generate_colored_noise(signal, snr, -2)

def generate_blue_noise(signal, snr):
    #f spectrum
    return generate_colored_noise(signal, snr, 1)

def generate_violet_noise(signal, snr):
    #f^2 spectrum
    return generate_colored_noise(signal, snr, 2)

def generate_grey_noise(signal, snr):
    return generate_colored_noise(signal, snr, -0.5)  # Custom exponent for perceptual adjustment

def scale_noise(signal, noise, snr):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    scaling_factor = np.sqrt(signal_power / (10 ** (snr / 10) * noise_power))
    return signal + scaling_factor * noise