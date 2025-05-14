import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import medfilt
import sounddevice as sd

class FxLMS:
    def __init__(self, L, mu, secondary_path_impulse_response):
        self.L = L
        self.mu = mu
        self.w = np.zeros(self.L)  # Adaptive filter weights
        #self.w = np.random.randn(self.L)
        self.u = np.zeros(self.L)  # Input buffer
        self.secondary_path_impulse_response = secondary_path_impulse_response

    def predict(self, x):
        self.u[1:] = self.u[:-1]  # Shift buffer
        self.u[0] = x  # Insert new input
        return np.dot(self.w, self.u)  # Predicted output
    
    def adapt(self, error, filtered_x):
        self.w = self.w - self.mu * error * filtered_x  # Update weights based on the secondary-path-filtered reference

    def estimate(self, x, d):
        y = self.predict(x)  # Get predicted output from the adaptive filter
        filtered_x = x
        #filtered_x = np.convolve(x, self.secondary_path_impulse_response, mode='full')[-self.L:]  # Filter the reference signal through the secondary path
        e = d - y  # Compute the error signal
        self.adapt(e, filtered_x)  # Adapt filter weights
        return e, y
    
# Load primary and secondary path
primary_path_data = loadmat("python/primary_path.mat")
secondary_path_data = loadmat("python/secondary_path.mat")

# Extract the impulse responses
primary_impulse_response = primary_path_data['sim_imp'].flatten()[:2000]  # flatten to convert to 1D array
secondary_impulse_response = secondary_path_data['sim_imp'].flatten()[:1000]

# Generate input signal
fs = 44100  # Sampling frequency
duration = 10 # Total time of simulation in seconds
t = np.arange(0, duration, 1/fs)
freq = 500  # Frequency in Hz
sinusoid = np.sin(2 * np.pi * freq * t)
max_val = np.max(abs(sinusoid))  # Normalize using max amplitude of original signal

def add_noise(signal, snr_db):
    noise = np.random.normal(0, 1, len(signal))
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * noise
    return signal + noise

# Add noise to create a noisy input
noisy_signal = sinusoid
#snr_levels = [-10, 0, 10]
snr_levels = [0]
for snr in snr_levels:
    noisy_signal = add_noise(noisy_signal, snr)

# Initialize FxLMS
L = 2 ** 9 # Filter length
mu = 1 * 10 ** (-4) # Learning rate
fxlms = FxLMS(L, mu, secondary_impulse_response)

# Filter the signal
error_signal = np.zeros(len(noisy_signal))
filtered_signal = np.zeros(len(noisy_signal))
primary_output = np.convolve(noisy_signal, primary_impulse_response, mode='full')[:len(noisy_signal)]
secondary_output = np.convolve(primary_output, secondary_impulse_response, mode='full')[:len(noisy_signal)]
for n in tqdm(range(len(noisy_signal))):
    error_signal[n], filtered_signal[n] = fxlms.estimate(secondary_output[n], noisy_signal[n])

# Performance Evaluation
def evaluate_performance(error_signal, fs, duration):
    steady_state_start = int(0.8 * duration * fs)  # Last 2 sec
    steady_state_error = 20 * np.log10(np.mean(np.abs(error_signal[steady_state_start:])) + 1e-8)

    # Convergence speed: Find the first index where error falls below 10% of initial
    initial_error = np.mean(np.abs(error_signal[:fs]))  # First second avg error
    threshold = 0.1 * initial_error
    convergence_idx = np.where(np.abs(error_signal) < threshold)[0]

    # Ensure convergence index exists
    if len(convergence_idx) > 0:
        convergence_time = convergence_idx[0] / fs
    else:
        convergence_time = float('inf')  # Indicate no convergence

    return steady_state_error, convergence_time

steady_state_error, convergence_time = evaluate_performance(error_signal, fs, duration)
print(f"Steady-State Error: {steady_state_error:.2f} dB")
print(f"Convergence Time: {convergence_time:.2f} sec")

# Convert to dBFS
def convert_to_dbfs(signal):
    return 20 * np.log10((np.abs(signal) / (max_val + 1e-8)) + 1e-8)

sinusoid_dbfs = convert_to_dbfs(sinusoid)
noisy_signal_dbfs = convert_to_dbfs(noisy_signal)
filtered_signal_dbfs = convert_to_dbfs(filtered_signal)
error_signal_dbfs = convert_to_dbfs(error_signal)

# Smoothing methods (Median Filter)
sinusoid_dbfs_median = medfilt(sinusoid_dbfs, kernel_size=401)
noisy_signal_dbfs_median = medfilt(noisy_signal_dbfs, kernel_size=401)
filtered_signal_dbfs_median = medfilt(filtered_signal_dbfs, kernel_size=401)
error_signal_dbfs_median = medfilt(error_signal_dbfs, kernel_size=401)

# Plot results
plt.figure()

plt.subplot(3,1,1)
plt.plot(t, sinusoid_dbfs_median, label="Original Signal", alpha=0.5)
plt.plot(t, noisy_signal_dbfs_median, label="Noisy Signal", alpha=0.5)
plt.plot(t, filtered_signal_dbfs_median, label="Filtered Signal", alpha=0.5)
plt.plot(t, error_signal_dbfs_median, label="Error Signal", alpha=0.5)
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (dBFS) - median")
#plt.title("FxLMS Filtering")
plt.grid()

plt.subplot(3,1,2)
plt.plot(t, sinusoid_dbfs, label="Original Signal", alpha=0.5)
plt.plot(t, noisy_signal_dbfs, label="Noisy Signal", alpha=0.5)
plt.plot(t, filtered_signal_dbfs, label="Filtered Signal", alpha=0.5)
plt.plot(t, error_signal_dbfs, label="Error Signal", alpha=0.5)
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (dBFS)")
#plt.title("FxLMS Filtering")
plt.grid()

plt.subplot(3,1,3)
plt.plot(t, sinusoid, label="Original Signal", alpha=0.5)
plt.plot(t, noisy_signal, label="Noisy Signal", alpha=0.5)
plt.plot(t, filtered_signal, label="Filtered Signal", alpha=0.5)
plt.plot(t, error_signal, label="Error Signal", alpha=0.5)
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (Volt)")
#plt.title("FxLMS Filtering")
plt.grid()

plt.show()

# Listen to results
# Function to play audio
def play_audio(signal, fs, message):
    print(f"Playing {message}...")
    sd.play(signal, samplerate=fs)
    sd.wait()

# Play original sinusoid
#play_audio(sinusoid, fs, "Original Sinusoid")

# Play noisy signal
play_audio(noisy_signal[:int(len(noisy_signal)/3)], fs, "Noisy Signal")

# Play filtered signal
play_audio(filtered_signal[:int(len(filtered_signal)/2)], fs, "Filtered Signal")