import tkinter as tk
from tkinter import ttk
import threading
import time
from utils.audio_utils import play_audio
from main import run_anc
from utils.plot import plot_results

# Create the root window first
root = tk.Tk()
root.title("Adaptive Filtering")

# Global variables
noisy_audio = None
filtered_audio = None
is_playing = False
progress_var = tk.DoubleVar(value=0.0)
start_time = None  
total_time = 0  # Store total execution time
convergence_speed = 0  # Store convergence time
steady_state_error = 0  # Store steady-state error

def start_algorithm():
    global start_time
    disable_buttons()
    start_time = time.time()

    # Get user input
    algorithm = algo_var.get()
    mu = mu_entry.get()
    L = L_entry.get()
    snr = snr_entry.get()
    noise_type = noise_var.get()

    try:
        mu = float(mu)
        L = int(L)
        snr = float(snr)
    except ValueError:
        status_label.config(text="Error: Invalid input!", fg="red")
        enable_buttons()
        return

    # Run ANC in separate thread to avoid blocking GUI
    threading.Thread(target=run_anc, args=(algorithm, L, mu, snr, noise_type, update_progress, on_anc_complete), daemon=True).start()

def disable_buttons():
    # Disable buttons during processing
    start_button.config(state=tk.DISABLED)
    play_noisy_btn.config(state=tk.DISABLED)
    play_filtered_btn.config(state=tk.DISABLED)
    graph_button.config(state=tk.DISABLED)

def enable_buttons():
    # Enable buttons after processing
    start_button.config(state=tk.NORMAL)
    graph_button.config(state=tk.NORMAL)

def update_progress(progress):
    # Update progress bar and estimated time
    root.after(0, lambda: progress_var.set(progress))  # Ensure thread-safe update
    elapsed_time = time.time() - start_time
    estimated_time = (elapsed_time / progress) * (100 - progress) if progress > 0 else 0
    minutes, seconds = divmod(int(estimated_time), 60)
    root.after(0, lambda: time_label.config(text=f"Time remaining: {minutes:02}:{seconds:02}"))

def on_anc_complete(reference_signal, noisy_signal, filtered_signal, error_signal, t, fs, exec_time, conv_time, steady_error):
    # Callback when ANC finishes
    global noisy_audio, filtered_audio, total_time, convergence_speed, steady_state_error, stored_t, stored_fs
    global stored_reference_signal, stored_error_signal

    noisy_audio = noisy_signal
    filtered_audio = filtered_signal
    stored_reference_signal = reference_signal
    stored_error_signal = error_signal
    stored_t = t
    stored_fs = fs

    total_time = exec_time
    convergence_speed = conv_time
    steady_state_error = steady_error

    root.after(0, lambda: status_label.config(text="Finished", fg="green"))
    root.after(0, enable_buttons)
    root.after(0, lambda: play_noisy_btn.config(state=tk.NORMAL))
    root.after(0, lambda: play_filtered_btn.config(state=tk.NORMAL))
    root.after(0, lambda: graph_button.config(state=tk.NORMAL))

    root.after(0, lambda: total_time_label.config(text=f"Total Time: {total_time:.2f} sec"))
    root.after(0, lambda: conv_time_label.config(text=f"Convergence Speed: {convergence_speed:.2f} sec"))
    root.after(0, lambda: error_label.config(text=f"Steady-State Error: {steady_state_error:.2f} dB"))

def open_graphs():
    global stored_reference_signal, noisy_audio, filtered_audio, stored_error_signal, stored_t, stored_fs
    root.after(0, lambda: plot_results(stored_reference_signal, noisy_audio, filtered_audio, stored_error_signal, stored_t, stored_fs))

def play_noisy_signal():
    global is_playing
    if is_playing:
        return
    is_playing = True

    # Disable only the play buttons
    play_noisy_btn.config(state=tk.DISABLED)
    play_filtered_btn.config(state=tk.DISABLED)

    def play():
        try:
            play_audio(noisy_audio)
        except Exception as e:
            print(f"Error playing noisy signal: {e}")
        finally:
            # Re-enable play buttons but keep Graphs enabled
            root.after(0, lambda: play_noisy_btn.config(state=tk.NORMAL))
            root.after(0, lambda: play_filtered_btn.config(state=tk.NORMAL))
            global is_playing
            is_playing = False

    threading.Thread(target=play, daemon=True).start()

def play_filtered_signal():
    global is_playing
    if is_playing:
        return
    is_playing = True

    # Disable only the play buttons
    play_noisy_btn.config(state=tk.DISABLED)
    play_filtered_btn.config(state=tk.DISABLED)

    def play():
        try:
            play_audio(filtered_audio)
        except Exception as e:
            print(f"Error playing filtered signal: {e}")
        finally:
            # Re-enable play buttons but keep "Graphs" enabled
            root.after(0, lambda: play_noisy_btn.config(state=tk.NORMAL))
            root.after(0, lambda: play_filtered_btn.config(state=tk.NORMAL))
            global is_playing
            is_playing = False

    threading.Thread(target=play, daemon=True).start()

# GUI Elements
tk.Label(root, text="Select Algorithm:").grid(row=0, column=0)
algo_var = tk.StringVar(value="LMS")
algo_menu = ttk.Combobox(root, textvariable=algo_var, values=["LMS", "NLMS", "FxLMS", "FxNLMS"])
algo_menu.grid(row=0, column=1)

tk.Label(root, text="Mu:").grid(row=1, column=0)
mu_entry = tk.Entry(root)
mu_entry.grid(row=1, column=1)

tk.Label(root, text="L:").grid(row=2, column=0)
L_entry = tk.Entry(root)
L_entry.grid(row=2, column=1)

tk.Label(root, text="Select Noise Type:").grid(row=3, column=0)
noise_var = tk.StringVar(value="White")
noise_menu = ttk.Combobox(root, textvariable=noise_var, values=["White", "Pink", "Brownian", "Violet", "Grey", "Blue"])
noise_menu.grid(row=3, column=1)

tk.Label(root, text="SNR (dB):").grid(row=4, column=0)
snr_entry = tk.Entry(root)
snr_entry.grid(row=4, column=1)

start_button = tk.Button(root, text="Start", command=start_algorithm)
start_button.grid(row=5, column=0, columnspan=2)

play_noisy_btn = tk.Button(root, text="Play Noisy Signal", state=tk.DISABLED, command=play_noisy_signal)
play_noisy_btn.grid(row=6, column=0)

play_filtered_btn = tk.Button(root, text="Play Filtered Signal", state=tk.DISABLED, command=play_filtered_signal)
play_filtered_btn.grid(row=6, column=1)

graph_button = tk.Button(root, text="Graphs", state=tk.DISABLED, command=open_graphs)
graph_button.grid(row=7, column=0, columnspan=2)

status_label = tk.Label(root, text="", fg="black")
status_label.grid(row=8, column=0, columnspan=2)

time_label = tk.Label(root, text="Time remaining: 00:00", fg="black")
time_label.grid(row=9, column=0, columnspan=2)

progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
progress_bar.grid(row=10, column=0, columnspan=2, pady=10)

total_time_label = tk.Label(root, text="Total Time: --")
total_time_label.grid(row=11, column=0, columnspan=2)

conv_time_label = tk.Label(root, text="Convergence Speed: --")
conv_time_label.grid(row=12, column=0, columnspan=2)

error_label = tk.Label(root, text="Steady-State Error: --")
error_label.grid(row=13, column=0, columnspan=2)

root.mainloop()