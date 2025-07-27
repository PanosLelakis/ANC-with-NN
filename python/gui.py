import tkinter as tk
from tkinter import ttk
import threading
import time
import tkinter.font as tkfont
from utils.audio_utils import play_audio, stop_audio
from main import run_anc
from utils.plot import *

# Window size parameters
aspect_ratio = 3 / 8
initial_width = 200
initial_height = int(initial_width / aspect_ratio)

# Create the root window first
root = tk.Tk()
root.title("ANC with NN - Panos Lelakis")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = int(screen_width * 0.3)  # 30% of screen width
window_height = int(screen_height * 0.8)  # 80% of screen height
x_pos = int((screen_width - window_width) / 2)
y_pos = int((screen_height - window_height) / 2)
root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
root.minsize(initial_width, initial_height)

# Global variables
noisy_audio = None
filtered_audio = None
is_playing = False
progress_var = tk.DoubleVar(value=0.0)
start_time = None  
total_time = 0
convergence_speed = 0
steady_state_error = 0
algorithm = ""
mu = 0
L = 0
snr = 0
noise_type = ""
stored_initial_weights = None
stored_final_weights = None
stored_signal_after_primary = None
stored_signal_after_secondary = None
all_buttons = []
resizing = False # flag to avoid recursion

def start_algorithm():
    global start_time, algorithm, mu, L, snr, noise_type

    reset_result_labels()
    disable_buttons()
    progress_var.set(0.0)
    status_label.config(text="Running...", fg="black")

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
        start_button.config(state=tk.NORMAL)
        return

    # Run ANC in separate thread to avoid blocking GUI
    threading.Thread(target=run_anc, args=(algorithm, L, mu, snr, noise_type, update_progress, on_anc_complete), daemon=True).start()

def disable_buttons():
    global all_buttons
    for b in all_buttons:
        b.config(state=tk.DISABLED)

def enable_buttons():
    global all_buttons
    for b in all_buttons:
        b.config(state=tk.NORMAL)

def reset_result_labels():
    total_time_label.config(text="Total Time: --")
    conv_time_label.config(text="Convergence Speed: --")
    error_label.config(text="Steady-State Error: --")
    status_label.config(text="Running...", fg="black")

def update_progress(progress):
    # Update progress bar and estimated time
    root.after(0, lambda: progress_var.set(progress))  # Ensure thread-safe update
    elapsed_time = time.time() - start_time
    estimated_time = (elapsed_time / progress) * (100 - progress) if progress > 0 else 0
    minutes, seconds = divmod(int(estimated_time), 60)
    root.after(0, lambda: time_label.config(text=f"Time remaining: {minutes:02}:{seconds:02}"))

def on_anc_complete(reference_signal, noisy_signal, filtered_signal, error_signal, 
                   t, fs, exec_time, conv_time, steady_error, initial_weights,
                   final_weights, primary_ir, secondary_ir,
                   signal_after_primary, signal_after_secondary):
    
    # Store all the data
    global stored_initial_weights, stored_final_weights, stored_primary_ir, stored_secondary_ir
    global stored_signal_after_primary, stored_signal_after_secondary
    global noisy_audio, filtered_audio, total_time, convergence_speed, steady_state_error, stored_t, stored_fs
    global stored_reference_signal, stored_error_signal
    
    stored_initial_weights = initial_weights
    stored_final_weights = final_weights
    stored_primary_ir = primary_ir
    stored_secondary_ir = secondary_ir
    noisy_audio = noisy_signal
    filtered_audio = filtered_signal
    stored_reference_signal = reference_signal
    stored_error_signal = error_signal
    stored_t = t
    stored_fs = fs

    total_time = exec_time
    convergence_speed = conv_time
    steady_state_error = steady_error

    stored_signal_after_primary = signal_after_primary
    stored_signal_after_secondary = signal_after_secondary

    root.after(0, lambda: status_label.config(text="Finished", fg="green"))
    root.after(0, enable_buttons)
    #if stored_signal_after_secondary == None:
        #secondary_path_btn.config(state=tk.DISABLED)

    root.after(0, lambda: total_time_label.config(text=f"Total Time: {total_time:.2f} sec"))
    if convergence_speed is not None:
        root.after(0, lambda: conv_time_label.config(text=f"Convergence Speed: {convergence_speed:.2f} ms"))
    else:
        root.after(0, lambda: conv_time_label.config(text="Convergence Speed: Not achieved"))
    root.after(0, lambda: error_label.config(text=f"Steady-State Error: {steady_state_error:.2f} dB"))

def play_noisy_signal():
    global is_playing

    if is_playing and play_noisy_btn["text"] == "Stop playing":
        stop_audio()
        is_playing = False
        reset_play_buttons()
        return

    is_playing = True
    play_noisy_btn.config(text="Stop playing", state=tk.NORMAL)
    play_filtered_btn.config(state=tk.DISABLED)

    threading.Thread(target=play_audio_thread, args=[noisy_audio], daemon=True).start()

def play_filtered_signal():
    global is_playing

    if is_playing and play_filtered_btn["text"] == "Stop playing":
        stop_audio()
        is_playing = False
        reset_play_buttons()
        return

    is_playing = True
    play_filtered_btn.config(text="Stop playing", state=tk.NORMAL)
    play_noisy_btn.config(state=tk.DISABLED)

    threading.Thread(target=play_audio_thread, args=[filtered_audio], daemon=True).start()

def play_audio_thread(audio):
        try:
            play_audio(audio)
        except Exception as e:
            print(f"Error playing signal: {e}")
        finally:
            root.after(0, reset_play_buttons)

def reset_play_buttons():
    global is_playing
    is_playing = False
    play_noisy_btn.config(state=tk.NORMAL, text="Play Noisy Signal")
    play_filtered_btn.config(state=tk.NORMAL, text="Play Filtered Signal")

def plot_filter():
    global algorithm, mu, L, noise_type, snr, convergence_speed, steady_state_error
    global stored_initial_weights, stored_final_weights, stored_fs
    plot_filter_weights(
        fs=stored_fs,
        w_final=stored_final_weights,
        #stored_initial_weights,
        algorithm_name=algorithm,
        mu=mu,
        L=L,
        noise_type=noise_type,
        snr=snr,
        convergence_time=convergence_speed,
        steady_state_error=steady_state_error
    )

def plot_primary_path_effect():
    global algorithm, mu, L, noise_type, snr, convergence_speed, steady_state_error
    global stored_t, stored_fs, stored_primary_ir, stored_signal_after_primary
    plot_path_analysis(
        stored_primary_ir,
        noisy_audio,
        stored_signal_after_primary,
        stored_t,
        stored_fs,
        title_prefix="Primary",
        algorithm_name=algorithm,
        mu=mu,
        L=L,
        noise_type=noise_type,
        snr=snr,
        convergence_time=convergence_speed,
        steady_state_error=steady_state_error
    )

def plot_secondary_path_effect():
    global algorithm, mu, L, noise_type, snr, convergence_speed, steady_state_error
    global stored_t, stored_fs, stored_secondary_ir, stored_signal_after_secondary
    plot_path_analysis(
        stored_secondary_ir,
        noisy_audio,
        stored_signal_after_secondary,
        stored_t,
        stored_fs,
        title_prefix="Secondary",
        algorithm_name=algorithm,
        mu=mu,
        L=L,
        noise_type=noise_type,
        snr=snr,
        convergence_time=convergence_speed,
        steady_state_error=steady_state_error
    )

def plot_error():
    global algorithm, mu, L, noise_type, snr, convergence_speed, steady_state_error
    global stored_error_signal, stored_signal_after_primary, stored_t, stored_fs
    plot_error_analysis(
        stored_error_signal,
        stored_t,
        stored_fs,
        passive_cancelling=stored_signal_after_primary,
        algorithm_name=algorithm,
        mu=mu,
        L=L,
        noise_type=noise_type,
        snr=snr,
        convergence_time=convergence_speed,
        steady_state_error=steady_state_error
    )

def plot_signal():
    global stored_reference_signal, noisy_audio, filtered_audio, stored_t
    global algorithm, mu, L, noise_type, snr, convergence_speed, steady_state_error
    plot_signal_flow(
        stored_reference_signal,
        noisy_audio,
        filtered_audio,
        stored_t,
        algorithm_name=algorithm,
        mu=mu,
        L=L,
        noise_type=noise_type,
        snr=snr,
        convergence_time=convergence_speed,
        steady_state_error=steady_state_error
    )

# Create a custom font style
font_size = max(11, int(screen_height / 70))  # Minimum size 10, scales with screen
default_font = tkfont.nametofont("TkDefaultFont")
default_font.configure(size=font_size)  # Increase default font size
root.option_add("*Font", default_font)

# GUI Elements
tk.Label(root, text="Select Algorithm:", font=default_font).grid(row=0, column=0)
algo_var = tk.StringVar(value="LMS")
algo_menu = ttk.Combobox(root, textvariable=algo_var, values=["LMS", "NLMS", "FxLMS", "FxNLMS"], font=default_font)
algo_menu.grid(row=0, column=1)

tk.Label(root, text="Mu:", font=default_font).grid(row=1, column=0)
mu_entry = tk.Entry(root, font=default_font)
mu_entry.grid(row=1, column=1)

tk.Label(root, text="L:", font=default_font).grid(row=2, column=0)
L_entry = tk.Entry(root, font=default_font)
L_entry.grid(row=2, column=1)

tk.Label(root, text="Select Noise Type:", font=default_font).grid(row=3, column=0)
noise_var = tk.StringVar(value="White")
noise_menu = ttk.Combobox(root, textvariable=noise_var, values=["White", "Pink", "Brownian", "Violet", "Grey", "Blue"], font=default_font)
noise_menu.grid(row=3, column=1)

tk.Label(root, text="SNR (dB):", font=default_font).grid(row=4, column=0)
snr_entry = tk.Entry(root, font=default_font)
snr_entry.grid(row=4, column=1)

start_button = tk.Button(root, text="Start", font=default_font, command=start_algorithm)
start_button.grid(row=5, column=0, columnspan=2)
all_buttons.append(start_button)

play_noisy_btn = tk.Button(root, text="Play Noisy Signal", state=tk.DISABLED, font=default_font, command=play_noisy_signal)
play_noisy_btn.grid(row=6, column=0)
all_buttons.append(play_noisy_btn)

play_filtered_btn = tk.Button(root, text="Play Filtered Signal", state=tk.DISABLED, font=default_font, command=play_filtered_signal)
play_filtered_btn.grid(row=6, column=1)
all_buttons.append(play_filtered_btn)

status_label = tk.Label(root, text="", fg="black", font=default_font)
status_label.grid(row=7, column=0, columnspan=2)

time_label = tk.Label(root, text="Time remaining: 00:00", fg="black", font=default_font)
time_label.grid(row=8, column=0, columnspan=2)

progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
progress_bar.grid(row=9, column=0, columnspan=2, pady=10)

total_time_label = tk.Label(root, text="Total Time: --", font=default_font)
total_time_label.grid(row=10, column=0, columnspan=2)

conv_time_label = tk.Label(root, text="Convergence Speed: --", font=default_font)
conv_time_label.grid(row=11, column=0, columnspan=2)

error_label = tk.Label(root, text="Steady-State Error: --", font=default_font)
error_label.grid(row=12, column=0, columnspan=2)

# --- Diagnostic Graphs Frame ---
diagnostics_frame = tk.LabelFrame(root, text="Graphs", font=default_font)
diagnostics_frame.grid(row=13, column=0, columnspan=2, pady=10)

filter_weights_btn = tk.Button(diagnostics_frame, text="Filter Weights", state=tk.DISABLED, font=default_font, command=plot_filter)
filter_weights_btn.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
all_buttons.append(filter_weights_btn)

primary_path_btn = tk.Button(diagnostics_frame, text="Primary Path", state=tk.DISABLED, font=default_font, command=plot_primary_path_effect)
primary_path_btn.grid(row=1, column=0, sticky="ew", padx=5, pady=2)
all_buttons.append(primary_path_btn)

secondary_path_btn = tk.Button(diagnostics_frame, text="Secondary Path", state=tk.DISABLED, font=default_font, command=plot_secondary_path_effect)
secondary_path_btn.grid(row=2, column=0, sticky="ew", padx=5, pady=2)
all_buttons.append(secondary_path_btn)

error_analysis_btn = tk.Button(diagnostics_frame, text="Error Analysis", state=tk.DISABLED, font=default_font, command=plot_error)
error_analysis_btn.grid(row=3, column=0, sticky="ew", padx=5, pady=2)
all_buttons.append(error_analysis_btn)

signal_flow_btn = tk.Button(diagnostics_frame, text="Signal Flow", state=tk.DISABLED, font=default_font, command=plot_signal)
signal_flow_btn.grid(row=4, column=0, sticky="ew", padx=5, pady=2)
all_buttons.append(signal_flow_btn)

def on_resizee(event):
    # Lock aspect ratio
    desired_height = int(event.width / aspect_ratio)
    if event.height != desired_height:
        root.geometry(f"{event.width}x{desired_height}")
        return  # Skip font scaling until next event

    # Adjust font size based on height
    scale = event.height / initial_height
    new_font_size = int(10 * scale)

    new_font = tkfont.Font(size=new_font_size)

    # Apply font to all children
    def apply_font_recursive(widget):
        try:
            widget.configure(font=new_font)
        except:
            pass
        for child in widget.winfo_children():
            apply_font_recursive(child)

    apply_font_recursive(root)

def on_resize(event):
    global resizing
    if resizing:
        return  # prevent recursion

    resizing = True

    try:
        # Lock aspect ratio
        desired_width = event.width
        desired_height = int(desired_width / aspect_ratio)

        # Prevent infinite loop by checking if size is already close
        if abs(event.height - desired_height) > 2:
            root.geometry(f"{desired_width}x{desired_height}")

        # Dynamic font resizing (proportional to height)
        font_size = max(8, int(desired_height / 40))  # tune divisor as needed
        default_font.configure(size=font_size)

    finally:
        resizing = False

# Add padding to all widgets
#for child in root.winfo_children():
    #child.grid_configure(padx=5, pady=5)
    
#for child in diagnostics_frame.winfo_children():
    #child.grid_configure(padx=5, pady=5)

# Make the progress bar thicker
progress_bar.config(length=window_width-10)  # Subtract some pixels for padding

for i in range(root.grid_size()[0]):
    root.columnconfigure(i, weight=1)

for i in range(root.grid_size()[1]):
    root.rowconfigure(i, weight=1)

#root.resizable(True, True)
#root.bind("<Configure>", on_resize)

root.mainloop()