import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
import gui_single
import gui_multi

class SharedState:
    """Holds shared Tk variables, references to widgets, and simulation data."""
    # Tk variables used across panels
    algo_var: tk.StringVar = None
    noise_source_var: tk.StringVar = None
    noise_var: tk.StringVar = None
    wav_file_path: tk.StringVar = None

    # Common entries (assigned in gui_single)
    L_entry = None
    mu_entry = None
    duration_entry = None
    bands_entry = None

    # Progress + status (assigned in gui_single)
    progress_var = None
    progress_bar = None
    status_label = None

    # Buttons (for enable/disable as a group)
    all_buttons = None

    # Stored signals/state set by single-run completion
    stored_reference_signal = None
    stored_noisy_signal = None
    stored_signal_after_primary = None
    stored_signal_after_secondary = None
    stored_error_signal = None
    stored_filtered_signal = None
    stored_error_signal_raw = None
    stored_filtered_signal_raw = None
    stored_t = None
    stored_fs = 44100
    stored_initial_weights = None
    stored_final_weights = None
    stored_primary_ir = None
    stored_secondary_ir = None
    stored_in_power = None
    stored_out_power = None

    # Callbacks that multi panel may call
    start_single_run_cb = None  # set by gui_single
    disable_buttons_cb = None   # set by gui_single
    enable_buttons_cb = None    # set by gui_single

    # Best (mu, L) remembered by multi panel
    best_mu = None
    best_L = None

    # Single-run timing
    single_start_time = None
    eta_label = None

    # Raw mic-level audio (before/after) for playback
    stored_before_signal_raw = None
    stored_after_signal_raw = None

    stored_convergence_speed = None
    stored_steady_state_error = None
    stored_execution_time = None

    # Links to other widgets
    wav_label_ref = None  # set by gui_single

    # Last multi-run artifacts for plotting
    last_ranked = None
    last_mu_vals = None
    last_L_vals  = None

def build_and_run():
    root = tk.Tk()
    root.title("ANC with NN — Single & Multi Run")

    paned = ttk.PanedWindow(root, orient="horizontal")
    paned.pack(fill="both", expand=True)

    left_frame  = tk.Frame(paned)
    right_frame = tk.Frame(paned)
    paned.add(left_frame,  weight=1)
    paned.add(right_frame, weight=1)

    default_font = tkfont.Font(size=10)
    header_font  = tkfont.Font(size=12, weight="bold")

    state = SharedState()
    # Only these have defaults
    state.algo_var = tk.StringVar(value="FxLMS")
    state.noise_source_var = tk.StringVar(value="Stationary")
    state.noise_var = tk.StringVar(value="White")
    state.wav_file_path = tk.StringVar(value="")
    state.all_buttons = []

    gui_single.build_single_ui(left_frame, state, default_font, header_font)
    gui_multi.build_multi_ui(right_frame, state, default_font, header_font)

    for f in (left_frame, right_frame):
        f.grid_columnconfigure(0, weight=0)
        f.grid_columnconfigure(1, weight=1)

    root.mainloop()

if __name__ == "__main__":
    build_and_run()