import tkinter as tk
import traceback
from tkinter import ttk
import tkinter.font as tkfont
from gui.gui_single import build_single_ui
from gui.gui_multi import build_multi_ui
from gui.gui_nn import build_nn_ui

class SharedState:
    """Holds shared Tk variables, references to widgets, and simulation data."""
    # Tk variables used across panels
    algo_var: tk.StringVar = None
    noise_source_var: tk.StringVar = None
    noise_var: tk.StringVar = None
    wav_file_path: tk.StringVar = None

    # NN dataset settings
    nn_dataset_root_var: tk.StringVar = None
    nn_processed_root_var: tk.StringVar = None
    nn_checkpoint_path_var: tk.StringVar = None

    # NN training settings
    nn_target_fs_var: tk.StringVar = None
    nn_crop_sec_var: tk.StringVar = None
    nn_epochs_var: tk.StringVar = None
    nn_batch_size_var: tk.StringVar = None
    nn_lr_var: tk.StringVar = None

    # NN model settings
    nn_conv_channels_var: tk.StringVar = None
    nn_lstm_hidden_var: tk.StringVar = None
    nn_delay_m_var: tk.StringVar = None

    # NN status
    nn_status_label = None

    # Common entries (assigned in gui_single)
    L_entry = None
    mu_entry = None
    duration_entry = None
    bands_entry = None

    # Progress + status (assigned in gui_single)
    progress_var = None
    progress_bar = None
    status_label = None

    # Buttons list (for mass enabling/disabling)
    all_buttons = None

    # Stored signals/state set by single-run completion
    stored_reference_signal = None
    stored_noisy_signal = None
    stored_signal_after_primary = None
    stored_signal_after_secondary = None
    stored_error_signal = None
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
    stored_divergence = False

    # Links to other widgets
    wav_label_ref = None  # set by gui_single

    last_best_combo = None

def build_and_run():
    import queue
    
    # Build main program window
    root = tk.Tk()
    root.title("ANC with NN — Single & Multi Run")

    # Main tabs
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # Tab frames
    single_frame = tk.Frame(notebook)
    multi_frame = tk.Frame(notebook)
    nn_frame = tk.Frame(notebook)

    # Add tabs
    notebook.add(single_frame, text="Single Run")
    notebook.add(multi_frame, text="Multi Run")
    notebook.add(nn_frame, text="Neural Network")

    default_font = tkfont.Font(size=10)
    header_font  = tkfont.Font(size=12, weight="bold")

    state = SharedState()
    # Set default values
    state.algo_var = tk.StringVar(value="FxNLMS")
    state.noise_source_var = tk.StringVar(value="Stationary")
    state.noise_var = tk.StringVar(value="White")
    state.wav_file_path = tk.StringVar(value="")
    state.all_buttons = [] # All buttons list
    state.ui_drain_after_id = None

    # NN dataset defaults
    state.nn_dataset_root_var = tk.StringVar(value="python/dataset/dataset_1")
    state.nn_processed_root_var = tk.StringVar(value="python/dataset/processed/dataset_1_16k_40s")
    state.nn_checkpoint_path_var = tk.StringVar(value="")

    # NN training defaults
    state.nn_target_fs_var = tk.StringVar(value="16000")
    state.nn_crop_sec_var = tk.StringVar(value="40")
    state.nn_epochs_var = tk.StringVar(value="30")
    state.nn_batch_size_var = tk.StringVar(value="1")
    state.nn_lr_var = tk.StringVar(value="0.001")

    # NN model defaults
    state.nn_conv_channels_var = tk.StringVar(value="16,32")
    state.nn_lstm_hidden_var = tk.StringVar(value="128")
    state.nn_delay_m_var = tk.StringVar(value="0")

     # Build tab contents
    build_single_ui(single_frame, state, default_font, header_font)
    build_multi_ui(multi_frame, state, default_font, header_font)
    build_nn_ui(nn_frame, state, default_font, header_font)

    # Column layout
    for f in (single_frame, multi_frame, nn_frame):
        f.grid_columnconfigure(0, weight=0)
        f.grid_columnconfigure(1, weight=1)

    state.root = root
    state._locked_widget_states = {}
    state.is_locked = False
    state.is_closing = False
    state.ui_queue = queue.Queue()

    def ui_call(fn, *args, **kwargs):
        if getattr(state, "is_closing", False):
            return
        try:
            state.ui_queue.put_nowait((fn, args, kwargs))
        except Exception as e:
            print("UI callback error:", e)
            traceback.print_exc()

    def _drain_ui_queue():
        # Runs on main Tk thread only
        try:
            while True:
                fn, args, kwargs = state.ui_queue.get_nowait()
                try:
                    fn(*args, **kwargs)
                except Exception as e:
                    print("UI callback error:", e)
                    traceback.print_exc()
        except queue.Empty:
            pass
        # schedule next drain
        try:
            if root.winfo_exists() and not getattr(state, "is_closing", False):
                state.ui_drain_after_id = root.after(15, _drain_ui_queue)
        except Exception as e:
            print("UI callback error:", e)
            traceback.print_exc()

    state.ui_call = ui_call
    root.after(15, _drain_ui_queue)

    def _walk_widgets(w):
        for ch in w.winfo_children():
            yield ch
            yield from _walk_widgets(ch)

    def lock_ui(allow_widgets=()):
        """
        Disable every widget that has a 'state' option, except those in allow_widgets.
        Preserves original state so we can restore precisely (e.g. readonly combobox).
        """
        allow = set(allow_widgets)
        if state.is_locked:
            return
        state.is_locked = True
        state._locked_widget_states = {}

        for w in _walk_widgets(root):
            if w in allow:
                continue
            try:
                if "state" in w.keys():
                    prev = w.cget("state")
                    state._locked_widget_states[w] = prev
                    w.configure(state="disabled")
            except Exception as e:
                print("UI callback error:", e)
                traceback.print_exc()

    def unlock_ui():
        if not state.is_locked:
            return
        for w, prev in list(state._locked_widget_states.items()):
            try:
                if w.winfo_exists():
                    w.configure(state=prev)
            except Exception as e:
                print("UI callback error:", e)
                traceback.print_exc()
        state._locked_widget_states = {}
        state.is_locked = False

    state.lock_ui = lock_ui
    state.unlock_ui = unlock_ui

    def on_close():
        state.is_closing = True

        # cancel scheduled UI drain
        try:
            if getattr(state, "ui_drain_after_id", None) is not None:
                root.after_cancel(state.ui_drain_after_id)
                state.ui_drain_after_id = None
        except Exception as e:
            print("UI callback error:", e)
            traceback.print_exc()

        try:
            from utils.audio import stop_audio
            stop_audio()
        except Exception as e:
            print("UI callback error:", e)
            traceback.print_exc()

        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()

# Typical __name__ type of shit
if __name__ == "__main__":
    build_and_run() # Build that shit and kick it alive