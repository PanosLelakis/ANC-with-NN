import tkinter as tk
import traceback
from tkinter import ttk
import tkinter.font as tkfont
from gui.gui_single import build_single_ui
from gui.gui_multi import build_multi_ui

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

    # Links to other widgets
    wav_label_ref = None  # set by gui_single

    last_best_combo = None

def build_and_run():
    import queue
    
    # Build main program window
    root = tk.Tk()
    root.title("ANC with NN — Single & Multi Run")

    paned = ttk.PanedWindow(root, orient="horizontal")
    paned.pack(fill="both", expand=True)

    left_frame  = tk.Frame(paned) # Single run gui
    right_frame = tk.Frame(paned) # Multi run gui
    paned.add(left_frame,  weight=1)
    paned.add(right_frame, weight=1)

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

    # Add single (left) and multi (right) run guis to main window
    build_single_ui(left_frame, state, default_font, header_font)
    build_multi_ui(right_frame, state, default_font, header_font)

    for f in (left_frame, right_frame):
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