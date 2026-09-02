import tkinter as tk
import traceback
from tkinter import ttk
import tkinter.font as tkfont
from gui.gui_single import build_single_ui
from gui.gui_multi import build_multi_ui
from gui.gui_nn import build_nn_ui
from engine.engine_common import load_paths

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
    nn_backend_var: tk.StringVar = None

    # Single Run model
    single_nn_checkpoint_path_var: tk.StringVar = None

    # Multi Run model
    multi_nn_checkpoint_path_var: tk.StringVar = None

    # NN training settings
    nn_target_fs_var: tk.StringVar = None
    nn_crop_sec_var: tk.StringVar = None
    nn_epochs_var: tk.StringVar = None
    nn_batch_size_var: tk.StringVar = None
    nn_lr_var: tk.StringVar = None

    # NN model settings
    nn_conv_layers_var: tk.StringVar = None
    nn_conv_channels_var: tk.StringVar = None
    nn_lstm_layers_var: tk.StringVar = None
    nn_lstm_hidden_var: tk.StringVar = None
    nn_delay_m_var: tk.StringVar = None
    nn_optimizer_var: tk.StringVar = None

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

    # Last complete Single Run result
    last_single_result = None

    # Normalized playback signals
    play_input = None
    play_anc_off = None
    play_anc_on = None

    # Timings
    single_start_time = None
    eta_label = None

    # Links to other widgets
    wav_label_ref = None  # set by gui_single

    last_best_combo = None

    # Preloaded ANC paths
    anc_paths = None

def build_and_run():
    import queue
    
    # Build main window
    root = tk.Tk()
    root.title("ANC with NN — Panos Lelakis")

    # Main fonts
    default_font = tkfont.Font(size=10)
    header_font = tkfont.Font(size=12, weight="bold")
    tab_font = tkfont.Font(size=15, weight="bold")

    # Tab style
    style = ttk.Style(root)
    style.configure(
        "Large.TNotebook.Tab",
        font=tab_font,
        padding=(24, 12)
    )

    # Main tabs
    notebook = ttk.Notebook(root, style="Large.TNotebook")
    notebook.pack(fill="both", expand=True)

    # Tab frames
    single_frame = tk.Frame(notebook)
    multi_frame = tk.Frame(notebook)
    nn_frame = tk.Frame(notebook)

    # Add tabs
    notebook.add(single_frame, text="Single Run")
    notebook.add(multi_frame, text="Multi Run")
    notebook.add(nn_frame, text="Neural Network")

    state = SharedState()

    state.notebook = notebook

    # Set default values
    state.algo_var = tk.StringVar(value="FxNLMS")
    state.noise_source_var = tk.StringVar(value="Stationary")
    state.noise_var = tk.StringVar(value="White")
    state.wav_file_path = tk.StringVar(value="")
    state.all_buttons = [] # All buttons list
    state.ui_drain_after_id = None

    # NN dataset defaults
    state.nn_dataset_root_var = tk.StringVar(
        value="dataset/trimmed_selection_train_validate"
    )
    state.nn_processed_root_var = tk.StringVar(
        value="dataset/selection_processed"
    )
    state.nn_checkpoint_path_var = tk.StringVar(value="")

    # Store Single Run model
    state.single_nn_checkpoint_path_var = tk.StringVar(
        value=""
    )

    # Store Multi Run model
    state.multi_nn_checkpoint_path_var = tk.StringVar(
        value=""
    )

    state.nn_backend_var = tk.StringVar(value="PyTorch")

    # NN model defaults
    state.nn_conv_layers_var = tk.StringVar(value="5")  # Original encoder depth
    state.nn_conv_channels_var = tk.StringVar(value="16,32,64,128,256")  # Original encoder channels
    state.nn_lstm_layers_var = tk.StringVar(value="2")  # Original grouped LSTM depth
    state.nn_lstm_hidden_var = tk.StringVar(value="1024")  # Original total recurrent width
    state.nn_delay_m_var = tk.StringVar(value="0")  # Start with no prediction delay

    # NN training defaults
    state.nn_target_fs_var = tk.StringVar(value="16000")
    state.nn_crop_sec_var = tk.StringVar(value="10")
    state.nn_epochs_var = tk.StringVar(value="30")  # Reference epoch count
    state.nn_batch_size_var = tk.StringVar(value="1")  # Keep project batch size
    state.nn_lr_var = tk.StringVar(value="0.001")  # Reference learning rate
    state.nn_optimizer_var = tk.StringVar(value="AMSGrad")  # Reference optimizer

    # Load ANC paths once at startup
    state.anc_paths = load_paths()

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
    state._locked_tab_states = {}
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

        # Lock other tabs
        state._locked_tab_states = {}

        try:
            current_tab = state.notebook.index("current")
            total_tabs = state.notebook.index("end")

            for idx in range(total_tabs):
                prev_state = state.notebook.tab(idx, "state")
                state._locked_tab_states[idx] = prev_state

                if idx != current_tab:
                    state.notebook.tab(idx, state="disabled")

        except Exception as e:
            print("UI callback error:", e)
            traceback.print_exc()

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
        
        # Unlock tabs
        try:
            for idx, prev_state in list(state._locked_tab_states.items()):
                state.notebook.tab(idx, state=prev_state)

            state._locked_tab_states = {}

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