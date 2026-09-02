import tkinter as tk
from tkinter import ttk, filedialog
import threading
import gc
import numpy as np
import time
import os
from utils.logger import init_log, log_case
from engine.engine_single import run_anc
from utils.plot import (
    plot_filter_weights, plot_path_analysis, plot_error_analysis, plot_signal_flow,
    plot_noise_spectrogram, plot_error_spectrogram, plot_band_attenuation
)
from utils.audio import play_audio, stop_audio
from utils.time_utils import estimate_eta
from utils.result_saver import save_case_artifacts

def build_single_ui(parent, state, default_font, header_font):
    # Local playback state
    is_playing = False
    play_token = 0
    
    # ---------------- helpers ----------------

    def _poll_playback(token):
        nonlocal is_playing, play_token
        if token != play_token:
            return

        from utils.audio import is_audio_active
        if is_audio_active():
            state.root.after(50, _poll_playback, token)
        else:
            is_playing = False
            reset_play_buttons()
            state.unlock_ui()
    
    def set_result_buttons(enabled):
        # Select button state
        button_state = tk.NORMAL if enabled else tk.DISABLED

        # Update result buttons
        for button in state.all_buttons:
            button.config(state=button_state)

        # Neural Network has no filter weights
        result = state.last_single_result

        if enabled and result is not None and result["algorithm"] == "Neural Network":
            fw_btn.config(state=tk.DISABLED)

    def reset_result_labels():
        conv_val.config(text="-")
        sse_val.config(text="-")
        inpow_val.config(text="-")
        outpow_val.config(text="-")
        exec_val.config(text="-")
        state.status_label.config(text="", fg="black")

    def reset_sim_state():
        # Clear previous result
        state.last_single_result = None

        # Clear playback signals
        state.play_input = None
        state.play_anc_off = None
        state.play_anc_on = None

        # Disable result actions
        set_result_buttons(False)

        # Clear displayed metrics
        reset_result_labels()

    def on_noise_source_change():
        mode = state.noise_source_var.get()

        if mode == "Stationary":
            noise_menu.config(state="readonly")
            wav_btn.config(state=tk.DISABLED)
            wav_label.config(text="No file selected")
        else:
            noise_menu.config(state=tk.DISABLED)
            wav_btn.config(state=tk.NORMAL)
        
        validate_single_ready()

    def on_algorithm_change(*_):
        # Check Neural Network selection
        is_neural = state.algo_var.get() == "Neural Network"

        # Update adaptive parameter fields
        state.L_entry.config(state=tk.DISABLED if is_neural else tk.NORMAL)
        state.mu_entry.config(state=tk.DISABLED if is_neural else tk.NORMAL)

        # Update backend field
        backend_menu.config(
            state=(
                "readonly"
                if is_neural
                else tk.DISABLED
            )
        )

        # Update model button
        model_btn.config(
            state=(
                tk.NORMAL
                if is_neural
                else tk.DISABLED
            )
        )

        # Refresh Start button
        validate_single_ready()
    
    def select_model_file():
        # Read current model path
        current_path = (
            state.single_nn_checkpoint_path_var
            .get()
            .strip()
        )

        # Select starting folder
        initial_dir = (
            os.path.dirname(current_path)
            if current_path
            else os.path.join(
                os.getcwd(),
                "models"
            )
        )

        # Open checkpoint selector
        path = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[
                (
                    "PyTorch checkpoints",
                    "*.pt"
                ),
                (
                    "All files",
                    "*.*"
                )
            ]
        )

        # Stop when no file is selected
        if not path:
            return

        # Store selected checkpoint
        state.single_nn_checkpoint_path_var.set(
            path
        )

        # Refresh Start button
        validate_single_ready()

    def select_wav_file():
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        
        if path:
            try:
                stop_audio()
            except Exception:
                pass
            try:
                from matplotlib import pyplot as plt
                plt.close('all')
            except Exception:
                pass
            gc.collect()
            reset_sim_state()
            state.wav_file_path.set(path)
            wav_label.config(text=path.split("/")[-1])
            state.status_label.config(text="WAV selected. Press Start to run.", fg="black")
        
        validate_single_ready()

    def select_model_file():
        # Read current model
        current_path = (
            state.single_nn_checkpoint_path_var
            .get()
            .strip()
        )

        # Select initial folder
        initial_dir = (
            os.path.dirname(current_path)
            if current_path
            else os.path.join(
                os.getcwd(),
                "models"
            )
        )

        # Select checkpoint
        path = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[
                (
                    "PyTorch checkpoints",
                    "*.pt"
                ),
                (
                    "All files",
                    "*.*"
                )
            ]
        )

        # Stop after cancellation
        if not path:
            return

        # Store checkpoint
        state.single_nn_checkpoint_path_var.set(
            path
        )

        # Show selected model
        model_label.config(
            text=(
                f"{os.path.basename(os.path.dirname(path))}/"
                f"{os.path.basename(path)}"
            )
        )

        # Refresh Start button
        validate_single_ready()

    def on_anc_complete(result):
        # Store complete simulation result
        state.last_single_result = result

        # Read metrics
        conv_ms = result["conv_ms"]
        sse_db = result["sse_db"]
        in_power = result["in_power"]
        out_power = result["out_power"]
        exec_time = result["exec_time"]
        divergence = bool(result["divergence"])

        # Show metrics
        conv_val.config(text="N/A" if conv_ms is None else f"{conv_ms:.2f} ms")
        sse_val.config(text=f"{sse_db:.2f} dBr")
        inpow_val.config(text=f"{in_power:.3f}")
        outpow_val.config(text=f"{out_power:.3f}")
        exec_val.config(text=f"{exec_time:.2f} s")

        # Show simulation status
        if divergence:
            state.status_label.config(text="Divergence Detected", fg="red")
        else:
            state.status_label.config(text="Done.", fg="green")

        # Complete progress
        state.progress_var.set(100.0)
        state.progress_bar.update_idletasks()

        # Read playback signals
        input_signal = np.nan_to_num(
            result["noisy"],
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        anc_off = np.nan_to_num(
            result["before_raw"],
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        anc_on = np.nan_to_num(
            result["after_raw"],
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # Use the same playback scale for fair comparison
        max_abs = max(
            float(np.max(np.abs(input_signal))),
            float(np.max(np.abs(anc_off))),
            float(np.max(np.abs(anc_on))),
            1e-6
        )

        scale = min(1.0, 0.99 / max_abs)

        # Store playback signals
        state.play_input = np.clip(
            input_signal * scale,
            -1.0,
            1.0
        ).astype(np.float32)

        state.play_anc_off = np.clip(
            anc_off * scale,
            -1.0,
            1.0
        ).astype(np.float32)

        state.play_anc_on = np.clip(
            anc_on * scale,
            -1.0,
            1.0
        ).astype(np.float32)

        # Log simulation
        log_case(
            stage="single",
            status="diverged" if divergence else "ok",
            algorithm=result["algorithm"],
            source=result["source"],
            noise_label=result["noise_label"],
            L=int(result["L"]),
            mu=float(result["mu"]),
            conv_ms=conv_ms,
            sse_db=sse_db,
            exec_time=exec_time,
            in_power=in_power,
            out_power=out_power,
            save_path="",
            message="Divergence detected." if divergence else "",
            divergence=divergence
        )

        # Unlock GUI
        state.unlock_ui()

        # Enable result actions
        set_result_buttons(True)

        # Restore source widgets
        on_noise_source_change()

    def show_run_error(message):
        # Unlock GUI
        state.unlock_ui()

        # Reset ETA
        state.eta_label.config(text="ETA --:--")

        # Show error
        state.status_label.config(text=f"Simulation error: {message}", fg="red")

        # Refresh Start button
        validate_single_ready()

    def start_algorithm():
        # Read validated simulation inputs
        algorithm = state.algo_var.get()
        duration = float(state.duration_entry.get())
        noise_source = state.noise_source_var.get()
        noise_wav_path = state.wav_file_path.get()
        nn_backend = state.nn_backend_var.get().lower()

        # Read noise label
        if noise_source == "WAV":
            noise_type = os.path.basename(noise_wav_path)
        else:
            noise_type = state.noise_var.get()

        # Initialize checkpoint path
        nn_checkpoint_path = None

        # Read algorithm-specific values
        if algorithm == "Neural Network":
            L = 0
            mu = 0.0

                        # Read selected checkpoint
            nn_checkpoint_path = (
                state.single_nn_checkpoint_path_var
                .get()
                .strip()
            )

            # Require selected model
            if not nn_checkpoint_path:
                state.status_label.config(
                    text=(
                        "Select a Neural Network "
                        "model first."
                    ),
                    fg="red"
                )

                return

            # Check ONNX model
            if nn_backend == "onnx":
                onnx_path = os.path.splitext(nn_checkpoint_path)[0] + ".onnx"

                if not os.path.exists(onnx_path):
                    state.status_label.config(
                        text=f"ONNX model not found: {onnx_path}",
                        fg="red"
                    )
                    return

        else:
            L = int(state.L_entry.get())
            mu = float(state.mu_entry.get())

        # Stop previous playback
        stop_audio()

        # Close previous plots
        try:
            from matplotlib import pyplot as plt
            plt.close("all")
        except Exception:
            pass

        # Clear previous result
        reset_sim_state()

        # Release unused objects
        gc.collect()

        # Prepare GUI
        start_btn.config(state=tk.DISABLED)
        state.status_label.config(text="Running…", fg="black")
        state.progress_var.set(0.0)
        state.progress_bar.update_idletasks()
        state.single_start_time = time.time()
        state.eta_label.config(text="ETA --:--")
        state.lock_ui()

        # Initialize log
        init_log(run_kind="single", clear=True)

        def progress_callback(percentage):
            state.ui_call(update_progress, percentage)

        def worker():
            try:
                # Run simulation
                result = run_anc(
                    algorithm_name=algorithm,
                    L=L,
                    mu=mu,
                    noise_source=noise_source,
                    noise_type=noise_type,
                    noise_wav_path=noise_wav_path,
                    duration=duration,
                    progress_callback=progress_callback,
                    nn_checkpoint_path=nn_checkpoint_path,
                    nn_backend=nn_backend,
                    paths=state.anc_paths
                )

                # Send result to GUI
                state.ui_call(on_anc_complete, result)

            except Exception as error:
                # Send error to GUI
                state.ui_call(show_run_error, str(error))

        # Start worker thread
        threading.Thread(target=worker, daemon=True).start()

    def validate_single_ready(*_):
        """Enable Start only when all required fields are filled and valid."""
        if getattr(state, "is_locked", False):
            try:
                start_btn.config(state=tk.DISABLED)
            except Exception:
                pass
            return
        
        ok = True

        # Require model for Neural Network
        if state.algo_var.get() == "Neural Network":
            ok &= bool(
                state.single_nn_checkpoint_path_var
                .get()
                .strip()
            )

        # Algorithm must be selected
        ok &= bool(state.algo_var.get())
        # Numeric fields
        if state.algo_var.get() == "Neural Network":
            numeric_entries = (state.duration_entry,)
        else:
            numeric_entries = (state.L_entry, state.mu_entry, state.duration_entry)

        for e in numeric_entries:
            if e is None:
                ok = False
                break

            txt = e.get().strip()

            if not txt:
                ok = False
                break

        if ok:
            try:
                # Read duration
                duration_value = float(state.duration_entry.get())

                # Check duration
                ok = duration_value > 0.0

                # Check adaptive parameters
                if state.algo_var.get() != "Neural Network":
                    # Read filter length
                    L_value = int(state.L_entry.get())

                    # Read step size
                    mu_value = float(state.mu_entry.get())

                    # Check adaptive parameters
                    ok = (
                        ok
                        and L_value > 0
                        and mu_value > 0.0
                    )

            except Exception:
                # Mark invalid input
                ok = False

        # Require model for Neural Network
        if state.algo_var.get() == "Neural Network":
            ok &= bool(
                state.single_nn_checkpoint_path_var
                .get()
                .strip()
            )

        # WAV path required if WAV chosen
        if state.noise_source_var.get() == "WAV":
            ok &= bool(state.wav_file_path.get().strip())
        start_btn.config(state=(tk.NORMAL if ok else tk.DISABLED))
    
    def update_progress(pct):
        try:
            state.progress_var.set(float(pct))
            state.progress_bar.update_idletasks()
            # ETA
            if state.single_start_time is not None and pct > 0:
                state.eta_label.config(
                    text=estimate_eta(state.single_start_time, float(pct), 100.0)
                )
            else:
                state.eta_label.config(text="ETA --:--")
        except Exception:
            pass

    def reset_play_buttons():
        nonlocal is_playing

        is_playing = False

        play_input_btn.config(
            state=tk.NORMAL,
            text="Play Input"
        )

        play_anc_off_btn.config(
            state=tk.NORMAL,
            text="Play ANC OFF"
        )

        play_anc_on_btn.config(
            state=tk.NORMAL,
            text="Play ANC ON"
        )

    def toggle_play(button, audio_data):
        nonlocal is_playing, play_token

        if is_playing:
            play_token += 1
            stop_audio()
            is_playing = False
            reset_play_buttons()
            state.unlock_ui()
            return

        is_playing = True
        play_token += 1
        token = play_token

        button.config(
            text="Stop playing",
            state=tk.NORMAL
        )

        state.lock_ui(
            allow_widgets=(button,)
        )

        play_audio(
            audio_data,
            sample_rate=int(
                state.last_single_result["fs"]
            )
        )

        state.root.after(
            50,
            _poll_playback,
            token
        )

    def plot_metadata(result, save_dir):
        # Return common plot metadata
        return {
            "algorithm_name": result["algorithm"],
            "mu": result["mu"],
            "L": result["L"],
            "noise_type": result["noise_label"],
            "convergence_time": result["conv_ms"],
            "steady_state_error": result["sse_db"],
            "save_dir": save_dir
        }
    
    def plot_filter(save_dir=None):
        result = state.last_single_result
        plot_filter_weights(result["fs"], result["wf"], **plot_metadata(result, save_dir))

    def plot_primary_path_effect(save_dir=None):
        result = state.last_single_result
        plot_path_analysis(
            result["pir"], result["noisy"], result["d"], result["fs"], "Primary",
            **plot_metadata(result, save_dir)
        )

    def plot_secondary_path_effect(save_dir=None):
        result = state.last_single_result
        plot_path_analysis(
            result["sir"], result["noisy"], result["z"], result["fs"], "Secondary",
            **plot_metadata(result, save_dir)
        )

    def plot_error(save_dir=None):
        result = state.last_single_result
        plot_error_analysis(
            result["error"], result["t"], result["fs"],
            passive_cancelling=result["before_raw"],
            noisy_signal=result["noisy"],
            **plot_metadata(result, save_dir)
        )

    def plot_signal(save_dir=None):
        result = state.last_single_result
        plot_signal_flow(
            result["reference"], result["noisy"], result["error"], result["t"],
            **plot_metadata(result, save_dir)
        )

    def plot_noise_spec(save_dir=None):
        result = state.last_single_result
        plot_noise_spectrogram(result["noisy"], result["fs"], save_dir=save_dir)

    def plot_error_spec(save_dir=None):
        result = state.last_single_result
        plot_error_spectrogram(result["error"], result["fs"], save_dir=save_dir)

    def plot_band_attn(save_dir=None):
        result = state.last_single_result
        bands_str = state.bands_text.get("1.0", "end-1c").strip()

        plot_band_attenuation(
            result["before_raw"], result["after_raw"], result["fs"],
            bands_str=bands_str,
            **plot_metadata(result, save_dir)
        )
    
    # --- Save Results (single run) ---
    def save_single_results():
        # Read complete simulation result
        payload = state.last_single_result

        # Read original noise source
        source = payload["source"]

        # Read original noise label
        noise_label = payload["noise_label"]

        # Read custom frequency bands
        bands_str = state.bands_text.get("1.0", "end-1c").strip()

        # Build results root
        results_root = os.path.join(os.getcwd(), "results")

        # Show saving status
        state.status_label.config(text="Saving results...", fg="black")

        # Disable result buttons
        set_result_buttons(False)

        # Lock GUI
        state.lock_ui()

        def finish_save(message, color):
            # Unlock GUI
            state.unlock_ui()

            # Restore result buttons
            set_result_buttons(True)

            # Show final status
            state.status_label.config(text=message, fg=color)

        def worker():
            # Save complete result
            try:
                # Save artifacts
                metadata = save_case_artifacts(
                    payload=payload,
                    alg=payload["algorithm"],
                    src=source,
                    nlabel=noise_label,
                    L=payload["L"],
                    mu=payload["mu"],
                    base_root=results_root,
                    save_plots=True,
                    save_audio_file=True,
                    bands_str=bands_str
                )

            except Exception as error:
                # Show save error
                state.ui_call(
                    finish_save,
                    f"Save failed: {error}",
                    "red"
                )

                # Stop worker
                return

            # Show saved folder
            state.ui_call(
                finish_save,
                f"Saved to: {metadata['save_path']}",
                "green"
            )

        # Start saving thread
        threading.Thread(
            target=worker,
            daemon=True
        ).start()
    
        # ---------------- UI ----------------

    # Build title
    tk.Label(
        parent,
        text="Single Run",
        font=header_font
    ).grid(
        row=0,
        column=0,
        columnspan=2,
        sticky="w"
    )

    # Build parameters frame
    parameters_frame = ttk.LabelFrame(
        parent,
        text="Parameters"
    )

    parameters_frame.grid(
        row=1,
        column=0,
        columnspan=2,
        sticky="ew",
        padx=5,
        pady=5
    )

    # Expand value column
    parameters_frame.grid_columnconfigure(
        1,
        weight=1
    )

    # Add Algorithm label
    tk.Label(
        parameters_frame,
        text="Algorithm:",
        font=default_font
    ).grid(
        row=0,
        column=0,
        sticky="e"
    )

    # Build algorithm row
    algorithm_frame = tk.Frame(
        parameters_frame
    )

    algorithm_frame.grid(
        row=0,
        column=1,
        sticky="w"
    )

    # Build algorithm menu
    algo_menu = ttk.Combobox(
        algorithm_frame,
        textvariable=state.algo_var,
        values=[
            "LMS",
            "NLMS",
            "FxLMS",
            "FxNLMS",
            "Neural Network"
        ],
        state="readonly",
        width=15
    )

    algo_menu.pack(
        side="left"
    )

    # Handle algorithm selection
    algo_menu.bind(
        "<<ComboboxSelected>>",
        on_algorithm_change
    )

    # Add backend label
    tk.Label(
        algorithm_frame,
        text="NN Backend:",
        font=default_font
    ).pack(
        side="left",
        padx=(10, 2)
    )

    # Build backend menu
    backend_menu = ttk.Combobox(
        algorithm_frame,
        textvariable=state.nn_backend_var,
        values=[
            "PyTorch",
            "ONNX"
        ],
        state=tk.DISABLED,
        width=8
    )

    backend_menu.pack(
        side="left"
    )

    # Add model label
    tk.Label(
        parameters_frame,
        text="NN Model:",
        font=default_font
    ).grid(
        row=1,
        column=0,
        sticky="e"
    )

    # Build model row
    model_frame = tk.Frame(
        parameters_frame
    )

    model_frame.grid(
        row=1,
        column=1,
        sticky="ew"
    )

    # Expand model path
    model_frame.grid_columnconfigure(
        1,
        weight=1
    )

    # Build model button
    model_btn = tk.Button(
        model_frame,
        text="Select Model",
        command=select_model_file,
        state=tk.DISABLED
    )

    model_btn.grid(
        row=0,
        column=0,
        sticky="w"
    )

    # Show selected model path
    model_label = tk.Label(
        model_frame,
        textvariable=(
            state.single_nn_checkpoint_path_var
        ),
        font=default_font,
        anchor="w",
        wraplength=500
    )

    model_label.grid(
        row=0,
        column=1,
        sticky="ew",
        padx=(5, 0)
    )

    # Add step-size label
    tk.Label(
        parameters_frame,
        text="μ:",
        font=default_font
    ).grid(
        row=2,
        column=0,
        sticky="e"
    )

    # Build step-size entry
    state.mu_entry = tk.Entry(
        parameters_frame,
        width=10
    )

    state.mu_entry.grid(
        row=2,
        column=1,
        sticky="w"
    )

    state.mu_entry.bind(
        "<KeyRelease>",
        validate_single_ready
    )

    # Add filter-length label
    tk.Label(
        parameters_frame,
        text="L (taps):",
        font=default_font
    ).grid(
        row=3,
        column=0,
        sticky="e"
    )

    # Build filter-length entry
    state.L_entry = tk.Entry(
        parameters_frame,
        width=10
    )

    state.L_entry.grid(
        row=3,
        column=1,
        sticky="w"
    )

    state.L_entry.bind(
        "<KeyRelease>",
        validate_single_ready
    )

    # Add duration label
    tk.Label(
        parameters_frame,
        text="Duration (sec):",
        font=default_font
    ).grid(
        row=4,
        column=0,
        sticky="e"
    )

    # Build duration entry
    state.duration_entry = tk.Entry(
        parameters_frame,
        width=10
    )

    state.duration_entry.grid(
        row=4,
        column=1,
        sticky="w"
    )

    state.duration_entry.bind(
        "<KeyRelease>",
        validate_single_ready
    )

    # Add source label
    tk.Label(
        parameters_frame,
        text="Noise Source:",
        font=default_font
    ).grid(
        row=5,
        column=0,
        sticky="e"
    )

    # Build source row
    source_frame = tk.Frame(
        parameters_frame
    )

    source_frame.grid(
        row=5,
        column=1,
        sticky="w"
    )

    # Build stationary option
    tk.Radiobutton(
        source_frame,
        text="Stationary",
        variable=state.noise_source_var,
        value="Stationary",
        command=on_noise_source_change
    ).pack(
        side="left"
    )

    # Build WAV option
    tk.Radiobutton(
        source_frame,
        text="WAV",
        variable=state.noise_source_var,
        value="WAV",
        command=on_noise_source_change
    ).pack(
        side="left"
    )

    # Add noise-type label
    tk.Label(
        parameters_frame,
        text="Noise Type:",
        font=default_font
    ).grid(
        row=6,
        column=0,
        sticky="e"
    )

    # Build noise-type menu
    noise_menu = ttk.Combobox(
        parameters_frame,
        textvariable=state.noise_var,
        values=[
            "White",
            "Pink",
            "Brownian",
            "Violet",
            "Grey",
            "Blue"
        ],
        state="readonly",
        width=12
    )

    noise_menu.grid(
        row=6,
        column=1,
        sticky="w"
    )

    noise_menu.bind(
        "<<ComboboxSelected>>",
        validate_single_ready
    )

    # Add WAV label
    tk.Label(
        parameters_frame,
        text="Noise WAV:",
        font=default_font
    ).grid(
        row=7,
        column=0,
        sticky="e"
    )

    # Build WAV row
    wav_select_frame = tk.Frame(
        parameters_frame
    )

    wav_select_frame.grid(
        row=7,
        column=1,
        sticky="ew"
    )

    # Expand WAV filename
    wav_select_frame.grid_columnconfigure(
        1,
        weight=1
    )

    # Build WAV button
    wav_btn = tk.Button(
        wav_select_frame,
        text="Select WAV",
        command=select_wav_file,
        state=tk.DISABLED
    )

    wav_btn.grid(
        row=0,
        column=0,
        sticky="w"
    )

    # Show selected WAV
    wav_label = tk.Label(
        wav_select_frame,
        text="No file selected",
        font=default_font,
        anchor="w"
    )

    wav_label.grid(
        row=0,
        column=1,
        sticky="ew",
        padx=(5, 0)
    )

    # Store WAV label reference
    state.wav_label_ref = wav_label

    # Build Start button
    start_btn = tk.Button(
        parameters_frame,
        text="Start",
        command=start_algorithm,
        state=tk.DISABLED
    )

    start_btn.grid(
        row=8,
        column=0,
        columnspan=2,
        sticky="ew",
        pady=(5, 0)
    )

    # Build progress frame
    progress_frame = ttk.LabelFrame(
        parent,
        text="Progress"
    )

    progress_frame.grid(
        row=2,
        column=0,
        columnspan=2,
        sticky="ew",
        padx=5,
        pady=5
    )

    # Expand progress frame
    progress_frame.grid_columnconfigure(
        0,
        weight=1
    )

    # Build progress value
    state.progress_var = tk.DoubleVar(
        value=0.0
    )

    # Build progress bar
    state.progress_bar = ttk.Progressbar(
        progress_frame,
        maximum=100.0,
        variable=state.progress_var
    )

    state.progress_bar.grid(
        row=0,
        column=0,
        sticky="ew"
    )

    # Build ETA label
    state.eta_label = tk.Label(
        progress_frame,
        text="ETA --:--",
        font=default_font,
        anchor="w"
    )

    state.eta_label.grid(
        row=1,
        column=0,
        sticky="w"
    )

    # Build status label
    state.status_label = tk.Label(
        progress_frame,
        text="",
        font=default_font,
        anchor="w"
    )

    state.status_label.grid(
        row=2,
        column=0,
        sticky="ew"
    )

    # Build metrics frame
    metrics_frame = ttk.LabelFrame(
        parent,
        text="Metrics"
    )

    metrics_frame.grid(
        row=3,
        column=0,
        columnspan=2,
        sticky="ew",
        padx=5,
        pady=5
    )

    # Expand metric value column
    metrics_frame.grid_columnconfigure(
        1,
        weight=1
    )

    # Add execution-time label
    tk.Label(
        metrics_frame,
        text="Execution time (sec):",
        font=default_font
    ).grid(
        row=0,
        column=0,
        sticky="e"
    )

    # Build execution-time value
    exec_val = tk.Label(
        metrics_frame,
        text="-",
        font=default_font
    )

    exec_val.grid(
        row=0,
        column=1,
        sticky="w"
    )

    # Add convergence label
    tk.Label(
        metrics_frame,
        text="Convergence time (msec):",
        font=default_font
    ).grid(
        row=1,
        column=0,
        sticky="e"
    )

    # Build convergence value
    conv_val = tk.Label(
        metrics_frame,
        text="-",
        font=default_font
    )

    conv_val.grid(
        row=1,
        column=1,
        sticky="w"
    )

    # Add SSE label
    tk.Label(
        metrics_frame,
        text="Steady state error (dBr):",
        font=default_font
    ).grid(
        row=2,
        column=0,
        sticky="e"
    )

    # Build SSE value
    sse_val = tk.Label(
        metrics_frame,
        text="-",
        font=default_font
    )

    sse_val.grid(
        row=2,
        column=1,
        sticky="w"
    )

    # Add ANC OFF label
    tk.Label(
        metrics_frame,
        text="Power (ANC OFF):",
        font=default_font
    ).grid(
        row=3,
        column=0,
        sticky="e"
    )

    # Build ANC OFF value
    inpow_val = tk.Label(
        metrics_frame,
        text="-",
        font=default_font
    )

    inpow_val.grid(
        row=3,
        column=1,
        sticky="w"
    )

    # Add ANC ON label
    tk.Label(
        metrics_frame,
        text="Power (ANC ON):",
        font=default_font
    ).grid(
        row=4,
        column=0,
        sticky="e"
    )

    # Build ANC ON value
    outpow_val = tk.Label(
        metrics_frame,
        text="-",
        font=default_font
    )

    outpow_val.grid(
        row=4,
        column=1,
        sticky="w"
    )

    # Build playback frame
    playback_frame = ttk.LabelFrame(
        parent,
        text="Playback"
    )

    playback_frame.grid(
        row=4,
        column=0,
        columnspan=2,
        sticky="ew",
        padx=5,
        pady=5
    )

    # Use equal playback widths
    playback_frame.grid_columnconfigure(
        0,
        weight=1,
        uniform="playback_buttons"
    )

    playback_frame.grid_columnconfigure(
        1,
        weight=1,
        uniform="playback_buttons"
    )

    playback_frame.grid_columnconfigure(
        2,
        weight=1,
        uniform="playback_buttons"
    )

    # Build input playback button
    play_input_btn = tk.Button(
        playback_frame,
        text="Play Input",
        command=lambda: toggle_play(
            play_input_btn,
            state.play_input
        ),
        state=tk.DISABLED
    )

    play_input_btn.grid(
        row=0,
        column=0,
        sticky="ew"
    )

    # Build ANC OFF playback button
    play_anc_off_btn = tk.Button(
        playback_frame,
        text="Play ANC OFF",
        command=lambda: toggle_play(
            play_anc_off_btn,
            state.play_anc_off
        ),
        state=tk.DISABLED
    )

    play_anc_off_btn.grid(
        row=0,
        column=1,
        sticky="ew"
    )

    # Build ANC ON playback button
    play_anc_on_btn = tk.Button(
        playback_frame,
        text="Play ANC ON",
        command=lambda: toggle_play(
            play_anc_on_btn,
            state.play_anc_on
        ),
        state=tk.DISABLED
    )

    play_anc_on_btn.grid(
        row=0,
        column=2,
        sticky="ew"
    )

    # Build plots frame
    plots_frame = ttk.LabelFrame(
        parent,
        text="Plots"
    )

    plots_frame.grid(
        row=5,
        column=0,
        columnspan=2,
        sticky="nsew",
        padx=5,
        pady=5
    )

    # Use equal plot widths
    for column in range(4):
        plots_frame.grid_columnconfigure(
            column,
            weight=1,
            uniform="plot_buttons"
        )

    # Expand custom-band row
    plots_frame.grid_rowconfigure(
        3,
        weight=1
    )

    # Build filter-weight button
    fw_btn = tk.Button(
        plots_frame,
        text="Filter Weights",
        command=plot_filter,
        state=tk.DISABLED
    )

    fw_btn.grid(
        row=0,
        column=0,
        sticky="ew"
    )

    # Build primary-path button
    pp_btn = tk.Button(
        plots_frame,
        text="Primary Path",
        command=plot_primary_path_effect,
        state=tk.DISABLED
    )

    pp_btn.grid(
        row=0,
        column=1,
        sticky="ew"
    )

    # Build secondary-path button
    sp_btn = tk.Button(
        plots_frame,
        text="Secondary Path",
        command=plot_secondary_path_effect,
        state=tk.DISABLED
    )

    sp_btn.grid(
        row=0,
        column=2,
        sticky="ew"
    )

    # Build error-analysis button
    ea_btn = tk.Button(
        plots_frame,
        text="Error Analysis",
        command=plot_error,
        state=tk.DISABLED
    )

    ea_btn.grid(
        row=0,
        column=3,
        sticky="ew"
    )

    # Build signal-flow button
    sf_btn = tk.Button(
        plots_frame,
        text="Signal Flow",
        command=plot_signal,
        state=tk.DISABLED
    )

    sf_btn.grid(
        row=1,
        column=0,
        sticky="ew"
    )

    # Build noise-spectrogram button
    spec_btn = tk.Button(
        plots_frame,
        text="Noise Spectrogram",
        command=plot_noise_spec,
        state=tk.DISABLED
    )

    spec_btn.grid(
        row=1,
        column=1,
        sticky="ew"
    )

    # Build error-spectrogram button
    err_spec_btn = tk.Button(
        plots_frame,
        text="Error Spectrogram",
        command=plot_error_spec,
        state=tk.DISABLED
    )

    err_spec_btn.grid(
        row=1,
        column=2,
        sticky="ew"
    )

    # Build band-attenuation button
    band_btn = tk.Button(
        plots_frame,
        text="Band Attenuation",
        command=plot_band_attn,
        state=tk.DISABLED
    )

    band_btn.grid(
        row=1,
        column=3,
        sticky="ew"
    )

    # Build save button
    save_btn = tk.Button(
        plots_frame,
        text="Save Results",
        command=save_single_results,
        state=tk.DISABLED
    )

    save_btn.grid(
        row=2,
        column=0,
        columnspan=4,
        sticky="ew",
        pady=(4, 0)
    )

    # Add custom-band label
    tk.Label(
        plots_frame,
        text="Custom Bands (Hz):",
        font=default_font
    ).grid(
        row=3,
        column=0,
        sticky="ne"
    )

    # Build custom-band frame
    bands_frame = tk.Frame(
        plots_frame
    )

    bands_frame.grid(
        row=3,
        column=1,
        columnspan=3,
        sticky="nsew"
    )

    # Expand custom-band text
    bands_frame.grid_columnconfigure(
        0,
        weight=1
    )

    bands_frame.grid_rowconfigure(
        0,
        weight=1
    )

    # Build custom-band text
    state.bands_text = tk.Text(
        bands_frame,
        height=3,
        width=20
    )

    # Insert default bands
    state.bands_text.insert(
        tk.INSERT,
        (
            "0-500, 500-1000, "
            "1000-3000, 3000-5000, "
            "5000-10000"
        )
    )

    state.bands_text.grid(
        row=0,
        column=0,
        sticky="nsew"
    )

    # Add resize grip
    ttk.Sizegrip(
        bands_frame
    ).grid(
        row=1,
        column=0,
        sticky="se"
    )

    # Expand plots section
    parent.grid_rowconfigure(
        5,
        weight=1
    )

    # Store result buttons
    state.all_buttons.extend([
        play_input_btn,
        play_anc_off_btn,
        play_anc_on_btn,
        save_btn,
        fw_btn,
        pp_btn,
        sp_btn,
        ea_btn,
        sf_btn,
        spec_btn,
        err_spec_btn,
        band_btn
    ])

    # Expose Single Run callback
    state.start_single_run_cb = (
        start_algorithm
    )

    # Initialize source widgets
    parent.after(
        0,
        on_noise_source_change
    )

    # Initialize algorithm widgets
    parent.after(
        0,
        on_algorithm_change
    )