import os
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
import time
import threading
from engine.engine_multi import (
    build_grid,
    build_run_combinations,
    group_results_by_combo,
    run_multi_sim
)
from utils.logger import init_log
from utils.plot import plot_hparam_heatmap, plot_convtime_vs_mu, plot_sse_vs_L
from utils.time_utils import estimate_eta

def build_multi_ui(parent, state, default_font, header_font):
    ranked = None
    mu_vals = None
    L_vals = None

    # --- Multi-select controls state ---
    alg_options = ["LMS","NLMS","FxLMS","FxNLMS", "Neural Network"]
    alg_vars = {name: tk.BooleanVar(value=(name=="FxNLMS")) for name in alg_options}

    color_options = ["White","Pink","Brownian","Violet","Grey","Blue"]
    color_vars = {c: tk.BooleanVar(value=(c=="White")) for c in color_options}
    color_cbs = {} # store Checkbutton widgets by color name

    include_stationary_var = tk.BooleanVar(value=True)
    include_wav_var = tk.BooleanVar(value=False)

    mr_wav_paths = [] # list of selected WAV full paths

    def set_multi_action_buttons(enabled):
        # Select button state
        button_state = (
            tk.NORMAL
            if enabled
            else tk.DISABLED
        )

        # Store action buttons
        action_buttons = (
            run_best_btn,
            show_heatmap_btn,
            show_conv_btn,
            show_sse_btn
        )

        # Update every button
        for button in action_buttons:
            # Apply selected state
            button.config(state=button_state)

    def clear_best_metrics():
        # Clear best mu
        best_mu_val.config(text="μ:")

        # Clear best L
        best_L_val.config(text="L:")

        # Clear convergence metric
        best_conv_val.config(text="Convergence speed:")

        # Clear SSE metric
        best_sse_val.config(text="SSE:")

    # --- WAV selection (multi-file) ---
    def select_wav_files():
        nonlocal mr_wav_paths
        paths = filedialog.askopenfilenames(filetypes=[("WAV files", "*.wav")])
        if paths:
            mr_wav_paths = list(paths)
            names = [os.path.basename(p) for p in mr_wav_paths]
            mr_wav_label.config(text=", ".join(names) if names else "No file selected")
        validate_multi_ready()

    def read_multi_inputs(require_adaptive):
        # Read common inputs
        values = {
            "duration": float(
                mr_duration_entry
                .get()
                .strip()
            ),
            "alpha": float(
                alpha_entry
                .get()
                .strip()
            ),
            "mu_scale": mu_scale_var.get(),
            "save_mode": (
                save_mode_var
                .get()
                .lower()
            )
        }

        # Read adaptive parameter grid
        if require_adaptive:
            values.update({
                "mu_min": float(
                    mu_min_entry
                    .get()
                    .strip()
                ),
                "mu_max": float(
                    mu_max_entry
                    .get()
                    .strip()
                ),
                "mu_steps": int(
                    mu_steps_entry
                    .get()
                    .strip()
                ),
                "L_min": int(
                    L_min_entry
                    .get()
                    .strip()
                ),
                "L_max": int(
                    L_max_entry
                    .get()
                    .strip()
                ),
                "L_steps": int(
                    L_steps_entry
                    .get()
                    .strip()
                )
            })

        # Use placeholder grid for NN only
        else:
            values.update({
                "mu_min": 0.0,
                "mu_max": 0.0,
                "mu_steps": 1,
                "L_min": 0,
                "L_max": 0,
                "L_steps": 1
            })

        return values

    def read_multi_selection():
        # Read selected algorithms
        algorithms = [
            name
            for name, variable in alg_vars.items()
            if variable.get()
        ]

        # Read selected stationary noises
        colors = [
            name
            for name, variable in color_vars.items()
            if variable.get()
        ] if include_stationary_var.get() else []

        # Read selected WAV files
        wav_paths = (
            list(mr_wav_paths)
            if include_wav_var.get()
            else []
        )

        # Return selections
        return {
            "algorithms": algorithms,
            "colors": colors,
            "wav_paths": wav_paths
        }

    def validate_multi_ready(*_):
        # Keep Start disabled while GUI is locked
        if getattr(state, "is_locked", False):
            try:
                start_multi_btn.config(
                    state=tk.DISABLED
                )
            except Exception:
                pass

            return

        try:
            # Read selected options
            selection = read_multi_selection()

            # Check adaptive algorithm selection
            require_adaptive = any(
                algorithm != "Neural Network"
                for algorithm
                in selection["algorithms"]
            )

            # Read numeric inputs
            values = read_multi_inputs(
                require_adaptive
            )

            # Check common numeric inputs
            numeric_ok = (
                values["duration"] > 0.0
                and 0.0
                <= values["alpha"]
                <= 1.0
            )

            # Check adaptive grid
            if require_adaptive:
                numeric_ok = (
                    numeric_ok
                    and 0.0
                    < values["mu_min"]
                    <= values["mu_max"]
                    and values["mu_steps"] > 0
                    and 0
                    < values["L_min"]
                    <= values["L_max"]
                    and values["L_steps"] > 0
                )

            # Check selections
            selection_ok = (
                bool(selection["algorithms"])
                and bool(
                    selection["colors"]
                    or selection["wav_paths"]
                )
            )

            ok = (
                numeric_ok
                and selection_ok
            )

        except (TypeError, ValueError):
            ok = False

        try:
            start_multi_btn.config(
                state=(
                    tk.NORMAL
                    if ok
                    else tk.DISABLED
                )
            )
        except Exception:
            pass
    
    def toggle_source_widgets(*_):
        # WAV picker enabled only if WAV source checked
        try:
            mr_wav_btn.config(state=(tk.NORMAL if include_wav_var.get() else tk.DISABLED))
        except Exception:
            pass
        if not include_wav_var.get():
            mr_wav_paths.clear()
            try:
                mr_wav_label.config(text="No file selected")
            except Exception:
                pass

        # Stationary colors enabled only if Stationary checked
        st_on = bool(include_stationary_var.get())
        if not st_on:
            # untick all colors
            for v in color_vars.values():
                v.set(False)
        # disable/enable the color checkboxes
        for cb in color_cbs.values():
            try:
                cb.config(state=(tk.NORMAL if st_on else tk.DISABLED))
            except Exception:
                pass

        validate_multi_ready()

    # ---------- actions ----------
    def show_heatmap():
        nonlocal ranked, mu_vals, L_vals
        if ranked is None or mu_vals is None or L_vals is None:
            mr_status.config(text="No results to plot yet.", fg="red")
            return
        try:
            plot_hparam_heatmap(ranked, mu_vals, L_vals)
        except Exception as e:
            mr_status.config(text=f"Heatmap error: {e}", fg="red")

    def show_combo_plot(plot_function):
        # Check available results
        if not ranked:
            # Show missing result message
            mr_status.config(text="No results to plot.", fg="red")

            # Stop plotting
            return

        try:
            # Group results by combination
            grouped_results = group_results_by_combo(ranked)

            # Process every combination
            for key, rows in grouped_results.items():
                # Read combination values
                algorithm, _, noise_label = key

                # Skip Neural Network parameter plots
                if algorithm == "Neural Network":
                    continue

                # Create plot
                plot_function(
                    rows,
                    save_dir=None,
                    algorithm_name=algorithm,
                    noise_type=noise_label
                )

        except Exception as error:
            # Show plot error
            mr_status.config(
                text=f"Plot error: {error}",
                fg="red"
            )

    def show_conv_vs_mu():
        # Plot convergence against mu
        show_combo_plot(plot_convtime_vs_mu)

    def show_sse_vs_L():
        # Plot SSE against L
        show_combo_plot(plot_sse_vs_L)
    
    def run_best_from_multi():
        # Get best hyperparams and combo metadata
        best = state.last_best_combo

        if best is None:
            # Show missing result message
            mr_status.config(text="No best result yet. Run multi-run first.", fg="red")

            # Stop action
            return

        # copy μ, L
        state.mu_entry.delete(0, "end")
        state.mu_entry.insert(0, f"{best['mu']:.6g}")
        state.L_entry.delete(0, "end")
        state.L_entry.insert(0, f"{best['L']}")

        # algorithm
        state.algo_var.set(best["algorithm"])

        # noise settings -> Single Run panel
        if best["source"] == "Stationary":
            state.noise_source_var.set("Stationary")
            state.noise_var.set(best["noise_label"])
            state.wav_file_path.set("")
            
            if state.wav_label_ref is not None:
                state.wav_label_ref.config(text="No file selected")
        else:
            state.noise_source_var.set("WAV")
            state.noise_var.set("White")
            state.wav_file_path.set(best.get("wav_path",""))
            
            if state.wav_label_ref is not None:
                fname = os.path.basename(best.get("wav_path",""))
                state.wav_label_ref.config(text=fname)

        # duration (mirror multi-run)
        try:
            dur_txt = mr_duration_entry.get().strip()
            if dur_txt:
                state.duration_entry.delete(0, "end")
                state.duration_entry.insert(0, dur_txt)
        except Exception:
            pass

        if state.start_single_run_cb:
            state.start_single_run_cb()

    def update_multi_progress(
        completed,
        total,
        start_time
    ):
        # Compute progress percentage
        percentage = (100.0 * completed / total)

        # Update progress bar
        mr_progress_var.set(percentage)

        # Update status and ETA
        mr_status.config(
            text=(
                f"{completed}/{total} — "
                f"{estimate_eta(start_time, completed, total)}"
            ),
            fg="black"
        )

    def update_save_progress(
        completed,
        total
    ):
        # Compute save percentage
        percentage = (100.0 * completed / total)

        # Update progress bar
        mr_progress_var.set(percentage)

        # Update status
        mr_status.config(
            text=f"Saving {completed}/{total}…",
            fg="black"
        )

    def show_multi_error(message):
        # Unlock GUI
        state.unlock_ui()

        # Show error message
        mr_status.config(text=message, fg="red")

        # Validate GUI fields
        validate_multi_ready()

    def start_multi_run():
        nonlocal mu_vals, L_vals, ranked

        # Read validated selections
        selection = read_multi_selection()

        # Read selected values
        sel_algs = selection["algorithms"]
        sel_cols = selection["colors"]
        sel_wavs = selection["wav_paths"]

        # Check whether adaptive algorithms are selected
        require_adaptive = any(
            algorithm != "Neural Network"
            for algorithm in sel_algs
        )

        # Read validated numeric inputs
        values = read_multi_inputs(
            require_adaptive
        )

        # Read common values
        dur = values["duration"]
        mode = values["save_mode"]
        alpha = values["alpha"]

        # Build adaptive parameter grid
        if require_adaptive:
            mu_vals, L_vals, muL = build_grid(
                mu_min=values["mu_min"],
                mu_max=values["mu_max"],
                mu_steps=values["mu_steps"],
                L_min=values["L_min"],
                L_max=values["L_max"],
                L_steps=values["L_steps"],
                mu_scale=values["mu_scale"]
            )

        # Use placeholder grid for NN only
        else:
            mu_vals = np.array(
                [0.0],
                dtype=float
            )

            L_vals = np.array(
                [0],
                dtype=int
            )

            muL = [(0.0, 0)]

        # Read Neural Network settings
        nn_checkpoint_path = None
        nn_backend = (
            state.nn_backend_var
            .get()
            .lower()
        )

        # Validate Neural Network checkpoint
        if "Neural Network" in sel_algs:
            # Read selected checkpoint
            nn_checkpoint_path = (
                state.nn_checkpoint_path_var
                .get()
                .strip()
            )

            # Use default best checkpoint
            if not nn_checkpoint_path:
                nn_checkpoint_path = os.path.join(
                    state.nn_processed_root_var.get(),
                    "checkpoints",
                    "best.pt"
                )

            # Check checkpoint file
            if not os.path.exists(
                nn_checkpoint_path
            ):
                mr_status.config(
                    text=(
                        f"Checkpoint not found: "
                        f"{nn_checkpoint_path}"
                    ),
                    fg="red"
                )
                return

            # Check ONNX model
            if nn_backend == "onnx":
                onnx_path = (
                    os.path.splitext(
                        nn_checkpoint_path
                    )[0]
                    + ".onnx"
                )

                if not os.path.exists(onnx_path):
                    mr_status.config(
                        text=(
                            f"ONNX model not found: "
                            f"{onnx_path}"
                        ),
                        fg="red"
                    )
                    return

        # Initialize common ANC log
        init_log(
            run_kind="multi",
            clear=True
        )

        # Build simulation combinations
        combos = build_run_combinations(
            sel_algs,
            sel_cols,
            sel_wavs
        )

        # Count adaptive and NN combinations
        adaptive_combo_count = sum(
            combination[0] != "Neural Network"
            for combination in combos
        )

        neural_combo_count = sum(
            combination[0] == "Neural Network"
            for combination in combos
        )

        # Count simulations
        total = (
            len(muL) * adaptive_combo_count
            + neural_combo_count
        )

        # UI prep
        start_multi_btn.config(state=tk.DISABLED)
        set_multi_action_buttons(False)
        mr_status.config(text=f"Queued {total} simulations…", fg="black")
        mr_progress_var.set(0.0)
        state.lock_ui()

        # Worker thread: run parallel with live progress
        def worker():
            # Start GUI timer
            start_t = time.time()

            # Build results folder
            results_root = os.path.join(
                os.getcwd(),
                "results"
            )

            # Forward engine progress to GUI
            def progress_callback(
                completed,
                total
            ):
                # Schedule GUI update
                state.ui_call(
                    update_multi_progress,
                    completed,
                    total,
                    start_t
                )

            def save_progress_callback(
                completed,
                total
            ):
                # Send save progress to GUI
                state.ui_call(
                    update_save_progress,
                    completed,
                    total
                )

            try:
                # Run Multi Run
                sim_result = run_multi_sim(
                    grid=muL,
                    combinations=combos,
                    mu_values=mu_vals,
                    L_values=L_vals,
                    duration=dur,
                    alpha=alpha,
                    save_mode=mode,
                    results_root=results_root,
                    paths=state.anc_paths,
                    progress_callback=progress_callback,
                    save_progress_callback=save_progress_callback,
                    nn_checkpoint_path=nn_checkpoint_path,
                    nn_backend=nn_backend
                )

                # Read ranked results
                ranked_local = sim_result["ranked"]

                # Read unique combination count
                unique_combo_count = sim_result["unique_combo_count"]

                # Read execution time
                elapsed = sim_result["execution_time"]

            except Exception as error:
                # Show engine error
                state.ui_call(
                    show_multi_error,
                    str(error)
                )

                # Stop worker
                return

            def ui_done():
                # Access outer ranked variable
                nonlocal ranked

                # Store ranked results
                ranked = ranked_local
                
                if not ranked:
                    # Clear stored combination
                    state.last_best_combo = None

                    # Clear displayed metrics
                    clear_best_metrics()

                    # Show error status
                    mr_status.config(text="No valid results.", fg="red")

                    # Unlock GUI
                    state.unlock_ui()

                    # Refresh Start button
                    validate_multi_ready()

                    # Stop completion
                    return

                # Store complete best result
                best = ranked[0]
                state.last_best_combo = best

                is_neural = (
                    best["algorithm"]
                    == "Neural Network"
                )

                # Check if run contains one combination
                single_combo = (unique_combo_count == 1)
                
                # Show metrics for one combination
                if single_combo:
                    if is_neural:
                        best_mu_val.config(
                            text="μ: N/A"
                        )

                        best_L_val.config(
                            text="L: N/A"
                        )

                        best_conv_val.config(
                            text="Convergence speed: N/A"
                        )

                    else:
                        best_mu_val.config(
                            text=f"μ: {best['mu']:.6g}"
                        )

                        best_L_val.config(
                            text=f"L: {best['L']:d}"
                        )

                        best_conv_val.config(
                            text=(
                                f"Convergence speed: "
                                f"{best['conv_ms']:.2f} ms"
                            )
                        )

                    best_sse_val.config(
                        text=(
                            f"SSE: "
                            f"{best['sse_db']:.2f} dBr"
                        )
                    )

                else:
                    clear_best_metrics()
                
                mr_exec_label.config(text=f"Execution time (sec): {elapsed:.2f}")
                
                # Unlock GUI
                state.unlock_ui()

                # Refresh Start button
                validate_multi_ready()

                # Enable result actions
                set_multi_action_buttons(single_combo)

                # Disable parameter plots for Neural Network
                if single_combo and is_neural:
                    show_heatmap_btn.config(
                        state=tk.DISABLED
                    )

                    show_conv_btn.config(
                        state=tk.DISABLED
                    )

                    show_sse_btn.config(
                        state=tk.DISABLED
                    )

                # Complete progress bar
                mr_progress_var.set(100.0)

                # Show All result status
                if mode == "all":
                    # Show saved path
                    mr_status.config(
                        text=f"Done. Results saved to: {results_root}",
                        fg="green"
                    )

                # Show Best result status
                elif mode == "best":
                    # Show saved path
                    mr_status.config(
                        text=f"Saved best results to: {results_root}",
                        fg="green"
                    )

                elif single_combo and is_neural:
                    mr_status.config(
                        text=(
                            f"Done. Neural Network: "
                            f"sse={best['sse_db']:.2f} dBr"
                        ),
                        fg="green"
                    )

                # Show no-save result status
                else:
                    # Show best result
                    mr_status.config(
                        text=(
                            f"Done. Best: "
                            f"L={best['L']}, "
                            f"μ={best['mu']:.6g}, "
                            f"score={best['score']:.3f}, "
                            f"conv={best['conv_ms']:.2f} ms, "
                            f"sse={best['sse_db']:.2f} dBr"
                        ),
                        fg="green"
                    )

            state.ui_call(ui_done)

        threading.Thread(target=worker, daemon=True).start()
    
    def on_alpha_change(*_):
        # Read alpha text
        text = alpha_entry.get().strip()

        try:
            # Convert alpha
            alpha = float(text)

            # Check alpha range
            if not 0.0 <= alpha <= 1.0:
                raise ValueError

            # Show score formula
            alpha_info.config(
                text=(
                    f"Preference = "
                    f"{alpha:.2f}*Convergence + "
                    f"{1.0 - alpha:.2f}*SSE"
                ),
                fg="black"
            )

        except Exception:
            # Show invalid alpha
            alpha_info.config(
                text="Preference factor must be between 0 and 1",
                fg="red"
            )

        # Refresh Start button
        validate_multi_ready()

    # ---------- UI ----------
    tk.Label(parent, text="Multi-Run", font=header_font).grid(row=0, column=0, columnspan=2, sticky="w")

    # --- Algorithms (multi-select) ---
    tk.Label(parent, text="Algorithms:", font=default_font).grid(row=1, column=0, sticky="ne")
    alg_frame = tk.Frame(parent)
    alg_frame.grid(row=1, column=1, sticky="w")
    for i, name in enumerate(alg_options):
        tk.Checkbutton(alg_frame, text=name, variable=alg_vars[name],
                    command=validate_multi_ready).grid(row=0, column=i, sticky="w")

    # Duration
    tk.Label(parent, text="Duration (s):", font=default_font).grid(row=2, column=0, sticky="e")
    mr_duration_entry = tk.Entry(parent, width=10)
    mr_duration_entry.grid(row=2, column=1, sticky="w")
    mr_duration_entry.bind("<KeyRelease>", validate_multi_ready)

    # --- Sources to include ---
    tk.Label(parent, text="Sources:", font=default_font).grid(row=3, column=0, sticky="ne")
    srcs = tk.Frame(parent)
    srcs.grid(row=3, column=1, sticky="w")
    tk.Checkbutton(srcs, text="Stationary", variable=include_stationary_var,
               command=toggle_source_widgets).pack(side="left")
    tk.Checkbutton(srcs, text="WAV", variable=include_wav_var,
               command=toggle_source_widgets).pack(side="left")

    # --- Stationary colors (multi-select) ---
    tk.Label(parent, text="Noise Colors:", font=default_font).grid(row=4, column=0, sticky="ne")
    col_frame = tk.Frame(parent)
    col_frame.grid(row=4, column=1, sticky="w")

    for i, c in enumerate(color_options):
        cb = tk.Checkbutton(col_frame, text=c, variable=color_vars[c],
                            command=validate_multi_ready)
        cb.grid(row=0, column=i, sticky="w")
        color_cbs[c] = cb

    tk.Label(parent, text="Noise WAV(s):", font=default_font).grid(row=5, column=0, sticky="e")
    wav_select_frame = tk.Frame(parent); wav_select_frame.grid(row=5, column=1, columnspan=2, sticky="ew")
    mr_wav_btn = tk.Button(wav_select_frame, text="Select WAVs", command=select_wav_files, state=tk.DISABLED)
    mr_wav_btn.grid(row=0, column=0, sticky="w")
    mr_wav_label = tk.Label(wav_select_frame, text="No file selected", font=default_font)
    mr_wav_label.grid(row=0, column=1, sticky="w")
    parent.after(0, toggle_source_widgets)

    # μ grid
    tk.Label(parent, text="μ min:", font=default_font).grid(row=6, column=0, sticky="e")
    mu_min_entry = tk.Entry(parent, width=10); mu_min_entry.grid(row=6, column=1, sticky="w")
    mu_min_entry.bind("<KeyRelease>", validate_multi_ready)

    tk.Label(parent, text="μ max:", font=default_font).grid(row=7, column=0, sticky="e")
    mu_max_entry = tk.Entry(parent, width=10); mu_max_entry.grid(row=7, column=1, sticky="w")
    mu_max_entry.bind("<KeyRelease>", validate_multi_ready)

    tk.Label(parent, text="μ steps:", font=default_font).grid(row=8, column=0, sticky="e")
    mu_steps_entry = tk.Entry(parent, width=10); mu_steps_entry.grid(row=8, column=1, sticky="w")
    mu_steps_entry.bind("<KeyRelease>", validate_multi_ready)

    tk.Label(parent, text="μ scale:", font=default_font).grid(row=9, column=0, sticky="e")
    mu_scale_var = tk.StringVar(value="log")
    mu_scale_menu = ttk.Combobox(parent, textvariable=mu_scale_var, values=["log","linear"], state="readonly", width=8)
    mu_scale_menu.grid(row=9, column=1, sticky="w")

    # L grid
    tk.Label(parent, text="L min:", font=default_font).grid(row=10, column=0, sticky="e")
    L_min_entry = tk.Entry(parent, width=10); L_min_entry.grid(row=10, column=1, sticky="w")
    L_min_entry.bind("<KeyRelease>", validate_multi_ready)

    tk.Label(parent, text="L max:", font=default_font).grid(row=11, column=0, sticky="e")
    L_max_entry = tk.Entry(parent, width=10); L_max_entry.grid(row=11, column=1, sticky="w")
    L_max_entry.bind("<KeyRelease>", validate_multi_ready)

    tk.Label(parent, text="L steps:", font=default_font).grid(row=12, column=0, sticky="e")
    L_steps_entry = tk.Entry(parent, width=10); L_steps_entry.grid(row=12, column=1, sticky="w")
    L_steps_entry.bind("<KeyRelease>", validate_multi_ready)

    # μ–L trade-off
    tk.Label(parent, text="Metric trade-off factor a:", font=default_font).grid(row=13, column=0, sticky="e")
    alpha_entry = tk.Entry(parent, width=10)
    alpha_entry.insert(0, "0.5")
    alpha_entry.grid(row=13, column=1, sticky="w")
    alpha_entry.bind("<KeyRelease>", on_alpha_change)

    alpha_info = tk.Label(parent, text="Preference = a * Conv_time + (1-a) * SSE", font=default_font, anchor="w")
    alpha_info.grid(row=14, column=0, columnspan=2, sticky="w")

    # --- Save Results mode ---
    tk.Label(parent, text="Save Results:", font=default_font).grid(row=15, column=0, sticky="e")
    save_mode_var = tk.StringVar(value="All")
    save_frame = tk.Frame(parent)
    save_frame.grid(row=15, column=1, sticky="w")
    for txt in ["None","Best","All"]:
        tk.Radiobutton(save_frame, text=txt, variable=save_mode_var, value=txt,
                    command=validate_multi_ready).pack(side="left")

    # Progress + status (multi-run)
    mr_progress_var = tk.DoubleVar(value=0.0)
    mr_progress = ttk.Progressbar(parent, maximum=100.0, variable=mr_progress_var)
    mr_progress.grid(row=16, column=0, columnspan=2, sticky="ew")

    mr_status = tk.Label(parent, text="", font=default_font, anchor="w")
    mr_status.grid(row=17, column=0, columnspan=2, sticky="w")

    mr_exec_label = tk.Label(parent, text="Execution time (sec): ", font=default_font, anchor="w")
    mr_exec_label.grid(row=18, column=0, columnspan=2, sticky="w")

    # Best metrics rows
    best_metrics_frame = tk.Frame(parent)
    best_metrics_frame.grid(row=19, column=0, rowspan=2, columnspan=2, sticky="ew")

    best_mu_val = tk.Label(best_metrics_frame, text="μ:", font=default_font)
    best_mu_val.grid(row=0, column=0, sticky="w")

    best_L_val = tk.Label(best_metrics_frame, text="L:", font=default_font)
    best_L_val.grid(row=0, column=1, sticky="w")

    best_conv_val = tk.Label(best_metrics_frame, text="Convergence speed:", font=default_font)
    best_conv_val.grid(row=1, column=0, sticky="w")

    best_sse_val = tk.Label(best_metrics_frame, text="SSE:", font=default_font)
    best_sse_val.grid(row=1, column=1, sticky="w")

    # Start / Run Best
    start_multi_btn = tk.Button(parent, text="Start Multi-Run", command=start_multi_run, state=tk.DISABLED)
    start_multi_btn.grid(row=21, column=0, columnspan=2, sticky="ew")
    run_best_btn   = tk.Button(parent, text="Run Best (from Multi-Run)",
                               command=run_best_from_multi, state=tk.DISABLED)
    run_best_btn.grid(row=22, column=0, columnspan=2, sticky="ew")

    plot_row = tk.Frame(parent)
    plot_row.grid(row=23, column=0, columnspan=3, sticky="ew")
    plot_row.grid_columnconfigure(0, weight=1)
    plot_row.grid_columnconfigure(1, weight=1)
    plot_row.grid_columnconfigure(2, weight=1)

    show_heatmap_btn = tk.Button(plot_row, text="Plot Heatmap", command=show_heatmap, state=tk.DISABLED)
    show_heatmap_btn.grid(row=0, column=0, sticky="ew")

    show_conv_btn = tk.Button(plot_row, text="Plot Convergence vs μ", command=show_conv_vs_mu, state=tk.DISABLED)
    show_conv_btn.grid(row=0, column=1, sticky="ew")

    show_sse_btn = tk.Button(plot_row, text="Plot SSE vs L", command=show_sse_vs_L, state=tk.DISABLED)
    show_sse_btn.grid(row=0, column=2, sticky="ew")