import tkinter as tk
from tkinter import ttk, filedialog
import threading
import gc
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
from utils.logger import init_log#, log_case
from engine.engine_single import run_anc
from utils.plot import (
    plot_filter_weights, plot_path_analysis, plot_error_analysis, plot_signal_flow,
    plot_noise_spectrogram, plot_error_spectrogram, plot_band_attenuation
)
from utils.audio import play_audio, stop_audio, save_wav

def build_single_ui(parent, state, default_font, header_font):
    # local playback state
    is_playing = False
    mu = None
    L = None
    duration = None
    algorithm = None
    noise_type = None
    wav_file_path = None
    
    # ---------------- helpers ----------------

    def disable_buttons():
        for b in state.all_buttons:
            try:
                b.config(state=tk.DISABLED)
            except:
                pass

    def enable_buttons():
        for b in state.all_buttons:
            try:
                b.config(state=tk.NORMAL)
            except:
                pass

    def reset_result_labels():
        conv_val.config(text="-")
        sse_val.config(text="-")
        inpow_val.config(text="-")
        outpow_val.config(text="-")
        exec_val.config(text="-")
        state.status_label.config(text="", fg="black")

    def reset_sim_state():
        state.stored_reference_signal = None
        state.stored_noisy_signal = None
        state.stored_signal_after_primary = None
        state.stored_signal_after_secondary = None
        state.stored_error_signal = None
        state.stored_t = None
        state.stored_initial_weights = None
        state.stored_final_weights = None
        state.stored_primary_ir = None
        state.stored_secondary_ir = None
        state.stored_in_power = None
        state.stored_out_power = None
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

    def select_wav_file():
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        
        if path:
            try: stop_audio()
            except: pass
            try: plt.close('all')
            except: pass
            gc.collect()
            reset_sim_state()
            state.wav_file_path.set(path)
            wav_label.config(text=path.split("/")[-1])
            state.status_label.config(text="WAV selected. Press Start to run.", fg="black")
        
        validate_single_ready()

    def on_anc_complete(reference_signal, noisy_signal, error_signal,
                    t, fs, exec_time, conv_time, steady_error_db,
                    initial_weights, final_weights, primary_ir, secondary_ir,
                    signal_after_primary, signal_after_secondary,
                    in_power, out_power,
                    before_signal_raw, after_signal_raw):
        # store
        state.stored_reference_signal = reference_signal
        state.stored_noisy_signal = noisy_signal
        state.stored_error_signal = error_signal
        state.stored_signal_after_primary = signal_after_primary
        state.stored_signal_after_secondary = signal_after_secondary
        state.stored_fs = fs
        state.stored_t = t
        state.stored_initial_weights = initial_weights
        state.stored_final_weights = final_weights
        state.stored_primary_ir = primary_ir
        state.stored_secondary_ir = secondary_ir
        state.stored_in_power = in_power
        state.stored_out_power = out_power
        state.stored_execution_time = exec_time
        state.stored_convergence_speed = conv_time
        state.stored_steady_state_error = steady_error_db
        # raw mic-level (before/after)
        state.stored_before_signal_raw = before_signal_raw
        state.stored_after_signal_raw  = after_signal_raw

        conv_val.config(text=f"{0.0 if conv_time is None else float(conv_time):.2f} ms")
        sse_val.config(text=f"{float(steady_error_db):.2f} dB")
        inpow_val.config(text=f"{float(in_power):.3f}")
        outpow_val.config(text=f"{float(out_power):.3f}")
        exec_val.config(text=f"{float(exec_time):.2f} s")
        state.status_label.config(text="Done.", fg="green")
        state.progress_var.set(100.0)
        state.progress_bar.update_idletasks()

        # Prepare normalized copies for playback to avoid clipping; equal scale for before/after
        before_signal_raw = np.nan_to_num(before_signal_raw, nan=0.0, posinf=0.0, neginf=0.0)
        after_signal_raw  = np.nan_to_num(after_signal_raw,  nan=0.0, posinf=0.0, neginf=0.0)

        max_abs = max(float(np.max(np.abs(before_signal_raw))),
                        float(np.max(np.abs(after_signal_raw))), 1e-6)
        scale = min(1.0, 0.99 / max_abs)
        state.play_before = np.clip(before_signal_raw * scale, -1.0, 1.0).astype(np.float32)
        state.play_after = np.clip(after_signal_raw * scale, -1.0, 1.0).astype(np.float32)

        try:
            alg = state.algo_var.get()
            src = state.noise_source_var.get()
            nlabel = (os.path.basename(state.wav_file_path.get()) if src=="WAV" else state.noise_var.get())
            from utils.logger import log_case
            log_case(stage="single", status="ok",
                    algorithm=alg, source=src, noise_label=nlabel,
                    L=int(state.L_entry.get()), mu=float(state.mu_entry.get()),
                    conv_ms=state.stored_convergence_speed, sse_db=state.stored_steady_state_error,
                    exec_time=state.stored_execution_time,
                    in_power=state.stored_in_power, out_power=state.stored_out_power,
                    save_path="", message="")
        except Exception:
            pass

        # enable plot/audio after a run
        state.unlock_ui()
        enable_buttons()
        on_noise_source_change()

    def start_algorithm():
        nonlocal L, mu, duration, algorithm, noise_type, wav_file_path

        try:
            L  = int(state.L_entry.get())
            mu = float(state.mu_entry.get())
            duration = float(state.duration_entry.get())
        except Exception as e:
            state.status_label.config(text=f"Input error: {e}", fg="red"); return

        stop_audio()
        try:
            plt.close('all')
        except:
            pass
        disable_buttons()
        gc.collect()
        reset_result_labels()
        start_btn.config(state=tk.DISABLED)
        state.status_label.config(text="Running…", fg="black")
        state.progress_var.set(0)
        state.progress_bar.update_idletasks()

        state.single_start_time = time.time()
        state.eta_label.config(text="ETA --:--")
        
        algorithm = state.algo_var.get()

        if state.noise_source_var.get() == "WAV":
            noise_type = os.path.basename(state.wav_file_path.get())
        else:
            noise_type = state.noise_var.get()
        
        state.lock_ui()

        init_log(run_kind="single", clear=True, log_dir=os.path.join(os.getcwd(), "results"))

        def progress_cb(pct):
            state.ui_call(update_progress, pct)

        def completion_cb(*args):
            state.ui_call(on_anc_complete, *args)

        threading.Thread(
            target=run_anc,
            args=(
                algorithm, L, mu, state.noise_source_var.get(),
                noise_type, state.wav_file_path.get(),
                duration, progress_cb, completion_cb
            ),
            kwargs={"metrics_callback": None},
            daemon=True
        ).start()

    def validate_single_ready(*_):
        """Enable Start only when all required fields are filled and valid."""
        if getattr(state, "is_locked", False):
            try:
                start_btn.config(state=tk.DISABLED)
            except Exception:
                pass
            return
        
        ok = True
        # Algorithm must be selected
        ok &= bool(state.algo_var.get())
        # Numeric fields
        for e in (state.L_entry, state.mu_entry, state.duration_entry):
            if e is None:
                ok = False
                break
            txt = e.get().strip()
            if not txt:
                ok = False
                break
        if ok:
            try:
                int(state.L_entry.get())
                float(state.mu_entry.get())
                float(state.duration_entry.get())
            except Exception:
                ok = False
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
                elapsed = time.time() - state.single_start_time
                remaining = elapsed * (100.0 - float(pct)) / float(pct)
                m, s = divmod(int(max(0, remaining)), 60)
                state.eta_label.config(text=f"ETA {m:02d}:{s:02d}")
            else:
                state.eta_label.config(text="ETA --:--")
        except Exception:
            pass

    def reset_play_buttons():
        nonlocal is_playing

        is_playing = False
        play_before_btn.config(state=tk.NORMAL, text="Play Input")
        play_after_btn.config(state=tk.NORMAL, text="Play Output")

    def toggle_play_before():
        nonlocal is_playing
        if is_playing and play_before_btn["text"] == "Stop playing":
            stop_audio()
            is_playing = False
            reset_play_buttons()
            state.unlock_ui()
            return
        is_playing = True
        play_before_btn.config(text="Stop playing", state=tk.NORMAL)
        state.lock_ui(allow_widgets=(play_before_btn,))
        def _runner():
            play_audio(state.play_before, sample_rate=int(state.stored_fs))
            state.ui_call(reset_play_buttons)
            state.ui_call(state.unlock_ui)
        threading.Thread(target=_runner, daemon=True).start()

    def toggle_play_after():
        nonlocal is_playing
        
        if is_playing and play_after_btn["text"] == "Stop playing":
            stop_audio()
            is_playing = False
            reset_play_buttons()
            state.unlock_ui()
            return
        is_playing = True
        play_after_btn.config(text="Stop playing", state=tk.NORMAL)
        state.lock_ui(allow_widgets=(play_after_btn,))
        def _runner():
            play_audio(state.play_after, sample_rate=int(state.stored_fs))
            state.ui_call(reset_play_buttons)
            state.ui_call(state.unlock_ui)
        threading.Thread(target=_runner, daemon=True).start()
    
    def plot_filter(save_dir=None):
        nonlocal mu, L, algorithm, noise_type

        plot_filter_weights(
            fs=state.stored_fs, w_final=state.stored_final_weights,
            algorithm_name=algorithm, mu=mu, L=L, noise_type=noise_type,
            convergence_time=state.stored_convergence_speed,
            steady_state_error=state.stored_steady_state_error, save_dir=save_dir
        )

    def plot_primary_path_effect(save_dir=None):
        nonlocal mu, L, algorithm, noise_type

        plot_path_analysis(
            state.stored_primary_ir, state.stored_noisy_signal, state.stored_signal_after_primary, state.stored_fs,
            title_prefix="Primary", algorithm_name=algorithm, mu=mu, L=L, noise_type=noise_type,
            convergence_time=state.stored_convergence_speed, steady_state_error=state.stored_steady_state_error, save_dir=save_dir
        )

    def plot_secondary_path_effect(save_dir=None):
        nonlocal mu, L, algorithm, noise_type

        if state.stored_signal_after_secondary is None:
            state.status_label.config(text="Secondary path not available", fg="red")
            return
        
        plot_path_analysis(
            state.stored_secondary_ir, state.stored_noisy_signal, state.stored_signal_after_secondary, state.stored_fs,
            title_prefix="Secondary", algorithm_name=algorithm, mu=mu, L=L, noise_type=noise_type,
            convergence_time=state.stored_convergence_speed, steady_state_error=state.stored_steady_state_error, save_dir=save_dir
        )

    def plot_error(save_dir=None):
        nonlocal mu, L, algorithm, noise_type

        plot_error_analysis(
            state.stored_error_signal, state.stored_t, state.stored_fs,
            passive_cancelling=state.stored_before_signal_raw, noisy_signal=state.stored_noisy_signal,
            algorithm_name=algorithm, mu=mu, L=L, noise_type=noise_type,
            convergence_time=state.stored_convergence_speed,
            steady_state_error=state.stored_steady_state_error, save_dir=save_dir
        )

    def plot_signal(save_dir=None):
        nonlocal mu, L, algorithm, noise_type

        plot_signal_flow(
            state.stored_reference_signal, state.stored_noisy_signal, state.stored_error_signal, state.stored_t,
            algorithm_name=algorithm, mu=mu, L=L, noise_type=noise_type,
            convergence_time=state.stored_convergence_speed, steady_state_error=state.stored_steady_state_error, save_dir=save_dir
        )
    
    def plot_noise_spec(save_dir=None):
        
        if state.stored_noisy_signal is None:
            state.status_label.config(text="No noise signal available.", fg="red")
            return
        
        plot_noise_spectrogram(state.stored_noisy_signal, state.stored_fs, save_dir=save_dir)
    
    def plot_error_spec(save_dir=None):
        
        if state.stored_error_signal is None:
            state.status_label.config(text="No error signal available.", fg="red")
            return
        
        plot_error_spectrogram(state.stored_error_signal, state.stored_fs, save_dir=save_dir)

    def plot_band_attn(save_dir=None):
        nonlocal mu, L, algorithm, noise_type
        
        if state.stored_signal_after_primary is None or state.stored_error_signal is None:
            state.status_label.config(text="Run a simulation first.", fg="red")
            return
        
        bands_str = state.bands_text.get("1.0", "end-1c").strip()
        plot_band_attenuation(state.stored_before_signal_raw, state.stored_after_signal_raw,
                              state.stored_fs, bands_str=bands_str, algorithm_name=algorithm,
                              mu=mu, L=L, noise_type=noise_type, convergence_time=state.stored_convergence_speed,
                              steady_state_error=state.stored_steady_state_error, save_dir=save_dir)
    
    def write_metrics(save_dir=None):
        nonlocal mu, L, algorithm, noise_type
        meta = dict(algorithm=algorithm, L=L, mu=mu, noise_label=noise_type,
                    fs=int(state.stored_fs),
                    exec_time=round(float(state.stored_execution_time or 0.0), 2),
                    conv_ms=round(float(0.0 if state.stored_convergence_speed is None else state.stored_convergence_speed), 2),
                    sse_db=round(float(state.stored_steady_state_error), 2),
                    in_power=round(float(state.stored_in_power or 0.0), 3),
                    out_power=round(float(state.stored_out_power or 0.0), 3))
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(meta, f, indent=2)
    
    # --- Save Results (single run) ---
    def save_single_results():
        try:
            alg = state.algo_var.get()
            L_local  = int(state.L_entry.get())
            mu_local = float(state.mu_entry.get())
            nlabel = (os.path.basename(state.wav_file_path.get())
                    if state.noise_source_var.get()=="WAV"
                    else state.noise_var.get())
            safe_noise = "".join(c for c in nlabel if c.isalnum() or c in (" ","-","_")).strip().replace(" ","_")
            base_root = os.path.join(os.getcwd(), "results", alg, safe_noise)
            base = os.path.join(base_root, f"L{int(L_local)}_mu{float(mu_local):.6g}")
            os.makedirs(base, exist_ok=True)
        except Exception as e:
            state.status_label.config(text=f"Save failed: {e}", fg="red")
            return

        # build the todo list
        jobs = []

        # 1) metrics.json
        jobs.append(("metrics", lambda: write_metrics(save_dir=base)))

        # 2) audio WAV
        jobs.append(("audio_wav", lambda: save_wav(state.stored_before_signal_raw, state.stored_after_signal_raw,
                                                   state.stored_fs, base)))

        # 3) figures
        jobs.append(("filter_weights", lambda: plot_filter(save_dir=base)))
        jobs.append(("primary_path", lambda: plot_primary_path_effect(save_dir=base)))
        if state.stored_signal_after_secondary is not None:
            jobs.append(("secondary_path", lambda: plot_secondary_path_effect(save_dir=base)))
        jobs.append(("error_analysis", lambda: plot_error(save_dir=base)))
        jobs.append(("signal_flow", lambda: plot_signal(save_dir=base)))
        jobs.append(("band_attenuation", lambda: plot_band_attn(save_dir=base)))
        jobs.append(("noise_spectrogram", lambda: plot_noise_spec(save_dir=base)))
        jobs.append(("error_spectrogram", lambda: plot_error_spec(save_dir=base)))

        total = len(jobs)
        state.status_label.config(text=f"Saving 0/{total}…", fg="black")
        #state.disable_buttons_cb()
        disable_buttons()
        state.lock_ui()

        def _worker():
            done = 0
            for name, fn in jobs:
                try:
                    fn()
                except Exception as e:
                    # If error, report and continue with next job
                    msg = f"{name} failed: {e}"
                    state.ui_call(state.status_label.config, text=msg, fg="red")
                done += 1
                state.ui_call(state.status_label.config, text=f"Saving {done}/{total}…", fg="black")
            def _finish():
                state.unlock_ui()
                #state.enable_buttons_cb()
                enable_buttons()
                state.status_label.config(text=f"Saved to: {base}", fg="green")
            state.ui_call(_finish)
        threading.Thread(target=_worker, daemon=True).start()
    
    # ---------------- UI ----------------
    # Title
    tk.Label(parent, text="Single Run", font=header_font).grid(row=0, column=0, columnspan=2, sticky="w")

    # Algorithm
    tk.Label(parent, text="Algorithm:", font=default_font).grid(row=1, column=0, sticky="e")
    algo_menu = ttk.Combobox(parent, textvariable=state.algo_var, values=["LMS","NLMS","FxLMS","FxNLMS"], state="readonly", width=10)
    algo_menu.grid(row=1, column=1, sticky="w")
    algo_menu.bind("<<ComboboxSelected>>", validate_single_ready)

    # μ first (row=2)
    tk.Label(parent, text="μ:", font=default_font).grid(row=2, column=0, sticky="e")
    state.mu_entry = tk.Entry(parent, width=10)
    state.mu_entry.grid(row=2, column=1, sticky="w")
    state.mu_entry.bind("<KeyRelease>", validate_single_ready)

    # L next (row=3)
    tk.Label(parent, text="L (taps):", font=default_font).grid(row=3, column=0, sticky="e")
    state.L_entry = tk.Entry(parent, width=10)
    state.L_entry.grid(row=3, column=1, sticky="w")
    state.L_entry.bind("<KeyRelease>", validate_single_ready)

    # Duration (row=4)
    tk.Label(parent, text="Duration (sec):", font=default_font).grid(row=4, column=0, sticky="e")
    state.duration_entry = tk.Entry(parent, width=10)
    state.duration_entry.grid(row=4, column=1, sticky="w")
    state.duration_entry.bind("<KeyRelease>", validate_single_ready)

    tk.Label(parent, text="Noise Source:", font=default_font).grid(row=5, column=0, sticky="e")
    src_frame = tk.Frame(parent); src_frame.grid(row=5, column=1, sticky="w")
    tk.Radiobutton(src_frame, text="Stationary", variable=state.noise_source_var, value="Stationary",
                   command=lambda:(on_noise_source_change(), validate_single_ready())).pack(side="left")
    tk.Radiobutton(src_frame, text="WAV", variable=state.noise_source_var, value="WAV",
                   command=lambda:(on_noise_source_change(), validate_single_ready())).pack(side="left")

    tk.Label(parent, text="Noise Type:", font=default_font).grid(row=6, column=0, sticky="e")
    noise_menu = ttk.Combobox(parent, textvariable=state.noise_var,
                              values=["White","Pink","Brownian","Violet","Grey","Blue"],
                              state="readonly", width=12)
    noise_menu.grid(row=6, column=1, sticky="w")
    noise_menu.bind("<<ComboboxSelected>>", validate_single_ready)

    tk.Label(parent, text="Noise WAV:", font=default_font).grid(row=7, column=0, sticky="e")
    wav_select_frame = tk.Frame(parent)
    wav_select_frame.grid(row=7, column=1, columnspan=2, sticky="ew")
    wav_btn = tk.Button(wav_select_frame, text="Select WAV", command=select_wav_file, state=tk.DISABLED)
    wav_btn.grid(row=0, column=0, sticky="w")
    wav_label = tk.Label(wav_select_frame, text="No file selected", font=default_font)
    wav_label.grid(row=0, column=1, sticky="w")
    state.wav_label_ref = wav_label

    start_btn = tk.Button(parent, text="Start", command=start_algorithm, state=tk.DISABLED)
    start_btn.grid(row=8, column=0, columnspan=2, sticky="ew")

    state.progress_var = tk.DoubleVar(value=0.0)
    state.progress_bar = ttk.Progressbar(parent, maximum=100.0, variable=state.progress_var)
    state.progress_bar.grid(row=9, column=0, columnspan=2, sticky="ew")

    state.eta_label = tk.Label(parent, text="ETA --:--", font=default_font, anchor="w")
    state.eta_label.grid(row=10, column=0, columnspan=2, sticky="w")

    state.status_label = tk.Label(parent, text="", font=default_font, anchor="w")
    state.status_label.grid(row=11, column=0, columnspan=2, sticky="w")

    tk.Label(parent, text="Execution time (sec):", font=default_font).grid(row=12, column=0, sticky="e")
    exec_val = tk.Label(parent, text="-", font=default_font)
    exec_val.grid(row=12, column=1, sticky="w")

    tk.Label(parent, text="Convergence speed (msec):", font=default_font).grid(row=13, column=0, sticky="e")
    conv_val = tk.Label(parent, text="-", font=default_font); conv_val.grid(row=13, column=1, sticky="w")

    tk.Label(parent, text="Steady state error (dB):", font=default_font).grid(row=14, column=0, sticky="e")
    sse_val = tk.Label(parent, text="-", font=default_font); sse_val.grid(row=14, column=1, sticky="w")

    tk.Label(parent, text="Power (ANC OFF):", font=default_font).grid(row=15, column=0, sticky="e")
    inpow_val = tk.Label(parent, text="-", font=default_font); inpow_val.grid(row=15, column=1, sticky="w")

    tk.Label(parent, text="Power (ANC ON):", font=default_font).grid(row=16, column=0, sticky="e")
    outpow_val = tk.Label(parent, text="-", font=default_font); outpow_val.grid(row=16, column=1, sticky="w")

    play_before_btn = tk.Button(parent, text="Play Input", command=toggle_play_before, state=tk.DISABLED)
    play_before_btn.grid(row=17, column=0, sticky="ew")

    play_after_btn = tk.Button(parent, text="Play Output", command=toggle_play_after, state=tk.DISABLED)
    play_after_btn.grid(row=17, column=1, sticky="ew")

    graphs_frame = tk.Frame(parent)
    graphs_frame.grid(row=18, column=0, rowspan=2, columnspan=2, sticky="ew")
    
    fw_btn = tk.Button(graphs_frame, text="Filter Weights", command=plot_filter, state=tk.DISABLED)
    fw_btn.grid(row=0, column=0, sticky="ew")
    
    pp_btn = tk.Button(graphs_frame, text="Primary Path", command=plot_primary_path_effect, state=tk.DISABLED)
    pp_btn.grid(row=0, column=1, sticky="ew")

    sp_btn = tk.Button(graphs_frame, text="Secondary Path", command=plot_secondary_path_effect, state=tk.DISABLED)
    sp_btn.grid(row=0, column=2, sticky="ew")
    
    ea_btn = tk.Button(graphs_frame, text="Error Analysis", command=plot_error, state=tk.DISABLED)
    ea_btn.grid(row=0, column=3, sticky="ew")

    sf_btn = tk.Button(graphs_frame, text="Signal Flow", command=plot_signal, state=tk.DISABLED)
    sf_btn.grid(row=1, column=0, sticky="ew")

    spec_btn = tk.Button(graphs_frame, text="Noise Spectrogram", command=plot_noise_spec, state=tk.DISABLED)
    spec_btn.grid(row=1, column=1, sticky="ew")

    err_spec_btn = tk.Button(graphs_frame, text="Error Spectrogram", command=plot_error_spec, state=tk.DISABLED)
    err_spec_btn.grid(row=1, column=2, sticky="ew")
    
    band_btn = tk.Button(graphs_frame, text="Band Attenuation", command=plot_band_attn, state=tk.DISABLED)
    band_btn.grid(row=1, column=3, sticky="ew")

    save_btn = tk.Button(parent, text="Save Results", command=save_single_results, state=tk.DISABLED)
    save_btn.grid(row=20, column=0, columnspan=2, sticky="ew")
    
    tk.Label(parent, text="Custom Bands (Hz):", font=default_font).grid(row=21, column=0, sticky="ne")
    state.bands_text = tk.Text(parent, height=3, width=20)
    state.bands_text.insert(tk.INSERT, "0-500, 500-1000, 1000-3000, 3000-5000, 5000-10000")
    state.bands_text.grid(row=21, column=1, sticky="ew")

    # Buttons group for global enabling/disabling
    state.all_buttons.extend([
        play_before_btn, play_after_btn, save_btn, fw_btn, pp_btn,
        sp_btn, ea_btn, sf_btn, spec_btn, err_spec_btn, band_btn
    ])

    # Expose callback to other panels
    state.start_single_run_cb = start_algorithm

    # Initial validation
    parent.after(0, on_noise_source_change)
    parent.after(0, validate_single_ready)