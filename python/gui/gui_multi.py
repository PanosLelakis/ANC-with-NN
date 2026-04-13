import os
import json
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
import time
import math
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from engine.engine_single import run_anc_headless
from engine.engine_multi import score_results, count_unique_combos
from utils.logger import log_case, init_log
from utils.plot import plot_hparam_heatmap, plot_convtime_vs_mu, plot_sse_vs_L
from utils.audio import save_wav

def build_multi_ui(parent, state, default_font, header_font):
    ranked = None
    mu_vals = None
    L_vals = None

    # --- Multi-select controls state ---
    alg_options = ["LMS","NLMS","FxLMS","FxNLMS"]
    alg_vars = {name: tk.BooleanVar(value=(name=="FxNLMS")) for name in alg_options}

    color_options = ["White","Pink","Brownian","Violet","Grey","Blue"]
    color_vars = {c: tk.BooleanVar(value=(c=="White")) for c in color_options}
    color_cbs = {}  # store Checkbutton widgets by color name

    include_stationary_var = tk.BooleanVar(value=True)
    include_wav_var        = tk.BooleanVar(value=False)

    mr_wav_paths = []  # list of selected WAV full paths

    def set_multi_action_buttons(run_best=False, heatmap=False, conv=False, sse=False):
        run_best_btn.config(state=tk.NORMAL if run_best else tk.DISABLED)
        show_heatmap_btn.config(state=tk.NORMAL if heatmap else tk.DISABLED)
        show_conv_btn.config(state=tk.NORMAL if conv else tk.DISABLED)
        show_sse_btn.config(state=tk.NORMAL if sse else tk.DISABLED)
        
    # --- WAV selection (multi-file) ---
    def select_wav_files():
        nonlocal mr_wav_paths
        paths = filedialog.askopenfilenames(filetypes=[("WAV files", "*.wav")])
        if paths:
            mr_wav_paths = list(paths)
            names = [os.path.basename(p) for p in mr_wav_paths]
            mr_wav_label.config(text=", ".join(names) if names else "No file selected")
        validate_multi_ready()

    def validate_multi_ready(*_):
        if getattr(state, "is_locked", False):
            try:
                start_multi_btn.config(state=tk.DISABLED)
            except Exception:
                pass
            return
        
        ok = True
        # at least one algorithm
        if not any(v.get() for v in alg_vars.values()):
            ok = False
        # duration numeric
        try:
            float(mr_duration_entry.get().strip())
        except Exception:
            ok = False
        # μ/L fields
        for e in (mu_min_entry, mu_max_entry, mu_steps_entry, L_min_entry, L_max_entry, L_steps_entry):
            if not e.get().strip():
                ok = False
        if ok:
            try:
                float(mu_min_entry.get())
                float(mu_max_entry.get())
                int(mu_steps_entry.get())
                int(L_min_entry.get())
                int(L_max_entry.get())
                int(L_steps_entry.get())
            except Exception:
                ok = False
        # at least one source and corresponding selection
        src_ok = False
        if include_stationary_var.get() and any(v.get() for v in color_vars.values()):
            src_ok = True
        if include_wav_var.get() and len(mr_wav_paths) > 0:
            src_ok = True
        ok = ok and src_ok

        try:
            start_multi_btn.config(state=(tk.NORMAL if ok else tk.DISABLED))
        except Exception:
            pass

    def fmt_eta(seconds):
        if seconds is None or not math.isfinite(seconds) or seconds < 0:
            return "ETA --:--"
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"ETA {h:02d}:{m:02d}:{s:02d}" if h else f"ETA {m:02d}:{s:02d}"

    def build_mu_values(mu_min, mu_max, mu_steps, scale):
        mu_steps = max(1, int(mu_steps))
        if scale == "log":
            return np.exp(np.linspace(np.log(mu_min), np.log(mu_max), mu_steps))
        else:
            return np.linspace(mu_min, mu_max, mu_steps)

    def build_L_values(L_min, L_max, L_steps):
        L_steps = max(1, int(L_steps))
        vals = np.linspace(int(L_min), int(L_max), L_steps)
        vals = np.unique(np.round(vals).astype(int))
        return vals
    
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
            except:
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

    def show_conv_vs_mu():
        nonlocal ranked
        if ranked is None:
            mr_status.config(text="No results to plot yet.", fg="red")
            return

        # unique (alg, src, noise)
        combos = []
        seen = set()
        for r in ranked:
            k = (r.get("algorithm",""), r.get("source",""), r.get("noise_label",""))
            if k not in seen:
                seen.add(k)
                combos.append(k)

        try:
            for (alg, src, nlabel) in combos:
                rows = [x for x in ranked if x.get("algorithm")==alg and x.get("source")==src and x.get("noise_label")==nlabel]
                plot_convtime_vs_mu(rows, save_dir=None,
                    algorithm_name=alg, noise_type=nlabel)
        except Exception as e:
            mr_status.config(text=f"Plot error: {e}", fg="red")

    def show_sse_vs_L():
        nonlocal ranked
        if ranked is None:
            mr_status.config(text="No results to plot yet.", fg="red")
            return

        combos = []
        seen = set()
        for r in ranked:
            k = (r.get("algorithm",""), r.get("source",""), r.get("noise_label",""))
            if k not in seen:
                seen.add(k)
                combos.append(k)

        try:
            for (alg, src, nlabel) in combos:
                rows = [x for x in ranked if x.get("algorithm")==alg and x.get("source")==src and x.get("noise_label")==nlabel]
                plot_sse_vs_L(rows, save_dir=None,
                    algorithm_name=alg, noise_type=nlabel)
        except Exception as e:
            mr_status.config(text=f"Plot error: {e}", fg="red")
    
    def _run_best_from_multi(state):
        # need best hyperparams *and* the combo metadata
        best = getattr(state, "last_best_combo", None)
        if best is None or state.best_mu is None or state.best_L is None:
            mr_status.config(text="No best result yet. Run multi-run first.", fg="red"); return

        # copy μ, L
        state.mu_entry.delete(0, "end")
        state.mu_entry.insert(0, f"{state.best_mu:.6g}")
        state.L_entry.delete(0, "end")
        state.L_entry.insert(0, f"{state.best_L}")

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

    def start_multi_run():
        nonlocal mu_vals, L_vals, ranked

        try:
            dur = float(mr_duration_entry.get())
            mu_min = float(mu_min_entry.get())
            mu_max = float(mu_max_entry.get())
            mu_steps = int(mu_steps_entry.get())
            L_min  = int(L_min_entry.get())
            L_max  = int(L_max_entry.get())
            L_steps  = int(L_steps_entry.get())
            mscale = mu_scale_var.get()
        except Exception as e:
            mr_status.config(text=f"Input error: {e}", fg="red")
            return

        sel_algs = [a for a,v in alg_vars.items() if v.get()]
        sel_cols = [c for c,v in color_vars.items() if v.get()] if include_stationary_var.get() else []
        sel_wavs = list(mr_wav_paths) if include_wav_var.get() else []

        mu_vals = build_mu_values(mu_min, mu_max, mu_steps, mscale)
        L_vals  = build_L_values(L_min, L_max, L_steps)
        muL = [(float(mu), int(L)) for L in L_vals for mu in mu_vals]

        init_log(run_kind="multi", clear=True, log_dir=os.path.join(os.getcwd(), "results"))

        # Build (algorithm, source, type/path) combinations
        combos = []
        for alg in sel_algs:
            if include_stationary_var.get():
                for col in sel_cols:
                    combos.append((alg, "Stationary", col, ""))  # noise_type=col
            if include_wav_var.get():
                for w in sel_wavs:
                    combos.append((alg, "WAV", os.path.basename(w), w))  # label + path

        total = len(muL) * len(combos)
        if total == 0:
            mr_status.config(text="Empty grid.", fg="red")
            return

        # UI prep
        start_multi_btn.config(state=tk.DISABLED)
        set_multi_action_buttons(False, False, False, False)
        mr_status.config(text=f"Queued {total} simulations…", fg="black")
        mr_progress_var.set(0.0)
        state.lock_ui()

        # Worker thread: run parallel with live progress
        def worker():
            start_t = time.time()
            results = []
            done = 0
            with ProcessPoolExecutor(max_workers=None) as ex:
                fut_meta = {}
                for (mu, L) in muL:
                    for (alg, src, nlabel, wfp) in combos:
                        fut = ex.submit(run_anc_headless, alg, int(L), float(mu),
                                        src, nlabel, wfp, dur)
                        fut_meta[fut] = {"algorithm": alg, "source": src, "noise_label": nlabel, "wav_path": wfp}
                for fut in as_completed(fut_meta):
                    meta = fut_meta[fut]
                    try:
                        r = fut.result()
                        r.update(meta)
                        results.append(r)
                        # simulation ok (round to GUI style)
                        log_case(stage="simulate", status="ok",
                                algorithm=meta["algorithm"], source=meta["source"], noise_label=meta["noise_label"],
                                L=int(r.get("L", 0)), mu=float(r.get("mu", 0.0)),
                                conv_ms=round(float(r.get("conv_ms", 0.0)), 2),
                                sse_db=round(float(r.get("sse_db", 0.0)), 2),
                                exec_time=None,
                                in_power=round(float(r.get("in_power", 0.0)), 3),
                                out_power=round(float(r.get("out_power", 0.0)), 3),
                                save_path="", message="")
                    except Exception as e:
                        # simulation error
                        log_case(stage="simulate", status="error",
                                algorithm=meta["algorithm"], source=meta["source"], noise_label=meta["noise_label"],
                                L=None, mu=None, conv_ms=None, sse_db=None, exec_time=None,
                                in_power=None, out_power=None, save_path="", message=str(e))
                    done += 1
                    elapsed = time.time() - start_t
                    eta = (elapsed / done) * (total - done) if done else None
                    pct = 100.0 * done / total
                    def ui_update():
                        mr_progress_var.set(pct)
                        mr_status.config(text=f"{done}/{total} — {fmt_eta(eta)}")
                    state.ui_call(ui_update)

            ranked_local = score_results(results, duration_s=dur, w_conv=0.5, w_sse=0.5, normalize="dataset",
                                         mu_vals=mu_vals, L_vals=L_vals,
                                         alpha=float(alpha_entry.get().strip()), lambda_muL=0.1)

            def ui_done():
                nonlocal ranked
                ranked = ranked_local
                elapsed = time.time() - start_t
                if not ranked:
                    mr_status.config(text="No valid results.", fg="red")
                    state.unlock_ui()
                    start_multi_btn.config(state=tk.NORMAL)
                    validate_multi_ready()
                    return
                best = ranked[0]
                state.best_mu = float(best["mu"]); state.best_L = int(best["L"])
                state.last_best_combo = {  # remember which combo produced the best
                    "algorithm":   best.get("algorithm",""),
                    "source":      best.get("source",""),
                    "noise_label": best.get("noise_label",""),
                    "wav_path":    best.get("wav_path",""),
                }
                mr_progress_var.set(100.0)
                mr_status.config(
                    text=(f"Done. Best: L={state.best_L}, μ={state.best_mu:.6g}, "
                        f"score={best['score']:.3f}, conv={best['conv_ms']:.2f} ms, sse={best['sse_db']:.2f} dB"),
                    fg="green"
                )

                unique_noise_types = count_unique_combos(combos)
                
                if unique_noise_types == 1:
                    # show best metrics normally
                    best_mu_val.config(text=f"μ: {state.best_mu:.6g}")
                    best_L_val.config(text=f"L: {state.best_L:d}")
                    best_conv_val.config(text=f"Convergence speed: {best['conv_ms']:.2f} ms")
                    best_sse_val.config(text=f"SSE: {best['sse_db']:.2f} dB")
                
                start_multi_btn.config(state=tk.NORMAL)
                mr_exec_label.config(text=f"Execution time (sec): {elapsed:.2f}")

                # Optional saving
                mode = save_mode_var.get().lower()
                if mode == "none":
                    state.unlock_ui()
                    validate_multi_ready()
                    if unique_noise_types == 1:
                        set_multi_action_buttons(True, True, True, True)
                    else:
                        set_multi_action_buttons(False, False, False, False)
                else:
                    # remain locked during saving
                    set_multi_action_buttons(False, False, False, False)
                    start_multi_btn.config(state=tk.DISABLED)
                    state.lock_ui()
                    # count "heatmap per combo" + "case saves"
                    unique_combos = []
                    seen = set()
                    for (alg, src, nlabel, wfp) in combos:
                        k = (alg, src, nlabel)
                        if k not in seen:
                            seen.add(k)
                            unique_combos.append((alg, src, nlabel, wfp))

                    heatmap_jobs = len(unique_combos)
                    case_jobs = 1 if mode == "best" else (len(muL) * len(combos))
                    total_save_jobs = heatmap_jobs + case_jobs

                    mr_progress_var.set(0.0)
                    mr_status.config(text=f"Saving 0/{total_save_jobs}…", fg="black")
                    
                    def _save_all_or_best():
                        from engine.engine_single import run_anc # full sim for complete signals
                        from utils import plot as U

                        def _safe_name(s):
                            return "".join(c for c in s if c.isalnum() or c in (" ","-","_")).strip().replace(" ","_")

                        saved = 0
                        def bump_save_progress():
                            nonlocal saved
                            saved += 1
                            pct = 100.0 * saved / max(1, total_save_jobs)
                            state.ui_call(mr_progress_var.set, pct)
                            state.ui_call(mr_status.config, text=f"Saving {saved}/{total_save_jobs}…", fg="black")

                        seen_combo = set()
                        for (alg, src, nlabel, wfp) in combos:
                            key = (alg, src, nlabel)
                            if key in seen_combo: continue
                            seen_combo.add(key)
                            base_root = os.path.join(os.getcwd(), "results", alg, _safe_name(nlabel))
                            os.makedirs(base_root, exist_ok=True)
                            # Filter 'ranked' to the rows for this combo
                            ranked_for_combo = [r for r in ranked if r.get("algorithm")==alg and
                                                                r.get("source")==src and
                                                                r.get("noise_label")==nlabel]
                            if ranked_for_combo:
                                U.plot_hparam_heatmap(ranked_for_combo, mu_vals, L_vals, save_dir=base_root)
                                U.plot_convtime_vs_mu(ranked_for_combo, save_dir=base_root, algorithm_name=alg, noise_type=nlabel)
                                U.plot_sse_vs_L(ranked_for_combo, save_dir=base_root, algorithm_name=alg, noise_type=nlabel)
                            bump_save_progress()
                        
                        def _run_and_save(alg, src, nlabel, wfp, Lx, mux):
                            # ---- Create folder up front so every (L,μ) is visible even if we fail later
                            base_root = os.path.join(os.getcwd(), "results", alg, _safe_name(nlabel))
                            base = os.path.join(base_root, f"L{int(Lx)}_mu{float(mux):.6g}") if mode == "all" else base_root
                            os.makedirs(base, exist_ok=True)

                            payload = {}
                            
                            def _dummy_prog(_):
                                pass
                            
                            def _cb(ref, noisy, err, tt, fs, exect, convt, ssedb,
                                    w0, wf, pir, sir, d_stream, z_stream, in_pow, out_pow,
                                    before_raw, after_raw):
                                payload.update(dict(
                                    reference=ref, noisy=noisy, error=err, t=tt, fs=fs,
                                    exec_time=exect, conv_ms=convt, sse_db=ssedb,
                                    w0=w0, wf=wf, pir=pir, sir=sir,
                                    d=d_stream, z=z_stream,
                                    in_power=in_pow, out_power=out_pow,
                                    before_raw=before_raw, after_raw=after_raw
                                ))

                            try:
                                # run the simulation
                                run_anc(alg, int(Lx), float(mux), src, nlabel,
                                        ("" if src=="Stationary" else wfp),
                                        dur, _dummy_prog, _cb)
                                
                                # Save audio output as .wav file
                                save_wav(payload["before_raw"], payload["after_raw"], payload["fs"], base)

                                # --- detect divergence here (weights NaN/Inf or gigantic norm)
                                diverged = False
                                try:
                                    wf = np.asarray(payload.get("wf", []), dtype=float)
                                    if wf.size == 0 or (not np.all(np.isfinite(wf))) or (np.linalg.norm(wf) > 1e4):
                                        diverged = True
                                except Exception:
                                    diverged = True

                                # metrics.json (rounded to GUI style)
                                meta = dict(
                                    algorithm=alg, L=int(Lx), mu=float(mux),
                                    noise_source=src, noise_label=nlabel,
                                    fs=int(payload["fs"]),
                                    exec_time=round(float(payload["exec_time"]), 2),
                                    conv_ms=round(float(0.0 if payload["conv_ms"] is None else payload["conv_ms"]), 2),
                                    sse_db=round(float(payload["sse_db"]), 2),
                                    in_power=round(float(payload["in_power"]), 3),
                                    out_power=round(float(payload["out_power"]), 3),
                                    status=("diverged" if diverged else "ok"),
                                )
                                with open(os.path.join(base, "metrics.json"), "w") as f:
                                    json.dump(meta, f, indent=2)

                                if diverged:
                                    with open(os.path.join(base, "error.txt"), "w") as f:
                                        f.write("Divergence detected (non-finite or huge weights).")

                                # --- save a single spectrogram per noise (if not already there)
                                if not os.path.exists(os.path.join(base_root, "noise_spectrogram.png")):
                                    U.plot_noise_spectrogram(payload["noisy"], payload["fs"], save_dir=base_root)

                                if not diverged:  # skip heavy plotting if diverged
                                    U.plot_band_attenuation(payload["before_raw"], payload["after_raw"], payload["fs"],
                                                            save_dir=base, algorithm_name=alg, mu=mux, L=Lx, noise_type=nlabel,
                                                            convergence_time=payload.get("conv_ms"), steady_state_error=payload.get("sse_db"))
                                    U.plot_error_spectrogram(payload["error"], payload["fs"], save_dir=base)
                                    U.plot_filter_weights(payload["fs"], payload["wf"], alg, mux, Lx, nlabel,
                                                        payload["conv_ms"], payload["sse_db"], save_dir=base)
                                    U.plot_path_analysis(payload["pir"], payload["noisy"], payload["d"], payload["fs"],
                                                        "Primary", alg, mux, Lx, nlabel, payload["conv_ms"], payload["sse_db"], save_dir=base)
                                    if payload["z"] is not None:
                                        U.plot_path_analysis(payload["sir"], payload["noisy"], payload["z"], payload["fs"],
                                                            "Secondary", alg, mux, Lx, nlabel, payload["conv_ms"], payload["sse_db"], save_dir=base)
                                    U.plot_error_analysis(payload["after_raw"], payload["t"], payload["fs"],
                                                            passive_cancelling=payload["before_raw"],
                                                            noisy_signal=payload["noisy"],
                                                            algorithm_name=alg, mu=mux, L=Lx, noise_type=nlabel,
                                                            convergence_time=payload["conv_ms"], steady_state_error=payload["sse_db"],
                                                            save_dir=base)
                                    U.plot_signal_flow(payload["reference"], payload["noisy"], payload["error"], payload["t"],
                                                    alg, mux, Lx, nlabel, payload["conv_ms"], payload["sse_db"], save_dir=base)

                                log_case(stage="save", status=("diverged" if diverged else "ok"),
                                        algorithm=alg, source=src, noise_label=nlabel,
                                        L=int(Lx), mu=float(mux),
                                        conv_ms=meta["conv_ms"], sse_db=meta["sse_db"],
                                        exec_time=meta["exec_time"], in_power=meta["in_power"], out_power=meta["out_power"],
                                        save_path=base, message="")
                            except Exception as e:
                                # record failure for this (μ,L)
                                try:
                                    with open(os.path.join(base, "error.txt"), "w") as f:
                                        f.write(str(e))
                                except Exception:
                                    pass
                                try:
                                    log_case(stage="save", status="error",
                                            algorithm=alg, source=src, noise_label=nlabel,
                                            L=int(Lx), mu=float(mux),
                                            conv_ms=None, sse_db=None, exec_time=None, in_power=None, out_power=None,
                                            save_path=base, message=str(e))
                                except Exception:
                                    pass
                            bump_save_progress()
                        
                        if mode == "best":
                            b = state.last_best_combo
                            _run_and_save(b["algorithm"], b["source"], b["noise_label"], b.get("wav_path",""),
                                        state.best_L, state.best_mu)
                        else:  # "all"
                            for (mu, L) in muL:
                                for (alg, src, nlabel, wfp) in combos:
                                    _run_and_save(alg, src, nlabel, wfp, L, mu)

                        # --- final UI update after saving finishes ---
                        def _save_done():
                            mr_progress_var.set(100.0)
                            state.unlock_ui()
                            if unique_noise_types == 1:
                                set_multi_action_buttons(True, True, True, True)
                            else:
                                set_multi_action_buttons(False, False, False, False)
                            validate_multi_ready()  # refresh Start button enable state
                            mr_status.config(text=f"Saved all results to: {os.path.join(os.getcwd(),'results')}", fg="green")

                        state.ui_call(_save_done)
  
                    threading.Thread(target=_save_all_or_best, daemon=False).start()

            state.ui_call(ui_done)

        threading.Thread(target=worker, daemon=True).start()
    
    def on_alpha_change(*_):
        txt = alpha_entry.get().strip()
        try:
            a = float(txt); a = max(0.0, min(1.0, a))
            alpha_info.config(text=f"Preference = {a:.2f}*μ + {1.0-a:.2f}*L (lower favored)", fg="black")
        except Exception:
            alpha_info.config(text="Preference = a*μ + (1-a)*L (lower favored) - invalid a", fg="red")
        validate_multi_ready()

    # Initial setup and validation
    parent.after(0, validate_multi_ready)

    # ---------- UI ----------
    tk.Label(parent, text="Multi-Run", font=header_font).grid(row=0, column=0, columnspan=2, sticky="w")

    # --- Algorithms (multi-select) ---
    tk.Label(parent, text="Algorithms:", font=default_font).grid(row=1, column=0, sticky="ne")
    alg_frame = tk.Frame(parent); alg_frame.grid(row=1, column=1, sticky="w")
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
               command=lambda:(validate_multi_ready(), toggle_source_widgets())).pack(side="left")

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
    tk.Label(parent, text="μ trade-off factor:", font=default_font).grid(row=13, column=0, sticky="e")
    alpha_entry = tk.Entry(parent, width=10)
    alpha_entry.insert(0, "0.5")
    alpha_entry.grid(row=13, column=1, sticky="w")
    alpha_entry.bind("<KeyRelease>", on_alpha_change)

    alpha_info = tk.Label(parent, text="Preference = a*μ + (1-a)*L (lower favored)", font=default_font, anchor="w")
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
                               command=lambda: _run_best_from_multi(state), state=tk.DISABLED)
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