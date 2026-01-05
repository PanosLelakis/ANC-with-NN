import threading
import numpy as np
from scipy.signal import spectrogram
from utils.smoothing import *
from utils.convert_to_dbfs import convert_to_dbfs
from utils.fft_transform import compute_fft

def _new_fig(headless: bool, figsize=(9.0, 5.4), dpi=110):
    """
    Create a figure safely:
      - headless=True: Agg canvas
      - interactive: size scaled to ~60% of screen width and ~55% of height (if available)
    """
    if headless or (threading.current_thread() is not threading.main_thread()):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        fig = Figure(figsize=figsize, dpi=dpi)
        FigureCanvas(fig)
        return fig, None
    else:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=figsize, dpi=dpi)
        # try to resize according to screen
        try:
            mgr = fig.canvas.manager
            sw = sh = None
            # TkAgg
            if hasattr(mgr, "window") and hasattr(mgr.window, "winfo_screenwidth"):
                sw = mgr.window.winfo_screenwidth()
                sh = mgr.window.winfo_screenheight()
            # QtAgg
            elif hasattr(mgr, "window") and hasattr(mgr.window, "screen"):
                geo = mgr.window.screen().availableGeometry()
                sw, sh = geo.width(), geo.height()
            if sw and sh:
                target_w = int(0.60 * sw)
                target_h = int(0.55 * sh)
                w_in = max(6.0, target_w / float(dpi))
                h_in = max(3.6, target_h / float(dpi))
                fig.set_size_inches(w_in, h_in, forward=True)
        except Exception:
            pass
        return fig, plt

def figure_title_metadata(fig,
                          algorithm_name="", mu=None, L=None, noise_type="",
                          convergence_time=None, steady_state_error=None,
                          title_line3="", fontsize=9.5, y=0.98):
    """Write a 1-3 line suptitle on the provided Figure without touching pyplot/backend."""
    title_line1 = f"Algo: {algorithm_name} | μ: {mu} | L: {L} | Noise: {noise_type}"

    if convergence_time is None:  conv_str = "Conv. Speed: N/A"
    else:                         conv_str = f"Conv. Speed: {convergence_time:.2f} ms"
    if steady_state_error is None: sse_str = "SSE: N/A"
    else:                          sse_str = f"SSE: {steady_state_error:.2f} dB"

    title_line2 = f"{conv_str} | {sse_str}"
    if title_line3:
        full = f"{title_line1}\n{title_line2}\n{title_line3}"
    else:
        full = f"{title_line1}\n{title_line2}"
    fig.suptitle(full, fontsize=fontsize, fontweight="bold", y=y)

def annotate_convergence(ax, t, y_dbfs_smooth, conv_ms):
    """Draw a vertical dashed line and a marker at the convergence time."""
    if conv_ms is None: return
    try:
        t_conv = float(conv_ms) / 1000.0
    except Exception:
        return
    if not np.isfinite(t_conv): return
    # find nearest index
    idx = int(np.argmin(np.abs(np.asarray(t) - t_conv)))
    x = float(t[idx])
    y = float(y_dbfs_smooth[idx]) if (0 <= idx < len(y_dbfs_smooth)) else 0.0
    ax.axvline(x=x, linestyle="--", linewidth=1.5, color="k", alpha=0.7, label="Convergence time")
    ax.plot([x], [y], marker="o", markersize=7, color="k", antialiased=True)

def beautify_plot(axis, title, xlabel, ylabel, xlim_left, xlim_right, ylim_bottom, ylim_top, xscale=None, yscale=None):
    #axis = fig.gca()
    axis.set_title(title, fontsize=9)
    if xscale is not None:
        axis.set_xscale(xscale)
    if yscale is not None:
        axis.set_yscale(yscale)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if (xlim_left is not None) and (xlim_right is not None):
        axis.set_xlim(xlim_left, xlim_right)
    if (ylim_bottom is not None) and (ylim_top is not None):
        axis.set_ylim(ylim_bottom, ylim_top)
    axis.legend(loc="upper right")
    axis.grid()
    axis.get_figure().tight_layout(rect=[0, 0.06, 1, 0.90])

def save_plot(fig, save_dir, name):
    import os
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, name), dpi=150, bbox_inches="tight")
    fig.clear()

def dispose_fig(fig, plt, save_dir):
    try:
        if save_dir is not None:
            if plt is not None:
                plt.close(fig)
            else:
                fig.clear()
        else:
            import gc
            fig.canvas.mpl_connect("close_event", lambda evt: (evt.canvas.figure.clear(), gc.collect()))
    except Exception:
        pass

def plot_filter_weights(fs, w_final, #w_initial, 
                algorithm_name="", mu=None, L=None, noise_type="",
                convergence_time=None, steady_state_error=None,
                save_dir=None):
    
    freqs, w_fft = compute_fft(w_final, fs, n_fft=2**14)
    fig1, plt = _new_fig(headless=bool(save_dir), figsize=(13, 8), dpi=120)
    figure_title_metadata(fig1, algorithm_name, mu, L, noise_type,
                      convergence_time, steady_state_error, "Filter Weights")
    fig1.gca().plot(w_final, label="Final Weights", antialiased=True)
    beautify_plot(fig1.gca(), "Filter Weights (Time Domain)", "Coefficient Index",
                  "Weight Value", 0, L-1, -1, 1, xscale=None)
    if save_dir:
        save_plot(fig1, save_dir, "filter_weights_time.png")
    dispose_fig(fig1, plt, save_dir)

    fig2, plt = _new_fig(headless=bool(save_dir), figsize=(13, 8), dpi=120)
    figure_title_metadata(fig2, algorithm_name, mu, L, noise_type,
                      convergence_time, steady_state_error, "Filter Weights")
    fig2.gca().plot(freqs, w_fft, label="FFT of Weights", antialiased=True)
    beautify_plot(fig2.gca(), "Filter Weights (Frequency Domain)", "Frequency (Hz)",
                  "Magnitude (dB)", 10, 10000, -20, 10, xscale="log")
    if save_dir:
        save_plot(fig2, save_dir, "filter_weights_fft.png")
    dispose_fig(fig2, plt, save_dir)

def plot_path_analysis(path_ir, signal_before, signal_after, t, fs, title_prefix,
                 algorithm_name="", mu=None, L=None, noise_type="",
                 convergence_time=None, steady_state_error=None, save_dir=None):

    freqs_path, path_fft = compute_fft(path_ir, fs, n_fft=2**14)
    freqs_before, before_fft = compute_fft(signal_before, fs, n_fft=2**14)
    freqs_after, after_fft = compute_fft(signal_after, fs, n_fft=2**14)

    path_fft_s = smooth_fractional_octave(freqs_path,  path_fft,  fraction=6, scale=8.0, min_bins=11,  max_bins=199)
    before_s = smooth_fractional_octave(freqs_before, before_fft, fraction=6, scale=8.0, min_bins=11,  max_bins=199)
    after_s = smooth_fractional_octave(freqs_after,  after_fft,  fraction=6, scale=8.0, min_bins=11,  max_bins=199)

    fig1, plt = _new_fig(headless=bool(save_dir), figsize=(13, 8), dpi=120)
    figure_title_metadata(fig1, algorithm_name, mu, L, noise_type,
                      convergence_time, steady_state_error, f"{title_prefix} Path Frequency Domain")
    fig1.gca().plot(freqs_path, path_fft_s, label="Frequency Response (smoothed)", antialiased=True)
    beautify_plot(fig1.gca(), f"{title_prefix} Path Frequency Response", "Frequency (Hz)",
                  "Magnitude (dB)", 10, 10000, -10, 20, xscale="log")
    if save_dir:
        save_plot(fig1, save_dir, f"{title_prefix.lower()}_path_fft.png")
    dispose_fig(fig1, plt, save_dir)

    # Second figure: input vs after path (FFT)
    fig2, plt = _new_fig(headless=bool(save_dir), figsize=(13, 8), dpi=120)
    figure_title_metadata(fig2, algorithm_name, mu, L, noise_type,
                      convergence_time, steady_state_error, f"{title_prefix} Path Frequency Domain")
    fig2.gca().plot(freqs_before, before_s,   label="Input (smoothed)", antialiased=True)
    fig2.gca().plot(freqs_after,  after_s,    label="After (smoothed)", antialiased=True)
    beautify_plot(fig2.gca(), f"Signal Before & After {title_prefix} Path (FFT)", "Frequency (Hz)",
                  "Magnitude (dB)", 10, 10000, 20, 80, xscale="log")
    if save_dir:
        save_plot(fig2, save_dir, f"{title_prefix.lower()}_before_after_fft.png")
    dispose_fig(fig2, plt, save_dir)

def plot_error_analysis(error_signal, t, fs, passive_cancelling=None,
                 algorithm_name="", mu=None, L=None, noise_type="",
                 convergence_time=None, steady_state_error=None, save_dir=None):

    # Use last 20% of samples
    start_idx = int(0.8 * len(error_signal))
    active_segment = error_signal[start_idx:]
    freqs, error_fft = compute_fft(active_segment, fs, n_fft=2**14)

    if passive_cancelling is not None:
        ref_max = np.max(np.abs(passive_cancelling)) + 1e-12
        passive_dbfs = convert_to_dbfs(passive_cancelling, ref_max)
        error_dbfs   = convert_to_dbfs(error_signal,      ref_max)
        passive_dbfs_smooth = smooth_time_median(passive_dbfs, fs, window_ms=120)
        passive_segment = passive_cancelling[start_idx:]
        _, passive_fft = compute_fft(passive_segment, fs, n_fft=2**14)
        passive_fft_s = smooth_fractional_octave(freqs, passive_fft, fraction=6, scale=8.0, min_bins=11,  max_bins=199)
    else:
        # fall back to per-signal
        error_dbfs   = convert_to_dbfs(error_signal, np.max(np.abs(error_signal)) + 1e-12)
    error_dbfs_smooth   = smooth_time_median(error_dbfs,   fs, window_ms=120)
    error_fft_s   = smooth_fractional_octave(freqs, error_fft,   fraction=6, scale=8.0, min_bins=11,  max_bins=199)
    
    fig1, plt = _new_fig(headless=bool(save_dir), figsize=(13, 8), dpi=120)
    figure_title_metadata(fig1, algorithm_name, mu, L, noise_type,
                      convergence_time, steady_state_error, "Error Signal Analysis")
    if passive_cancelling is not None:
        fig1.gca().plot(t, passive_dbfs_smooth, label="Passive Cancelling", linestyle="--", alpha=0.7, antialiased=True)
        fig1.gca().plot(t, error_dbfs_smooth, label="Active Cancelling", linestyle="-", alpha=0.7, antialiased=True)
        title = "Active - Passive Cancelling (Time Domain)"
    else:
        fig1.gca().plot(t, error_dbfs_smooth, label="Active Cancelling", linestyle="-", antialiased=True)
        title = "Active Cancelling (Time Domain)"
    
    annotate_convergence(fig1.gca(), t, error_dbfs_smooth, convergence_time)
    beautify_plot(fig1.gca(), title, "Time (sec)", "Amplitude (dBFS)", 0, 1, -30, 0)

    if save_dir:
        save_plot(fig1, save_dir, "error_time.png")
    dispose_fig(fig1, plt, save_dir)

    fig2, plt = _new_fig(headless=bool(save_dir), figsize=(13, 8), dpi=120)
    figure_title_metadata(fig2, algorithm_name, mu, L, noise_type,
                      convergence_time, steady_state_error, "Error Signal Analysis")
    if passive_cancelling is not None:
        fig2.gca().plot(freqs, passive_fft_s,label="Passive FFT (smoothed)", antialiased=True)
        fig2.gca().plot(freqs, error_fft_s,  label="Active FFT (smoothed)", antialiased=True)
        title = "Active - Passive Cancelling (Frequency Domain)"
    else:
        fig2.gca().plot(freqs, error_fft_s,  label="Active FFT (smoothed)", antialiased=True)
        title = "Active Cancelling (Frequency Domain)"
    
    beautify_plot(fig2.gca(), title, "Frequency (Hz)", "Magnitude (dB)", 10, 10000, 20, 60, xscale="log")
    
    if save_dir:
        save_plot(fig2, save_dir, "error_fft.png")
    dispose_fig(fig2, plt, save_dir)

def plot_signal_flow(reference, noisy, error, t,
                 algorithm_name="", mu=None, L=None, noise_type="",
                 convergence_time=None, steady_state_error=None, save_dir=None):
    
    reference_dbfs = convert_to_dbfs(reference, np.max(np.abs(reference)) + 1e-12)
    noisy_dbfs = convert_to_dbfs(noisy, np.max(np.abs(noisy)) + 1e-12)
    error_dbfs = convert_to_dbfs(error, np.max(np.abs(error)) + 1e-12)

    fig1, plt = _new_fig(headless=bool(save_dir))
    axs = fig1.subplots(2, 2)
    ax00, ax01, ax10, ax11 = axs[0,0], axs[0,1], axs[1,0], axs[1,1]
    figure_title_metadata(fig1, algorithm_name, mu, L, noise_type,
                      convergence_time, steady_state_error, "Signal Flow Comparison")

    ax00.plot(t, reference, label="Reference", alpha=0.7, antialiased=True)
    ax00.plot(t, noisy, label="Noisy", alpha=0.7, antialiased=True)
    beautify_plot(ax00, "Reference vs Noisy (Time Domain)", "Time (sec)", "Amplitude", 0, 1, -4, 4)

    ax01.plot(t, noisy, label="Noisy", alpha=0.7, antialiased=True)
    ax01.plot(t, error, label="Error", alpha=0.7, antialiased=True)
    annotate_convergence(ax01, t, error, convergence_time)
    beautify_plot(ax01, "Noisy vs Error (Time Domain)", "Time (sec)", "Amplitude", 0, 1, -4, 4)

    ax10.plot(t, reference_dbfs, label="Reference", alpha=0.7, antialiased=True)
    ax10.plot(t, noisy_dbfs, label="Noisy", alpha=0.7, antialiased=True)
    beautify_plot(ax10, "Reference vs Noisy (Time Domain)", "Time (sec)", "Amplitude (dBFS)", 0, 1, -160, 0)

    ax11.plot(t, noisy_dbfs, label="Noisy", alpha=0.7, antialiased=True)
    ax11.plot(t, error_dbfs, label="Error", alpha=0.7, antialiased=True)
    annotate_convergence(ax11, t, error_dbfs, convergence_time)
    beautify_plot(ax11, "Noisy vs Error (Time Domain)", "Time (sec)", "Amplitude (dBFS)", 0, 1, -60, 0)

    if save_dir:
        save_plot(fig1, save_dir, "signal_flow.png")
    dispose_fig(fig1, plt, save_dir)

def plot_noise_spectrogram(signal, fs, nperseg=1024, noverlap=512, save_dir=None):
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap,
                            scaling='spectrum', mode='magnitude')
    Sxx_db = 20 * np.log10(Sxx + 1e-12)

    fig1, plt = _new_fig(headless=bool(save_dir))
    beautify_plot(fig1.gca(), "Noise Spectrogram", "Time (sec)", "Frequency (Hz)", None, None, 10, 10000, yscale="log")
    im = fig1.gca().pcolormesh(t, f, Sxx_db, shading='gouraud')
    cbar = fig1.colorbar(im)
    cbar.set_label('Magnitude (dB)')
    if save_dir:
        save_plot(fig1, save_dir, "noise_spectrogram.png")
    dispose_fig(fig1, plt, save_dir)

def _band_edges_from_string(bands_str):
    # Example: "0-500, 500-1000, 1000-3000"
    bands = []
    if not bands_str.strip():
        return bands
    for token in bands_str.split(","):
        token = token.strip()
        if "-" in token:
            a, b = token.split("-")
            try:
                f1 = float(a.strip())
                f2 = float(b.strip())
                if f2 > f1 >= 0:
                    bands.append((f1, f2))
            except Exception:
                pass
    return bands

def _band_powers(x, fs):
    # last 20% steady-state
    N = len(x); seg = x[int(0.8*N):]
    # Hann to reduce leakage
    win = np.hanning(len(seg))
    seg = seg * win
    X = np.fft.rfft(seg)
    freqs = np.fft.rfftfreq(len(seg), 1.0/fs)
    psd = (np.abs(X)**2) / (np.sum(win**2) + 1e-12)  # proportional; absolute scale cancels in ratios
    return freqs, psd

def plot_band_attenuation(d_signal, e_signal, fs, bands=None, bands_str="", save_dir=None):
    if bands is None or len(bands) == 0:
        bands = _band_edges_from_string(bands_str)
        if len(bands) == 0:
            bands = [(0,500), (500,1000), (1000,3000), (3000,5000), (5000,10000)]

    f_d, P_d = _band_powers(d_signal, fs)
    _, P_e = _band_powers(e_signal, fs)
    att_db = []
    labels = []
    for (f1, f2) in bands:
        idx = np.where((f_d >= f1) & (f_d < f2))[0]
        Pd = np.sum(P_d[idx]) + 1e-20
        Pe = np.sum(P_e[idx]) + 1e-20
        att_db.append(10.0 * np.log10(Pd / Pe))
        labels.append(f"{int(f1)}–{int(f2)}")

    fig1, plt = _new_fig(headless=bool(save_dir))
    ax = fig1.gca()
    x = np.arange(len(labels))
    ax.bar(x, att_db, width=0.6)
    beautify_plot(ax, "Band Attenuation (steady-state)", "Frequency band (Hz)", "Attenuation (dB)", None, None, None, None)
    for i, v in enumerate(att_db):
        ax.text(i, v, f"{v:.1f} dB", ha='center', va='bottom')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(axis='y', alpha=0.3)
    if save_dir:
        save_plot(fig1, save_dir, "band_attenuation.png")
    dispose_fig(fig1, plt, save_dir)

def plot_hparam_heatmap(ranked, mu_vals, L_vals, save_dir=None):
    score = np.full((len(L_vals), len(mu_vals)), np.nan, dtype=float)
    for r in ranked:
        L = int(r["L"]); mu = float(r["mu"])
        try:
            i = np.where(L_vals == L)[0][0]
            j = np.where(np.isclose(mu_vals, mu))[0][0]
            score[i, j] = r["score"]
        except Exception:
            pass

    fig1, plt = _new_fig(headless=bool(save_dir))
    ax = fig1.gca()
    im = ax.imshow(score, aspect="auto", origin="lower",
                   extent=[0, len(mu_vals)-1, 0, len(L_vals)-1])
    beautify_plot(ax, "Hyperparameter Score (lower is better)", "mu", "L", None, None, None, None)
    fig1.colorbar(im, label="score")
    ax.set_yticks(np.arange(len(L_vals)))
    ax.set_yticklabels([str(int(L)) for L in L_vals])
    ax.set_xticks(np.arange(len(mu_vals)))
    ax.set_xticklabels([f"{m:.3g}" for m in mu_vals], rotation=45)
    if save_dir:
        save_plot(fig1, save_dir, "param_heatmap.png")
    dispose_fig(fig1, plt, save_dir)