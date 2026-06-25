import os
import json
import numpy as np
from utils.audio import save_wav
from utils import plot as U

def safe_name(s):
    return "".join(c for c in str(s) if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")

def save_case_artifacts(payload, alg, src, nlabel, L, mu, base_root, save_plots=True, save_audio_file=True):
    base = os.path.join(base_root, alg, safe_name(nlabel), f"L{int(L)}_mu{float(mu):.6g}")
    os.makedirs(base, exist_ok=True)

    diverged = bool(payload.get("divergence", False))

    try:
        wf = np.asarray(payload.get("wf", []), dtype=float)
        if wf.size == 0 or (not np.all(np.isfinite(wf))) or (np.linalg.norm(wf) > 1e4):
            diverged = True
    except Exception:
        diverged = True

    in_power = float(payload["in_power"])
    out_power = float(payload["out_power"])

    attenuation_db = 10.0 * np.log10((out_power + 1e-12) / (in_power + 1e-12))

    meta = dict(
        algorithm=alg,
        L=int(L),
        mu=float(mu),
        noise_source=src,
        noise_label=nlabel,
        fs=int(payload["fs"]),
        exec_time=round(float(payload["exec_time"]), 2),
        conv_ms=None if payload["conv_ms"] is None else round(float(payload["conv_ms"]), 2),
        sse_db=None if payload["sse_db"] is None else round(float(payload["sse_db"]), 2),
        in_power=round(in_power, 3),
        out_power=round(out_power, 3),
        attenuation_db=round(float(attenuation_db), 2),
        divergence=diverged,
        status=("diverged" if diverged else "ok"),
        save_path=base
    )

    with open(os.path.join(base, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if save_audio_file:
        save_wav(payload["before_raw"], payload["after_raw"], payload["fs"], base)

    if diverged:
        with open(os.path.join(base, "error.txt"), "w") as f:
            f.write("Divergence detected.")
        return meta

    if save_plots:
        base_noise = os.path.join(base_root, alg, safe_name(nlabel))

        if not os.path.exists(os.path.join(base_noise, "noise_spectrogram.pdf")):
            U.plot_noise_spectrogram(payload["noisy"], payload["fs"], save_dir=base_noise)

        U.plot_band_attenuation(
            payload["before_raw"], payload["after_raw"], payload["fs"],
            save_dir=base,
            algorithm_name=alg,
            mu=mu,
            L=L,
            noise_type=nlabel,
            convergence_time=payload.get("conv_ms"),
            steady_state_error=payload.get("sse_db")
        )

        U.plot_error_spectrogram(payload["error"], payload["fs"], save_dir=base)

        U.plot_filter_weights(
            payload["fs"], payload["wf"],
            alg, mu, L, nlabel,
            payload["conv_ms"],
            payload["sse_db"],
            save_dir=base
        )

        U.plot_path_analysis(
            payload["pir"], payload["noisy"], payload["d"], payload["fs"],
            "Primary", alg, mu, L, nlabel,
            payload["conv_ms"],
            payload["sse_db"],
            save_dir=base
        )

        if payload["z"] is not None:
            U.plot_path_analysis(
                payload["sir"], payload["noisy"], payload["z"], payload["fs"],
                "Secondary", alg, mu, L, nlabel,
                payload["conv_ms"],
                payload["sse_db"],
                save_dir=base
            )

        U.plot_error_analysis(
            payload["after_raw"], payload["t"], payload["fs"],
            passive_cancelling=payload["before_raw"],
            noisy_signal=payload["noisy"],
            algorithm_name=alg,
            mu=mu,
            L=L,
            noise_type=nlabel,
            convergence_time=payload["conv_ms"],
            steady_state_error=payload["sse_db"],
            save_dir=base
        )

        U.plot_signal_flow(
            payload["reference"], payload["noisy"], payload["error"], payload["t"],
            algorithm_name=alg,
            mu=mu,
            L=L,
            noise_type=nlabel,
            convergence_time=payload["conv_ms"],
            steady_state_error=payload["sse_db"],
            save_dir=base
        )

    return meta

def append_run_summary(row, results_root="results"):
    import csv
    import os
    import json

    os.makedirs(results_root, exist_ok=True)

    path = os.path.join(results_root, "run_summary.csv")

    fieldnames = [
        "run_kind",
        "algorithm",
        "noise_source",
        "noise_label",
        "L",
        "mu",
        "score",
        "conv_ms",
        "sse_db",
        "power_anc_off",
        "power_anc_on",
        "attenuation_db",
        "divergence",
        "status",
        "save_path",
        "avg_pnc_dbr",
        "band_attenuation"
    ]

    new_file = (not os.path.exists(path)) or os.path.getsize(path) == 0

    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        if new_file:
            f.write("sep=,\n")

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if new_file:
            writer.writeheader()

        writer.writerow({
            "run_kind": row.get("run_kind", "multi"),
            "algorithm": row.get("algorithm", ""),
            "noise_source": row.get("noise_source", row.get("source", "")),
            "noise_label": row.get("noise_label", ""),
            "L": row.get("L", ""),
            "mu": row.get("mu", ""),
            "score": row.get("score", ""),
            "conv_ms": row.get("conv_ms", ""),
            "sse_db": row.get("sse_db", ""),
            "power_anc_off": row.get("power_anc_off", row.get("in_power", "")),
            "power_anc_on": row.get("power_anc_on", row.get("out_power", "")),
            "attenuation_db": row.get("attenuation_db", ""),
            "divergence": row.get("divergence", False),
            "status": row.get("status", ""),
            "save_path": row.get("save_path", ""),
            "avg_pnc_dbr": row.get("avg_pnc_dbr", ""),
            "band_attenuation": json.dumps(row.get("band_attenuation", {}), ensure_ascii=False)
        })