import numpy as np
from joblib import Parallel, delayed
from engine.engine_single import simulate_once

def _linspace_inclusive(a, b, n):
    n = int(n)
    a = float(a)
    b = float(b)
    if n <= 1:
        return np.array([a], dtype=float)
    vals = np.linspace(a, b, n, dtype=float)
    vals[0] = a
    vals[-1] = b
    return vals

def _logspace_inclusive(a, b, n):
    n = int(n)
    a = float(a)
    b = float(b)
    if n <= 1:
        return np.array([a], dtype=float)
    vals = np.geomspace(a, b, n, dtype=float)
    vals[0] = a
    vals[-1] = b
    return vals

def build_grid(mu_min, mu_max, mu_steps, L_min, L_max, L_steps, mu_scale="log"):
    mu_vals = _logspace_inclusive(mu_min, mu_max, mu_steps) if mu_scale.lower() == "log" \
              else _linspace_inclusive(mu_min, mu_max, mu_steps)
    L_vals = np.unique(np.round(_linspace_inclusive(L_min, L_max, L_steps)).astype(int))
    grid = [(float(mu), int(L)) for L in L_vals for mu in mu_vals]
    return mu_vals, L_vals, grid

def average_replicates(res_list):
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in res_list:
        buckets[(int(r["L"]), float(r["mu"]))].append(r)
    aggregated = []
    for (L, mu), items in buckets.items():
        conv_ms = float(np.median([it["conv_ms"] for it in items]))
        sse_db  = float(np.median([it["sse_db"]  for it in items]))
        in_p    = float(np.median([it["in_power"] for it in items]))
        out_p   = float(np.median([it["out_power"] for it in items]))
        fs      = items[0]["fs"]
        divergence = any(bool(it.get("divergence", False)) for it in items)
        aggregated.append({
            "L": L, "mu": mu, "conv_ms": conv_ms, "sse_db": sse_db,
            "in_power": in_p, "out_power": out_p, "fs": fs, "divergence":divergence})
    return aggregated

def run_grid_parallel(algorithm_name, noise_source, noise_type, noise_wav_path,
                      duration, mu_min, mu_max, mu_steps, L_min, L_max, L_steps,
                      mu_scale="log", replicates=1, n_jobs=-1):
    mu_vals, L_vals, _ = build_grid(mu_min, mu_max, mu_steps, L_min, L_max, L_steps, mu_scale)
    jobs = []
    for L in L_vals:
        for mu in mu_vals:
            for _ in range(int(replicates)):
                jobs.append((algorithm_name, int(L), float(mu), noise_source,
                             noise_type, noise_wav_path, duration))
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(simulate_once)(*args) for args in jobs
    )
    aggregated = average_replicates(results)
    ranked = score_results(aggregated, duration_s=duration, a=0.5, normalize="dataset")
    return {"ranked": ranked, "mu_vals": mu_vals, "L_vals": L_vals}

def score_results(results, duration_s, a=0.5, normalize="dataset",
                  mu_vals=None, L_vals=None, lambda_muL=0.0):
    
    if not results:
        return []
    
    a = float(np.clip(a, 0.0, 1.0))

    conv = np.array([r["conv_ms"] for r in results], dtype=float)
    sse_db = np.array([r["sse_db"] for r in results], dtype=float)
    divergence_flags = np.array([bool(r.get("divergence", False)) for r in results], dtype=bool)
    
    # Convert duration from sec to msec
    duration_ms = 1000.0 * float(duration_s)

    # Replace invalid convergence values with large value
    # If convergence is Nan or +- inf, treat as bad
    conv = np.nan_to_num(
        conv,
        nan=duration_ms,
        posinf=duration_ms,
        neginf=duration_ms
    )

    # Treat convergence at the first allowed detection instant as unreliable.
    # These cases are assigned the full simulation duration so they receive a bad score.
    MIN_VALID_CONV_MS = 20.0
    conv = np.where(conv <= MIN_VALID_CONV_MS + 1e-6, duration_ms, conv)

    # Replace invalid SSE values with large value
    # If SSE is 0 or +- inf, treat as bad
    sse_db = np.nan_to_num(  
        sse_db,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # -------------------------
    # Normalize convergence time
    # -------------------------

    if normalize == "dataset":  # Use current dataset for normalization.
        conv_min = np.min(conv)  # Best convergence time in this run
        conv_max = np.max(conv)  # Worst convergence time in this run
        conv_span = max(conv_max - conv_min, 1e-12)  # Avoid division by zero
        conv_norm = (conv - conv_min) / conv_span  # Normalize convergence time to 0–1
    else:
        conv_norm = conv / max(duration_ms, 1e-12)  # Normalize by total simulation duration
        conv_norm = np.clip(conv_norm, 0.0, 1.0)  # Keep values inside 0–1

    # -------------------------
    # Normalize SSE
    # -------------------------

    sse_min = np.min(sse_db)  # Best SSE in this run
    sse_max = np.max(sse_db)  # Worst SSE in this run
    sse_span = max(sse_max - sse_min, 1e-12)  # Avoid division by zero

    sse_norm = (sse_db - sse_min) / sse_span  # Normalize SSE to 0–1

    # -------------------------
    # Final score
    # -------------------------

    score_core = a * conv_norm + (1.0 - a) * sse_norm  # Calculate score core

    if (mu_vals is not None) and (L_vals is not None) and len(mu_vals) and len(L_vals):
        mu_min, mu_max = float(np.min(mu_vals)), float(np.max(mu_vals))
        L_min,  L_max  = float(np.min(L_vals)),  float(np.max(L_vals))
        span_mu = max(mu_max - mu_min, 1e-12)
        span_L  = max(L_max  - L_min,  1e-12)

        mus = np.array([r["mu"] for r in results], dtype=float)
        Ls  = np.array([r["L"]  for r in results], dtype=float)

        mu_norm = (mus - mu_min) / span_mu
        L_norm  = (Ls  - L_min)  / span_L

        # Small secondary penalty, not controlled by user
        penalty = 0.5 * mu_norm + 0.5 * L_norm
        scores = score_core + lambda_muL * penalty
    else:
        scores = score_core
    scores = np.where(divergence_flags, scores + 1e6, scores)
    ranked_local = []
    for r, s, cn, re, conv_used in zip(results, scores, conv_norm, sse_norm, conv):
        item = {
            "L": int(r["L"]),
            "mu": float(r["mu"]),
            "conv_ms": float(conv_used),
            "sse_db": float(r["sse_db"]),
            "power_anc_off": float(r["in_power"]),
            "power_anc_on": float(r["out_power"]),
            "score": float(s),
            "conv_norm": float(cn),
            "rms_e": float(re),
            "algorithm": r.get("algorithm",""),
            "source": r.get("source",""),
            "noise_label": r.get("noise_label",""),
            "wav_path": r.get("wav_path",""),
            "divergence": bool(r.get("divergence", False)),
            "status": r.get("status", "diverged" if bool(r.get("divergence", False)) else "ok"),
            "save_path": r.get("save_path", ""),
            "avg_pnc_dbr": r.get("avg_pnc_dbr", None),
            "band_attenuation": r.get("band_attenuation", {})
        }
        ranked_local.append(item)
    
    ranked_local.sort(key=lambda d: d["score"])
    return ranked_local

def count_unique_combos(combos):
    noise_types = set()
    for (alg, src, nlabel, _) in combos:
        if src == "Stationary":
            noise_types.add((alg, "Stationary", nlabel))
        else:
            noise_types.add((alg, "WAV", nlabel))
    return len(noise_types)


def run_multi_case(algorithm, L, mu, source, noise_label, wav_path,
                   duration, save_mode="none", results_root="results"):
    """
    One simulation job for Multi Run.

    save_mode:
    - "none": run only metrics
    - "all": run full simulation and save this case immediately
    """
    from engine.engine_single import run_anc_headless, run_anc_capture
    from utils.result_saver import save_case_artifacts

    if save_mode == "all":
        payload = run_anc_capture(
            algorithm, L, mu, source, noise_label,
            "" if source == "Stationary" else wav_path,
            duration
        )

        meta = save_case_artifacts(
            payload=payload,
            alg=algorithm,
            src=source,
            nlabel=noise_label,
            L=L,
            mu=mu,
            base_root=results_root,
            save_plots=True,
            save_audio_file=True
        )

        result = dict(
            mu=float(mu),
            L=int(L),
            conv_ms=0.0 if meta["conv_ms"] is None else float(meta["conv_ms"]),
            sse_db=float(meta["sse_db"]) if meta["sse_db"] is not None else float("nan"),
            in_power=float(meta["in_power"]),
            out_power=float(meta["out_power"]),
            fs=int(payload["fs"]),
            divergence=bool(meta["divergence"]),
            status=meta["status"],
            save_path=meta["save_path"]
        )

    else:
        result = run_anc_headless(
            algorithm, L, mu, source, noise_label,
            "" if source == "Stationary" else wav_path,
            duration
        )
        result["save_path"] = ""

    result.update(dict(
        algorithm=algorithm,
        source=source,
        noise_label=noise_label,
        wav_path=wav_path
    ))

    return result