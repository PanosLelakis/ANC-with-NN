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
        aggregated.append({
            "L": L, "mu": mu, "conv_ms": conv_ms, "sse_db": sse_db,
            "in_power": in_p, "out_power": out_p, "fs": fs})
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
    ranked = score_results(aggregated, duration_s=duration, w_conv=0.5, w_sse=0.5, normalize="dataset")
    return {"ranked": ranked, "mu_vals": mu_vals, "L_vals": L_vals}

def score_results(results, duration_s, w_conv=0.5, w_sse=0.5, normalize="dataset",
                    mu_vals=None, L_vals=None, alpha=0.5, lambda_muL=0.1):
    
    # If no simulation results, return empty list
    if not results:
        return []
    
    # Convert to numpy arrays
    conv = np.array([r["conv_ms"] for r in results], dtype=float)
    sse_db = np.array([r["sse_db"] for r in results], dtype=float)
    
    # Calculate RMS error from SSE in dB
    rms_e = np.power(10.0, sse_db / 20.0)
    duration_ms = 1000.0 * float(duration_s)
    
    # Normalize convergence time
    if normalize == "dataset":
        finite = np.isfinite(conv)
        denom = np.max(conv[finite]) if np.any(finite) else duration_ms
        denom = max(denom, 1e-6)
        conv_norm = np.clip(conv / denom, 0, 10)
    else:
        conv_norm = np.clip(conv / max(duration_ms, 1e-6), 0, 10)

    # Calculate score as weighted sum of normalized convergence time and RMS error
    score_core = w_conv * conv_norm + w_sse * rms_e

    # mu-L preference penalty (lower mu and/or lower L favored as a->1 or a->0)
    if (mu_vals is not None) and (L_vals is not None) and len(mu_vals) and len(L_vals):
        mu_min, mu_max = float(np.min(mu_vals)), float(np.max(mu_vals))
        L_min,  L_max  = float(np.min(L_vals)),  float(np.max(L_vals))
        span_mu = max(mu_max - mu_min, 1e-12)
        span_L  = max(L_max  - L_min,  1e-12)
        mus = np.array([r["mu"] for r in results], dtype=float)
        Ls  = np.array([r["L"]  for r in results], dtype=float)
        mu_norm = (mus - mu_min) / span_mu
        L_norm  = (Ls  - L_min)  / span_L
        penalty = alpha * mu_norm + (1.0 - alpha) * L_norm
        scores = score_core + lambda_muL * penalty
    else:
        scores = score_core
    
    # Keep all metadata for Run Best and saving
    ranked_local = [] # Ranked list
    for r, s, cn, re in zip(results, scores, conv_norm, rms_e):
        item = {
            "L": int(r["L"]),
            "mu": float(r["mu"]),
            "conv_ms": float(r["conv_ms"]),
            "sse_db": float(r["sse_db"]),
            "power_anc_off": float(r["in_power"]),
            "power_anc_on": float(r["out_power"]),
            "score": float(s),
            "conv_norm": float(cn),
            "rms_e": float(re),
            "algorithm": r.get("algorithm",""),
            "source": r.get("source",""),
            "noise_label": r.get("noise_label",""),
            "wav_path": r.get("wav_path","")
        }
        ranked_local.append(item)
    
    # Sort by score (lower is better)
    ranked_local.sort(key=lambda d: d["score"])
    
    # Return ranked list
    return ranked_local

def count_unique_combos(combos):
    noise_types = set()
    for (alg, src, nlabel, _) in combos:
        if src == "Stationary":
            noise_types.add((alg, "Stationary", nlabel))
        else:
            noise_types.add((alg, "WAV", nlabel))
    return len(noise_types)