import numpy as np
from joblib import Parallel, delayed
from engine.engine_single import simulate_once

def _linspace_inclusive(a, b, n):
    n = int(n)
    if n <= 1: return np.array([float(a)], dtype=float)
    return np.linspace(float(a), float(b), n)

def _logspace_inclusive(a, b, n):
    n = int(n)
    if n <= 1: return np.array([float(a)], dtype=float)
    return np.exp(np.linspace(np.log(float(a)), np.log(float(b)), n))

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

def score_results(results, duration_s, w_conv=0.5, w_sse=0.5, normalize="dataset"):
    duration_ms = 1000.0 * float(duration_s)
    conv = np.array([r["conv_ms"] for r in results], dtype=float)
    sse_db = np.array([r["sse_db"] for r in results], dtype=float)
    rms_e = np.power(10.0, sse_db / 20.0)  # [0,1], lower is better

    if normalize == "dataset":
        finite = np.isfinite(conv)
        denom = np.max(conv[finite]) if np.any(finite) else duration_ms
        denom = max(denom, 1e-6)
        conv_norm = np.clip(conv / denom, 0, 10)
    else:
        conv_norm = np.clip(conv / max(duration_ms, 1e-6), 0, 10)

    scores = w_conv * conv_norm + w_sse * rms_e
    ranked = []
    for r, s, cn, re in zip(results, scores, conv_norm, rms_e):
        x = dict(r); x["score"] = float(s); x["conv_norm"] = float(cn); x["rms_e"] = float(re)
        ranked.append(x)
    ranked.sort(key=lambda d: d["score"])
    return ranked

def run_grid_parallel(algorithm_name, noise_source, noise_type, noise_wav_path,
                      duration, mu_min, mu_max, mu_steps, L_min, L_max, L_steps,
                      mu_scale="log", replicates=1, n_jobs=-1, base_seed=1234):
    mu_vals, L_vals, grid = build_grid(mu_min, mu_max, mu_steps, L_min, L_max, L_steps, mu_scale)
    jobs = []
    for L in L_vals:
        for mu in mu_vals:
            for rep in range(int(replicates)):
                seed = base_seed + 7919*rep + 97*int(L) + int(1e6*float(mu))
                jobs.append((algorithm_name, int(L), float(mu),
                             noise_source, noise_type, noise_wav_path,
                             duration, seed))
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(simulate_once)(*args) for args in jobs
    )
    aggregated = average_replicates(results)
    ranked = score_results(aggregated, duration_s=duration, w_conv=0.5, w_sse=0.5, normalize="dataset")
    return {"ranked": ranked, "mu_vals": mu_vals, "L_vals": L_vals}