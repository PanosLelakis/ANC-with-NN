import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import plot as plot_utils
from utils.export_results import export_thesis_tables
from utils.logger import log_case
from utils.result_saver import append_run_summary, safe_name, save_case_artifacts
from engine.engine_single import run_anc, make_noise

def get_noise_key(source, noise_label, wav_path):
    # Keep WAV files with the same filename separate
    path_key = (wav_path if source == "WAV" else "")

    # Return unique noise key
    return (source, noise_label, path_key)

def build_noise_bank(combinations, duration, paths):
    # Read path sampling rate
    fs = int(paths[0])

    # Compute signal length
    sample_count = int(float(duration) * fs)

    # Initialize noise bank
    noise_bank = {}

    # Process selected noises
    for _, source, noise_label, wav_path in combinations:
        # Build unique noise key
        key = get_noise_key(source, noise_label, wav_path)

        # Skip already prepared noise
        if key in noise_bank:
            continue

        # Create or load noise once
        noise = make_noise(
            sample_count,
            fs,
            source,
            noise_label,
            wav_path
        ).astype(np.float32,copy=False)

        # Prevent accidental modification
        noise.setflags(write=False)

        # Store prepared noise
        noise_bank[key] = noise

    # Return prepared noises
    return noise_bank

def get_result_key(result):
    # Build result key
    return (
        result.get("algorithm", ""),
        result.get("source", ""),
        result.get("noise_label", "")
    )

def group_results_by_combo(results):
    # Initialize grouped results
    grouped_results = {}

    # Read every result
    for result in results:
        # Build combination key
        key = get_result_key(result)

        # Create group when missing
        if key not in grouped_results:
            grouped_results[key] = []

        # Add result to group
        grouped_results[key].append(result)

    # Return grouped results
    return grouped_results

def select_best_by_combo(ranked_results):
    # Initialize best results
    best_by_combo = {}

    # Read ranked results in score order
    for result in ranked_results:
        # Build combination key
        key = get_result_key(result)

        # Keep first result for this combination
        if key not in best_by_combo:
            # Store best result
            best_by_combo[key] = result

    # Return best result per combination
    return best_by_combo

def get_unique_combinations(combinations):
    # Initialize unique combinations
    unique_combinations = []

    # Initialize stored keys
    seen = set()

    # Read all combinations
    for combination in combinations:
        # Unpack combination
        algorithm, source, noise_label, _ = combination

        # Build unique key
        key = (algorithm, source, noise_label)

        # Skip duplicate combination
        if key in seen:
            continue

        # Store combination key
        seen.add(key)

        # Store full combination
        unique_combinations.append(combination)

    # Return unique combinations
    return unique_combinations

def build_run_combinations(algorithms, colors, wav_paths):
    # Initialize combinations
    combinations = []

    # Add stationary combinations
    for algorithm in algorithms:
        for color in colors:
            combinations.append((algorithm, "Stationary", color, ""))

    # Add WAV combinations
    for algorithm in algorithms:
        for wav_path in wav_paths:
            noise_label = os.path.basename(wav_path)
            combinations.append((algorithm, "WAV", noise_label, wav_path))

    # Return combinations
    return combinations

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

def score_results(
    results,
    duration_s,
    alpha=0.5
):
    # Stop when no results exist
    if not results:
        return []

    # Separate adaptive and NN results
    adaptive_results = [
        result
        for result in results
        if result.get("algorithm")
        != "Neural Network"
    ]

    neural_results = [
        result
        for result in results
        if result.get("algorithm")
        == "Neural Network"
    ]

    # Initialize ranked results
    ranked_local = []

    # Rank adaptive algorithm results
    if adaptive_results:
        # Normalize preference factor
        alpha = float(
            np.clip(
                alpha,
                0.0,
                1.0
            )
        )

        # Read metrics
        conv = np.array(
            [
                result["conv_ms"]
                for result in adaptive_results
            ],
            dtype=float
        )

        sse_db = np.array(
            [
                result["sse_db"]
                for result in adaptive_results
            ],
            dtype=float
        )

        divergence_flags = np.array(
            [
                bool(
                    result.get(
                        "divergence",
                        False
                    )
                )
                for result in adaptive_results
            ],
            dtype=bool
        )

        # Convert duration to milliseconds
        duration_ms = (
            1000.0
            * float(duration_s)
        )

        # Replace invalid convergence values
        conv = np.nan_to_num(
            conv,
            nan=duration_ms,
            posinf=duration_ms,
            neginf=duration_ms
        )

        # Treat immediate convergence as unreliable
        min_valid_conv_ms = 20.0

        conv = np.where(
            conv <= min_valid_conv_ms + 1e-6,
            duration_ms,
            conv
        )

        # Replace invalid SSE values
        sse_db = np.nan_to_num(
            sse_db,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # Normalize convergence
        conv_min = np.min(conv)
        conv_max = np.max(conv)
        conv_span = max(
            conv_max - conv_min,
            1e-12
        )

        conv_norm = (
            conv - conv_min
        ) / conv_span

        # Normalize SSE
        sse_min = np.min(sse_db)
        sse_max = np.max(sse_db)
        sse_span = max(
            sse_max - sse_min,
            1e-12
        )

        sse_norm = (
            sse_db - sse_min
        ) / sse_span

        # Compute adaptive score
        scores = (
            alpha * conv_norm
            + (1.0 - alpha) * sse_norm
        )

        scores = np.where(
            divergence_flags,
            scores + 1e6,
            scores
        )

        # Build adaptive rows
        for (
            result,
            score,
            conv_value,
            sse_value,
            conv_used
        ) in zip(
            adaptive_results,
            scores,
            conv_norm,
            sse_norm,
            conv
        ):
            ranked_local.append({
                "L": int(result["L"]),
                "mu": float(result["mu"]),
                "conv_ms": float(conv_used),
                "sse_db": float(
                    result["sse_db"]
                ),
                "power_anc_off": float(
                    result["in_power"]
                ),
                "power_anc_on": float(
                    result["out_power"]
                ),
                "score": float(score),
                "conv_norm": float(
                    conv_value
                ),
                "sse_norm": float(
                    sse_value
                ),
                "algorithm": result.get(
                    "algorithm",
                    ""
                ),
                "source": result.get(
                    "source",
                    ""
                ),
                "noise_label": result.get(
                    "noise_label",
                    ""
                ),
                "wav_path": result.get(
                    "wav_path",
                    ""
                ),
                "divergence": bool(
                    result.get(
                        "divergence",
                        False
                    )
                ),
                "status": result.get(
                    "status",
                    (
                        "diverged"
                        if bool(
                            result.get(
                                "divergence",
                                False
                            )
                        )
                        else "ok"
                    )
                ),
                "save_path": result.get(
                    "save_path",
                    ""
                ),
                "avg_pnc_dbr": result.get(
                    "avg_pnc_dbr"
                ),
                "band_attenuation": result.get(
                    "band_attenuation",
                    {}
                )
            })

        # Sort adaptive results
        ranked_local.sort(
            key=lambda item: item["score"]
        )

    # Add Neural Network results
    for result in neural_results:
        ranked_local.append({
            "L": 0,
            "mu": 0.0,
            "conv_ms": None,
            "sse_db": float(
                result["sse_db"]
            ),
            "power_anc_off": float(
                result["in_power"]
            ),
            "power_anc_on": float(
                result["out_power"]
            ),
            "score": float("nan"),
            "conv_norm": float("nan"),
            "sse_norm": float("nan"),
            "algorithm": "Neural Network",
            "source": result.get(
                "source",
                ""
            ),
            "noise_label": result.get(
                "noise_label",
                ""
            ),
            "wav_path": result.get(
                "wav_path",
                ""
            ),
            "divergence": bool(
                result.get(
                    "divergence",
                    False
                )
            ),
            "status": result.get(
                "status",
                (
                    "diverged"
                    if bool(
                        result.get(
                            "divergence",
                            False
                        )
                    )
                    else "ok"
                )
            ),
            "save_path": result.get(
                "save_path",
                ""
            ),
            "avg_pnc_dbr": result.get(
                "avg_pnc_dbr"
            ),
            "band_attenuation": result.get(
                "band_attenuation",
                {}
            )
        })

    return ranked_local

def save_multi_summary(ranked_results, duration, results_root):
    # Build summary file path
    summary_path = os.path.join(results_root, "run_summary.csv")

    # Remove previous summary
    if os.path.exists(summary_path):
        os.remove(summary_path)

    # Process ranked results
    for result in ranked_results:
        # Read ANC OFF power
        input_power = float(result.get("power_anc_off", 0.0))

        # Read ANC ON power
        output_power = float(result.get("power_anc_on", 0.0))

        # Compute attenuation in dB
        attenuation_db = 10.0 * np.log10((output_power + 1e-12) / (input_power + 1e-12))

        # Copy result
        row = dict(result)

        # Add run type
        row["run_kind"] = "multi"

        # Add attenuation
        row["attenuation_db"] = float(attenuation_db)

        # Save summary row
        append_run_summary(row, results_root=results_root)

    # Export thesis tables
    export_thesis_tables(results_root, duration_s=duration)

def log_multi_error(metadata, error):
    # Write failed simulation log
    log_case(
        stage="simulate",
        status="error",
        algorithm=metadata.get("algorithm", ""),
        source=metadata.get("source", ""),
        noise_label=metadata.get("noise_label", ""),
        L=metadata.get("L"),
        mu=metadata.get("mu"),
        conv_ms=None,
        sse_db=None,
        exec_time=None,
        in_power=None,
        out_power=None,
        save_path="",
        message=str(error)
    )

def log_multi_result(result, stage):
    # Read divergence flag
    diverged = bool(result.get("divergence", False))

    # Select log status
    status = ("diverged" if diverged else "ok")

    # Use saved status when available
    if stage == "save":
        status = result.get("status", status)

    # Read convergence time
    conv_value = result.get("conv_ms")

    logged_conv = (
        None
        if conv_value is None
        else round(
            float(conv_value),
            2
        )
    )

    # Write log row
    log_case(
        stage=stage,
        status=status,
        algorithm=result.get("algorithm", ""),
        source=result.get("source", ""),
        noise_label=result.get("noise_label", ""),
        L=int(result.get("L", 0)),
        mu=float(result.get("mu", 0.0)),
        conv_ms=logged_conv,
        sse_db=round(float(result.get("sse_db", 0.0)), 2),
        exec_time=None,
        in_power=round(float(result.get("in_power", 0.0)), 3),
        out_power=round(float(result.get("out_power", 0.0)), 3),
        save_path=result.get("save_path", ""),
        message=("Divergence detected." if diverged else ""),
        divergence=diverged)

def build_multi_result(payload, L, mu):
    # Read convergence time
    conv_ms = payload.get("conv_ms")

    # Read steady-state error
    sse_db = payload.get("sse_db")

    # Read divergence status
    divergence = bool(payload.get("divergence", False))

    # Build compact Multi Run result
    return {
        "mu": float(mu),
        "L": int(L),
        "conv_ms": (
            None
            if conv_ms is None
            else float(conv_ms)
        ),
        "sse_db": (float("nan") if sse_db is None else float(sse_db)),
        "in_power": float(payload["in_power"]),
        "out_power": float(payload["out_power"]),
        "fs": int(payload["fs"]),
        "divergence": divergence,
        "status": ("diverged" if divergence else "ok"),
        "save_path": "",
        "avg_pnc_dbr": payload.get("avg_pnc_dbr"),
        "band_attenuation": payload.get("band_attenuation", {})
    }

def run_multi_payload(
    algorithm,
    L,
    mu,
    source,
    noise_label,
    wav_path,
    duration,
    preloaded_noise,
    paths,
    nn_checkpoint_path=None,
    nn_backend="pytorch"
):
    # Select simulation WAV path
    simulation_wav_path = (wav_path if source == "WAV" else "")

    # Run one ANC simulation
    return run_anc(
        algorithm_name=algorithm,
        L=L,
        mu=mu,
        noise_source=source,
        noise_type=noise_label,
        noise_wav_path=simulation_wav_path,
        duration=duration,
        progress_callback=None,
        nn_checkpoint_path=nn_checkpoint_path,
        nn_backend=nn_backend,
        preloaded_noise=preloaded_noise,
        paths=paths
    )

def run_and_save_case(
    algorithm,
    L,
    mu,
    source,
    noise_label,
    wav_path,
    duration,
    results_root,
    preloaded_noise,
    paths,
    nn_checkpoint_path=None,
    nn_backend="pytorch"
):
    # Run complete simulation
    payload = run_multi_payload(
        algorithm=algorithm,
        L=L,
        mu=mu,
        source=source,
        noise_label=noise_label,
        wav_path=wav_path,
        duration=duration,
        nn_checkpoint_path=nn_checkpoint_path,
        nn_backend=nn_backend,
        preloaded_noise=preloaded_noise,
        paths=paths
    )

    # Save simulation artifacts
    metadata = save_case_artifacts(
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

    # Build Multi Run result
    result = build_multi_result(
        payload=payload,
        L=L,
        mu=mu
    )

    # Store saved status
    result["divergence"] = bool(metadata["divergence"])

    # Store saved status text
    result["status"] = metadata["status"]

    # Store result folder
    result["save_path"] = metadata["save_path"]

    # Return result and metadata
    return result, metadata

def run_multi_case(
    algorithm,
    L,
    mu,
    source,
    noise_label,
    wav_path,
    duration,
    preloaded_noise,
    paths,
    save_mode="none",
    results_root="results",
    nn_checkpoint_path=None,
    nn_backend="pytorch"
):
    """
    One simulation job for Multi Run.

    save_mode:
    - "none": run only metrics
    - "all": run full simulation and save this case immediately
    """

    if save_mode == "all":
        # Run and save full case
        result, _ = run_and_save_case(
            algorithm=algorithm,
            L=L,
            mu=mu,
            source=source,
            noise_label=noise_label,
            wav_path=wav_path,
            duration=duration,
            results_root=results_root,
            preloaded_noise=preloaded_noise,
            paths=paths,
            nn_checkpoint_path=nn_checkpoint_path,
            nn_backend=nn_backend
        )

    else:
        # Run simulation
        payload = run_multi_payload(
            algorithm=algorithm,
            L=L,
            mu=mu,
            source=source,
            noise_label=noise_label,
            wav_path=wav_path,
            duration=duration,
            preloaded_noise=preloaded_noise,
            paths=paths,
            nn_checkpoint_path=nn_checkpoint_path,
            nn_backend=nn_backend
        )

        # Keep only Multi Run metrics
        result = build_multi_result(
            payload=payload,
            L=L,
            mu=mu
        )

    result.update(dict(
        algorithm=algorithm,
        source=source,
        noise_label=noise_label,
        wav_path=wav_path
    ))

    return result

def save_one_combo_summary(
    ranked_results,
    combination,
    mu_values,
    L_values,
    results_root
):
    # Unpack combination
    algorithm, source, noise_label, _ = combination

    # Neural Network has no mu-L parameter grid
    if algorithm == "Neural Network":
        return

    # Build output folder
    output_folder = os.path.join(results_root, algorithm, safe_name(noise_label))

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Select combination results
    combination_results = [
        result
        for result in ranked_results
        if result.get("algorithm") == algorithm
        and result.get("source") == source
        and result.get("noise_label") == noise_label
    ]

    # Stop if no results exist
    if not combination_results:
        return

    # Save score heatmap
    plot_utils.plot_hparam_heatmap(
        combination_results,
        mu_values,
        L_values,
        save_dir=output_folder
    )

    # Save convergence plot
    plot_utils.plot_convtime_vs_mu(
        combination_results,
        save_dir=output_folder,
        algorithm_name=algorithm,
        noise_type=noise_label
    )

    # Save SSE plot
    plot_utils.plot_sse_vs_L(
        combination_results,
        save_dir=output_folder,
        algorithm_name=algorithm,
        noise_type=noise_label
    )

def save_combo_summary_plots(
    ranked_results,
    combinations,
    mu_values,
    L_values,
    results_root
):
    # Find unique combinations
    unique_combinations = get_unique_combinations(combinations)

    # Process each combination
    for combination in unique_combinations:
        # Save combination plots
        save_one_combo_summary(
            ranked_results=ranked_results,
            combination=combination,
            mu_values=mu_values,
            L_values=L_values,
            results_root=results_root
        )

def run_multi_sim(
    grid,
    combinations,
    mu_values,
    L_values,
    duration,
    alpha,
    save_mode,
    results_root,
    paths,
    progress_callback=None,
    save_progress_callback=None,
    nn_checkpoint_path=None,
    nn_backend="pytorch"
):
    # Start timer
    start_time = time.time()

    # Normalize save mode
    save_mode = str(save_mode).lower()

    # Separate adaptive and Neural Network combinations
    adaptive_combinations = [
        combination
        for combination in combinations
        if combination[0] != "Neural Network"
    ]

    neural_combinations = [
        combination
        for combination in combinations
        if combination[0] == "Neural Network"
    ]

    # Count simulations
    total = (
        len(grid) * len(adaptive_combinations)
        + len(neural_combinations)
    )

    # Count unique combinations
    unique_combo_count = len(
        get_unique_combinations(combinations)
    )

    # Initialize results
    results = []

    # Initialize completed count
    completed = 0

    # Prepare each selected noise once
    noise_bank = build_noise_bank(
        combinations,
        duration,
        paths
    )

    # Create process pool
    with ProcessPoolExecutor() as executor:
        # Store future metadata
        future_metadata = {}

        # Submit adaptive simulations
        for mu, L in grid:
            for (
                algorithm,
                source,
                noise_label,
                wav_path
            ) in adaptive_combinations:
                # Build noise key
                noise_key = get_noise_key(
                    source,
                    noise_label,
                    wav_path
                )

                # Read prepared noise
                preloaded_noise = (
                    noise_bank[noise_key]
                )

                # Submit simulation
                future = executor.submit(
                    run_multi_case,
                    algorithm,
                    int(L),
                    float(mu),
                    source,
                    noise_label,
                    wav_path,
                    duration,
                    preloaded_noise,
                    paths,
                    save_mode,
                    results_root,
                    None,
                    "pytorch"
                )

                # Store metadata
                future_metadata[future] = {
                    "algorithm": algorithm,
                    "source": source,
                    "noise_label": noise_label,
                    "wav_path": wav_path,
                    "L": int(L),
                    "mu": float(mu)
                }

        # Submit one Neural Network simulation per noise
        for (
            algorithm,
            source,
            noise_label,
            wav_path
        ) in neural_combinations:
            # Build noise key
            noise_key = get_noise_key(
                source,
                noise_label,
                wav_path
            )

            # Read prepared noise
            preloaded_noise = noise_bank[
                noise_key
            ]

            # Submit fixed Neural Network simulation
            future = executor.submit(
                run_multi_case,
                algorithm,
                0,
                0.0,
                source,
                noise_label,
                wav_path,
                duration,
                preloaded_noise,
                paths,
                save_mode,
                results_root,
                nn_checkpoint_path,
                nn_backend
            )

            # Store metadata
            future_metadata[future] = {
                "algorithm": algorithm,
                "source": source,
                "noise_label": noise_label,
                "wav_path": wav_path,
                "L": 0,
                "mu": 0.0
            }

        # Read completed simulations
        for future in as_completed(
            future_metadata
        ):
            metadata = future_metadata[
                future
            ]

            try:
                # Read result
                result = future.result()

                # Add metadata
                result.update(metadata)

                # Store result
                results.append(result)

                # Log simulation
                log_multi_result(
                    result,
                    stage="simulate"
                )

                # Log saved case
                if save_mode == "all":
                    log_multi_result(
                        result,
                        stage="save"
                    )

            except Exception as error:
                log_multi_error(
                    metadata,
                    error
                )

            # Increase completed count
            completed += 1

            # Send progress update
            if progress_callback is not None:
                progress_callback(
                    completed,
                    total
                )

    # Rank completed simulations
    ranked_results = score_results(results=results, duration_s=duration, alpha=alpha)

    # Save summary files
    save_multi_summary(
        ranked_results=ranked_results,
        duration=duration,
        results_root=results_root
    )

    # Select best result per combination
    best_by_combo = select_best_by_combo(ranked_results)

    # Save best results
    if save_mode == "best":
        # Save one best case per combination
        save_best_results(
            ranked_results=ranked_results,
            best_by_combo=best_by_combo,
            combinations=combinations,
            mu_values=mu_values,
            L_values=L_values,
            duration=duration,
            results_root=results_root,
            noise_bank=noise_bank,
            paths=paths,
            progress_callback=save_progress_callback,
            nn_checkpoint_path=nn_checkpoint_path,
            nn_backend=nn_backend
        )

    # Save all summary plots
    elif save_mode == "all":
        # Save plots for every combination
        save_combo_summary_plots(
            ranked_results=ranked_results,
            combinations=combinations,
            mu_values=mu_values,
            L_values=L_values,
            results_root=results_root
        )

    # Compute execution time
    execution_time = time.time() - start_time

    # Return Multi Run results
    return {
        "ranked": ranked_results,
        "execution_time": execution_time,
        "unique_combo_count": unique_combo_count
    }

def save_one_best_case(
    best_result,
    duration,
    results_root,
    noise_bank,
    paths,
    nn_checkpoint_path=None,
    nn_backend="pytorch"
):
    # Read algorithm
    algorithm = best_result.get("algorithm", "")

    # Read noise source
    source = best_result.get("source", "")

    # Read noise label
    noise_label = best_result.get("noise_label", "")

    # Read WAV path
    wav_path = best_result.get("wav_path", "")

    # Read filter parameters
    if algorithm == "Neural Network":
        L = 0
        mu = 0.0
    else:
        L = int(best_result.get("L"))
        mu = float(best_result.get("mu"))

    # Build original noise key
    noise_key = get_noise_key(source, noise_label, wav_path)

    # Read original Multi Run noise
    preloaded_noise = noise_bank[noise_key]

    # Run and save selected case
    _, metadata = run_and_save_case(
        algorithm=algorithm,
        L=L,
        mu=mu,
        source=source,
        noise_label=noise_label,
        wav_path=wav_path,
        duration=duration,
        results_root=results_root,
        preloaded_noise=preloaded_noise,
        paths=paths,
        nn_checkpoint_path=nn_checkpoint_path,
        nn_backend=nn_backend
    )

    # Write save log
    log_case(
        stage="save",
        status=metadata["status"],
        algorithm=algorithm,
        source=source,
        noise_label=noise_label,
        L=L,
        mu=mu,
        conv_ms=metadata["conv_ms"],
        sse_db=metadata["sse_db"],
        exec_time=metadata["exec_time"],
        in_power=metadata["in_power"],
        out_power=metadata["out_power"],
        save_path=metadata["save_path"],
        message=("Divergence detected." if metadata["divergence"] else ""),
        divergence=metadata["divergence"]
    )

    # Return saved metadata
    return metadata

def save_best_case_error(best_result, results_root, error):
    # Read algorithm
    algorithm = best_result.get("algorithm", "")

    # Read noise source
    source = best_result.get("source", "")

    # Read noise label
    noise_label = best_result.get("noise_label", "")

    # Read filter length
    L = int(best_result.get("L"))

    # Read step size
    mu = float(best_result.get("mu"))

    # Build result folder
    output_folder = os.path.join(
        results_root,
        algorithm,
        safe_name(noise_label),
        f"L{L}_mu{mu:.6g}"
    )

    # Create result folder
    os.makedirs(output_folder, exist_ok=True)

    # Build error file path
    error_path = os.path.join(output_folder, "error.txt")

    # Write error file
    with open(error_path, "w", encoding="utf-8") as file:
        # Write error message
        file.write(str(error))

    # Write error log
    log_case(
        stage="save",
        status="error",
        algorithm=algorithm,
        source=source,
        noise_label=noise_label,
        L=L,
        mu=mu,
        conv_ms=None,
        sse_db=None,
        exec_time=None,
        in_power=None,
        out_power=None,
        save_path=output_folder,
        message=str(error)
    )

def log_summary_error(combination, error):
    # Unpack combination
    algorithm, source, noise_label, _ = combination

    # Write summary error log
    log_case(
        stage="save_summary",
        status="error",
        algorithm=algorithm,
        source=source,
        noise_label=noise_label,
        L=None,
        mu=None,
        conv_ms=None,
        sse_db=None,
        exec_time=None,
        in_power=None,
        out_power=None,
        save_path="",
        message=str(error)
    )

def save_best_results(
    ranked_results,
    best_by_combo,
    combinations,
    mu_values,
    L_values,
    duration,
    results_root,
    noise_bank,
    paths,
    progress_callback=None,
    nn_checkpoint_path=None,
    nn_backend="pytorch"
):
    # Start timer
    start_time = time.time()

    # Find unique combinations
    unique_combinations = get_unique_combinations(combinations)

    # Read best results
    best_results = list(best_by_combo.values())

    # Count total save jobs
    total_jobs = len(unique_combinations) + len(best_results)

    # Initialize completed jobs
    completed_jobs = 0

    # Save summary plots
    for combination in unique_combinations:
        try:
            # Save combination plots
            save_one_combo_summary(
                ranked_results=ranked_results,
                combination=combination,
                mu_values=mu_values,
                L_values=L_values,
                results_root=results_root
            )

        except Exception as error:
            # Log plot error
            log_summary_error(combination, error)

        # Increase completed jobs
        completed_jobs += 1

        # Update progress
        if progress_callback is not None:
            progress_callback(completed_jobs, total_jobs)

    # Save best simulation cases
    for best_result in best_results:
        try:
            # Save case with original noise
            save_one_best_case(
                best_result=best_result,
                duration=duration,
                results_root=results_root,
                noise_bank=noise_bank,
                paths=paths,
                nn_checkpoint_path=nn_checkpoint_path,
                nn_backend=nn_backend
            )

        except Exception as error:
            # Save error information
            save_best_case_error(
                best_result=best_result,
                results_root=results_root,
                error=error
            )

        # Increase completed jobs
        completed_jobs += 1

        # Update progress
        if progress_callback is not None:
            progress_callback(completed_jobs, total_jobs)

    # Compute execution time
    execution_time = time.time() - start_time

    # Return save result
    return {
        "saved_jobs": completed_jobs,
        "total_jobs": total_jobs,
        "execution_time": execution_time
    }