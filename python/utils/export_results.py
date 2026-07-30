import os
import pandas as pd
import json

# Small tolerance so values close to duration are treated as non-converged
CONV_MS_TOL = 1e-6

def remove_non_converged_cases(group, duration_ms=None):
    # Keep only rows with a valid convergence time
    group = group[group["conv_ms"].notna()].copy()

    # If duration is known, remove cases that reached the end of the simulation
    if duration_ms is not None:
        group = group[group["conv_ms"] < float(duration_ms) - CONV_MS_TOL].copy()

    # Return only useful rows for the performance-range table
    return group

def read_run_summary(run_path):
    # Read the first line to check if the CSV contains the Excel separator hint.
    with open(run_path, "r", encoding="utf-8-sig") as f:
        first_line = f.readline().strip().lower()

    # Skip the first line only when it is the Excel separator hint.
    skiprows = 1 if first_line == "sep=," else 0

    # Read the actual CSV content into a pandas DataFrame.
    df = pd.read_csv(run_path, skiprows=skiprows)

    # Support older run_summary files that used "source" instead of "noise_source".
    if "noise_source" not in df.columns and "source" in df.columns:
        df["noise_source"] = df["source"]

    # Convert the divergence column to real boolean values.
    if "divergence" in df.columns:
        df["divergence"] = (
            df["divergence"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )

    # Convert numeric columns to numeric values when they exist.
    for col in ["mu", "L", "conv_ms", "sse_db", "score", "avg_pnc_dbr"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Return the cleaned DataFrame.
    return df

def write_excel_csv(df, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        # Excel separator
        f.write("sep=,\n")

        # Write DataFrame content
        df.to_csv(f, index=False)

def fmt_sig(value, digits):
    # Return empty text for missing values
    if pd.isna(value):
        return ""

    # Format the number with the requested significant digits
    return f"{float(value):.{int(digits)}g}"

def fmt_int(value):
    # Return empty text for missing values
    if pd.isna(value):
        return ""

    # Format L as integer
    return str(int(float(value)))

def parse_band_attenuation(value):
    # Return empty dictionary for missing values.
    if pd.isna(value):
        return {}

    # If already dictionary, return it.
    if isinstance(value, dict):
        return value

    # Try to parse JSON string.
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Return empty dictionary if parsing fails.
    return {}

def add_band_columns(df):
    # Work on a copy.
    df = df.copy()

    # If the band attenuation column does not exist, return unchanged DataFrame.
    if "band_attenuation" not in df.columns:
        return df, []

    # Parse JSON strings into dictionaries.
    parsed_bands = df["band_attenuation"].apply(parse_band_attenuation)

    # Collect all band labels that appear in the run.
    band_labels = []

    for band_dict in parsed_bands:
        for key in band_dict.keys():
            if key not in band_labels:
                band_labels.append(key)

    # Sort bands by their lower frequency.
    def band_start(label):
        try:
            return float(str(label).split("-")[0])
        except Exception:
            return 0.0

    band_labels = sorted(band_labels, key=band_start)

    # Create one numeric column per band.
    for band in band_labels:
        col_name = f"Band {band} (dBr)"
        df[col_name] = parsed_bands.apply(
            lambda d: pd.to_numeric(d.get(band, None), errors="coerce")
        )

    # Return DataFrame and generated band columns.
    return df, [f"Band {band} (dBr)" for band in band_labels]

def format_noise_impact_table(df, include_mu_L=False, band_cols=None):
    # Use empty list if no band columns exist.
    if band_cols is None:
        band_cols = []

    # Create output DataFrame.
    out = pd.DataFrame()

    # Always include noise.
    out["Noise"] = df["noise_label"].astype(str)

    # Include μ and L only for noise_impact_2.
    if include_mu_L:
        out["μ"] = df["mu"].apply(lambda x: fmt_sig(x, 3))
        out["L"] = df["L"].apply(fmt_int)
    else:
        out["Avg. PNC (dBr)"] = df["avg_pnc_dbr"].apply(lambda x: fmt_sig(x, 4))

    # Common metrics.
    out["SSE (dBr)"] = df["sse_db"].apply(lambda x: fmt_sig(x, 4))
    out["Conv. Speed (ms)"] = df["conv_ms"].apply(lambda x: fmt_sig(x, 4))
    out["Score"] = df["score"].apply(lambda x: fmt_sig(x, 4))

    # Add one column per frequency band.
    for col in band_cols:
        out[col] = df[col].apply(lambda x: fmt_sig(x, 4))

    # Return formatted table.
    return out

def latex_escape(text):
    # Convert input to string
    text = str(text)

    # Escape special LaTeX characters
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }

    # Apply all replacements.
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Return LaTeX-safe text.
    return text

def build_algorithm_performance_range(non_divergent, duration_ms=None):
    # Define the final columns for the thesis-oriented CSV.
    columns = [
        "Algorithm",
        "μ",
        "L",
        "Conv. Speed (ms)",
        "SSE (dBr)",
        "Score",
        "Case"
    ]

    # Return an empty table with the correct columns if there are no valid simulations.
    if non_divergent.empty:
        return pd.DataFrame(columns=columns)

    # Work on a copy so the original DataFrame is not modified.
    non_divergent = non_divergent.copy()

    # Keep only rows with valid values.
    non_divergent = non_divergent[
        non_divergent["conv_ms"].notna()
        & non_divergent["sse_db"].notna()
        & non_divergent["score"].notna()
    ].copy()

    # If simulation duration is known, remove cases that reached the end of the simulation.
    # These cases are treated as non-converged and are not useful in the performance range.
    if duration_ms is not None:
        non_divergent = non_divergent[
            non_divergent["conv_ms"] < float(duration_ms) - 1e-6
        ].copy()

    non_divergent = non_divergent[
        non_divergent["conv_ms"] > 20.0 + 1e-6
    ].copy()

    # Return empty table if all rows were removed.
    if non_divergent.empty:
        return pd.DataFrame(columns=columns)

    # Store selected rows.
    selected_rows = []

    # Process each algorithm separately.
    for algorithm, group in non_divergent.groupby("algorithm", sort=True):
        group = group.copy()

        # 1) Top 3 simulations by score.
        top3 = group.sort_values("score", ascending=True).head(3)

        for idx, (_, row) in enumerate(top3.iterrows(), start=1):
            row = row.copy()
            row["case_label"] = f"Top Score {idx}"
            selected_rows.append(row)

        # 2) Best convergence speed.
        best_conv = group.sort_values("conv_ms", ascending=True).head(1)

        for _, row in best_conv.iterrows():
            row = row.copy()
            row["case_label"] = "Best Conv. Speed"
            selected_rows.append(row)

        # 3) Best SSE.
        # More negative SSE is better.
        best_sse = group.sort_values("sse_db", ascending=True).head(1)

        for _, row in best_sse.iterrows():
            row = row.copy()
            row["case_label"] = "Best SSE"
            selected_rows.append(row)

        # 4) Worst score that is still non-divergent and converged.
        worst_score = group.sort_values("score", ascending=False).head(1)

        for _, row in worst_score.iterrows():
            row = row.copy()
            row["case_label"] = "Worst Score"
            selected_rows.append(row)

    # If nothing was selected, return empty table.
    if not selected_rows:
        return pd.DataFrame(columns=columns)

    # Convert selected rows to DataFrame.
    selected_df = pd.DataFrame(selected_rows)

    # Build final formatted table.
    out = pd.DataFrame({
        "Algorithm": selected_df["algorithm"].astype(str),
        "μ": selected_df["mu"].apply(lambda x: fmt_sig(x, 6)),
        "L": selected_df["L"].apply(fmt_int),
        "Conv. Speed (ms)": selected_df["conv_ms"].apply(lambda x: fmt_sig(x, 4)),
        "SSE (dBr)": selected_df["sse_db"].apply(lambda x: fmt_sig(x, 4)),
        "Score": selected_df["score"].apply(lambda x: fmt_sig(x, 5)),
        "Case": selected_df["case_label"].astype(str),
    })

    return out[columns]

def write_algorithm_performance_range_latex(table_df, path):
    # Open the LaTeX output text file.
    with open(path, "w", encoding="utf-8") as f:
        # Write xltabular preamble for six columns.
        f.write(r"\begin{xltabular}{\textwidth}{%" + "\n")
        f.write(r"    >{\RaggedRight\arraybackslash\bfseries}p{0.14\textwidth}" + "\n")
        f.write(r"    Y Y Y Y Y Y" + "\n")
        f.write(r"}" + "\n")

        # Write caption and label.
        f.write(r"\caption{Ενδεικτικό λειτουργικό εύρος και απόδοση των αλγορίθμων στο Multi Run.}" + "\n")
        f.write(r"\label{tab:algorithm_performance_range}\\" + "\n\n")

        # Write first header.
        f.write(r"\toprule" + "\n")
        f.write(r"\textbf{Algorithm} & \textbf{$\mu$} & \textbf{$L$} & \textbf{Conv. Speed (ms)} & \textbf{SSE (dBr)} & \textbf{Score} & \textbf{Case} \\" + "\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\endfirsthead" + "\n\n")

        # Write repeated header for the next pages.
        f.write(r"\toprule" + "\n")
        f.write(r"\textbf{Algorithm} & \textbf{$\mu$} & \textbf{$L$} & \textbf{Conv. Speed (ms)} & \textbf{SSE (dBr)} & \textbf{Score} & \textbf{Case} \\" + "\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\endhead" + "\n\n")

        # Write one LaTeX row per simulation.
        for _, row in table_df.iterrows():
            f.write(
                f"{latex_escape(row['Algorithm'])} & "
                f"{row['μ']} & "
                f"{row['L']} & "
                f"{row['Conv. Speed (ms)']} & "
                f"{row['SSE (dBr)']} & "
                f"{row['Score']} & "
                f"{latex_escape(row['Case'])} \\\\"
                "\n"
            )

        # Write table ending.
        f.write("\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{xltabular}" + "\n")

def export_thesis_tables(results_dir="results", duration_s=None):
    # Build the path of the main run summary file
    run_path = os.path.join(results_dir, "run_summary.csv")

    # Stop if run_summary.csv does not exist
    if not os.path.exists(run_path):
        return

    # Read and clean run_summary.csv
    df = read_run_summary(run_path)

    # Convert duration from seconds to milliseconds, if it was provided
    duration_ms = None if duration_s is None else float(duration_s) * 1000.0

    # Keep only completed simulations and divergence cases.
    valid = df[df["status"].astype(str).isin(["ok", "diverged"])].copy()

    # Keep only non-divergent simulations for best-performance tables.
    non_divergent = valid[(valid["divergence"] == False) & valid["score"].notna()].copy()

    # Best per algorithm/source/noise.
    best_by_combo = (
        non_divergent
        .sort_values("score")
        .groupby(["algorithm", "noise_source", "noise_label"], as_index=False)
        .first()
    )

    # Save best-by-combo table.
    write_excel_csv(best_by_combo, os.path.join(results_dir, "best_by_combo.csv"))

    # Best per noise.
    best_by_noise = (
        non_divergent
        .sort_values("score")
        .groupby(["noise_label"], as_index=False)
        .first()
    )

    # Save best-by-noise table.
    write_excel_csv(best_by_noise, os.path.join(results_dir, "best_by_noise.csv"))

    # Top 10 overall.
    top10 = (
        non_divergent
        .sort_values("score")
        .head(10)
    )

    # Save top-10 table.
    write_excel_csv(top10, os.path.join(results_dir, "top10_results.csv"))

    # Select divergence cases
    div_cases = valid[
        valid["divergence"] == True
    ].copy()

    # Save divergence cases table.
    write_excel_csv(div_cases, os.path.join(results_dir, "divergence_cases.csv"))

    # Build thesis-oriented table with four best and one worst non-divergent case per algorithm.
    algorithm_range = build_algorithm_performance_range(
        non_divergent,
        duration_ms=duration_ms
    )

    # Save the thesis-oriented table as CSV.
    write_excel_csv(algorithm_range, os.path.join(results_dir, "algorithm_performance_range.csv"))

    # Save the same table as LaTeX code in a TXT file.
    write_algorithm_performance_range_latex(
        algorithm_range,
        os.path.join(results_dir, "algorithm_performance_range_latex.txt")
    )

    # Save thesis tables for the effect of noise type.
    export_noise_impact_tables(non_divergent, results_dir)

def write_noise_impact_latex(table_df, path, caption, label):
    # Number of columns in the table.
    n_cols = len(table_df.columns)

    # First column is wider, remaining columns use Y.
    col_spec = (
        r">{\RaggedRight\arraybackslash\bfseries}p{0.18\textwidth}"
        + "\n    "
        + " ".join(["Y"] * (n_cols - 1))
    )

    # Convert headers to LaTeX.
    latex_headers = []

    for col in table_df.columns:
        if col == "μ":
            latex_headers.append(r"\textbf{$\mu$}")
        elif col == "L":
            latex_headers.append(r"\textbf{$L$}")
        else:
            latex_headers.append(r"\textbf{" + latex_escape(col) + r"}")

    header_line = " & ".join(latex_headers) + r" \\"

    # Write LaTeX code.
    with open(path, "w", encoding="utf-8") as f:
        f.write(r"\begin{xltabular}{\textwidth}{%" + "\n")
        f.write("    " + col_spec + "\n")
        f.write(r"}" + "\n")
        f.write(r"\caption{" + caption + r"}" + "\n")
        f.write(r"\label{" + label + r"}\\" + "\n\n")

        f.write(r"\toprule" + "\n")
        f.write(header_line + "\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\endfirsthead" + "\n\n")

        f.write(r"\toprule" + "\n")
        f.write(header_line + "\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\endhead" + "\n\n")

        for _, row in table_df.iterrows():
            values = [latex_escape(row[col]) for col in table_df.columns]
            f.write(" & ".join(values) + r" \\" + "\n")

        f.write("\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{xltabular}" + "\n")

def export_noise_impact_tables(non_divergent, results_dir):
    # Keep only FxNLMS results.
    fxnlms = non_divergent[
        non_divergent["algorithm"].astype(str) == "FxNLMS"
    ].copy()

    # Stop if there are no FxNLMS results.
    if fxnlms.empty:
        return

    # Expand band attenuation JSON into one column per band.
    fxnlms, band_cols = add_band_columns(fxnlms)

    # ------------------------------------------------------------
    # noise_impact_1.csv
    # Same μ, L, algorithm, duration, and a.
    # One row per noise, sorted by score.
    # ------------------------------------------------------------

    agg_dict = {
        "avg_pnc_dbr": "mean",
        "sse_db": "mean",
        "conv_ms": "mean",
        "score": "mean",
    }

    for col in band_cols:
        agg_dict[col] = "mean"

    noise_impact_1_raw = (
        fxnlms
        .groupby("noise_label", as_index=False)
        .agg(agg_dict)
        .sort_values("score", ascending=True)
    )

    noise_impact_1 = format_noise_impact_table(
        noise_impact_1_raw,
        include_mu_L=False,
        band_cols=band_cols
    )

    write_excel_csv(
        noise_impact_1,
        os.path.join(results_dir, "noise_impact_1.csv")
    )

    write_noise_impact_latex(
        noise_impact_1,
        os.path.join(results_dir, "noise_impact_1_latex.txt"),
        caption=r"Επίδραση του είδους θορύβου για σταθερές παραμέτρους του FxNLMS.",
        label=r"tab:noise_impact_1"
    )

    # ------------------------------------------------------------
    # noise_impact_2.csv
    # Best μ, L per noise, based on score.
    # ------------------------------------------------------------

    noise_impact_2_raw = (
        fxnlms
        .sort_values("score", ascending=True)
        .groupby("noise_label", as_index=False)
        .first()
        .sort_values("score", ascending=True)
    )

    noise_impact_2 = format_noise_impact_table(
        noise_impact_2_raw,
        include_mu_L=True,
        band_cols=band_cols
    )

    write_excel_csv(
        noise_impact_2,
        os.path.join(results_dir, "noise_impact_2.csv")
    )

    write_noise_impact_latex(
        noise_impact_2,
        os.path.join(results_dir, "noise_impact_2_latex.txt"),
        caption=r"Βέλτιστη παραμετροποίηση FxNLMS ανά είδος θορύβου.",
        label=r"tab:noise_impact_2"
    )