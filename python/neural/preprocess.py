from pathlib import Path
import csv
import math
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

def _read_wav_mono(path):
    # Read wav file
    fs, data = wavfile.read(str(path))

    # Convert to float
    data = np.asarray(data)

    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        data = data.astype(np.float32) / scale
    else:
        data = data.astype(np.float32)

    # Convert to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    return int(fs), data.astype(np.float32)

def _resample_if_needed(x, source_fs, target_fs):
    # Check sampling rate
    source_fs = int(source_fs)
    target_fs = int(target_fs)

    if source_fs == target_fs:
        return x.astype(np.float32)

    # Rational resampling
    g = math.gcd(source_fs, target_fs)
    up = target_fs // g
    down = source_fs // g

    y = resample_poly(x, up, down)
    return y.astype(np.float32)

def _crop_or_loop(x, target_len):
    # Fixed length signal
    target_len = int(target_len)

    if len(x) == 0:
        raise ValueError("Empty audio file")

    if len(x) >= target_len:
        return x[:target_len].astype(np.float32)

    # Loop short signals
    reps = int(np.ceil(target_len / len(x)))
    y = np.tile(x, reps)[:target_len]
    return y.astype(np.float32)

def _normalize_unit_power(x):
    # Remove DC
    x = x.astype(np.float32)
    x = x - np.mean(x)

    # Unit power
    power = float(np.mean(x ** 2))

    if power < 1e-12:
        raise ValueError("Silent or near-silent audio file")

    x = x / np.sqrt(power + 1e-12)
    return x.astype(np.float32)

def _process_one_file(input_path, output_path, target_fs, crop_sec):
    # Load audio
    source_fs, x = _read_wav_mono(input_path)

    # Original duration
    original_duration = len(x) / float(source_fs)

    # Resample audio
    x = _resample_if_needed(x, source_fs, target_fs)

    # Crop audio
    target_len = int(float(crop_sec) * int(target_fs))
    x = _crop_or_loop(x, target_len)

    # Normalize audio
    x = _normalize_unit_power(x)

    # Save audio
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(output_path), int(target_fs), x.astype(np.float32))

    return {
        "original_path": str(input_path),
        "processed_path": str(output_path),
        "original_fs": int(source_fs),
        "target_fs": int(target_fs),
        "original_duration_sec": round(float(original_duration), 4),
        "processed_duration_sec": round(float(crop_sec), 4),
        "normalization": "unit_power"
    }

def preprocess_noise_dataset(dataset_root, processed_root, target_fs=16000, crop_sec=40, progress_callback=None):
    # Dataset paths
    dataset_root = Path(dataset_root)
    processed_root = Path(processed_root)

    target_fs = int(target_fs)
    crop_sec = float(crop_sec)

    # Split folders
    split_names = ["train", "validate"]

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    all_jobs = []

    for split in split_names:
        split_dir = dataset_root / split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {split_dir}")

        wav_files = sorted(split_dir.rglob("*.wav"))

        for wav_path in wav_files:
            rel_path = wav_path.relative_to(split_dir)
            out_path = processed_root / split / rel_path
            all_jobs.append((split, wav_path, out_path))

    if not all_jobs:
        raise RuntimeError("No wav files found in train or validate folders")

    # Output folders
    processed_root.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = []

    total = len(all_jobs)

    for idx, (split, input_path, output_path) in enumerate(all_jobs, start=1):
        try:
            row = _process_one_file(
                input_path=input_path,
                output_path=output_path,
                target_fs=target_fs,
                crop_sec=crop_sec,
            )

            row["split"] = split
            rows.append(row)

        except Exception as e:
            skipped.append({
                "split": split,
                "original_path": str(input_path),
                "error": str(e),
            })

        # Progress update
        if progress_callback is not None:
            pct = 100.0 * idx / float(total)
            progress_callback(pct)

    # Metadata file
    metadata_path = processed_root / "metadata.csv"

    fieldnames = [
        "split",
        "original_path",
        "processed_path",
        "original_fs",
        "target_fs",
        "original_duration_sec",
        "processed_duration_sec",
        "normalization"
    ]

    with open(metadata_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "total_files": total,
        "processed_files": len(rows),
        "skipped_files": len(skipped),
        "processed_root": str(processed_root),
        "metadata_path": str(metadata_path),
        "skipped": skipped,
    }