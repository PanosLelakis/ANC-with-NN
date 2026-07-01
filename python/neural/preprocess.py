from pathlib import Path
import math
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from joblib import Parallel, delayed

def read_wav_mono(path):
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

def resample_if_needed(x, source_fs, target_fs):
    # Check sampling rate
    source_fs = int(source_fs)
    target_fs = int(target_fs)

    if source_fs == target_fs:
        return x.astype(np.float32)

    # Resample
    gcd = math.gcd(source_fs, target_fs) # greatest common divisor
    up = target_fs // gcd
    down = source_fs // gcd

    y = resample_poly(x, up, down) # resample_poly instead of resample better for non-periodic signals
    return y.astype(np.float32)

def crop_or_loop(x, target_len):
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

def normalize_unit_power(x):
    # Remove DC
    x = x.astype(np.float32)
    x = x - np.mean(x)

    # Unit power
    power = float(np.mean(x ** 2))

    if power < 1e-12:
        raise ValueError("Silent or near-silent audio file")

    x = x / np.sqrt(power + 1e-12)
    return x.astype(np.float32)

def process_one_file(input_path, output_path, target_fs, crop_sec):
    # Load audio and convert to mono
    source_fs, x = read_wav_mono(input_path)

    # Original duration
    original_duration = len(x) / float(source_fs)

    # Resample audio
    x = resample_if_needed(x, source_fs, target_fs)

    # Crop audio
    target_len = int(float(crop_sec) * int(target_fs))
    x = crop_or_loop(x, target_len)

    # Normalize audio
    x = normalize_unit_power(x)

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

def process_one_job(job):
    # Worker job
    split, input_path, output_path, target_fs, crop_sec = job

    try:
        process_one_file(
            input_path=input_path,
            output_path=output_path,
            target_fs=target_fs,
            crop_sec=crop_sec
        )

        return {
            "ok": True,
            "split": split,
            "original_path": str(input_path)
        }

    except Exception as e:
        return {
            "ok": False,
            "split": split,
            "original_path": str(input_path),
            "error": str(e)
        }

def preprocess_noise_dataset(dataset_root, processed_root, target_fs=16000, crop_sec=40, progress_callback=None, n_jobs=-1):
    # Dataset paths
    dataset_root = Path(dataset_root)
    processed_root = Path(processed_root)

    target_fs = int(target_fs)
    crop_sec = float(crop_sec)

    # Split folders
    split_names = ["train", "validate"]

    all_jobs = []

    for split in split_names:
        split_dir = dataset_root / split

        wav_files = sorted(split_dir.rglob("*.wav"))

        for wav_path in wav_files:
            rel_path = wav_path.relative_to(split_dir)
            out_path = processed_root / split / rel_path
            all_jobs.append((split, wav_path, out_path, target_fs, crop_sec))

    # Output folders
    processed_root.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    skipped = []

    total = len(all_jobs)

    # Parallel preprocessing
    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        return_as="generator_unordered")(
        delayed(process_one_job)(job)
        for job in all_jobs
    )

    for idx, item in enumerate(results, start=1):
        # Finished job
        if item["ok"]:
            processed_count += 1
        else:
            skipped.append({
                "split": item["split"],
                "original_path": item["original_path"],
                "error": item["error"]
            })

        # Progress update
        if progress_callback is not None:
            pct = 100.0 * idx / float(total)

            try:
                progress_callback(pct, idx, total)
            except TypeError:
                progress_callback(pct)

    return {
        "total_files": total,
        "processed_files": processed_count,
        "skipped_files": len(skipped),
        "processed_root": str(processed_root),
        "skipped": skipped
    }