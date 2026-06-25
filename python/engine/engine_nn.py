from pathlib import Path

def get_noise_label(path):
    # Known noise names
    name = Path(path).stem.lower()

    if "airport" in name:
        return "airport"
    if "street" in name:
        return "street"
    if "subway" in name:
        return "subway"

    return "unknown"

def inspect_dataset_summary(dataset_root):
    # Dataset folders
    root = Path(dataset_root)
    train_dir = root / "train"
    validate_dir = root / "validate"

    # WAV files
    train_files = sorted(train_dir.glob("*.wav")) if train_dir.exists() else []
    validate_files = sorted(validate_dir.glob("*.wav")) if validate_dir.exists() else []

    # Noise labels
    labels = [get_noise_label(p) for p in train_files + validate_files]
    unique_labels = sorted(set(labels))

    # Folder check
    dataset_ok = train_dir.exists() and validate_dir.exists()

    return {
        "dataset_ok": dataset_ok,
        "train_count": len(train_files),
        "validate_count": len(validate_files),
        "classes": unique_labels,
    }

def preprocess_dataset(dataset_root, processed_root, target_fs, crop_sec):
    # Backend placeholder
    return {
        "ok": False,
        "message": "Preprocessing not connected yet",
    }

def start_training(config):
    # Backend placeholder
    return {
        "ok": False,
        "message": "Training not connected yet",
    }

def run_validation(checkpoint_path):
    # Backend placeholder
    return {
        "ok": False,
        "message": "Validation not connected yet",
    }