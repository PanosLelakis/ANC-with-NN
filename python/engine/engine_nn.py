from pathlib import Path
from neural.preprocess import preprocess_noise_dataset

def preprocess_dataset(dataset_root, processed_root, target_fs, crop_sec, progress_callback=None):
    # Run preprocessing
    try:
        result = preprocess_noise_dataset(
            dataset_root=dataset_root,
            processed_root=processed_root,
            target_fs=int(target_fs),
            crop_sec=float(crop_sec),
            progress_callback=progress_callback,
        )

        message = (
            f"Preprocessing completed. "
            f"Processed {result['processed_files']}/{result['total_files']} files. "
            f"Skipped {result['skipped_files']} files."
        )

        return {
            "ok": True,
            "message": message,
            "result": result,
        }

    except Exception as e:
        return {
            "ok": False,
            "message": str(e),
            "result": None,
        }

def start_training(config):
    # Backend placeholder
    return {
        "ok": False,
        "message": "Training not connected yet",
    }

def run_validation(checkpoint_path, processed_root):
    # Backend placeholder
    return {
        "ok": False,
        "message": "Validation not connected yet",
    }