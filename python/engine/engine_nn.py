import time
from pathlib import Path
from neural.preprocess import preprocess_noise_dataset
from neural.nn_logger import log_nn_event
from neural.train import train_model

def preprocess_dataset(dataset_root, processed_root, target_fs, crop_sec, progress_callback=None):
    # Preprocessing parameters
    params = {
        "target_fs": int(target_fs),
        "crop_sec": float(crop_sec),
        "total_files": ""
    }

    # Start measuring execution time
    start_time = time.time()

    try:
        result = preprocess_noise_dataset(
            dataset_root=dataset_root,
            processed_root=processed_root,
            target_fs=int(target_fs),
            crop_sec=float(crop_sec),
            progress_callback=progress_callback,
            n_jobs=-1
        )

        # Log file count
        params["total_files"] = result["total_files"]

        # Calculate execution time in minutes
        execution_time = time.time() - start_time

        message = (
            f"Preprocessing completed. "
            f"Processed {result['processed_files']}/{result['total_files']} files. "
            f"Skipped {result['skipped_files']} files."
        )

        log_nn_event(
            stage="preprocessing",
            status="success",
            execution_time=execution_time,
            parameters=params,
            message=message
        )

        return {
            "ok": True,
            "message": message,
            "result": result,
            "execution_time": execution_time
        }

    except Exception as e:
        execution_time = time.time() - start_time
        message = str(e)

        log_nn_event(
            stage="preprocessing",
            status="error",
            execution_time=execution_time,
            parameters=params,
            message=message
        )

        return {
            "ok": False,
            "message": message,
            "result": None,
            "execution_time": execution_time
        }

def start_training(config, progress_callback=None):
    # Start timer
    start_time = time.time()

    try:
        # Run training
        result = train_model(
            config=config,
            progress_callback=progress_callback
        )

        # Compute execution time
        execution_time = time.time() - start_time

        # Build message
        message = (
            f"Training completed. "
            f"Best validation loss: {result['best_val_loss']:.8f}"
        )

        # Log result
        log_nn_event(
            stage="training",
            status="success",
            execution_time=execution_time,
            parameters=config,
            message=message
        )

        # Add execution time
        result["execution_time"] = execution_time
        result["message"] = message

        # Return result to GUI
        return result

    except Exception as e:
        # Compute execution time
        execution_time = time.time() - start_time

        # Error message
        message = str(e)

        # Log error
        log_nn_event(
            stage="training",
            status="error",
            execution_time=execution_time,
            parameters=config,
            message=message
        )

        # Return error result
        return {
            "ok": False,
            "message": message,
            "execution_time": execution_time
        }

def run_validation(checkpoint_path, processed_root):
    # Validation placeholder
    params = {
        "checkpoint_path": checkpoint_path,
        "processed_root": processed_root,
    }

    start_time = time.time()
    execution_time = time.time() - start_time
    message = "Validation not connected yet"

    log_nn_event(
        stage="validation",
        status="not_started",
        execution_time=execution_time,
        parameters=params,
        message=message
    )

    return {
        "ok": False,
        "message": message,
        "execution_time": execution_time
    }