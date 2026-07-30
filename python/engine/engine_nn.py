import time
from neural.preprocess import preprocess_noise_dataset
from neural.nn_logger import log_nn_event
from neural.train import train_model
from neural.validate import validate_checkpoint
from neural.onnx_backend import export_checkpoint_to_onnx

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

def start_training(config, paths, progress_callback=None):
    # Start timer
    start_time = time.time()

    try:
        # Run training
        result = train_model(
            config=config,
            paths=paths,
            progress_callback=progress_callback
        )

        # Compute execution time
        execution_time = time.time() - start_time

        # Build message
        message = (
            f"Training completed. "
            f"Best epoch: {result['best_epoch']}. "
            f"Best validation loss: "
            f"{result['best_val_loss']:.8f}. "
            f"Best validation NMSE: "
            f"{result['best_val_nmse_db']:.2f} dB."
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

def run_validation(
    checkpoint_path,
    processed_root,
    paths,
    progress_callback=None
):
    # Store validation parameters
    params = {
        "checkpoint_path": checkpoint_path,
        "processed_root": processed_root
    }

    # Start timer
    start_time = time.time()

    try:
        # Run validation
        result = validate_checkpoint(
            checkpoint_path=checkpoint_path,
            processed_root=processed_root,
            paths=paths,
            progress_callback=progress_callback
        )

        # Compute execution time
        execution_time = time.time() - start_time

        # Build success message
        message = (
            f"Validation completed. "
            f"ANC OFF: "
            f"{result['anc_off_dbr']:.2f} dBr. "
            f"ANC ON: "
            f"{result['anc_on_dbr']:.2f} dBr. "
            f"Residual change: "
            f"{result['residual_change_db']:.2f} dB."
        )

        # Log validation
        log_nn_event(
            stage="validation",
            status="success",
            execution_time=execution_time,
            parameters=params,
            message=message
        )

        # Add execution time
        result["execution_time"] = execution_time

        # Add result message
        result["message"] = message

        # Return result
        return result

    except Exception as error:
        # Compute execution time
        execution_time = time.time() - start_time

        # Build error message
        message = str(error)

        # Log error
        log_nn_event(
            stage="validation",
            status="error",
            execution_time=execution_time,
            parameters=params,
            message=message
        )

        # Return error result
        return {
            "ok": False,
            "message": message,
            "execution_time": execution_time
        }

def export_onnx_model(checkpoint_path):
    # Start timer
    start_time = time.time()

    # Store export parameters
    params = {"checkpoint_path": checkpoint_path}

    try:
        # Export model
        onnx_path = export_checkpoint_to_onnx(checkpoint_path)

        # Compute execution time
        execution_time = time.time() - start_time

        # Build success message
        message = f"ONNX model exported to: {onnx_path}"

        # Log export
        log_nn_event(
            stage="onnx_export",
            status="success",
            execution_time=execution_time,
            parameters=params,
            message=message
        )

        # Return success result
        return {
            "ok": True,
            "message": message,
            "onnx_path": onnx_path,
            "execution_time": execution_time
        }

    except Exception as error:
        # Compute execution time
        execution_time = time.time() - start_time

        # Build error message
        message = str(error)

        # Log error
        log_nn_event(
            stage="onnx_export",
            status="error",
            execution_time=execution_time,
            parameters=params,
            message=message
        )

        # Return error result
        return {
            "ok": False,
            "message": message,
            "execution_time": execution_time
        }