import math
from torch.utils.data import DataLoader
from neural.dataset import NoiseDataset
from neural.inference import load_trained_model, run_anc_inference
from neural.train import load_paths_as_tensors

def validate_checkpoint(
    checkpoint_path,
    processed_root,
    paths,
    progress_callback=None
):
    # Load trained model
    model, config, device = load_trained_model(checkpoint_path)

    # Create validation dataset
    validation_dataset = NoiseDataset(
        processed_root=processed_root,
        split="validate"
    )

    # Read batch size
    batch_size = int(config.get("batch_size", 1))

    # Create validation loader
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Read sampling rate
    target_fs = int(config.get("target_fs", 16000))

    # Load ANC paths
    primary_path, secondary_path = load_paths_as_tensors(
        paths=paths,
        device=device,
        target_fs=target_fs
    )

    # Initialize total powers
    total_loss = 0.0
    total_baseline_loss = 0.0
    total_input_power = 0.0

    # Initialize sample count
    total_items = 0

    # Count validation batches
    total_batches = len(validation_loader)

    # Process validation batches
    for batch_index, x in enumerate(
        validation_loader,
        start=1
    ):
        # Move input to device
        x = x.to(device)

        # Compute noisy input power
        input_power = x.pow(2).mean()

        # Run inference
        signals = run_anc_inference(
            model=model,
            x=x,
            primary_path=primary_path,
            secondary_path=secondary_path
        )

        # Read batch losses
        loss = signals["loss"]
        baseline_loss = signals["baseline_loss"]

        # Read current batch size
        current_batch_size = x.shape[0]

        # Add weighted losses
        total_loss += (float(loss.item()) * current_batch_size)

        total_baseline_loss += (float(baseline_loss.item()) * current_batch_size)

        total_input_power += (
            float(input_power.item())
            * current_batch_size
        )

        # Add processed items
        total_items += current_batch_size

        # Update progress
        if progress_callback is not None:
            # Calculate progress
            progress = 100.0 * batch_index / total_batches

            # Send progress
            progress_callback(
                progress,
                batch_index,
                total_batches
            )

    # Compute average validation losses
    validation_loss = (total_loss / max(1, total_items))

    baseline_validation_loss = (total_baseline_loss / max(1, total_items))

    # Compute average noisy input power
    input_validation_power = (
        total_input_power
        / max(1, total_items)
    )

    # Convert powers to dBr
    anc_off_dbr = 10.0 * math.log10(
        (baseline_validation_loss + 1e-12)
        / (input_validation_power + 1e-12)
    )

    anc_on_dbr = 10.0 * math.log10(
        (validation_loss + 1e-12)
        / (input_validation_power + 1e-12)
    )

    # Compute ANC ON change relative to ANC OFF
    residual_change_db = (
        anc_on_dbr
        - anc_off_dbr
    )

    # Return validation result
    return {
    "ok": True,
    "validation_loss": float(
        validation_loss
    ),
    "baseline_validation_loss": float(
        baseline_validation_loss
    ),
    "input_validation_power": float(
        input_validation_power
    ),
    "anc_off_dbr": float(
        anc_off_dbr
    ),
    "anc_on_dbr": float(
        anc_on_dbr
    ),
    "residual_change_db": float(
        residual_change_db
    ),
    "validation_nmse_db": float(
        residual_change_db
    ),
    "validation_files": len(
        validation_dataset
    ),
    "device": str(device),
    "config": config
}