import math
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from neural.dataset import build_noise_datasets
from neural.model import build_model, parse_conv_channels
from neural.checkpoints import (
    make_checkpoint_dir,
    get_checkpoint_paths,
    save_checkpoint,
    save_model_info
)
from neural.preprocess import resample_if_needed
from neural.features import get_stft_params
from neural.nn_logger import (
    initialize_training_history,
    append_training_history
)

def load_paths_as_tensors(
    paths,
    device,
    target_fs
):
    # Read startup paths
    source_fs, primary_path, secondary_path = paths

    # Resample primary path
    resampled_primary_path = resample_if_needed(
        primary_path,
        source_fs,
        target_fs
    )

    # Resample secondary path
    resampled_secondary_path = resample_if_needed(
        secondary_path,
        source_fs,
        target_fs
    )

    # Convert primary path to tensor
    primary_path_tensor = torch.from_numpy(
        resampled_primary_path
    ).float().to(
        device
    )

    # Convert secondary path to tensor
    secondary_path_tensor = torch.from_numpy(
        resampled_secondary_path
    ).float().to(
        device
    )

    # Return path tensors
    return (
        primary_path_tensor,
        secondary_path_tensor
    )

def parse_training_config(config):
    # Convert GUI strings to typed values
    typed_config = {
        "processed_root": str(config.get("processed_root", "")),
        "target_fs": int(config.get("target_fs", 16000)),
        "conv_layers": int(config.get("conv_layers", 2)),
        "conv_channels": str(config.get("conv_channels", "16,32")),
        "lstm_layers": int(config.get("lstm_layers", 1)),
        "lstm_hidden": int(config.get("lstm_hidden", 128)),
        "delay_m": int(config.get("delay_m", 0)),
        "epochs": int(config.get("epochs", 30)),
        "batch_size": int(config.get("batch_size", 1)),
        "learning_rate": float(config.get("learning_rate", 0.001)),
        "optimizer": str(config.get("optimizer", "AMSGrad"))
    }

    # Validate processed root
    if not typed_config["processed_root"]:
        raise ValueError("Processed root is empty")

    # Validate convolution settings
    conv_channels = parse_conv_channels(typed_config["conv_channels"])

    if len(conv_channels) != typed_config["conv_layers"]:
        raise ValueError(
            f"Conv layers must match conv channels. "
            f"Got conv_layers={typed_config['conv_layers']} "
            f"and conv_channels={typed_config['conv_channels']}"
        )

    # Validate training settings
    if typed_config["epochs"] <= 0:
        raise ValueError("Epochs must be positive")

    if typed_config["batch_size"] <= 0:
        raise ValueError("Batch size must be positive")

    if typed_config["learning_rate"] <= 0:
        raise ValueError("Learning rate must be positive")

    # Return typed config
    return typed_config

def get_device():
    # Select cuda if available
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Select cpu otherwise
    return torch.device("cpu")

def create_optimizer(model, optimizer_name, learning_rate):
    # Normalize optimizer name
    optimizer_name = str(optimizer_name)
    learning_rate = float(learning_rate)

    if optimizer_name == "AMSGrad":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            amsgrad=True
        )

    elif optimizer_name == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def causal_fir_filter(x, h):
    # Apply causal FIR filter
    if x.dim() == 1:
        x = x.unsqueeze(0)

    # Prepare signal
    x_3d = x.unsqueeze(1)

    # Prepare impulse response
    h = h.reshape(-1)

    taps = h.numel()

    # Flip filter for convolution
    kernel = h.flip(0).reshape(1, 1, taps)

    # Left padding for causal filtering
    x_padded = F.pad(x_3d, (taps - 1, 0))

    # Apply convolution
    y = F.conv1d(x_padded, kernel)

    # Return signal shape [batch, samples]
    return y.squeeze(1)

def compute_anc_loss(
    model,
    x,
    primary_path,
    secondary_path,
    return_baseline=False
):
    # Compute desired signal
    d = causal_fir_filter(x, primary_path)

    # Compute neural network output
    y = model(x)

    # Compute secondary path signal
    a = causal_fir_filter(y, secondary_path)

    # Match lengths
    length = min(d.shape[1], a.shape[1])

    d = d[:, :length]
    a = a[:, :length]

    # Compute residual error
    e = d - a

    # Compute ANC ON loss
    loss = torch.mean(e ** 2)

    # Return only training loss
    if not return_baseline:
        return loss

    # Compute ANC OFF loss
    baseline_loss = torch.mean(d ** 2)

    # Return validation losses
    return loss, baseline_loss

def train_one_epoch(model, train_loader, optimizer, primary_path, secondary_path, device):
    # Set model to training mode
    model.train()

    # Initialize loss values
    total_loss = 0.0
    total_items = 0

    for x in train_loader:
        # Move batch to device
        x = x.to(device)

        # Reset gradients
        optimizer.zero_grad(set_to_none=True)

        # Compute loss
        loss = compute_anc_loss(
            model=model,
            x=x,
            primary_path=primary_path,
            secondary_path=secondary_path
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Accumulate loss
        batch_size = x.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    # Return average loss
    return total_loss / max(1, total_items)

def validate_one_epoch(
    model,
    val_loader,
    primary_path,
    secondary_path,
    device
):
    # Set model to evaluation mode
    model.eval()

    # Initialize loss values
    total_loss = 0.0
    total_baseline_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for x in val_loader:
            # Move batch to device
            x = x.to(device)

            # Compute validation losses
            loss, baseline_loss = compute_anc_loss(
                model=model,
                x=x,
                primary_path=primary_path,
                secondary_path=secondary_path,
                return_baseline=True
            )

            # Accumulate losses
            batch_size = x.shape[0]

            total_loss += (float(loss.item()) * batch_size)

            total_baseline_loss += (float(baseline_loss.item()) * batch_size)

            total_items += batch_size

    # Compute average losses
    validation_loss = (total_loss / max(1, total_items))

    baseline_validation_loss = (total_baseline_loss / max(1, total_items))

    # Compute normalized validation error
    validation_nmse_db = 10.0 * math.log10((validation_loss + 1e-12) / (baseline_validation_loss + 1e-12))

    # Return validation metrics
    return (
        validation_loss,
        baseline_validation_loss,
        validation_nmse_db
    )

def train_model(config, paths, progress_callback=None):
    # Build dataset loaders
    # Build model
    # Load paths
    # Run epoch loop
    # Save best checkpoint
    # Save last checkpoint
    # Return training result

    # Start training timer
    training_start_time = time.time()
    
    # Parse training config
    config = parse_training_config(config)

    # Select device
    device = get_device()

    # Build datasets
    train_dataset, val_dataset = build_noise_datasets(
        processed_root=config["processed_root"]
    )

    # Build data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0
    )

    # Build model
    model = build_model(config)
    model = model.to(device)

    # Load primary and secondary paths
    primary_path, secondary_path = load_paths_as_tensors(
        paths=paths,
        device=device,
        target_fs=config["target_fs"]
    )

    # Create optimizer
    optimizer = create_optimizer(
        model=model,
        optimizer_name=config["optimizer"],
        learning_rate=config["learning_rate"]
    )

    # Create checkpoint paths
    checkpoint_dir = make_checkpoint_dir(config["processed_root"])
    best_path, last_path = get_checkpoint_paths(checkpoint_dir)

    # Create new training history log
    history_path = initialize_training_history()

    # Initialize training state
    best_val_loss = float("inf")
    best_epoch = 0
    best_train_loss = float("inf")
    best_val_anc_off_loss = float("inf")
    best_val_nmse_db = float("inf")
    history = []

    # Training loop
    for epoch in range(1, config["epochs"] + 1):
        # Train one epoch
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            primary_path=primary_path,
            secondary_path=secondary_path,
            device=device
        )

        # Validate one epoch
        val_loss, val_anc_off_loss, val_nmse_db = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            primary_path=primary_path,
            secondary_path=secondary_path,
            device=device
        )

        # Save last checkpoint
        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_anc_off_loss=val_anc_off_loss,
            val_nmse_db=val_nmse_db,
            config=config
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_epoch = epoch
            best_train_loss = train_loss
            best_val_anc_off_loss = val_anc_off_loss
            best_val_nmse_db = val_nmse_db

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_anc_off_loss=val_anc_off_loss,
                val_nmse_db=val_nmse_db,
                config=config
            )

        # Store epoch history
        history_row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_anc_off_loss": float(
                val_anc_off_loss
            ),
            "val_nmse_db": float(
                val_nmse_db
            )
        }

        history.append(history_row)

        # Write epoch history
        append_training_history(
            epoch=epoch,
            training_loss=train_loss,
            validation_loss=val_loss,
            validation_anc_off_loss=val_anc_off_loss,
            validation_nmse_db=val_nmse_db
        )

        # Progress update
        if progress_callback is not None:
            pct = 100.0 * epoch / float(config["epochs"])

            try:
                progress_callback(pct, epoch, config["epochs"], train_loss, val_loss)
            except TypeError:
                progress_callback(pct)


    # Compute training execution time
    training_execution_time = (
        time.time()
        - training_start_time
    )

    # Count model parameters
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    # Read STFT information
    stft_params = get_stft_params(
        config["target_fs"]
    )

    # Read final history row
    final_row = history[-1]

    # Build model information
    model_information = {
        "model_name": model.__class__.__name__,
        "best_checkpoint_file": best_path.name,
        "last_checkpoint_file": last_path.name,
        "best_checkpoint_path": str(best_path),
        "last_checkpoint_path": str(last_path),
        "training_history_path": str(history_path),
        "device": str(device),
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(
            trainable_parameters
        ),
        "best_checkpoint_size_mb": round(
            best_path.stat().st_size
            / (1024 ** 2),
            4
        ),
        "last_checkpoint_size_mb": round(
            last_path.stat().st_size
            / (1024 ** 2),
            4
        ),
        "training_files": int(
            len(train_dataset)
        ),
        "validation_files": int(
            len(val_dataset)
        ),
        "requested_epochs": int(
            config["epochs"]
        ),
        "completed_epochs": int(
            len(history)
        ),
        "best_epoch": int(best_epoch),
        "best_training_loss": float(
            best_train_loss
        ),
        "best_validation_loss": float(
            best_val_loss
        ),
        "best_validation_anc_off_loss": float(
            best_val_anc_off_loss
        ),
        "best_validation_nmse_db": float(
            best_val_nmse_db
        ),
        "final_epoch": int(
            final_row["epoch"]
        ),
        "final_training_loss": float(
            final_row["train_loss"]
        ),
        "final_validation_loss": float(
            final_row["val_loss"]
        ),
        "final_validation_anc_off_loss": float(
            final_row["val_anc_off_loss"]
        ),
        "final_validation_nmse_db": float(
            final_row["val_nmse_db"]
        ),
        "training_execution_time_sec": float(
            training_execution_time
        ),
        "stft": {
            "n_fft": int(
                stft_params["n_fft"]
            ),
            "win_length": int(
                stft_params["win_length"]
            ),
            "hop_length": int(
                stft_params["hop_length"]
            ),
            "frequency_bins": int(
                stft_params["n_fft"] // 2 + 1
            )
        },
        "config": dict(config)
    }

    # Save model information
    model_info_path = save_model_info(
        checkpoint_dir=checkpoint_dir,
        information=model_information
    )

    # Return training result
    return {
        "ok": True,
        "message": "Training completed",
        "device": str(device),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "model_info_path": str(model_info_path),
        "training_history_path": str(history_path),
        "best_epoch": int(best_epoch),
        "best_train_loss": float(
            best_train_loss
        ),
        "best_val_loss": float(
            best_val_loss
        ),
        "best_val_anc_off_loss": float(
            best_val_anc_off_loss
        ),
        "best_val_nmse_db": float(
            best_val_nmse_db
        ),
        "history": history,
        "config": config
    }