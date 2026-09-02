import math
import time
import random
import numpy as np
import torch
import shutil
import torch.nn.functional as F
from torch.utils.data import DataLoader
from neural.dataset import build_noise_datasets
from neural.model import build_model, parse_conv_channels
from neural.checkpoints import (
    make_model_dir,
    get_checkpoint_paths,
    save_checkpoint,
    save_model_info
)
from neural.preprocess import resample_if_needed
from neural.features import (
    get_stft_params,
    compute_frequency_balanced_residual_loss
)
from neural.nn_logger import (
    initialize_training_history,
    append_training_history
)

# Silence loss weight
SILENCE_LOSS_WEIGHT = 1.0

# Silence power threshold
SILENCE_POWER_THRESHOLD = 1e-12

# Fixed training seed
TRAINING_SEED = 28

# Frequency loss weight
FREQUENCY_LOSS_WEIGHT = 0.0

# Frequency normalization floor
FREQUENCY_POWER_FLOOR_RATIO = 0.01

# Training STFT window
TRAINING_WINDOW_TYPE = "hamming"

TRAINING_ARCHITECTURE = "deep_anc_original"  # Use reference Deep ANC architecture
TRAINING_FRAME_MS = 20  # Use original 20-ms frame
TRAINING_HOP_MS = 10  # Use original 10-ms frame shift

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

    # Compute resampling gain correction
    path_gain = (
        float(source_fs)
        / float(target_fs)
    )

    # Correct primary-path gain
    resampled_primary_path = (
        resampled_primary_path
        * path_gain
    )

    # Correct secondary-path gain
    resampled_secondary_path = (
        resampled_secondary_path
        * path_gain
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

def parse_training_config(config):  # Convert GUI values to model configuration
    typed_config = {  # Build typed configuration
        "architecture": str(  # Store selected architecture
            config.get(
                "architecture",
                TRAINING_ARCHITECTURE
            )
        ).lower(),
        "processed_root": str(config.get("processed_root", "")),  # Store processed dataset
        "target_fs": int(config.get("target_fs", 16000)),  # Store sampling rate
        "conv_layers": int(config.get("conv_layers", 5)),  # Store encoder depth
        "conv_channels": str(config.get("conv_channels", "16,32,64,128,256")),  # Store encoder channels
        "lstm_layers": int(config.get("lstm_layers", 2)),  # Store recurrent depth
        "lstm_hidden": int(config.get("lstm_hidden", 1024)),  # Store recurrent width
        "lstm_groups": int(config.get("lstm_groups", 2)),  # Store recurrent group count
        "delay_m": int(config.get("delay_m", 0)),  # Store frame prediction delay
        "epochs": int(config.get("epochs", 30)),  # Store epoch count
        "batch_size": int(config.get("batch_size", 1)),  # Store batch size
        "learning_rate": float(config.get("learning_rate", 0.001)),  # Store learning rate
        "optimizer": str(config.get("optimizer", "AMSGrad")),  # Store optimizer
        "frame_ms": int(config.get("frame_ms", TRAINING_FRAME_MS)),  # Store STFT frame duration
        "hop_ms": int(config.get("hop_ms", TRAINING_HOP_MS)),  # Store STFT frame shift
        "window_type": str(config.get("window_type", TRAINING_WINDOW_TYPE)).lower(),  # Store STFT window
        "silence_loss_weight": float(SILENCE_LOSS_WEIGHT),  # Store silence loss weight
        "silence_power_threshold": float(SILENCE_POWER_THRESHOLD),  # Store silence threshold
        "seed": int(config.get("seed", TRAINING_SEED)),  # Store deterministic seed
        "frequency_loss_weight": float(FREQUENCY_LOSS_WEIGHT),  # Store frequency loss weight
        "frequency_power_floor_ratio": float(FREQUENCY_POWER_FLOOR_RATIO)  # Store frequency floor
    }

    if not typed_config["processed_root"]:  # Check dataset path
        raise ValueError("Processed root is empty")  # Reject missing dataset

    if typed_config["architecture"] not in (  # Check supported architectures
        "simple_crn",  # Existing simplified architecture
        "deep_anc_original"  # Reference architecture
    ):
        raise ValueError("Unknown Neural Network architecture")  # Reject unknown architecture

    if typed_config["architecture"] == "deep_anc_original":  # Apply fixed reference geometry
        if typed_config["target_fs"] != 16000:  # Require reference sampling rate
            raise ValueError("Deep ANC Original requires target_fs = 16000")  # Reject incompatible sampling rate

        typed_config["conv_layers"] = 5  # Fix encoder depth
        typed_config["conv_channels"] = "16,32,64,128,256"  # Fix encoder channels
        typed_config["lstm_layers"] = 2  # Fix grouped LSTM depth
        typed_config["lstm_hidden"] = 1024  # Fix total recurrent width
        typed_config["lstm_groups"] = 2  # Fix recurrent group count
        typed_config["frame_ms"] = 20  # Fix original frame duration
        typed_config["hop_ms"] = 10  # Fix original frame shift
        typed_config["window_type"] = "hamming"  # Fix original STFT window

    conv_channels = parse_conv_channels(  # Parse convolution channel string
        typed_config["conv_channels"]  # Read configured channels
    )

    if len(conv_channels) != typed_config["conv_layers"]:  # Check channel count
        raise ValueError(  # Reject inconsistent encoder configuration
            f"Conv layers must match conv channels. "
            f"Got conv_layers={typed_config['conv_layers']} "
            f"and conv_channels={typed_config['conv_channels']}"
        )

    if typed_config["epochs"] <= 0:  # Check epoch count
        raise ValueError("Epochs must be positive")  # Reject invalid epoch count

    if typed_config["batch_size"] <= 0:  # Check batch size
        raise ValueError("Batch size must be positive")  # Reject invalid batch size

    if typed_config["learning_rate"] <= 0:  # Check learning rate
        raise ValueError("Learning rate must be positive")  # Reject invalid learning rate

    if typed_config["frame_ms"] <= 0:  # Check frame duration
        raise ValueError("Frame duration must be positive")  # Reject invalid frame duration

    if typed_config["hop_ms"] <= 0:  # Check frame shift
        raise ValueError("Frame shift must be positive")  # Reject invalid frame shift

    if typed_config["window_type"] not in (  # Check supported windows
        "rectangular",  # Existing window
        "hamming"  # Reference window
    ):
        raise ValueError("Window type must be rectangular or hamming")  # Reject unknown window

    return typed_config  # Return validated configuration

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
    return_baseline=False,
    return_components=False
):
    # Compute desired signal
    d = causal_fir_filter(x, primary_path)

    # Compute Neural Network outputs
    #y, Y_channels = model(
        #x,
        #return_stft=True
    #)
    y = model(x)

    # Compute secondary path signal
    a = causal_fir_filter(y, secondary_path)

    # Match lengths
    length = min(d.shape[1], a.shape[1])

    d = d[:, :length]
    a = a[:, :length]

    # Compute residual error
    e = d - a

    # Old loss formula
    # loss = torch.mean(e ** 2)

    # Compute ANC residual loss
    anc_loss = torch.mean(
        e ** 2
    )

    input_power = None  # Avoid unnecessary power calculation

    silence_loss = (  # Initialize disabled silence loss
        anc_loss * 0.0  # Keep differentiable zero
    )

    frequency_loss = (  # Initialize disabled frequency loss
        anc_loss * 0.0  # Keep differentiable zero
    )

    if SILENCE_LOSS_WEIGHT > 0.0:  # Compute silence loss only when enabled
        input_power = torch.mean(  # Compute input power per item
            x ** 2,  # Square waveform samples
            dim=1  # Average sample dimension
        )

        silence_mask = (  # Detect silence examples
            input_power < SILENCE_POWER_THRESHOLD  # Compare with silence threshold
        )

        if torch.any(silence_mask):  # Check whether batch contains silence
            silence_loss = torch.mean(  # Penalize controller activity on silence
                y[silence_mask] ** 2  # Compute controller output power
            )

    if FREQUENCY_LOSS_WEIGHT > 0.0:  # Compute frequency loss only when enabled
        if input_power is None:  # Compute input power if not already available
            input_power = torch.mean(  # Compute input power per item
                x ** 2,  # Square waveform samples
                dim=1  # Average sample dimension
            )

        non_silence_mask = (  # Select normal noise samples
            input_power >= SILENCE_POWER_THRESHOLD  # Exclude silence
        )

        frequency_loss = compute_frequency_balanced_residual_loss(  # Compute auxiliary frequency loss
            desired_signal=d,  # Desired microphone signal
            residual_signal=e,  # ANC residual
            fs=model.fs,  # Sampling rate
            valid_mask=non_silence_mask,  # Use non-silence items
            power_floor_ratio=FREQUENCY_POWER_FLOOR_RATIO,  # Apply normalization floor
            window_type=model.window_type,  # Use model window
            frame_ms=model.frame_ms,  # Use model frame duration
            hop_ms=model.hop_ms  # Use model frame shift
        )

    # Compute weighted silence loss
    weighted_silence_loss = (
        SILENCE_LOSS_WEIGHT
        * silence_loss
    )

    # Compute weighted frequency loss
    weighted_frequency_loss = (
        FREQUENCY_LOSS_WEIGHT
        * frequency_loss
    )

    # Compute total loss
    loss = (
        anc_loss
        + weighted_silence_loss
        + weighted_frequency_loss
    )

    # Store loss components
    loss_components = {
        "anc_loss": anc_loss,
        "silence_loss": silence_loss,
        "weighted_silence_loss": (
            weighted_silence_loss
        ),
        "frequency_loss": frequency_loss,
        "weighted_frequency_loss": (
            weighted_frequency_loss
        ),
        "total_loss": loss
    }

    # Return training loss and components
    if not return_baseline:
        if return_components:
            return (
                loss,
                loss_components
            )

        return loss

    # Compute ANC OFF loss
    baseline_loss = torch.mean(
        d ** 2
    )

    # Return validation losses and components
    if return_components:
        return (
            loss,
            anc_loss,
            baseline_loss,
            loss_components
        )

    # Return validation losses
    return (
        loss,
        anc_loss,
        baseline_loss
    )

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    primary_path,
    secondary_path,
    device
):
    # Set model to training mode
    model.train()

    # Initialize total loss
    total_loss = 0.0

    # Initialize component totals
    component_totals = {
        "anc_loss": 0.0,
        "silence_loss": 0.0,
        "weighted_silence_loss": 0.0,
        "frequency_loss": 0.0,
        "weighted_frequency_loss": 0.0,
        "total_loss": 0.0
    }

    # Initialize item count
    total_items = 0

    for x in train_loader:
        # Move batch to device
        x = x.to(
            device
        )

        # Reset gradients
        optimizer.zero_grad(
            set_to_none=True
        )

        # Compute loss and components
        (
            loss,
            loss_components
        ) = compute_anc_loss(
            model=model,
            x=x,
            primary_path=primary_path,
            secondary_path=secondary_path,
            return_components=True
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Read batch size
        batch_size = x.shape[0]

        # Accumulate total loss
        total_loss += (
            float(
                loss.item()
            )
            * batch_size
        )

        # Accumulate loss components
        for name in component_totals:
            component_totals[name] += (
                float(
                    loss_components[
                        name
                    ].item()
                )
                * batch_size
            )

        # Accumulate item count
        total_items += batch_size

    # Prevent zero division
    item_count = max(
        1,
        total_items
    )

    # Compute average total loss
    average_loss = (
        total_loss
        / item_count
    )

    # Compute average loss components
    average_components = {
        name: value / item_count
        for name, value
        in component_totals.items()
    }

    # Return epoch averages
    return (
        average_loss,
        average_components
    )

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

    # Initialize ANC residual loss
    total_anc_loss = 0.0

    # Initialize ANC OFF loss
    total_baseline_loss = 0.0

    # Initialize item count
    total_items = 0

    with torch.no_grad():
        for x in val_loader:
            # Move batch to device
            x = x.to(device)

            # Compute validation losses
            loss, anc_loss, baseline_loss = compute_anc_loss(
                model=model,
                x=x,
                primary_path=primary_path,
                secondary_path=secondary_path,
                return_baseline=True
            )

            # Accumulate losses
            batch_size = x.shape[0]

            # Accumulate total objective
            total_loss += (
                float(loss.item())
                * batch_size
            )

            # Accumulate ANC residual loss
            total_anc_loss += (
                float(anc_loss.item())
                * batch_size
            )

            # Accumulate ANC OFF loss
            total_baseline_loss += (
                float(baseline_loss.item())
                * batch_size
            )

            # Accumulate item count
            total_items += batch_size

    # Compute total validation loss
    validation_loss = (
        total_loss
        / max(
            1,
            total_items
        )
    )

    # Compute ANC residual validation loss
    validation_anc_loss = (
        total_anc_loss
        / max(
            1,
            total_items
        )
    )

    # Compute ANC OFF validation loss
    baseline_validation_loss = (
        total_baseline_loss
        / max(
            1,
            total_items
        )
    )

    # Compute normalized ANC error
    validation_nmse_db = (
        10.0
        * math.log10(
            (
                validation_anc_loss
                + 1e-12
            )
            / (
                baseline_validation_loss
                + 1e-12
            )
        )
    )

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

    # Read training seed
    seed = int(
        config["seed"]
    )

    # Seed Python
    random.seed(
        seed
    )

    # Seed NumPy
    np.random.seed(
        seed
    )

    # Seed PyTorch
    torch.manual_seed(
        seed
    )

    # Seed CUDA when available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

        # Use deterministic CUDA behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Select device
    device = get_device()

    # Build datasets
    train_dataset, val_dataset = build_noise_datasets(
        processed_root=config["processed_root"]
    )

    # Create deterministic training shuffle
    train_generator = torch.Generator()

    train_generator.manual_seed(
        seed
    )

    # Build data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=train_generator
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

    # Create model folder and checkpoint paths
    model_dir = make_model_dir(config)

    best_path, last_path = (
        get_checkpoint_paths(
            model_dir
        )
    )

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
        (
            train_loss,
            train_components
        ) = train_one_epoch(
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
            "train_loss": float(
                train_loss
            ),
            "train_anc_loss": float(
                train_components[
                    "anc_loss"
                ]
            ),
            "train_silence_loss": float(
                train_components[
                    "silence_loss"
                ]
            ),
            "train_weighted_silence_loss": float(
                train_components[
                    "weighted_silence_loss"
                ]
            ),
            "train_frequency_loss": float(
                train_components[
                    "frequency_loss"
                ]
            ),
            "train_weighted_frequency_loss": float(
                train_components[
                    "weighted_frequency_loss"
                ]
            ),
            "val_loss": float(
                val_loss
            ),
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
            training_anc_loss=(
                train_components[
                    "anc_loss"
                ]
            ),
            training_silence_loss=(
                train_components[
                    "silence_loss"
                ]
            ),
            training_weighted_silence_loss=(
                train_components[
                    "weighted_silence_loss"
                ]
            ),
            training_frequency_loss=(
                train_components[
                    "frequency_loss"
                ]
            ),
            training_weighted_frequency_loss=(
                train_components[
                    "weighted_frequency_loss"
                ]
            ),
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
    stft_params = get_stft_params(  # Read actual checkpoint STFT settings
        config["target_fs"],  # Sampling rate
        frame_ms=config["frame_ms"],  # Saved frame duration
        hop_ms=config["hop_ms"]  # Saved frame shift
    )

    # Read final history row
    final_row = history[-1]

    # Copy training history into model folder
    model_history_path = (
        model_dir
        / "training_history.csv"
    )

    shutil.copy2(
        history_path,
        model_history_path
    )

    # Build model information
    model_information = {
        "model_name": model.__class__.__name__,
        "best_checkpoint_file": best_path.name,
        "last_checkpoint_file": last_path.name,
        "best_checkpoint_path": str(best_path),
        "last_checkpoint_path": str(last_path),
        "model_directory": str(
            model_dir
        ),

        "training_history_path": str(
            model_history_path
        ),
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
        model_dir=model_dir,
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
        "training_history_path": str(
            model_history_path
        ),
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
        "config": config,
        "model_directory": str(
            model_dir
        ),
    }