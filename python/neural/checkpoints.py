from pathlib import Path
import torch
import json
from neural.model import build_model

# There are 2 checkpoints (.pt files):
# best: is saved when validation loss is better than before
# last: is saved at every epoch

def get_models_root():
    # Build models root
    models_root = (
        Path.cwd()
        / "models"
    )

    # Create models root
    models_root.mkdir(
        parents=True,
        exist_ok=True
    )

    # Return models root
    return models_root

def build_model_folder_name(config):
    # Format learning rate
    learning_rate = (
        f"{float(config['learning_rate']):.10f}"
    )

    learning_rate = (
        learning_rate
        .rstrip("0")
        .rstrip(".")
    )

    if learning_rate.startswith("0."):
        learning_rate = learning_rate[2:]

    # Format convolution channels
    conv_channels = "_".join(
        part.strip()
        for part in str(
            config["conv_channels"]
        ).split(",")
        if part.strip()
    )

    architecture = str(  # Read model architecture
        config.get(
            "architecture",
            "simple_crn"
        )
    ).lower()

    # Build model folder name
    return (  # Build descriptive model folder
        f"{architecture}_"  # Store architecture name
        f"{int(config['epochs'])}ep_"  # Store epoch count
        f"{learning_rate}lr_"  # Store learning rate
        f"{int(config['conv_layers'])}conv_"  # Store convolution depth
        f"{conv_channels}_"  # Store convolution channels
        f"{int(config['lstm_layers'])}lstm_"  # Store recurrent depth
        f"{int(config['lstm_hidden'])}_"  # Store recurrent width
        f"{int(config['delay_m'])}M"  # Store prediction delay
    )

def make_model_dir(config):
    # Build base folder
    models_root = get_models_root()
    base_name = build_model_folder_name(
        config
    )

    model_dir = (
        models_root
        / base_name
    )

    # Keep repeated trainings separate
    suffix = 2

    while model_dir.exists():
        model_dir = (
            models_root
            / f"{base_name}_{suffix}"
        )

        suffix += 1

    # Create model folder
    model_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    # Return model folder
    return model_dir

def get_latest_checkpoint_path():
    # Find trained checkpoints
    candidates = list(
        get_models_root().glob(
            "*/best.pt"
        )
    )

    # Return empty path if none exists
    if not candidates:
        return ""

    # Find most recently modified model
    latest = max(
        candidates,
        key=lambda path: (
            path.stat().st_mtime
        )
    )

    # Return checkpoint path
    return str(latest)

def get_checkpoint_paths(checkpoint_dir):
    # Build best and last paths

    # Checkpoint folder path
    checkpoint_dir = Path(checkpoint_dir)

    # Checkpoint paths
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"

    # Return paths
    return best_path, last_path

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    config,
    val_anc_off_loss=None,
    val_nmse_db=None
):
    # Build checkpoint dictionary
    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_anc_off_loss": (
            None
            if val_anc_off_loss is None
            else float(val_anc_off_loss)
        ),
        "val_nmse_db": (
            None
            if val_nmse_db is None
            else float(val_nmse_db)
        ),
        "config": dict(config)
    }

    # Checkpoint path
    path = Path(path)

    # Save checkpoint
    torch.save(
        checkpoint,
        str(path)
    )

def save_model_info(
    model_dir,
    information
):
    # Build model information path
    info_path = (
        Path(model_dir)
        / "model_info.json"
    )

    # Create checkpoint folder
    info_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save model information
    with open(
        info_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            information,
            file,
            ensure_ascii=False,
            indent=4
        )

    # Return model information path
    return str(info_path)

def load_model_checkpoint(path, device):
    # Load checkpoint
    checkpoint = torch.load(str(Path(path)), map_location=device)

    # Read model config
    config = checkpoint["config"]

    # Build model
    model = build_model(config).to(device)

    # Load trained weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set inference mode
    model.eval()

    # Return model and config
    return model, config