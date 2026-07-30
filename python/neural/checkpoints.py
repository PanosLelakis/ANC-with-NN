from pathlib import Path
import torch
import json
from neural.model import build_model

# There are 2 checkpoints (.pt files):
# best: is saved when validation loss is better than before
# last: is saved at every epoch

def make_checkpoint_dir(processed_root):
    # Create checkpoint folder

    # Checkpoint folder path
    checkpoint_dir = Path(processed_root) / "checkpoints"

    # Create folder if it does not exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Return checkpoint folder directory
    return checkpoint_dir

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

def save_model_info(checkpoint_dir, information):
    # Build model information path
    info_path = (
        Path(checkpoint_dir)
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