from pathlib import Path
import torch

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

def save_checkpoint(path, model, optimizer, epoch, train_loss, val_loss, config):
    # Save model state

    # Build checkpoint dictionary
    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "config": dict(config)
    }

    # Checkpoint path
    path = Path(path)

    # Save checkpoint
    torch.save(checkpoint, str(path))

def load_checkpoint(path, model, optimizer=None):
    # Load model state
    
    # Checkpoint path
    path = Path(path)
    
    # Load checkpoint
    checkpoint = torch.load(
        str(path),
        map_location = torch.device("cpu")
    )
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state if provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Return checkpoint
    return checkpoint