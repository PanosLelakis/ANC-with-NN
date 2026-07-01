from pathlib import Path
import torch
from scipy.io import wavfile
from torch.utils.data import Dataset as TorchDataset

def list_wav_files(path):
    # Root folder
    root = Path(path)

    # Raise error if folder does not exist
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    # Keep only .wav files
    wav_files = sorted(root.rglob("*.wav"))

    # Raise error if no wav files found
    if len(wav_files) == 0:
        raise RuntimeError(f"No wav files found in: {root}")

    # Return list of wav files
    return wav_files

def read_wav_as_tensor(path):
    # Read wav file
    _, data = wavfile.read(str(path))

    # Convert to tensor
    tensor = torch.from_numpy(data).float()

    return tensor

def build_noise_datasets(processed_root):
    # Train dataset
    train_dataset = NoiseDataset(
        processed_root=processed_root,
        split="train"
    )

    # Validation dataset
    val_dataset = NoiseDataset(
        processed_root=processed_root,
        split="validate"
    )

    return train_dataset, val_dataset

class NoiseDataset(TorchDataset):
    def __init__(self, processed_root, split):
        # Dataset settings
        self.processed_root = Path(processed_root)
        self.split = str(split)
        self.split_dir = self.processed_root / self.split

        # WAV files
        self.paths = list_wav_files(self.split_dir)

    def __len__(self):
        # Dataset size
        return len(self.paths)

    def __getitem__(self, index):
        # Audio path
        path = self.paths[index]

        # Read audio
        x = read_wav_as_tensor(path)

        # Return item
        return x