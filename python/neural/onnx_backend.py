from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from neural.checkpoints import load_model_checkpoint
from neural.features import (
    apply_frame_delay,
    signal_to_stft_channels,
    stft_channels_to_signal
)

class OnnxCRN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, stft_input):
        # Run Neural Network core
        return self.model.forward_stft(stft_input)


def get_onnx_path(checkpoint_path):
    # Use same filename with ONNX extension
    return Path(checkpoint_path).with_suffix(".onnx")


def export_checkpoint_to_onnx(checkpoint_path):
    # Use CPU for export
    device = torch.device("cpu")

    # Load trained model
    model, _ = load_model_checkpoint(checkpoint_path, device)

    # Create export wrapper
    export_model = OnnxCRN(model).eval()

    # Build example STFT input
    example_input = torch.zeros(
        1,
        2,
        model.freq_bins,
        10,
        dtype=torch.float32
    )

    # Build output path
    onnx_path = get_onnx_path(checkpoint_path)

    # Export Neural Network core
    torch.onnx.export(
        export_model,
        example_input,
        str(onnx_path),
        input_names=["stft_input"],
        output_names=["stft_output"],
        dynamic_axes={
            "stft_input": {3: "frames"},
            "stft_output": {3: "frames"}
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False
    )

    # Validate exported model
    import onnx

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # Return exported model path
    return str(onnx_path)


def load_onnx_session(checkpoint_path, backend="onnx"):
    # Import ONNX Runtime only when needed
    import onnxruntime as ort

    # Normalize backend name
    backend = str(backend).lower()

    # Build ONNX path
    onnx_path = get_onnx_path(checkpoint_path)

    # Stop when model has not been exported
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    # Read model config
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    config = checkpoint["config"]

    # Create NPU session
    if backend == "npu":
        # Check AMD execution provider
        available_providers = ort.get_available_providers()

        if "VitisAIExecutionProvider" not in available_providers:
            raise RuntimeError(
                "VitisAIExecutionProvider is not available. "
                "Run the program from the Ryzen AI conda environment."
            )

        # Read BF16 configuration path
        config_path = Path(__file__).with_name("vai_ep_config.json")

        if not config_path.exists():
            raise FileNotFoundError(f"Vitis AI config not found: {config_path}")

        # Create compiled model cache folder
        cache_dir = onnx_path.parent / "npu_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Use a new cache after every ONNX export
        cache_key = f"{onnx_path.stem}_{onnx_path.stat().st_mtime_ns}"

        # Configure Vitis AI
        provider_options = [{
            "config_file": str(config_path),
            "cache_dir": str(cache_dir),
            "cache_key": cache_key,
            "enable_cache_file_io_in_mem": 0
        }]

        # Create NPU session
        session = ort.InferenceSession(
            str(onnx_path),
            providers=["VitisAIExecutionProvider"],
            provider_options=provider_options
        )

    # Create CPU session
    else:
        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"]
        )

    # Show active execution providers
    print("ONNX Runtime providers:", session.get_providers())

    # ONNX preprocessing remains on CPU
    device = torch.device("cpu")

    # Return session information
    return session, config, device

def run_onnx_model(session, x, config):
    # Read model settings
    fs = int(config.get("target_fs", 16000))
    delay_m = int(config.get("delay_m", 0))
    signal_length = int(x.shape[1])

    # Convert signal to STFT channels
    stft_input = signal_to_stft_channels(x.detach().cpu(), fs)

    # Apply the same frame delay as PyTorch
    stft_input = apply_frame_delay(stft_input, delay_m)

    # Convert ONNX input to NumPy
    input_array = stft_input.contiguous().numpy().astype(np.float32, copy=False)

    # Read ONNX input name
    input_name = session.get_inputs()[0].name

    # Run ONNX inference
    output_array = session.run(None, {input_name: input_array})[0]

    # Convert output back to tensor
    stft_output = torch.from_numpy(output_array)

    # Convert predicted STFT to time signal
    return stft_channels_to_signal(stft_output, signal_length, fs)