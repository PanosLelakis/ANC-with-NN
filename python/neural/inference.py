import torch
from neural.checkpoints import load_model_checkpoint
from neural.onnx_backend import load_onnx_session, run_onnx_model
from neural.train import causal_fir_filter

def load_trained_model(checkpoint_path):
    # Select PyTorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load PyTorch model
    model, config = load_model_checkpoint(checkpoint_path, device)

    # Return loaded objects
    return model, config, device

def load_inference_model(checkpoint_path, backend="pytorch"):
    # Normalize backend name
    backend = str(backend).lower()

    # Load ONNX model
    if backend == "onnx":
        return load_onnx_session(checkpoint_path)

    # Load PyTorch model
    return load_trained_model(checkpoint_path)

def run_anc_inference(
    model,
    x,
    primary_path,
    secondary_path,
    backend="pytorch",
    config=None
):
    # Normalize backend name
    backend = str(backend).lower()

    # Disable gradient calculation
    with torch.inference_mode():
        # Compute desired signal
        d = causal_fir_filter(x, primary_path)

        # Compute controller output
        if backend == "onnx":
            y = run_onnx_model(model, x, config)
        else:
            y = model(x)

        # Compute secondary path output
        a = causal_fir_filter(y, secondary_path)

        # Find common signal length
        length = min(d.shape[1], a.shape[1])

        # Match signal lengths
        d = d[:, :length]
        y = y[:, :length]
        a = a[:, :length]

        # Compute residual error
        e = d - a

        # Compute ANC ON loss
        loss = torch.mean(e ** 2)

        # Compute ANC OFF loss
        baseline_loss = torch.mean(d ** 2)

    # Return inference signals
    return {
        "d": d,
        "y": y,
        "a": a,
        "e": e,
        "loss": loss,
        "baseline_loss": baseline_loss
    }