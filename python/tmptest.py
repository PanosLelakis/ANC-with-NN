from neural.train import train_model


def progress_callback(pct, epoch=None, total_epochs=None, train_loss=None, val_loss=None):
    # Print training progress
    if epoch is not None:
        print(
            f"Progress: {pct:.1f}% | "
            f"Epoch {epoch}/{total_epochs} | "
            f"Train loss: {train_loss:.8f} | "
            f"Validation loss: {val_loss:.8f}"
        )
    else:
        print(f"Progress: {pct:.1f}%")


def main():
    # Training test config
    config = {
        "processed_root": "python/dataset/processed",
        "target_fs": 16000,
        "conv_layers": 2,
        "conv_channels": "16,32",
        "lstm_layers": 1,
        "lstm_hidden": 128,
        "delay_m": 0,
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 0.001,
        "optimizer": "AMSGrad"
    }

    # Run training
    result = train_model(
        config=config,
        progress_callback=progress_callback
    )

    # Print result
    print("Training result:")
    print("ok:", result["ok"])
    print("message:", result["message"])
    print("device:", result["device"])
    print("best checkpoint:", result["best_checkpoint"])
    print("last checkpoint:", result["last_checkpoint"])
    print("best validation loss:", result["best_val_loss"])
    print("history:", result["history"])


if __name__ == "__main__":
    main()