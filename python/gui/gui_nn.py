import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog

def build_nn_ui(parent, state, default_font, header_font):
    # Local helpers
    def select_dataset_root():
        path = filedialog.askdirectory(
            initialdir=state.nn_dataset_root_var.get() or "."
        )

        if path:
            state.nn_dataset_root_var.set(path)
            inspect_dataset()

    def select_processed_root():
        path = filedialog.askdirectory(
            initialdir=state.nn_processed_root_var.get() or "."
        )

        if path:
            state.nn_processed_root_var.set(path)

    def select_checkpoint():
        path = filedialog.askopenfilename(
            filetypes=[("PyTorch checkpoints", "*.pt"), ("All files", "*.*")]
        )

        if path:
            state.nn_checkpoint_path_var.set(path)
            checkpoint_label.config(text=os.path.basename(path))

    def get_noise_label(path):
        # Known noise names
        name = path.stem.lower()

        if "airport" in name:
            return "airport"
        if "street" in name:
            return "street"
        if "subway" in name:
            return "subway"

        return "unknown"

    def inspect_dataset():
        root = Path(state.nn_dataset_root_var.get())
        train_dir = root / "train"
        validate_dir = root / "validate"

        train_files = sorted(train_dir.glob("*.wav")) if train_dir.exists() else []
        validate_files = sorted(validate_dir.glob("*.wav")) if validate_dir.exists() else []

        labels = [get_noise_label(p) for p in train_files + validate_files]
        unique_labels = sorted(set(labels))

        train_count_label.config(text=str(len(train_files)))
        validate_count_label.config(text=str(len(validate_files)))
        classes_label.config(text=", ".join(unique_labels) if unique_labels else "-")

        if train_dir.exists() and validate_dir.exists():
            nn_status.config(text="Dataset found", fg="green")
        else:
            nn_status.config(text="Dataset folders not found", fg="red")

    def preprocess_dataset():
        # Backend placeholder
        nn_status.config(text="Preprocessing not connected yet", fg="black")

    def start_training():
        # Backend placeholder
        nn_status.config(text="Training not connected yet", fg="black")

    def stop_training():
        # Backend placeholder
        nn_status.config(text="Stop training not connected yet", fg="black")

    def run_validation():
        # Backend placeholder
        nn_status.config(text="Validation not connected yet", fg="black")

    # Title
    tk.Label(parent, text="Neural Network", font=header_font).grid(
        row=0, column=0, columnspan=3, sticky="w"
    )

    # Dataset frame
    dataset_frame = ttk.LabelFrame(parent, text="Dataset")
    dataset_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    dataset_frame.grid_columnconfigure(1, weight=1)

    tk.Label(dataset_frame, text="Dataset root:", font=default_font).grid(
        row=0, column=0, sticky="e"
    )

    dataset_entry = tk.Entry(
        dataset_frame,
        textvariable=state.nn_dataset_root_var,
        width=55
    )
    dataset_entry.grid(row=0, column=1, sticky="ew")

    tk.Button(
        dataset_frame,
        text="Browse",
        command=select_dataset_root
    ).grid(row=0, column=2, sticky="ew")

    tk.Label(dataset_frame, text="Processed root:", font=default_font).grid(
        row=1, column=0, sticky="e"
    )

    processed_entry = tk.Entry(
        dataset_frame,
        textvariable=state.nn_processed_root_var,
        width=55
    )
    processed_entry.grid(row=1, column=1, sticky="ew")

    tk.Button(
        dataset_frame,
        text="Browse",
        command=select_processed_root
    ).grid(row=1, column=2, sticky="ew")

    tk.Label(dataset_frame, text="Target fs:", font=default_font).grid(
        row=2, column=0, sticky="e"
    )

    tk.Entry(
        dataset_frame,
        textvariable=state.nn_target_fs_var,
        width=10
    ).grid(row=2, column=1, sticky="w")

    tk.Label(dataset_frame, text="Crop duration:", font=default_font).grid(
        row=3, column=0, sticky="e"
    )

    tk.Entry(
        dataset_frame,
        textvariable=state.nn_crop_sec_var,
        width=10
    ).grid(row=3, column=1, sticky="w")

    tk.Label(dataset_frame, text="Normalization:", font=default_font).grid(
        row=4, column=0, sticky="e"
    )

    tk.Label(dataset_frame, text="Unit power", font=default_font).grid(
        row=4, column=1, sticky="w"
    )

    tk.Button(
        dataset_frame,
        text="Inspect Dataset",
        command=inspect_dataset
    ).grid(row=5, column=0, sticky="ew")

    tk.Button(
        dataset_frame,
        text="Preprocess Dataset",
        command=preprocess_dataset
    ).grid(row=5, column=1, sticky="ew")

    # Dataset summary frame
    summary_frame = ttk.LabelFrame(parent, text="Dataset Summary")
    summary_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

    tk.Label(summary_frame, text="Train files:", font=default_font).grid(
        row=0, column=0, sticky="e"
    )
    train_count_label = tk.Label(summary_frame, text="-", font=default_font)
    train_count_label.grid(row=0, column=1, sticky="w")

    tk.Label(summary_frame, text="Validation files:", font=default_font).grid(
        row=1, column=0, sticky="e"
    )
    validate_count_label = tk.Label(summary_frame, text="-", font=default_font)
    validate_count_label.grid(row=1, column=1, sticky="w")

    tk.Label(summary_frame, text="Classes:", font=default_font).grid(
        row=2, column=0, sticky="e"
    )
    classes_label = tk.Label(summary_frame, text="-", font=default_font)
    classes_label.grid(row=2, column=1, sticky="w")

    # Model frame
    model_frame = ttk.LabelFrame(parent, text="Model")
    model_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

    tk.Label(model_frame, text="Architecture:", font=default_font).grid(
        row=0, column=0, sticky="e"
    )
    tk.Label(model_frame, text="Simplified CRN", font=default_font).grid(
        row=0, column=1, sticky="w"
    )

    tk.Label(model_frame, text="Conv layers:", font=default_font).grid(
        row=1, column=0, sticky="e"
    )
    tk.Label(model_frame, text="2", font=default_font).grid(
        row=1, column=1, sticky="w"
    )

    tk.Label(model_frame, text="Conv channels:", font=default_font).grid(
        row=2, column=0, sticky="e"
    )
    tk.Entry(
        model_frame,
        textvariable=state.nn_conv_channels_var,
        width=10
    ).grid(row=2, column=1, sticky="w")

    tk.Label(model_frame, text="LSTM layers:", font=default_font).grid(
        row=3, column=0, sticky="e"
    )
    tk.Label(model_frame, text="1", font=default_font).grid(
        row=3, column=1, sticky="w"
    )

    tk.Label(model_frame, text="LSTM hidden:", font=default_font).grid(
        row=4, column=0, sticky="e"
    )
    tk.Entry(
        model_frame,
        textvariable=state.nn_lstm_hidden_var,
        width=10
    ).grid(row=4, column=1, sticky="w")

    tk.Label(model_frame, text="Delay M:", font=default_font).grid(
        row=5, column=0, sticky="e"
    )
    tk.Entry(
        model_frame,
        textvariable=state.nn_delay_m_var,
        width=10
    ).grid(row=5, column=1, sticky="w")

    tk.Label(model_frame, text="Loudspeaker:", font=default_font).grid(
        row=6, column=0, sticky="e"
    )
    tk.Label(model_frame, text="Linear", font=default_font).grid(
        row=6, column=1, sticky="w"
    )

    # Training frame
    training_frame = ttk.LabelFrame(parent, text="Training")
    training_frame.grid(row=4, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

    tk.Label(training_frame, text="Epochs:", font=default_font).grid(
        row=0, column=0, sticky="e"
    )
    tk.Entry(
        training_frame,
        textvariable=state.nn_epochs_var,
        width=10
    ).grid(row=0, column=1, sticky="w")

    tk.Label(training_frame, text="Batch size:", font=default_font).grid(
        row=1, column=0, sticky="e"
    )
    tk.Entry(
        training_frame,
        textvariable=state.nn_batch_size_var,
        width=10
    ).grid(row=1, column=1, sticky="w")

    tk.Label(training_frame, text="Learning rate:", font=default_font).grid(
        row=2, column=0, sticky="e"
    )
    tk.Entry(
        training_frame,
        textvariable=state.nn_lr_var,
        width=10
    ).grid(row=2, column=1, sticky="w")

    tk.Label(training_frame, text="Optimizer:", font=default_font).grid(
        row=3, column=0, sticky="e"
    )
    tk.Label(training_frame, text="AMSGrad", font=default_font).grid(
        row=3, column=1, sticky="w"
    )

    tk.Button(
        training_frame,
        text="Start Training",
        command=start_training
    ).grid(row=4, column=0, sticky="ew")

    tk.Button(
        training_frame,
        text="Stop Training",
        command=stop_training,
        state=tk.DISABLED
    ).grid(row=4, column=1, sticky="ew")

    # Validation frame
    validation_frame = ttk.LabelFrame(parent, text="Validation")
    validation_frame.grid(row=5, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    validation_frame.grid_columnconfigure(1, weight=1)

    tk.Label(validation_frame, text="Checkpoint:", font=default_font).grid(
        row=0, column=0, sticky="e"
    )

    checkpoint_label = tk.Label(
        validation_frame,
        text="No checkpoint selected",
        font=default_font
    )
    checkpoint_label.grid(row=0, column=1, sticky="w")

    tk.Button(
        validation_frame,
        text="Select",
        command=select_checkpoint
    ).grid(row=0, column=2, sticky="ew")

    tk.Button(
        validation_frame,
        text="Run Validation",
        command=run_validation
    ).grid(row=1, column=0, columnspan=3, sticky="ew")

    # Progress frame
    progress_frame = ttk.LabelFrame(parent, text="Progress")
    progress_frame.grid(row=6, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

    nn_progress_var = tk.DoubleVar(value=0.0)

    nn_progress = ttk.Progressbar(
        progress_frame,
        maximum=100.0,
        variable=nn_progress_var
    )
    nn_progress.grid(row=0, column=0, columnspan=3, sticky="ew")

    tk.Label(progress_frame, text="Training loss:", font=default_font).grid(
        row=1, column=0, sticky="e"
    )
    tk.Label(progress_frame, text="-", font=default_font).grid(
        row=1, column=1, sticky="w"
    )

    tk.Label(progress_frame, text="Validation loss:", font=default_font).grid(
        row=2, column=0, sticky="e"
    )
    tk.Label(progress_frame, text="-", font=default_font).grid(
        row=2, column=1, sticky="w"
    )

    # Status
    nn_status = tk.Label(parent, text="", font=default_font, anchor="w")
    nn_status.grid(row=7, column=0, columnspan=3, sticky="ew", padx=5)

    state.nn_status_label = nn_status

    # Initial inspection
    parent.after(0, inspect_dataset)