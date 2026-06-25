import os
import tkinter as tk
from tkinter import ttk, filedialog

from engine.engine_nn import (
    inspect_dataset_summary,
    preprocess_dataset,
    start_training,
    run_validation,
)

def build_nn_ui(parent, state, default_font, header_font):
    # Widget references
    widgets = {}

    # GUI callbacks
    callbacks = build_nn_callbacks(state, widgets)

    # Main layout
    build_nn_title(parent, header_font)
    build_dataset_frame(parent, state, default_font, callbacks)
    build_summary_frame(parent, default_font, widgets)
    build_model_frame(parent, state, default_font)
    build_training_frame(parent, state, default_font, callbacks)
    build_validation_frame(parent, state, default_font, widgets, callbacks)
    build_progress_frame(parent, default_font, widgets)
    build_status_label(parent, state, default_font, widgets)

    # Initial dataset check
    parent.after(0, callbacks["inspect_dataset"])

def build_nn_callbacks(state, widgets):
    def select_dataset_root():
        # Select dataset folder
        path = filedialog.askdirectory(
            initialdir=state.nn_dataset_root_var.get() or "."
        )

        if path:
            state.nn_dataset_root_var.set(path)
            inspect_dataset()

    def select_processed_root():
        # Select processed folder
        path = filedialog.askdirectory(
            initialdir=state.nn_processed_root_var.get() or "."
        )

        if path:
            state.nn_processed_root_var.set(path)

    def select_checkpoint():
        # Select checkpoint file
        path = filedialog.askopenfilename(
            filetypes=[("PyTorch checkpoints", "*.pt"), ("All files", "*.*")]
        )

        if path:
            state.nn_checkpoint_path_var.set(path)
            widgets["checkpoint_label"].config(text=os.path.basename(path))

    def inspect_dataset():
        # Dataset summary
        summary = inspect_dataset_summary(state.nn_dataset_root_var.get())

        widgets["train_count_label"].config(text=str(summary["train_count"]))
        widgets["validate_count_label"].config(text=str(summary["validate_count"]))

        classes = summary["classes"]
        widgets["classes_label"].config(
            text=", ".join(classes) if classes else "-"
        )

        if summary["dataset_ok"]:
            widgets["status_label"].config(text="Dataset found", fg="green")
        else:
            widgets["status_label"].config(text="Dataset folders not found", fg="red")

    def preprocess_clicked():
        # Preprocess request
        result = preprocess_dataset(
            dataset_root=state.nn_dataset_root_var.get(),
            processed_root=state.nn_processed_root_var.get(),
            target_fs=state.nn_target_fs_var.get(),
            crop_sec=state.nn_crop_sec_var.get(),
        )

        color = "green" if result["ok"] else "black"
        widgets["status_label"].config(text=result["message"], fg=color)

    def start_training_clicked():
        # Training config
        config = {
            "dataset_root": state.nn_dataset_root_var.get(),
            "processed_root": state.nn_processed_root_var.get(),
            "target_fs": state.nn_target_fs_var.get(),
            "crop_sec": state.nn_crop_sec_var.get(),
            "conv_layers": state.nn_conv_layers_var.get(),
            "conv_channels": state.nn_conv_channels_var.get(),
            "lstm_layers": state.nn_lstm_layers_var.get(),
            "lstm_hidden": state.nn_lstm_hidden_var.get(),
            "delay_m": state.nn_delay_m_var.get(),
            "epochs": state.nn_epochs_var.get(),
            "batch_size": state.nn_batch_size_var.get(),
            "learning_rate": state.nn_lr_var.get(),
            "optimizer": state.nn_optimizer_var.get(),
        }

        result = start_training(config)

        color = "green" if result["ok"] else "black"
        widgets["status_label"].config(text=result["message"], fg=color)

    def run_validation_clicked():
        # Validation request
        result = run_validation(state.nn_checkpoint_path_var.get())

        color = "green" if result["ok"] else "black"
        widgets["status_label"].config(text=result["message"], fg=color)

    return {
        "select_dataset_root": select_dataset_root,
        "select_processed_root": select_processed_root,
        "select_checkpoint": select_checkpoint,
        "inspect_dataset": inspect_dataset,
        "preprocess_clicked": preprocess_clicked,
        "start_training_clicked": start_training_clicked,
        "run_validation_clicked": run_validation_clicked,
    }

def build_nn_title(parent, header_font):
    tk.Label(parent, text="Neural Network", font=header_font).grid(
        row=0, column=0, columnspan=3, sticky="w"
    )

def build_dataset_frame(parent, state, default_font, callbacks):
    # Dataset frame
    dataset_frame = ttk.LabelFrame(parent, text="Dataset")
    dataset_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    dataset_frame.grid_columnconfigure(1, weight=1)

    tk.Label(dataset_frame, text="Dataset root:", font=default_font).grid(
        row=0, column=0, sticky="e"
    )

    tk.Entry(
        dataset_frame,
        textvariable=state.nn_dataset_root_var,
        width=55
    ).grid(row=0, column=1, sticky="ew")

    tk.Button(
        dataset_frame,
        text="Browse",
        command=callbacks["select_dataset_root"]
    ).grid(row=0, column=2, sticky="ew")

    tk.Label(dataset_frame, text="Processed root:", font=default_font).grid(
        row=1, column=0, sticky="e"
    )

    tk.Entry(
        dataset_frame,
        textvariable=state.nn_processed_root_var,
        width=55
    ).grid(row=1, column=1, sticky="ew")

    tk.Button(
        dataset_frame,
        text="Browse",
        command=callbacks["select_processed_root"]
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

    # Dataset buttons
    button_frame = tk.Frame(dataset_frame)
    button_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(4, 0))
    button_frame.grid_columnconfigure(0, weight=1)
    button_frame.grid_columnconfigure(1, weight=1)

    tk.Button(
        button_frame,
        text="Inspect Dataset",
        command=callbacks["inspect_dataset"]
    ).grid(row=0, column=0, sticky="ew", padx=(0, 2))

    tk.Button(
        button_frame,
        text="Preprocess Dataset",
        command=callbacks["preprocess_clicked"]
    ).grid(row=0, column=1, sticky="ew", padx=(2, 0))

def build_summary_frame(parent, default_font, widgets):
    # Dataset summary frame
    summary_frame = ttk.LabelFrame(parent, text="Dataset Summary")
    summary_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

    tk.Label(summary_frame, text="Train files:", font=default_font).grid(
        row=0, column=0, sticky="e"
    )

    widgets["train_count_label"] = tk.Label(
        summary_frame,
        text="-",
        font=default_font
    )
    widgets["train_count_label"].grid(row=0, column=1, sticky="w", padx=(0, 20))

    tk.Label(summary_frame, text="Validation files:", font=default_font).grid(
        row=0, column=2, sticky="e"
    )

    widgets["validate_count_label"] = tk.Label(
        summary_frame,
        text="-",
        font=default_font
    )
    widgets["validate_count_label"].grid(row=0, column=3, sticky="w", padx=(0, 20))

    tk.Label(summary_frame, text="Classes:", font=default_font).grid(
        row=0, column=4, sticky="e"
    )

    widgets["classes_label"] = tk.Label(
        summary_frame,
        text="-",
        font=default_font
    )
    widgets["classes_label"].grid(row=0, column=5, sticky="w")

def build_model_frame(parent, state, default_font):
    # Model frame
    model_frame = ttk.LabelFrame(parent, text="Model")
    model_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

    tk.Label(model_frame, text="Architecture:", font=default_font).grid(
        row=0, column=0, sticky="e"
    )
    tk.Label(model_frame, text="Simplified CRN", font=default_font).grid(
        row=0, column=1, sticky="w", padx=(0, 20)
    )

    tk.Label(model_frame, text="Delay M:", font=default_font).grid(
        row=0, column=2, sticky="e"
    )
    tk.Entry(
        model_frame,
        textvariable=state.nn_delay_m_var,
        width=10
    ).grid(row=0, column=3, sticky="w")

    tk.Label(model_frame, text="Conv layers:", font=default_font).grid(
        row=1, column=0, sticky="e"
    )
    tk.Entry(
        model_frame,
        textvariable=state.nn_conv_layers_var,
        width=10
    ).grid(row=1, column=1, sticky="w", padx=(0, 20))

    tk.Label(model_frame, text="Conv channels:", font=default_font).grid(
        row=1, column=2, sticky="e"
    )
    tk.Entry(
        model_frame,
        textvariable=state.nn_conv_channels_var,
        width=10
    ).grid(row=1, column=3, sticky="w")

    tk.Label(model_frame, text="LSTM layers:", font=default_font).grid(
        row=2, column=0, sticky="e"
    )
    tk.Entry(
        model_frame,
        textvariable=state.nn_lstm_layers_var,
        width=10
    ).grid(row=2, column=1, sticky="w", padx=(0, 20))

    tk.Label(model_frame, text="LSTM hidden:", font=default_font).grid(
        row=2, column=2, sticky="e"
    )
    tk.Entry(
        model_frame,
        textvariable=state.nn_lstm_hidden_var,
        width=10
    ).grid(row=2, column=3, sticky="w")

def build_training_frame(parent, state, default_font, callbacks):
    # Training frame
    training_frame = ttk.LabelFrame(parent, text="Training")
    training_frame.grid(row=4, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

    training_frame.grid_columnconfigure(1, weight=1)
    training_frame.grid_columnconfigure(3, weight=1)

    tk.Label(training_frame, text="Epochs:", font=default_font).grid(
        row=0, column=0, sticky="e"
    )
    tk.Entry(
        training_frame,
        textvariable=state.nn_epochs_var,
        width=10
    ).grid(row=0, column=1, sticky="w", padx=(0, 20))

    tk.Label(training_frame, text="Batch size:", font=default_font).grid(
        row=0, column=2, sticky="e"
    )
    tk.Entry(
        training_frame,
        textvariable=state.nn_batch_size_var,
        width=10
    ).grid(row=0, column=3, sticky="w")

    tk.Label(training_frame, text="Learning rate:", font=default_font).grid(
        row=1, column=0, sticky="e"
    )
    tk.Entry(
        training_frame,
        textvariable=state.nn_lr_var,
        width=10
    ).grid(row=1, column=1, sticky="w", padx=(0, 20))

    tk.Label(training_frame, text="Optimizer:", font=default_font).grid(
        row=1, column=2, sticky="e"
    )

    ttk.Combobox(
        training_frame,
        textvariable=state.nn_optimizer_var,
        values=["AMSGrad", "Adam", "AdamW", "SGD", "RMSprop"],
        state="readonly",
        width=10
    ).grid(row=1, column=3, sticky="w")

    tk.Button(
        training_frame,
        text="Start Training",
        command=callbacks["start_training_clicked"]
    ).grid(row=2, column=0, columnspan=4, sticky="ew", pady=(4, 0))

def build_validation_frame(parent, state, default_font, widgets, callbacks):
    # Validation frame
    validation_frame = ttk.LabelFrame(parent, text="Validation")
    validation_frame.grid(row=5, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    validation_frame.grid_columnconfigure(1, weight=1)

    tk.Label(validation_frame, text="Checkpoint:", font=default_font).grid(
        row=0, column=0, sticky="e"
    )

    widgets["checkpoint_label"] = tk.Label(
        validation_frame,
        text="No checkpoint selected",
        font=default_font
    )
    widgets["checkpoint_label"].grid(row=0, column=1, sticky="w")

    tk.Button(
        validation_frame,
        text="Select",
        command=callbacks["select_checkpoint"]
    ).grid(row=0, column=2, sticky="ew")

    tk.Button(
        validation_frame,
        text="Run Validation",
        command=callbacks["run_validation_clicked"]
    ).grid(row=1, column=0, columnspan=3, sticky="ew")

def build_progress_frame(parent, default_font, widgets):
    # Progress frame
    progress_frame = ttk.LabelFrame(parent, text="Progress")
    progress_frame.grid(row=6, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

    for col in range(4):
        progress_frame.grid_columnconfigure(col, weight=1)

    widgets["progress_var"] = tk.DoubleVar(value=0.0)

    widgets["progress_bar"] = ttk.Progressbar(
        progress_frame,
        maximum=100.0,
        variable=widgets["progress_var"]
    )
    widgets["progress_bar"].grid(row=0, column=0, columnspan=4, sticky="ew")

    tk.Label(progress_frame, text="Training loss:", font=default_font).grid(
        row=1, column=0, sticky="e"
    )

    widgets["training_loss_label"] = tk.Label(
        progress_frame,
        text="-",
        font=default_font
    )
    widgets["training_loss_label"].grid(row=1, column=1, sticky="w", padx=(0, 20))

    tk.Label(progress_frame, text="Validation loss:", font=default_font).grid(
        row=1, column=2, sticky="e"
    )

    widgets["validation_loss_label"] = tk.Label(
        progress_frame,
        text="-",
        font=default_font
    )
    widgets["validation_loss_label"].grid(row=1, column=3, sticky="w")

def build_status_label(parent, state, default_font, widgets):
    # Status label
    widgets["status_label"] = tk.Label(
        parent,
        text="",
        font=default_font,
        anchor="w"
    )
    widgets["status_label"].grid(row=7, column=0, columnspan=3, sticky="ew", padx=5)

    state.nn_status_label = widgets["status_label"]