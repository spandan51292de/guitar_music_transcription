import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import torch
import config


def plot_spectrogram(
    spectrogram_tensor,
    sr,
    hop_length,
    ax=None,
    title="CQT Spectrogram",
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    spec_np = spectrogram_tensor.cpu().numpy()

    img = librosa.display.specshow(
        spec_np,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="cqt_hz",
        ax=ax,
        cmap="magma",
        bins_per_octave=config.BINS_PER_OCTAVE_CQT,
    )

    ax.set_title(title)
    ax.figure.colorbar(img, ax=ax, format="%+2.0f dB")

    return ax


def plot_onset_labels(
    onset_targets_tensor,
    sr,
    hop_length,
    num_strings=config.DEFAULT_NUM_STRINGS,
    ax=None,
    title="Onset Labels",
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 2))

    onsets_np = onset_targets_tensor.cpu().numpy()

    if onsets_np.ndim == 2 and onsets_np.shape[1] == num_strings:
        onsets_np = onsets_np.T

    num_frames = (
        onsets_np.shape[1]
        if onsets_np.ndim == 2
        else len(onsets_np)
    )

    max_time = num_frames * hop_length / sr

    ax.imshow(
        onsets_np,
        aspect="auto",
        origin="lower",
        cmap="gray_r",
        extent=[0, max_time, -0.5, num_strings - 0.5],
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    ax.set_yticks(np.arange(num_strings))
    ax.set_yticklabels(
        [f"String {i}" for i in range(num_strings)]
    )

    ax.set_ylabel("String")
    ax.set_xlabel("Time (s)")
    ax.set_title(title)

    return ax


def plot_fret_labels(
    fret_targets_tensor,
    sr,
    hop_length,
    num_strings=config.DEFAULT_NUM_STRINGS,
    max_fret_display=(
        config.MAX_FRETS + config.FRET_SILENCE_CLASS_OFFSET
    ),
    ax=None,
    title="Fret Labels",
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 2))

    frets_np = fret_targets_tensor.cpu().numpy()

    if frets_np.ndim == 2 and frets_np.shape[1] == num_strings:
        frets_np = frets_np.T

    num_frames = (
        frets_np.shape[1]
        if frets_np.ndim == 2
        else len(frets_np)
    )

    max_time = num_frames * hop_length / sr

    custom_cmap = plt.cm.get_cmap(
        "viridis",
        max_fret_display + 1,
    )

    img = ax.imshow(
        frets_np,
        aspect="auto",
        origin="lower",
        cmap=custom_cmap,
        extent=[0, max_time, -0.5, num_strings - 0.5],
        vmin=0,
        vmax=max_fret_display,
        interpolation="nearest",
    )

    colorbar = ax.figure.colorbar(img, ax=ax)

    colorbar.set_label(
        f"Fret Number (silence = {max_fret_display})"
    )

    ax.set_yticks(np.arange(num_strings))
    ax.set_yticklabels(
        [f"String {i}" for i in range(num_strings)]
    )

    ax.set_ylabel("String")
    ax.set_xlabel("Time (s)")
    ax.set_title(title)

    return ax


def plot_sample_data_summary(
    features_tensor,
    onset_labels_tensor,
    fret_labels_tensor,
    sampling_rate,
    hop_len,
    track_id_display=None,
):
    fig, axs = plt.subplots(
        3,
        1,
        figsize=(15, 10),
        sharex=False,
    )

    main_title = "Sample Data Visualization (Ground Truth)"

    if track_id_display:
        main_title = (
            f"{main_title}\nTrack: {track_id_display}"
        )

    fig.suptitle(
        main_title,
        fontsize=16,
    )

    plot_spectrogram(
        features_tensor,
        sr=sampling_rate,
        hop_length=hop_len,
        ax=axs[0],
    )

    plot_onset_labels(
        onset_labels_tensor,
        sr=sampling_rate,
        hop_length=hop_len,
        ax=axs[1],
    )

    plot_fret_labels(
        fret_labels_tensor,
        sr=sampling_rate,
        hop_length=hop_len,
        max_fret_display=(
            config.MAX_FRETS
            + config.FRET_SILENCE_CLASS_OFFSET
        ),
        ax=axs[2],
    )

    plt.tight_layout(
        rect=[0, 0.03, 1, 0.95]
    )

    plt.show()
    plt.close(fig)


def plot_training_history(
    history_data,
    output_save_path=None,
    figure_size=(16, 10),
):
    if (
        not history_data
        or not history_data.get("train_total_loss")
    ):
        return

    epochs = range(
        1,
        len(history_data["train_total_loss"]) + 1,
    )

    fig, axs = plt.subplots(
        2,
        2,
        figsize=figure_size,
    )

    fig.suptitle(
        "Training History Summary",
        fontsize=18,
    )

    # Loss plot
    ax = axs[0, 0]

    ax.plot(
        epochs,
        history_data.get("train_total_loss", []),
        "o-",
        label="Training Loss",
    )

    ax.plot(
        epochs,
        history_data.get("val_total_loss", []),
        "o-",
        label="Validation Loss",
    )

    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()


    # Learning rate plot
    ax = axs[0, 1]

    ax.plot(
        epochs,
        history_data.get("lr", []),
        "o-",
        label="Learning Rate",
        color="purple",
    )

    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()


    # TDR metrics plot
    ax = axs[1, 0]

    ax.plot(
        epochs,
        history_data.get("val_tdr_f1_at_0.5", []),
        "o-",
        label="TDR F1 (notes)",
    )

    ax.plot(
        epochs,
        history_data.get("val_tdr_precision_at_0.5", []),
        "s--",
        label="TDR Precision",
        alpha=0.6,
    )

    ax.plot(
        epochs,
        history_data.get("val_tdr_recall_at_0.5", []),
        "^-",
        label="TDR Recall",
        alpha=0.6,
    )

    ax.set_title(
        "Note-Level Metrics (TDR @ threshold=0.5)"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.grid(True)
    ax.legend()
    ax.set_ylim(0, 1.05)


    # Frame/event metrics plot
    ax = axs[1, 1]

    ax.plot(
        epochs,
        history_data.get("val_mpe_f1", []),
        "o-",
        label="MPE F1 (frames)",
    )

    ax.plot(
        epochs,
        history_data.get(
            "val_onset_f1_event_at_0.5",
            [],
        ),
        "s-",
        label="Onset F1 (event @ threshold=0.5)",
    )

    ax.set_title(
        "Frame-Level and Event-Level Metrics"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.grid(True)
    ax.legend()
    ax.set_ylim(0, 1.05)


    plt.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    if output_save_path:
        try:
            plt.savefig(output_save_path)
        except Exception as e:
            print(
                f"Error while saving training history plot: {e}"
            )

    plt.show()
    plt.close(fig)