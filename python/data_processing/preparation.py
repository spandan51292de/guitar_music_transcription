import os
import jams
import numpy as np
import torch
import librosa
import mirdata
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import config


def prepare_track_splits(
    data_home,
    problematic_files_list,
    test_split_fraction,
    validation_split_fraction,
    seed,
    output_dir_for_ids=None,
):
    try:
        guitarset_loader = mirdata.initialize("guitarset", data_home=data_home)
        all_track_ids = guitarset_loader.track_ids
        print(f"Found {len(all_track_ids)} total track IDs in GuitarSet.")
    except Exception as e:
        print(f"Error while initializing mirdata or retrieving track IDs: {e}")
        return {"train": [], "validation": [], "test": []}

    filtered_track_ids = [
        track_id
        for track_id in all_track_ids
        if os.path.splitext(os.path.basename(track_id))[0]
        not in problematic_files_list
    ]
    print(
        f"Number of track IDs after removing problematic files: {len(filtered_track_ids)}"
    )

    track_ids_split_map = {"train": [], "validation": [], "test": []}

    if not filtered_track_ids:
        print("No track IDs available after filtering.")
        return track_ids_split_map

    train_val_ids, test_ids = train_test_split(
        filtered_track_ids,
        test_size=test_split_fraction,
        random_state=seed,
        shuffle=True,
    )
    track_ids_split_map["test"] = test_ids

    if (1 - test_split_fraction) > 0 and len(train_val_ids) > 0:
        relative_val_split_size = (
            validation_split_fraction / (1 - test_split_fraction)
            if (1 - test_split_fraction) > 0
            else 0
        )

        if relative_val_split_size > 0 and len(train_val_ids) > 1:
            train_ids, validation_ids = train_test_split(
                train_val_ids,
                test_size=relative_val_split_size,
                random_state=seed,
                shuffle=True,
            )
            track_ids_split_map["train"] = train_ids
            track_ids_split_map["validation"] = validation_ids
        elif len(train_val_ids) > 0:
            track_ids_split_map["train"] = train_val_ids
            track_ids_split_map["validation"] = []
    elif len(train_val_ids) > 0:
        track_ids_split_map["train"] = train_val_ids
        track_ids_split_map["validation"] = []

    print("\nDataset split:")
    print(f"  Training: {len(track_ids_split_map['train'])} tracks")
    print(f"  Validation: {len(track_ids_split_map['validation'])} tracks")
    print(f"  Test: {len(track_ids_split_map['test'])} tracks")

    if output_dir_for_ids:
        os.makedirs(output_dir_for_ids, exist_ok=True)

        for split_name, ids in track_ids_split_map.items():
            split_specific_dir = os.path.join(output_dir_for_ids, split_name)
            os.makedirs(split_specific_dir, exist_ok=True)

            with open(
                os.path.join(output_dir_for_ids, f"{split_name}_ids.txt"), "w"
            ) as f:
                for track_id in ids:
                    f.write(f"{track_id}\n")

        print(
            f"\nSaved track ID lists for each split in: {output_dir_for_ids}"
        )

    return track_ids_split_map


def extract_annotations_from_jams(jams_file_path):
    notes = []

    jam_data = jams.load(jams_file_path)
    note_midi_annotations = jam_data.search(namespace="note_midi")

    for annotation_obj in note_midi_annotations:
        if not (
            annotation_obj.annotation_metadata
            and hasattr(annotation_obj.annotation_metadata, "data_source")
        ):
            continue

        string_num_str = annotation_obj.annotation_metadata.data_source

        if isinstance(string_num_str, str) and string_num_str.isdigit():
            string_num = int(string_num_str)
        else:
            continue

        if string_num not in config.OPEN_STRING_PITCHES_JAMS:
            continue

        open_string_pitch = config.OPEN_STRING_PITCHES_JAMS[string_num]

        for obs in annotation_obj.data:
            onset_sec = float(obs.time)
            duration_sec = float(obs.duration)
            offset_sec = onset_sec + duration_sec
            pitch_midi = float(obs.value)

            fret_num = int(round(pitch_midi - open_string_pitch))

            if fret_num < 0:
                fret_num = 0

            notes.append(
                (
                    onset_sec,
                    offset_sec,
                    string_num,
                    fret_num,
                    pitch_midi,
                )
            )

    notes.sort(key=lambda x: x[0])
    return notes


def process_single_track(
    track_object,
    output_file_base,
    target_sr,
    hop_size,
    num_cqt_bins,
    cqt_bins_per_octave,
    cqt_fmin,
):
    features_path = f"{output_file_base}_features.pt"
    labels_path = f"{output_file_base}_labels.pt"

    if os.path.exists(features_path) and os.path.exists(labels_path):
        return "skipped"

    audio_file_path = None

    if (
        hasattr(track_object, "audio_mix_path")
        and track_object.audio_mix_path
        and os.path.exists(track_object.audio_mix_path)
    ):
        audio_file_path = track_object.audio_mix_path

    elif (
        hasattr(track_object, "audio_mic_path")
        and track_object.audio_mic_path
        and os.path.exists(track_object.audio_mic_path)
    ):
        audio_file_path = track_object.audio_mic_path

    if not audio_file_path:
        print(f"  Error: No available audio file for {track_object.track_id}")
        return "error"

    try:
        audio, _ = librosa.load(audio_file_path, sr=target_sr, mono=True)

        cqt_spectrogram = librosa.cqt(
            y=audio,
            sr=target_sr,
            hop_length=hop_size,
            fmin=cqt_fmin,
            n_bins=num_cqt_bins,
            bins_per_octave=cqt_bins_per_octave,
        )

        log_cqt_spectrogram = librosa.amplitude_to_db(
            np.abs(cqt_spectrogram), ref=np.max
        )

    except Exception as e:
        print(
            f"  Error while processing audio for {track_object.track_id}: {e}"
        )
        return "error"

    if (
        not hasattr(track_object, "jams_path")
        or not track_object.jams_path
        or not os.path.exists(track_object.jams_path)
    ):
        print(f"  Error: Missing JAMS file for {track_object.track_id}")
        return "error"

    try:
        annotations = extract_annotations_from_jams(track_object.jams_path)

    except Exception as e:
        print(
            f"  Error while extracting JAMS annotations for {track_object.track_id}: {e}"
        )
        return "error"

    if not annotations:
        print(f"  Error: No annotations found in JAMS file for {track_object.track_id}")
        return "error"

    torch.save(
        torch.tensor(log_cqt_spectrogram, dtype=torch.float32),
        features_path,
    )

    labels_array = np.array(annotations, dtype=np.float32)
    torch.save(torch.from_numpy(labels_array), labels_path)

    return "processed"


def preprocess_guitarset_data(
    guitarset_data_home,
    processed_output_base_dir,
    track_ids_map,
    audio_sample_rate,
    audio_hop_length,
    audio_n_cqt_bins,
    audio_cqt_bins_per_octave,
    audio_cqt_fmin,
):
    print("Starting GuitarSet preprocessing.")
    print(f"Dataset directory (guitarset_data_home): {guitarset_data_home}")
    print(
        f"Output directory (processed_output_base_dir): {processed_output_base_dir}"
    )

    try:
        guitarset_instance = mirdata.initialize(
            "guitarset", data_home=guitarset_data_home
        )

    except Exception as e:
        print(
            f"Critical error: Failed to initialize GuitarSet. Error: {e}"
        )
        return

    processing_stats = {
        "train": {"processed": 0, "skipped": 0, "errors": 0},
        "validation": {"processed": 0, "skipped": 0, "errors": 0},
        "test": {"processed": 0, "skipped": 0, "errors": 0},
    }

    for split_type, track_id_list_for_split in track_ids_map.items():

        if not track_id_list_for_split:
            print(f"\nNo tracks to process in the {split_type} split.")
            continue

        print(
            f"\nProcessing {split_type} split ({len(track_id_list_for_split)} tracks)"
        )

        current_split_output_dir = os.path.join(
            processed_output_base_dir, split_type
        )
        os.makedirs(current_split_output_dir, exist_ok=True)

        for current_track_id in tqdm(
            track_id_list_for_split,
            desc=f"Processing {split_type}",
            unit="track",
        ):
            try:
                track_data_object = guitarset_instance.track(current_track_id)

                track_id_filename_base = os.path.splitext(
                    os.path.basename(current_track_id)
                )[0]

                output_file_path_base = os.path.join(
                    current_split_output_dir,
                    track_id_filename_base,
                )

                status_msg = process_single_track(
                    track_data_object,
                    output_file_path_base,
                    audio_sample_rate,
                    audio_hop_length,
                    audio_n_cqt_bins,
                    audio_cqt_bins_per_octave,
                    audio_cqt_fmin,
                )

                if status_msg in processing_stats[split_type]:
                    processing_stats[split_type][status_msg] += 1
                else:
                    print(
                        f"Warning: Unknown status '{status_msg}' for track {current_track_id}"
                    )
                    processing_stats[split_type]["errors"] += 1

            except mirdata.core.errors.TrackIdError:
                print(
                    f"  Error: Track ID '{current_track_id}' not found in mirdata GuitarSet."
                )
                processing_stats[split_type]["errors"] += 1

            except Exception as e:
                print(
                    f"  Unexpected error (outside process_single_track) for track {current_track_id}: {e}"
                )
                processing_stats[split_type]["errors"] += 1

        print()

    print("\n--- Preprocessing Summary ---")

    total_tracks_for_processing = sum(
        len(val) for val in track_ids_map.values()
    )

    print(
        f"Total tracks scheduled for processing: {total_tracks_for_processing}"
    )

    for split_key_name in ["train", "validation", "test"]:
        if split_key_name in processing_stats:
            print(f"  {split_key_name.capitalize()} split:")
            print(
                f"    Newly processed: {processing_stats[split_key_name]['processed']}"
            )
            print(
                f"    Skipped (already existed): {processing_stats[split_key_name]['skipped']}"
            )
            print(
                f"    Errors: {processing_stats[split_key_name]['errors']}"
            )

    print(
        f"Processed data has been saved to: {processed_output_base_dir}"
    )