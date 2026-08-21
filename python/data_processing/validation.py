import torch
import config


def validate_tensor_shapes_and_types(cqt, onsets, frets, track_identifier, shape_params):
    is_valid = True
    calculated_num_frames = -1

    if isinstance(cqt, torch.Tensor):
        if cqt.ndim == 2 and cqt.shape[0] == shape_params['N_BINS_CQT']:
            calculated_num_frames = cqt.shape[1]
            if cqt.dtype != torch.float32:
                print(f"  [FAIL] {track_identifier} - cqt: data type is {cqt.dtype}, expected torch.float32")
                is_valid = False
        else:
            print(f"  [FAIL] {track_identifier} - cqt: unexpected shape {cqt.shape}. Expected [{shape_params['N_BINS_CQT']}, num_frames]")
            is_valid = False
    else:
        print(f"  [FAIL] {track_identifier} - cqt: object is not a tensor")
        is_valid = False

    if calculated_num_frames == -1:
        print(f"  [FAIL] {track_identifier} - Unable to determine the number of frames from CQT.")
        return False

    if isinstance(onsets, torch.Tensor):
        if not (onsets.ndim == 2 and onsets.shape[0] == calculated_num_frames and onsets.shape[1] == config.DEFAULT_NUM_STRINGS):
            print(f"  [FAIL] {track_identifier} - onsets: unexpected shape {onsets.shape}. Expected [{calculated_num_frames}, {config.DEFAULT_NUM_STRINGS}]")
            is_valid = False
        if onsets.dtype != torch.float32:
            print(f"  [FAIL] {track_identifier} - onsets: data type is {onsets.dtype}, expected torch.float32")
            is_valid = False
    else:
        print(f"  [FAIL] {track_identifier} - onsets: object is not a tensor")
        is_valid = False

    if isinstance(frets, torch.Tensor):
        if not (frets.ndim == 2 and frets.shape[0] == calculated_num_frames and frets.shape[1] == config.DEFAULT_NUM_STRINGS):
            print(f"  [FAIL] {track_identifier} - frets: unexpected shape {frets.shape}. Expected [{calculated_num_frames}, {config.DEFAULT_NUM_STRINGS}]")
            is_valid = False
        if frets.dtype != torch.long:
            print(f"  [FAIL] {track_identifier} - frets: data type is {frets.dtype}, expected torch.long")
            is_valid = False
    else:
        print(f"  [FAIL] {track_identifier} - frets: object is not a tensor")
        is_valid = False

    if calculated_num_frames == 0:
        print(f"  [WARN] {track_identifier} - Track contains 0 frames.")

    return is_valid


def validate_tensor_values(cqt, onsets, frets, track_identifier):
    is_valid = True

    if not torch.all((onsets == 0.0) | (onsets == 1.0)):
        print(f"  [FAIL] {track_identifier} - onsets: values are not exclusively 0.0 or 1.0. Min: {onsets.min()}, Max: {onsets.max()}")
        is_valid = False

    if onsets.sum() == 0:
        print(f"  [WARN] {track_identifier} - onsets: all values are 0.")

    max_allowed_fret_val = config.MAX_FRETS + config.FRET_SILENCE_CLASS_OFFSET
    if torch.any(frets < 0) or torch.any(frets > max_allowed_fret_val):
        print(f"  [FAIL] {track_identifier} - frets: contains values outside valid range [0, {max_allowed_fret_val}].")
        is_valid = False

    if torch.isnan(cqt).any():
        print(f"  [FAIL] {track_identifier} - cqt: contains NaN values.")
        is_valid = False

    if torch.isinf(cqt).any():
        print(f"  [FAIL] {track_identifier} - cqt: contains Inf values.")
        is_valid = False

    return is_valid


def run_full_data_validation(dataset_to_validate, validation_shape_params):
    print(f"\n--- Starting Full Data Validation ({len(dataset_to_validate)} tracks) ---")

    if not dataset_to_validate:
        print("Dataset is empty. Nothing to validate.")
        return

    all_items_valid_shape_type = True
    all_items_valid_values = True

    validation_stats = {
        'total_frames': 0,
        'total_onsets': 0,
        'frames_with_any_onset': 0,
        'min_frames': float('inf'),
        'max_frames': 0,
        'track_id_min_frames': "",
        'track_id_max_frames': ""
    }

    for item_idx in range(len(dataset_to_validate)):
        try:
            cqt, (onsets, frets), raw_labels, current_track_id = dataset_to_validate[item_idx]

            print(f"\nValidating track {item_idx + 1}/{len(dataset_to_validate)}: {current_track_id}")

            if not validate_tensor_shapes_and_types(cqt, onsets, frets, current_track_id, validation_shape_params):
                all_items_valid_shape_type = False

            if not validate_tensor_values(cqt, onsets, frets, current_track_id):
                all_items_valid_values = False

            num_frames_in_item = cqt.shape[1]
            validation_stats['total_frames'] += num_frames_in_item

            if num_frames_in_item < validation_stats['min_frames']:
                validation_stats['min_frames'] = num_frames_in_item
                validation_stats['track_id_min_frames'] = current_track_id

            if num_frames_in_item > validation_stats['max_frames']:
                validation_stats['max_frames'] = num_frames_in_item
                validation_stats['track_id_max_frames'] = current_track_id

            validation_stats['total_onsets'] += onsets.sum().item()
            validation_stats['frames_with_any_onset'] += (onsets.sum(dim=1) > 0).sum().item()

        except Exception as e:
            error_track_id = f"unknown_track_index_{item_idx}"
            print(f"  [FATAL ERROR] {error_track_id} - Unable to load or process sample: {e}")
            all_items_valid_shape_type = False

    print("\n--- Validation Summary ---")
    if not dataset_to_validate:
        return

    print(f"Processed {len(dataset_to_validate)} tracks.")
    print("[OK] All processed tracks have valid tensor shapes and data types." if all_items_valid_shape_type else "[ERROR] Problems found with tensor shapes or data types.")
    print("[OK] All tensors in processed tracks have valid value ranges." if all_items_valid_values else "[ERROR/WARNING] Problems found with tensor values.")

    print("\nOverall Statistics:")
    print(f"  Total frames: {validation_stats['total_frames']}")
    if len(dataset_to_validate) > 0 and validation_stats['total_frames'] > 0:
        print(f"  Average frames per track: {validation_stats['total_frames'] / len(dataset_to_validate):.2f}")
    
    print(f"  Minimum frames: {validation_stats['min_frames']} (track: {validation_stats['track_id_min_frames']})")
    print(f"  Maximum frames: {validation_stats['max_frames']} (track: {validation_stats['track_id_max_frames']})")
    print(f"  Total onsets: {validation_stats['total_onsets']}")

    print("--- End of Validation ---")