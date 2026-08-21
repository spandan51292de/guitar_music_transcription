import os
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import torchaudio
import mirdata
import random
from scipy.signal import convolve, iirfilter, lfilter
import config


def _augment_time_stretch(audio_data, stretch_rate, current_raw_labels):
    stretched_audio = librosa.effects.time_stretch(
        y=audio_data, rate=stretch_rate
    )
    adjusted_labels = current_raw_labels

    if current_raw_labels is not None and current_raw_labels.numel() > 0:
        adjusted_labels = current_raw_labels.clone()
        adjusted_labels[:, 0] /= stretch_rate
        adjusted_labels[:, 1] /= stretch_rate

    return stretched_audio, adjusted_labels


def _augment_add_noise(audio_data, noise_amplitude):
    noise_signal = np.random.randn(len(audio_data)) * noise_amplitude
    return audio_data + noise_signal.astype(audio_data.dtype)


def _augment_random_gain(audio_data, gain_value):
    return audio_data * gain_value


def _augment_reverb(audio_data, sample_rate):
    params = config.DATASET_TRAIN_AUGMENTATION_REVERB_PARAMS

    if params["enabled"] and random.random() < params["probability"]:
        decay_seconds = random.uniform(*params["decay_seconds_range"])
        wet_level = random.uniform(*params["wet_level_range"])

        decay_samples = int(decay_seconds * sample_rate)

        if decay_samples <= 0:
            return audio_data

        impulse_response = (
            np.random.randn(decay_samples)
            * np.exp(-np.linspace(0, 5, decay_samples))
        )

        reverbed_audio = convolve(
            audio_data,
            impulse_response,
            mode="same"
        )

        reverbed_audio_norm = (
            reverbed_audio /
            (np.max(np.abs(reverbed_audio)) + 1e-8)
        ) * np.max(np.abs(audio_data))

        mixed_audio = (
            (1 - wet_level) * audio_data
            + wet_level * reverbed_audio_norm
        )

        return mixed_audio.astype(audio_data.dtype)

    return audio_data


def _augment_eq(audio_data, sample_rate):
    params = config.DATASET_TRAIN_AUGMENTATION_EQ_PARAMS

    if params["enabled"] and random.random() < params["probability"]:
        nyquist = 0.5 * sample_rate

        low = random.uniform(
            *params["low_cutoff_hz_range"]
        ) / nyquist

        high = random.uniform(
            *params["high_cutoff_hz_range"]
        ) / nyquist

        if low >= high:
            return audio_data

        b, a = iirfilter(
            N=4,
            Wn=[low, high],
            btype="band",
            ftype="butter"
        )

        filtered_audio = lfilter(
            b,
            a,
            audio_data
        )

        return filtered_audio.astype(audio_data.dtype)

    return audio_data


def _augment_clipping(audio_data):
    params = config.DATASET_TRAIN_AUGMENTATION_CLIPPING_PARAMS

    if params["enabled"] and random.random() < params["probability"]:
        clip_threshold = random.uniform(
            *params["threshold_range"]
        )

        return np.clip(
            audio_data,
            -clip_threshold,
            clip_threshold
        )

    return audio_data


class GuitarSetTabDataset(Dataset):
    def __init__(
        self,
        processed_data_base_dir,
        data_split_name,
        audio_hop_length,
        audio_sample_rate,
        max_fret_value,
        audio_n_cqt_bins,
        audio_cqt_bins_per_octave,
        audio_cqt_fmin,
        label_transform_function=None,
        guitarset_data_home=None,
        enable_audio_augmentations=True,
        aug_p_time_stretch=0.5,
        aug_time_stretch_limits=(0.85, 1.15),
        aug_p_add_noise=0.5,
        aug_noise_level_limits=(0.0005, 0.005),
        aug_p_random_gain=0.5,
        aug_gain_limits=(0.7, 1.3),
        enable_specaugment=True,
        specaug_time_mask_param=40,
        specaug_freq_mask_param=20,
        specaug_num_time_masks=1,
        specaug_num_freq_masks=1,
    ):
        self.processed_data_base_dir = processed_data_base_dir
        self.data_split_name = data_split_name

        self.current_split_data_dir = os.path.join(
            self.processed_data_base_dir,
            self.data_split_name
        )

        self.audio_sample_rate = audio_sample_rate
        self.audio_hop_length = audio_hop_length
        self.max_fret_value = max_fret_value

        self.audio_n_cqt_bins = audio_n_cqt_bins
        self.audio_cqt_bins_per_octave = audio_cqt_bins_per_octave
        self.audio_cqt_fmin = audio_cqt_fmin

        self.label_transform_function = label_transform_function
        self.guitarset_data_home = guitarset_data_home

        self.guitarset_loader_instance = None

        self.enable_audio_augmentations = (
            enable_audio_augmentations
            if self.data_split_name == "train"
            else False
        )
        
        self.aug_p_time_stretch = aug_p_time_stretch
        self.aug_time_stretch_limits = aug_time_stretch_limits
        self.aug_p_add_noise = aug_p_add_noise
        self.aug_noise_level_limits = aug_noise_level_limits
        self.aug_p_random_gain = aug_p_random_gain
        self.aug_gain_limits = aug_gain_limits

        self.enable_specaugment = (
            enable_specaugment
            if self.data_split_name == "train"
            else False
        )

        if self.enable_specaugment:
            time_masks = [
                torchaudio.transforms.TimeMasking(
                    time_mask_param=specaug_time_mask_param
                )
                for _ in range(specaug_num_time_masks)
            ]
            freq_masks = [
                torchaudio.transforms.FrequencyMasking(
                    freq_mask_param=specaug_freq_mask_param
                )
                for _ in range(specaug_num_freq_masks)
            ]
            self.specaugment_transform_op = torch.nn.Sequential(
                *(time_masks + freq_masks)
            )

        if self.enable_audio_augmentations:
            if self.guitarset_data_home is None:
                raise ValueError(
                    "`guitarset_data_home` must be provided for the "
                    "'train' split when audio augmentations are enabled."
                )
            self.guitarset_loader_instance = mirdata.initialize(
                "guitarset",
                data_home=self.guitarset_data_home
            )
            if not self.guitarset_loader_instance.track_ids:
                raise RuntimeError(
                    f"mirdata.GuitarSet did not find any tracks in "
                    f"guitarset_data_home='{self.guitarset_data_home}'."
                )

        self.feature_sources = []
        self.label_file_paths = []
        self.base_track_ids = []
        self.full_track_ids = []

        ids_list_file_path = os.path.join(
            self.processed_data_base_dir,
            f"{self.data_split_name}_ids.txt"
        )

        if not os.path.exists(ids_list_file_path):
            raise FileNotFoundError(
                f"Split ID file for '{data_split_name}' was not found: "
                f"{ids_list_file_path}"
            )

        with open(ids_list_file_path, "r") as f_ids:
            for line_content in f_ids:
                full_track_id_from_file = line_content.strip()

                if not full_track_id_from_file:
                    continue

                base_track_id_for_filename = os.path.splitext(
                    os.path.basename(full_track_id_from_file)
                )[0]

                label_file_path = os.path.join(
                    self.current_split_data_dir,
                    f"{base_track_id_for_filename}_labels.pt"
                )

                if not os.path.exists(label_file_path):
                    continue

                if self.enable_audio_augmentations:
                    if self.guitarset_loader_instance is None:
                        continue
                    try:
                        track_metadata = (
                            self.guitarset_loader_instance.track(
                                full_track_id_from_file
                            )
                        )
                        audio_source_path = None

                        if (
                            hasattr(track_metadata, "audio_mix_path")
                            and track_metadata.audio_mix_path
                            and os.path.exists(track_metadata.audio_mix_path)
                        ):
                            audio_source_path = track_metadata.audio_mix_path
                        elif (
                            hasattr(track_metadata, "audio_mic_path")
                            and track_metadata.audio_mic_path
                            and os.path.exists(track_metadata.audio_mic_path)
                        ):
                            audio_source_path = track_metadata.audio_mic_path

                        if audio_source_path:
                            self.feature_sources.append(audio_source_path)
                            self.label_file_paths.append(label_file_path)
                            self.base_track_ids.append(base_track_id_for_filename)
                            self.full_track_ids.append(full_track_id_from_file)
                    except mirdata.core.errors.TrackIdError:
                        pass
                    except Exception as e:
                        print(
                            f"Error while retrieving audio path for "
                            f"{full_track_id_from_file}: {e}"
                        )
                else:
                    feature_file_path = os.path.join(
                        self.current_split_data_dir,
                        f"{base_track_id_for_filename}_features.pt"
                    )

                    if os.path.exists(feature_file_path):
                        self.feature_sources.append(feature_file_path)
                        self.label_file_paths.append(label_file_path)
                        self.base_track_ids.append(base_track_id_for_filename)
                        self.full_track_ids.append(full_track_id_from_file)

        if not self.feature_sources:
            print(
                f"Warning: No data files found for split "
                f"'{self.data_split_name}' in "
                f"'{self.current_split_data_dir}'. "
                f"The dataset will be empty."
            )

    def __len__(self):
        return len(self.feature_sources)


    def __getitem__(self, item_idx):

        if item_idx >= len(self.feature_sources):
            raise IndexError(
                f"Index {item_idx} is out of range "
                f"for dataset size {len(self.feature_sources)}"
            )

        feature_source_path = self.feature_sources[item_idx]
        labels_file_path = self.label_file_paths[item_idx]

        loaded_raw_labels = torch.load(
            labels_file_path,
            weights_only=False
        )

        labels_for_transform = loaded_raw_labels

        if self.enable_audio_augmentations:

            audio_file_path = feature_source_path

            audio_data, sr_loaded = librosa.load(
                audio_file_path,
                sr=self.audio_sample_rate,
                mono=True
            )

            if sr_loaded != self.audio_sample_rate:
                audio_data = librosa.resample(
                    y=audio_data,
                    orig_sr=sr_loaded,
                    target_sr=self.audio_sample_rate
                )

            current_labels_for_stretch = loaded_raw_labels

            if np.random.rand() < self.aug_p_time_stretch:
                if current_labels_for_stretch is loaded_raw_labels:
                    current_labels_for_stretch = loaded_raw_labels.clone()

                stretch_factor_val = np.random.uniform(
                    self.aug_time_stretch_limits[0],
                    self.aug_time_stretch_limits[1]
                )

                audio_data, current_labels_for_stretch = (
                    _augment_time_stretch(
                        audio_data,
                        stretch_factor_val,
                        current_labels_for_stretch
                    )
                )

            labels_for_transform = current_labels_for_stretch

            if np.random.rand() < self.aug_p_random_gain:
                gain_factor_val = np.random.uniform(
                    self.aug_gain_limits[0],
                    self.aug_gain_limits[1]
                )

                audio_data = _augment_random_gain(
                    audio_data,
                    gain_factor_val
                )

            if np.random.rand() < self.aug_p_add_noise:
                noise_level_val = np.random.uniform(
                    self.aug_noise_level_limits[0],
                    self.aug_noise_level_limits[1]
                )

                audio_data = _augment_add_noise(
                    audio_data,
                    noise_level_val
                )

            audio_data = _augment_reverb(
                audio_data,
                self.audio_sample_rate
            )

            audio_data = _augment_eq(
                audio_data,
                self.audio_sample_rate
            )

            audio_data = _augment_clipping(
                audio_data
            )

            cqt_spec = librosa.cqt(
                y=audio_data,
                sr=self.audio_sample_rate,
                hop_length=self.audio_hop_length,
                fmin=self.audio_cqt_fmin,
                n_bins=self.audio_n_cqt_bins,
                bins_per_octave=self.audio_cqt_bins_per_octave,
            )

            log_cqt_spec = librosa.amplitude_to_db(
                np.abs(cqt_spec),
                ref=np.max
            )

            input_features = torch.tensor(
                log_cqt_spec,
                dtype=torch.float32
            )

        else:
            precomputed_features_path = feature_source_path

            input_features = torch.load(
                precomputed_features_path,
                weights_only=False
            )

        if self.enable_specaugment:
            input_features = input_features.unsqueeze(0)
            input_features = self.specaugment_transform_op(input_features)
            input_features = input_features.squeeze(0)

        output_labels_tuple = (
            self.label_transform_function(
                labels_for_transform,
                input_features,
                self.audio_hop_length,
                self.audio_sample_rate,
                self.max_fret_value,
            )
            if self.label_transform_function
            else labels_for_transform
        )

        track_id = self.full_track_ids[item_idx]

        return (
            input_features,
            output_labels_tuple,
            loaded_raw_labels,
            track_id,
        )


    def get_full_track_id_for_item(self, item_idx):
        if item_idx >= len(self.full_track_ids):
            raise IndexError(
                f"Index {item_idx} is outside the range of "
                f"full_track_ids ({len(self.full_track_ids)})"
            )

        return self.full_track_ids[item_idx]
    
def create_frame_level_labels(
    raw_annotation_tensor,
    feature_map_tensor,
    frame_hop_length,
    audio_sr,
    fret_max_value,
):

    if (
        frame_hop_length is None
        or audio_sr is None
        or fret_max_value is None
    ):
        raise ValueError(
            "Missing parameters in create_frame_level_labels"
        )


    num_audio_frames = feature_map_tensor.shape[1]

    num_guitar_strings = config.DEFAULT_NUM_STRINGS


    onset_targets_matrix = torch.zeros(
        (num_audio_frames, num_guitar_strings),
        dtype=torch.float32
    )

    fret_targets_matrix = torch.full(
        (num_audio_frames, num_guitar_strings),
        fret_max_value + config.FRET_SILENCE_CLASS_OFFSET,
        dtype=torch.long,
    )


    time_duration_per_frame = (
        frame_hop_length / audio_sr
    )


    if (
        raw_annotation_tensor is not None
        and raw_annotation_tensor.numel() > 0
    ):

        for i in range(raw_annotation_tensor.shape[0]):

            onset_time_sec = (
                raw_annotation_tensor[i, 0].item()
            )

            offset_time_sec = (
                raw_annotation_tensor[i, 1].item()
            )

            string_index_val = int(
                raw_annotation_tensor[i, 2].item()
            )

            fret_number_val = int(
                raw_annotation_tensor[i, 3].item()
            )


            onset_frame_idx = min(
                max(
                    0,
                    int(
                        round(
                            onset_time_sec
                            / time_duration_per_frame
                        )
                    )
                ),
                num_audio_frames - 1,
            )


            offset_frame_idx = min(
                max(
                    0,
                    int(
                        round(
                            offset_time_sec
                            / time_duration_per_frame
                        )
                    )
                ),
                num_audio_frames - 1,
            )


            if not (
                0 <= string_index_val < num_guitar_strings
            ):
                continue


            if onset_frame_idx < num_audio_frames:

                onset_targets_matrix[
                    onset_frame_idx,
                    string_index_val
                ] = 1.0


            encoded_fret_value = (

                min(
                    fret_number_val,
                    fret_max_value
                )

                if fret_number_val >= 0

                else (
                    fret_max_value
                    + config.FRET_SILENCE_CLASS_OFFSET
                )
            )


            for current_frame_idx in range(
                onset_frame_idx,
                offset_frame_idx + 1
            ):

                if current_frame_idx < num_audio_frames:

                    fret_targets_matrix[
                        current_frame_idx,
                        string_index_val
                    ] = encoded_fret_value


    return (
        onset_targets_matrix,
        fret_targets_matrix
    )