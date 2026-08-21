import torch
from torch.nn.utils.rnn import pad_sequence
import config


def collate_fn_pad(batch_data):
    features_list = []
    onset_targets_list = []
    fret_targets_list = []
    raw_labels_list = []
    track_ids_list = []
    sequence_lengths = []

    for (
        features_sample,
        (onset_targets_sample, fret_targets_sample),
        raw_labels_sample,
        track_id_sample,
    ) in batch_data:
        features_list.append(features_sample.T)
        onset_targets_list.append(onset_targets_sample)
        fret_targets_list.append(fret_targets_sample)
        raw_labels_list.append(raw_labels_sample)
        track_ids_list.append(track_id_sample)
        sequence_lengths.append(features_sample.shape[1])

    padded_features_batch = pad_sequence(
        features_list, batch_first=True, padding_value=0.0
    ).permute(0, 2, 1)

    padded_onset_targets_batch = pad_sequence(
        onset_targets_list, batch_first=True, padding_value=0.0
    )
    padded_fret_targets_batch = pad_sequence(
        fret_targets_list, batch_first=True, padding_value=config.FRET_PADDING_VALUE
    )

    padded_labels_tuple = (padded_onset_targets_batch, padded_fret_targets_batch)
    lengths_tensor_batch = torch.tensor(sequence_lengths, dtype=torch.int64)

    return (
        padded_features_batch,
        padded_labels_tuple,
        lengths_tensor_batch,
        raw_labels_list,
        track_ids_list,
    )
