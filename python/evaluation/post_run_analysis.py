import os
import numpy as np
import torch
from tqdm import tqdm
from evaluation import tablature_export, metrics
from training import note_conversion_utils
import config


def find_optimal_threshold_for_tdr(
    all_onset_probs, all_fret_preds, all_raw_labels, config_obj
):
    best_f1, optimal_threshold = -1.0, config.DEFAULT_TDR_THRESHOLD

    thresholds = np.arange(
        config_obj.TDR_THRESH_SEARCH_MIN,
        config_obj.TDR_THRESH_SEARCH_MAX + config_obj.TDR_THRESH_SEARCH_STEP,
        config_obj.TDR_THRESH_SEARCH_STEP,
    )

    for threshold in tqdm(thresholds, desc="Optimizing TDR threshold"):
        f1_scores = []

        for i in range(len(all_onset_probs)):
            predicted_notes = note_conversion_utils.frames_to_notes_for_eval(
                (all_onset_probs[i] > threshold).float(),
                all_fret_preds[i],
                config_obj.HOP_LENGTH,
                config_obj.SAMPLE_RATE,
                config_obj.MAX_FRETS,
            )

            tdr_result = metrics.calculate_note_level_metrics(
                predicted_notes,
                all_raw_labels[i],
            )

            f1_scores.append(tdr_result["tdr_f1"])

        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            optimal_threshold = threshold

    return optimal_threshold, best_f1


def run_final_evaluation_and_visualization(
    model,
    test_dataloader,
    device,
    config_obj,
    output_dir,
    num_top_bottom_files_to_show=3,
):
    model.eval()

    (
        all_onset_probs,
        all_fret_preds,
        all_raw_labels,
        all_onset_gt,
        all_fret_gt,
        all_track_ids,
    ) = ([], [], [], [], [], [])

    print("Step 1/5: Collecting predictions and labels from the entire test set...")

    with torch.no_grad():
        try:
            for (
                features,
                labels,
                lengths,
                raw_labels_batch,
                track_ids_batch,
            ) in tqdm(test_dataloader, desc="Processing test dataset"):

                features = features.to(device)

                onset_targets_b, fret_targets_b = labels

                onset_logits, fret_logits = model(features)

                onset_probs_b = torch.sigmoid(onset_logits)
                fret_pred_indices_b = torch.argmax(fret_logits, dim=-1)

                for j in range(features.size(0)):
                    true_len = lengths[j].item()

                    all_onset_probs.append(
                        onset_probs_b[j, :true_len, :].cpu()
                    )

                    all_fret_preds.append(
                        fret_pred_indices_b[j, :true_len, :].cpu()
                    )

                    all_onset_gt.append(
                        onset_targets_b[j, :true_len, :].cpu()
                    )

                    all_fret_gt.append(
                        fret_targets_b[j, :true_len, :].cpu()
                    )

                    all_raw_labels.append(raw_labels_batch[j])

                all_track_ids.extend(track_ids_batch)

        except ValueError:
            print(
                "\nCRITICAL ERROR: Failed to unpack data from the DataLoader."
            )
            return None, None

    print("Step 2/5: Searching for the optimal TDR threshold...")

    optimal_tdr_threshold, _ = find_optimal_threshold_for_tdr(
        all_onset_probs,
        all_fret_preds,
        all_raw_labels,
        config_obj,
    )

    print(f"Optimal TDR threshold found: {optimal_tdr_threshold:.4f}")

    print("Step 3/5: Computing per-file metrics to create a ranking...")

    file_scores = []

    for i in range(len(all_track_ids)):
        pred_notes = note_conversion_utils.frames_to_notes_for_eval(
            (all_onset_probs[i] > optimal_tdr_threshold).float(),
            all_fret_preds[i],
            config_obj.HOP_LENGTH,
            config_obj.SAMPLE_RATE,
            config_obj.MAX_FRETS,
        )

        tdr_result = metrics.calculate_note_level_metrics(
            pred_notes,
            all_raw_labels[i],
        )

        file_scores.append(
            {
                "track_id": all_track_ids[i],
                "f1_score": tdr_result["tdr_f1"],
                "index": i,
            }
        )

    sorted_files = sorted(
        file_scores,
        key=lambda x: x["f1_score"],
        reverse=True,
    )

    print("Step 4/5: Computing final averaged metrics...")

    final_metrics = metrics.full_evaluation(
        model,
        test_dataloader,
        device,
        config_obj,
        optimal_tdr_threshold,
    )

    print(
        "Step 5/5: Generating tablature files for the best and worst predictions..."
    )

    os.makedirs(output_dir, exist_ok=True)

    top_files = sorted_files[:num_top_bottom_files_to_show]
    worst_files = sorted_files[-num_top_bottom_files_to_show:]

    print(
        f"\n--- Top {num_top_bottom_files_to_show} Results (Highest TDR F1) ---"
    )

    for file_info in top_files:
        print(
            f"  - File: {file_info['track_id']}, "
            f"F1 Score: {file_info['f1_score']:.4f}"
        )

        idx = file_info["index"]

        tablature_export.generate_text_tablature_comparison(
            all_onset_probs[idx],
            all_fret_preds[idx],
            all_onset_gt[idx],
            all_fret_gt[idx],
            file_info["track_id"],
            optimal_tdr_threshold,
            config_obj.MAX_FRETS,
            output_dir,
        )

    print(
        f"\n--- Bottom {num_top_bottom_files_to_show} Results (Lowest TDR F1) ---"
    )

    for file_info in worst_files:
        print(
            f"  - File: {file_info['track_id']}, "
            f"F1 Score: {file_info['f1_score']:.4f}"
        )

        idx = file_info["index"]

        tablature_export.generate_text_tablature_comparison(
            all_onset_probs[idx],
            all_fret_preds[idx],
            all_onset_gt[idx],
            all_fret_gt[idx],
            file_info["track_id"],
            optimal_tdr_threshold,
            config_obj.MAX_FRETS,
            output_dir,
        )

    print(f"\nDone. Tablature files have been saved to: {output_dir}")

    return final_metrics, sorted_files