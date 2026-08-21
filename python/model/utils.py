import os
import torch
import traceback
import json
from model import architecture
import config


def load_best_model(model_class, model_path, run_config_path, device):
    if not os.path.exists(model_path):
        print(f"CRITICAL ERROR: Model file not found at '{model_path}'")
        return None

    if not os.path.exists(run_config_path):
        print(
            f"CRITICAL ERROR: Configuration file not found at '{run_config_path}'"
        )
        return None

    print(
        f"--- Attempting to load model from file: {os.path.basename(model_path)} ---"
    )

    try:
        with open(run_config_path, "r", encoding="utf-8") as f:
            run_config = json.load(f)

        hyperparams = run_config["hyperparameters_tuned"]
        static_params = run_config["static_parameters"]

        cnn_hyperparams = {
            "input_channels": hyperparams.get("CNN_INPUT_CHANNELS", config.CNN_INPUT_CHANNELS),
            "output_channels_list": hyperparams.get("CNN_OUTPUT_CHANNELS_LIST"),
            "kernel_sizes": hyperparams.get("CNN_KERNEL_SIZES"),
            "strides": hyperparams.get("CNN_STRIDES"),
            "paddings": hyperparams.get("CNN_PADDINGS"),
            "pooling_kernels": hyperparams.get("CNN_POOLING_KERNELS"),
            "pooling_strides": hyperparams.get("CNN_POOLING_STRIDES"),
        }
        
        cnn_hyperparams = {k: v for k, v in cnn_hyperparams.items() if v is not None}

        temp_cnn_model = architecture.TabCNN(**cnn_hyperparams)

        with torch.no_grad():
            n_bins_cqt = static_params.get("N_BINS_CQT", 168)
            dummy_input = torch.randn(1, 1, n_bins_cqt, 32)
            cnn_output = temp_cnn_model(dummy_input)

            calculated_cnn_out_dim = (
                cnn_output.shape[1] * cnn_output.shape[2]
            )

        del temp_cnn_model
        
        crnn_cnn_kwargs = {f"cnn_{k}": v for k, v in cnn_hyperparams.items()}

        model_init_params = {
            "num_frames_rnn_input_dim": calculated_cnn_out_dim,
            "rnn_type": hyperparams.get(
                "RNN_TYPE",
                config.RNN_TYPE_DEFAULT,
            ),
            "rnn_hidden_size": hyperparams["RNN_HIDDEN_SIZE"],
            "rnn_layers": hyperparams["RNN_LAYERS"],
            "rnn_dropout": hyperparams["RNN_DROPOUT"],
            "rnn_bidirectional": hyperparams.get(
                "RNN_BIDIRECTIONAL",
                config.RNN_BIDIRECTIONAL_DEFAULT,
            ),
            **crnn_cnn_kwargs 
        }

        print(
            f"Reconstructed model initialization parameters: "
            f"{model_init_params}"
        )

        loaded_model = model_class(**model_init_params)

        print(f"Loading model weights onto device: {device.type}")

        state_dict = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )

        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {
                k[7:]: v for k, v in state_dict.items()
            }

        loaded_model.load_state_dict(state_dict)

        loaded_model.to(device)
        loaded_model.eval()

        print(
            f"Model successfully loaded and moved to device: {device}"
        )

        return loaded_model

    except Exception as e:
        print(
            f"\nCRITICAL ERROR during model loading: {e}"
        )
        traceback.print_exc()
        return None