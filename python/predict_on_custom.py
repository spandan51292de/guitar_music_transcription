import os
import glob
import json
import torch
import numpy as np
import librosa
import config
from model.architecture import GuitarTabCRNN
from model.utils import load_best_model
from vizualization import plotting

def preprocess_custom_audio(audio_path, device):
    print(f"Loading and resampling audio: {os.path.basename(audio_path)}")
    audio, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE)
    
    print("Computing Constant-Q Transform (CQT)...")
    cqt = librosa.cqt(
        audio, 
        sr=sr, 
        hop_length=config.HOP_LENGTH, 
        fmin=config.FMIN_CQT, 
        n_bins=config.N_BINS_CQT, 
        bins_per_octave=config.BINS_PER_OCTAVE_CQT
    )
    
    cqt_mag = np.abs(cqt)
    cqt_log = librosa.amplitude_to_db(cqt_mag, ref=np.max)
        
    tensor_cqt = torch.tensor(cqt_log, dtype=torch.float32).unsqueeze(0).to(device)
    return tensor_cqt

def get_latest_run_dir(artifacts_dir):
    runs = [os.path.join(artifacts_dir, d) for d in os.listdir(artifacts_dir) if d.startswith("run_")]
    runs = [r for r in runs if os.path.exists(os.path.join(r, "best_model.pth"))]
    if not runs:
        raise FileNotFoundError("No valid training runs found with a saved best_model.pth.")
    runs.sort(key=os.path.getmtime, reverse=True)
    return runs[0]

def run_custom_prediction(artifacts_dir, audio_dir, device):
    print("\n" + "="*50)
    print("INITIALIZING CUSTOM INFERENCE PIPELINE")
    print("="*50)
    
    latest_run_dir = get_latest_run_dir(artifacts_dir)
    model_path = os.path.join(latest_run_dir, "best_model.pth")
    config_path = os.path.join(latest_run_dir, "run_configuration.json")
    
    print(f"Targeting model directory: {os.path.basename(latest_run_dir)}")
    
    model = load_best_model(
        model_class=GuitarTabCRNN,
        model_path=model_path,
        run_config_path=config_path,
        device=device
    )
    model.eval()
    
    audio_files = []
    for ext in ["*.wav", "*.mp3", "*.flac"]:
        audio_files.extend(glob.glob(os.path.join(audio_dir, ext)))
        
    if not audio_files:
        print(f"WARNING: No audio files found in {audio_dir}. Please add some recordings.")
        return
        
    print(f"Found {len(audio_files)} custom audio tracks for processing.")
    
    output_dir = os.path.join(audio_dir, "transcriptions")
    os.makedirs(output_dir, exist_ok=True)
    
    optimal_threshold = 0.5100 
    
    with torch.no_grad():
        for audio_path in audio_files:
            file_name = os.path.basename(audio_path)
            print(f"\nTranscribing: {file_name}...")
            
            input_tensor = preprocess_custom_audio(audio_path, device)
            
            onset_preds, tdr_preds = model(input_tensor)
            
            binary_tdr = (torch.sigmoid(tdr_preds) >= optimal_threshold).cpu().numpy().astype(int)
            
            binary_tdr = binary_tdr.squeeze(0)
            
            output_array_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_prediction.npy")
            np.save(output_array_path, binary_tdr)
            
            print(f"Success! Prediction array saved to: {output_array_path}")

    print("\n" + "="*50)
    print("CUSTOM INFERENCE COMPLETE")
    print("="*50)