import os
import shutil
import tempfile
import torch
import numpy as np
import pretty_midi
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import config
from model.architecture import GuitarTabCRNN
from model.utils import load_best_model
from predict_on_custom import preprocess_custom_audio, get_latest_run_dir


ml_models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[SYSTEM] Booting AI Engine...")
    try:
        artifacts_dir = os.path.join(os.getcwd(), "results", "hyperparam_search")
        latest_run_dir = get_latest_run_dir(artifacts_dir)
        model_path = os.path.join(latest_run_dir, "best_model.pth")
        config_path = os.path.join(latest_run_dir, "run_configuration.json")
        
        model = load_best_model(GuitarTabCRNN, model_path, config_path, device)
        model.eval()
        ml_models["transcriber"] = model
        print(f"[SYSTEM] Model loaded successfully on {device.type.upper()}")
        yield
    except Exception as e:
        print(f"[FATAL] Failed to load ML model: {str(e)}")
        raise e
    finally:
        ml_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[SYSTEM] AI Engine gracefully shut down.")

app = FastAPI(
    title="GuitarTab AI Engine",
    description="Automated Music Transcription API for Fingerstyle Guitar",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def cleanup_temp_dir(dir_path: str):
    """Background task to delete temporary files after the response is sent."""
    try:
        shutil.rmtree(dir_path)
        print(f"[CLEANUP] Deleted temporary workspace: {dir_path}")
    except Exception as e:
        print(f"[WARNING] Failed to clean up {dir_path}: {e}")

def tensor_to_midi(binary_tdr: np.ndarray, output_path: str):
    """Converts the 3D numpy tensor into a standard MIDI file."""
    guitar = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Guitar (nylon)'))
    string_midi_bases = [40, 45, 50, 55, 59, 64]
    frame_duration = config.HOP_LENGTH / config.SAMPLE_RATE
    silence_fret_index = binary_tdr.shape[2] - 1

    active_notes = {}
    
    for frame_idx in range(binary_tdr.shape[0]):
        time_sec = frame_idx * frame_duration
        active_strings, active_frets = np.where(binary_tdr[frame_idx] == 1)
        current_frame_notes = {}
        
        for string_idx, fret in zip(active_strings, active_frets):
            if fret != silence_fret_index and string_idx not in current_frame_notes:
                current_frame_notes[string_idx] = fret

        for string_idx in list(active_notes.keys()):
            active_fret, start_time = active_notes[string_idx]
            if string_idx not in current_frame_notes or current_frame_notes[string_idx] != active_fret:
                note = pretty_midi.Note(
                    velocity=100, 
                    pitch=string_midi_bases[string_idx] + active_fret, 
                    start=start_time, 
                    end=time_sec
                )
                guitar.notes.append(note)
                del active_notes[string_idx]
                
        for string_idx, fret in current_frame_notes.items():
            if string_idx not in active_notes:
                active_notes[string_idx] = (fret, time_sec)

    midi_data = pretty_midi.PrettyMIDI()
    midi_data.instruments.append(guitar)
    midi_data.write(output_path)

@app.get("/health")
async def health_check():
    """Simple up-check for load balancers and container orchestration."""
    return {"status": "healthy", "gpu_enabled": torch.cuda.is_available()}

@app.post("/transcribe")
async def transcribe_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Ingests an audio file, runs the AI inference pipeline, and returns a MIDI file.
    """
    if not file.filename.endswith(('.wav', '.mp3', '.flac')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload .wav, .mp3, or .flac")

    temp_dir = tempfile.mkdtemp()
    raw_audio_path = os.path.join(temp_dir, file.filename)
    output_midi_path = os.path.join(temp_dir, "transcription.mid")
    
    background_tasks.add_task(cleanup_temp_dir, temp_dir)

    try:
        with open(raw_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"[INFERENCE] Processing requested file: {file.filename}")

        input_tensor = preprocess_custom_audio(raw_audio_path, device)
        
        model = ml_models["transcriber"]
        with torch.no_grad():
            onset_preds, tdr_preds = model(input_tensor)
            
        optimal_threshold = 0.5100
        binary_tdr = (torch.sigmoid(tdr_preds) >= optimal_threshold).cpu().numpy().astype(int).squeeze(0)
        
        tensor_to_midi(binary_tdr, output_midi_path)
        
        return FileResponse(
            path=output_midi_path, 
            media_type="audio/midi", 
            filename=f"transcribed_{os.path.splitext(file.filename)[0]}.mid"
        )

    except Exception as e:
        print(f"[ERROR] Inference pipeline failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during transcription.")