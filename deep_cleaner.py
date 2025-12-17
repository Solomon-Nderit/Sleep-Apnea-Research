import os
import time
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
import soundfile as sf
import librosa
import numpy as np

# --- CONFIGURATION ---
INPUT_FOLDER = "./experiment_data/SNR_-5dB"
OUTPUT_FOLDER = "./experiment_data/Denoised_DeepLearning"
TARGET_SR = 16000

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# 1. Load the Pre-trained Model (Runs on CPU if no GPU)
# "htdemucs" is the latest hybrid transformer model
print("⏳ Loading Demucs Neural Net (this grabs ~100MB)...")
model = get_model(name="htdemucs")
model.cpu()
model.eval()

files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.wav')]
files = files[:10] # LIMIT TO 10 FILES FIRST (It is slow on CPU!)

print(f"🧠 Starting AI Cleanup on {len(files)} files...")
processing_times = []

for i, f in enumerate(files):
    in_path = os.path.join(INPUT_FOLDER, f)
    out_path = os.path.join(OUTPUT_FOLDER, f)
    
    # 2. Load Audio & Prep for Model
    # Demucs expects specific tensor shapes
    # Use soundfile directly to avoid torchaudio backend issues on Windows
    wav_np, sr = sf.read(in_path)
    wav = torch.from_numpy(wav_np).float()
    
    if wav.ndim == 1:
        wav = wav.unsqueeze(0) # (Time,) -> (1, Time)
    else:
        wav = wav.t() # (Time, Channels) -> (Channels, Time)
    
    # Resample if needed (Demucs usually wants 44.1k internally but handles it)
    # We will just feed it what we have, but it expects stereo/channels dim
    if wav.shape[0] == 1:
        wav = torch.cat([wav, wav]) # Duplicate mono to stereo

    # Add batch dimension: [1, 2, Samples]
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    wav = wav[None] 

    # 3. RUN INFERENCE (The Heavy Lifting)
    start_time = time.time()
    
    with torch.no_grad():
        # Demucs splits audio into: Drums, Bass, Other, Vocals
        # We assume "Snore" is closest to "Vocals" or the residual
        sources = apply_model(model, wav, shifts=0)[0]
    
    end_time = time.time()
    
    # 4. Extract the Result
    # Index 3 is usually 'vocals' (breathing/speech)
    # We might also want to check 'Other' depending on the sound. 
    # Let's try VOCALS first as it captures human sounds best.
    clean_audio = sources[3].cpu().numpy()
    
    # Convert back to Mono
    clean_audio = np.mean(clean_audio, axis=0) 

    processing_times.append(end_time - start_time)
    
    # 5. Save
    sf.write(out_path, clean_audio, 44100) # Demucs outputs 44.1k
    
    print(f"   Processed {f} in {end_time - start_time:.2f}s")

avg_time = np.mean(processing_times) * 1000
print("-" * 40)
print(f"✅ AI Denoising Complete.")
print(f"⚡ Average Time: {avg_time:.2f} ms per file (Slow!)")