import os
import time
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np

# --- CONFIGURATION ---
INPUT_FOLDER = "./experiment_data/SNR_-5dB"  # We are tackling the HARDEST level
OUTPUT_FOLDER = "./experiment_data/Denoised_Spectral"
TARGET_SR = 16000

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.wav')]
files = files[:50] # Process 50 files

print(f"🧹 Starting Cleanup on {len(files)} files...")
print(f"   Method: Spectral Gating (Stationary Noise Assumption)")

processing_times = []

for i, f in enumerate(files):
    in_path = os.path.join(INPUT_FOLDER, f)
    out_path = os.path.join(OUTPUT_FOLDER, f)
    
    # 1. Load Noisy Audio
    data, sr = librosa.load(in_path, sr=TARGET_SR)
    
    # 2. THE CLEANING PROCESS
    # We measure how long it takes (Critical for your research paper!)
    start_time = time.time()
    
    # prop_decrease=1.0 means "remove 100% of the noise you find"
    # stationary=True means "assume the noise is a constant hum (like a fan)"
    reduced_noise = nr.reduce_noise(y=data, sr=sr, stationary=True, prop_decrease=0.9)
    
    end_time = time.time()
    
    duration = end_time - start_time
    processing_times.append(duration)
    
    # 3. Save
    sf.write(out_path, reduced_noise, sr)
    
    if i % 10 == 0:
        print(f"   Processed {i}/{len(files)} files...")

avg_time = np.mean(processing_times) * 1000 # Convert to ms
print("-" * 40)
print(f"✅ Denoising Complete.")
print(f"⚡ Average Processing Time: {avg_time:.2f} ms per second of audio")
print(f"📂 Saved to: {OUTPUT_FOLDER}")