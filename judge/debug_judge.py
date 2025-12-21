import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import csv
import io
import requests
import os

# --- SETUP ---
MODEL_URL = "https://tfhub.dev/google/yamnet/1"
MAP_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
TARGET_SR = 16000

# Load Model & Map
print("Loading Model...")
model = hub.load(MODEL_URL)

# Get Class Names
print("Getting Class Map...")
response = requests.get(MAP_URL)
class_names = []
reader = csv.DictReader(io.StringIO(response.text))
for row in reader:
    class_names.append(row['display_name'])

def diagnose_file(file_path):
    print(f"\n🎧 LISTENING TO: {os.path.basename(file_path)}")
    
    # 1. Load Audio
    wav_data, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
    
    # 2. Check Volume (If this is 0, the file is empty)
    max_vol = np.max(np.abs(wav_data))
    print(f"   - Max Volume: {max_vol:.4f}")
    
    if max_vol < 0.01:
        print("   ⚠️ WARNING: Audio is near silent!")
    
    # 3. Normalize
    if max_vol > 0:
        wav_data = wav_data / max_vol

    # 4. Run Inference
    scores, embeddings, spectrogram = model(wav_data)
    
    # 5. Get Top 5 Predictions
    mean_scores = np.mean(scores, axis=0)
    top_indices = np.argsort(mean_scores)[::-1][:5] # Sort descending
    
    print("   🤖 MODEL PREDICTIONS:")
    for i in top_indices:
        print(f"      {class_names[i]}: {mean_scores[i]:.4f}")

# --- RUN DIAGNOSTIC ---
# Point this to one of your CLEAN snores (inside clean_snores folder)
# Change '1_1.wav' to whatever actual filename you have
clean_folder = "./clean_snores"
files = os.listdir(clean_folder)

if files:
    test_file = os.path.join(clean_folder, files[7]) # Test the first file found
    diagnose_file(test_file)
else:
    print("No clean files found.")