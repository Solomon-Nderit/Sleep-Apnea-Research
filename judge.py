import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import csv
import io
import requests
import os

# --- CONFIGURATION ---
MODEL_URL = "https://tfhub.dev/google/yamnet/1"
MAP_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
TARGET_SR = 16000

# Function to download and parse the Class Map automatically
def load_class_names():
    print("Downloading YAMNet Class Map...")
    response = requests.get(MAP_URL)
    response.raise_for_status()
    
    class_names = []
    # Parse CSV from string
    csv_file = io.StringIO(response.text)
    reader = csv.DictReader(csv_file)
    for row in reader:
        class_names.append(row['display_name'])
    return class_names

# 1. Load the Model and Map
print("Loading YAMNet Model (this takes a minute the first time)...")
model = hub.load(MODEL_URL)
class_names = load_class_names()

# Find the exact index for "Snoring"
try:
    SNORE_INDEX = class_names.index("Snoring")
    print(f"✅ Judge Ready! Tracking class: 'Snoring' (Index {SNORE_INDEX})")
except ValueError:
    print("❌ Error: Could not find 'Snoring' in class map.")
    exit()

def evaluate_audio(file_path):
    # Load audio
    wav_data, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
    
    # Normalize to -1.0 to 1.0 (YAMNet expects this)
    if np.max(np.abs(wav_data)) > 0:
        wav_data = wav_data / np.max(np.abs(wav_data))

    # Run Inference
    # YAMNet returns [scores, embeddings, spectrogram]
    scores, embeddings, spectrogram = model(wav_data)
    
    # Scores is a list of predictions for every 0.48s frame.
    # We take the MEAN (average) score across the whole clip.
    mean_scores = np.mean(scores, axis=0)
    
    # Get the confidence for Snoring
    snore_confidence = mean_scores[SNORE_INDEX]
    return snore_confidence

# --- TEST RUN ---
# Let's test it on one file from your "Hard" (-5dB) dataset
# Make sure you have run mixer.py first!

# noisy_folder = "./experiment_data/SNR_10dB"
noisy_folder = 'clean_snores'  

if not os.path.exists(noisy_folder):
    print("❌ Folder not found. Run mixer.py first!")
else:
    print("\n--- JUDGING START ---")
    files = [f for f in os.listdir(noisy_folder) if f.endswith('.wav')]
    
    if len(files) > 0:
        test_file = files[0] # Pick the first file
        file_path = os.path.join(noisy_folder, test_file)
        
        print(f"Judging File: {test_file}")
        score = evaluate_audio(file_path)
        
        print(f"🔊 Snoring Confidence: {score:.4f} (0.0 to 1.0)")
        
        if score < 0.2:
            print("Verdict: The Judge can barely hear the snore.")
        else:
            print("Verdict: The Judge hears it clearly!")
    else:
        print("No files found in the noise folder.")