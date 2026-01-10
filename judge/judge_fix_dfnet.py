import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import os
import csv
from scipy.spatial.distance import cosine

# --- CONFIGURATION ---
MODEL_URL = "https://tfhub.dev/google/yamnet/1"
TARGET_SR = 16000

# DIRECTORIES
EXPERIMENT_DATA_DIR = "../experiment_data"
CLEAN_DIR = "../clean_snores"
DFNET_ROOT = "../experiment_data/Denoised_DFNet" # Targeting ONLY this folder

print("⏳ Loading YAMNet (this takes a moment)...")
model = hub.load(MODEL_URL)

def get_embedding(file_path):
    try:
        wav_data, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
        if np.max(np.abs(wav_data)) > 0:
            wav_data = wav_data / np.max(np.abs(wav_data))
        _, embeddings, _ = model(wav_data)
        return np.mean(embeddings.numpy(), axis=0)
    except Exception as e:
        print(f"   ⚠️ Error reading {os.path.basename(file_path)}: {e}")
        return np.zeros(1024)

def compare_files(clean_path, noisy_path):
    if not os.path.exists(clean_path) or not os.path.exists(noisy_path):
        return 0.0
    emb_clean = get_embedding(clean_path)
    emb_noisy = get_embedding(noisy_path)
    # 1 - Cosine Distance = Similarity
    return 1 - cosine(emb_clean, emb_noisy)

# --- THE FIX LOOP ---

print(f"🚀 Starting Surgical Fix on: {DFNET_ROOT}")

# Walk through all subfolders in Denoised_DFNet (0dB/airplane, -5dB/baby, etc.)
for root, dirs, files in os.walk(DFNET_ROOT):
    wav_files = [f for f in files if f.endswith('.wav')]
    
    if not wav_files:
        continue

    # Determine which SNR folder we are in (-5dB or 0dB) based on the path
    # This helps us find the matching Noisy file
    current_snr = "-5dB" if "-5dB" in root else "0dB"
    category = os.path.basename(root) # e.g. "airplane"
    
    print(f"\n📂 Re-Judging: {category} ({current_snr})")
    
    csv_lines = []
    header = "FILENAME,0dB,-5dB,DeepFilterNet" # Standard Header
    csv_lines.append(header)
    
    for f in wav_files:
        # --- THE CRITICAL FIX ---
        # 1. Strip the suffix to find the "Real" filename
        # "1_181_..._-5dB_DeepFilterNet3.wav" -> "1_181_..._-5dB.wav"
        clean_name_candidate = f.replace("_DeepFilterNet3.wav", ".wav")
        
        # 2. Further strip to find the Original Clean Snore
        # "1_181_plus_..._-5dB.wav" -> "1_181.wav" (Using your logic)
        if "_plus_" in clean_name_candidate:
            original_clean_name = clean_name_candidate.split("_plus_")[0] + ".wav"
        elif "_noise_" in clean_name_candidate:
            original_clean_name = clean_name_candidate.split("_noise_")[0] + ".wav"
        else:
            original_clean_name = clean_name_candidate # Fallback

        clean_path = os.path.join(CLEAN_DIR, original_clean_name)
        
        # 3. Find the Noisy Baselines
        # We need to construct the path to the noisy files in SNR_-5dB and SNR_0dB
        # They will NOT have the _DeepFilterNet3 suffix.
        
        # Filename in the noisy folders is just "1_181_..._-5dB.wav"
        baseline_filename_neg5 = clean_name_candidate.replace("0dB", "-5dB")
        baseline_filename_0 = clean_name_candidate.replace("-5dB", "0dB")

        path_neg5 = os.path.join(EXPERIMENT_DATA_DIR, "SNR_-5dB", category, baseline_filename_neg5)
        path_0 = os.path.join(EXPERIMENT_DATA_DIR, "SNR_0dB", category, baseline_filename_0)
        path_denoised = os.path.join(root, f)

        # 4. Calculate Scores
        # We only re-calculate if we actually found the clean reference
        if os.path.exists(clean_path):
            score_0 = compare_files(clean_path, path_0)
            score_neg5 = compare_files(clean_path, path_neg5)
            score_denoised = compare_files(clean_path, path_denoised)
            
            # 5. Add to CSV
            csv_lines.append(f"{f},{score_0:.4f},{score_neg5:.4f},{score_denoised:.4f}")
            print(f"   ✅ {f} -> {score_denoised:.4f}")
        else:
            print(f"   ❌ Clean ref not found for: {f}")
            # Write zeros so we don't break the CSV structure
            csv_lines.append(f"{f},0.0000,0.0000,0.0000")

    # Overwrite the CSV
    csv_out_path = os.path.join(root, "judge_stats.csv")
    with open(csv_out_path, "w") as f_out:
        f_out.write("\n".join(csv_lines))
    print(f"💾 Saved corrected stats to: {csv_out_path}")

print("\n✨ All DeepFilterNet folders fixed.")