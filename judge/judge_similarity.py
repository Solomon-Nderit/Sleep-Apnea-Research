import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import os
from scipy.spatial.distance import cosine

# --- CONFIGURATION ---
MODEL_URL = "https://tfhub.dev/google/yamnet/1"
TARGET_SR = 16000

print("Loading YAMNet...")
model = hub.load(MODEL_URL)

def get_embedding(file_path):
    # 1. Load and Normalize
    wav_data, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
    if np.max(np.abs(wav_data)) > 0:
        wav_data = wav_data / np.max(np.abs(wav_data))
    
    # 2. Get Embedding (The "Fingerprint")
    # YAMNet returns: [scores, embeddings, spectrogram]
    _, embeddings, _ = model(wav_data)
    
    # 3. Average the embeddings across the clip to get one vector
    # shape becomes (1024,)
    return np.mean(embeddings.numpy(), axis=0)

def compare_files(clean_path, noisy_path):
    # Get fingerprints
    emb_clean = get_embedding(clean_path)
    emb_noisy = get_embedding(noisy_path)
    
    # Calculate Similarity (1 - Cosine Distance)
    # Result: 0.0 to 1.0 (Higher is better)
    similarity = 1 - cosine(emb_clean, emb_noisy)
    return similarity

# --- THE EXPERIMENT LOOP ---
# We will iterate through all Denoised folders, SNRs, and Categories.
# For each folder, we process the files inside and generate a judge_stats.csv.

EXPERIMENT_DATA_DIR = "../experiment_data"
CLEAN_DIR = "../clean_snores"

DENOISED_FOLDERS = [
    "Denoised_Spectral",
    "Denoised_DeepLearning",
    "Denoised_Wiener",
    "Denoised_Wavelet",
    "Denoised_DFNet"
]

METHOD_MAP = {
    "Denoised_Spectral": "Spectral",
    "Denoised_DeepLearning": "Demucs",
    "Denoised_Wiener": "Wiener",
    "Denoised_Wavelet": "Wavelet",
    "Denoised_DFNet": "DeepFilterNet"
}

SNRS = ["-5dB", "0dB"]

for method_folder in DENOISED_FOLDERS:
    method_name = METHOD_MAP.get(method_folder, method_folder)

    for snr in SNRS:
        # Path to the method/snr root (e.g. ../experiment_data/Denoised_Spectral/-5dB)
        method_snr_path = os.path.join(EXPERIMENT_DATA_DIR, method_folder, snr)
        
        if not os.path.exists(method_snr_path):
            # print(f"Skipping {method_snr_path} (Not found)")
            continue
        
        # List categories (subfolders)
        try:
            categories = [d for d in os.listdir(method_snr_path) if os.path.isdir(os.path.join(method_snr_path, d))]
        except OSError:
            continue
            
        for category in categories:
            current_process_dir = os.path.join(method_snr_path, category)
            
            # Noisy Baselines
            noisy_hard_dir = os.path.join(EXPERIMENT_DATA_DIR, "SNR_-5dB", category)
            noisy_med_dir = os.path.join(EXPERIMENT_DATA_DIR, "SNR_0dB", category)
            
            files_to_check = sorted([f for f in os.listdir(current_process_dir) if f.endswith('.wav')])
            
            if not files_to_check:
                continue

            print(f"Processing: {current_process_dir}")
            
            # Prepare CSV Data
            csv_lines = []
            header = f"FILENAME,0dB,-5dB,{method_name}"
            csv_lines.append(header)

            for f in files_to_check:
                # Parse filename to find the original clean snore
                clean_filename = None
                if "_plus_" in f:
                    clean_filename = f.split("_plus_")[0] + ".wav"
                elif "_noise_" in f:
                    clean_filename = f.split("_noise_")[0] + ".wav"
                
                if not clean_filename or not os.path.exists(os.path.join(CLEAN_DIR, clean_filename)):
                    continue

                clean_p = os.path.join(CLEAN_DIR, clean_filename)
                
                # Construct paths for all versions
                # Handle SNR mapping for the Noisy Baselines
                
                f_0dB = f
                f_minus5dB = f
                
                if "-5dB" in f:
                    f_0dB = f.replace("-5dB", "0dB")
                elif "0dB" in f:
                    f_minus5dB = f.replace("0dB", "-5dB")
                
                # Noisy paths
                noisy_med_p = os.path.join(noisy_med_dir, f_0dB)
                noisy_hard_p = os.path.join(noisy_hard_dir, f_minus5dB)
                
                # Current Denoised File Path
                current_file_p = os.path.join(current_process_dir, f)

                # Helper to score if exists
                def get_score(path):
                    if os.path.exists(path):
                        return compare_files(clean_p, path)
                    return 0.0

                # Calculate scores
                s_med = get_score(noisy_med_p)
                s_hard = get_score(noisy_hard_p)
                s_current = get_score(current_file_p)

                # Add to CSV
                line = f"{f},{s_med:.4f},{s_hard:.4f},{s_current:.4f}"
                csv_lines.append(line)
            
            # Save CSV
            csv_path = os.path.join(current_process_dir, "judge_stats.csv")
            with open(csv_path, "w") as f_out:
                f_out.write("\n".join(csv_lines))
            print(f"  -> Saved {csv_path}")

print("-" * 100)
print("Processing Complete.")