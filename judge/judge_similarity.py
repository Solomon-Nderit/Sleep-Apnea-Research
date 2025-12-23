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
# We will compare a Clean file vs. its Noisy version
# to establish the "Baseline Score" before you try to clean it.

# Allow category selection
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--category", default="airplane", help="Category to judge (folder name)")
args = parser.parse_args()
CATEGORY = args.category

clean_dir = "../clean_snores"
noisy_dir_hard = f"../experiment_data/SNR_-5dB/{CATEGORY}"
noisy_dir_med = f"../experiment_data/SNR_0dB/{CATEGORY}"

# Denoised folders
cleaned_dir_spectral = f"../experiment_data/Denoised_Spectral/{CATEGORY}"
cleaned_dir_demucs = f"../experiment_data/Denoised_DeepLearning/{CATEGORY}"
cleaned_dir_wiener = f"../experiment_data/Denoised_Wiener/{CATEGORY}"
cleaned_dir_wavelet = f"../experiment_data/Denoised_Wavelet/{CATEGORY}"
cleaned_dir_dfnet = f"../experiment_data/Denoised_DFNet/{CATEGORY}"

# We iterate through the Demucs folder (or you can change this to noisy_dir_hard)
# to find files that have been processed.
reference_dir = cleaned_dir_demucs
if not os.path.exists(reference_dir):
    print(f"Warning: Reference directory {reference_dir} does not exist. Falling back to Noisy directory.")
    reference_dir = noisy_dir_hard

if not os.path.exists(reference_dir):
    print(f"Error: No directory found to list files from. Checked {cleaned_dir_demucs} and {noisy_dir_hard}")
    exit()

files_to_check = sorted([f for f in os.listdir(reference_dir) if f.endswith('.wav')])

print(f"\nJudging Category: {CATEGORY}")
print(f"Reference Directory: {reference_dir}")
print(f"{'FILENAME':<30} | {'0dB':<7} | {'-5dB':<7} | {'Spec':<7} | {'Demucs':<7} | {'Wiener':<7} | {'Wave':<7} | {'DFNet':<7}")
print("-" * 100)

for f in files_to_check:
    # Parse filename to find the original clean snore
    # Expected formats: 
    # 1. "1_0_plus_1-100032-A-0_-5dB.wav" (New)
    # 2. "1_0_noise_-5dB.wav" (Old)
    
    clean_filename = None
    if "_plus_" in f:
        clean_filename = f.split("_plus_")[0] + ".wav"
    elif "_noise_" in f:
        clean_filename = f.split("_noise_")[0] + ".wav"
    
    if not clean_filename:
        # Fallback: try to guess if it starts with "1_0" style
        # This is risky, but let's try splitting by first underscore if it looks like a snore ID
        pass

    if not clean_filename or not os.path.exists(os.path.join(clean_dir, clean_filename)):
        # print(f"Skipping {f}: Could not find clean source {clean_filename}")
        continue

    clean_p = os.path.join(clean_dir, clean_filename)
    
    # Construct paths for all versions
    # The file 'f' usually contains the SNR suffix (e.g. -5dB)
    # We need to construct the 0dB filename for the "Medium" noise column
    
    if "-5dB" in f:
        f_0dB = f.replace("-5dB", "0dB")
    else:
        # If we are judging 0dB files, then f is f_0dB
        f_0dB = f
        # And we might need to guess the -5dB name? 
        # Let's assume we are iterating -5dB files mostly.
    
    noisy_med_p = os.path.join(noisy_dir_med, f_0dB)
    noisy_hard_p = os.path.join(noisy_dir_hard, f) # Assuming f is the -5dB version
    
    # Denoised files usually match the input filename (f)
    cleaned_spectral_p = os.path.join(cleaned_dir_spectral, f)
    cleaned_dl_p = os.path.join(cleaned_dir_demucs, f)
    cleaned_wiener_p = os.path.join(cleaned_dir_wiener, f)
    cleaned_wavelet_p = os.path.join(cleaned_dir_wavelet, f)
    
    # DeepFilterNet adds a suffix like "_DeepFilterNet3"
    # We check for the standard name first, then try the suffix
    cleaned_dfnet_p = os.path.join(cleaned_dir_dfnet, f)
    if not os.path.exists(cleaned_dfnet_p):
        # Try with suffix
        f_dfnet = f.replace(".wav", "_DeepFilterNet3.wav")
        cleaned_dfnet_p = os.path.join(cleaned_dir_dfnet, f_dfnet)

    # Helper to score if exists
    def get_score(path):
        if os.path.exists(path):
            return compare_files(clean_p, path)
        return 0.0

    # We only print if we have at least the noisy file or one cleaned file
    if os.path.exists(noisy_hard_p) or os.path.exists(cleaned_dl_p):
        s_med = get_score(noisy_med_p)
        s_hard = get_score(noisy_hard_p)
        s_spec = get_score(cleaned_spectral_p)
        s_dl = get_score(cleaned_dl_p)
        s_wien = get_score(cleaned_wiener_p)
        s_wave = get_score(cleaned_wavelet_p)
        s_dfnet = get_score(cleaned_dfnet_p)

        print(f"{f[:28]:<30} | {s_med:.3f}   | {s_hard:.3f}   | {s_spec:.3f}   | {s_dl:.3f}   | {s_wien:.3f}   | {s_wave:.3f}   | {s_dfnet:.3f}")

print("-" * 100)
print("Interpretation:")
print("1.00 = Perfect Match (Clean)")
print("0.90+ = Excellent Quality")
print("<0.50 = Heavy Distortion/Noise")