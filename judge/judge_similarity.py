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

clean_dir = "../clean_snores"
noisy_dir_hard = "../experiment_data/SNR_-5dB"
noisy_dir_med = "../experiment_data/SNR_0dB"
cleaned_dir = "../experiment_data/Denoised_Spectral"
cleaned_dir_demucs = "../experiment_data/Denoised_DeepLearning"
cleaned_dir_wiener = "../experiment_data/Denoised_Wiener"
cleaned_dir_wavelet = "../experiment_data/Denoised_Wavelet"


# Let's test the first 50 files
clean_files = sorted(os.listdir(clean_dir))[:50]


print(f"\n{'FILENAME':<20} | {'0dB SCORE':<10} | {'-5dB SCORE':<10} | {'NoiseReduce':<10} | Demucs | Wiener  | Wavelet ")


print("-" * 50)

for f in clean_files:
    # Construct paths
    # Note: Mixer script added suffix to filenames
    clean_p = os.path.join(clean_dir, f)
    
    # Find corresponding noisy files
    # The mixer saved them as: "1_noise_-5dB.wav" (example)
    # Adjust this line to match your exact file naming from mixer.py
    fname_base = f[:-4] # remove .wav
    noisy_med_p = os.path.join(noisy_dir_med, f"{fname_base}_noise_0dB.wav")
    noisy_hard_p = os.path.join(noisy_dir_hard, f"{fname_base}_noise_-5dB.wav")
    cleaned_p = os.path.join(cleaned_dir, f"{fname_base}_noise_-5dB.wav")
    cleaned_dl_p = os.path.join(cleaned_dir_demucs, f"{fname_base}_noise_-5dB.wav")
    cleaned_wiener_p = os.path.join(cleaned_dir_wiener, f"{fname_base}_noise_-5dB.wav")
    cleaned_wavelet_p = os.path.join(cleaned_dir_wavelet, f"{fname_base}_noise_-5dB.wav")


    if os.path.exists(noisy_med_p) and os.path.exists(noisy_hard_p) and os.path.exists(cleaned_p) and os.path.exists(cleaned_dl_p) and os.path.exists(cleaned_wiener_p) and os.path.exists(cleaned_wavelet_p):
        score_med = compare_files(clean_p, noisy_med_p)
        score_hard = compare_files(clean_p, noisy_hard_p)
        score_cleaned = compare_files(clean_p, cleaned_p)
        score_cleaned_dl = compare_files(clean_p, cleaned_dl_p)
        score_cleaned_wiener = compare_files(clean_p, cleaned_wiener_p)
        score_cleaned_wavelet = compare_files(clean_p, cleaned_wavelet_p)

        print(f"{f:<20} | {score_med:.4f}     | {score_hard:.4f}     | {score_cleaned:.4f} | {score_cleaned_dl:.4f} | {score_cleaned_wiener:.4f} | {score_cleaned_wavelet:.4f} ")
    else:
        print(f"{f:<20} | MISSING FILES")

print("-" * 50)
print("Interpretation:")
print("1.00 = Perfect Match (Clean)")
print("0.90+ = Excellent Quality")
print("<0.50 = Heavy Distortion/Noise")