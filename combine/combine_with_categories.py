import os
import random
import numpy as np
import soundfile as sf
import librosa
import csv
from collections import defaultdict

# --- CONFIGURATION (Must match your folder names) ---
CLEAN_DIR = "./clean_snores"
NOISE_DIR = "./noise_samples"
OUTPUT_DIR = "./experiment_data"
CSV_PATH = "./ESC-50-master/meta/esc50.csv"
TARGET_SR = 16000 
FILES_PER_CATEGORY = 10

def load_categories(csv_path):
    """
    Returns a dictionary: { 'category_name': ['file1.wav', 'file2.wav', ...] }
    """
    category_map = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            category_map[row['category']].append(row['filename'])
    return category_map

def create_dataset():
    # 1. Load Categories
    print(f"Loading categories from {CSV_PATH}...")
    category_map = load_categories(CSV_PATH)
    
    # 2. Get Clean Snore List
    clean_files = [f for f in os.listdir(CLEAN_DIR) if f.endswith('.wav')]
    print(f"Found {len(clean_files)} clean snore files.")

    if len(clean_files) < FILES_PER_CATEGORY:
        print(f"Error: Not enough clean snores. Need at least {FILES_PER_CATEGORY}.")
        return

    snr_levels = [10, 0, -5] # 10dB (Clear), 0dB (Equal), -5dB (Noisy)
    
    print(f"Processing {len(category_map)} categories. Generating {FILES_PER_CATEGORY} files per category per SNR level.")

    # 3. Iterate by Category
    for category, noise_files in category_map.items():
        # Filter noise files to ensure they exist in our noise directory
        available_noises = [f for f in noise_files if os.path.exists(os.path.join(NOISE_DIR, f))]
        
        if len(available_noises) < FILES_PER_CATEGORY:
            print(f"Warning: Category '{category}' has only {len(available_noises)} available noise files. Skipping or using fewer.")
            if len(available_noises) == 0:
                continue
            # If fewer than 10, just use what we have
            current_batch_size = len(available_noises)
        else:
            current_batch_size = FILES_PER_CATEGORY

        # Select Noises (Fixed for this category across all SNRs)
        # We take the first N or random N. Let's take random N for variety.
        selected_noises = random.sample(available_noises, current_batch_size)
        
        # Select Clean Snores (Randomly for this category)
        # We sample with replacement if needed, but sample without replacement is better if we have enough.
        # Since we have ~500 snores and need 10, we can sample without replacement easily.
        selected_snores = random.sample(clean_files, current_batch_size)
        
        # Pair them up
        pairs = list(zip(selected_snores, selected_noises))
        
        print(f"  Category: {category} -> {len(pairs)} pairs")

        # Generate mixes for this category
        for clean_file, noise_file in pairs:
            # Load Audio (Load once per pair to be efficient)
            clean_path = os.path.join(CLEAN_DIR, clean_file)
            noise_path = os.path.join(NOISE_DIR, noise_file)
            
            try:
                clean_sig, _ = librosa.load(clean_path, sr=TARGET_SR)
                noise_sig, _ = librosa.load(noise_path, sr=TARGET_SR)
            except Exception as e:
                print(f"    Error loading files {clean_file} or {noise_file}: {e}")
                continue

            # Crop Noise to match Snore Length
            if len(noise_sig) > len(clean_sig):
                noise_sig = noise_sig[:len(clean_sig)]
            else:
                # If noise is too short, loop it
                tile = int(np.ceil(len(clean_sig)/len(noise_sig)))
                noise_sig = np.tile(noise_sig, tile)[:len(clean_sig)]

            # Generate the 3 versions for this SPECIFIC pair
            for snr in snr_levels:
                # -- MATH START --
                clean_pwr = np.mean(clean_sig ** 2)
                noise_pwr = np.mean(noise_sig ** 2)
                
                if noise_pwr > 0:
                    target_noise_pwr = clean_pwr / (10 ** (snr / 10))
                    scale = np.sqrt(target_noise_pwr / noise_pwr)
                    mixed_sig = clean_sig + (noise_sig * scale)
                else:
                    mixed_sig = clean_sig
                
                # Normalize
                if np.max(np.abs(mixed_sig)) > 1.0:
                    mixed_sig /= np.max(np.abs(mixed_sig))
                # -- MATH END --

                # Save
                # Structure: experiment_data/SNR_{snr}dB/{category}/
                out_folder = os.path.join(OUTPUT_DIR, f"SNR_{snr}dB", category)
                os.makedirs(out_folder, exist_ok=True)
                
                # Naming: cleanName_noiseName_SNR.wav to be descriptive
                # clean_file is like "1_0.wav", noise_file is like "1-100032-A-0.wav"
                out_name = f"{clean_file[:-4]}_plus_{noise_file[:-4]}_{snr}dB.wav"
                sf.write(os.path.join(out_folder, out_name), mixed_sig, TARGET_SR)

    print("Done! Check the 'experiment_data' folder.")

if __name__ == "__main__":
    create_dataset()
