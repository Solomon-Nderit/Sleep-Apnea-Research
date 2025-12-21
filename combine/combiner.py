import os
import random
import numpy as np
import soundfile as sf
import librosa

# --- CONFIGURATION (Must match your folder names) ---
CLEAN_DIR = "./clean_snores"
NOISE_DIR = "./noise_samples"
OUTPUT_DIR = "./experiment_data"
TARGET_SR = 16000 

def create_dataset():
    # 1. Setup Output Folders
    snr_levels = [10, 0, -5] # 10dB (Clear), 0dB (Equal), -5dB (Noisy)
    
    for snr in snr_levels:
        path = os.path.join(OUTPUT_DIR, f"SNR_{snr}dB")
        os.makedirs(path, exist_ok=True)

    # 2. Get File Lists
    clean_files = [f for f in os.listdir(CLEAN_DIR) if f.endswith('.wav')]
    noise_files = [f for f in os.listdir(NOISE_DIR) if f.endswith('.wav')]
    
    # We only use the first 50 snores for this experiment to save time
    clean_files = clean_files[:50] 
    
    print(f"Processing {len(clean_files)} snores at 3 different noise levels...")

    # 3. The Mixing Loop
    for clean_file in clean_files:
        # Load Clean Snore (1 second)
        clean_path = os.path.join(CLEAN_DIR, clean_file)
        clean_sig, _ = librosa.load(clean_path, sr=TARGET_SR)
        
        # Pick a Random Noise
        noise_file = random.choice(noise_files)
        noise_path = os.path.join(NOISE_DIR, noise_file)
        noise_sig, _ = librosa.load(noise_path, sr=TARGET_SR)

        # Crop Noise to match Snore Length
        if len(noise_sig) > len(clean_sig):
            noise_sig = noise_sig[:len(clean_sig)]
        else:
            # If noise is too short, loop it (rare for ESC-50)
            tile = int(np.ceil(len(clean_sig)/len(noise_sig)))
            noise_sig = np.tile(noise_sig, tile)[:len(clean_sig)]

        # Generate the 3 versions
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
            
            # Normalize to prevent distortion (clipping)
            if np.max(np.abs(mixed_sig)) > 1.0:
                mixed_sig /= np.max(np.abs(mixed_sig))
            # -- MATH END --

            # Save
            out_name = f"{clean_file[:-4]}_noise_{snr}dB.wav"
            out_folder = os.path.join(OUTPUT_DIR, f"SNR_{snr}dB")
            sf.write(os.path.join(out_folder, out_name), mixed_sig, TARGET_SR)

    print("Done! Check the 'experiment_data' folder.")

if __name__ == "__main__":
    create_dataset()