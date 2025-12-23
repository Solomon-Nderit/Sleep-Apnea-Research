import os
import subprocess
import sys

# --- CONFIGURATION ---
BASE_INPUT_DIR = "experiment_data"
SNR_LEVELS = ["SNR_0dB", "SNR_-5dB"] # User requested 0dB and -5dB specifically

# Map script filenames to their output folder names
METHODS = {
    "noise_reduce.py": "Denoised_Spectral",
    "deepfilternet_cleaner.py": "Denoised_DFNet",
    "demucs_cleaner.py": "Denoised_DeepLearning",
    "wavelet_filtering.py": "Denoised_Wavelet",
    "wiener_filtering.py": "Denoised_Wiener"
}

def main():
    # Verify base directory
    if not os.path.exists(BASE_INPUT_DIR):
        print(f"Error: Base directory '{BASE_INPUT_DIR}' not found.")
        return

    # Iterate through SNR levels
    for snr_folder in SNR_LEVELS:
        current_input_root = os.path.join(BASE_INPUT_DIR, snr_folder)
        
        if not os.path.exists(current_input_root):
            print(f"Warning: SNR folder '{current_input_root}' not found. Skipping.")
            continue

        # Get all categories in this SNR folder
        categories = [d for d in os.listdir(current_input_root) if os.path.isdir(os.path.join(current_input_root, d))]
        
        if not categories:
            print(f"Warning: No category folders found in '{current_input_root}'.")
            continue

        print(f"\n=== Processing SNR Level: {snr_folder} ({len(categories)} categories) ===")

        # Iterate through categories
        for category in categories:
            input_dir = os.path.join(current_input_root, category)
            print(f"\n  📂 Category: {category}")
            
            # Run each method
            for script_name, output_base_name in METHODS.items():
                # Construct Output Path: experiment_data/{Method}/{Level}/{Category}
                # Extract "0dB" from "SNR_0dB"
                level_name = snr_folder.replace("SNR_", "") 
                
                output_dir = os.path.join(BASE_INPUT_DIR, output_base_name, level_name, category)
                
                # Construct the command
                # We use sys.executable to ensure we use the same python interpreter
                script_path = os.path.join("methods", script_name)
                
                cmd = [
                    sys.executable, script_path,
                    "--input-dir", input_dir,
                    "--output-dir", output_dir
                ]
                
                print(f"    🚀 Running {script_name}...")
                # print(f"       Command: {' '.join(cmd)}")
                
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"    ❌ Error running {script_name}: {e}")
                except Exception as e:
                    print(f"    ❌ Unexpected error: {e}")

    print("\n✅ All experiments completed.")

if __name__ == "__main__":
    main()
