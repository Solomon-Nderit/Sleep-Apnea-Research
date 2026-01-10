import os
import time
import csv
import sys
import types
import torch
import librosa
import soundfile as sf
import numpy as np

# --- 1. COMPATIBILITY PATCH ---
try:
    import torchaudio
    if not hasattr(torchaudio, 'backend'):
        dummy_backend = types.ModuleType("torchaudio.backend")
        dummy_common = types.ModuleType("torchaudio.backend.common")
        if hasattr(torchaudio, 'AudioMetaData'):
            dummy_common.AudioMetaData = torchaudio.AudioMetaData
        else:
            from dataclasses import dataclass
            @dataclass
            class AudioMetaData:
                sample_rate: int
                num_frames: int
                num_channels: int
                bits_per_sample: int
                encoding: str
            dummy_common.AudioMetaData = AudioMetaData
        sys.modules["torchaudio.backend"] = dummy_backend
        sys.modules["torchaudio.backend.common"] = dummy_common
except ImportError:
    pass

from df.enhance import enhance, init_df

# --- CONFIGURATION ---
BASE_INPUT_DIR = "experiment_data"
BASE_OUTPUT_DIR = "experiment_data/Denoised_DFNet"
SNR_FOLDERS = ["SNR_0dB", "SNR_-5dB"]

def get_snr_level_from_filename(filename):
    if "_10dB" in filename: return "10dB"
    if "_0dB" in filename: return "0dB"
    if "_-5dB" in filename: return "-5dB"
    return "Unknown"

def save_category_stats(stats, output_dir):
    csv_path = os.path.join(output_dir, "processing_stats.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "category", "snr_level", "method", "processing_time_ms"])
        writer.writeheader()
        writer.writerows(stats)
    print(f"   📊 Stats log saved to: {csv_path}")

def main():
    print("🚀 Starting DeepFilterNet Standalone Runner (Librosa Bypass Mode)")
    
    # 2. LOAD MODEL
    print("⏳ Loading Model into Memory...")
    model, df_state, _ = init_df()
    target_sr = df_state.sr() # Usually 48000
    print(f"✅ Model Loaded. Target Sample Rate: {target_sr}Hz")

    for snr_folder in SNR_FOLDERS:
        input_snr_path = os.path.join(BASE_INPUT_DIR, snr_folder)
        if not os.path.exists(input_snr_path):
            continue
            
        categories = [d for d in os.listdir(input_snr_path) if os.path.isdir(os.path.join(input_snr_path, d))]
        
        print(f"\n=== Processing Level: {snr_folder} ===")
        
        for category in categories:
            input_cat_path = os.path.join(input_snr_path, category)
            clean_level_name = snr_folder.replace("SNR_", "")
            output_cat_path = os.path.join(BASE_OUTPUT_DIR, clean_level_name, category)
            
            os.makedirs(output_cat_path, exist_ok=True)
            
            files = [f for f in os.listdir(input_cat_path) if f.endswith('.wav')]
            if not files:
                continue

            print(f"📂 Category: {category} ({len(files)} files)")
            stats = []
            
            for f in files:
                in_file = os.path.join(input_cat_path, f)
                out_file = os.path.join(output_cat_path, f) # Variable defined as out_file
                
                try:
                    # 1. Load Audio
                    audio_np, _ = librosa.load(in_file, sr=target_sr, mono=True)
                    
                    # 2. Convert to Tensor
                    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
                    
                    # --- TIMER START ---
                    start_time = time.time()
                    
                    # 3. Run Inference
                    enhanced_audio = enhance(model, df_state, audio_tensor)
                    
                    end_time = time.time()
                    # --- TIMER END ---
                    
                    duration_ms = (end_time - start_time) * 1000
                    
                    # 4. Save
                    enhanced_np = enhanced_audio.cpu().detach().numpy().flatten()
                    sf.write(out_file, enhanced_np, target_sr) # FIX: Use out_file here
                    
                    stats.append({
                        "filename": f,
                        "category": category,
                        "snr_level": get_snr_level_from_filename(f),
                        "method": "DeepFilterNet",
                        "processing_time_ms": round(duration_ms, 2)
                    })
                    
                    if len(stats) <= 2: 
                        print(f"   -> {f} processed in {duration_ms:.2f}ms")
                        
                except Exception as e:
                    print(f"   ❌ Error on {f}: {e}")

            save_category_stats(stats, output_cat_path)

    print("\n✅ All DeepFilterNet tasks finished.")

if __name__ == "__main__":
    main()