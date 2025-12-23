import os
import time
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
import soundfile as sf
import librosa
import numpy as np
import argparse
import csv

def process_file(model, in_path, out_path):
    # 2. Load Audio & Prep for Model
    # Demucs expects specific tensor shapes
    # Use soundfile directly to avoid torchaudio backend issues on Windows
    wav_np, sr = sf.read(in_path)
    wav = torch.from_numpy(wav_np).float()
    
    if wav.ndim == 1:
        wav = wav.unsqueeze(0) # (Time,) -> (1, Time)
    else:
        wav = wav.t() # (Time, Channels) -> (Channels, Time)
    
    # Resample if needed (Demucs usually wants 44.1k internally but handles it)
    # We will just feed it what we have, but it expects stereo/channels dim
    if wav.shape[0] == 1:
        wav = torch.cat([wav, wav]) # Duplicate mono to stereo

    # Add batch dimension: [1, 2, Samples]
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    wav = wav[None] 

    # 3. RUN INFERENCE (The Heavy Lifting)
    start_time = time.time()
    
    with torch.no_grad():
        # Demucs splits audio into: Drums, Bass, Other, Vocals
        # We assume "Snore" is closest to "Vocals" or the residual
        sources = apply_model(model, wav, shifts=0)[0]
    
    end_time = time.time()
    
    # 4. Extract the Result
    # Index 3 is usually 'vocals' (breathing/speech)
    # We might also want to check 'Other' depending on the sound. 
    # Let's try VOCALS first as it captures human sounds best.
    clean_audio = sources[3].cpu().numpy()
    
    # Convert back to Mono
    clean_audio = np.mean(clean_audio, axis=0) 
    
    # 5. Save
    sf.write(out_path, clean_audio, 44100) # Demucs outputs 44.1k
    
    print(f"   Processed {os.path.basename(in_path)} in {end_time - start_time:.2f}s")
    return (end_time - start_time) * 1000 # ms

def get_snr_level(filename):
    # Expected format: ..._10dB.wav, ..._0dB.wav, ..._-5dB.wav
    if "_10dB" in filename: return "10dB"
    if "_0dB" in filename: return "0dB"
    if "_-5dB" in filename: return "-5dB"
    return "Unknown"

def save_stats(stats, output_dir):
    csv_path = os.path.join(output_dir, "processing_stats.csv")
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "category", "snr_level", "method", "processing_time_ms"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(stats)
    print(f"📊 Stats saved to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Demucs Cleaner")
    parser.add_argument("--input-dir", help="Input directory containing .wav files")
    parser.add_argument("--output-dir", help="Output directory for processed files")
    parser.add_argument("--input-file", help="Single input .wav file")
    parser.add_argument("--output-file", help="Single output .wav file")
    
    args = parser.parse_args()

    # 1. Load the Pre-trained Model (Runs on CPU if no GPU)
    # "htdemucs" is the latest hybrid transformer model
    print("⏳ Loading Demucs Neural Net (this grabs ~100MB)...")
    model = get_model(name="htdemucs")
    model.cpu()
    model.eval()

    stats = []
    METHOD_NAME = "Demucs"

    if args.input_dir and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        files = [f for f in os.listdir(args.input_dir) if f.endswith('.wav')]
        print(f"🧠 Starting AI Cleanup on {len(files)} files...")
        
        category = os.path.basename(args.input_dir.rstrip(os.sep))
        
        for f in files:
            in_path = os.path.join(args.input_dir, f)
            out_path = os.path.join(args.output_dir, f)
            duration_ms = process_file(model, in_path, out_path)
            
            stats.append({
                "filename": f,
                "category": category,
                "snr_level": get_snr_level(f),
                "method": METHOD_NAME,
                "processing_time_ms": round(duration_ms, 2)
            })
            
        save_stats(stats, args.output_dir)
        print(f"✅ AI Denoising Complete.")

    elif args.input_file and args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        duration_ms = process_file(model, args.input_file, args.output_file)
        
        category = os.path.basename(os.path.dirname(args.input_file))
        filename = os.path.basename(args.input_file)
        
        stats.append({
            "filename": filename,
            "category": category,
            "snr_level": get_snr_level(filename),
            "method": METHOD_NAME,
            "processing_time_ms": round(duration_ms, 2)
        })
        
        save_stats(stats, os.path.dirname(args.output_file))
        print(f"✅ AI Denoising Complete.")
        
    else:
        print("Error: Please provide either --input-dir and --output-dir OR --input-file and --output-file")

if __name__ == "__main__":
    main()
