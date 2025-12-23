import os
import librosa
import soundfile as sf
from scipy.signal import wiener
import numpy as np
import argparse
import time
import csv

TARGET_SR = 16000

def process_file(in_path, out_path):
    start_time = time.time()
    data, _ = librosa.load(in_path, sr=TARGET_SR)
    
    # Apply Wiener Filter
    # mysize: size of the window to estimate local variance
    cleaned = wiener(data, mysize=None, noise=None)
    
    sf.write(out_path, cleaned, TARGET_SR)
    end_time = time.time()
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
    parser = argparse.ArgumentParser(description="Wiener Filtering Cleaner")
    parser.add_argument("--input-dir", help="Input directory containing .wav files")
    parser.add_argument("--output-dir", help="Output directory for processed files")
    parser.add_argument("--input-file", help="Single input .wav file")
    parser.add_argument("--output-file", help="Single output .wav file")
    
    args = parser.parse_args()
    
    stats = []
    METHOD_NAME = "Wiener"

    if args.input_dir and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        files = [f for f in os.listdir(args.input_dir) if f.endswith('.wav')]
        print(f"Processing {len(files)} files...")
        
        category = os.path.basename(args.input_dir.rstrip(os.sep))
        
        for f in files:
            in_path = os.path.join(args.input_dir, f)
            out_path = os.path.join(args.output_dir, f)
            duration_ms = process_file(in_path, out_path)
            
            stats.append({
                "filename": f,
                "category": category,
                "snr_level": get_snr_level(f),
                "method": METHOD_NAME,
                "processing_time_ms": round(duration_ms, 2)
            })
            
        save_stats(stats, args.output_dir)
        print("✅ Wiener Filtering Complete.")
        
    elif args.input_file and args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        duration_ms = process_file(args.input_file, args.output_file)
        
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
        print(f"✅ Wiener Filtering Complete. File saved to: {args.output_file}")
        
    else:
        print("Error: Please provide either --input-dir and --output-dir OR --input-file and --output-file")

if __name__ == "__main__":
    main()
