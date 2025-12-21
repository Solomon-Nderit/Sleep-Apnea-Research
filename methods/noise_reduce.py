import os
import time
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
import argparse

TARGET_SR = 16000

def process_file(in_path, out_path):
    # 1. Load Noisy Audio
    data, sr = librosa.load(in_path, sr=TARGET_SR)
    
    # 2. THE CLEANING PROCESS
    # We measure how long it takes (Critical for your research paper!)
    start_time = time.time()
    
    # prop_decrease=1.0 means "remove 100% of the noise you find"
    # stationary=True means "assume the noise is a constant hum (like a fan)"
    reduced_noise = nr.reduce_noise(y=data, sr=sr, stationary=True, prop_decrease=0.9)
    
    end_time = time.time()
    
    duration = end_time - start_time
    
    # 3. Save
    sf.write(out_path, reduced_noise, sr)
    return duration

def main():
    parser = argparse.ArgumentParser(description="Noise Reduce (Spectral Gating) Cleaner")
    parser.add_argument("--input-dir", help="Input directory containing .wav files")
    parser.add_argument("--output-dir", help="Output directory for processed files")
    parser.add_argument("--input-file", help="Single input .wav file")
    parser.add_argument("--output-file", help="Single output .wav file")
    
    args = parser.parse_args()
    
    processing_times = []

    if args.input_dir and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        files = [f for f in os.listdir(args.input_dir) if f.endswith('.wav')]
        print(f"🧹 Starting Cleanup on {len(files)} files...")
        print(f"   Method: Spectral Gating (Stationary Noise Assumption)")
        
        for i, f in enumerate(files):
            in_path = os.path.join(args.input_dir, f)
            out_path = os.path.join(args.output_dir, f)
            duration = process_file(in_path, out_path)
            processing_times.append(duration)
            
            if i % 10 == 0:
                print(f"   Processed {i}/{len(files)} files...")
                
        if processing_times:
            avg_time = np.mean(processing_times) * 1000 # Convert to ms
            print("-" * 40)
            print(f"✅ Denoising Complete.")
            print(f"⚡ Average Processing Time: {avg_time:.2f} ms per file")
            print(f"📂 Saved to: {args.output_dir}")

    elif args.input_file and args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        process_file(args.input_file, args.output_file)
        print(f"✅ Denoising Complete. File saved to: {args.output_file}")
        
    else:
        print("Error: Please provide either --input-dir and --output-dir OR --input-file and --output-file")

if __name__ == "__main__":
    main()
