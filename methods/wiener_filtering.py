import os
import librosa
import soundfile as sf
from scipy.signal import wiener
import numpy as np
import argparse

TARGET_SR = 16000

def process_file(in_path, out_path):
    data, _ = librosa.load(in_path, sr=TARGET_SR)
    
    # Apply Wiener Filter
    # mysize: size of the window to estimate local variance
    cleaned = wiener(data, mysize=None, noise=None)
    
    sf.write(out_path, cleaned, TARGET_SR)

def main():
    parser = argparse.ArgumentParser(description="Wiener Filtering Cleaner")
    parser.add_argument("--input-dir", help="Input directory containing .wav files")
    parser.add_argument("--output-dir", help="Output directory for processed files")
    parser.add_argument("--input-file", help="Single input .wav file")
    parser.add_argument("--output-file", help="Single output .wav file")
    
    args = parser.parse_args()

    if args.input_dir and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        files = [f for f in os.listdir(args.input_dir) if f.endswith('.wav')]
        print(f"Processing {len(files)} files...")
        for f in files:
            in_path = os.path.join(args.input_dir, f)
            out_path = os.path.join(args.output_dir, f)
            process_file(in_path, out_path)
        print("✅ Wiener Filtering Complete.")
        
    elif args.input_file and args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        process_file(args.input_file, args.output_file)
        print(f"✅ Wiener Filtering Complete. File saved to: {args.output_file}")
        
    else:
        print("Error: Please provide either --input-dir and --output-dir OR --input-file and --output-file")

if __name__ == "__main__":
    main()
