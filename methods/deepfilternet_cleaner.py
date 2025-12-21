import os
import subprocess
import argparse

def run_deepfilternet_file(input_path, output_path):
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing: {input_path} -> {output_dir}")
    # DeepFilterNet usually takes an output directory with -o
    subprocess.run(["deepFilter", input_path, "-o", output_dir])

def main():
    parser = argparse.ArgumentParser(description="DeepFilterNet Cleaner Wrapper")
    parser.add_argument("--input-dir", help="Input directory containing .wav files")
    parser.add_argument("--output-dir", help="Output directory for processed files")
    parser.add_argument("--input-file", help="Single input .wav file")
    parser.add_argument("--output-file", help="Single output .wav file")
    
    args = parser.parse_args()

    if args.input_dir and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        files = [f for f in os.listdir(args.input_dir) if f.endswith('.wav')]
        print(f"🧹 Starting DeepFilterNet Cleanup on {len(files)} files...")
        for f in files:
            input_path = os.path.join(args.input_dir, f)
            output_path = os.path.join(args.output_dir, f)
            run_deepfilternet_file(input_path, output_path)
        print(f"✅ DeepFilterNet Denoising Complete. Files saved to: {args.output_dir}")
        
    elif args.input_file and args.output_file:
        run_deepfilternet_file(args.input_file, args.output_file)
        print(f"✅ DeepFilterNet Denoising Complete. File saved to: {args.output_file}")
        
    else:
        print("Error: Please provide either --input-dir and --output-dir OR --input-file and --output-file")

if __name__ == "__main__":
    main()

