# Sleep Apnea Denoising Research Codebase Report

This report documents the structure and functionality of the codebase used for sleep apnea/snore denoising research. Combining clean snore samples with environmental noises at various SNR levels, the project evaluates multiple denoising algorithms using deep learning-based metrics.

## 1. Project Overview
The goal of this research is to evaluate the effectiveness of different audio denoising techniques (both traditional DSP and modern Deep Learning) in recovering "Snore" sounds from noisy environments. The pipeline consists of:
1.  **Data Generation**: Mixing clean snores with categorized noise (ESC-50 dataset).
2.  **Denoising**: Applying 5 different cleaning methods to the noisy data.
3.  **Evaluation**: Using YAMNet (a pre-trained audio classifier) to measure:
    *   **Class Recovery**: Can the model recognize the cleaned audio as a "Snore"?
    *   **Similarity**: How close is the cleaned audio fingerprint to the original clean snore?

## 2. Project Structure
*(Repetitive .wav files omitted for brevity)*

```text
root/
├── Analysis.ipynb                 # Jupyter Notebook for visualizing results and stats
├── run_all_experiments.py         # Main orchestrator script to run all denoising methods
├── yamnet_class_map.csv           # Class definitions for the YAMNet model
├── guide.md                       # Documentation on SNR levels
│
├── clean_snores/                  # Source folder for clean/ground-truth snore audio
│
├── combine/                       # Scripts for generating the synthetic dataset
│   ├── combine_with_categories.py # Advanced mixer: Uses ESC-50 categories (dog, rain, etc.)
│   └── combiner.py                # Simple mixer: Random noise mixing
│
├── ESC-50-master/                 # Metadata/Tools for the Environmental Sound Classification dataset
│   └── meta/
│       └── esc50.csv              # Maps filenames to categories (e.g., 1-100032-A-0.wav -> "dog")
│
├── experiment_data/               # The main data storage (Generated & Processed)
│   ├── SNR_10dB/                  # Noisy Data (Input) - High SNR (Clearer)
│   ├── SNR_0dB/                   # Noisy Data (Input) - Mid SNR
│   ├── SNR_-5dB/                  # Noisy Data (Input) - Low SNR (Noisier)
│   │   ├── airplane/              # Organized by noise category
│   │   └── ...
│   │
│   ├── Denoised_DeepLearning/     # Results from Demucs
│   ├── Denoised_DFNet/            # Results from DeepFilterNet
│   ├── Denoised_Spectral/         # Results from Spectral Gating
│   ├── Denoised_Wavelet/          # Results from Wavelet Denoising
│   └── Denoised_Wiener/           # Results from Wiener Filtering
│
├── judge/                         # Evaluation Scripts
│   ├── judge.py                   # Calculates "Snore" class probability (YAMNet)
│   ├── judge_similarity.py        # Calculates Cosine Similarity (Embeddings)
│   ├── judge_fix_dfnet.py         # Utility likely for fixing specific DFNet output structural issues
│   └── debug_judge.py             # Debugging tool for the judge
│
├── methods/                       # The Denoising Algorithms (Cleaners)
│   ├── deepfilternet_cleaner.py   # DeepFilterNet wrapper
│   ├── demucs_cleaner.py          # Demucs wrapper
│   ├── noise_reduce.py            # Spectral Gating (noisereduce lib)
│   ├── wavelet_filtering.py       # Wavelet Thresholding
│   ├── wiener_filtering.py        # Wiener Filter
│   └── run_dfnet_standalone.py    # Standalone runner for DFNet
│
├── noise_samples/                 # Source folder for noise audio files (from ESC-50)
├── results/                       # Likely destination for final CSV reports/metric outputs
└── useful scripts/                # Helper utilities
    └── upsampler.py               # (Currently empty/placeholder)
```

## 3. Detailed Component Breakdown

### A. Data Generation (`combine/`)
The foundation of the experiment is the creation of a **synthetic dataset**.
*   **`combine_with_categories.py`**: This script reads the `ESC-50` metadata. It takes clean snores from `clean_snores/` and mixes them with noises from `noise_samples/` based on their category (e.g., "dog", "rain").
*   It generates three versions of every file corresponding to SNR levels:
    *   **10dB**: Snore is significantly louder than noise.
    *   **0dB**: Snore and noise are equal volume.
    *   **-5dB**: Noise is louder than snore (Critical test case).
*   **Output**: Files are saved to `experiment_data/SNR_XdB/{category}/`.

### B. Denoising Methods (`methods/`)
The `run_all_experiments.py` script orchestrates the execution of these methods. It iterates through the SNR folders and applies the following 5 techniques:

1.  **`noise_reduce.py` (Spectral Gating)**:
    *   **Method**: Uses the `noisereduce` library. Performs spectral gating assuming stationary noise (`stationary=True`).
    *   **Output Folder**: `Denoised_Spectral`
    *   **Pros**: Fast, good for constant hums (fans). **Cons**: Can introduce "musical noise" artifacts.

2.  **`deepfilternet_cleaner.py` (DeepFilterNet)**:
    *   **Method**: A wrapper around the `DeepFilterNet` command-line tool (`deepFilter`). This is a Low-latency Deep Learning model specifically designed for speech enhancement.
    *   **Output Folder**: `Denoised_DFNet`
    *   **Note**: Requires the CLI tool to be installed in the environment.

3.  **`demucs_cleaner.py` (Demucs)**:
    *   **Method**: Uses Facebook's **Demucs** source separation model. It attempts to separate the "Vocals" track (mapped to human snoring here) from the rest of the audio.
    *   **Output Folder**: `Denoised_DeepLearning`
    *   **Implementation**: Loads audio into PyTorch tensors and runs inference.

4.  **`wavelet_filtering.py` (Wavelet)**:
    *   **Method**: Uses Discrete Wavelet Transform (DWT).
    *   **Algorithm**: Decomposes signal using `db4` wavelet -> Estimates noise sigma -> Applies Soft Thresholding -> Reconstructs signal.
    *   **Output Folder**: `Denoised_Wavelet`
    *   **Type**: Traditional DSP.

5.  **`wiener_filtering.py` (Wiener)**:
    *   **Method**: Uses `scipy.signal.wiener`. This is a statistical approach that minimizes the mean square error between the estimated random process and the desired process.
    *   **Output Folder**: `Denoised_Wiener`
    *   **Type**: Traditional DSP.

### C. Evaluation (`judge/`)
Subjective listening is too slow, so "AI Judges" are used.

*   **`judge.py` (Classification Metric)**:
    *   Uses **YAMNet** (via TensorFlow Hub), an Audio Event Classifier trained on AudioSet.
    *   It feeds the *denoised* audio into YAMNet and checks the confidence score for the class **"Snoring"** (Index 0 or specific index in map).
    *   **Hypothesis**: A better denoising job results in the AI being more confident that the sound is a snore.

*   **`judge_similarity.py` (Similarity Metric)**:
    *   Extracts the **Embeddings** (fingerprints) from the penultimate layer of YAMNet.
    *   Calculates the **Cosine Similarity** between the *Denoised Snore embedding* and the *Original Clean Snore embedding*.
    *   **Score**: 0.0 to 1.0. Higher means the audio content is preserved more accurately.

### D. Workflow Summarized
1.  **Generate**: Clean Snores + Noise -> `SNR_-5dB/dog/mixed.wav`
2.  **Run Experiments**: `run_all_experiments.py` -> calls `methods/*.py` -> `Denoised_DFNet/-5dB/dog/mixed.wav`
3.  **Judge**: `judge/*.py` reads the denoised files -> generates stats (likely CSVs).
4.  **Analyze**: `Analysis.ipynb` reads the stats -> Generates Plots/Reports.
