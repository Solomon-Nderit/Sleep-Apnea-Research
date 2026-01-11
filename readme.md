# Benchmarking Deep Learning vs. Signal Processing for Real-Time Sleep Apnea Detection

*An investigation into the "Efficiency-Accuracy Trade-off" of running noise suppression algorithms on mobile edge hardware.*

---

## 📌 Project Overview

Smartphone-based acoustic monitoring is a scalable solution for diagnosing Obstructive Sleep Apnea (OSA). However, uncontrolled bedroom environments (fans, sirens, snoring partners) introduce noise that masks diagnostic cues. While the industry trends toward massive Generative AI models for denoising, these are often too heavy for real-time mobile execution.

**This research benchmarks 5 distinct architectures**—ranging from classical DSP to state-of-the-art Generative AI—to find the optimal balance between *Semantic Preservation* and *Inference Latency* on CPU-constrained hardware.

### Key Outcome
The study identifies an **"Efficiency Paradox"** where 40-year-old mathematical filters (Spectral Gating) outperform modern Deep Learning on specific noise types while running **100x faster**.

---

## 🏗️ Repository Structure

This codebase is organized into three pipeline stages: **Generation**, **Denoising**, and **Evaluation**.

```bash
├── combine/                  # 1. DATA GENERATION
│   ├── combine_with_categories.py  # Mixes clean snores + ESC-50 noise at specific SNRs
│   └── combiner.py                 # Basic random mixer
│
├── methods/                  # 2. DENOISING ALGORITHMS
│   ├── demucs_cleaner.py           # Generative AI (Demucs - Meta)
│   ├── deepfilternet_cleaner.py    # Lightweight AI (DeepFilterNet)
│   ├── noise_reduce.py             # Spectral Gating (Stationary noise removal)
│   ├── wiener_filtering.py         # Wiener Filter (Statistical estimation)
│   ├── wavelet_filtering.py        # Wavelet Thresholding (db4)
│
├── judge/                    # 3. EVALUATION (The "Judge")
│   ├── judge.py                    # Classification confidence (Is it a snore?)
│   ├── judge_similarity.py         # Cosine Similarity of YAMNet embeddings
│   └── yamnet_class_map.csv        # Class definitions
│
├── run_all_experiments.py    # Main orchestrator: Runs the full benchmark loop
└── Analysis.ipynb            # Visualizes efficiency/accuracy trade-offs
```

---

## 🔬 Methodology

### 1. Synthetic "Stress-Test" Dataset
To simulate hostile acoustic environments, we engineered a synthetic dataset using `combine/combine_with_categories.py`.
*   **Source**: Clean biomedical snore recordings mixed with environmental noise from the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50).
*   **Noise Profiles**:
    *   *Stationary*: Fans, AC hum, engine drone.
    *   *Transient*: Dog barks, sirens, sudden impacts.
*   **SNR Levels**: Tested at **0dB** (Equal volume) and **-5dB** (Noise louder than signal).

### 2. The Contenders (Algorithms)
We benchmarked five architectures representing the history of audio processing:
*   **Demucs (Meta)**: Hybrid Transformer/U-Net. State-of-the-art source separation.
*   **DeepFilterNet**: Low-latency Deep Learning for speech enhancement.
*   **Spectral Gating**: Threshold-based frequency subtraction (using `noisereduce`).
*   **Wiener Filtering**: Minimum mean square error estimation.
*   **Wavelet Denoising**: Discrete Wavelet Transform (db4) with Soft Thresholding.

### 3. The Pivot: Objective Evaluation (YAMNet)
Initial attempts to use a custom-trained classifier introduced circular bias (the model overfitted to synthetic noise patterns). This repo implements a pivoted evaluation protocol (`judge/judge_similarity.py`):
*   **Metric**: Cosine Similarity of Feature Embeddings.
*   **Model**: YAMNet (Pre-trained on AudioSet).
*   **Logic**: Instead of training a new judge, we extract embeddings from the "Clean" snore and the "Denoised" output. High similarity means the algorithm preserved the semantic identity of the sound without "hallucinating" artifacts.

> **⚠️ Legacy Code Archive**: The source code for the initial (failed) custom classifier and earlier experimental iterations can be examined for historical context here: [Solomon-Nderit-Evaluating-Noise-Reduction-Techniques](Solomon-Nderit-Evaluating-Noise-Reduction-Techniques)

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy scipy librosa torch tensorflow tensorflow_hub noisereduce demucs
```
*Note: DeepFilterNet requires a specific Rust/Cargo installation. See `methods/deepfilternet_cleaner.py` for details.*

### Running the Benchmarks
The `run_all_experiments.py` script is the main entry point. It will iterate through your `experiment_data/` folders and apply all 5 cleaners.

```bash
# 1. Generate Data (Edit script to point to your raw audio sources)
python combine/combine_with_categories.py

# 2. Run Benchmarks
python run_all_experiments.py

# 3. Evaluate Results
python judge/judge_similarity.py
```

---

## 📊 Key Results

| Architecture | Type | Latency (CPU) | Performance (Stationary) | Performance (Transient) |
| :--- | :--- | :--- | :--- | :--- |
| **Demucs** | Gen AI | ~4.0s (Slow) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **DeepFilterNet** | Light AI | ~0.08s | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Spectral Gating** | DSP | ~0.02s (Fast) | ⭐⭐⭐⭐ | ⭐ |

### The Efficiency Paradox
Massive AI models (Demucs) provide excellent separation but disqualify themselves for real-time mobile use due to 4s+ latency.

### The "Dumb" Math Win
For fan noise (the most common bedroom sound), simple **Spectral Gating** outperformed Generative AI in signal fidelity while consuming **<1% of the compute**.

### Proposed Solution: "AdaptiveSleep"
Based on this code, I propose a Cascaded Architecture:
> Use ultra-low-power Spectral Gating as a default "sentinel" for steady noise, and dynamically wake the AI model only when transient events (sirens/barks) are detected.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📬 Contact

Solomon Nderitu Wahome - Non-Trivial Fellow