# Environmental Sound Recognition Under Unseen & Adverse Conditions

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Torchaudio](https://img.shields.io/badge/Torchaudio-2.0%2B-red.svg)](https://pytorch.org/audio/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Official repository for the Introduction to Deep Learning (DL2026) course project: **Multi-label Environmental Sound Classification under Unseen and Heavy Noise Corruptions**.

---

## 📖 Overview

Real-world acoustic environments present severe corruptions from non-stationary background noise and reverberation. While state-of-the-art Audio Spectrogram Transformers (AST) achieve exceptional performance on curated clean datasets, their accuracy collapses when evaluated under out-of-distribution noise (e.g. dropping from **0.5068** mAP down to **0.2040** at $-5\text{ dB}$ SNR).

This project investigates and resolves this robustness gap on the **FSD50K** (200-class) benchmark:
1. **Architectural Comparison (Setup 1):** Systematic benchmarking of CNN baselines (ResNet-18, ResNet-34, EfficientNet-B0, EfficientNet-B4) vs. Vision Transformers (ViT-Tiny, ViT-Small).
2. **Noise Robustness & Zero-Data Augmentation (Setup 2):** Developing purely algorithmic DSP-based colored noise ($1/f^\alpha$) and Schroeder room reverberation augmentations to establish noise invariance without external datasets.
3. **Comprehensive Ablation Studies (Setup 3):** Dissecting data augmentations (SpecAugment vs. Mixup), classification token pooling strategies (Dual-Token, GAP, GAP+Max, Attention Pooling), and model scaling.

---

## 📂 Repository Structure

```text
.
├── config/                     # Configuration definitions
├── data/                       # Dataset loaders, augmentations, and samplers
│   ├── dataset.py              # FSD50K multi-label dataset pipeline
│   ├── augment.py              # SpecAugment, Mixup, ColoredNoise, Reverb
│   └── sampler.py              # DistributedWeightedSampler for class balancing
├── models/                     # Deep learning model architectures
│   ├── classifier.py           # Unified classifier head (Dual-Token, GAP, GAP+Max, Attention)
│   ├── encoder.py              # ASTEncoder (DeiT ViT-Tiny, Small, Base)
│   ├── resnet_encoder.py       # ResNet-18 and ResNet-34 CNN encoders
│   ├── efficientnet_encoder.py # EfficientNet-B0 and EfficientNet-B4 CNN encoders
│   └── denoiser.py             # Self-supervised autoencoder backbone
├── metrics/                    # Multi-label evaluation metrics
│   ├── classification.py       # mAP, mAUC, Micro/Macro F1, Top-1/Top-5 Hit
│   └── reconstruction.py       # SDR, SNR, MSE evaluation
├── scripts/                    # Training, evaluation, and inference CLI utilities
│   ├── train_classifier.py     # Main end-to-end classifier training script
│   ├── eval_clean_checkpoint.py# Standalone FSD50K clean held-out evaluation
│   ├── eval_benchmark.py       # Multi-condition unseen benchmark evaluator
│   ├── eval_augmentation_comparison.py # 6-way augmentation ablation evaluator
│   ├── eval_all_checkpoints.py # Batch benchmark evaluator across SNR tiers
│   └── inference.py            # Standalone single-file and batch inference engine
├── metadata/                   # Benchmark CSV results and evaluation manifests
├── report/                     # LaTeX research paper, figures, and compiled PDF
├── DATA.md                     # Comprehensive dataset, augmentation & robustness documentation
└── README.md                   # Project overview and reproduction guide
```

---

## 🛠️ Installation & Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/TotallyNotMinh/DL2026-22-36.git
cd DL2026-22-36
```

### 2. Create Virtual Environment
```bash
conda create -n dl_audio python=3.12 -y
conda activate dl_audio

# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install numpy scipy pandas tqdm soundfile torchinfo
```

---

## 📊 Dataset Setup

Follow [`DATA.md`](DATA.md) for full dataset specifications, data augmentation formulations, and robustness benchmark details. The dataset directory structure should be arranged as follows:

```text
data/fsd50k/
├── FSD50K.ground_truth/
│   ├── dev.csv              # train/val split annotations
│   ├── eval.csv             # evaluation annotations
│   └── vocabulary.csv       # class index, name, mid
├── FSD50K.dev_audio_16k/    # training and validation WAV files
└── FSD50K.eval_audio_16k/   # evaluation WAV files
```

---

## 💾 Pre-trained Weights & Model Checkpoints

All trained model weights, evaluation checkpoints (ResNet, EfficientNet, ViT-Tiny, ViT-Small, Denoising Encoders), and experiment archives (containing `.pth` and `.zip` files) are hosted on Google Drive:
- 🔗 **Google Drive Checkpoints Folder:** [Download Pre-trained Model Weights & Checkpoints](https://drive.google.com/drive/folders/1ZCv6w_MN8kF1f3m0npEY96q4KrkWGROi?usp=sharing)

To evaluate or run inference with these checkpoints, download and extract them into the `checkpoints/` directory:
```bash
mkdir -p checkpoints
# Place or unzip downloaded checkpoint files (.pth) into checkpoints/
```

---

## 🚀 Steps for Reproducing Main Experiments

> **Note:** All commands assume the conda environment is active and the working directory is the repo root.
> Checkpoints are saved to `checkpoints/` by default. Adjust `--data-path` to match your local dataset path.

---

### Step 1 — Data Download & Preprocessing

FSD50K is downloaded directly via the Kaggle CLI (pre-resampled to 16 kHz):

```bash
pip install kaggle
# Ensure your ~/.kaggle/kaggle.json API token is configured
kaggle datasets download -d yousirui1/fsd50k -p data/ --unzip
```

Expected layout after download:
```text
data/fsd50k/
├── FSD50K.ground_truth/
│   ├── dev.csv              # 36,796 clips (train + val)
│   ├── eval.csv             # 10,231 clips (test)
│   └── vocabulary.csv       # 200 class labels
├── FSD50K.dev_audio_16k/    # *.wav  (training & validation)
└── FSD50K.eval_audio_16k/   # *.wav  (evaluation / held-out)
```

---

### Step 2 — Baseline CNN Architectures (Setup 1)

All CNN baselines use full SpecAugment + Mixup augmentation.
Convolutional networks use $\eta = 5 \times 10^{-4}$ (10× higher than ViT):

```bash
# ResNet-18
python scripts/train_classifier.py \
    --data-path data/fsd50k \
    --encoder resnet18 \
    --batch-size 16 \
    --num-epoch 30 \
    --encoder-lr 5e-4 \
    --head-lr 5e-4 \
    --weight-decay 1e-4 \
    --lr-scheduler ast_step \
    --freq-mask 48 --time-mask 192 \
    --mixup-prob 0.5 --mixup-alpha 0.5 \
    --time-shift 10 --noise-level 0.05

# ResNet-34
python scripts/train_classifier.py \
    --data-path data/fsd50k \
    --encoder resnet34 \
    --batch-size 16 \
    --num-epoch 30 \
    --encoder-lr 5e-4 \
    --head-lr 5e-4 \
    --weight-decay 1e-4 \
    --lr-scheduler ast_step \
    --freq-mask 48 --time-mask 192 \
    --mixup-prob 0.5 --mixup-alpha 0.5 \
    --time-shift 10 --noise-level 0.05

# EfficientNet-B0
python scripts/train_classifier.py \
    --data-path data/fsd50k \
    --encoder efficientnet_b0 \
    --batch-size 16 \
    --num-epoch 30 \
    --encoder-lr 5e-4 \
    --head-lr 5e-4 \
    --weight-decay 1e-4 \
    --lr-scheduler ast_step \
    --freq-mask 48 --time-mask 192 \
    --mixup-prob 0.5 --mixup-alpha 0.5 \
    --time-shift 10 --noise-level 0.05

# EfficientNet-B4
python scripts/train_classifier.py \
    --data-path data/fsd50k \
    --encoder efficientnet_b4 \
    --batch-size 16 \
    --num-epoch 30 \
    --encoder-lr 5e-4 \
    --head-lr 5e-4 \
    --weight-decay 1e-4 \
    --lr-scheduler ast_step \
    --freq-mask 48 --time-mask 192 \
    --mixup-prob 0.5 --mixup-alpha 0.5 \
    --time-shift 10 --noise-level 0.05
```

---

### Step 3 — Augmentation Ablation Study (Setup 3a)

All six variants use ViT-Tiny with the dual-token pooling baseline (`--pooling None`) for 30 epochs.
The only variables are the active augmentation flags:

```bash
# Condition 1: no_aug — no data augmentation whatsoever
python scripts/train_classifier.py \
    --data-path data/fsd50k \
    --encoder ast --arch tiny \
    --batch-size 12 --num-epoch 30 --patience 30 \
    --encoder-lr 5e-5 --head-lr 5e-5 --weight-decay 1e-4 \
    --lr-scheduler ast_step \
    --freq-mask 0 --time-mask 0 \
    --mixup-prob 0.0 --noise-level 0.0 \
    --time-shift 0 --colored-noise-prob 0.0 --reverb-prob 0.0

# Condition 2: time_mask — SpecAugment time masking only (T=192 frames)
python scripts/train_classifier.py \
    --data-path data/fsd50k \
    --encoder ast --arch tiny \
    --batch-size 12 --num-epoch 30 --patience 30 \
    --encoder-lr 5e-5 --head-lr 5e-5 --weight-decay 1e-4 \
    --lr-scheduler ast_step \
    --freq-mask 0 --time-mask 192 \
    --mixup-prob 0.0 --noise-level 0.0 \
    --time-shift 0 --colored-noise-prob 0.0 --reverb-prob 0.0

# Condition 3: freq_mask — SpecAugment frequency masking only (F=48 bins)
python scripts/train_classifier.py \
    --data-path data/fsd50k \
    --encoder ast --arch tiny \
    --batch-size 12 --num-epoch 30 --patience 30 \
    --encoder-lr 5e-5 --head-lr 5e-5 --weight-decay 1e-4 \
    --lr-scheduler ast_step \
    --freq-mask 48 --time-mask 0 \
    --mixup-prob 0.0 --noise-level 0.0 \
    --time-shift 0 --colored-noise-prob 0.0 --reverb-prob 0.0

# Condition 4: time_freq_mask — Both SpecAugment masks (T=192, F=48)
python scripts/train_classifier.py \
    --data-path data/fsd50k \
    --encoder ast --arch tiny \
    --batch-size 12 --num-epoch 30 --patience 30 \
    --encoder-lr 5e-5 --head-lr 5e-5 --weight-decay 1e-4 \
    --lr-scheduler ast_step \
    --freq-mask 48 --time-mask 192 \
    --mixup-prob 0.0 --noise-level 0.0 \
    --time-shift 0 --colored-noise-prob 0.0 --reverb-prob 0.0

# Condition 5: mixup — Mixup only (α=0.5, p=0.5), no SpecAugment
python scripts/train_classifier.py \
    --data-path data/fsd50k \
    --encoder ast --arch tiny \
    --batch-size 12 --num-epoch 30 --patience 30 \
    --encoder-lr 5e-5 --head-lr 5e-5 --weight-decay 1e-4 \
    --lr-scheduler ast_step \
    --freq-mask 0 --time-mask 0 \
    --mixup-prob 0.5 --mixup-alpha 0.5 --noise-level 0.0 \
    --time-shift 0 --colored-noise-prob 0.0 --reverb-prob 0.0

# Condition 6: full — SpecAugment + Mixup combined (best clean accuracy)
python scripts/train_classifier.py \
    --data-path data/fsd50k \
    --encoder ast --arch tiny \
    --batch-size 12 --num-epoch 30 --patience 30 \
    --encoder-lr 5e-5 --head-lr 5e-5 --weight-decay 1e-4 \
    --lr-scheduler ast_step \
    --freq-mask 48 --time-mask 192 \
    --mixup-prob 0.5 --mixup-alpha 0.5 --noise-level 0.0 \
    --time-shift 0 --colored-noise-prob 0.0 --reverb-prob 0.0
```

---

### Step 4 — Pooling Strategy Ablation (Setup 3b)

All four variants use ViT-Tiny with the full augmentation stack (SpecAugment + Mixup + time-shift + Gaussian noise):

```bash
# Baseline: Dual-Token (CLS + DIST tokens, default —no --pooling flag or --pooling None)
python scripts/train_classifier.py \
    --data-path data/fsd50k \
    --encoder ast --arch tiny \
    --batch-size 12 --num-epoch 30 --patience 30 \
    --encoder-lr 5e-5 --head-lr 5e-5 --weight-decay 1e-4 \
    --lr-scheduler ast_step \
    --freq-mask 48 --time-mask 192 \
    --mixup-prob 0.5 --mixup-alpha 0.5 \
    --time-shift 10 --noise-level 0.05 \
    --colored-noise-prob 0.0 --reverb-prob 0.0

# GAP: Global Average Pooling over all 1,188 patch tokens
python scripts/train_classifier.py \
    --data-path data/fsd50k \
    --encoder ast --arch tiny \
    --pooling gap \
    --batch-size 12 --num-epoch 30 --patience 30 \
    --encoder-lr 5e-5 --head-lr 5e-5 --weight-decay 1e-4 \
    --lr-scheduler ast_step \
    --freq-mask 48 --time-mask 192 \
    --mixup-prob 0.5 --mixup-alpha 0.5 \
    --time-shift 10 --noise-level 0.05 \
    --colored-noise-prob 0.0 --reverb-prob 0.0

# GAP+Max: Concatenation of GAP and Global Max Pooling
python scripts/train_classifier.py \
    --data-path data/fsd50k \
    --encoder ast --arch tiny \
    --pooling gap_max \
    --batch-size 12 --num-epoch 30 --patience 30 \
    --encoder-lr 5e-5 --head-lr 5e-5 --weight-decay 1e-4 \
    --lr-scheduler ast_step \
    --freq-mask 48 --time-mask 192 \
    --mixup-prob 0.5 --mixup-alpha 0.5 \
    --time-shift 10 --noise-level 0.05 \
    --colored-noise-prob 0.0 --reverb-prob 0.0

# Attention Pooling: Learned softmax-weighted aggregation over patch tokens
python scripts/train_classifier.py \
    --data-path data/fsd50k \
    --encoder ast --arch tiny \
    --pooling attention \
    --batch-size 12 --num-epoch 30 --patience 30 \
    --encoder-lr 5e-5 --head-lr 5e-5 --weight-decay 1e-4 \
    --lr-scheduler ast_step \
    --freq-mask 48 --time-mask 192 \
    --mixup-prob 0.5 --mixup-alpha 0.5 \
    --time-shift 10 --noise-level 0.05 \
    --colored-noise-prob 0.0 --reverb-prob 0.0
```

---

### Step 5 — ViT-Small Architecture Scaling (Setup 1 / Setup 3c)

```bash
python scripts/train_classifier.py \
    --data-path data/fsd50k \
    --encoder ast --arch small \
    --batch-size 12 --num-epoch 30 --patience 30 \
    --encoder-lr 5e-5 --head-lr 5e-5 --weight-decay 1e-4 \
    --lr-scheduler ast_step \
    --freq-mask 48 --time-mask 192 \
    --mixup-prob 0.5 --mixup-alpha 0.5 \
    --time-shift 10 --noise-level 0.05
```

---

### Step 6 — Proposed Robust Model (Setup 2)

ViT-Tiny + Attention Pooling + Mixup + Colored Noise ($1/f^\alpha$) + Schroeder Reverberation.
SpecAugment is disabled to prevent conflicting with learned noise invariance:

```bash
python scripts/train_classifier.py \
    --data-path data/fsd50k \
    --encoder ast --arch tiny \
    --pooling attention \
    --batch-size 12 --num-epoch 30 --patience 30 \
    --encoder-lr 5e-5 --head-lr 5e-5 --weight-decay 1e-4 \
    --lr-scheduler ast_step \
    --freq-mask 0 --time-mask 0 \
    --mixup-prob 0.5 --mixup-alpha 0.5 \
    --time-shift 10 --noise-level 0.05 \
    --colored-noise-prob 0.5 \
    --colored-noise-min-snr -5.0 \
    --colored-noise-max-snr 20.0 \
    --reverb-prob 0.3 \
    --reverb-min-t60 0.15 \
    --reverb-max-t60 0.60
```

---

### Step 7 — Multi-GPU DDP Training (e.g. Dual-GPU Kaggle / Server)

```bash
torchrun --nproc_per_node=2 scripts/train_classifier.py \
    --data-path data/fsd50k \
    --encoder ast --arch tiny \
    --pooling attention \
    --batch-size 12 --num-epoch 30 --patience 30 \
    --encoder-lr 5e-5 --head-lr 5e-5 --weight-decay 1e-4 \
    --lr-scheduler ast_step \
    --freq-mask 0 --time-mask 0 \
    --mixup-prob 0.5 --mixup-alpha 0.5 \
    --time-shift 10 --noise-level 0.05 \
    --colored-noise-prob 0.5 \
    --colored-noise-min-snr -5.0 \
    --colored-noise-max-snr 20.0 \
    --reverb-prob 0.3 \
    --reverb-min-t60 0.15 \
    --reverb-max-t60 0.60
```

---

### Step 8 — Evaluation on Clean & Adverse Noise Benchmarks

```bash
# Standalone evaluation on clean held-out test split (10,231 clips)
# Auto-detects architecture and pooling head from checkpoint state dict
python scripts/eval_clean_checkpoint.py \
    --data-path data/fsd50k \
    --checkpoint checkpoints/vit-tiny-mixup-colored-noises.pth \
    --encoder ast \
    --arch tiny \
    --pooling auto

# Multi-condition noise evaluation across all 6 augmentation ablation checkpoints
# Requires: metadata/fsd50k_evaluation_metadata.csv (40,925-row ACE noise manifest, seed=42)
# ACE noise source: data/ace/Single/  (see DATA.md §5 for ACE dataset setup)
python scripts/eval_augmentation_comparison.py \
    --data-path data/fsd50k \
    --checkpoint-dir checkpoints \
    --metadata-path metadata/fsd50k_evaluation_metadata.csv \
    --output-dir metadata

# Comprehensive evaluation across all architectures and SNR tiers (+10, +5, 0, -5 dB)
# Pre-synthesized multi-SNR evaluation dataset: https://www.kaggle.com/datasets/knuckleizmad/fsd50k-eval-various-snrs
python scripts/eval_all_checkpoints.py \
    --data-path data/fsd50k \
    --checkpoint-dir checkpoints \
    --metadata-path metadata/fsd50k_evaluation_metadata.csv \
    --output-csv metadata/new_checkpoints_noise_robustness.csv
```

> **Evaluation Dataset:** The pre-rendered multi-condition noisy evaluation benchmark across varying SNRs can also be downloaded directly from Kaggle:
> - 🔗 **Kaggle Dataset:** [FSD50K Evaluation under Various SNRs](https://www.kaggle.com/datasets/knuckleizmad/fsd50k-eval-various-snrs)

---

### Step 9 — Running Inference / Audio Demo

```bash
# Single-file prediction (top-5 classes above threshold)
python scripts/inference.py \
    --audio sample.wav \
    --checkpoint checkpoints/best_classifier.pth \
    --arch tiny \
    --top-k 5 \
    --threshold 0.20

# Batch prediction over an audio directory, output to JSON
python scripts/inference.py \
    --audio-dir path/to/wavs/ \
    --checkpoint checkpoints/best_classifier.pth \
    --arch tiny \
    --output predictions.json
```




## 👥 Team & Member Contributions

| Member Name | Student ID | Role | Core Deliverable |
| :--- | :--- | :--- | :--- |
| **Dao Chi Trung** | 23BA14295 | CNN Baselines | EfficientNet-B0 and B4 baselines, model complexity profiling |
| **Pham Hong Van** | 23BA14318 | CNN Baselines | ResNet-18 and ResNet-34 baselines, model profiling and evaluation |
| **Vu Thi Kim Oanh** | 23BA14225 | Augmentation Ablations | SpecAugment vs. Mixup 6-way ablation study |
| **Nguyen Thi Ngoc Anh** | 2411095 | Classification Layer Ablation | Dual-Token vs. GAP vs. GAP+Max vs. Attention Pooling on 1,188 patch tokens |
| **Pham Gia Anh** | 2410084 | Model Scaling Comparison | ViT-Tiny vs. ViT-Small architecture comparison and FLOP/latency profiling |
| **Nguyen Khai Minh** | 2410607 | Dataset Engineering & Documentation | ACE noise benchmark synthesis, FSD50K EDA, `DATA.md` |
| **Dang Nhat Minh** | 2410667 | Robustness Pipeline | Algorithmic DSP colored noise & reverb augmentations, benchmark evaluation pipeline |
