# Environmental Sound Recognition Under Unseen & Adverse Conditions

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Torchaudio](https://img.shields.io/badge/Torchaudio-2.0%2B-red.svg)](https://pytorch.org/audio/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Official repository for the Introduction to Deep Learning (DL2026) course project: **Multi-label Environmental Sound Classification under Unseen and Heavy Noise Corruptions**.

---

## 📖 Overview

Real-world acoustic environments present severe corruptions from non-stationary background noise and reverberation. While state-of-the-art Audio Spectrogram Transformers (AST) achieve exceptional performance on curated clean datasets, their accuracy collapses when evaluated under out-of-distribution noise (e.g. dropping from **0.5066** mAP down to **0.2040** at $-5\text{ dB}$ SNR).

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
│   ├── classifier.py           # Multi-label classifier head with pooling ablations
│   ├── encoder.py              # ASTEncoder (DeiT ViT-Tiny, Small, Base)
│   ├── resnet_encoder.py       # ResNet-18 and ResNet-34 CNN encoders
│   └── denoiser.py             # Self-supervised autoencoder backbone
├── metrics/                    # Multi-label evaluation metrics
│   ├── classification.py       # mAP, mAUC, Micro/Macro F1, Top-1/Top-5 Hit
│   └── reconstruction.py       # SDR, SNR, MSE evaluation
├── scripts/                    # Training, evaluation, and inference CLI utilities
│   ├── train_classifier.py     # Main end-to-end classifier training script
│   ├── eval_benchmark.py       # Automated multi-condition SNR benchmark evaluator
│   ├── inference.py            # Standalone single-file and batch inference engine
│   └── profile_model.py        # Model complexity (FLOPs, parameters) profiler
├── report/                     # Project report templates and guidelines
│   └── PROJECT_REPORT_TEMPLATE.md
├── DATA.md                     # Comprehensive dataset documentation
└── README.md                   # Project overview and reproduction guide
```

---

## 🛠️ Installation & Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/TotallyNotMinh/Environmental-Sound-Recognition-Under-Unseen-Conditions.git
cd Environmental-Sound-Recognition-Under-Unseen-Conditions
```

### 2. Create Virtual Environment
```bash
conda create -n dl_audio python=3.12 -y
conda activate dl_audio

# Install PyTorch with CUDA support (adjust cuda version as needed)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install numpy scipy pandas tqdm soundfile torchinfo
```

---

## 📊 Dataset Setup

Follow [`DATA.md`](DATA.md) for full dataset specifications and links. The dataset directory structure should be arranged as follows:

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

## 🚀 Steps for Reproducing Main Experiments

### 1. Training the Main ViT-Tiny Classifier (Single GPU)
```bash
python scripts/train_classifier.py \
    --data-path data/fsd50k \
    --arch tiny \
    --batch-size 12 \
    --num-epoch 30 \
    --colored-noise-prob 0.5 \
    --reverb-prob 0.3 \
    --checkpoint-dir checkpoints/vit_tiny/
```

### 2. Training with Multi-GPU DDP (e.g. Dual-GPU Kaggle T4 x2)
```bash
torchrun --nproc_per_node=2 scripts/train_classifier.py \
    --data-path /kaggle/input/datasets/yousirui1/fsd50k/fsd50k \
    --arch tiny \
    --batch-size 12 \
    --num-workers 2 \
    --num-epoch 30 \
    --patience 30 \
    --colored-noise-prob 0.5 \
    --reverb-prob 0.3 \
    --checkpoint-dir /kaggle/working/checkpoints/vit_tiny_ddp/
```

### 3. Evaluating on the Clean & Noisy Robustness Benchmark
```bash
python scripts/eval_benchmark.py \
    --checkpoint checkpoints/vit_tiny/best_classifier.pth \
    --data-path data/fsd50k \
    --arch tiny \
    --batch-size 16
```

### 4. Running Inference / Audio Demo
```bash
# Run prediction on a single audio file
python scripts/inference.py \
    --audio sample.wav \
    --checkpoint checkpoints/vit_tiny/best_classifier.pth \
    --arch tiny \
    --top-k 5

# Run batch prediction on a directory
python scripts/inference.py \
    --audio-dir path/to/wavs/ \
    --checkpoint checkpoints/vit_tiny/best_classifier.pth \
    --output predictions.csv
```

---

## 👥 Team & Member Contributions

| Member | Role | Core Deliverable |
| :--- | :--- | :--- |
| **Đào Chí Trung** | CNN Baselines | EfficientNet-B0 and B4 baselines, model complexity profiling |
| **Phạm Hồng Vân** | CNN Baselines | ResNet-18 and ResNet-34 baselines, model profiling and evaluation |
| **Vũ Thị Kim Oanh** | Augmentation Ablations | SpecAugment vs. Mixup 6-way ablation study |
| **Nguyễn Thị Ngọc Ánh** | Classification Layer Ablation | Dual-Token vs. GAP vs. GAP+Max vs. Attention Pooling on 1,188 patch tokens |
| **Phạm Gia Anh** | Model Scaling Comparison | ViT-Tiny vs. ViT-Small architecture comparison and FLOP/latency profiling |
| **Nguyễn Khải Minh** | Dataset Engineering & Documentation | ACE noise benchmark synthesis, FSD50K EDA, `DATA.md` |
| **Đặng Nhật Minh** | Pretraining, Robustness Pipeline | Self-supervised ACAD pretraining, DSP colored noise & reverb, benchmark pipeline |
