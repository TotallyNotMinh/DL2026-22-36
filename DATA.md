# Dataset, Augmentation & Robustness Documentation (`DATA.md`)

This document provides complete documentation for the datasets, partitions, audio preprocessing pipelines, data augmentations, empirical ablation studies, out-of-distribution noise robustness, and decision threshold calibration for environmental sound recognition on FSD50K.

---

## 1. Official Datasets & Metadata

### 1.1 Primary Benchmark: FSD50K (Freesound Dataset 50K)
* **Official URL:** [Zenodo Record 4060432](https://zenodo.org/record/4060432)
* **Kaggle Distribution:** [yousirui1/fsd50k](https://www.kaggle.com/datasets/yousirui1/fsd50k)
* **Dataset Version:** 1.0 (Fonseca et al., IEEE/ACM TASLP 2021)
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Audio Characteristics:**
  * Total duration: ~108 hours of sound events.
  * Native sampling rates: Diverse (downsampled to 16 kHz for experimentation).
  * Taxonomy: 200 sound classes hierarchically drawn from the AudioSet Ontology.
  * Task: Multi-label sound event classification.

### 1.2 Acoustic Noise Benchmark: ACE Dataset (Acoustic Characterization of Environments)
* **Official Website & Corpus Link:** [ACE Challenge - Imperial College London](http://ee.ic.ac.uk/naylor/ACEweb/index.html)
* **Official Reference:** Eaton et al., "The ACE Challenge – Corpus Description and Performance Evaluation", IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2016.
* **Dataset Version:** 1.0 (ACE Corpus)
* **Noise Types & Environments:** Real-world ambient background noises captured across physical indoor rooms (offices, lecture rooms, meeting rooms) containing genuine HVAC ventilation fans, ambient human speech babble, and computer cooling equipment.
* **Noise Matching & Calibration Protocol:**
  * Each clean FSD50K evaluation clip ($10,231$ clips, $10.0\text{ s}$ duration) is paired with a randomly selected continuous noise segment from the ACE corpus.
  * Root-Mean-Square (RMS) power is calculated for both clean signal $x(t)$ and noise segment $n(t)$.
  * Noise scaling factor is calibrated to achieve the exact target SNR:
    $$\sigma_{\text{noise}} = \frac{\text{RMS}(x)}{\text{RMS}(n) \cdot 10^{\text{SNR}/20}}$$
  * Four standardized SNR tiers are generated: $+10\text{ dB}$, $+5\text{ dB}$, $0\text{ dB}$, and $-5\text{ dB}$ ($40,924$ corrupted evaluation audio files manifest in `metadata/fsd50k_evaluation_metadata.csv`).

---

## 2. Dataset Splits & Directory Structure

The repository expects the following standard directory layout matching `yousirui1/fsd50k`:

```text
data_path/
├── FSD50K.ground_truth/
│   ├── dev.csv              # Training and validation split annotations
│   ├── eval.csv             # Evaluation split annotations
│   └── vocabulary.csv       # 200 class indices, label names, and AudioSet MIDs
├── FSD50K.dev_audio_16k/    # 40,966 WAV clips resampled to 16,000 Hz
└── FSD50K.eval_audio_16k/   # 10,231 WAV clips resampled to 16,000 Hz
```

### Partition Breakdown
| Split Name | Source File | Audio Directory | Number of Clips | Total Hours | Split Role |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Train** | `dev.csv` (`split == "train"`) | `FSD50K.dev_audio_16k/` | 36,796 | ~80.2h | Model training with data augmentations |
| **Val** | `dev.csv` (`split == "val"`) | `FSD50K.dev_audio_16k/` | 4,170 | ~11.8h | Hyperparameter selection & checkpointing |
| **Eval (Clean)**| `eval.csv` | `FSD50K.eval_audio_16k/` | 10,231 | ~24.1h | Primary held-out clean evaluation |
| **Eval (Noisy)**| Benchmark Manifest | `data/evaluation_fsd50k/` | 10,231 $\times$ 4 | ~96.4h | Robustness evaluation under 4 SNR levels |

### 2.1 Downloading FSD50K

**Option A — Kaggle CLI (Recommended)**

The Kaggle distribution by `yousirui1` contains audio pre-resampled to 16 kHz, saving the resampling step:
```bash
pip install kaggle
# Place your ~/.kaggle/kaggle.json API token first
kaggle datasets download -d yousirui1/fsd50k -p data/ --unzip
# Result: data/fsd50k/{FSD50K.ground_truth/, FSD50K.dev_audio_16k/, FSD50K.eval_audio_16k/}
```

**Option B — Zenodo (Official, requires manual resampling)**
```bash
# Download the five ground-truth and audio archives from Zenodo record 4060432
wget -c "https://zenodo.org/record/4060432/files/FSD50K.ground_truth.zip" -P data/fsd50k/
wget -c "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z01"   -P data/fsd50k/
wget -c "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z02"   -P data/fsd50k/
wget -c "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip"   -P data/fsd50k/
wget -c "https://zenodo.org/record/4060432/files/FSD50K.eval_audio.zip"  -P data/fsd50k/

# Unzip (multi-part)
cd data/fsd50k && zip -s 0 FSD50K.dev_audio.zip --out combined_dev.zip && unzip combined_dev.zip
unzip FSD50K.ground_truth.zip && unzip FSD50K.eval_audio.zip

# Resample to 16 kHz using torchaudio (run from repo root)
python - <<'EOF'
import torchaudio, os
from pathlib import Path
for split, src_dir, dst_dir in [
    ("dev",  "data/fsd50k/FSD50K.dev_audio",  "data/fsd50k/FSD50K.dev_audio_16k"),
    ("eval", "data/fsd50k/FSD50K.eval_audio", "data/fsd50k/FSD50K.eval_audio_16k"),
]:
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    for f in Path(src_dir).glob("*.wav"):
        wav, sr = torchaudio.load(f)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        torchaudio.save(str(Path(dst_dir) / f.name), wav, 16000)
EOF
```

### 2.2 Downloading & Generating the ACE Noise Benchmark

**Step 1 — Download ACE Corpus**

The ACE corpus is freely available from Imperial College London:
```bash
# Download Single-channel ambient noise recordings
wget -c "http://ee.ic.ac.uk/naylor/ACEweb/Data/Single.zip" -P data/ace/
cd data/ace && unzip Single.zip
# Expected layout: data/ace/Single/{Anechoic,Office_1,Office_2,...}/*.wav
```

**Step 2 — Generate the Noise Evaluation Benchmark**

The pre-built evaluation manifest `metadata/fsd50k_evaluation_metadata.csv` (40,925 rows, `seed=42`) is already committed to the repository. To regenerate the noisy audio files from this manifest:
```bash
python scripts/eval_benchmark.py \
    --data-path data/fsd50k \
    --ace-path data/ace/Single \
    --metadata-path metadata/fsd50k_evaluation_metadata.csv \
    --output-dir data/evaluation_fsd50k \
    --seed 42
```

This writes pre-mixed noisy WAV files to `data/evaluation_fsd50k/{snr_10,snr_5,snr_0,snr_neg5}/`.
If you prefer on-the-fly mixing (no pre-written WAVs), the evaluation scripts accept the raw ACE path directly and re-compute the mixing at inference time.

---



## 3. Audio Preprocessing Pipeline

All audio signals are processed deterministically following the standardized Audio Spectrogram Transformer (AST) pipeline:

1. **Mono Downmixing & Resampling:**
   * Multichannel audio is averaged across channels to single-channel mono.
   * Audio is polyphase resampled to $f_s = 16,000\text{ Hz}$ using `torchaudio.functional.resample`.
2. **Temporal Windowing & Cropping:**
   * Target clip duration: $T = 10.0\text{ seconds}$ ($L = 160,000\text{ samples}$).
   * **Train Split:** Random start index $t_{\text{start}} \in [0, \max(0, L_{\text{raw}} - L)]$.
   * **Val/Eval Splits:** Deterministic center crop $t_{\text{start}} = \frac{1}{2}(L_{\text{raw}} - L)$.
   * Clips shorter than $10.0\text{ s}$ are zero-padded at the end to $160,000$ samples.
3. **Log-Mel Spectrogram Transformation:**
   * STFT Window Length: $N_{\text{win}} = 400$ ($25\text{ ms}$) with Hann window.
   * Hop Size: $N_{\text{hop}} = 160$ ($10\text{ ms}$).
   * FFT Size: $N_{\text{fft}} = 1,024$.
   * Mel Filterbank: $N_{\text{mels}} = 128$ filters spanning $0\text{ Hz} - 8,000\text{ Hz}$.
   * Dynamic Range Compression: $\log(\text{Mel} + 10^{-6})$.
4. **Global Normalization:**
   * Feature standardization computed over the entire training corpus:
     $$\tilde{S} = \frac{S - \mu}{2 \sigma}, \quad \mu = -4.2677393, \quad \sigma = 4.5689974$$
   * Final Spectrogram Matrix: Fixed shape of $1 \times 128 \times 1,000$ (Channel $\times$ Freq $\times$ Time).

### 3.1 Acoustic Noise Benchmark (ACE Corpus) Generation Protocol

To establish a standardized out-of-distribution evaluation benchmark without training split contamination, we synthesize the ACE noise benchmark across 4 standardized SNR tiers:
1. **Source Noise Corpus:** Ambient noise recordings from the Acoustic Characterization of Environments (ACE) corpus, capturing real-world indoor acoustic spaces (offices, lecture rooms, meeting halls) with authentic background sources (HVAC fans, babble, computer cooling).
2. **Audio Processing & 10s Windowing:** Ambient audio is standardized to $16,000\text{ Hz}$ mono PCM. Continuous ambient segments are selected and trimmed into exact $10.0\text{ s}$ windows ($160,000$ samples) matching the FSD50K clip length.
3. **Calibrated SNR Additive Mixing:** For each of the 10,231 clean held-out evaluation clips $x(t)$, a background noise segment $n(t)$ is scaled and mixed at calibrated Signal-to-Noise Ratios:
   $$\text{RMS}(x) = \sqrt{\frac{1}{L}\sum_{t=1}^L x(t)^2}, \quad \sigma_{\text{noise}} = \frac{\text{RMS}(x)}{10^{\gamma_{\text{dB}}/20}}$$
   $$x_{\text{corrupt}}(t) = x(t) + \sigma_{\text{noise}} \cdot \frac{n(t)}{\text{RMS}(n(t))}$$
   Evaluated across four SNR tiers: $+10\text{ dB}$ (mild), $+5\text{ dB}$ (moderate), $0\text{ dB}$ (equal energy), and $-5\text{ dB}$ (severe corruption), producing $40,924$ total evaluation instances.
4. **Deterministic Reproducibility:** All noise segment choices, temporal start offsets, and additive scaling factors were generated using a fixed global seed (`seed = 42`), ensuring an identical, fair test bed across all baseline and robust checkpoints.

---

## 4. Data Augmentation Techniques & Formulations

The framework implements six primary data augmentation methods spanning spectral masking, acoustic superposition, and digital signal processing (DSP) environmental simulation.

| Technique | Domain | Core Parameters | Physical / Regularization Mechanism |
| :--- | :--- | :--- | :--- |
| **Time Masking** | Log-mel spectrogram | $T_{\text{mask}} \le 192$ frames ($1.92\text{ s}$), $N=2$ | Zeroes out temporal bands across all frequencies. Forces the model to classify events from partial clips without relying on transient onset bursts. |
| **Frequency Masking** | Log-mel spectrogram | $F_{\text{mask}} \le 48$ mel bins, $N=2$ | Zeroes out spectral channels across all time steps. Prevents overfitting to specific resonant frequencies, harmonics, or microphone frequency response peaks. |
| **SpecAugment** | Log-mel spectrogram | Joint Time ($192$) + Freq ($48$) masking | Joint 2D masking creating structured missing time-frequency patches that regularize high-capacity vision encoders. |
| **Mixup** | Waveform / Spectrogram | $\alpha_{\text{mix}} = 0.5$, $p=0.5$ | Convex linear combination of sample pairs and multi-label ground truth. Physically simulates acoustic wave superposition (polyphonic acoustic scenes). |
| **Continuous Colored Noise ($1/f^\alpha$)** | Raw waveform | $\alpha \sim \mathcal{U}(0, 2)$, $\text{SNR} \sim \mathcal{U}(-5, 20)\text{ dB}$ | Continuous spectral slope synthesis modeling natural acoustic power-law decay ($S(f) \propto 1/f^\alpha$) from white ($\alpha=0$) to pink ($\alpha=1$) and brown ($\alpha=2$). |
| **Schroeder Reverberation** | Raw waveform | $T_{60} \sim \mathcal{U}(0.15, 0.60)\text{ s}$, $v(t) \sim \mathcal{N}(0, 1)$ | Statistical Room Impulse Response (RIR) circular convolution via FFT, modeling room acoustic reflections and decaying reverberant tails. |

### 4.1 Mixup Formulation
Given two audio instances $(x_a, y_a)$ and $(x_b, y_b)$:
$$\lambda \sim \text{Beta}(\alpha, \alpha)$$
$$\tilde{x} = \lambda x_a + (1 - \lambda) x_b, \quad \tilde{y} = \lambda y_a + (1 - \lambda) y_b$$
In multi-label audio classification, linear interpolation in label space corresponds directly to polyphonic acoustic mixtures, encouraging linear behavior across intermediate layer representations.

### 4.2 Continuous Spectral-Slope Colored Noise Injection ($1/f^\alpha$)
Natural ambient acoustic noise (traffic, HVAC, wind turbulence) exhibits non-flat power spectra:
1. **White Gaussian Generator:** Generate $w(t) \sim \mathcal{N}(0, 1)$ of length $L = 160,000$ samples ($10.0\text{ s}$ at $16\text{ kHz}$).
2. **Spectral Filtering:**
   $$W(f) = \mathcal{F}\{w(t)\}, \quad H(f) = \frac{1}{\max(f, f_{\text{min}})^{\alpha/2}}, \quad f_{\text{min}} = 10\text{ Hz}$$
   $$\alpha \sim \mathcal{U}(0.0, 2.0), \quad \text{SNR}_{\text{dB}} \sim \mathcal{U}(-5.0, 20.0)\text{ dB}$$
3. **Calibrated Energy Scaling:**
   $$n_{\text{raw}}(t) = \mathcal{F}^{-1}\{W(f) \cdot H(f)\}$$
   $$\text{RMS}(x) = \sqrt{\frac{1}{L}\sum_{t=1}^L x(t)^2}, \quad \sigma_{\text{noise}} = \frac{\text{RMS}(x)}{10^{\text{SNR}_{\text{dB}} / 20}}$$
   $$x_{\text{aug}}(t) = x(t) + \sigma_{\text{noise}} \cdot \frac{n_{\text{raw}}(t)}{\text{RMS}(n_{\text{raw}}(t))}$$

### 4.3 Schroeder Room Reverberation Modeling
Simulates late diffuse room reflections without pre-recorded impulse response files:
$$h(t) = e^{-\frac{6.9078 \cdot t}{T_{60}}} \cdot v(t), \quad 0 \le t \le T_{\text{max}}, \quad v(t) \sim \mathcal{N}(0, 1)$$
$$x_{\text{reverb}}(t) = \mathcal{F}^{-1}\{\mathcal{F}\{x(t)\} \cdot \mathcal{F}\{h(t)\}\}$$
$$x_{\text{aug}}(t) = (1 - \text{wet}) \cdot x(t) + \text{wet} \cdot x_{\text{reverb}}(t) \cdot \left(\frac{\text{RMS}(x)}{\text{RMS}(x_{\text{reverb}})}\right)$$
where $T_{60} \in [0.15, 0.60]\text{ s}$ and $\text{wet} \in [0.10, 0.40]$.

---

## 5. Systematic In-Distribution Ablation Study (Clean FSD50K)

Evaluated across 6 isolated 30-epoch training runs on the identical ViT-Tiny (CLS+DIST backbone, $5.7\text{M}$ parameters) under fixed baseline settings:

| Configuration | Time Mask | Freq Mask | Mixup | Val mAP | Val mAUC | Static F1 ($\tau=0.50$) | Calib. Micro-F1 ($\tau^*$) | Top-1 Hit | Top-5 Hit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Augmentation** | $\times$ | $\times$ | $\times$ | 0.5419 | 0.8999 | 0.6533 | 0.6620 ($\tau=0.35$) | 73.84% | 86.59% |
| **Time Masking Only** | $\checkmark$ | $\times$ | $\times$ | 0.5396 | 0.8877 | 0.6521 | 0.6623 ($\tau=0.35$) | 73.19% | 85.95% |
| **Freq Masking Only** | $\times$ | $\checkmark$ | $\times$ | 0.5417 | 0.9169 | 0.6448 | 0.6589 ($\tau=0.35$) | 73.76% | 87.05% |
| **SpecAugment (Time+Freq)** | $\checkmark$ | $\checkmark$ | $\times$ | 0.5401 | 0.9242 | 0.6420 | 0.6585 ($\tau=0.35$) | 74.08% | 87.79% |
| **Mixup Only** | $\times$ | $\times$ | $\checkmark$ | **0.5534** | 0.9174 | 0.6517 | **0.6695** ($\tau=0.30$) | **75.32%** | 88.06% |
| **Full Pipeline (SpecAug + Mixup)** | $\checkmark$ | $\checkmark$ | $\checkmark$ | 0.5380 | **0.9351** | 0.6362 | 0.6674 ($\tau=0.30$) | 74.56% | **88.71%** |

### Key Findings:
1. **Spectral Masking Delivers Negligible In-Distribution Gain:** Without Mixup, masking methods fluctuate within $\le 0.23\%$ mAP of baseline ($0.5396\text{--}0.5419$), which is within seed noise.
2. **Mixup Provides True Regularization:** Standalone Mixup yields $+1.15\%$ Val mAP ($0.5534$), $+1.75\%$ mAUC ($0.9174$), and $+1.48\%$ Top-1 Hit ($75.32\%$).
3. **Compound Augmentation Penalty:** Applying SpecAugment on top of Mixup drops mAP by $-1.54\%$ ($0.5534 \to 0.5380$) due to over-masking already diluted audio mixtures.

---

## 6. Out-of-Distribution Noise Robustness Across SNR Levels

Evaluated on the held-out FSD50K evaluation set ($10,231$ clips) against real-world indoor ambient acoustic noise from the ACE Challenge across 4 SNR tiers:

| Augmentation Configuration | Clean mAP | SNR +10 dB | SNR +5 dB | SNR 0 dB | SNR -5 dB | Relative Drop (Clean $\to$ -5 dB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Augmentation** | 0.4869 | 0.3345 | 0.2799 | 0.2154 | 0.1551 | $-68.1\%$ |
| **Time Masking Only** | 0.4859 | 0.3369 | 0.2826 | 0.2183 | 0.1575 | $-67.6\%$ |
| **Freq Masking Only** | 0.4933 | 0.3438 | 0.2897 | 0.2303 | 0.1714 | $-65.3\%$ |
| **SpecAugment (Time+Freq)** | 0.4972 | 0.3470 | 0.2981 | 0.2388 | 0.1768 | $-64.4\%$ |
| **Mixup Only** | **0.5124** | **0.3866** | **0.3342** | **0.2679** | **0.2000** | **$-61.0\%$** |
| **Full Pipeline (SpecAug + Mixup)** | 0.5056 | 0.3657 | 0.3167 | 0.2571 | 0.1926 | $-61.9\%$ |

### Comparison with DSP-Augmented Training:
When trained with **Continuous Colored Noise + Schroeder Reverberation + Mixup + Attention Pooling** (the Proposed Robust ViT-Tiny, canonical evaluation on held-out test split):
- **Clean:** $0.5303$ mAP ($77.08\%$ Top-1, $0.9040$ mAUC)
- **SNR +10 dB:** $0.4677$ mAP ($+20.9\%$ relative gain over matched attention-pool; $+27.4\%$ over baseline)
- **SNR +5 dB:** $0.4328$ mAP ($+27.5\%$ relative gain over matched attention-pool; $+35.4\%$ over baseline)
- **SNR 0 dB:** $0.3828$ mAP ($+36.9\%$ relative gain over matched attention-pool; $+45.7\%$ over baseline)
- **SNR -5 dB:** $0.3182$ mAP ($+49.9\%$ relative gain over matched attention-pool; $+56.0\%$ over baseline)

---

## 7. Decision Threshold Sweeping & Calibration

Because multi-label audio contains extreme prior class sparsity ($P(y_c=1) \approx 0.0065$), standard $\tau = 0.50$ evaluation severely penalizes recall. Regularization (smoothing + Mixup) naturally compresses sigmoid logits:

| Threshold ($\tau$) | Operating Regime | No Aug: Precision | No Aug: Recall | No Aug: Micro-F1 | Mixup: Precision | Mixup: Recall | Mixup: Micro-F1 | Mixup: Macro-F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.10 | High Sensitivity | 0.3852 | 0.7691 | 0.5133 | 0.4011 | 0.7910 | 0.5323 | 0.4441 |
| 0.15 | Balanced Sensitive | 0.5287 | 0.7119 | 0.6068 | 0.5460 | 0.7307 | 0.6250 | 0.5192 |
| **0.20** | **Deployment ($\tau_{\text{eval}}$)** | **0.6066** | **0.6762** | **0.6395** | **0.6249** | **0.6889** | **0.6554** | **0.5439** |
| **0.25** | **Opt. Macro-F1 ($\tau^*$)** | 0.6598 | 0.6480 | 0.6539 | 0.6784 | 0.6555 | 0.6668 | **0.5454** |
| **0.30** | **Opt. Micro-F1 ($\tau^*$)** | 0.7008 | 0.6269 | 0.6618 | 0.7165 | 0.6284 | **0.6695** | 0.5375 |
| 0.35 | High Precision | 0.7291 | 0.6062 | 0.6620 | 0.7485 | 0.6052 | 0.6693 | 0.5304 |
| 0.40 | Conservative | 0.7578 | 0.5876 | 0.6619 | 0.7718 | 0.5837 | 0.6647 | 0.5188 |
| 0.50 | Default Uncalibrated | 0.8026 | 0.5508 | 0.6533 | 0.8182 | 0.5415 | 0.6517 | 0.4790 |
| 0.60 | High Confidence | 0.8385 | 0.5128 | 0.6364 | 0.8498 | 0.4953 | 0.6258 | 0.4260 |

Calibrating the decision threshold from $\tau=0.50$ down to $\tau=0.20$ recovers $\approx +14.7\%$ absolute recall ($0.5415 \to 0.6889$) and improves Macro-F1 from $0.4790$ to $0.5439$.

---

## 8. Implementation Code & Metadata References

The corresponding PyTorch modules, scripts, and evaluation metadata are organized as follows:

* **Dataset Loading & Preprocessing:** [`data/dataset.py`](file:///home/totallynotminh/Documents/IntroDL/data/dataset.py)
* **Data Augmentations:** [`data/augment.py`](file:///home/totallynotminh/Documents/IntroDL/data/augment.py)
  * `FrequencyMasking`, `TimeMasking`, `SpecAugment`
  * `mixup_samples`, `mixup_batch`, `Mixup`
  * `RandomColoredNoise` ($1/f^\alpha$ power-law filter)
  * `RandomReverberation` (Schroeder room impulse response)
* **Class-Balanced Sampler:** [`data/sampler.py`](file:///home/totallynotminh/Documents/IntroDL/data/sampler.py)
* **Training & Evaluation Scripts:**
  * [`scripts/train_classifier.py`](file:///home/totallynotminh/Documents/IntroDL/scripts/train_classifier.py)
  * [`scripts/eval_clean_checkpoint.py`](file:///home/totallynotminh/Documents/IntroDL/scripts/eval_clean_checkpoint.py)
  * [`scripts/eval_augmentation_comparison.py`](file:///home/totallynotminh/Documents/IntroDL/scripts/eval_augmentation_comparison.py)
  * [`scripts/eval_all_checkpoints.py`](file:///home/totallynotminh/Documents/IntroDL/scripts/eval_all_checkpoints.py)
  * [`scripts/inference.py`](file:///home/totallynotminh/Documents/IntroDL/scripts/inference.py)
* **Metadata & Benchmarks:**
  * [`metadata/augmentation_val_comparison.csv`](file:///home/totallynotminh/Documents/IntroDL/metadata/augmentation_val_comparison.csv): Clean validation metrics across ablations.
  * [`metadata/augmentation_noise_robustness.csv`](file:///home/totallynotminh/Documents/IntroDL/metadata/augmentation_noise_robustness.csv): Evaluation across ACE noise conditions.
  * [`metadata/augmentation_threshold_sweep.csv`](file:///home/totallynotminh/Documents/IntroDL/metadata/augmentation_threshold_sweep.csv): Fine-grained threshold sweep grid.
  * [`metadata/all_undefended_architectures_noise_robustness.csv`](file:///home/totallynotminh/Documents/IntroDL/metadata/all_undefended_architectures_noise_robustness.csv): Undefended architectural baseline robustness.
  * [`metadata/final_model_evaluation_metrics.csv`](file:///home/totallynotminh/Documents/IntroDL/metadata/final_model_evaluation_metrics.csv): Evaluation metrics for the proposed robust model.
