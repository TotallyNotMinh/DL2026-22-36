# Dataset Documentation (`DATA.md`)

This document specifies the datasets, splits, preprocessing pipelines, acoustic corruption protocols, and reproduction scripts utilized in the **Environmental Sound Recognition Under Unseen Conditions** project.

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
  * Noise scaling factor is calibrated to achieve the exact target SNR: $\sigma_{\text{noise}} = \frac{\text{RMS}(x)}{\text{RMS}(n) \cdot 10^{\text{SNR}/20}}$.
  * Four standardized SNR tiers are generated: $+10\text{ dB}$, $+5\text{ dB}$, $0\text{ dB}$, and $-5\text{ dB}$ ($40,924$ corrupted evaluation audio files manifest in `metadata/fsd50k_evaluation_metadata.csv`).

---

## 2. Dataset Splits & Structure

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

---

## 3. Audio Preprocessing Procedure

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

---

## 4. Acoustic Corruption & Benchmark Protocols

To measure model robustness under unseen real-world acoustic degradation without using external training data, our benchmark incorporates:

### 4.1 Pure Noise Benchmark (ACE Noise)
Evaluation audio clips ($10,231$ clips) are mixed with background noise at 4 calibrated SNR tiers:
$$\alpha = \sqrt{\frac{P_{\text{signal}}}{P_{\text{noise}} \cdot 10^{\text{SNR}/10}}}$$

* **`clean`**: Original FSD50K evaluation clips.
* **`snr_10`**: Foreground + Noise at $+10\text{ dB SNR}$ (mild background interference).
* **`snr_5`**: Foreground + Noise at $+5\text{ dB SNR}$ (moderate interference).
* **`snr_0`**: Foreground + Noise at $0\text{ dB SNR}$ (equal signal and noise power).
* **`snr_neg5`**: Foreground + Noise at $-5\text{ dB SNR}$ (heavy noise corruption).

### 4.2 Algorithmic Colored Noise & Reverberation Augmentation
Implemented in `data/augment.py` for training without downloading external noise libraries:
* **`RandomColoredNoise`**: Continuous spectral slope filtering $S(f) \propto 1/f^\alpha$ ($\alpha \in [0.0, 2.0]$ for white, pink, and brown noise) + random frequency band-pass masks at $\text{SNR} \in [-5, 20]\text{ dB}$.
* **`RandomReverberation`**: Schroeder late reverberation using decaying Gaussian noise with decay times $T_{60} \in [0.15, 0.60]\text{ s}$ convolved via composite fast FFT padding.

---

## 5. Scripts Required for Reproduction

All preprocessing, training, and evaluation scripts are available in the repository:

* **Dataset Loading & Preprocessing:** [`data/dataset.py`](file:///home/totallynotminh/Documents/IntroDL/data/dataset.py)
* **Data Augmentations:** [`data/augment.py`](file:///home/totallynotminh/Documents/IntroDL/data/augment.py)
* **Class-Balanced Sampler:** [`data/sampler.py`](file:///home/totallynotminh/Documents/IntroDL/data/sampler.py)
* **Classifier Training Script:** [`scripts/train_classifier.py`](file:///home/totallynotminh/Documents/IntroDL/scripts/train_classifier.py)
* **Inference & Demo Script:** [`scripts/inference.py`](file:///home/totallynotminh/Documents/IntroDL/scripts/inference.py)
* **Benchmark Evaluation Engine:** [`scripts/eval_benchmark.py`](file:///home/totallynotminh/Documents/IntroDL/scripts/eval_benchmark.py)

---

## 6. Pretrained Weights & Checkpoint Links

* **Pretrained ViT-Tiny Checkpoint:** [Kaggle Model: knuckleizmad/vit-tiny-fsd50k-best-checkpoint](https://www.kaggle.com/models/knuckleizmad/vit-tiny-fsd50k-best-checkpoint)
* **Meta DeiT ViT-Tiny Weights:** Automatically fetched via `torch.hub` (`deit_tiny_distilled_patch16_224`)
* **Meta DeiT ViT-Small Weights:** Automatically fetched via `torch.hub` (`deit_small_distilled_patch16_224`)
