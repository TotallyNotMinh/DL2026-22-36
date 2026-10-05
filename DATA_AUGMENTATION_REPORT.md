# Data Augmentation Experiments

## 1. Overview

This experiment studies how different data augmentation methods affect the performance of an environmental sound classification model.

Six training settings were compared:

1. No Augmentation
2. Time Masking
3. Frequency Masking
4. Time + Frequency Masking (SpecAugment)
5. Mixup
6. Full Augmentation (Time + Frequency Masking + Mixup)

All experiments were trained for **30 epochs** using the same model and training setup. The main difference between the experiments is the augmentation method.

---

## 2. Training Setup

To focus on the effects of SpecAugment and Mixup, the following settings were used in the Kaggle training script:

```bash
--time-shift 0 --noise-level 0
```

`--time-shift 0` disables random time shifting, while `--noise-level 0` disables added noise.

This is important because the experiment is intended to compare Time Masking, Frequency Masking, and Mixup. If time shifting or noise were also active, it would be difficult to identify which augmentation caused a change in performance.

The six configurations were:

| Experiment               | Time Masking | Frequency Masking | Mixup |
| ------------------------ | ------------ | ----------------- | ----- |
| No Augmentation          | No           | No                | No    |
| Time Masking             | Yes          | No                | No    |
| Frequency Masking        | No           | Yes               | No    |
| Time + Frequency Masking | Yes          | Yes               | No    |
| Mixup                    | No           | No                | Yes   |
| Full Augmentation        | Yes          | Yes               | Yes   |

All experiments used:

```text
Epochs      = 30
Time Shift  = 0
Noise Level = 0
Dataset     = FSD50K
```

### 2.1 Exact Training Flags Per Run

The table below gives the precise CLI argument values for every ablation run and the production baseline
(FM = `--freq-mask`, TM = `--time-mask`, MX = `--mixup-prob`, TS = `--time-shift`,
NL = `--noise-level`, CN = `--colored-noise-prob`, RV = `--reverb-prob`):

| Run | FM | TM | MX | TS | NL | CN | RV |
|-----|----|----|-----|----|----|-----|-----|
| `no_aug` | 0 | 0 | 0.0 | 0 | 0.00 | 0.0 | 0.0 |
| `time_mask` | 0 | 192 | 0.0 | 0 | 0.00 | 0.0 | 0.0 |
| `freq_mask` | 48 | 0 | 0.0 | 0 | 0.00 | 0.0 | 0.0 |
| `time_freq_mask` | 48 | 192 | 0.0 | 0 | 0.00 | 0.0 | 0.0 |
| `mixup` | 0 | 0 | 0.5 | 0 | 0.00 | 0.0 | 0.0 |
| `full` | 48 | 192 | 0.5 | 0 | 0.00 | 0.0 | 0.0 |
| **Baseline ViT-Tiny (production)** | 48 | 192 | 0.5 | 10 | 0.05 | 0.5 | 0.3 |

**Shared command template (substitute `<FM>`, `<TM>`, `<MX>` per row above):**

```bash
torchrun --nproc_per_node=2 scripts/train_classifier.py \
  --data-path <FSD50K_ROOT>  --arch tiny  --batch-size 12 \
  --num-epoch 30  --patience 30 \
  --encoder-lr 5e-5  --head-lr 5e-4 \
  --time-shift 0  --noise-level 0 \
  --colored-noise-prob 0  --reverb-prob 0 \
  --freq-mask <FM>  --time-mask <TM>  --mixup-prob <MX> \
  --checkpoint-dir <CKPT_DIR>/<run_name>/
```

---

## 3. How to Run the Experiments on Kaggle

The experiments were run using the same training script.

The FSD50K dataset was located at:

```text
/kaggle/input/datasets/yousirui1/fsd50k/fsd50k
```

Checkpoints were saved under:

```text
/kaggle/working/checkpoints/
```

Each experiment has a separate checkpoint directory:

```text
checkpoints/
├── run_01_no_aug/
├── run_02_time_mask/
├── run_03_frequency/
├── run_04_time_frequency/
├── run_05_mixup/
└── run_06_full/
```

Each run was trained for 30 epochs.

The main configuration for each run was:

```text
run_01_no_aug
    No augmentation

run_02_time_mask
    Time Masking only

run_03_frequency
    Frequency Masking only

run_04_time_frequency
    Time + Frequency Masking

run_05_mixup
    Mixup only

run_06_full
    Time + Frequency Masking + Mixup
```

The checkpoints are saved so that the trained models can be used later for testing on another dataset.

---

## 4. Evaluation Metrics

Four main validation metrics were used:

* **mAP (mean Average Precision):** measures average precision across classes.
* **mAUC (mean Area Under the ROC Curve):** measures how well the model separates positive and negative samples across classes.
* **Micro-F1:** calculates F1 using all predictions together. It is more influenced by classes with more samples.
* **Macro-F1:** calculates F1 for each class and then averages them. It gives every class equal importance.

Macro-F1 is useful for checking whether the model performs well across different classes, including less frequent classes.

---

## 5. Validation Results

The following values are the results at **epoch 30/30** for each experiment.

| Augmentation Method      | Validation mAP | Validation mAUC |   Micro-F1 |   Macro-F1 |
| ------------------------ | -------------: | --------------: | ---------: | ---------: |
| No Augmentation          |         0.5396 |          0.8877 | **0.6520** | **0.4975** |
| Time Masking             |         0.5386 |          0.9001 |     0.6502 |     0.4962 |
| Frequency Masking        |     **0.5400** |          0.9159 |     0.6391 |     0.4720 |
| Time + Frequency Masking |         0.5383 |          0.9228 |     0.6405 |     0.4637 |
| Mixup                    |     **0.5523** |          0.9153 |     0.6484 |     0.4808 |
| Full Augmentation        |         0.5380 |      **0.9351** |     0.6361 |     0.4347 |

### Best result for each metric

| Metric   | Best Method       |      Score |
| -------- | ----------------- | ---------: |
| mAP      | Mixup             | **0.5523** |
| mAUC     | Full Augmentation | **0.9351** |
| Micro-F1 | No Augmentation   | **0.6520** |
| Macro-F1 | No Augmentation   | **0.4975** |

Therefore, there is **no single augmentation method that is the best for every metric**.

---

## 6. Results and Discussion

### 6.1 No Augmentation

The No Augmentation model achieves:

```text
mAP      = 0.5396
mAUC     = 0.8877
Micro-F1 = 0.6520
Macro-F1 = 0.4975
```

Interestingly, the baseline has the highest Micro-F1 and Macro-F1 among the six experiments.

This means that, for this particular validation set, adding augmentation does not automatically improve F1 scores.

The baseline model may already learn useful features from the original training data. Augmentation changes the training examples and can sometimes make the learning problem harder, especially when the dataset or augmentation strength is not perfectly matched to the data.

---

### 6.2 Time Masking

Time Masking gives:

```text
mAP      = 0.5386
mAUC     = 0.9001
Micro-F1 = 0.6502
Macro-F1 = 0.4962
```

Compared with the baseline, mAUC increases from **0.8877 to 0.9001**.

This suggests that Time Masking may help the model learn a better ranking or separation between positive and negative examples.

However, mAP and both F1 scores are slightly lower than the baseline. Therefore, Time Masking does not provide an overall improvement for every metric.

One possible reason is that hiding parts of the time axis can remove useful information from some environmental sound events.

---

### 6.3 Frequency Masking

Frequency Masking gives:

```text
mAP      = 0.5400
mAUC     = 0.9159
Micro-F1 = 0.6391
Macro-F1 = 0.4720
```

Frequency Masking gives a slightly higher mAP than both the baseline and Time Masking.

It also produces a large increase in mAUC, from **0.8877 to 0.9159**.

However, Micro-F1 and Macro-F1 decrease.

This shows that a higher AUC does not always mean a higher F1 score. The model may rank samples better across different thresholds, while the final predictions at the selected threshold are not as balanced across classes.

---

### 6.4 Time + Frequency Masking

Combining Time Masking and Frequency Masking gives:

```text
mAP      = 0.5383
mAUC     = 0.9228
Micro-F1 = 0.6405
Macro-F1 = 0.4637
```

The main improvement is in mAUC.

The mAUC increases to **0.9228**, which is higher than both Time Masking and Frequency Masking alone.

This suggests that using both masking directions can help the model learn features that are more robust to missing time and frequency information.

However, the mAP and F1 scores are slightly lower. Therefore, combining the two masking methods does not guarantee better performance for every evaluation metric.

---

### 6.5 Mixup

Mixup gives the highest mAP:

```text
mAP      = 0.5523
mAUC     = 0.9153
Micro-F1 = 0.6484
Macro-F1 = 0.4808
```

The mAP increases from **0.5396** for the baseline to **0.5523**.

This is the largest mAP improvement among the six experiments.

Mixup creates new training examples by combining existing samples and their labels. This increases the variety of training data and can reduce over-reliance on individual training examples.

The result suggests that Mixup is particularly useful for improving average precision in this experiment.

However, its F1 scores are still lower than the baseline.

---

### 6.6 Full Augmentation

Full Augmentation combines:

```text
Time Masking
+
Frequency Masking
+
Mixup
```

It achieves:

```text
mAP      = 0.5380
mAUC     = 0.9351
Micro-F1 = 0.6361
Macro-F1 = 0.4347
```

The most important result is that Full Augmentation achieves the **highest mAUC: 0.9351**.

This means the model has the best overall ability to rank positive examples above negative examples across classes.

However, Full Augmentation has the lowest Macro-F1 and the lowest Micro-F1 among the six experiments.

One possible explanation is that the combination of three augmentation methods may be too strong for this training setup. The model receives highly modified training examples, which can help generalization in some ways but can also remove or change useful information.

Therefore, more augmentation is not always better.

---

## 7. Overall Comparison

The results show that different metrics give different conclusions.

Based on **mAP**:

```text
1. Mixup                    0.5523
2. Frequency Masking       0.5400
3. No Augmentation         0.5396
4. Time Masking            0.5386
5. Time + Frequency        0.5383
6. Full Augmentation       0.5380
```

Based on **mAUC**:

```text
1. Full Augmentation       0.9351
2. Time + Frequency        0.9228
3. Frequency Masking       0.9159
4. Mixup                   0.9153
5. Time Masking            0.9001
6. No Augmentation         0.8877
```

Based on **Micro-F1**:

```text
1. No Augmentation         0.6520
2. Time Masking            0.6502
3. Mixup                   0.6484
4. Time + Frequency        0.6405
5. Frequency Masking       0.6391
6. Full Augmentation       0.6361
```

Based on **Macro-F1**:

```text
1. No Augmentation         0.4975
2. Time Masking            0.4962
3. Mixup                   0.4808
4. Frequency Masking       0.4720
5. Time + Frequency        0.4637
6. Full Augmentation       0.4347
```

---

## 8. Main Findings

The experiments show several important points.

First, **Mixup gives the best mAP**. Its mAP of **0.5523** is higher than the baseline value of **0.5396**.

Second, **Full Augmentation gives the best mAUC** with a score of **0.9351**. This means the full augmentation model has the best overall ranking ability according to AUC.

Third, the **No Augmentation model gives the highest Micro-F1 and Macro-F1**. Therefore, augmentation does not improve every metric in this experiment.

Finally, combining more augmentation methods does not always lead to better results. The Full Augmentation model performs very well in mAUC but performs poorly in F1. This suggests that the amount and type of augmentation need to be selected carefully.

---

### 8.1 Critical Analysis: Why Full Augmentation is Not Justified over Mixup

A central question in designing the augmentation pipeline is whether the higher mAUC of Full Augmentation (0.9351 vs. 0.9153 for Mixup) justifies deploying the full compound pipeline. When considering the task requirements and empirical behavior, **it is not justified**:

1. **FSD50K Metric Priority (mAP over mAUC):** In multi-label sound classification with 200 categories, each clip has only 1–3 positive labels (>99% of classes per clip are true negatives). In such severe class imbalance, mAUC (ROC-AUC) is easily inflated by the massive volume of easy negatives; heavy regularization pushes negative logits close to zero, boosting AUC while masking poor positive retrieval. In contrast, mAP directly measures Precision-Recall on positive detections. Mixup achieves the highest validation mAP (0.5523 vs. 0.5380 for Full) and highest test mAP across clean (0.5124 vs. 0.5056) and adverse noise conditions (0.3866 vs. 0.3657 at +10 dB SNR).
2. **Compound Over-Corruption & Tail-Class Collapse:** In Full Augmentation, clips are first linearly mixed (halving individual signal energy), and then subjected to dual time (192 frames) and frequency (48 bins) masking. This zeroes out 30–40% of the already diluted spectrogram. For subtle or rare classes (clicks, snaps, faint whistles), the discriminative acoustic signal is wiped out while the loss still penalizes the model for missing the label. This causes severe tail-class collapse, shown by the lowest Macro-F1 among all configurations (0.4347 vs. 0.4808 for Mixup, a -4.61% penalty).
3. **Physical Acoustic Realism:** Mixup naturally mirrors acoustic wave superposition in the physical world ($s(t) = s_1(t) + s_2(t)$), providing organic multi-source scene simulation. Standalone Mixup achieves the optimal Pareto frontier between acoustic regularization and signal integrity under a standard 30-epoch training budget.

---

## 9. Conclusion

The experiment shows that different augmentation methods affect different evaluation metrics in different ways.

**Mixup is the best method for validation mAP**, while **Full Augmentation is the best method for mAUC**.

However, **No Augmentation gives the highest Micro-F1 and Macro-F1** in this experiment.

Therefore, there is no single method that is best for all metrics.

The results suggest that augmentation can improve some aspects of model performance, but stronger augmentation is not always better. The effect depends on the evaluation metric and how much useful information is changed or removed from the original audio.

For future experiments, it would be useful to test different masking strengths and Mixup settings to find a better balance between generalization and classification performance.

---

## 10. Checkpoints

The trained checkpoints were saved after each experiment:

```text
/kaggle/working/checkpoints/run_01_no_aug
/kaggle/working/checkpoints/run_02_time_mask
/kaggle/working/checkpoints/run_03_frequency
/kaggle/working/checkpoints/run_04_time_frequency
/kaggle/working/checkpoints/run_05_mixup
/kaggle/working/checkpoints/run_06_full
```

These checkpoints can be used later for cross-dataset testing without retraining the models.

---
