#!/usr/bin/env python3
"""
Evaluate a specific checkpoint on FSD50K Clean Evaluation Split (10,231 clips)
"""

import os
import sys
import csv
import argparse
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import torch
torch.set_num_threads(8)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.classifier import Classifer
from metrics.classification import compute_average_precision, compute_roc_auc


class EvalAudioDataset(Dataset):
    def __init__(
        self,
        file_paths: list[str],
        labels_list: list[str],
        class_to_idx: dict[str, int],
        sample_rate: int = 16000,
        duration_sec: float = 10.0,
        n_mels: int = 128,
        n_fft: int = 1024,
        win_length: int = 400,
        hop_length: int = 160,
        target_frames: int = 1000,
        norm_mean: float = -4.2677393,
        norm_std: float = 4.5689974,
    ):
        self.file_paths = file_paths
        self.labels_list = labels_list
        self.class_to_idx = class_to_idx
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * duration_sec)
        self.target_frames = target_frames
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=2.0,
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, sr = torchaudio.load(file_path)
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if sr != self.sample_rate:
            waveform = AF.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)

        length = waveform.shape[-1]
        if length > self.target_len:
            start = (length - self.target_len) // 2
            waveform = waveform[..., start : start + self.target_len]
        elif length < self.target_len:
            pad = self.target_len - length
            waveform = F.pad(waveform, (0, pad))

        mel = self.mel_transform(waveform)
        log_mel = torch.log(mel.clamp(min=1e-6))
        log_mel = (log_mel - self.norm_mean) / (self.norm_std * 2.0)

        if log_mel.shape[-1] > self.target_frames:
            log_mel = log_mel[..., : self.target_frames]
        elif log_mel.shape[-1] < self.target_frames:
            log_mel = F.pad(log_mel, (0, self.target_frames - log_mel.shape[-1]))

        target = torch.zeros(200, dtype=torch.float32)
        raw_labels = str(self.labels_list[idx])
        sep = ";" if ";" in raw_labels else ","
        for l in raw_labels.split(sep):
            l = l.strip().replace('"', "")
            if l in self.class_to_idx:
                target[self.class_to_idx[l]] = 1.0

        return log_mel, target


def compute_metrics(probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> dict:
    N, num_classes = probs.shape
    ap_list = []
    auc_list = []
    for c in range(num_classes):
        ap = compute_average_precision(targets[:, c], probs[:, c])
        if ap is not None:
            ap_list.append(ap)
        auc = compute_roc_auc(targets[:, c], probs[:, c])
        if auc is not None:
            auc_list.append(auc)

    mean_ap = float(np.mean(ap_list)) if ap_list else 0.0
    mean_auc = float(np.mean(auc_list)) if auc_list else 0.0

    bin_preds = (probs >= threshold).astype(int)
    bin_targets = targets.astype(int)

    tp = np.sum((bin_preds == 1) & (bin_targets == 1))
    fp = np.sum((bin_preds == 1) & (bin_targets == 0))
    fn = np.sum((bin_preds == 0) & (bin_targets == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    micro_f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)

    class_f1s = []
    for c in range(num_classes):
        c_tp = np.sum((bin_preds[:, c] == 1) & (bin_targets[:, c] == 1))
        c_fp = np.sum((bin_preds[:, c] == 1) & (bin_targets[:, c] == 0))
        c_fn = np.sum((bin_preds[:, c] == 0) & (bin_targets[:, c] == 1))
        denom = 2 * c_tp + c_fp + c_fn
        if denom > 0:
            class_f1s.append((2 * c_tp) / denom)
        elif np.sum(bin_targets[:, c]) > 0:
            class_f1s.append(0.0)
    macro_f1 = float(np.mean(class_f1s)) if class_f1s else 0.0

    top1_indices = np.argmax(probs, axis=1)
    top1_hit = float(np.mean([bin_targets[i, top1_indices[i]] == 1 for i in range(N)]))

    top5_indices = np.argsort(-probs, axis=1)[:, :5]
    top5_hit = float(np.mean([np.any(bin_targets[i, top5_indices[i]] == 1) for i in range(N)]))

    return {
        "mAP": mean_ap,
        "mAUC": mean_auc,
        "micro_f1": float(micro_f1),
        "macro_f1": macro_f1,
        "precision": float(precision),
        "recall": float(recall),
        "top1_hit": top1_hit,
        "top5_hit": top5_hit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vit-tiny.pth", help="Path to checkpoint (.pth)")
    parser.add_argument("--data-path", type=str, default="data/fsd50k", help="Path to FSD50K dataset root directory")
    parser.add_argument("--encoder", type=str, default="ast", choices=["ast", "resnet18", "resnet34", "efficientnet_b0", "efficientnet_b4"])
    parser.add_argument("--arch", type=str, default="tiny", choices=["tiny", "small", "base"])
    parser.add_argument("--pooling", type=str, default="auto", choices=["auto", "cls_dist", "gap", "gap_max", "attention"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = Path(args.data_path)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    vocab_file = data_root / "FSD50K.ground_truth" / "vocabulary.csv"
    if not vocab_file.is_file():
        vocab_file = data_root / "vocabulary.csv"

    class_to_idx = {}
    with open(vocab_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                class_to_idx[str(row[1]).strip()] = int(row[0])

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    print(f"Loading {args.encoder if args.encoder != 'ast' else args.arch} from {ckpt_path.name}...")

    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    if any(k.startswith("features.") for k in sd.keys()):
        sd = {("encoder." + k if k.startswith("features.") else k): v for k, v in sd.items()}

    pooling = args.pooling
    if pooling == "auto":
        if any(k.startswith("pool.") for k in sd.keys()):
            pooling = "attention"
        elif "head_norm.weight" in sd and sd["head_norm.weight"].shape[0] == 384 and args.arch == "tiny":
            pooling = "gap_max"
        else:
            pooling = "cls_dist"
    print(f"Resolved pooling mode: {pooling}")

    if args.encoder == "ast":
        tok_dim = 192 if args.arch == "tiny" else (384 if args.arch == "small" else 768)
        num_head = 3 if args.arch == "tiny" else (6 if args.arch == "small" else 12)
        model = Classifer(encoder_type="ast", tok_dim=tok_dim, num_head=num_head, num_layer=12, num_classes=200, pooling=pooling, pretrained_dino=False).to(device)
    else:
        model = Classifer(encoder_type=args.encoder, num_classes=200, pretrained_dino=False).to(device)

    model.load_state_dict(sd, strict=True)
    model.eval()

    eval_csv = data_root / "FSD50K.ground_truth" / "eval.csv"
    clean_eval_dir = data_root / "FSD50K.eval_audio_16k"
    clean_df = pd.read_csv(eval_csv)
    
    clean_paths = []
    clean_labels = []
    for _, row in clean_df.iterrows():
        fpath = clean_eval_dir / f"{row['fname']}.wav"
        if fpath.is_file():
            clean_paths.append(str(fpath))
            clean_labels.append(str(row["labels"]))

    print(f"Evaluating {len(clean_paths)} clean test clips for {ckpt_path.name}...")
    clean_ds = EvalAudioDataset(clean_paths, clean_labels, class_to_idx)
    clean_loader = DataLoader(clean_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for specs, tgts in tqdm(clean_loader, desc=f"Eval {ckpt_path.name}"):
            specs = specs.to(device, non_blocking=True)
            all_targets.append(tgts.numpy())
            logits = model(specs)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    mets_20 = compute_metrics(probs, targets, threshold=0.20)
    mets_50 = compute_metrics(probs, targets, threshold=0.50)

    print("\n" + "=" * 80)
    print(f"Clean Evaluation Results for {ckpt_path.name} (10,231 clips):")
    print("=" * 80)
    print(f"mAP:                  {mets_20['mAP']:.4f}")
    print(f"mAUC:                 {mets_20['mAUC']:.4f}")
    print(f"Micro-F1 (tau=0.20):  {mets_20['micro_f1']:.4f}")
    print(f"Macro-F1 (tau=0.20):  {mets_20['macro_f1']:.4f}")
    print(f"Micro-F1 (tau=0.50):  {mets_50['micro_f1']:.4f}")
    print(f"Macro-F1 (tau=0.50):  {mets_50['macro_f1']:.4f}")
    print(f"Precision (tau=0.20): {mets_20['precision']:.4f}")
    print(f"Recall (tau=0.20):    {mets_20['recall']:.4f}")
    print(f"Top-1 Hit:            {mets_20['top1_hit']*100:.2f}%")
    print(f"Top-5 Hit:            {mets_20['top5_hit']*100:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
