#!/usr/bin/env python3
"""
Evaluate 6 new checkpoints across FSD50K Benchmark conditions:
1. ViT-Tiny + Attention Pooling
2. ViT-Tiny + GAP Pooling
3. ViT-Tiny + GAP + Max Pooling
4. ResNet-34 Baseline
5. ViT-Tiny (Full Aug + Colored Noise + Reverb)
6. ViT-Tiny (Mixup + Colored Noise + Reverb)

Conditions:
- In-Distribution Val (4,170 clips)
- Clean Eval (10,231 clips)
- Out-of-Distribution Noisy Eval: snr_10, snr_5, snr_0, snr_neg5 (10,231 clips each)
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path

# Cap threads
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

from models.encoder import ASTEncoder
from models.resnet_encoder import ResNetEncoder
from data.dataset import FSD50KDataset
from metrics.classification import compute_average_precision, compute_roc_auc


# =========================================================================
# Model Architecture Builders
# =========================================================================
class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        w = self.score(x).softmax(dim=1)
        return (w * x).sum(dim=1)


class ASTClassifier(nn.Module):
    def __init__(self, tok_dim=192, num_classes=200, pooling=None):
        super().__init__()
        self.encoder = ASTEncoder(
            tok_dim=tok_dim,
            c_in=1,
            overlap=6,
            patch_size=16,
            size=(128, 1000),
            num_head=3,
            num_layer=12,
            use_dino=True,
            pretrained_dino=False,
        )
        self.pooling = pooling
        if pooling == "attention":
            self.pool = AttentionPool(tok_dim)

        head_in = 2 * tok_dim if pooling == "gap_max" else tok_dim
        self.head_norm = nn.LayerNorm(head_in, eps=1e-6)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(head_in, num_classes)

    def forward(self, x):
        tokens = self.encoder(x)  # (B, 2 + 1188, tok_dim)
        if self.pooling == "gap":
            pooled = tokens[:, 2:].mean(dim=1)
        elif self.pooling == "gap_max":
            patch_tokens = tokens[:, 2:]
            pooled = torch.cat([patch_tokens.mean(dim=1), patch_tokens.amax(dim=1)], dim=-1)
        elif self.pooling == "attention":
            pooled = self.pool(tokens[:, 2:])
        else:  # CLS + DIST dual token
            pooled = (tokens[:, 0] + tokens[:, 1]) / 2.0

        pooled = self.head_norm(pooled)
        pooled = self.dropout(pooled)
        return self.head(pooled)


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.encoder = ResNetEncoder(model_name="resnet34", pretrained=False, in_channels=1)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.dropout(feat)
        return self.head(feat)


MODEL_SPECS = [
    {
        "id": "vit_attn_pool",
        "name": "ViT-Tiny (Attention Pool)",
        "file": "attention.zip",
        "type": "ast",
        "pooling": "attention",
    },
    {
        "id": "vit_gap_pool",
        "name": "ViT-Tiny (GAP Pool)",
        "file": "gap.zip",
        "type": "ast",
        "pooling": "gap",
    },
    {
        "id": "vit_gap_max_pool",
        "name": "ViT-Tiny (GAP+Max Pool)",
        "file": "gap-max.zip",
        "type": "ast",
        "pooling": "gap_max",
    },
    {
        "id": "resnet34_baseline",
        "name": "ResNet-34 Baseline",
        "file": "resnet34.zip",
        "type": "resnet34",
        "pooling": None,
    },
    {
        "id": "vit_full_noise_reverb",
        "name": "ViT-Tiny (Full Aug+Noise+Reverb)",
        "file": "full-colored-noise.zip",
        "type": "ast",
        "pooling": None,
    },
    {
        "id": "vit_mixup_noise_reverb",
        "name": "ViT-Tiny (Mixup+Noise+Reverb)",
        "file": "mixup-colored-noise.zip",
        "type": "ast",
        "pooling": None,
    },
]


# =========================================================================
# Audio Dataset
# =========================================================================
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

        # Center pad or crop
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


def find_audio_path(path_str: str, noisy_data_dir: Path, clean_eval_dir: Path) -> str | None:
    if os.path.isfile(path_str):
        return path_str
    
    cand = PROJECT_ROOT / path_str
    if cand.is_file():
        return str(cand)
    
    fname = Path(path_str).name
    # Check if under noisy condition folder
    parts = Path(path_str).parts
    if len(parts) >= 2 and parts[-2].startswith("snr_"):
        cond = parts[-2]
        cand2 = noisy_data_dir / cond / fname
        if cand2.is_file():
            return str(cand2)
    
    # Check clean eval dir
    cand3 = clean_eval_dir / fname
    if cand3.is_file():
        return str(cand3)

    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate new checkpoints")
    parser.add_argument("--data-root", type=str, default="data/fsd50k")
    parser.add_argument("--noisy-data-dir", type=str, default="data/evaluation_fsd50k")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--metadata-csv", type=str, default="metadata/fsd50k_evaluation_metadata.csv")
    parser.add_argument("--output-dir", type=str, default="metadata")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--skip-val", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load taxonomy
    data_root = Path(args.data_root)
    vocab_file = data_root / "FSD50K.ground_truth" / "vocabulary.csv"
    if not vocab_file.is_file():
        vocab_file = PROJECT_ROOT / "data" / "fsd50k" / "FSD50K.ground_truth" / "vocabulary.csv"
    
    class_to_idx = {}
    with open(vocab_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                class_to_idx[str(row[1]).strip()] = int(row[0])
    print(f"Loaded taxonomy with {len(class_to_idx)} classes.")

    # Load models
    ckpt_dir = Path(args.checkpoint_dir)
    models = {}
    for spec in MODEL_SPECS:
        path = ckpt_dir / spec["file"]
        if not path.is_file():
            print(f"Warning: {path} not found, skipping {spec['id']}")
            continue
        print(f"Loading {spec['name']} from {path.name}...")
        if spec["type"] == "resnet34":
            model = ResNetClassifier(num_classes=200)
        else:
            model = ASTClassifier(tok_dim=192, num_classes=200, pooling=spec["pooling"])
        
        ckpt = torch.load(path, map_location=device)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        model.to(device)
        model.eval()
        models[spec["id"]] = (spec, model)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    # -------------------------------------------------------------
    # 1. In-Distribution Validation Set (4,170 samples)
    # -------------------------------------------------------------
    if not args.skip_val:
        print("\n" + "=" * 80)
        print("Evaluating on In-Distribution FSD50K Validation Split (4,170 clips)")
        print("=" * 80)
        val_ds = FSD50KDataset(root_dir=str(data_root), split="val", num_classes=200, use_augment=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        val_probs = {mid: [] for mid in models}
        val_targets = []

        with torch.no_grad():
            for specs, tgts in tqdm(val_loader, desc="Validation"):
                specs = specs.to(device, non_blocking=True)
                val_targets.append(tgts.numpy())
                for mid, (spec, m) in models.items():
                    logits = m(specs)
                    val_probs[mid].append(torch.sigmoid(logits).cpu().numpy())

        val_targets = np.concatenate(val_targets, axis=0)
        for mid in models:
            probs = np.concatenate(val_probs[mid], axis=0)
            mets = compute_metrics(probs, val_targets, threshold=args.threshold)
            mets_50 = compute_metrics(probs, val_targets, threshold=0.50)
            res = {
                "condition": "val",
                "model_id": mid,
                "model_name": models[mid][0]["name"],
                "samples": len(val_targets),
                "mAP": mets["mAP"],
                "mAUC": mets["mAUC"],
                "micro_f1_t20": mets["micro_f1"],
                "macro_f1_t20": mets["macro_f1"],
                "micro_f1_t50": mets_50["micro_f1"],
                "macro_f1_t50": mets_50["macro_f1"],
                "top1_hit": mets["top1_hit"],
                "top5_hit": mets["top5_hit"],
            }
            all_results.append(res)
            print(f"[{mid:<22}] val: mAP={res['mAP']:.4f} | mAUC={res['mAUC']:.4f} | Micro-F1(0.2)={res['micro_f1_t20']:.4f} | Top1={res['top1_hit']*100:.2f}%")

    # -------------------------------------------------------------
    # 2. Benchmark Conditions: Clean + 4 Noisy (snr_10, snr_5, snr_0, snr_neg5)
    # -------------------------------------------------------------
    meta_df = pd.read_csv(args.metadata_csv)
    noisy_data_dir = Path(args.noisy_data_dir)
    clean_eval_dir = data_root / "FSD50K.eval_audio_16k"
    if not clean_eval_dir.is_dir():
        clean_eval_dir = data_root / "FSD50K.eval_audio"

    # Evaluate Clean
    eval_csv = data_root / "FSD50K.ground_truth" / "eval.csv"
    if eval_csv.is_file():
        print("\n" + "=" * 80)
        print("Evaluating condition: clean (FSD50K eval split, 10,231 clips)")
        print("=" * 80)
        clean_df = pd.read_csv(eval_csv)
        clean_paths = []
        clean_labels = []
        for _, row in clean_df.iterrows():
            fname = f"{row['fname']}.wav"
            fpath = clean_eval_dir / fname
            if fpath.is_file():
                clean_paths.append(str(fpath))
                clean_labels.append(str(row["labels"]))
        
        if clean_paths:
            clean_ds = EvalAudioDataset(clean_paths, clean_labels, class_to_idx)
            clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            clean_probs = {mid: [] for mid in models}
            clean_targets = []

            with torch.no_grad():
                for specs, tgts in tqdm(clean_loader, desc="Clean Eval"):
                    specs = specs.to(device, non_blocking=True)
                    clean_targets.append(tgts.numpy())
                    for mid, (spec, m) in models.items():
                        logits = m(specs)
                        clean_probs[mid].append(torch.sigmoid(logits).cpu().numpy())

            clean_targets = np.concatenate(clean_targets, axis=0)
            for mid in models:
                probs = np.concatenate(clean_probs[mid], axis=0)
                mets = compute_metrics(probs, clean_targets, threshold=args.threshold)
                mets_50 = compute_metrics(probs, clean_targets, threshold=0.50)
                res = {
                    "condition": "clean",
                    "model_id": mid,
                    "model_name": models[mid][0]["name"],
                    "samples": len(clean_targets),
                    "mAP": mets["mAP"],
                    "mAUC": mets["mAUC"],
                    "micro_f1_t20": mets["micro_f1"],
                    "macro_f1_t20": mets["macro_f1"],
                    "micro_f1_t50": mets_50["micro_f1"],
                    "macro_f1_t50": mets_50["macro_f1"],
                    "top1_hit": mets["top1_hit"],
                    "top5_hit": mets["top5_hit"],
                }
                all_results.append(res)
                print(f"[{mid:<22}] clean: mAP={res['mAP']:.4f} | mAUC={res['mAUC']:.4f} | Micro-F1(0.2)={res['micro_f1_t20']:.4f} | Top1={res['top1_hit']*100:.2f}%")

    # Evaluate Noisy Conditions
    for cond in ["snr_10", "snr_5", "snr_0", "snr_neg5"]:
        print("\n" + "=" * 80)
        print(f"Evaluating condition: {cond} (10,231 clips)")
        print("=" * 80)
        cond_df = meta_df[meta_df["condition"] == cond]
        cond_paths = []
        cond_labels = []
        for _, row in cond_df.iterrows():
            p = find_audio_path(str(row["output_path"]), noisy_data_dir, clean_eval_dir)
            if p is not None:
                cond_paths.append(p)
                cond_labels.append(str(row["labels"]))

        if not cond_paths:
            print(f"No audio files found for {cond}, skipping.")
            continue

        cond_ds = EvalAudioDataset(cond_paths, cond_labels, class_to_idx)
        cond_loader = DataLoader(cond_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        cond_probs = {mid: [] for mid in models}
        cond_targets = []

        with torch.no_grad():
            for specs, tgts in tqdm(cond_loader, desc=f"Eval {cond}"):
                specs = specs.to(device, non_blocking=True)
                cond_targets.append(tgts.numpy())
                for mid, (spec, m) in models.items():
                    logits = m(specs)
                    cond_probs[mid].append(torch.sigmoid(logits).cpu().numpy())

        cond_targets = np.concatenate(cond_targets, axis=0)
        for mid in models:
            probs = np.concatenate(cond_probs[mid], axis=0)
            mets = compute_metrics(probs, cond_targets, threshold=args.threshold)
            mets_50 = compute_metrics(probs, cond_targets, threshold=0.50)
            res = {
                "condition": cond,
                "model_id": mid,
                "model_name": models[mid][0]["name"],
                "samples": len(cond_targets),
                "mAP": mets["mAP"],
                "mAUC": mets["mAUC"],
                "micro_f1_t20": mets["micro_f1"],
                "macro_f1_t20": mets["macro_f1"],
                "micro_f1_t50": mets_50["micro_f1"],
                "macro_f1_t50": mets_50["macro_f1"],
                "top1_hit": mets["top1_hit"],
                "top5_hit": mets["top5_hit"],
            }
            all_results.append(res)
            print(f"[{mid:<22}] {cond}: mAP={res['mAP']:.4f} | mAUC={res['mAUC']:.4f} | Micro-F1(0.2)={res['micro_f1_t20']:.4f} | Top1={res['top1_hit']*100:.2f}%")

    # Save summary
    res_df = pd.DataFrame(all_results)
    out_csv = out_dir / "new_checkpoints_noise_robustness.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")


if __name__ == "__main__":
    main()
