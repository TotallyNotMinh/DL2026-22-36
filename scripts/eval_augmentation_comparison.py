#!/usr/bin/env python3
"""
scripts/eval_augmentation_comparison.py

Comprehensive evaluation script to compare all 6 augmentation ablation checkpoints:
  1. no_aug:          No Augmentation
  2. time_mask:       Time Masking
  3. freq_mask:       Frequency Masking
  4. time_freq_mask:  Time + Frequency Masking (SpecAugment)
  5. mixup:           Mixup Only
  6. full:            Full Augmentation (Time + Freq + Mixup)

Designed to run seamlessly on both local machines and Kaggle environments:
  - Supports --data-path, --checkpoint-dir, --metadata-path, --output-dir
  - Auto-resolves checkpoint naming patterns (.zip, .pth, directory checkpoints)
  - Memory-safe: disables aggressive memory pinning by default to prevent OOM
  - Supports --skip-val to focus directly on out-of-distribution noise conditions
"""

import os
import sys
import gc
import json
import time
import argparse
from pathlib import Path

# Enforce thread caps
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T
import numpy as np
import pandas as pd
from tqdm import tqdm

torch.set_num_threads(min(8, torch.get_num_threads()))

from models.classifier import Classifer
from data.dataset import FSD50KDataset
from metrics.classification import (
    compute_average_precision,
    compute_roc_auc,
)

MODEL_NAMES = [
    ("no_aug", "No Augmentation"),
    ("time_mask", "Time Masking"),
    ("freq_mask", "Frequency Masking"),
    ("time_freq_mask", "Time + Freq Masking"),
    ("mixup", "Mixup Only"),
    ("full", "Full Augmentation"),
]


def matches_model(path_str: str, model_name: str) -> bool:
    fn = path_str.lower()
    if model_name == "no_aug":
        return "noaug" in fn or "no_aug" in fn or "no-aug" in fn or "run_01" in fn
    elif model_name == "time_freq_mask":
        return "time_freq" in fn or "time-freq" in fn or "run_04" in fn
    elif model_name == "time_mask":
        return ("time_mask" in fn or "time-mask" in fn or "run_02" in fn) and "freq" not in fn
    elif model_name == "freq_mask":
        return ("freq_mask" in fn or "freq-mask" in fn or "frequency" in fn or "run_03" in fn) and "time" not in fn
    elif model_name == "mixup":
        return "mixup" in fn or "run_05" in fn
    elif model_name == "full":
        return "full" in fn or "run_06" in fn
    return False


def find_checkpoint(base_dir: Path, model_name: str) -> Path | None:
    for p in sorted(base_dir.rglob("*")):
        if p.is_file() and p.suffix in [".pth", ".pt", ".bin", ".zip"] and matches_model(p.name, model_name):
            return p
        elif p.is_dir() and matches_model(p.name, model_name):
            if (p / "data.pkl").is_file():
                return p
            for sub in ["best_classifier.pth", "best_classifier", "model.pth"]:
                if (p / sub).exists():
                    return p / sub
    return None


def load_model(ckpt_path: Path, device: torch.device) -> Classifer:
    model = Classifer(
        tok_dim=192,
        num_classes=200,
        c_in=1,
        overlap=6,
        patch_size=16,
        size=(128, 1000),
        num_head=3,
        num_layer=12,
        pretrained_dino=False,
        use_cls_dist=True,
    ).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


class EvalAudioDataset(Dataset):
    """
    Unified dataset for evaluation clips (both clean and noisy ACE benchmark).
    """
    def __init__(
        self,
        file_paths: list[str],
        labels_list: list[str],
        class_to_idx: dict[str, int],
        sample_rate: int = 16000,
        duration_sec: float = 10.0,
        target_frames: int = 1000,
    ):
        self.file_paths = file_paths
        self.labels_list = labels_list
        self.class_to_idx = class_to_idx
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * duration_sec)
        self.target_frames = target_frames

        self.norm_mean = -4.2677393
        self.norm_std = 4.5689974

        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            win_length=400,
            hop_length=160,
            n_mels=128,
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

        # Target multi-hot
        target = torch.zeros(200, dtype=torch.float32)
        raw_labels = str(self.labels_list[idx])
        sep = ";" if ";" in raw_labels else ","
        for l in raw_labels.split(sep):
            l = l.strip().replace('"', "")
            if l in self.class_to_idx:
                target[self.class_to_idx[l]] = 1.0

        return log_mel, target


def compute_metrics_from_probs(probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> dict:
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 6 Augmentation Ablation Models")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/fsd50k",
        help="Path to FSD50K dataset root (default: data/fsd50k or /kaggle/input/datasets/yousirui1/fsd50k/fsd50k)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing trained model checkpoints (default: checkpoints or /kaggle/working/checkpoints)",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="metadata/fsd50k_evaluation_metadata.csv",
        help="Path to evaluation noise metadata manifest CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="metadata",
        help="Directory to save output CSV and JSON benchmark results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="DataLoader batch size (default: 64)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default: 4)",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=False,
        help="Enable pin_memory in DataLoader (default: False for RAM safety)",
    )
    parser.add_argument(
        "--skip-val",
        action="store_true",
        default=False,
        help="Skip Part A (Validation Split) and execute only Part B (Noise Benchmark)",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="clean,snr_10,snr_5,snr_0,snr_neg5",
        help="Comma-separated evaluation conditions to test",
    )
    parser.add_argument(
        "--noisy-data-dir",
        type=str,
        default=None,
        help="Directory containing synthesized noisy evaluation audio (e.g. data/evaluation_fsd50k or /kaggle/input/.../evaluation_fsd50k)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device (cuda or cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 80)
    print("ASTATIC BENCHMARK: 6-WAY AUGMENTATION ABLATION EVALUATION")
    print("=" * 80)
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Data Root     : {args.data_path}")
    print(f"Noisy Dir     : {args.noisy_data_dir or 'Auto-discover'}")
    print(f"Checkpoint Dir: {args.checkpoint_dir}")
    print(f"Metadata Path : {args.metadata_path}")
    print(f"Output Dir    : {args.output_dir}")
    print(f"Batch Size    : {args.batch_size}")
    print(f"Num Workers   : {args.num_workers}")
    print(f"Pin Memory    : {args.pin_memory}")
    print("=" * 80)

    # 1. Load taxonomy / vocabulary
    data_root = Path(args.data_path)
    vocab_candidates = [
        data_root / "FSD50K.ground_truth" / "vocabulary.csv",
        PROJECT_ROOT / "data" / "fsd50k" / "FSD50K.ground_truth" / "vocabulary.csv",
        PROJECT_ROOT / "config" / "fsd50k_200_classes.json",
    ]

    class_to_idx = {}
    idx_to_class = {}

    for vc in vocab_candidates:
        if vc.is_file() and vc.suffix == ".csv":
            df_v = pd.read_csv(vc, header=None)
            for _, row in df_v.iterrows():
                cid = int(row[0])
                cname = str(row[1]).strip()
                class_to_idx[cname] = cid
                idx_to_class[cid] = cname
            break
        elif vc.is_file() and vc.suffix == ".json":
            with open(vc, "r", encoding="utf-8") as f:
                c200 = json.load(f)
            for k, v in c200.items():
                cid = v["class_id"]
                class_to_idx[k] = cid
                idx_to_class[cid] = k
            break

    print(f"Loaded {len(class_to_idx)} classes into taxonomy.")

    # 2. Locate and load model checkpoints
    ckpt_dir = Path(args.checkpoint_dir)
    loaded_models = {}
    model_labels = {}

    for name, label in MODEL_NAMES:
        ckpt_path = find_checkpoint(ckpt_dir, name)
        if ckpt_path is None:
            print(f"  [!] Checkpoint for {name} ({label}) NOT found in {ckpt_dir}, skipping.")
            continue
        print(f"  • Loading {name:<15} ({label}) from {ckpt_path.name}...", end="", flush=True)
        m = load_model(ckpt_path, device)
        loaded_models[name] = m
        model_labels[name] = label
        print(" loaded.")

    if not loaded_models:
        raise RuntimeError(f"No valid checkpoints found in {ckpt_dir}!")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # PART A: In-Distribution Validation Set
    # -------------------------------------------------------------
    if not args.skip_val:
        print("\n" + "=" * 80)
        print("PART A: FSD50K Validation Split Inference (4,170 clips)")
        print("=" * 80)

        val_dataset = FSD50KDataset(
            root_dir=str(data_root),
            split="val",
            num_classes=200,
            use_augment=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

        val_probs = {name: [] for name in loaded_models}
        val_targets = []

        with torch.no_grad():
            for specs, tgts in tqdm(val_loader, desc="Validating"):
                specs = specs.to(device, non_blocking=True)
                val_targets.append(tgts.numpy())
                for name, model in loaded_models.items():
                    logits = model(specs)
                    val_probs[name].append(torch.sigmoid(logits).cpu().numpy())

        val_targets = np.concatenate(val_targets, axis=0)
        for name in val_probs:
            val_probs[name] = np.concatenate(val_probs[name], axis=0)

        val_summary = []
        for name in loaded_models:
            probs = val_probs[name]
            mets = compute_metrics_from_probs(probs, val_targets, threshold=0.50)

            best_macro_f1 = 0.0
            best_macro_t = 0.50
            best_micro_f1 = 0.0
            best_micro_t = 0.50

            for t in np.linspace(0.05, 0.90, 18):
                sm = compute_metrics_from_probs(probs, val_targets, threshold=float(t))
                if sm["macro_f1"] > best_macro_f1:
                    best_macro_f1 = sm["macro_f1"]
                    best_macro_t = float(t)
                if sm["micro_f1"] > best_micro_f1:
                    best_micro_f1 = sm["micro_f1"]
                    best_micro_t = float(t)

            val_summary.append({
                "model": name,
                "label": model_labels[name],
                "mAP": round(mets["mAP"], 4),
                "mAUC": round(mets["mAUC"], 4),
                "micro_f1_0.50": round(mets["micro_f1"], 4),
                "macro_f1_0.50": round(mets["macro_f1"], 4),
                "precision_0.50": round(mets["precision"], 4),
                "recall_0.50": round(mets["recall"], 4),
                "top1_hit": round(mets["top1_hit"], 4),
                "top5_hit": round(mets["top5_hit"], 4),
                "best_macro_f1": round(best_macro_f1, 4),
                "opt_macro_thresh": round(best_macro_t, 2),
                "best_micro_f1": round(best_micro_f1, 4),
                "opt_micro_thresh": round(best_micro_t, 2),
            })

        val_df = pd.DataFrame(val_summary)
        val_csv = output_dir / "augmentation_val_comparison.csv"
        val_df.to_csv(val_csv, index=False)
        print("\nValidation Summary Table:")
        print(val_df.to_string(index=False))

        del val_probs, val_targets, val_loader, val_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # PART B: Out-of-Distribution Noise Benchmark
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART B: Unseen Noise Robustness Benchmark (ACE Evaluation Dataset)")
    print("=" * 80)

    # 1. Clean evaluation audio paths
    eval_csv_path = data_root / "FSD50K.ground_truth" / "eval.csv"
    if not eval_csv_path.is_file():
        eval_csv_path = PROJECT_ROOT / "data" / "fsd50k" / "FSD50K.ground_truth" / "eval.csv"

    clean_eval_df = pd.read_csv(eval_csv_path)
    clean_audio_dir = data_root / "FSD50K.eval_audio_16k"
    if not clean_audio_dir.is_dir():
        clean_audio_dir = PROJECT_ROOT / "data" / "fsd50k" / "FSD50K.eval_audio_16k"

    clean_paths = [str(clean_audio_dir / f"{fname}.wav") for fname in clean_eval_df["fname"]]
    clean_labels = clean_eval_df["labels"].tolist()

    # 2. Noisy ACE metadata and directory discovery
    meta_csv_path = Path(args.metadata_path)
    if not meta_csv_path.is_file():
        meta_csv_path = PROJECT_ROOT / "metadata" / "fsd50k_evaluation_metadata.csv"

    noisy_meta_df = pd.read_csv(meta_csv_path) if meta_csv_path.is_file() else None

    noisy_dir_candidates = [
        Path(args.noisy_data_dir) if args.noisy_data_dir else None,
        data_root / "evaluation_fsd50k",
        data_root.parent / "evaluation_fsd50k",
        PROJECT_ROOT / "data" / "evaluation_fsd50k",
        PROJECT_ROOT / "evaluation_fsd50k",
    ]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for d in kaggle_input.iterdir():
            if d.is_dir():
                noisy_dir_candidates.append(d / "evaluation_fsd50k")
                noisy_dir_candidates.append(d / "data" / "evaluation_fsd50k")

    detected_noisy_dir = None
    for cand_dir in noisy_dir_candidates:
        if cand_dir and cand_dir.is_dir():
            detected_noisy_dir = cand_dir
            break

    if detected_noisy_dir:
        print(f"Resolved noisy evaluation directory: {detected_noisy_dir}")
    else:
        print("Notice: No dedicated noisy evaluation directory detected. Will attempt standard relative paths.")

    active_conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    noise_results = []
    noise_csv = output_dir / "augmentation_noise_robustness.csv"
    if noise_csv.is_file():
        try:
            existing_df = pd.read_csv(noise_csv)
            noise_results = existing_df.to_dict(orient="records")
            print(f"Loaded {len(noise_results)} existing result rows from {noise_csv.name}")
        except Exception:
            noise_results = []

    for cond in active_conditions:
        print(f"\n--> Evaluating Condition: {cond.upper()}...")
        existing_models_for_cond = {r["model"] for r in noise_results if r.get("condition") == cond}
        if set(loaded_models.keys()).issubset(existing_models_for_cond):
            print(f"  [i] Condition '{cond}' already evaluated in {noise_csv.name}, skipping.")
            continue

        if cond == "clean":
            cond_paths = clean_paths
            cond_labels = clean_labels
        else:
            if noisy_meta_df is None:
                print(f"  [!] Missing noisy metadata manifest, skipping {cond}.")
                continue
            sub = noisy_meta_df[noisy_meta_df["condition"] == cond]
            # Resolve paths
            cond_paths = []
            missing_paths = []
            for p in sub["output_path"]:
                cand = None
                p_rel = p.replace("data/evaluation_fsd50k/", "")
                candidates = []
                if detected_noisy_dir:
                    candidates.extend([
                        detected_noisy_dir / p_rel,
                        detected_noisy_dir / p,
                        detected_noisy_dir / cond / Path(p).name,
                    ])
                candidates.extend([
                    Path(p),
                    PROJECT_ROOT / p,
                    PROJECT_ROOT / "data" / p,
                    data_root / p,
                    data_root.parent / p,
                ])
                for c in candidates:
                    if c.is_file():
                        cand = c
                        break
                if cand is not None:
                    cond_paths.append(str(cand))
                else:
                    missing_paths.append(p)

            if missing_paths:
                print(f"  [!] Condition '{cond}': {len(missing_paths)}/{len(sub)} audio files not found.")
                if len(cond_paths) == 0:
                    print(f"  [!] Skipping condition '{cond}' because no audio files exist.")
                    continue
                else:
                    print(f"  [!] Evaluating on {len(cond_paths)} available files.")

            cond_labels = sub["labels"].tolist()[: len(cond_paths)]

        cond_dataset = EvalAudioDataset(cond_paths, cond_labels, class_to_idx)
        cond_loader = DataLoader(
            cond_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

        cond_probs = {name: [] for name in loaded_models}
        cond_targets = []

        with torch.no_grad():
            for specs, tgts in tqdm(cond_loader, desc=f"Eval {cond}"):
                specs = specs.to(device, non_blocking=True)
                cond_targets.append(tgts.numpy())
                for name, model in loaded_models.items():
                    logits = model(specs)
                    cond_probs[name].append(torch.sigmoid(logits).cpu().numpy())

        cond_targets = np.concatenate(cond_targets, axis=0)

        for name in loaded_models:
            c_probs = np.concatenate(cond_probs[name], axis=0)
            mets = compute_metrics_from_probs(c_probs, cond_targets, threshold=0.50)
            noise_results.append({
                "condition": cond,
                "model": name,
                "label": model_labels[name],
                "samples": len(cond_targets),
                "mAP": round(mets["mAP"], 4),
                "mAUC": round(mets["mAUC"], 4),
                "micro_f1": round(mets["micro_f1"], 4),
                "macro_f1": round(mets["macro_f1"], 4),
                "top1_hit": round(mets["top1_hit"], 4),
                "top5_hit": round(mets["top5_hit"], 4),
            })
            print(f"    [{name:<15}] mAP: {mets['mAP']:.4f} | mAUC: {mets['mAUC']:.4f} | Micro-F1: {mets['micro_f1']:.4f}")

        del cond_probs, cond_targets, cond_loader, cond_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Incremental saving after each condition completes
        noise_df = pd.DataFrame(noise_results)
        noise_df.to_csv(noise_csv, index=False)
        with open(output_dir / "augmentation_noise_robustness.json", "w", encoding="utf-8") as f:
            json.dump(noise_results, f, indent=2)

    if noise_results:
        noise_df = pd.DataFrame(noise_results)
        noise_csv = output_dir / "augmentation_noise_robustness.csv"
        noise_df.to_csv(noise_csv, index=False)
        print(f"\nNoise Robustness Benchmark saved to {noise_csv}")

        # Pivot Table
        pivot_conditions = [c for c in active_conditions if c in noise_df["condition"].unique()]
        mAP_pivot = noise_df.pivot(index="model", columns="condition", values="mAP")[pivot_conditions]
        print("\nmAP by Noise Condition Summary:")
        print(mAP_pivot.to_string())

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETE. All artifacts written to {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
