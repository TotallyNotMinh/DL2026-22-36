#!/usr/bin/env python3
"""
scripts/eval_val_split.py

Evaluate the AST Classifier on the exact FSD50K validation split (dev.csv split='val')
used during training to verify against the checkpoint's recorded validation performance.
"""

import argparse
import os
import sys
from pathlib import Path
import time

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.classifier import Classifer
from data.dataset import FSD50KDataset
from metrics.classification import MultiLabelClassificationMetrics

# Enforce thread limits to avoid saturating CPU
torch.set_num_threads(min(8, torch.get_num_threads()))

ARCH_CONFIGS = {
    "tiny": {"tok_dim": 192, "num_head": 3, "num_layer": 12, "name": "DeiT ViT-Tiny"},
    "small": {"tok_dim": 384, "num_head": 6, "num_layer": 12, "name": "DeiT ViT-Small"},
    "base": {"tok_dim": 768, "num_head": 12, "num_layer": 12, "name": "DeiT ViT-Base"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AST Classifier on FSD50K split")
    parser.add_argument("--checkpoint", type=str, default="data/best_classifier.pth", help="Path to checkpoint .pth")
    parser.add_argument("--data-path", type=str, default="data/fsd50k", help="Path to FSD50K root")
    parser.add_argument("--split", type=str, default="val", choices=["val", "eval", "test", "train"], help="Dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--arch", type=str, default="tiny", choices=["tiny", "small", "base"], help="Model architecture")
    parser.add_argument("--pooling", type=str, default=None, choices=[None, "gap", "gap_max", "attention"], help="Pooling strategy: None (baseline dual-token CLS+DIST), gap, gap_max, or attention")
    parser.add_argument("--duration-sec", type=float, default=10.0, help="Audio duration in seconds")
    parser.add_argument("--target-frames", type=int, default=1000, help="Spectrogram target frames")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 70)
    print(f"FSD50K '{args.split.upper()}' Split Evaluation (Raw Inference, No Augmentation)")
    print("=" * 70)
    print(f"  • Checkpoint:     {args.checkpoint}")
    print(f"  • Dataset root:   {args.data_path}")
    print(f"  • Split:          {args.split}")
    print(f"  • Device:         {device}")
    print(f"  • Batch size:     {args.batch_size}")
    print(f"  • PyTorch threads:{torch.get_num_threads()}")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    # Load dataset
    val_dataset = FSD50KDataset(
        root_dir=args.data_path,
        split=args.split,
        duration_sec=args.duration_sec,
        target_frames=args.target_frames,
        num_classes=200,
        normalize=True,
    )
    print(f"  • Dataset size:   {len(val_dataset)} clips")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Instantiate model
    arch_cfg = ARCH_CONFIGS[args.arch]
    model = Classifer(
        tok_dim=arch_cfg["tok_dim"],
        num_classes=200,
        c_in=1,
        overlap=6,
        patch_size=16,
        size=(128, args.target_frames),
        num_head=arch_cfg["num_head"],
        num_layer=arch_cfg["num_layer"],
        pretrained_dino=False,
        use_cls_dist=True,
        pooling=args.pooling,
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        reported_epoch = ckpt.get("epoch", "N/A")
        reported_best_map = ckpt.get("best_map", None)
    else:
        state_dict = ckpt
        reported_epoch = "N/A"
        reported_best_map = None

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    print(f"  • Checkpoint Epoch:    {reported_epoch}")
    if reported_best_map is not None:
        print(f"  • Checkpoint Best mAP: {reported_best_map:.4f} ({reported_best_map:.6f})")

    criterion = nn.BCEWithLogitsLoss().to(device)
    metrics = MultiLabelClassificationMetrics(num_classes=200)

    running_val_loss = 0.0
    start_time = time.time()

    print("\nRunning evaluation on validation split...")
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=True)
        for spectrograms, targets in pbar:
            spectrograms = spectrograms.to(device)
            targets = targets.to(device)

            logits = model(spectrograms)
            val_loss = criterion(logits, targets)

            running_val_loss += val_loss.item()
            metrics.update(logits, targets)
            pbar.set_postfix({"val_bce": f"{val_loss.item():.4f}"})

    total_time = time.time() - start_time
    avg_val_loss = running_val_loss / max(1, len(val_loader))
    val_results = metrics.compute()

    val_map = val_results["mAP"]
    val_mauc = val_results["mAUC"]
    micro_f1 = val_results["micro_f1"]
    macro_f1 = val_results["macro_f1"]
    top1_hit = val_results["top1_hit"]
    top5_hit = val_results["top5_hit"]

    throughput = len(val_dataset) / total_time

    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS (Split: {args.split.upper()})")
    print("=" * 70)
    print(f"  • Samples Evaluated: {len(val_dataset):,}")
    print(f"  • Elapsed Time:      {total_time:.2f}s ({throughput:.2f} samples/sec)")
    print(f"  • BCE Loss:          {avg_val_loss:.4f}")
    print(f"  • mAP:               {val_map:.4f} ({val_map:.6f})")
    print(f"  • mAUC:              {val_mauc:.4f} ({val_mauc:.6f})")
    print(f"  • Micro-F1:          {micro_f1:.4f}")
    print(f"  • Macro-F1:          {macro_f1:.4f}")
    print(f"  • Top-1 Hit Rate:    {top1_hit * 100:.2f}%")
    print(f"  • Top-5 Hit Rate:    {top5_hit * 100:.2f}%")
    if args.split == "val" and reported_best_map is not None:
        diff = val_map - reported_best_map
        print(f"  • Match Check:       Reported = {reported_best_map:.6f} vs Measured = {val_map:.6f} (Diff: {diff:+.6e})")
    print("=" * 70)


if __name__ == "__main__":
    main()
