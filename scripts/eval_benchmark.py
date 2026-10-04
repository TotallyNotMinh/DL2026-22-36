#!/usr/bin/env python3
"""
scripts/eval_benchmark.py

Comprehensive Evaluation Benchmark Runner for AST Sound Event Classifier
on the newly synthesized multi-event, multi-condition test benchmark:
- Conditions: clean, reverb (unseen room acoustics), snr_10, snr_5, snr_0, snr_neg5.
- Metrics: mAP, mAUC, Micro-F1, Macro-F1, Top-1 Hit, Top-5 Hit.
- Exports results to metadata/benchmark_eval_results.json & .csv.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T
from tqdm import tqdm

# Cap thread usage to prevent host CPU core saturation
torch.set_num_threads(min(8, torch.get_num_threads()))

from models.classifier import Classifer
from metrics.classification import MultiLabelClassificationMetrics


class BenchmarkAudioDataset(Dataset):
    """
    Dataset for loading and preprocessing evaluation benchmark audio clips.
    """
    def __init__(
        self,
        records: list[dict],
        class_to_idx: dict[str, int],
        num_classes: int = 200,
        sample_rate: int = 16000,
        duration_sec: float = 10.0,
        target_frames: int = 1000,
    ):
        self.records = records
        self.class_to_idx = class_to_idx
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
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
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        file_path = rec["output_path"]
        if not os.path.isfile(file_path):
            cand = PROJECT_ROOT / file_path
            if cand.is_file():
                file_path = str(cand)

        # Load audio waveform
        waveform, sr = torchaudio.load(file_path)
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if sr != self.sample_rate:
            waveform = AF.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)

        # Pad or center crop to 10.0s target length (160,000 samples)
        length = waveform.shape[-1]
        if length > self.target_len:
            start = (length - self.target_len) // 2
            waveform = waveform[..., start : start + self.target_len]
        elif length < self.target_len:
            pad = self.target_len - length
            waveform = F.pad(waveform, (0, pad))

        # Log-mel spectrogram
        mel = self.mel_transform(waveform)
        log_mel = torch.log(mel.clamp(min=1e-6))
        log_mel = (log_mel - self.norm_mean) / (self.norm_std * 2.0)

        if log_mel.shape[-1] > self.target_frames:
            log_mel = log_mel[..., :self.target_frames]
        elif log_mel.shape[-1] < self.target_frames:
            log_mel = F.pad(log_mel, (0, self.target_frames - log_mel.shape[-1]))

        # Target multi-hot vector
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        labels_str = str(rec.get("labels", ""))
        for l in labels_str.split(";"):
            l = l.strip()
            if l in self.class_to_idx:
                target[self.class_to_idx[l]] = 1.0

        return log_mel, target


def load_vocabulary(vocab_candidates: list[Path]) -> tuple[dict[str, int], dict[int, str]]:
    class_to_idx: dict[str, int] = {}
    idx_to_class: dict[int, str] = {}

    target_csv = None
    for p in vocab_candidates:
        if p.is_file():
            target_csv = p
            break

    if target_csv:
        with open(target_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    c_id = int(row[0].strip())
                    c_name = row[1].strip()
                except ValueError:
                    try:
                        c_id = int(row[1].strip())
                        c_name = row[0].strip()
                    except ValueError:
                        continue
                class_to_idx[c_name] = c_id
                idx_to_class[c_id] = c_name

    # Fallback to config json if csv didn't load 200 classes
    if len(class_to_idx) < 200:
        json_path = PROJECT_ROOT / "config" / "fsd50k_200_classes.json"
        if json_path.is_file():
            with open(json_path, "r", encoding="utf-8") as f:
                c200 = json.load(f)
            for k, v in c200.items():
                cid = v["class_id"]
                class_to_idx[k] = cid
                idx_to_class[cid] = k

    return class_to_idx, idx_to_class


def load_classifier(checkpoint_path: Path, device: torch.device) -> Classifer:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def evaluate_condition(
    model: Classifer,
    records: list[dict],
    class_to_idx: dict[str, int],
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 2,
    threshold: float = 0.2,
) -> dict[str, float]:
    dataset = BenchmarkAudioDataset(records, class_to_idx)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    metrics = MultiLabelClassificationMetrics(num_classes=200, threshold=threshold)

    with torch.no_grad():
        for spectrograms, targets in loader:
            spectrograms = spectrograms.to(device, non_blocking=True)
            logits = model(spectrograms)
            metrics.update(logits.cpu(), targets)

    return metrics.compute()


def run_evaluation(
    checkpoint_path: Path,
    metadata_csv: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    device_str: str = "auto",
    threshold: float = 0.2,
):
    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available())
        else (device_str if device_str != "auto" else "cpu")
    )
    print("=" * 75)
    print(" FSD50K UNSEEN CONDITIONS BENCHMARK EVALUATION")
    print("=" * 75)
    print(f"Checkpoint Path    : {checkpoint_path}")
    print(f"Metadata Manifest  : {metadata_csv}")
    print(f"Decision Threshold : {threshold}")
    print(f"Execution Device   : {device}")
    print("=" * 75)

    # 1. Load Vocabulary
    vocab_candidates = [
        PROJECT_ROOT / "data" / "fsd50k" / "FSD50K.ground_truth" / "vocabulary.csv",
        PROJECT_ROOT / "data" / "FSD50K.ground_truth" / "vocabulary.csv",
        PROJECT_ROOT / "data" / "vocabulary.csv",
    ]
    class_to_idx, idx_to_class = load_vocabulary(vocab_candidates)
    print(f"Loaded taxonomy    : {len(class_to_idx)} classes")

    # 2. Load Model
    print("Loading model weights...")
    model = load_classifier(checkpoint_path, device)
    print("Model loaded successfully.")

    # 3. Read Metadata Manifest
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Benchmark metadata manifest missing: {metadata_csv}")
    df = pd.read_csv(metadata_csv)
    print(f"Total test samples : {len(df):,} across conditions: {sorted(df['condition'].unique())}")

    # Standard order of evaluation conditions
    all_conditions = ["clean", "reverb", "snr_10", "snr_5", "snr_0", "snr_neg5"]
    active_conditions = [c for c in all_conditions if c in df["condition"].unique()]
    # Any other conditions present
    for c in df["condition"].unique():
        if c not in active_conditions:
            active_conditions.append(c)

    results_table = []
    print("\nRunning evaluation across acoustic conditions...")
    for cond in active_conditions:
        sub_df = df[df["condition"] == cond]
        records = sub_df.to_dict(orient="records")
        print(f"  • Evaluating condition: {cond:<10} ({len(records)} clips)...", end="", flush=True)

        res = evaluate_condition(
            model=model,
            records=records,
            class_to_idx=class_to_idx,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            threshold=threshold,
        )
        print(f" done. (mAP: {res['mAP']:.4f}, mAUC: {res['mAUC']:.4f}, F1: {res['micro_f1']:.4f})")

        results_table.append({
            "Condition": cond,
            "Samples": len(records),
            "mAP": round(res["mAP"], 4),
            "mAUC": round(res["mAUC"], 4),
            "Micro_F1": round(res["micro_f1"], 4),
            "Macro_F1": round(res["macro_f1"], 4),
            "Top1_Hit": round(res["top1_hit"], 4),
            "Top5_Hit": round(res["top5_hit"], 4),
        })

    # Summary Display
    results_df = pd.DataFrame(results_table)
    print("\n" + "=" * 75)
    print(" BENCHMARK EVALUATION RESULTS SUMMARY")
    print("=" * 75)
    print(results_df.to_string(index=False))
    print("=" * 75)

    # Save results
    output_dir = metadata_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_out = output_dir / "benchmark_eval_results.csv"
    json_out = output_dir / "benchmark_eval_results.json"
    results_df.to_csv(csv_out, index=False)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2)

    print(f"\nSaved evaluation metrics to:")
    print(f"  - CSV : {csv_out}")
    print(f"  - JSON: {json_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained AST classifier on FSD50K Unseen Conditions Benchmark."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "data" / "best_classifier.pth",
        help="Path to trained classifier checkpoint (.pth)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "metadata" / "fsd50k_evaluation_metadata.csv",
        help="Path to evaluation metadata manifest (.csv)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Probability decision threshold for F1 computation (default: 0.2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run evaluation on",
    )
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        metadata_csv=args.metadata,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_str=args.device,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
