import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T

# Cap thread usage to prevent host CPU core saturation
torch.set_num_threads(min(8, torch.get_num_threads()))

from models.classifier import Classifer

ARCH_CONFIGS = {
    "tiny": {"tok_dim": 192, "num_head": 3, "num_layer": 12, "name": "DeiT ViT-Tiny"},
    "small": {"tok_dim": 384, "num_head": 6, "num_layer": 12, "name": "DeiT ViT-Small"},
    "base": {"tok_dim": 768, "num_head": 12, "num_layer": 12, "name": "DeiT ViT-Base"},
}

DEFAULT_VOCAB_PATHS = [
    PROJECT_ROOT / "data" / "FSD50K.ground_truth" / "vocabulary.csv",
    PROJECT_ROOT / "data" / "vocabulary.csv",
    PROJECT_ROOT / "FSD50K.ground_truth" / "vocabulary.csv",
]


def resolve_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_vocabulary(vocab_path: Optional[Union[str, Path]] = None) -> Dict[int, str]:
    target_path = None
    if vocab_path and Path(vocab_path).is_file():
        target_path = Path(vocab_path)
    else:
        for p in DEFAULT_VOCAB_PATHS:
            if p.is_file():
                target_path = p
                break

    idx_to_label: Dict[int, str] = {}
    if target_path and target_path.is_file():
        with open(target_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    idx = int(row[0].strip())
                    label = row[1].strip()
                except ValueError:
                    try:
                        idx = int(row[1].strip())
                        label = row[0].strip()
                    except ValueError:
                        continue
                idx_to_label[idx] = label

    return idx_to_label


class AudioClassifier:
    """
    Inference pipeline for AST multi-label environmental sound event classifier.
    """
    def __init__(
        self,
        checkpoint_path: Union[str, Path] = PROJECT_ROOT / "data" / "best_classifier.pth",
        vocab_path: Optional[Union[str, Path]] = None,
        arch: str = "tiny",
        num_classes: int = 200,
        sample_rate: int = 16000,
        duration_sec: float = 10.0,
        target_frames: int = 1000,
        device: Union[str, torch.device] = "auto",
    ):
        self.device = resolve_device(str(device)) if isinstance(device, str) else device
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.target_len = int(sample_rate * duration_sec)
        self.target_frames = target_frames
        self.num_classes = num_classes

        # Normalization constants matching FSD50K training setup
        self.norm_mean = -4.2677393
        self.norm_std = 4.5689974

        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            win_length=400,
            hop_length=160,
            n_mels=128,
            center=True,
            power=2.0,
        )

        # Vocabulary
        self.idx_to_label = load_vocabulary(vocab_path)

        # Architecture config
        if arch not in ARCH_CONFIGS:
            raise ValueError(f"Unknown architecture '{arch}'. Supported: {list(ARCH_CONFIGS.keys())}")
        arch_cfg = ARCH_CONFIGS[arch]

        # Initialize model
        self.model = Classifer(
            tok_dim=arch_cfg["tok_dim"],
            num_classes=self.num_classes,
            c_in=1,
            overlap=6,
            patch_size=16,
            size=(128, self.target_frames),
            num_head=arch_cfg["num_head"],
            num_layer=arch_cfg["num_layer"],
            pretrained_dino=False,
            use_cls_dist=True,
        ).to(self.device)

        self._load_checkpoint(checkpoint_path)
        self.model.eval()

    def _load_checkpoint(self, checkpoint_path: Union[str, Path]):
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

        # Clean DDP module prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=True)

    def preprocess(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """
        Loads and transforms an audio file into a normalized log-Mel spectrogram tensor.
        Returns: tensor of shape (1, 1, 128, target_frames)
        """
        waveform, sr = torchaudio.load(str(audio_path))

        # Downmix stereo/multichannel to mono
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Resample to target sample rate
        if sr != self.sample_rate:
            waveform = AF.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)

        # Center-crop or zero-pad waveform
        length = waveform.shape[-1]
        if length > self.target_len:
            start = (length - self.target_len) // 2
            waveform = waveform[..., start : start + self.target_len]
        elif length < self.target_len:
            pad = self.target_len - length
            waveform = F.pad(waveform, (0, pad))

        # Compute log-mel spectrogram
        mel = self.mel_transform(waveform)
        log_mel = torch.log(mel.clamp(min=1e-6))
        log_mel = (log_mel - self.norm_mean) / (self.norm_std * 2.0)

        # Ensure target frame length
        if log_mel.shape[-1] > self.target_frames:
            log_mel = log_mel[..., :self.target_frames]
        elif log_mel.shape[-1] < self.target_frames:
            log_mel = F.pad(log_mel, (0, self.target_frames - log_mel.shape[-1]))

        # Add batch dimension: (1, 1, 128, target_frames)
        return log_mel.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(
        self,
        audio_path: Union[str, Path],
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Union[int, str, float]]]:
        """
        Runs inference on an audio file.
        Returns a list of dicts with: class_id, label, probability, logit.
        """
        input_tensor = self.preprocess(audio_path)
        logits = self.model(input_tensor)[0]
        probs = torch.sigmoid(logits)

        results = []
        if threshold is not None:
            indices = torch.where(probs >= threshold)[0]
            # Sort detected indices by probability descending
            sorted_order = torch.argsort(probs[indices], descending=True)
            indices = indices[sorted_order]
        else:
            k = min(top_k, self.num_classes)
            _, indices = torch.topk(probs, k=k)

        for idx_tensor in indices:
            idx = int(idx_tensor.item())
            score = float(probs[idx].item())
            logit_val = float(logits[idx].item())
            label = self.idx_to_label.get(idx, f"class_{idx}")
            results.append({
                "class_id": idx,
                "label": label,
                "probability": score,
                "logit": logit_val,
            })

        return results


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for AST sound event classifier on audio files")
    parser.add_argument(
        "--audio",
        type=str,
        default=str(PROJECT_ROOT / "data" / "100018.wav"),
        help="Path to single input audio file (.wav, .mp3, .flac)",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Optional directory containing multiple audio files to run inference on",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "data" / "best_classifier.pth"),
        help="Path to trained classifier checkpoint (.pth)",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default=None,
        help="Path to vocabulary.csv mapping class IDs to human-readable names",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="tiny",
        choices=["tiny", "small", "base"],
        help="ViT backbone architecture (default: tiny)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to display (default: 5)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Filter predictions above this probability threshold (e.g. 0.2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device (default: auto)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to export inference results (.json)",
    )
    return parser.parse_args()


def format_predictions(file_path: str, predictions: List[Dict]) -> str:
    lines = [f"\nFile: {file_path}"]
    lines.append(f"{'Class ID':<10} {'Label':<40} {'Probability':<15} {'Logit':<10}")
    lines.append("-" * 75)
    for p in predictions:
        lines.append(
            f"{p['class_id']:<10} {p['label']:<40} {p['probability']*100:>6.2f}% ({p['probability']:.4f})  {p['logit']:>7.3f}"
        )
    return "\n".join(lines)


def main():
    args = parse_args()
    classifier = AudioClassifier(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab_path,
        arch=args.arch,
        device=args.device,
    )

    audio_files = []
    if args.audio_dir:
        dir_path = Path(args.audio_dir)
        if not dir_path.is_dir():
            print(f"Error: audio directory '{args.audio_dir}' does not exist.", file=sys.stderr)
            sys.exit(1)
        valid_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        audio_files = [p for p in sorted(dir_path.iterdir()) if p.suffix.lower() in valid_exts]
        if not audio_files:
            print(f"No audio files found in '{args.audio_dir}'.", file=sys.stderr)
            sys.exit(1)
    else:
        audio_path = Path(args.audio)
        if not audio_path.is_file():
            print(f"Error: audio file '{args.audio}' does not exist.", file=sys.stderr)
            sys.exit(1)
        audio_files = [audio_path]

    all_results = {}
    for audio_file in audio_files:
        preds = classifier.predict(
            audio_path=audio_file,
            top_k=args.top_k,
            threshold=args.threshold,
        )
        all_results[str(audio_file)] = preds
        print(format_predictions(str(audio_file), preds))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nInference results saved to: {args.output}")


if __name__ == "__main__":
    main()
