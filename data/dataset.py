import os
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T


class ACADDataset(Dataset):
    """
    Dataset for the Automatic Contextual Audio Denoising (ACAD) benchmark (Luong et al., EUSIPCO 2026).
    Yields paired (noisy_mel, clean_mel) log-Mel spectrograms, each of shape [1, 128, 500].

    Time resolution:
      hop_length = 160 at 16 kHz -> 100 frames / second (100 * time).
      duration_sec = 5.0 -> exactly 500 time frames.

    Expected directory hierarchy on disk:
      root_dir/
      └── {split}/ (e.g. train, val, test)
          └── {scene}/ (Kitchen, Park, Restaurant, Restroom, Street, Subway)
              └── {clip_id}/
                  ├── {clip_id}.wav                      <-- Noisy mixture
                  └── isolated_events/
                      └── background0_{scene}.wav        <-- Clean target scene
    """
    def __init__(
        self,
        root_dir="data/acad",
        split="train",
        sample_rate=16000,
        duration_sec=5.0,
        n_mels=128,
        n_fft=1024,
        win_length=400,
        hop_length=160,
        target_frames=500,
        scenes=None,
        mock=False,
        mock_length=256,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.target_len = int(sample_rate * duration_sec)
        self.target_frames = target_frames
        self.scenes = scenes
        self.mock = mock
        self.mock_length = mock_length

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=2.0,
        )

        if not self.mock:
            self.samples = self._index_samples()
        else:
            self.samples = []

    def _find_split_dir(self):
        direct_split = self.root_dir / self.split
        if direct_split.is_dir():
            return direct_split
        nested_split = self.root_dir / "acad" / self.split
        if nested_split.is_dir():
            return nested_split
        return direct_split

    def _index_samples(self):
        split_dir = self._find_split_dir()
        if not split_dir.is_dir():
            return []

        samples = []
        scene_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        if self.scenes:
            scene_dirs = [d for d in scene_dirs if d.name in self.scenes]

        for scene_dir in scene_dirs:
            for clip_dir in scene_dir.iterdir():
                if not clip_dir.is_dir():
                    continue

                # Locate noisy audio file ({clip_id}.wav)
                noisy_file = clip_dir / f"{clip_dir.name}.wav"
                if not noisy_file.is_file():
                    wavs = [f for f in clip_dir.glob("*.wav") if f.is_file()]
                    if wavs:
                        noisy_file = wavs[0]
                    else:
                        continue

                # Locate clean audio inside isolated_events/
                iso_dir = clip_dir / "isolated_events"
                if not iso_dir.is_dir():
                    continue

                clean_candidates = list(iso_dir.glob("background*.wav"))
                if not clean_candidates:
                    clean_candidates = list(iso_dir.glob("*.wav"))

                if clean_candidates:
                    clean_file = clean_candidates[0]
                    samples.append((str(noisy_file), str(clean_file)))

        return sorted(samples, key=lambda x: x[0])

    def _load_audio(self, file_path):
        waveform, sr = torchaudio.load(file_path)

        # Convert multi-channel to mono
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Resample to target sample rate
        if sr != self.sample_rate:
            waveform = AF.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)

        return waveform

    def _waveform_to_log_mel(self, waveform):
        # Compute Mel spectrogram: [1, n_mels, time_frames]
        mel = self.mel_transform(waveform)
        log_mel = torch.log(mel.clamp(min=1e-6))

        # Adjust time dimension to exact target_frames (500)
        if log_mel.shape[-1] > self.target_frames:
            log_mel = log_mel[..., :self.target_frames]
        elif log_mel.shape[-1] < self.target_frames:
            log_mel = F.pad(log_mel, (0, self.target_frames - log_mel.shape[-1]))

        return log_mel

    def __len__(self):
        if self.mock:
            return self.mock_length
        return len(self.samples)

    def __getitem__(self, idx):
        if self.mock:
            clean_mel = torch.randn(1, 128, self.target_frames)
            noisy_mel = clean_mel + 0.1 * torch.randn(1, 128, self.target_frames)
            return noisy_mel, clean_mel

        if not self.samples:
            split_path = self._find_split_dir()
            raise FileNotFoundError(
                f"No paired ACAD audio samples found in '{self.root_dir}'. "
                f"Expected directory structure: {split_path}/<Scene>/<ClipID>/<ClipID>.wav "
                "with isolated_events/background0_*.wav"
            )

        noisy_path, clean_path = self.samples[idx]
        noisy_wave = self._load_audio(noisy_path)
        clean_wave = self._load_audio(clean_path)

        # Align lengths between noisy and clean waveforms
        min_len = min(noisy_wave.shape[-1], clean_wave.shape[-1])
        noisy_wave = noisy_wave[..., :min_len]
        clean_wave = clean_wave[..., :min_len]

        # Synchronous cropping to 5.0 seconds (target_len = 80,000 samples)
        if min_len > self.target_len:
            if self.split == "train":
                max_start = min_len - self.target_len
                start = random.randint(0, max_start)
            else:
                start = (min_len - self.target_len) // 2
            noisy_wave = noisy_wave[..., start : start + self.target_len]
            clean_wave = clean_wave[..., start : start + self.target_len]
        elif min_len < self.target_len:
            pad = self.target_len - min_len
            noisy_wave = F.pad(noisy_wave, (0, pad))
            clean_wave = F.pad(clean_wave, (0, pad))

        noisy_mel = self._waveform_to_log_mel(noisy_wave)
        clean_mel = self._waveform_to_log_mel(clean_wave)

        return noisy_mel, clean_mel
