"""
AudioMNIST Dataset
Copied from SSL_Seminar.ipynb

Dataset of spoken digits (0-9) from 60 speakers
Each speaker pronounces each digit multiple times
"""
from typing import Callable

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from torch.utils.data import Dataset, Subset


class LogMelSpectrogram(T.MelSpectrogram):
    """
    Log Mel Spectrogram transform
    Adds logarithm to mel spectrogram for better dynamic range
    """

    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [channels, time] or [time]

        Returns:
            log_mel_spec: [n_mels, time_frames]
        """
        return (super().forward(waveform) + self.eps).log()


class AudioMNISTDataset(Dataset):
    """
    AudioMNIST Dataset loader

    Directory structure:
        root/
            01/  # speaker 01
                0_01_0.wav  # digit_speaker_utterance.wav
                0_01_1.wav
                ...
            02/  # speaker 02
                ...

    Args:
        root: Path to AudioMNIST/data directory
        sr: Target sample rate (default: 16000)
        transform: Optional transform to apply to waveforms
    """

    def __init__(self, root, sr: int = 16000, transform: Callable | None = None):
        self.root = root
        self.sr = sr
        self.transform = transform
        self.items = self.list_wavs_and_labels(root)

    def __len__(self):
        return len(self.items)

    def list_wavs_and_labels(self, root: str):
        """
        List all wav files with their labels and speaker IDs

        Returns:
            items: List of tuples (wav_path, digit_label, speaker_id)
        """
        base = Path(root)
        speakers = sorted([p for p in base.iterdir() if p.is_dir()])
        items = []

        for sp in speakers:
            for wav in sorted(sp.glob("**/*.wav")):
                # Filename format: "digit_speaker_utterance.wav"
                # e.g., "9_10_0.wav" means digit 9, speaker 10, utterance 0
                name = wav.stem.split("_")
                digit = int(name[0])
                speaker_id = sp.name
                items.append((str(wav), digit, speaker_id))

        return items

    def load_wav(self, path):
        """
        Load and preprocess waveform

        Args:
            path: Path to wav file

        Returns:
            wav: [1, time] - mono waveform at target sample rate
        """
        wav, sr = torchaudio.load(path)  # [channels, time]

        # Resample if necessary
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)

        # Convert to mono if stereo
        wav = wav.mean(dim=0, keepdim=True)  # [1, time]

        return wav

    def __getitem__(self, idx):
        """
        Get a single sample

        Args:
            idx: Index

        Returns:
            wav: [1, time] - waveform
            label: int - digit label (0-9)
            speaker: str - speaker ID (optional, for splits)
        """
        path, label, speaker = self.items[idx]
        wav = self.load_wav(path)

        if self.transform:
            wav = self.transform(wav)

        return wav, label


def collate_fn(batch):
    """
    Collate function for DataLoader
    Pads waveforms to same length in batch

    Args:
        batch: List of tuples (wav, label)

    Returns:
        wavs: [batch_size, max_time] - padded waveforms
        labels: [batch_size] - labels
    """
    wavs, labels = zip(*batch)

    # Pad sequences to max length in batch
    # Input: list of [1, time_i], output: [batch_size, max_time]
    wavs = nn.utils.rnn.pad_sequence([w.squeeze(0) for w in wavs], batch_first=True)

    labels = torch.tensor(labels, dtype=torch.long)

    return wavs, labels


def split_by_speaker(dataset: AudioMNISTDataset, test_speakers: set):
    """
    Split dataset by speaker IDs
    Used to ensure speakers don't overlap between train/val/test

    Args:
        dataset: AudioMNISTDataset instance
        test_speakers: Set of speaker IDs to assign to test set

    Returns:
        train_indices: List of indices for training
        test_indices: List of indices for testing
    """
    train_idxs = []
    test_idxs = []

    for idx, (_, _, speaker) in enumerate(dataset.items):
        if speaker in test_speakers:
            test_idxs.append(idx)
        else:
            train_idxs.append(idx)

    return train_idxs, test_idxs


def create_audiomnist_splits(root, num_test_speakers=12):
    """
    Create train/validation splits for AudioMNIST

    Default split: 48 training speakers, 12 validation speakers (80/20 split)

    Args:
        root: Path to AudioMNIST/data directory
        num_test_speakers: Number of speakers for validation set

    Returns:
        train_dataset: Subset for training
        valid_dataset: Subset for validation
    """
    full_dataset = AudioMNISTDataset(root=root)

    all_speakers = sorted({speaker for (_, _, speaker) in full_dataset.items})

    # Use last N speakers for validation (deterministic split)
    valid_speakers = set(all_speakers[-num_test_speakers:])

    # Split indices by speaker
    train_idxs, valid_idxs = split_by_speaker(full_dataset, valid_speakers)

    train_dataset = Subset(full_dataset, train_idxs)
    valid_dataset = Subset(full_dataset, valid_idxs)

    return train_dataset, valid_dataset, full_dataset
