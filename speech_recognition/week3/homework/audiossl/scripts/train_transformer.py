"""
Experiment: Audio Spectrogram Transformer (AST-style) classifier for AudioMNIST.

Implements a transformer-based model operating on log-mel spectrograms with
SpecAugment and label smoothing.

References:
  Gong et al., "AST: Audio Spectrogram Transformer"
"""  # noqa: E501

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


from audiossl.data.dataset import (
    LogMelSpectrogram,
    create_audiomnist_splits,
)
from audiossl.losses import LabelSmoothingCrossEntropy
from audiossl.modeling import AudioSpectrogramTransformer

ROOT = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_with_lengths(
    batch: Iterable[Tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    waveforms, labels = zip(*batch)
    lengths = torch.tensor([w.shape[-1] for w in waveforms], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(
        [w.squeeze(0) for w in waveforms], batch_first=True
    )
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels_tensor


def pad_spectrogram_batch(
    spectrograms: Iterable[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([spec.shape[-1] for spec in spectrograms], dtype=torch.long)
    max_len = lengths.max().item()
    n_mels = spectrograms[0].shape[0]

    batch = torch.zeros(len(spectrograms), n_mels, max_len, dtype=torch.float32)
    for idx, spec in enumerate(spectrograms):
        batch[idx, :, : spec.shape[-1]] = spec

    return batch, lengths


def waveforms_to_spectrograms(
    waveforms: torch.Tensor,
    waveform_lengths: torch.Tensor,
    mel_transform: LogMelSpectrogram,
) -> Tuple[torch.Tensor, torch.Tensor]:
    specs = []
    for wav, wav_len in zip(waveforms, waveform_lengths):
        # [1, time]
        wav = wav[:wav_len].unsqueeze(0)
        spec = mel_transform(wav).squeeze(0).float()
        specs.append(spec)

    return pad_spectrogram_batch(specs)


class SpecAugment:
    """Minimal SpecAugment with time and frequency masking."""

    def __init__(
        self,
        freq_masks: int = 2,
        time_masks: int = 2,
        freq_mask_param: int = 8,
        time_mask_param: int = 30,
    ) -> None:
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def __call__(
        self, spectrograms: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        augmented = spectrograms.clone()
        batch_size, n_mels, max_len = augmented.shape

        for b in range(batch_size):
            t_len = int(lengths[b].item())

            for _ in range(self.freq_masks):
                mask_width = random.randint(0, min(self.freq_mask_param, n_mels))
                if mask_width == 0 or mask_width >= n_mels:
                    continue
                f0 = random.randint(0, n_mels - mask_width)
                augmented[b, f0 : f0 + mask_width, :t_len] = 0.0

            for _ in range(self.time_masks):
                if t_len <= 1:
                    break
                mask_width = random.randint(0, min(self.time_mask_param, t_len))
                if mask_width == 0 or mask_width >= t_len:
                    continue
                t0 = random.randint(0, t_len - mask_width)
                augmented[b, :, t0 : t0 + mask_width] = 0.0

        return augmented


@dataclass
class TrainingConfig:
    encoder_type: str = "spectrogram-transformer"
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_epochs: int = 40
    patience: int = 8
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    sample_rate: int = 16000
    n_mels: int = 64
    seed: int = 42
    data_root: str = "AudioMNIST/data"
    experiment_dir: str = "experiments/transformer"
    use_spec_augment: bool = True


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    mel_transform: LogMelSpectrogram,
    spec_augment: SpecAugment | None,
    grad_clip: float,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(dataloader, desc="Train", leave=False)
    for waveforms, waveform_lengths, labels in progress:
        spectrograms, spec_lengths = waveforms_to_spectrograms(
            waveforms, waveform_lengths, mel_transform
        )

        if spec_augment is not None:
            spectrograms = spec_augment(spectrograms, spec_lengths)

        spectrograms = spectrograms.to(device)
        spec_lengths = spec_lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(spectrograms, spec_lengths)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}", acc=f"{100.0 * correct / max(total, 1):.2f}%"
        )

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mel_transform: LogMelSpectrogram,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for waveforms, waveform_lengths, labels in tqdm(
        dataloader, desc="Val", leave=False
    ):
        spectrograms, spec_lengths = waveforms_to_spectrograms(
            waveforms, waveform_lengths, mel_transform
        )
        spectrograms = spectrograms.to(device)
        spec_lengths = spec_lengths.to(device)
        labels = labels.to(device)

        logits = model(spectrograms, spec_lengths)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Audio Spectrogram Transformer on AudioMNIST"
    )
    parser.add_argument("--epochs", type=int, default=TrainingConfig.num_epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument(
        "--weight-decay", type=float, default=TrainingConfig.weight_decay
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=TrainingConfig.label_smoothing
    )
    parser.add_argument(
        "--no-spec-augment", action="store_true", help="Disable SpecAugment"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        label_smoothing=args.label_smoothing,
        use_spec_augment=not args.no_spec_augment,
        seed=args.seed,
    )

    set_seed(config.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    exp_dir = Path(config.experiment_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    train_dataset, val_dataset, _ = create_audiomnist_splits(config.data_root)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_with_lengths,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_with_lengths,
    )

    mel_transform = LogMelSpectrogram(
        sample_rate=config.sample_rate,
        n_mels=config.n_mels,
        n_fft=1024,
        hop_length=256,
        f_min=20.0,
        f_max=config.sample_rate / 2,
    )

    model = AudioSpectrogramTransformer(
        n_mels=config.n_mels,
        d_model=192,
        n_heads=3,
        num_layers=4,
        dim_feedforward=384,
        dropout=0.15,
        max_seq_len=500,
        classifier_hidden_dim=256,
        num_classes=10,
    ).to(device)

    criterion = LabelSmoothingCrossEntropy(smoothing=config.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=config.learning_rate * 0.1
    )

    spec_augment = SpecAugment() if config.use_spec_augment else None

    best_val_acc = 0.0
    best_epoch = -1
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            mel_transform,
            spec_augment,
            config.grad_clip,
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, mel_transform
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Train: loss={train_loss:.4f}, acc={train_acc:.2f}% | "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), exp_dir / "best_model.pt")
            print(f"Saved new best model (val acc={best_val_acc:.2f}%).")

        if epoch - best_epoch >= config.patience:
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(best epoch {best_epoch}, best val acc {best_val_acc:.2f}%)."
            )
            break

    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")


if __name__ == "__main__":
    main()
