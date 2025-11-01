"""
Task 1: Supervised Baseline Training Script

Train supervised classifier on AudioMNIST for digit classification
Compare 1D (waveform) vs 2D (spectrogram) encoders
"""  # noqa: E501

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from audiossl.data import collate_fn, LogMelSpectrogram, create_audiomnist_splits
from audiossl.modeling import SupervisedAudioClassifier
from audiossl.utils.wandb_utils import log_audio_predictions_supervised


def train_epoch(
    model, dataloader, criterion, optimizer, device, spectrogram_transform=None
):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for waveforms, labels in pbar:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if model.encoder_type == "1d":
            logits = model(waveform=waveforms)
        elif model.encoder_type == "2d":
            spectrograms = []
            for wav in waveforms:
                spec = spectrogram_transform(wav.unsqueeze(0).cpu())
                spectrograms.append(spec)
            spectrograms = torch.stack(spectrograms).to(device)
            logits = model(spectrogram=spectrograms)
        else:  # both
            spectrograms = []
            for wav in waveforms:
                spec = spectrogram_transform(wav.unsqueeze(0).cpu())
                spectrograms.append(spec)
            spectrograms = torch.stack(spectrograms).to(device)
            logits = model(waveform=waveforms, spectrogram=spectrograms)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.2f}%"}
        )

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


@torch.no_grad()
def validate(model, dataloader, criterion, device, spectrogram_transform=None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for waveforms, labels in tqdm(dataloader, desc="Validation"):
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        if model.encoder_type == "1d":
            logits = model(waveform=waveforms)
        elif model.encoder_type == "2d":
            spectrograms = []
            for wav in waveforms:
                spec = spectrogram_transform(wav.unsqueeze(0).cpu())
                spectrograms.append(spec)
            spectrograms = torch.stack(spectrograms).to(device)
            logits = model(spectrogram=spectrograms)
        else:  # both
            spectrograms = []
            for wav in waveforms:
                spec = spectrogram_transform(wav.unsqueeze(0).cpu())
                spectrograms.append(spec)
            spectrograms = torch.stack(spectrograms).to(device)
            logits = model(waveform=waveforms, spectrogram=spectrograms)

        loss = criterion(logits, labels)

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def plot_training_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy curve
    ax2.plot(history["train_acc"], label="Train Accuracy")
    ax2.plot(history["val_acc"], label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Training curves saved to {save_path}")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train supervised classifier on AudioMNIST"
    )
    parser.add_argument(
        "--encoder-type",
        type=str,
        default="1d",
        choices=["1d", "2d", "both"],
        help="Encoder type: 1d (waveform), 2d (spectrogram), or both",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes")
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Audio sample rate"
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=15, help="Early stopping patience"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="AudioMNIST/data",
        help="Root directory for AudioMNIST data",
    )
    parser.add_argument(
        "--num-test-speakers", type=int, default=12, help="Number of test speakers"
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="experiments/supervised",
        help="Experiment directory",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=256, help="Hidden dimension for model"
    )
    parser.add_argument(
        "--n-mels", type=int, default=64, help="Number of mel filterbanks"
    )
    parser.add_argument(
        "--n-fft", type=int, default=400, help="FFT size for spectrogram"
    )
    parser.add_argument(
        "--hop-length", type=int, default=160, help="Hop length for spectrogram"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for dataloader"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=5,
        help="Patience for learning rate scheduler",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="Factor for learning rate scheduler",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    exp_dir = Path(args.exp_dir) / args.encoder_type
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "encoder_type": args.encoder_type,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "num_epochs": args.epochs,
        "num_classes": args.num_classes,
        "sample_rate": args.sample_rate,
        "early_stop_patience": args.early_stop_patience,
        "hidden_dim": args.hidden_dim,
        "seed": args.seed,
    }

    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    if not args.no_wandb:
        wandb.init(
            project="audiomnist-ssl",
            name=f"supervised-{args.encoder_type}",
            config=config,
            tags=["supervised", args.encoder_type, "task1"],
        )

    spectrogram_transform = None
    if args.encoder_type in ["2d", "both"]:
        spectrogram_transform = LogMelSpectrogram(
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
        )

    print("\nLoading AudioMNIST dataset...")
    train_dataset, val_dataset, full_dataset = create_audiomnist_splits(
        root=args.data_root, num_test_speakers=args.num_test_speakers
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"\nCreating supervised model with {args.encoder_type} encoder...")
    model = SupervisedAudioClassifier(
        encoder_type=args.encoder_type,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_acc = 0
    best_epoch = 0

    epochs_without_improvement = 0

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Early stopping: patience={args.early_stop_patience}")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, spectrogram_transform
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, spectrogram_transform
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")

        if not args.no_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                exp_dir / "best_model.pt",
            )
            print(f"  ✓ New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= args.early_stop_patience:
            print(
                f"\n  Early stopping triggered! No improvement for {args.early_stop_patience} epochs."
            )
            print(
                f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}"
            )
            break

        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                exp_dir / f"checkpoint_epoch_{epoch}.pt",
            )

        if epoch % 10 == 0 and not args.no_wandb:
            print("  Logging audio predictions to wandb...")
            log_audio_predictions_supervised(
                model=model,
                dataloader=val_loader,
                device=device,
                spectrogram_transform=spectrogram_transform,
                num_samples=5,
                epoch=epoch,
                sample_rate=args.sample_rate,
            )

    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, exp_dir / "training_curves.png")

    if not args.no_wandb:
        wandb.run.summary["best_val_accuracy"] = best_val_acc
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.log(
            {"training_curves": wandb.Image(str(exp_dir / "training_curves.png"))}
        )

        wandb.finish()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"Results saved to: {exp_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
