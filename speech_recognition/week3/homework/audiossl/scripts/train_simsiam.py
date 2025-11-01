"""
Task 3: SimSiam Non-Contrastive Learning Training Script

Train multi-format model using SimSiam (no negative pairs needed!)
Simple siamese network with stop-gradient

References:
- SimSiam: 2011.10566v1.pdf (Chen & He, CVPR 2021)
- Key idea: Maximize similarity between predictions and stop-gradiented targets
- No hyperparameters: no temperature, no momentum, no negative pairs :)
""" # noqa: E501

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
from audiossl.losses import SimSiamLoss
from audiossl.modeling import SimSiamMultiFormat
from audiossl.utils.wandb_utils import (
    log_audio_embeddings_ssl,
    log_audio_predictions_with_classifier,
)


def train_epoch(model, dataloader, criterion, optimizer, device, spectrogram_transform):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training (SimSiam)")
    for waveforms, _ in pbar:
        waveforms = waveforms.to(device)

        spectrograms = []
        for wav in waveforms:
            spec = spectrogram_transform(wav.unsqueeze(0).cpu())
            spectrograms.append(spec)
        spectrograms = torch.stack(spectrograms).to(device)

        optimizer.zero_grad()

        p1, p2, z1, z2, _, _ = model(waveforms, spectrograms)

        loss = criterion(p1, p2, z1, z2)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, criterion, device, spectrogram_transform):
    model.eval()
    total_loss = 0

    for waveforms, _ in tqdm(dataloader, desc="Validation (SimSiam)"):
        waveforms = waveforms.to(device)

        spectrograms = []
        for wav in waveforms:
            spec = spectrogram_transform(wav.unsqueeze(0).cpu())
            spectrograms.append(spec)
        spectrograms = torch.stack(spectrograms).to(device)

        p1, p2, z1, z2, _, _ = model(waveforms, spectrograms)

        loss = criterion(p1, p2, z1, z2)
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def linear_evaluation(
    model,
    train_loader,
    val_loader,
    device,
    spectrogram_transform,
    view="1d",
    num_epochs=20,
    sample_rate=16000,
    no_wandb=False,
):
    from audiossl.modeling import LinearClassifier

    for param in model.parameters():
        param.requires_grad = False

    classifier = LinearClassifier(input_dim=512, num_classes=10).to(device)

    optimizer = optim.Adam(classifier.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0

    for epoch in range(1, num_epochs + 1):
        classifier.train()
        correct = 0
        total = 0

        for waveforms, labels in tqdm(
            train_loader, desc=f"Linear Eval Epoch {epoch}/{num_epochs}"
        ):
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            if view == "1d":
                features = model.get_features(waveform=waveforms, view="1d")
            elif view == "2d":
                spectrograms = []
                for wav in waveforms:
                    spec = spectrogram_transform(wav.unsqueeze(0).cpu())
                    spectrograms.append(spec)
                spectrograms = torch.stack(spectrograms).to(device)
                features = model.get_features(spectrogram=spectrograms, view="2d")
            else:  # both
                spectrograms = []
                for wav in waveforms:
                    spec = spectrogram_transform(wav.unsqueeze(0).cpu())
                    spectrograms.append(spec)
                spectrograms = torch.stack(spectrograms).to(device)
                features = model.get_features(
                    waveform=waveforms, spectrogram=spectrograms, view="both"
                )

            optimizer.zero_grad()
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total

        classifier.eval()
        correct = 0
        total = 0

        for waveforms, labels in val_loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            if view == "1d":
                features = model.get_features(waveform=waveforms, view="1d")
            elif view == "2d":
                spectrograms = []
                for wav in waveforms:
                    spec = spectrogram_transform(wav.unsqueeze(0).cpu())
                    spectrograms.append(spec)
                spectrograms = torch.stack(spectrograms).to(device)
                features = model.get_features(spectrogram=spectrograms, view="2d")
            else:
                spectrograms = []
                for wav in waveforms:
                    spec = spectrogram_transform(wav.unsqueeze(0).cpu())
                    spectrograms.append(spec)
                spectrograms = torch.stack(spectrograms).to(device)
                features = model.get_features(
                    waveform=waveforms, spectrogram=spectrograms, view="both"
                )

            logits = classifier(features)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        val_acc = 100 * correct / total
        best_val_acc = max(best_val_acc, val_acc)

        print(
            f"Epoch {epoch}/{num_epochs} - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%"
        )

        if not no_wandb:
            wandb.log(
                {
                    f"linear_eval_{view}/train_acc": train_acc,
                    f"linear_eval_{view}/val_acc": val_acc,
                    f"linear_eval_{view}/epoch": epoch,
                }
            )

    # Log final predictions table to wandb
    if not no_wandb:
        print("  Logging predictions table to wandb...")
        log_audio_predictions_with_classifier(
            model=model,
            classifier=classifier,
            dataloader=val_loader,
            device=device,
            spectrogram_transform=spectrogram_transform,
            num_samples=10,
            epoch=num_epochs,
            sample_rate=sample_rate,
            view=view,
        )

    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    return best_val_acc


def plot_training_curves(history, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Loss curve
    ax.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(history["val_loss"], label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("SimSiam Loss (negative cosine similarity)", fontsize=12)
    ax.set_title("SimSiam NCL: Training and Validation Loss", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Training curves saved to {save_path}")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SimSiam Non-Contrastive Learning on AudioMNIST"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--projector-dim", type=int, default=128, help="Projector output dimension"
    )
    parser.add_argument(
        "--predictor-hidden", type=int, default=512, help="Predictor hidden dimension"
    )
    parser.add_argument(
        "--projector-hidden-dim",
        type=int,
        default=256,
        help="Projector hidden dimension",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Audio sample rate"
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=20, help="Early stopping patience"
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
        default="experiments/simsiam",
        help="Experiment directory",
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
        "--linear-eval-epochs",
        type=int,
        default=20,
        help="Number of epochs for linear evaluation",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=20,
        help="Save checkpoint every N epochs",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "method": "SimSiam Non-Contrastive Learning",
        "paper": "Chen & He, CVPR 2021 (arxiv:2011.10566)",
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "num_epochs": args.epochs,
        "projector_dim": args.projector_dim,
        "predictor_hidden": args.predictor_hidden,
        "projector_hidden_dim": args.projector_hidden_dim,
        "sample_rate": args.sample_rate,
        "early_stop_patience": args.early_stop_patience,
        "seed": args.seed,
        "notes": "No temperature, no momentum encoder, no negative pairs!",
    }

    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    if not args.no_wandb:
        wandb.init(
            project="audiomnist-ssl",
            name="simsiam-ncl",
            config=config,
            tags=["simsiam", "ncl", "ssl", "task3"],
        )

    # Create spectrogram transform
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

    print("\nCreating SimSiam multi-format model...")
    model = SimSiamMultiFormat(
        projector_hidden_dim=args.projector_hidden_dim,
        projector_output_dim=args.projector_dim,
        predictor_hidden_dim=args.predictor_hidden,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    criterion = SimSiamLoss()  # No hyperparameters!
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"\nStarting SimSiam training for {args.epochs} epochs...")
    print(f"Early stopping: patience={args.early_stop_patience}")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, spectrogram_transform
        )

        val_loss = validate(model, val_loader, criterion, device, spectrogram_transform)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")

        if not args.no_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "ssl/train_loss": train_loss,
                    "ssl/val_loss": val_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                exp_dir / "best_model.pt",
            )
            print(f"  New best model saved! Val Loss: {val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= args.early_stop_patience:
            print(
                f"\n⚠️  Early stopping triggered! No improvement for {args.early_stop_patience} epochs."
            )
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
            break

        if epoch % args.checkpoint_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                exp_dir / f"checkpoint_epoch_{epoch}.pt",
            )

            if not args.no_wandb:
                print("  Logging audio embeddings to wandb...")
                log_audio_embeddings_ssl(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    spectrogram_transform=spectrogram_transform,
                    num_samples=5,
                    epoch=epoch,
                    sample_rate=args.sample_rate,
                    view="1d",
                )
                log_audio_embeddings_ssl(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    spectrogram_transform=spectrogram_transform,
                    num_samples=5,
                    epoch=epoch,
                    sample_rate=args.sample_rate,
                    view="2d",
                )

    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, exp_dir / "training_curves.png")

    if not args.no_wandb:
        wandb.run.summary["ssl_best_val_loss"] = best_val_loss
        wandb.run.summary["ssl_best_epoch"] = best_epoch
        wandb.log(
            {"ssl_training_curves": wandb.Image(str(exp_dir / "training_curves.png"))}
        )

    print("\n" + "=" * 80)
    print("SIMSIAM TRAINING COMPLETE!")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print("=" * 80)

    print("\nPerforming linear evaluation on learned representations...")

    checkpoint = torch.load(exp_dir / "best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    acc_1d = linear_evaluation(
        model,
        train_loader,
        val_loader,
        device,
        spectrogram_transform,
        view="1d",
        num_epochs=args.linear_eval_epochs,
        sample_rate=args.sample_rate,
        no_wandb=args.no_wandb,
    )

    acc_2d = linear_evaluation(
        model,
        train_loader,
        val_loader,
        device,
        spectrogram_transform,
        view="2d",
        num_epochs=args.linear_eval_epochs,
        sample_rate=args.sample_rate,
        no_wandb=args.no_wandb,
    )

    acc_both = linear_evaluation(
        model,
        train_loader,
        val_loader,
        device,
        spectrogram_transform,
        view="both",
        num_epochs=args.linear_eval_epochs,
        sample_rate=args.sample_rate,
        no_wandb=args.no_wandb,
    )

    linear_eval_results = {
        "1d_encoder_accuracy": acc_1d,
        "2d_encoder_accuracy": acc_2d,
        "combined_accuracy": acc_both,
    }

    with open(exp_dir / "linear_eval_results.json", "w") as f:
        json.dump(linear_eval_results, f, indent=2)

    if not args.no_wandb:
        wandb.run.summary["linear_eval_1d"] = acc_1d
        wandb.run.summary["linear_eval_2d"] = acc_2d
        wandb.run.summary["linear_eval_combined"] = acc_both

        wandb.finish()

    print("\n" + "=" * 80)
    print("linear evaluation results:")
    print(f"  1D Encoder (waveform):    {acc_1d:.2f}%")
    print(f"  2D Encoder (spectrogram): {acc_2d:.2f}%")
    print(f"  Combined (both):          {acc_both:.2f}%")
    print(f"Results saved to: {exp_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
