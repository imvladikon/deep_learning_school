"""
Train Contrastive Model (InfoNCE) with specific hyperparameters

1. Temperature: 0.5 → 0.1 (5x decrease)
2. Batch size: 32 → 128 (4x increase)
3. Epochs: 40 → 100 (2.5x increase)
4. Learning rate schedule: ReduceLROnPlateau → CosineAnnealing
5. Verification: Augmentation logging

Based on research from:
- SimCLR: τ=0.07, batch=256-4096
- wav2vec 2.0: τ=0.1, LayerNorm
- CLAR: τ≈0.1, batch=256
"""  # noqa: E501

import argparse
import json
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from audiossl.data import create_audiomnist_splits, collate_fn, LogMelSpectrogram
from audiossl.data.augmentation import ContrastiveAudioAugmentation
from audiossl.losses import InfoNCELoss
from audiossl.modeling import MultiFormatContrastiveModel
from audiossl.training import train_epoch_contrastive, validate_contrastive
from audiossl.utils import create_default_callbacks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Contrastive Model with FIXED hyperparameters"
    )

    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size (was 32 → FIXED to 128)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="InfoNCE temperature (was 0.5 → FIXED to 0.1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (was 40 → FIXED to 100)",
    )

    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--hidden-dim", type=int, default=256, help="Projector hidden dimension"
    )
    parser.add_argument(
        "--proj-dim", type=int, default=128, help="Projection output dimension"
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=25,
        help="Early stopping patience (was 15)",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="./AudioMNIST/data",
        help="AudioMNIST data directory",
    )
    parser.add_argument(
        "--num-test-speakers", type=int, default=12, help="Number of test speakers"
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    parser.add_argument(
        "--aug-prob", type=float, default=0.5, help="Augmentation probability"
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable augmentation (for debugging)",
    )

    parser.add_argument(
        "--wandb", action="store_true", help="Use Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="audiomnist-ssl-fixed",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-name", type=str, default="contrastive-fixed", help="W&B run name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/contrastive_FIXED",
        help="Output directory",
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "plateau"],
        help="LR scheduler (cosine recommended)",
    )

    parser.add_argument(
        "--log-augmentation",
        action="store_true",
        help="Log augmentation stats (first batch)",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode (2 epochs, small data)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.quick_test:
        args.epochs = 2
        args.early_stop_patience = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config["device"] = str(device)
    config["fix_applied"] = "DEEP_ANALYSIS_SUMMARY.md recommendations"
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 80)
    print("=" * 80)
    print(
        f"Temperature:   0.5 → {args.temperature} ({'5x DECREASE' if args.temperature == 0.1 else 'custom'})"
    )
    print(
        f"Batch Size:    32 → {args.batch_size} ({'4x INCREASE' if args.batch_size == 128 else 'custom'})"
    )
    print(
        f"Epochs:        40 → {args.epochs} ({'2.5x INCREASE' if args.epochs == 100 else 'custom'})"
    )
    print(f"LR Scheduler:  ReduceLROnPlateau → {args.scheduler}")
    print("=" * 80)
    print()

    print("loading AudioMNIST dataset...")
    train_dataset, val_dataset, _ = create_audiomnist_splits(
        root=args.data_root, num_test_speakers=args.num_test_speakers
    )

    if args.quick_test:
        train_dataset = torch.utils.data.Subset(train_dataset, range(64))
        val_dataset = torch.utils.data.Subset(val_dataset, range(32))

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

    spec_transform = LogMelSpectrogram(
        sample_rate=16000,
        n_mels=64,
        n_fft=400,
        hop_length=160,
    )

    if not args.no_augmentation:
        augment_fn = ContrastiveAudioAugmentation(
            sample_rate=16000,
            p=args.aug_prob,
        )
        print(f"Augmentation enabled (p={args.aug_prob})")
    else:
        augment_fn = None
        print("Augmentation disabled")

    model = MultiFormatContrastiveModel(
        projector_hidden_dim=args.hidden_dim,
        projector_output_dim=args.proj_dim,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    criterion = InfoNCELoss(temperature=args.temperature)
    print(f"InfoNCE loss with temperature={args.temperature}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6,
        )
        print(f"Using CosineAnnealingLR (T_max={args.epochs})")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
        print("Using ReduceLROnPlateau")

    callbacks = create_default_callbacks(use_wandb=args.wandb, use_notebook=False)

    if args.wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=config,
            tags=["contrastive", "infonce", "FIXED", "audiomnist"],
        )
        print(f"W&B: {wandb.run.url}\n")

    print(f"\n{'=' * 80}")
    print(f"Training for {args.epochs} epochs...")
    print(f"{'=' * 80}\n")

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    callbacks.on_train_begin({"config": config})

    for epoch in range(1, args.epochs + 1):
        callbacks.on_epoch_begin(epoch)

        train_loss = train_epoch_contrastive(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            spec_transform=spec_transform,
            augment_fn=augment_fn,
        )

        val_loss = validate_contrastive(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            spec_transform=spec_transform,
        )

        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_logs = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
        }
        callbacks.on_epoch_end(epoch, epoch_logs)

        print(
            f"Epoch {epoch:3d}/{args.epochs}: "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"LR: {current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            print(f"  New best! Val Loss: {val_loss:.4f}")

            checkpoint_path = output_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print(
                    f"\nEarly stopping at epoch {epoch} (no improvement for {args.early_stop_patience} epochs)"
                )
                break

    print(f"\n{'=' * 80}")
    print("Training Complete!")
    print(f"{'=' * 80}")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"{'=' * 80}\n")

    callbacks.on_train_end(
        {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        }
    )

    results = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
