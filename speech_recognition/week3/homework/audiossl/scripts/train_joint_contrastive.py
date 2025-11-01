"""
Train Contrastive Model + Classifier JOINTLY

This script implements joint training (SSL + supervised simultaneously):
- Loss = α * L_contrastive + (1-α) * L_classification
- α can be constant or scheduled (e.g., start 80% SSL, end 20% SSL)
- Gradients from both objectives flow through encoder

Benefits:
- May converge faster than pure SSL + linear evaluation
- Encoder learns both semantic (SSL) and task-specific (supervised) features
- Can balance exploration (SSL) vs exploitation (supervised)

check: audiossl/training/joint_training.py
"""  # noqa: E501

import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from audiossl.data import create_audiomnist_splits, collate_fn, LogMelSpectrogram
from audiossl.data.augmentation import ContrastiveAudioAugmentation
from audiossl.modeling import MultiFormatContrastiveModel, ContrastiveWithLinearHead
from audiossl.losses import InfoNCELoss
from audiossl.training import (
    train_epoch_joint_contrastive,
    validate_joint_contrastive,
    AlphaScheduler,
)
from audiossl.utils import create_default_callbacks


def parse_args():
    parser = argparse.ArgumentParser(description="Joint Training: SSL + Supervised")

    # hparameters
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="InfoNCE temperature"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")

    # hparameters for joint training
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="SSL weight (constant mode)"
    )
    parser.add_argument(
        "--alpha-schedule",
        type=str,
        default="constant",
        choices=["constant", "linear", "cosine", "step"],
        help="Alpha scheduling strategy",
    )
    parser.add_argument(
        "--initial-alpha",
        type=float,
        default=0.7,
        help="Starting alpha (for scheduling)",
    )
    parser.add_argument(
        "--final-alpha", type=float, default=0.3, help="Ending alpha (for scheduling)"
    )

    # model related
    parser.add_argument(
        "--hidden-dim", type=int, default=256, help="Projector hidden dim"
    )
    parser.add_argument(
        "--proj-dim", type=int, default=128, help="Projection output dim"
    )
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes")

    # data related
    parser.add_argument(
        "--data-root", type=str, default="./AudioMNIST/data", help="Data directory"
    )
    parser.add_argument(
        "--num-test-speakers", type=int, default=12, help="Test speakers"
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--aug-prob", type=float, default=0.5, help="Augmentation probability"
    )

    # logging
    parser.add_argument("--wandb", action="store_true", help="Use W&B")
    parser.add_argument(
        "--wandb-project", type=str, default="audiomnist-ssl-joint", help="W&B project"
    )
    parser.add_argument(
        "--wandb-name", type=str, default="joint-contrastive", help="W&B run name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/joint_contrastive",
        help="Output dir",
    )

    # other
    parser.add_argument(
        "--early-stop-patience", type=int, default=25, help="Early stopping patience"
    )
    parser.add_argument(
        "--quick-test", action="store_true", help="Quick test (2 epochs)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.quick_test:
        args.epochs = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 80)
    print("joint training: SSL (InfoNCE) + Supervised (CrossEntropy)")
    print("=" * 80)
    print(f"Alpha schedule: {args.alpha_schedule}")
    if args.alpha_schedule == "constant":
        print(f"  α = {args.alpha} (constant)")
        print(
            f"  SSL weight: {args.alpha * 100:.0f}%, Supervised weight: {(1 - args.alpha) * 100:.0f}%"
        )
    else:
        print(
            f"  Initial α = {args.initial_alpha} ({args.initial_alpha * 100:.0f}% SSL)"
        )
        print(f"  Final α   = {args.final_alpha} ({args.final_alpha * 100:.0f}% SSL)")
        print(f"  Transition: {args.alpha_schedule}")
    print("=" * 80)
    print()

    print("Loading AudioMNIST dataset...")
    train_dataset, val_dataset, _ = create_audiomnist_splits(
        root=args.data_root, num_test_speakers=args.num_test_speakers
    )

    if args.quick_test:
        train_dataset = torch.utils.data.Subset(train_dataset, range(64))
        val_dataset = torch.utils.data.Subset(val_dataset, range(32))

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

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
        sample_rate=16000, n_mels=64, n_fft=400, hop_length=160
    )
    augment_fn = ContrastiveAudioAugmentation(sample_rate=16000, p=args.aug_prob)

    contrastive_model = MultiFormatContrastiveModel(
        projector_hidden_dim=args.hidden_dim,
        projector_output_dim=args.proj_dim,
    )

    model = ContrastiveWithLinearHead(
        contrastive_model=contrastive_model,
        num_classes=args.num_classes,
        freeze_encoder=False,
        input_dim=512,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} (encoder NOT frozen)")

    contrastive_criterion = InfoNCELoss(temperature=args.temperature)
    classification_criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    alpha_scheduler = AlphaScheduler(
        strategy=args.alpha_schedule,
        initial_alpha=args.initial_alpha
        if args.alpha_schedule != "constant"
        else args.alpha,
        final_alpha=args.final_alpha
        if args.alpha_schedule != "constant"
        else args.alpha,
        total_epochs=args.epochs,
    )

    callbacks = create_default_callbacks(use_wandb=args.wandb, use_notebook=False)

    if args.wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=config,
            tags=["joint", "contrastive", "supervised", "audiomnist"],
        )
        print(f"W&B: {wandb.run.url}\n")

    print(f"\n{'=' * 80}")
    print(f"Training for {args.epochs} epochs...")
    print(f"{'=' * 80}\n")

    best_val_acc = 0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    callbacks.on_train_begin({"config": config})

    for epoch in range(1, args.epochs + 1):
        callbacks.on_epoch_begin(epoch)

        alpha = alpha_scheduler.get_alpha(epoch - 1)

        train_loss, train_ssl, train_cls, train_acc = train_epoch_joint_contrastive(
            model=model,
            dataloader=train_loader,
            contrastive_criterion=contrastive_criterion,
            classification_criterion=classification_criterion,
            optimizer=optimizer,
            device=device,
            spec_transform=spec_transform,
            augment_fn=augment_fn,
            alpha=alpha,
        )

        val_loss, val_ssl, val_cls, val_acc = validate_joint_contrastive(
            model=model,
            dataloader=val_loader,
            contrastive_criterion=contrastive_criterion,
            classification_criterion=classification_criterion,
            device=device,
            spec_transform=spec_transform,
            alpha=alpha,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_logs = {
            "train_loss": train_loss,
            "train_ssl": train_ssl,
            "train_cls": train_cls,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_ssl": val_ssl,
            "val_cls": val_cls,
            "val_acc": val_acc,
            "learning_rate": current_lr,
            "alpha": alpha,
            "ssl_weight_pct": alpha * 100,
            "sup_weight_pct": (1 - alpha) * 100,
        }
        callbacks.on_epoch_end(epoch, epoch_logs)

        print(
            f"Epoch {epoch:3d}/{args.epochs}: "
            f"Loss={train_loss:.4f} (SSL={train_ssl:.4f}, CLS={train_cls:.4f}), "
            f"Acc={train_acc:.2f}% | "
            f"Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}%, "
            f"α={alpha:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            print(f"  New best! Val Acc: {val_acc:.2f}%")

            # Save checkpoint
            checkpoint_path = output_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "config": config,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\n{'=' * 80}")
    print("Training Complete!")
    print(f"{'=' * 80}")
    print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {output_dir}")
    print(f"{'=' * 80}\n")

    callbacks.on_train_end(
        {
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        }
    )

    results = {
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    if args.wandb:
        wandb.finish()

    print("\n✅ Done!\n")


if __name__ == "__main__":
    main()
