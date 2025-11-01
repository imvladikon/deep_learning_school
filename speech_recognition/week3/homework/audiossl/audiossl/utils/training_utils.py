from typing import Optional, Callable, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json

from audiossl.utils.callbacks import CallbackList, create_default_callbacks


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    train_step_fn: Callable,
    val_step_fn: Callable,
    scheduler: Optional[Any] = None,
    callbacks: Optional[CallbackList] = None,
    early_stop_patience: int = 15,
    save_dir: Optional[Path] = None,
    use_wandb: bool = False,
    use_notebook: Optional[bool] = None,
) -> dict:
    """
    Generic training loop with callback support.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of epochs
        train_step_fn: Function to execute training step
                       Should have signature: (model, batch, criterion, device) -> (loss, metrics)
        val_step_fn: Function to execute validation step
                     Should have signature: (model, batch, criterion, device) -> (loss, metrics)
        scheduler: Learning rate scheduler (optional)
        callbacks: CallbackList for custom behavior (optional)
        early_stop_patience: Patience for early stopping
        save_dir: Directory to save checkpoints (optional)
        use_wandb: Whether to use wandb (only if callbacks not provided)
        use_notebook: Whether to use notebook visualization (only if callbacks not provided)

    Returns:
        Dictionary with training history
    """
    # Create default callbacks if not provided
    if callbacks is None:
        callbacks = create_default_callbacks(
            use_wandb=use_wandb,
            use_notebook=use_notebook,
        )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0
    best_epoch = 0
    epochs_without_improvement = 0

    train_begin_logs = {
        "config": {
            "num_epochs": num_epochs,
            "early_stop_patience": early_stop_patience,
        }
    }

    callbacks.on_train_begin(train_begin_logs)

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Early stopping: patience={early_stop_patience}")
    print("=" * 80)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        callbacks.on_epoch_begin(epoch)

        train_loss, train_acc = train_step_fn(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = val_step_fn(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_logs = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        callbacks.on_epoch_end(epoch, epoch_logs)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0

            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                    },
                    save_dir / "best_model.pt",
                )
                print(f"  ✓ New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= early_stop_patience:
            print(
                f"\n⚠️  Early stopping triggered! No improvement for {early_stop_patience} epochs."
            )
            print(
                f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}"
            )
            break

        if save_dir is not None and epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                save_dir / f"checkpoint_epoch_{epoch}.pt",
            )

    if save_dir is not None:
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    train_end_logs = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "history": history,
    }
    callbacks.on_train_end(train_end_logs)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    if save_dir is not None:
        print(f"Results saved to: {save_dir}")
    print("=" * 80)

    return history
