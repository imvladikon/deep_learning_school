"""
Joint Training: Contrastive SSL + Supervised Classification

Train encoder with both self-supervised (InfoNCE) and supervised (CrossEntropy) objectives
simultaneously. This allows gradients from both tasks to flow through the encoder.

Benefits:
- Encoder learns both semantic representations (SSL) and task-specific features (supervised)
- May converge faster than pure SSL + linear evaluation
- Can balance exploration (SSL) with exploitation (supervised)

References:
- "Understanding Self-supervised Learning with Dual Deep Networks" (Tian et al., 2021)
- "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)
- Semi-supervised learning approaches in SimCLR, MoCo

Implementation inspired by semi-supervised learning protocols.
"""

import torch
import torch.nn as nn
from tqdm.auto import tqdm


def train_epoch_joint_contrastive(
    model,
    dataloader,
    contrastive_criterion,
    classification_criterion,
    optimizer,
    device,
    spec_transform,
    augment_fn=None,
    alpha=0.5,
    log_interval=None,
):
    """
    Train contrastive model + linear head jointly with both objectives

    Args:
        model: ContrastiveWithLinearHead (with freeze_encoder=False!)
        dataloader: Training dataloader with (waveform, label) pairs
        contrastive_criterion: InfoNCELoss for SSL
        classification_criterion: CrossEntropyLoss for supervised
        optimizer: Optimizer for ALL parameters (encoder + head)
        device: Device to train on
        spec_transform: LogMelSpectrogram transform
        augment_fn: Optional audio augmentation function
        alpha: Weight for contrastive loss (1-alpha for classification)
               alpha=1.0 → pure SSL, alpha=0.0 → pure supervised
               alpha=0.5 → 50-50 balance (recommended start)
        log_interval: If set, log detailed stats every N batches

    Returns:
        avg_loss: Average total loss
        avg_contrastive: Average contrastive loss
        avg_classification: Average classification loss
        avg_acc: Average classification accuracy

    Loss:
        L_total = α * L_contrastive + (1-α) * L_classification

    Note:
        - Encoder must NOT be frozen (set freeze_encoder=False)
        - Both losses backpropagate through encoder
        - Classification head only receives classification gradients
    """
    if model.freeze_encoder:
        raise ValueError(
            "Encoder is frozen! Set freeze_encoder=False for joint training."
        )

    model.train()

    total_loss = 0
    total_contrastive_loss = 0
    total_classification_loss = 0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc="Training (Joint)", leave=False)

    for batch_idx, (waveforms, labels) in enumerate(pbar):
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        # ========================================
        # Create augmented views for contrastive learning
        # ========================================
        if augment_fn is not None:
            # View 1: Augmented waveform (1D)
            view1 = augment_fn(waveforms)

            # View 2: Different augmentation (for diversity)
            view2_wav = augment_fn(waveforms)
        else:
            # Fallback: no augmentation
            view1 = waveforms
            view2_wav = waveforms

        # Create spectrograms for view2 (2D)
        spectrograms = []
        for wav in view2_wav:
            spec = spec_transform(wav.unsqueeze(0).cpu())
            spectrograms.append(spec)
        spectrograms = torch.stack(spectrograms).to(device)

        optimizer.zero_grad()

        # ========================================
        # Forward pass: Get embeddings AND features
        # ========================================
        # Contrastive model outputs:
        # - audio_emb, spec_emb: [B, proj_dim] for contrastive loss
        # - audio_feat, spec_feat: [B, 512] for classification
        audio_emb, spec_emb, audio_feat, spec_feat = model.contrastive_model(
            view1, spectrograms
        )

        # ========================================
        # Compute contrastive loss (SSL objective)
        # ========================================
        loss_contrastive = contrastive_criterion(audio_emb, spec_emb)

        # ========================================
        # Compute classification loss (supervised objective)
        # ========================================
        # Use 1D features (waveform encoder) for classification
        # Could also use 2D features or concatenation
        logits = model.classifier(audio_feat)
        loss_classification = classification_criterion(logits, labels)

        # ========================================
        # Combined loss
        # ========================================
        loss = alpha * loss_contrastive + (1 - alpha) * loss_classification

        # ========================================
        # Backward pass & optimization
        # ========================================
        loss.backward()
        optimizer.step()

        # ========================================
        # Track metrics
        # ========================================
        total_loss += loss.item()
        total_contrastive_loss += loss_contrastive.item()
        total_classification_loss += loss_classification.item()

        # Classification accuracy
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        # Update progress bar
        current_acc = 100.0 * total_correct / total_samples
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "ssl": f"{loss_contrastive.item():.4f}",
                "cls": f"{loss_classification.item():.4f}",
                "acc": f"{current_acc:.2f}%",
            }
        )

        # Detailed logging (optional)
        if log_interval is not None and (batch_idx + 1) % log_interval == 0:
            print(
                f"\n[Batch {batch_idx+1}] "
                f"Loss={loss.item():.4f} "
                f"(α*SSL={(alpha * loss_contrastive.item()):.4f} + "
                f"(1-α)*CLS={((1-alpha) * loss_classification.item()):.4f}), "
                f"Acc={current_acc:.2f}%"
            )

    # Compute epoch averages
    avg_loss = total_loss / len(dataloader)
    avg_contrastive = total_contrastive_loss / len(dataloader)
    avg_classification = total_classification_loss / len(dataloader)
    avg_acc = 100.0 * total_correct / total_samples

    return avg_loss, avg_contrastive, avg_classification, avg_acc


@torch.no_grad()
def validate_joint_contrastive(
    model, dataloader, contrastive_criterion, classification_criterion, device, spec_transform, alpha=0.5
):
    """
    Validate contrastive model + linear head jointly

    Args:
        model: ContrastiveWithLinearHead
        dataloader: Validation dataloader
        contrastive_criterion: InfoNCELoss
        classification_criterion: CrossEntropyLoss
        device: Device to evaluate on
        spec_transform: LogMelSpectrogram transform
        alpha: Weight for contrastive loss (same as training)

    Returns:
        avg_loss: Average total loss
        avg_contrastive: Average contrastive loss
        avg_classification: Average classification loss
        avg_acc: Average classification accuracy
    """
    model.eval()

    total_loss = 0
    total_contrastive_loss = 0
    total_classification_loss = 0
    total_correct = 0
    total_samples = 0

    for waveforms, labels in tqdm(dataloader, desc="Validation (Joint)", leave=False):
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        # No augmentation during validation
        view1 = waveforms
        view2_wav = waveforms

        # Create spectrograms
        spectrograms = []
        for wav in view2_wav:
            spec = spec_transform(wav.unsqueeze(0).cpu())
            spectrograms.append(spec)
        spectrograms = torch.stack(spectrograms).to(device)

        # Forward pass
        audio_emb, spec_emb, audio_feat, spec_feat = model.contrastive_model(
            view1, spectrograms
        )

        # Contrastive loss
        loss_contrastive = contrastive_criterion(audio_emb, spec_emb)

        # Classification loss
        logits = model.classifier(audio_feat)
        loss_classification = classification_criterion(logits, labels)

        # Combined loss
        loss = alpha * loss_contrastive + (1 - alpha) * loss_classification

        # Track metrics
        total_loss += loss.item()
        total_contrastive_loss += loss_contrastive.item()
        total_classification_loss += loss_classification.item()

        # Accuracy
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    # Compute averages
    avg_loss = total_loss / len(dataloader)
    avg_contrastive = total_contrastive_loss / len(dataloader)
    avg_classification = total_classification_loss / len(dataloader)
    avg_acc = 100.0 * total_correct / total_samples

    return avg_loss, avg_contrastive, avg_classification, avg_acc


# ========================================
# Annealing Strategies for Alpha
# ========================================


class AlphaScheduler:
    """
    Schedule for balancing SSL vs supervised objectives

    Start with more SSL (exploration), gradually increase supervised (exploitation)
    """

    def __init__(self, strategy="constant", initial_alpha=0.5, final_alpha=0.5, total_epochs=100):
        """
        Args:
            strategy: "constant", "linear", "cosine", "step"
            initial_alpha: Starting alpha value (epoch 0)
            final_alpha: Ending alpha value (last epoch)
            total_epochs: Total training epochs
        """
        self.strategy = strategy
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.total_epochs = total_epochs

    def get_alpha(self, epoch):
        """Get alpha for current epoch"""
        if self.strategy == "constant":
            return self.initial_alpha

        elif self.strategy == "linear":
            # Linear interpolation from initial to final
            progress = epoch / self.total_epochs
            return self.initial_alpha + (self.final_alpha - self.initial_alpha) * progress

        elif self.strategy == "cosine":
            # Cosine annealing (smooth transition)
            import math

            progress = epoch / self.total_epochs
            cosine_factor = (1 + math.cos(math.pi * progress)) / 2
            return self.final_alpha + (self.initial_alpha - self.final_alpha) * cosine_factor

        elif self.strategy == "step":
            # Step decay at 50% and 75% of training
            if epoch < self.total_epochs * 0.5:
                return self.initial_alpha
            elif epoch < self.total_epochs * 0.75:
                return (self.initial_alpha + self.final_alpha) / 2
            else:
                return self.final_alpha

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


# ========================================
# Example Usage
# ========================================

if __name__ == "__main__":
    """
    Example usage of joint training

    This demonstrates how to train with both SSL and supervised objectives.
    """
    print("Joint Training Example")
    print("=" * 80)

    # Example: Start with more SSL, gradually transition to supervised
    scheduler = AlphaScheduler(
        strategy="cosine",
        initial_alpha=0.8,  # 80% SSL at start
        final_alpha=0.2,  # 20% SSL at end (80% supervised)
        total_epochs=100,
    )

    print("\nAlpha schedule (SSL weight):")
    for epoch in [0, 25, 50, 75, 99]:
        alpha = scheduler.get_alpha(epoch)
        print(
            f"  Epoch {epoch:3d}: α={alpha:.3f} "
            f"(SSL weight={(alpha*100):.1f}%, "
            f"Supervised weight={((1-alpha)*100):.1f}%)"
        )

    print("\n" + "=" * 80)
    print("Usage in training loop:")
    print(
        """
    from audiossl.training.joint_training import (
        train_epoch_joint_contrastive,
        validate_joint_contrastive,
        AlphaScheduler,
    )
    from audiossl.modeling import ContrastiveWithLinearHead
    from audiossl.losses import InfoNCELoss
    import torch.nn as nn

    # Create model (encoder NOT frozen!)
    model = ContrastiveWithLinearHead(
        contrastive_model=your_contrastive_model,
        num_classes=10,
        freeze_encoder=False,  # ← IMPORTANT!
    ).to(device)

    # Create criteria
    contrastive_criterion = InfoNCELoss(temperature=0.1)
    classification_criterion = nn.CrossEntropyLoss()

    # Optimizer for ALL parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Alpha scheduler
    alpha_scheduler = AlphaScheduler(
        strategy="cosine",
        initial_alpha=0.7,  # Start with 70% SSL
        final_alpha=0.3,    # End with 30% SSL (70% supervised)
        total_epochs=100,
    )

    # Training loop
    for epoch in range(100):
        alpha = alpha_scheduler.get_alpha(epoch)

        loss, loss_ssl, loss_cls, acc = train_epoch_joint_contrastive(
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

        print(f"Epoch {epoch}: "
              f"Train Loss={loss:.4f} (SSL={loss_ssl:.4f}, CLS={loss_cls:.4f}, Acc={acc:.2f}%), "
              f"Val Loss={val_loss:.4f} (Acc={val_acc:.2f}%), "
              f"α={alpha:.3f}")
    """
    )

    print("\n" + "=" * 80)
    print("See DEEP_ANALYSIS_SUMMARY.md for more details!")
