"""
Contrastive learning (InfoNCE) training functions.
"""

import torch
from tqdm.auto import tqdm


def train_epoch_contrastive(model, dataloader, criterion, optimizer, device, spec_transform, augment_fn=None):
    """
    Train contrastive model (InfoNCE) for one epoch.

    Args:
        model: MultiFormatContrastiveModel
        dataloader: Training dataloader
        criterion: InfoNCELoss
        optimizer: Optimizer
        device: Device to train on
        spec_transform: LogMelSpectrogram transform
        augment_fn: Optional audio augmentation function (ContrastiveAudioAugmentation)

    Returns:
        Average loss
    """
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training (Contrastive)", leave=False)
    for waveforms, _ in pbar:
        waveforms = waveforms.to(device)

        # Apply augmentation to create two different views
        if augment_fn is not None:
            # View 1: Augmented waveform
            view1 = augment_fn(waveforms)

            # View 2: Different augmentation (for diversity)
            view2_wav = augment_fn(waveforms)
        else:
            # Fallback: no augmentation (old behavior)
            view1 = waveforms
            view2_wav = waveforms

        # Create spectrograms for view2
        spectrograms = []
        for wav in view2_wav:
            spec = spec_transform(wav.unsqueeze(0).cpu())
            spectrograms.append(spec)
        spectrograms = torch.stack(spectrograms).to(device)

        optimizer.zero_grad()

        # Forward pass with augmented views
        audio_emb, spec_emb, audio_feat, spec_feat = model(view1, spectrograms)

        # Contrastive loss (only use embeddings, not features)
        loss = criterion(audio_emb, spec_emb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def validate_contrastive(model, dataloader, criterion, device, spec_transform):
    """
    Validate contrastive model.

    Args:
        model: MultiFormatContrastiveModel
        dataloader: Validation dataloader
        criterion: InfoNCELoss
        device: Device to evaluate on
        spec_transform: LogMelSpectrogram transform

    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0

    for waveforms, _ in tqdm(dataloader, desc="Validation (Contrastive)", leave=False):
        waveforms = waveforms.to(device)

        spectrograms = []
        for wav in waveforms:
            spec = spec_transform(wav.unsqueeze(0).cpu())
            spectrograms.append(spec)
        spectrograms = torch.stack(spectrograms).to(device)

        audio_emb, spec_emb, audio_feat, spec_feat = model(waveforms, spectrograms)
        loss = criterion(audio_emb, spec_emb)
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss
