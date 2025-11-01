"""
SimSiam (Non-Contrastive Learning) training functions.
"""

import torch
from tqdm.auto import tqdm


def train_epoch_simsiam(model, dataloader, criterion, optimizer, device, spec_transform, augment_fn=None):
    """
    Train SimSiam model for one epoch.

    Args:
        model: SimSiamMultiFormat
        dataloader: Training dataloader
        criterion: SimSiamLoss
        optimizer: Optimizer
        device: Device to train on
        spec_transform: LogMelSpectrogram transform
        augment_fn: Optional audio augmentation function (ContrastiveAudioAugmentation)

    Returns:
        Average loss
    """
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training (SimSiam)", leave=False)
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
        p1, p2, z1, z2, audio_feat, spec_feat = model(view1, spectrograms)

        # SimSiam loss (negative cosine similarity with stop-gradient)
        # Only uses p1, p2, z1, z2; audio_feat and spec_feat are for linear eval
        loss = criterion(p1, p2, z1, z2)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def validate_simsiam(model, dataloader, criterion, device, spec_transform):
    """
    Validate SimSiam model.

    Args:
        model: SimSiamMultiFormat
        dataloader: Validation dataloader
        criterion: SimSiamLoss
        device: Device to evaluate on
        spec_transform: LogMelSpectrogram transform

    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0

    for waveforms, _ in tqdm(dataloader, desc="Validation (SimSiam)", leave=False):
        waveforms = waveforms.to(device)

        spectrograms = []
        for wav in waveforms:
            spec = spec_transform(wav.unsqueeze(0).cpu())
            spectrograms.append(spec)
        spectrograms = torch.stack(spectrograms).to(device)

        p1, p2, z1, z2, audio_feat, spec_feat = model(waveforms, spectrograms)
        loss = criterion(p1, p2, z1, z2)
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss
