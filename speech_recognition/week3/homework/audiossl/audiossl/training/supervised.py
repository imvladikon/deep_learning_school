"""
Supervised training functions.
"""

import torch
from tqdm.auto import tqdm


def train_epoch_supervised(model, dataloader, criterion, optimizer, device, spec_transform=None):
    """
    Train supervised model for one epoch.

    Args:
        model: SupervisedAudioClassifier model
        dataloader: Training dataloader
        criterion: Loss function (e.g., CrossEntropyLoss)
        optimizer: Optimizer
        device: Device to train on
        spec_transform: Optional LogMelSpectrogram transform for 2D/both encoders

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for waveforms, labels in pbar:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass based on encoder type
        if model.encoder_type == "1d":
            logits = model(waveform=waveforms)
        elif model.encoder_type == "2d":
            spectrograms = []
            for wav in waveforms:
                spec = spec_transform(wav.unsqueeze(0).cpu())
                spectrograms.append(spec)
            spectrograms = torch.stack(spectrograms).to(device)
            logits = model(spectrogram=spectrograms)
        else:  # both
            spectrograms = []
            for wav in waveforms:
                spec = spec_transform(wav.unsqueeze(0).cpu())
                spectrograms.append(spec)
            spectrograms = torch.stack(spectrograms).to(device)
            logits = model(waveform=waveforms, spectrogram=spectrograms)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # Track metrics
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
def validate_supervised(model, dataloader, criterion, device, spec_transform=None):
    """
    Validate supervised model.

    Args:
        model: SupervisedAudioClassifier model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to evaluate on
        spec_transform: Optional LogMelSpectrogram transform for 2D/both encoders

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for waveforms, labels in tqdm(dataloader, desc="Validation", leave=False):
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        # Forward pass based on encoder type
        if model.encoder_type == "1d":
            logits = model(waveform=waveforms)
        elif model.encoder_type == "2d":
            spectrograms = []
            for wav in waveforms:
                spec = spec_transform(wav.unsqueeze(0).cpu())
                spectrograms.append(spec)
            spectrograms = torch.stack(spectrograms).to(device)
            logits = model(spectrogram=spectrograms)
        else:  # both
            spectrograms = []
            for wav in waveforms:
                spec = spec_transform(wav.unsqueeze(0).cpu())
                spectrograms.append(spec)
            spectrograms = torch.stack(spectrograms).to(device)
            logits = model(waveform=waveforms, spectrogram=spectrograms)

        loss = criterion(logits, labels)

        # Track metrics
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy
