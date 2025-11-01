import torch
from tqdm.auto import tqdm


def train_epoch_linear_head(
    model, dataloader, criterion, optimizer, device, spec_transform=None, view="1d"
):
    """
    Train linear classifier on top of frozen SSL encoder for one epoch.

    Args:
        model: ContrastiveWithLinearHead or NCLWithLinearHead
        dataloader: Training dataloader
        criterion: Loss function (e.g., CrossEntropyLoss)
        optimizer: Optimizer (only classifier parameters)
        device: Device to train on
        spec_transform: Optional LogMelSpectrogram transform for 2D view
        view: "1d" or "2d" - which encoder to use

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    # Encoder stays frozen, only classifier trains
    if hasattr(model, "freeze_encoder") and model.freeze_encoder:
        if hasattr(model, "contrastive_model"):
            model.contrastive_model.eval()
        elif hasattr(model, "ncl_model"):
            model.ncl_model.eval()

    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Training Linear Head ({view})", leave=False)
    for waveforms, labels in pbar:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Prepare inputs based on view
        if view == "1d":
            logits = model(waveform=waveforms, view="1d")
        elif view == "2d":
            spectrograms = []
            for wav in waveforms:
                spec = spec_transform(wav.unsqueeze(0).cpu())
                spectrograms.append(spec)
            spectrograms = torch.stack(spectrograms).to(device)
            logits = model(spectrogram=spectrograms, view="2d")
        else:
            raise ValueError(f"view must be '1d' or '2d', got {view}")

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = logits.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        acc = 100.0 * correct / total
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.2f}%"})

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate_linear_head(
    model, dataloader, criterion, device, spec_transform=None, view="1d"
):
    """
    Validate linear classifier on top of frozen SSL encoder.

    Args:
        model: ContrastiveWithLinearHead or NCLWithLinearHead
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to evaluate on
        spec_transform: Optional LogMelSpectrogram transform for 2D view
        view: "1d" or "2d"

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    for waveforms, labels in tqdm(
        dataloader, desc=f"Validation Linear Head ({view})", leave=False
    ):
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        # Prepare inputs based on view
        if view == "1d":
            logits = model(waveform=waveforms, view="1d")
        elif view == "2d":
            spectrograms = []
            for wav in waveforms:
                spec = spec_transform(wav.unsqueeze(0).cpu())
                spectrograms.append(spec)
            spectrograms = torch.stack(spectrograms).to(device)
            logits = model(spectrogram=spectrograms, view="2d")
        else:
            raise ValueError(f"view must be '1d' or '2d', got {view}")

        loss = criterion(logits, labels)
        total_loss += loss.item()

        predicted = logits.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy
