import torch
import wandb
import numpy as np


@torch.no_grad()
def log_audio_predictions_supervised(
    model,
    dataloader,
    device,
    spectrogram_transform=None,
    num_samples=5,
    epoch=0,
    sample_rate=16000,
):
    """
    Log audio predictions to wandb for supervised model

    Args:
        model: Supervised classifier model
        dataloader: Validation dataloader
        device: torch device
        spectrogram_transform: Transform for 2D encoder (optional)
        num_samples: Number of samples to log
        epoch: Current epoch number
        sample_rate: Audio sample rate for wandb.Audio
    """
    model.eval()

    columns = ["idx", "audio", "true_label", "predicted_label", "confidence", "correct"]
    data = []

    for waveforms, labels in dataloader:
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
        else:
            spectrograms = []
            for wav in waveforms:
                spec = spectrogram_transform(wav.unsqueeze(0).cpu())
                spectrograms.append(spec)
            spectrograms = torch.stack(spectrograms).to(device)
            logits = model(waveform=waveforms, spectrogram=spectrograms)

        probs = torch.softmax(logits, dim=1)
        predicted = logits.argmax(dim=1)
        confidences = probs.max(dim=1)[0]

        for i in range(min(num_samples, len(waveforms))):
            audio_np = waveforms[i].cpu().numpy()
            true_label = labels[i].item()
            pred_label = predicted[i].item()
            confidence = confidences[i].item()
            is_correct = true_label == pred_label

            audio = wandb.Audio(audio_np, sample_rate=sample_rate)

            data.append(
                [
                    i,
                    audio,
                    true_label,
                    pred_label,
                    f"{confidence:.3f}",
                    "✓" if is_correct else "✗",
                ]
            )

        break

    table = wandb.Table(data=data, columns=columns)
    wandb.log({f"audio_predictions/epoch_{epoch}": table, "epoch": epoch})


@torch.no_grad()
def log_audio_embeddings_ssl(
    model,
    dataloader,
    device,
    spectrogram_transform,
    num_samples=5,
    epoch=0,
    sample_rate=16000,
    view="1d",
):
    """
    Log audio embeddings to wandb for SSL model (contrastive or SimSiam)
    Shows embeddings from learned representations

    Args:
        model: SSL model (contrastive or SimSiam)
        dataloader: Validation dataloader
        device: torch device
        spectrogram_transform: Transform for spectrograms
        num_samples: Number of samples to log
        epoch: Current epoch number
        sample_rate: Audio sample rate for wandb.Audio
        view: Which encoder to use ("1d", "2d", or "both")
    """
    model.eval()

    columns = ["idx", "audio", "true_label", "embedding_norm", "embedding_mean"]
    data = []

    for waveforms, labels in dataloader:
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

        for i in range(min(num_samples, len(waveforms))):
            audio_np = waveforms[i].cpu().numpy()
            true_label = labels[i].item()
            embedding = features[i].cpu().numpy()

            audio = wandb.Audio(audio_np, sample_rate=sample_rate)

            data.append(
                [
                    i,
                    audio,
                    true_label,
                    f"{np.linalg.norm(embedding):.3f}",
                    f"{embedding.mean():.3f}",
                ]
            )

        break

    table = wandb.Table(data=data, columns=columns)
    wandb.log({f"audio_embeddings_{view}/epoch_{epoch}": table, "epoch": epoch})


@torch.no_grad()
def log_audio_predictions_with_classifier(
    model,
    classifier,
    dataloader,
    device,
    spectrogram_transform,
    num_samples=5,
    epoch=0,
    sample_rate=16000,
    view="1d",
):
    """
    Log audio predictions using SSL model + linear classifier

    Args:
        model: SSL model (frozen encoder)
        classifier: Linear classifier (trained)
        dataloader: Validation dataloader
        device: torch device
        spectrogram_transform: Transform for spectrograms
        num_samples: Number of samples to log
        epoch: Current epoch number
        sample_rate: Audio sample rate for wandb.Audio
        view: Which encoder to use ("1d", "2d", or "both")
    """
    model.eval()
    classifier.eval()

    columns = ["idx", "audio", "true_label", "predicted_label", "confidence", "correct"]
    data = []

    for waveforms, labels in dataloader:
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

        logits = classifier(features)
        probs = torch.softmax(logits, dim=1)
        predicted = logits.argmax(dim=1)
        confidences = probs.max(dim=1)[0]

        for i in range(min(num_samples, len(waveforms))):
            audio_np = waveforms[i].cpu().numpy()
            true_label = labels[i].item()
            pred_label = predicted[i].item()
            confidence = confidences[i].item()
            is_correct = true_label == pred_label

            audio = wandb.Audio(audio_np, sample_rate=sample_rate)

            data.append(
                [
                    i,
                    audio,
                    true_label,
                    pred_label,
                    f"{confidence:.3f}",
                    "✓" if is_correct else "✗",
                ]
            )

        break

    table = wandb.Table(data=data, columns=columns)
    wandb.log({f"audio_predictions_{view}/epoch_{epoch}": table, "epoch": epoch})
