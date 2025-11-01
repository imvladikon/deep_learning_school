"""
Visualization of Learned Embeddings

Creates t-SNE and UMAP visualizations of learned representations
to understand how well the models separate different classes.

Usage:
    python3 src/evaluation/visualize_embeddings.py \\
        --model_path experiments/supervised/best_model.pt \\
        --model_type supervised \\
        --output_dir experiments/visualizations/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from audiossl.data import create_audiomnist_splits, collate_fn, LogMelSpectrogram
from audiossl.modeling import (
    SupervisedAudioClassifier,
    MultiFormatContrastiveModel,
    SimSiamMultiFormat,
)


def load_model(model_path, model_type, device):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location=device)

    if model_type == "supervised":
        model = SupervisedAudioClassifier(encoder_type="1d", num_classes=10)
    elif model_type == "contrastive":
        model = MultiFormatContrastiveModel()
    elif model_type == "simsiam":
        model = SimSiamMultiFormat()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


@torch.no_grad()
def extract_features(
    model, dataloader, model_type, view="1d", spectrogram_transform=None, device="cuda"
):
    """
    Extract features from model

    Args:
        model: Trained model
        dataloader: DataLoader
        model_type: 'supervised', 'contrastive', or 'simsiam'
        view: '1d', '2d', or 'both' (for SSL models)
        spectrogram_transform: Transform for creating spectrograms
        device: Device to use

    Returns:
        features: (N, D) array of features
        labels: (N,) array of labels
    """
    all_features = []
    all_labels = []

    for waveforms, labels in tqdm(dataloader, desc=f"Extracting {view} features"):
        waveforms = waveforms.to(device)

        if model_type == "supervised":
            # For supervised, extract features before classifier
            if hasattr(model, "encoder_1d"):
                features = model.encoder_1d(waveforms).squeeze(-1)
            else:
                # If combined encoder
                features = model.encoder_combined(waveforms).squeeze(-1)

        elif model_type in ["contrastive", "simsiam"]:
            # For SSL models, use get_features method
            if view == "1d":
                features = model.get_features(waveform=waveforms, view="1d")
            elif view == "2d":
                # Create spectrograms
                spectrograms = []
                for wav in waveforms:
                    spec = spectrogram_transform(wav.unsqueeze(0).cpu())
                    spectrograms.append(spec)
                spectrograms = torch.stack(spectrograms).to(device)
                features = model.get_features(spectrogram=spectrograms, view="2d")
            elif view == "both":
                # Create spectrograms
                spectrograms = []
                for wav in waveforms:
                    spec = spectrogram_transform(wav.unsqueeze(0).cpu())
                    spectrograms.append(spec)
                spectrograms = torch.stack(spectrograms).to(device)
                features = model.get_features(
                    waveform=waveforms, spectrogram=spectrograms, view="both"
                )

        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return features, labels


def plot_tsne(features, labels, title, save_path, perplexity=30):
    """Create t-SNE visualization"""
    print(f"Computing t-SNE (perplexity={perplexity})...")

    # Limit to 2000 samples for faster computation
    if len(features) > 2000:
        indices = np.random.choice(len(features), 2000, replace=False)
        features = features[indices]
        labels = labels[indices]

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    embeddings_2d = tsne.fit_transform(features)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color map for 10 classes
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for digit in range(10):
        mask = labels == digit
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[digit]],
            label=str(digit),
            alpha=0.6,
            s=20,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.legend(title="Digit", fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved t-SNE plot to {save_path}")
    plt.close()


def plot_umap(features, labels, title, save_path, n_neighbors=15, min_dist=0.1):
    """Create UMAP visualization"""
    try:
        import umap
    except ImportError:
        print("⚠️  UMAP not installed. Install with: pip install umap-learn")
        return

    print(f"Computing UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")

    # Limit to 2000 samples
    if len(features) > 2000:
        indices = np.random.choice(len(features), 2000, replace=False)
        features = features[indices]
        labels = labels[indices]

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embeddings_2d = reducer.fit_transform(features)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for digit in range(10):
        mask = labels == digit
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[digit]],
            label=str(digit),
            alpha=0.6,
            s=20,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.legend(title="Digit", fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved UMAP plot to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize learned embeddings")
    parser.add_argument("--data-root", type=str, default="AudioMNIST/data")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["supervised", "contrastive", "simsiam"],
    )
    parser.add_argument(
        "--view",
        type=str,
        default="1d",
        choices=["1d", "2d", "both"],
        help="Which encoder to use (for SSL models)",
    )
    parser.add_argument("--output_dir", type=str, default="experiments/visualizations/")
    parser.add_argument(
        "--method", type=str, default="both", choices=["tsne", "umap", "both"]
    )
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EMBEDDING VISUALIZATION")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Model type: {args.model_type}")
    print(f"View: {args.view}")
    print(f"Device: {device}")

    print("\nLoading AudioMNIST dataset...")
    data_root = args.data_root
    _, val_dataset, _ = create_audiomnist_splits(root=data_root, num_test_speakers=12)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    print(f"Validation samples: {len(val_dataset)}")

    # Load model
    print("\nLoading model...")
    model = load_model(args.model_path, args.model_type, device)

    # Create spectrogram transform if needed
    spectrogram_transform = None
    if args.model_type in ["contrastive", "simsiam"] and args.view in ["2d", "both"]:
        spectrogram_transform = LogMelSpectrogram(
            sample_rate=16000, n_mels=64, n_fft=400, hop_length=160
        )

    # Extract features
    print("\nExtracting features...")
    features, labels = extract_features(
        model, val_loader, args.model_type, args.view, spectrogram_transform, device
    )

    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    # Create visualizations
    title_prefix = f"{args.model_type.capitalize()}"
    if args.model_type != "supervised":
        title_prefix += f" ({args.view} encoder)"

    # t-SNE
    if args.method in ["tsne", "both"]:
        tsne_path = output_dir / f"tsne_{args.model_type}_{args.view}.png"
        plot_tsne(features, labels, f"t-SNE: {title_prefix}", tsne_path)

    # UMAP
    if args.method in ["umap", "both"]:
        umap_path = output_dir / f"umap_{args.model_type}_{args.view}.png"
        plot_umap(features, labels, f"UMAP: {title_prefix}", umap_path)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
