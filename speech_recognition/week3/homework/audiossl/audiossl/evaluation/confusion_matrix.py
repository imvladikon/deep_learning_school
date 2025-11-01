"""
Confusion Matrix and Per-Class Analysis

Creates confusion matrices and detailed per-class performance metrics
for trained models.

Usage:
    python3 src/evaluation/confusion_matrix.py \\
        --model_path experiments/supervised/best_model.pt \\
        --model_type supervised \\
        --output_dir experiments/visualizations/
"""

import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import json

from audiossl.modeling import (
    SupervisedAudioClassifier,
    MultiFormatContrastiveModel,
    SimSiamMultiFormat,
    LinearClassifier,
)
from audiossl.data import create_audiomnist_splits, collate_fn, LogMelSpectrogram
from torch.utils.data import DataLoader


def load_model_with_classifier(model_path, model_type, device):
    """Load model with classifier for evaluation"""
    checkpoint = torch.load(model_path, map_location=device)

    if model_type == "supervised":
        model = SupervisedAudioClassifier(encoder_type="1d", num_classes=10)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        return model, None

    elif model_type in ["contrastive", "simsiam"]:
        # Load SSL model
        if model_type == "contrastive":
            base_model = MultiFormatContrastiveModel()
        else:
            base_model = SimSiamMultiFormat()

        base_model.load_state_dict(checkpoint["model_state_dict"])
        base_model = base_model.to(device)
        base_model.eval()

        # Load linear classifier (should be in same directory)
        model_dir = Path(model_path).parent
        linear_eval_path = model_dir / "linear_classifier_1d.pt"  # Use 1D by default

        if not linear_eval_path.exists():
            raise FileNotFoundError(
                f"Linear classifier not found at {linear_eval_path}. "
                "Run linear evaluation first!"
            )

        classifier = LinearClassifier(input_dim=512, num_classes=10)
        classifier_checkpoint = torch.load(linear_eval_path, map_location=device)
        classifier.load_state_dict(classifier_checkpoint["classifier_state_dict"])
        classifier = classifier.to(device)
        classifier.eval()

        return base_model, classifier

    else:
        raise ValueError(f"Unknown model type: {model_type}")


@torch.no_grad()
def evaluate_model(
    model, classifier, dataloader, model_type, spectrogram_transform, device
):
    """
    Evaluate model and return predictions and labels

    Returns:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities (for confidence analysis)
    """
    all_labels = []
    all_preds = []
    all_probs = []

    for waveforms, labels in tqdm(dataloader, desc="Evaluating"):
        waveforms = waveforms.to(device)

        if model_type == "supervised":
            # Direct prediction
            logits = model(waveform=waveforms)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

        elif model_type in ["contrastive", "simsiam"]:
            # Extract features then classify
            features = model.get_features(waveform=waveforms, view="1d")
            logits = classifier(features)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

        all_labels.append(labels.numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)

    return y_true, y_pred, y_probs


def plot_confusion_matrix(y_true, y_pred, title, save_path, normalize=True):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        square=True,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
        xticklabels=range(10),
        yticklabels=range(10),
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved confusion matrix to {save_path}")
    plt.close()


def analyze_per_class_performance(y_true, y_pred, save_path):
    """Generate and save per-class performance metrics"""
    # Get classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=[str(i) for i in range(10)],
        output_dict=True,
        zero_division=0,
    )

    # Save as JSON
    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\nPer-Class Performance:")
    print("=" * 60)
    print(
        f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}"
    )
    print("=" * 60)

    for digit in range(10):
        digit_str = str(digit)
        precision = report[digit_str]["precision"]
        recall = report[digit_str]["recall"]
        f1 = report[digit_str]["f1-score"]
        support = report[digit_str]["support"]

        print(
            f"{digit:<10} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f} {support:<10.0f}"
        )

    print("=" * 60)
    print(
        f"{'Accuracy':<10} {'':<12} {'':<12} {report['accuracy']:<12.3f} {report['macro avg']['support']:<10.0f}"
    )
    print(
        f"{'Macro Avg':<10} {report['macro avg']['precision']:<12.3f} {report['macro avg']['recall']:<12.3f} {report['macro avg']['f1-score']:<12.3f} {report['macro avg']['support']:<10.0f}"
    )
    print("=" * 60)

    print(f"\nDetailed report saved to {save_path}")


def find_misclassified_examples(y_true, y_pred, y_probs, top_k=20):
    """Find most confidently misclassified examples"""
    # Find misclassified samples
    misclassified_mask = y_true != y_pred

    if misclassified_mask.sum() == 0:
        print("No misclassifications found!")
        return []

    # Get confidence of wrong predictions
    wrong_confidences = y_probs[misclassified_mask].max(axis=1)

    # Get indices of most confident wrong predictions
    misclassified_indices = np.where(misclassified_mask)[0]

    # Sort by confidence (descending)
    sorted_idx = np.argsort(-wrong_confidences)[:top_k]

    results = []
    for idx in sorted_idx:
        sample_idx = misclassified_indices[idx]
        true_label = y_true[sample_idx]
        pred_label = y_pred[sample_idx]
        confidence = wrong_confidences[idx]

        results.append(
            {
                "sample_idx": int(sample_idx),
                "true_label": int(true_label),
                "pred_label": int(pred_label),
                "confidence": float(confidence),
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate confusion matrix and error analysis"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["supervised", "contrastive", "simsiam"],
    )
    parser.add_argument("--output_dir", type=str, default="experiments/visualizations/")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--normalize", action="store_true", help="Normalize confusion matrix"
    )

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CONFUSION MATRIX & ERROR ANALYSIS")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {device}")

    # Load data
    print("\nLoading AudioMNIST dataset...")
    data_root = "AudioMNIST/data"
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
    model, classifier = load_model_with_classifier(
        args.model_path, args.model_type, device
    )

    # Create spectrogram transform if needed
    spectrogram_transform = LogMelSpectrogram(
        sample_rate=16000, n_mels=64, n_fft=400, hop_length=160
    )

    # Evaluate
    print("\nEvaluating model...")
    y_true, y_pred, y_probs = evaluate_model(
        model, classifier, val_loader, args.model_type, spectrogram_transform, device
    )

    # Calculate accuracy
    accuracy = (y_true == y_pred).mean() * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")

    # Plot confusion matrix
    cm_path = output_dir / f"confusion_matrix_{args.model_type}.png"
    title = f"Confusion Matrix: {args.model_type.capitalize()}"
    plot_confusion_matrix(y_true, y_pred, title, cm_path, normalize=args.normalize)

    # Per-class analysis
    report_path = output_dir / f"classification_report_{args.model_type}.json"
    analyze_per_class_performance(y_true, y_pred, report_path)

    # Find misclassified examples
    print("\nMost Confident Misclassifications:")
    print("=" * 60)
    misclassified = find_misclassified_examples(y_true, y_pred, y_probs, top_k=10)

    for i, example in enumerate(misclassified, 1):
        print(
            f"{i}. Sample #{example['sample_idx']}: "
            f"True={example['true_label']}, Pred={example['pred_label']}, "
            f"Confidence={example['confidence']:.3f}"
        )

    # Save misclassifications
    misc_path = output_dir / f"misclassifications_{args.model_type}.json"
    with open(misc_path, "w") as f:
        json.dump(misclassified, f, indent=2)
    print(f"\nMisclassifications saved to {misc_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
