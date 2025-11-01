"""
Relevant for
Task 4: Model Comparison and Analysis

Compare all three approaches:
1. Supervised baseline
2. InfoNCE contrastive learning
3. SimSiam non-contrastive learning

Metrics:
- Classification accuracy (train/val)
- Training efficiency (epochs to convergence)
- Representation quality (linear probe accuracy)
- Model complexity (parameters, FLOPs)
""" # noqa: E501
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def load_experiment_results(exp_dir):
    """Load results from an experiment directory"""
    exp_dir = Path(exp_dir)

    with open(exp_dir / "config.json", "r") as f:
        config = json.load(f)

    # training history
    with open(exp_dir / "history.json", "r") as f:
        history = json.load(f)

    # linear eval results if available (for SSL methods)
    linear_eval = None
    if (exp_dir / "linear_eval_results.json").exists():
        with open(exp_dir / "linear_eval_results.json", "r") as f:
            linear_eval = json.load(f)

    return {"config": config, "history": history, "linear_eval": linear_eval}


def plot_training_curves(results_dict, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for method, results in results_dict.items():
        history = results["history"]
        epochs = range(1, len(history["train_loss"]) + 1)
        ax.plot(
            epochs,
            history["train_loss"],
            label=f"{method} (train)",
            linewidth=2,
            alpha=0.7,
        )
        ax.plot(
            epochs,
            history["val_loss"],
            label=f"{method} (val)",
            linewidth=2,
            linestyle="--",
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for method, results in results_dict.items():
        history = results["history"]
        if "train_acc" in history:  # Supervised has accuracy
            epochs = range(1, len(history["train_acc"]) + 1)
            ax.plot(
                epochs,
                history["train_acc"],
                label=f"{method} (train)",
                linewidth=2,
                alpha=0.7,
            )
            ax.plot(
                epochs,
                history["val_acc"],
                label=f"{method} (val)",
                linewidth=2,
                linestyle="--",
            )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Classification Accuracy (Supervised)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Training curves saved to {save_path}")
    plt.close()


def plot_linear_probe_comparison(results_dict, save_path):
    data = []

    for method, results in results_dict.items():
        if results["linear_eval"]:
            data.append(
                {
                    "Method": method,
                    "Encoder": "1D (waveform)",
                    "Accuracy": results["linear_eval"]["1d_encoder_accuracy"],
                }
            )
            data.append(
                {
                    "Method": method,
                    "Encoder": "2D (spectrogram)",
                    "Accuracy": results["linear_eval"]["2d_encoder_accuracy"],
                }
            )
            data.append(
                {
                    "Method": method,
                    "Encoder": "Combined (both)",
                    "Accuracy": results["linear_eval"]["combined_accuracy"],
                }
            )

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df["Encoder"].unique()))
    width = 0.25
    methods = df["Method"].unique()

    for i, method in enumerate(methods):
        method_data = df[df["Method"] == method]
        accuracies = method_data["Accuracy"].values
        ax.bar(x + i * width, accuracies, width, label=method, alpha=0.8)

        for j, acc in enumerate(accuracies):
            ax.text(
                x[j] + i * width,
                acc + 0.5,
                f"{acc:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Encoder Type", fontsize=12, fontweight="bold")
    ax.set_ylabel("Linear Probe Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "SSL Methods: Linear Evaluation Protocol", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels(df["Encoder"].unique())
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Linear probe comparison saved to {save_path}")
    plt.close()


def create_comparison_table(results_dict):
    rows = []

    for method, results in results_dict.items():
        config = results["config"]
        history = results["history"]

        if "val_acc" in history:  # Supervised
            best_val = max(history["val_acc"])
            metric_name = "Val Accuracy"
        else:  # SSL
            best_val = min(history["val_loss"])
            metric_name = "Val Loss"

        epochs = len(history["train_loss"])

        linear_eval_str = "N/A"
        if results["linear_eval"]:
            best_linear = max(
                [
                    results["linear_eval"]["1d_encoder_accuracy"],
                    results["linear_eval"]["2d_encoder_accuracy"],
                    results["linear_eval"]["combined_accuracy"],
                ]
            )
            linear_eval_str = f"{best_linear:.2f}%"

        row = {
            "Method": method,
            "Learning Type": config.get("method", "Supervised"),
            "Epochs": epochs,
            f"Best {metric_name}": f"{best_val:.4f}"
            if "loss" in metric_name.lower()
            else f"{best_val:.2f}%",
            "Best Linear Probe": linear_eval_str,
            "Batch Size": config.get("batch_size", "N/A"),
            "Learning Rate": config.get("learning_rate", "N/A"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def generate_report(results_dict, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MULTI-FORMAT SSL AUDIO CLASSIFICATION: MODEL COMPARISON")
    print("=" * 80)

    comparison_df = create_comparison_table(results_dict)
    print("\n" + comparison_df.to_string(index=False))

    comparison_df.to_csv(output_dir / "comparison_table.csv", index=False)
    print(f"\nComparison table saved to {output_dir / 'comparison_table.csv'}")

    plot_training_curves(results_dict, output_dir / "training_comparison.png")

    ssl_results = {k: v for k, v in results_dict.items() if v["linear_eval"]}
    if ssl_results:
        plot_linear_probe_comparison(
            ssl_results, output_dir / "linear_probe_comparison.png"
        )

    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)

    for method, results in results_dict.items():
        print(f"\n{method}:")
        if results["linear_eval"]:
            print(
                f"  - Best linear probe accuracy: {max(results['linear_eval'].values()):.2f}%"
            )
            print(
                f"  - 1D encoder: {results['linear_eval']['1d_encoder_accuracy']:.2f}%"
            )
            print(
                f"  - 2D encoder: {results['linear_eval']['2d_encoder_accuracy']:.2f}%"
            )
            print(f"  - Combined: {results['linear_eval']['combined_accuracy']:.2f}%")
        else:
            if "val_acc" in results["history"]:
                print(
                    f"  - Best validation accuracy: {max(results['history']['val_acc']):.2f}%"
                )

    print("\n" + "=" * 80)
    print(f"All results saved to: {output_dir}")
    print("=" * 80)


def main():
    experiments = {
        "Supervised": "experiments/supervised",
        "InfoNCE (Contrastive)": "experiments/contrastive",
        "SimSiam (NCL)": "experiments/simsiam",
    }

    results_dict = {}
    for method, exp_dir in experiments.items():
        exp_path = Path(exp_dir)
        if exp_path.exists() and (exp_path / "history.json").exists():
            print(f"Loading {method} results from {exp_dir}...")
            results_dict[method] = load_experiment_results(exp_dir)
        else:
            print(f"Warning: {method} results not found at {exp_dir}")

    if not results_dict:
        print("No experiment results found. Train models first!")
        return

    output_dir = Path("experiments/comparison")
    generate_report(results_dict, output_dir)


if __name__ == "__main__":
    main()
