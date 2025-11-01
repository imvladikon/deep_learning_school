"""
Compare Results from All Experiments

This script compares results from all three tasks:
- Task 1: Supervised learning
- Task 2: Contrastive learning (InfoNCE)
- Task 3: Non-contrastive learning (SimSiam)

Creates comparison tables and plots.
"""  # noqa: E501

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_experiment_results(exp_dir: Path) -> Dict:
    if not exp_dir.exists():
        return None

    results = {
        "name": exp_dir.name,
        "path": str(exp_dir),
    }

    config_path = exp_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            results["config"] = json.load(f)

    history_path = exp_dir / "history.json"
    if history_path.exists():
        with open(history_path, "r") as f:
            results["history"] = json.load(f)

            if "val_acc" in results["history"]:
                val_accs = results["history"]["val_acc"]
                best_idx = np.argmax(val_accs)
                results["best_val_acc"] = val_accs[best_idx]
                results["best_epoch"] = best_idx + 1
                results["final_train_acc"] = results["history"]["train_acc"][best_idx]
                results["final_val_loss"] = results["history"]["val_loss"][best_idx]

    return results


def create_comparison_table(all_results: List[Dict]):
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(
        f"{'Experiment':<30} {'Best Val Acc':<15} {'Best Epoch':<12} {'Train Acc':<12}"
    )
    print("-" * 80)

    for result in all_results:
        if result and "best_val_acc" in result:
            name = result["name"]
            val_acc = result["best_val_acc"]
            epoch = result["best_epoch"]
            train_acc = result.get("final_train_acc", 0)
            print(f"{name:<30} {val_acc:>13.2f}% {epoch:>11} {train_acc:>10.2f}%")

    print("=" * 80)


def plot_training_curves_comparison(all_results: List[Dict], save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    results_with_history = [r for r in all_results if r and "history" in r]

    if not results_with_history:
        print("No training history found to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Training Loss
    ax = axes[0, 0]
    for result in results_with_history:
        history = result["history"]
        if "train_loss" in history:
            epochs = range(1, len(history["train_loss"]) + 1)
            ax.plot(epochs, history["train_loss"], label=result["name"], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    ax = axes[0, 1]
    for result in results_with_history:
        history = result["history"]
        if "val_loss" in history:
            epochs = range(1, len(history["val_loss"]) + 1)
            ax.plot(epochs, history["val_loss"], label=result["name"], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Validation Loss Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Training Accuracy
    ax = axes[1, 0]
    for result in results_with_history:
        history = result["history"]
        if "train_acc" in history:
            epochs = range(1, len(history["train_acc"]) + 1)
            ax.plot(epochs, history["train_acc"], label=result["name"], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training Accuracy Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Validation Accuracy
    ax = axes[1, 1]
    for result in results_with_history:
        history = result["history"]
        if "val_acc" in history:
            epochs = range(1, len(history["val_acc"]) + 1)
            ax.plot(epochs, history["val_acc"], label=result["name"], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Validation Accuracy Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / "training_curves_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to: {save_path}")
    plt.close()


def plot_final_accuracy_bar(all_results: List[Dict], save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    results_with_acc = [r for r in all_results if r and "best_val_acc" in r]
    results_with_acc.sort(key=lambda x: x["best_val_acc"], reverse=True)

    if not results_with_acc:
        print("No accuracy data found to plot")
        return

    names = [r["name"] for r in results_with_acc]
    accs = [r["best_val_acc"] for r in results_with_acc]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(names)), accs, color="steelblue", alpha=0.7)

    for i, (bar, acc) in enumerate(zip(bars, accs)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Final Validation Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = save_dir / "final_accuracy_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Accuracy comparison saved to: {save_path}")
    plt.close()


def main():
    project_root = Path(__file__).parent.parent
    exp_base = project_root / "experiments"
    results_dir = project_root / "results"

    print("=" * 80)
    print("COMPARING RESULTS")
    print("=" * 80)
    print(f"Looking for experiments in: {exp_base}")
    print()

    experiment_paths = [
        # Task 1: Supervised
        exp_base / "task1_supervised" / "supervised_1d",
        exp_base / "task1_supervised" / "supervised_2d",
        exp_base / "task1_supervised" / "supervised_both",
        # Task 2: Contrastive
        exp_base / "task2_contrastive" / "contrastive_infonce",
        # Task 3: NCL
        exp_base / "task3_ncl" / "simsiam",
    ]

    all_results = []
    for exp_path in experiment_paths:
        print(f"Loading: {exp_path.name}...", end=" ")
        result = load_experiment_results(exp_path)
        if result:
            all_results.append(result)
            print("✓")
        else:
            print("✗ (not found)")

    if not all_results:
        print("\nNo experiment results found!")
        print("Please run experiments first:")
        print("  bash scripts/run_experiments/07_run_all_experiments.sh")
        return

    create_comparison_table(all_results)

    print("\nGenerating comparison plots...")
    plot_training_curves_comparison(all_results, results_dir)
    plot_final_accuracy_bar(all_results, results_dir)

    results_with_acc = [r for r in all_results if "best_val_acc" in r]
    if results_with_acc:
        best_result = max(results_with_acc, key=lambda x: x["best_val_acc"])
        print("\n" + "=" * 80)
        print("BEST MODEL")
        print("=" * 80)
        print(f"Name: {best_result['name']}")
        print(f"Validation Accuracy: {best_result['best_val_acc']:.2f}%")
        print(f"Best Epoch: {best_result['best_epoch']}")
        print(f"Model Path: {best_result['path']}/best_model.pt")
        print("=" * 80)

    print(f"\nAll results saved to: {results_dir}")


if __name__ == "__main__":
    main()
