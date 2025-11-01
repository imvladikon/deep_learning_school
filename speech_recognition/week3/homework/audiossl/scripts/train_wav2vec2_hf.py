"""Fine-tuning HuggingFace wav2vec2-base on AudioMNIST"""

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from audiossl.data import create_audiomnist_splits


def create_hf_dataset_from_audiomnist(train_dataset, val_dataset, sample_rate=16000):
    def dataset_to_dict(dataset):
        """Convert PyTorch dataset to dictionary"""
        audio_dicts = []
        label_list = []

        for i in range(len(dataset)):
            waveform, label = dataset[i]

            audio_array = waveform.numpy()

            audio_dict = {"array": audio_array, "sampling_rate": sample_rate}

            audio_dicts.append(audio_dict)
            label_list.append(int(label))

        return {"audio": audio_dicts, "label": label_list}

    train_dict = dataset_to_dict(train_dataset)
    val_dict = dataset_to_dict(val_dataset)

    train_hf = Dataset.from_dict(train_dict)
    val_hf = Dataset.from_dict(val_dict)

    dataset_dict = DatasetDict({"train": train_hf, "validation": val_hf})

    print(
        f"Created HF dataset: {len(train_hf)} train, {len(val_hf)} validation samples"
    )

    return dataset_dict


def preprocess_function(examples, feature_extractor, max_length_seconds=1.0):
    """
    Preprocess audio samples for wav2vec2

    Args:
        examples: Batch of examples from dataset
        feature_extractor: Wav2Vec2FeatureExtractor
        max_length_seconds: Maximum audio length in seconds
    """
    audio_arrays = []
    for audio_dict in examples["audio"]:
        arr = audio_dict["array"]
        if isinstance(arr, list):
            arr = np.array(arr, dtype=np.float32)
        audio_arrays.append(arr)

    sampling_rate = feature_extractor.sampling_rate
    max_length = int(sampling_rate * max_length_seconds)

    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=sampling_rate,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
    )
    inputs["labels"] = examples["label"]
    return inputs


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    references = eval_pred.label_ids

    accuracy = accuracy_score(references, predictions)

    return {"accuracy": accuracy}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Wav2Vec2 on AudioMNIST")
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/wav2vec2-base",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training and evaluation",
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--max-length-seconds",
        type=float,
        default=1.0,
        help="Maximum audio length in seconds",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Audio sample rate"
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=3, help="Early stopping patience"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="AudioMNIST/data",
        help="Root directory for AudioMNIST data",
    )
    parser.add_argument(
        "--num-test-speakers", type=int, default=12, help="Number of test speakers"
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="experiments/hf_wav2vec2",
        help="Experiment directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--num-proc",
        type=int,
        default=4,
        help="Number of processes for dataset preprocessing",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=4,
        help="Number of workers for dataloader",
    )
    parser.add_argument(
        "--freeze-feature-extractor",
        action="store_true",
        help="Freeze feature extractor during training",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_wandb:
        wandb.init(
            project="audiomnist-ssl",
            name="wav2vec2-base-audiomnist",
            config={
                "model": args.model_name,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "num_epochs": args.epochs,
                "max_length_seconds": args.max_length_seconds,
                "early_stopping_patience": args.early_stopping_patience,
                "seed": args.seed,
            },
            tags=["wav2vec2", "hf", "bonus", "pretrained"],
        )

    print("\nLoading AudioMNIST dataset...")
    train_dataset, val_dataset, _ = create_audiomnist_splits(
        root=args.data_root, num_test_speakers=args.num_test_speakers
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    dataset_dict = create_hf_dataset_from_audiomnist(
        train_dataset, val_dataset, sample_rate=args.sample_rate
    )
    num_labels = 10

    print(f"\nLoading pretrained model: {args.model_name}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name)

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels, ignore_mismatched_sizes=True
    )

    # static analyzer complains about freeze_feature_encoder
    # so, just hasattr check here
    if args.freeze_feature_extractor and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )

    print("\nPreprocessing datasets...")

    def preprocess(examples):
        return preprocess_function(examples, feature_extractor, args.max_length_seconds)

    dataset_dict = dataset_dict.map(
        preprocess,
        remove_columns=["audio"],
        batched=True,
        batch_size=100,
        num_proc=args.num_proc,
    )

    training_args = TrainingArguments(
        output_dir=str(exp_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="wandb" if not args.no_wandb else "none",
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        ],
    )

    print("\n" + "=" * 80)
    print("STARTING FINE-TUNING")
    print("=" * 80)

    train_result = trainer.train()

    trainer.save_model(exp_dir / "final_model")

    print("\n" + "=" * 80)
    print("EVALUATING ON VALIDATION SET")
    print("=" * 80)

    eval_result = trainer.evaluate()

    print("\n" + "=" * 80)
    print("FINE-TUNING COMPLETE!")
    print(f"Best validation accuracy: {eval_result['eval_accuracy'] * 100:.2f}%")
    print(f"Results saved to: {exp_dir}")
    print("=" * 80)

    if not args.no_wandb:
        wandb.run.summary["best_val_accuracy"] = eval_result["eval_accuracy"]
        wandb.run.summary["train_runtime"] = train_result.metrics["train_runtime"]
        wandb.run.summary["train_samples_per_second"] = train_result.metrics[
            "train_samples_per_second"
        ]

        wandb.finish()

    return eval_result


if __name__ == "__main__":
    main()
