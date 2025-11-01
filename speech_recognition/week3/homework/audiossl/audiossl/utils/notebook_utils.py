import random
from typing import List, Literal, Callable

import matplotlib.pyplot as plt
import torch
from IPython.display import Audio, display, clear_output


class NotebookVisualizer:

    def __init__(self, figsize=(15, 5)):
        self.figsize = figsize
        self.train_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []

    def update(
            self,
            *,
            train_loss: float,
            val_loss: float,
            train_acc: float | None = None,
            val_acc: float | None = None,
            epoch: int | None = None,
    ):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        show_accuracy = train_acc is not None and val_acc is not None

        if train_acc is not None:
            self.train_accuracies.append(train_acc)

        if val_acc is not None:
            self.val_accuracies.append(val_acc)

        clear_output(wait=True)

        self.plot(show_accuracy=show_accuracy)

        if epoch is not None:
            print(f"\nEpoch {epoch}:")

        if show_accuracy:
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        else:
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")

    def plot(self, show_accuracy: bool = True):
        if not self.train_losses:
            return

        epochs = range(1, len(self.train_losses) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        ax1.plot(epochs, self.train_losses, "b-", label="Training Loss", linewidth=2)
        ax1.plot(epochs, self.val_losses, "r-", label="Validation Loss", linewidth=2)
        ax1.set_title("Training and Validation Loss", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if show_accuracy:
            ax2.plot(
                epochs, self.train_accuracies, "b-", label="Training Accuracy", linewidth=2
            )
            ax2.plot(
                epochs, self.val_accuracies, "r-", label="Validation Accuracy", linewidth=2
            )
            ax2.set_title(
                "Training and Validation Accuracy", fontsize=12, fontweight="bold"
            )
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("Accuracy (%)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_history(self):
        return {
            "train_loss": self.train_losses,
            "train_acc": self.train_accuracies,
            "val_loss": self.val_losses,
            "val_acc": self.val_accuracies,
        }


def is_notebook() -> bool:
    try:
        from IPython import get_ipython

        if get_ipython() is None:
            return False
        if "IPKernelApp" not in get_ipython().config:
            return False
        return True
    except (ImportError, AttributeError):
        return False


def listen_and_predict_ssl(
        model_with_head,
        dataset,
        spec_transform=None,
        view="1d",
        num_samples=5,
        sample_rate=16000,
        device="cuda",
):
    """
    Prediction function for SSL models with linear heads.
    Use listen_and_predict_head() instead - this is kept for backwards compatibility.
    """
    model_with_head.eval()

    for i in range(num_samples):
        sample_idx = random.randint(0, len(dataset) - 1)
        waveform, label = dataset[sample_idx]

        waveform_batch = waveform.unsqueeze(0).to(device)

        with torch.no_grad():
            if view == "1d":
                logits = model_with_head(waveform=waveform_batch, view=view)
            elif view == "2d":
                # spec_transform already returns [1, n_mels, time], don't add extra batch dim
                spec = spec_transform(waveform.unsqueeze(0).cpu())
                spec_batch = spec.to(device)  # Already has batch dimension from spec_transform
                logits = model_with_head(spectrogram=spec_batch, view=view)

            predicted = logits.argmax(dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item() * 100

        print("\n" + "=" * 50)
        print(f"Sample {i + 1}/{num_samples} (index: {sample_idx})")
        print(f"True Label:    {label}")
        print(f"Predicted:     {predicted}")
        print(f"Confidence:    {confidence:.1f}%")
        result_icon = "✅ CORRECT" if predicted == label else "❌ WRONG"
        print(f"Result:        {result_icon}")
        print("=" * 50)

        display(Audio(waveform.cpu().numpy(), rate=sample_rate))


def listen_and_predict_head(
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        spec_transform: Callable | None = None,
        view: Literal["1d", "2d"]="1d",
        num_samples: int=5,
        sample_rate: int=16000,
        device:str="cuda",
        model_type: Literal["linear_head", "transformer", "wav2vec2"]="linear_head",
) -> None:
    """
    - Task 2: Linear heads (ContrastiveWithLinearHead, NCLWithLinearHead)
    - Task 3.1: Transformer (AudioSpectrogramTransformer)
    - Task 3.2: Wav2Vec2 (Wav2Vec2ForSequenceClassification)
    """
    model.eval()

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for i, idx in enumerate(indices):
        waveform, label = dataset[idx]

        # waveform может быть [channels, time] или [time]
        # Нужно привести к [time] для Wav2Vec2
        if waveform.dim() > 1:
            waveform = waveform.squeeze()  # [channels, time] -> [time]

        print(f"\n{'=' * 60}")
        print(f"Sample {i + 1}/{num_samples} (index={idx})")
        print(f"True label: {label}")

        # Play audio
        display(Audio(waveform.cpu().numpy(), rate=sample_rate))

        with torch.no_grad():
            if model_type == "linear_head":
                if view == "1d":
                    waveform_batch = waveform.unsqueeze(0).to(device)
                    logits = model(waveform=waveform_batch, view=view)
                elif view == "2d":
                    spec = spec_transform(waveform.unsqueeze(0).cpu())
                    spec_batch = spec.to(device)
                    logits = model(spectrogram=spec_batch, view=view)

            elif model_type == "transformer":
                spec = spec_transform(waveform.unsqueeze(0).cpu()).squeeze(0)
                spec_batch = spec.unsqueeze(0).to(device)
                logits = model(spec_batch)

            elif model_type == "wav2vec2":
                # Wav2Vec2 ожидает [batch, sequence_length]
                # waveform уже [time] после squeeze выше
                waveform_batch = waveform.unsqueeze(0).to(device)  # [time] -> [1, time]

                # DEBUG: Проверка формы
                print(f"DEBUG: waveform_batch.shape = {waveform_batch.shape}")

                outputs = model(waveform_batch)
                logits = outputs.logits
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        confidence = probs[0, pred].item()

        is_correct = pred == label
        status = "✅ CORRECT" if is_correct else "❌ WRONG"

        print(f"Predicted: {pred} (confidence: {confidence * 100:.1f}%) {status}")

    print(f"\n{'=' * 60}\n")


def listen_and_predict(
        model,
        dataset,
        spectrogram_transform=None,
        num_samples=5,
        sample_rate=16_000,
        device="cuda",
):
    """
    Original function for supervised models (Task 1)
    """
    model.eval()

    for i in range(num_samples):
        sample_idx = random.randint(0, len(dataset) - 1)
        waveform, label = dataset[sample_idx]

        waveform_batch = waveform.unsqueeze(0).to(device)

        with torch.no_grad():
            if model.encoder_type == "1d":
                logits = model(waveform=waveform_batch)
            elif model.encoder_type == "2d":
                spec = spectrogram_transform(waveform.unsqueeze(0).cpu())
                spec_batch = spec.unsqueeze(0).to(device)
                logits = model(spectrogram=spec_batch)
            else:
                spec = spectrogram_transform(waveform.unsqueeze(0).cpu())
                spec_batch = spec.unsqueeze(0).to(device)
                logits = model(waveform=waveform_batch, spectrogram=spec_batch)

            predicted = logits.argmax(dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item() * 100

        print("\n" + "=" * 50)
        print(f"Sample {i + 1}/{num_samples} (index: {sample_idx})")
        print(f"True Label:    {label}")
        print(f"Predicted:     {predicted}")
        print(f"Confidence:    {confidence:.1f}%")
        result_icon = "✅ CORRECT" if predicted == label else "❌ WRONG"
        print(f"Result:        {result_icon}")
        print("=" * 50)

        display(Audio(waveform.cpu().numpy(), rate=sample_rate))


