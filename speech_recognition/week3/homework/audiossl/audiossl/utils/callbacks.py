from typing import Any, Dict, List, Optional
import wandb
from audiossl.utils.notebook_utils import NotebookVisualizer, is_notebook


class TrainingCallback:

    def __init__(self, key: str = ""):
        self.key = key

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of training."""
        ...

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of training."""
        ...

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of each epoch."""
        ...

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            logs: Dictionary containing metrics (train_loss, train_acc, val_loss, val_acc, etc.)
        """
        ...

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each training batch."""
        ...


class WandbCallback(TrainingCallback):
    def __init__(self, log_audio_every_n_epochs: int = 10):
        super().__init__(key="wandb")
        self.log_audio_every_n_epochs = log_audio_every_n_epochs

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        if wandb.run is None and logs:
            config = logs.get("config", {})
            wandb.init(
                project=config.get("project", "audiomnist-ssl"),
                name=config.get("run_name", None),
                config=config,
                tags=config.get("tags", []),
            )

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if not logs or wandb.run is None:
            return

        log_dict = {
            "epoch": epoch,
            "train/loss": logs.get("train_loss"),
            "train/accuracy": logs.get("train_acc"),
            "val/loss": logs.get("val_loss"),
            "val/accuracy": logs.get("val_acc"),
        }

        if "learning_rate" in logs:
            log_dict["learning_rate"] = logs["learning_rate"]

        wandb.log(log_dict)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        if wandb.run is None or not logs:
            return

        if "best_val_acc" in logs:
            wandb.run.summary["best_val_accuracy"] = logs["best_val_acc"]
        if "best_epoch" in logs:
            wandb.run.summary["best_epoch"] = logs["best_epoch"]

        if "training_curves_path" in logs:
            wandb.log(
                {"training_curves": wandb.Image(str(logs["training_curves_path"]))}
            )


class NotebookCallback(TrainingCallback):
    def __init__(self, figsize=(15, 5)):
        super().__init__(key="notebook")
        self.visualizer = NotebookVisualizer(figsize=figsize)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None):
        if not logs:
            return

        self.visualizer.update(
            train_loss=logs.get("train_loss", 0),
            train_acc=logs.get("train_acc"),
            val_loss=logs.get("val_loss", 0),
            val_acc=logs.get("val_acc"),
            epoch=epoch,
        )

    def get_history(self):
        return self.visualizer.get_history()


class CallbackList:
    def __init__(self, callbacks: Optional[List[TrainingCallback]] = None):
        self.callbacks = callbacks or []

    def append(self, callback: TrainingCallback):
        self.callbacks.append(callback)

    def __getitem__(self, name: str) -> TrainingCallback | None:
        for callback in self.callbacks:
            if callback.__class__.__name__ == name or callback.key == name:
                return callback
        return None

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)


def create_default_callbacks(
        use_wandb: bool = True,
        use_notebook: bool | None = None,
        figsize=(15, 5),
) -> CallbackList:
    callbacks = []

    if use_wandb:
        callbacks.append(WandbCallback())

    if use_notebook is None:
        use_notebook = is_notebook()

    if use_notebook:
        callbacks.append(NotebookCallback(figsize=figsize))

    return CallbackList(callbacks)
