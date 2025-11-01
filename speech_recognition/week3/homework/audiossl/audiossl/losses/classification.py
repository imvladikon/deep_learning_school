import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Args:
        smoothing: Smoothing factor in [0, 1). When 0, recovers standard CE.
    """  # noqa: E501

    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in the range [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("logits must have shape [batch_size, num_classes]")

        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError("targets must have shape [batch_size]")

        num_classes = logits.shape[1]
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth_value = self.smoothing / (num_classes - 1)
            target_dist = torch.full_like(log_probs, smooth_value)
            target_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = (-target_dist * log_probs).sum(dim=1).mean()
        return loss
