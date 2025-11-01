from typing import Literal

import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """
    Projection heads for self-supervised learning

    MLP projection head used in contrastive learning
    Projects encoder features to embedding space for contrastive loss

    From seminar: projects 512-dim encoder output to 128-dim embedding
    """  # noqa: E501

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 128,
        norm_type: Literal["batch", "layer"] = "batch",
    ):
        super(MLPProjector, self).__init__()

        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self._make_norm_layer(norm_type, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            self._make_norm_layer(norm_type, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _make_norm_layer(self, norm_type: str, dim: int) -> nn.Module:
        if norm_type == "batch":
            return nn.BatchNorm1d(dim)
        elif norm_type == "layer":
            return nn.LayerNorm(dim)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim] - encoder features
        Returns:
            x: [batch_size, output_dim] - projected embeddings
        """  # noqa: E501
        return self.projector(x)


class PredictorMLP(nn.Module):
    """
    Predictor MLP used in SimSiam and other non-contrastive methods
    Asymmetric architecture with stop-gradient on one branch
    """  # noqa: E501

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 512,
        output_dim: int = 128,
        norm_type: Literal["batch", "layer"] = "batch",
    ):
        super(PredictorMLP, self).__init__()

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self._make_norm_layer(norm_type, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _make_norm_layer(self, norm_type: str, dim: int) -> nn.Module:
        if norm_type == "batch":
            return nn.BatchNorm1d(dim)
        elif norm_type == "layer":
            return nn.LayerNorm(dim)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim] - projected embeddings
        Returns:
            x: [batch_size, output_dim] - predicted embeddings
        """  # noqa: E501
        return self.predictor(x)


class LinearClassifier(nn.Module):
    """
    Simple linear classifier (single layer)

    This is the standard approach for linear evaluation protocol in SSL.
    Fast and interpretable, commonly used for probing pretrained representations.
    """  # noqa: E501

    def __init__(self, input_dim: int = 512, num_classes: int = 10):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MLPClassifier(nn.Module):
    """
    MLP-based classifier with configurable depth and architecture.

    Based on research findings:
    - SimCLR: Uses 2-layer MLP (Linear → ReLU → Linear) for projection
    - BEATs: Uses Dropout → Linear for classification head
    - Common practice: Hidden layer helps learn non-linear decision boundaries

    This classifier provides better expressiveness than linear probe while
    remaining computationally efficient.

    Architecture options:
    - 'simple': Dropout → Linear (like BEATs)
    - 'mlp1': Linear → ReLU → Dropout → Linear (SimCLR-style, 1 hidden layer)
    - 'mlp2': Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear (2 hidden layers)
    - 'norm': Adds LayerNorm before each activation (more stable for audio)

    Args:
        input_dim: Input feature dimension
        num_classes: Number of target classes
        hidden_dim: Hidden layer dimension(s)
        dropout: Dropout probability (default: 0.1, following BEATs)
        arch: Architecture type
        norm_type: Normalization type ('none', 'layer', 'batch')
    """  # noqa: E501

    def __init__(
        self,
        input_dim: int = 512,
        num_classes: int = 10,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        arch: Literal["simple", "mlp1", "mlp2", "norm"] = "mlp1",
        norm_type: Literal["none", "layer", "batch"] = "none",
    ):
        super(MLPClassifier, self).__init__()

        self.arch = arch
        self.norm_type = norm_type

        if arch == "simple":
            # BEATs-style: just Dropout → Linear
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, num_classes),
            )

        elif arch == "mlp1":
            # SimCLR-style: 1 hidden layer with ReLU
            layers = [
                nn.Linear(input_dim, hidden_dim),
                self._make_norm_layer(norm_type, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            ]
            # Remove None layers (from 'none' norm type)
            self.classifier = nn.Sequential(*[l for l in layers if l is not None])

        elif arch == "mlp2":
            # 2 hidden layers for more expressiveness
            layers = [
                nn.Linear(input_dim, hidden_dim),
                self._make_norm_layer(norm_type, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                self._make_norm_layer(norm_type, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            ]
            self.classifier = nn.Sequential(*[l for l in layers if l is not None])

        elif arch == "norm":
            # With LayerNorm before activations (better for audio per research)
            layers = [
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            ]
            self.classifier = nn.Sequential(*layers)

        else:
            raise ValueError(f"Unknown architecture: {arch}")

    def _make_norm_layer(self, norm_type: str, dim: int) -> nn.Module | None:
        """Create normalization layer or None"""
        if norm_type == "batch":
            return nn.BatchNorm1d(dim)
        elif norm_type == "layer":
            return nn.LayerNorm(dim)
        elif norm_type == "none":
            return None
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim] - encoder features
        Returns:
            logits: [batch_size, num_classes] - class logits
        """
        return self.classifier(x)
