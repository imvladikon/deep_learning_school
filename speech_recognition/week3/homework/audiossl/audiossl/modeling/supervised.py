from typing import Literal

import torch
import torch.nn as nn
from audiossl.modeling.encoders import create_resnet1d, create_resnet2d


class SupervisedAudioClassifier(nn.Module):
    """
    Supervised Learning Model
    Simple baseline using encoder + classifier trained end-to-end

    Supervised classifier for audio digit classification

    Uses either 1D (waveform) or 2D (spectrogram) encoder
    with a classification head on top

    Args:
        encoder_type: "1d", "2d", or "both"
        num_classes: Number of output classes (default: 10 for digits 0-9)
        hidden_dim: Hidden dimension for classification head
    """  # noqa: E501

    def __init__(
        self,
        encoder_type: Literal["1d", "2d", "both"] = "1d",
        num_classes: int = 10,
        hidden_dim: int = 256,
        norm_type: Literal["batch", "layer"] = "batch",
    ):
        super().__init__()

        self.encoder_type = encoder_type

        if encoder_type == "1d":
            self.encoder_1d = create_resnet1d()
            encoder_output_dim = 512
        elif encoder_type == "2d":
            self.encoder_2d = create_resnet2d(img_channels=1)
            encoder_output_dim = 512
        elif encoder_type == "both":
            self.encoder_1d = create_resnet1d()
            self.encoder_2d = create_resnet2d(img_channels=1)
            encoder_output_dim = 1024  # Concatenated
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.classifier = nn.Sequential(
            nn.Linear(encoder_output_dim, hidden_dim),
            self._make_norm_layer(norm_type, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def _make_norm_layer(self, norm_type: str, dim: int) -> nn.Module:
        if norm_type == "batch":
            return nn.BatchNorm1d(dim)
        elif norm_type == "layer":
            return nn.LayerNorm(dim)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

    def forward(
        self,
        waveform: torch.Tensor | None = None,
        spectrogram: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            waveform: [batch_size, time] - raw waveform (for 1d or both)
            spectrogram: [batch_size, freq, time] - log mel spectrogram (for 2d or both)

        Returns:
            logits: [batch_size, num_classes] - class logits
        """  # noqa: E501
        if self.encoder_type == "1d":
            if waveform is None:
                raise ValueError("waveform required for 1d encoder")

            # add channel dimension: [B, T] -> [B, 1, T]
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(1)

            # [B, 1, T] -> [B, 512, 1]
            features = self.encoder_1d(waveform)
            features = features.squeeze(-1)  # [B, 512]

        elif self.encoder_type == "2d":
            if spectrogram is None:
                raise ValueError("spectrogram required for 2d encoder")

            # Add channel dimension: [B, F, T] -> [B, 1, F, T]
            if spectrogram.dim() == 3:
                spectrogram = spectrogram.unsqueeze(1)

            # Encode: [B, 1, F, T] -> [B, 512, 1, 1]
            features = self.encoder_2d(spectrogram)
            features = features.squeeze(-1).squeeze(-1)  # [B, 512]

        else:  # both
            if waveform is None or spectrogram is None:
                raise ValueError(
                    "Both waveform and spectrogram required for 'both' encoder"
                )

            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(1)

            # [B, 512]
            features_1d = self.encoder_1d(waveform).squeeze(-1)

            if spectrogram.dim() == 3:
                spectrogram = spectrogram.unsqueeze(1)

            # [B, 512]
            features_2d = self.encoder_2d(spectrogram).squeeze(-1).squeeze(-1)

            # # [B, 1024]
            features = torch.cat([features_1d, features_2d], dim=1)

        logits = self.classifier(features)
        return logits

    def get_features(
        self, waveform: torch.Tensor, spectrogram: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract features before classification head
        Useful for evaluation and visualization

        Returns:
            features: [batch_size, encoder_output_dim]
        """  # noqa: E501
        if self.encoder_type == "1d":
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(1)
            features = self.encoder_1d(waveform).squeeze(-1)

        elif self.encoder_type == "2d":
            if spectrogram.dim() == 3:
                spectrogram = spectrogram.unsqueeze(1)
            features = self.encoder_2d(spectrogram).squeeze(-1).squeeze(-1)

        else:  # both
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(1)
            features_1d = self.encoder_1d(waveform).squeeze(-1)

            if spectrogram.dim() == 3:
                spectrogram = spectrogram.unsqueeze(1)
            features_2d = self.encoder_2d(spectrogram).squeeze(-1).squeeze(-1)

            features = torch.cat([features_1d, features_2d], dim=1)

        return features
