"""
Multi-Format Contrastive Learning Model

Uses InfoNCE loss to learn representations from waveform + spectrogram views.
Inspired by multi-view contrastive learning approaches for audio.

References:
- "CLAR: Contrastive Learning of Auditory Representations" (arXiv:2103.06508)
  papers/2103.06508v3.pdf - Multi-format contrastive learning for audio
- "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"
  (Baevski et al., NeurIPS 2020, arXiv:2006.11477)
  papers/2010.09542v1.pdf - Contrastive learning from raw waveform
- Based on SSL_Seminar.ipynb implementation

Architecture inspired by SimCLR and wav2vec 2.0.
"""  # noqa: E501

from typing import Literal

import torch
import torch.nn as nn
from audiossl.modeling.encoders import create_resnet1d, create_resnet2d
from audiossl.modeling.projectors import MLPProjector, LinearClassifier


class MultiFormatContrastiveModel(nn.Module):
    """
    Multi-format contrastive learning model

    Architecture:
    1. Dual encoders: 1D (waveform) + 2D (spectrogram)
    2. Shared projection head: maps 512-dim features to 128-dim embeddings
    3. InfoNCE loss: contrastive loss between two views

    From seminar Net class

    Args:
        projector_hidden_dim: Hidden dimension for projector MLP
        projector_output_dim: Output embedding dimension
    """  # noqa: E501

    def __init__(
        self,
        projector_hidden_dim: int = 256,
        projector_output_dim: int = 128,
        input_dim: int = 512,
        norm_type: Literal["batch", "layer"] = "batch",
    ):
        super().__init__()
        # [B, 1, T] -> [B, 512, 1]
        self.encoder_1d = create_resnet1d()
        # [B, 1, F, T] -> [B, 512, 1, 1]
        self.encoder_2d = create_resnet2d(img_channels=1)

        # Shared projection head for contrastive learning
        # Projects encoder features to embedding space
        self.projector = MLPProjector(
            input_dim=input_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_output_dim,
            norm_type=norm_type
        )

    def forward(self, waveform, spectrogram):
        """
        Forward pass for contrastive learning

        Args:
            waveform: [batch_size, time] - raw waveform
            spectrogram: [batch_size, freq, time] - log mel spectrogram

        Returns:
            audio_emb: [batch_size, output_dim] - waveform embeddings (for contrastive loss)
            spec_emb: [batch_size, output_dim] - spectrogram embeddings (for contrastive loss)
            audio_feat: [batch_size, 512] - waveform features (for linear evaluation)
            spec_feat: [batch_size, 512] - spectrogram features (for linear evaluation)
        """  # noqa: E501

        if waveform.dim() == 2:
            # [B, 1, T]
            waveform = waveform.unsqueeze(1)

        if spectrogram.dim() == 3:
            # [B, 1, F, T]
            spectrogram = spectrogram.unsqueeze(1)

        # [B, 512, 1]
        audio_feat = self.encoder_1d(waveform)
        # [B, 512]
        audio_feat = audio_feat.squeeze(-1)

        # [B, 512, 1, 1]
        spec_feat = self.encoder_2d(spectrogram)
        # [B, 512]
        spec_feat = spec_feat.squeeze(-1).squeeze(-1)

        # Project to embedding space for contrastive loss
        # [B, output_dim]
        audio_emb = self.projector(audio_feat)
        # [B, output_dim]
        spec_emb = self.projector(spec_feat)
        return audio_emb, spec_emb, audio_feat, spec_feat

    def get_features(
        self,
        waveform: torch.Tensor | None = None,
        spectrogram: torch.Tensor | None = None,
        view: Literal["1d", "2d", "both"] = "both",
    ) -> torch.Tensor:
        """
        Extract encoder features (before projection)
        Used for linear evaluation

        Args:
            waveform: [batch_size, time]
            spectrogram: [batch_size, freq, time]
            view: "1d", "2d", or "both" (concatenated)

        Returns:
            features: [batch_size, feature_dim]
        """  # noqa: E501
        if view == "1d" or view == "both":
            if waveform is None:
                raise ValueError("waveform required for 1d view")
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(1)
            audio_feat = self.encoder_1d(waveform).squeeze(-1)

        if view == "2d" or view == "both":
            if spectrogram is None:
                raise ValueError("spectrogram required for 2d view")
            if spectrogram.dim() == 3:
                spectrogram = spectrogram.unsqueeze(1)
            spec_feat = self.encoder_2d(spectrogram).squeeze(-1).squeeze(-1)

        if view == "1d":
            return audio_feat
        elif view == "2d":
            return spec_feat
        else:  # both
            return torch.cat([audio_feat, spec_feat], dim=1)

    def get_embeddings(self, waveform=None, spectrogram=None, view="both"):
        """
        Extract projected embeddings (after projection head)
        Used for visualization and analysis

        Args:
            waveform: [batch_size, time]
            spectrogram: [batch_size, freq, time]
            view: "1d", "2d", or "both" (averaged)

        Returns:
            embeddings: [batch_size, output_dim]
        """  # noqa: E501
        if view == "1d" or view == "both":
            if waveform is None:
                raise ValueError("waveform required for 1d view")
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(1)
            audio_feat = self.encoder_1d(waveform).squeeze(-1)
            audio_emb = self.projector(audio_feat)

        if view == "2d" or view == "both":
            if spectrogram is None:
                raise ValueError("spectrogram required for 2d view")
            if spectrogram.dim() == 3:
                spectrogram = spectrogram.unsqueeze(1)
            spec_feat = self.encoder_2d(spectrogram).squeeze(-1).squeeze(-1)
            spec_emb = self.projector(spec_feat)

        if view == "1d":
            return audio_emb
        elif view == "2d":
            return spec_emb
        else:  # both - average embeddings
            return (audio_emb + spec_emb) / 2


class ContrastiveWithLinearHead(nn.Module):
    """
    Wrapper that adds linear classifier on top of contrastive model
    Used for linear evaluation protocol
    """  # noqa: E501

    def __init__(
        self,
        contrastive_model: nn.Module,
        num_classes: int = 10,
        freeze_encoder=True,
        input_dim: int = 512,
    ):
        super().__init__()

        self.contrastive_model = contrastive_model
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for param in self.contrastive_model.parameters():
                param.requires_grad = False

        self.classifier = LinearClassifier(input_dim=input_dim, num_classes=num_classes)

    def forward(
        self,
        waveform: torch.Tensor | None = None,
        spectrogram: torch.Tensor | None = None,
        view: Literal["1d", "2d"] = "1d",
    ) -> torch.Tensor:
        """
        Args:
            waveform: [batch_size, time]
            spectrogram: [batch_size, freq, time]
            view: "1d" or "2d" (which encoder to use)

        Returns:
            logits: [batch_size, num_classes]
        """  # noqa: E501
        with torch.set_grad_enabled(not self.freeze_encoder):
            features = self.contrastive_model.get_features(
                waveform=waveform, spectrogram=spectrogram, view=view
            )

        logits = self.classifier(features)
        return logits
