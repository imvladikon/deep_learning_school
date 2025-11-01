"""
Non-Contrastive Learning (NCL) Models

Implementations using:
- SimSiam (Simple Siamese with stop-gradient)
- Barlow Twins (Redundancy reduction)
- VICReg (Variance-Invariance-Covariance)

All adapt the multi-format (waveform + spectrogram) architecture.

References:
- "Exploring Simple Siamese Representation Learning" (Chen & He, CVPR 2021)
  arXiv:2011.10566, papers/2011.10566v1.pdf
- "Barlow Twins: Self-Supervised Learning via Redundancy Reduction" (Zbontar et al., ICML 2021)
  arXiv:2103.03230, papers/2103.03230v3.pdf
- "VICReg: Variance-Invariance-Covariance Regularization" (Bardes et al., ICLR 2022)
  arXiv:2105.04906, papers/2105.04906v3.pdf

See PAPER_REFERENCES.md for full citations and implementation details.
"""  # noqa: E501

from typing import Literal

import torch
import torch.nn as nn
from audiossl.modeling.encoders import create_resnet1d, create_resnet2d
from audiossl.modeling.projectors import MLPProjector, PredictorMLP, LinearClassifier


class SimSiamMultiFormat(nn.Module):
    """
    SimSiam adapted for multi-format audio learning

    Architecture:
    1. Dual encoders: 1D + 2D
    2. Shared projector: 512 -> 128
    3. Shared predictor: 128 -> 128 (with stop-gradient)

    Paper: https://arxiv.org/pdf/2011.10566

    Args:
        projector_hidden_dim: Hidden dim for projector
        projector_output_dim: Output embedding dim
        predictor_hidden_dim: Hidden dim for predictor
    """  # noqa: E501

    def __init__(
        self,
        projector_hidden_dim: int = 256,
        projector_output_dim: int = 128,
        predictor_hidden_dim: int = 512,
        input_dim: int = 512,
        img_channels: int = 1,
        norm_type: Literal["batch", "layer"] = "batch",

    ):
        super().__init__()

        self.encoder_1d = create_resnet1d()
        self.encoder_2d = create_resnet2d(img_channels=img_channels)

        self.projector = MLPProjector(
            input_dim=input_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_output_dim,
            norm_type=norm_type
        )

        self.predictor = PredictorMLP(
            input_dim=projector_output_dim,
            hidden_dim=predictor_hidden_dim,
            output_dim=projector_output_dim,
            norm_type=norm_type
        )

    def forward(self, waveform: torch.Tensor, spectrogram: torch.Tensor) -> tuple:
        """
        Args:
            waveform: [batch_size, time]
            spectrogram: [batch_size, freq, time]

        Returns:
            p1: [B, dim] - predictions from view 1 (waveform)
            p2: [B, dim] - predictions from view 2 (spectrogram)
            z1: [B, dim] - projections from view 1 (for stop-gradient)
            z2: [B, dim] - projections from view 2 (for stop-gradient)
            audio_feat: [B, 512] - waveform features (for linear eval)
            spec_feat: [B, 512] - spectrogram features (for linear eval)
        """  # noqa: E501
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(1)

        # [B, 512]
        audio_feat = self.encoder_1d(waveform).squeeze(-1)
        # [B, 512]
        spec_feat = self.encoder_2d(spectrogram).squeeze(-1).squeeze(-1)

        # [B, output_dim]
        z1 = self.projector(audio_feat)
        # [B, output_dim]
        z2 = self.projector(spec_feat)

        # [B, output_dim]
        p1 = self.predictor(z1)
        # [B, output_dim]
        p2 = self.predictor(z2)

        return p1, p2, z1, z2, audio_feat, spec_feat

    def get_features(
        self,
        waveform: torch.Tensor | None = None,
        spectrogram: torch.Tensor | None = None,
        view: Literal["1d", "2d", "both"] = "both",
    ) -> torch.Tensor:
        """Extract encoder features for linear evaluation"""
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
        else:
            return torch.cat([audio_feat, spec_feat], dim=1)


class BarlowTwinsMultiFormat(nn.Module):
    """
    Barlow Twins adapted for multi-format audio learning

    Paper: https://arxiv.org/pdf/2103.03230

    Args:
        projector_hidden_dim: Hidden dim for projector
        projector_output_dim: Output embedding dim (recommend 2048+ from paper)
    """  # noqa: E501

    def __init__(
        self,
        projector_hidden_dim: int = 256,
        projector_output_dim: int = 128,
        input_dim: int = 512,
        img_channels: int = 1,
    ):
        super().__init__()

        self.encoder_1d = create_resnet1d()
        self.encoder_2d = create_resnet2d(img_channels=img_channels)

        self.projector = MLPProjector(
            input_dim=input_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_output_dim,
        )

    def forward(self, waveform: torch.Tensor, spectrogram: torch.Tensor) -> tuple:
        """
        Forward pass for Barlow Twins

        Returns:
            z1: [B, dim] - embeddings from view 1
            z2: [B, dim] - embeddings from view 2
            audio_feat: [B, 512] - waveform features
            spec_feat: [B, 512] - spectrogram features
        """  # noqa: E501
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(1)

        audio_feat = self.encoder_1d(waveform).squeeze(-1)
        spec_feat = self.encoder_2d(spectrogram).squeeze(-1).squeeze(-1)

        z1 = self.projector(audio_feat)
        z2 = self.projector(spec_feat)

        return z1, z2, audio_feat, spec_feat

    def get_features(
        self,
        waveform: torch.Tensor | None = None,
        spectrogram: torch.Tensor | None = None,
        view: Literal["1d", "2d", "both"] = "both",
    ) -> torch.Tensor:
        """Extract encoder features for linear evaluation"""
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
        else:
            return torch.cat([audio_feat, spec_feat], dim=1)


class VICRegMultiFormat(nn.Module):
    """
    VICReg adapted for multi-format audio learning

    Paper: https://arxiv.org/pdf/2105.04906

    Args:
        projector_hidden_dim: Hidden dim for projector
        projector_output_dim: Output embedding dim (recommend 2048+ from paper)
    """  # noqa: E501

    def __init__(
        self,
        projector_hidden_dim: int = 256,
        projector_output_dim: int = 128,
        norm_type: Literal["batch", "layer"] = "batch",
    ):
        super().__init__()

        self.encoder_1d = create_resnet1d()
        self.encoder_2d = create_resnet2d(img_channels=1)

        self.projector = MLPProjector(
            input_dim=512,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_output_dim,
            norm_type=norm_type
        )

    def forward(self, waveform: torch.Tensor, spectrogram: torch.Tensor) -> tuple:
        """
        Returns:
            z1: [B, dim] - embeddings from view 1
            z2: [B, dim] - embeddings from view 2
            audio_feat: [B, 512] - waveform features
            spec_feat: [B, 512] - spectrogram features
        """  # noqa: E501
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(1)

        audio_feat = self.encoder_1d(waveform).squeeze(-1)
        spec_feat = self.encoder_2d(spectrogram).squeeze(-1).squeeze(-1)

        z1 = self.projector(audio_feat)
        z2 = self.projector(spec_feat)

        return z1, z2, audio_feat, spec_feat

    def get_features(self, waveform=None, spectrogram=None, view="both"):
        """Extract encoder features for linear evaluation"""
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
        else:
            return torch.cat([audio_feat, spec_feat], dim=1)


class NCLWithLinearHead(nn.Module):
    """
    Wrapper that adds linear classifier on top of NCL model
    Used for linear evaluation protocol
    """  # noqa: E501

    def __init__(
        self,
        ncl_model: nn.Module,
        num_classes: int = 10,
        freeze_encoder: bool = True,
        input_dim: int = 512,
    ):
        super().__init__()

        self.ncl_model = ncl_model
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for param in self.ncl_model.parameters():
                param.requires_grad = False

        self.classifier = LinearClassifier(input_dim=input_dim, num_classes=num_classes)

    def forward(
        self, waveform=None, spectrogram=None, view: Literal["1d", "2d"] = "1d"
    ):
        """
        Args:
            waveform: [batch_size, time]
            spectrogram: [batch_size, freq, time]
            view: "1d" or "2d"

        Returns:
            logits: [batch_size, num_classes]
        """  # noqa: E501
        with torch.set_grad_enabled(not self.freeze_encoder):
            features = self.ncl_model.get_features(
                waveform=waveform, spectrogram=spectrogram, view=view
            )

        logits = self.classifier(features)

        return logits
