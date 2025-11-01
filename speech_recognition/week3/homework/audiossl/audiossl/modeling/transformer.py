import torch
import torch.nn as nn


class AudioSpectrogramTransformer(nn.Module):
    """
    Transformer-based classifier that operates on log-mel spectrogram patches.
    Inspired by Gong et al., "AST: Audio Spectrogram Transformer"

    Transformer classifier for speech commands using log-mel spectrograms.
    """  # noqa: E501

    def __init__(
        self,
        n_mels: int = 64,
        d_model: int = 192,
        n_heads: int = 3,
        num_layers: int = 4,
        dim_feedforward: int = 384,
        dropout: float = 0.1,
        max_seq_len: int = 400,
        classifier_hidden_dim: int = 256,
        num_classes: int = 10,
    ) -> None:
        """
        :param n_mels: Number of mel frequency bins in the input spectrogram.
        :param d_model: Embedding dimension for the transformer.
        :param n_heads: Number of self-attention heads.
        :param num_layers: Number of transformer encoder layers.
        :param dim_feedforward: Hidden dimension inside the transformer feed-forward blocks.
        :param dropout: Dropout rate applied across the model.
        :param max_seq_len: Maximum number of spectrogram frames expected (controls positional embeddings).
        :param classifier_hidden_dim: Hidden dimension of the MLP classification head.
        :param num_classes: Number of target classes (default 10 for AudioMNIST digits).
        """  # noqa: E501
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_mels = n_mels
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Front-end projection from mel bins into model dimension
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Learnable class token and positional encoding (cls + frames)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, max_seq_len + 1, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize learnable tokens and embeddings."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

    def forward(
        self, spectrograms: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            spectrograms: Tensor of shape [B, n_mels, T] (padded).
            lengths: Optional tensor with original frame lengths per sample (without padding).

        Returns:
            logits: Tensor of shape [B, num_classes].
        """ # noqa: E501
        if spectrograms.dim() != 3:
            raise ValueError("Expected spectrograms with shape [B, n_mels, T]")

        batch_size, _, seq_len = spectrograms.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}. "
                "Increase max_seq_len when instantiating the model."
            )

        # Project mel bins into model dimension
        x = self.input_proj(spectrograms)  # [B, d_model, T]
        x = x.transpose(1, 2)  # [B, T, d_model]

        # Prepend class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, T + 1, d_model]

        # Add positional embeddings
        x = x + self.positional_embedding[:, : seq_len + 1, :]

        # Prepare key padding mask (True for padded positions)
        key_padding_mask = None
        if lengths is not None:
            if lengths.dim() != 1 or lengths.shape[0] != batch_size:
                raise ValueError("lengths must be a 1D tensor with batch_size elements")

            device = spectrograms.device
            frame_positions = torch.arange(seq_len, device=device).expand(
                batch_size, seq_len
            )
            padding_positions = frame_positions >= lengths.unsqueeze(1)

            key_padding_mask = torch.zeros(
                batch_size, seq_len + 1, dtype=torch.bool, device=device
            )
            key_padding_mask[:, 1:] = padding_positions

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        cls_representation = x[:, 0]  # [B, d_model]
        logits = self.classifier(cls_representation)

        return logits
