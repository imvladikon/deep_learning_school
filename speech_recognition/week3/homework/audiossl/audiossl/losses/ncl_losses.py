"""
Non-Contrastive Learning (NCL) Losses

Implementations of:
1. SimSiam - Simple Siamese network with stop-gradient
2. Barlow Twins - Redundancy reduction via cross-correlation matrix
3. VICReg - Variance-Invariance-Covariance Regularization

All these methods avoid the need for negative samples while preventing collapse.

References:
- "Exploring Simple Siamese Representation Learning" (Chen & He, CVPR 2021)
  arXiv:2011.10566, papers/2011.10566v1.pdf
- "Barlow Twins: Self-Supervised Learning via Redundancy Reduction" (Zbontar et al., ICML 2021)
  arXiv:2103.03230, papers/2103.03230v3.pdf
- "VICReg: Variance-Invariance-Covariance Regularization" (Bardes et al., ICLR 2022)
  arXiv:2105.04906, papers/2105.04906v3.pdf
- "Bootstrap Your Own Latent" (BYOL, Grill et al., NeurIPS 2020)
  arXiv:2006.07733, papers/2006.07733v3.pdf

See PAPER_REFERENCES.md for full citations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimSiamLoss(nn.Module):
    """
    SimSiam loss: Simple Siamese network with stop-gradient

    Paper: "Exploring Simple Siamese Representation Learning"
           https://arxiv.org/pdf/2011.10566

    Key idea: Maximize similarity between predictions and stop-gradient targets

    Loss = -0.5 * (D(p1, sg(z2)) + D(p2, sg(z1)))

    where:
    - p = predictor(z)
    - z = projector(encoder(x))
    - sg = stop_gradient
    - D = negative cosine similarity

    No hyperparameters needed!
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def cosine_similarity(p, z):
        """
        Negative cosine similarity (to minimize)

        Args:
            p: [batch_size, dim] - predictions
            z: [batch_size, dim] - targets (will be stop-gradiented by caller)

        Returns:
            similarity: [batch_size] - negative cosine similarities
        """
        # Normalize to unit vectors
        p = F.normalize(p, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)

        # Negative cosine similarity: -(p · z)
        return -(p * z).sum(dim=1).mean()

    def forward(self, p1, p2, z1, z2):
        """
        Compute SimSiam loss

        Args:
            p1: [batch_size, dim] - predictions from view 1
            p2: [batch_size, dim] - predictions from view 2
            z1: [batch_size, dim] - projections from view 1 (will be stop-gradiented)
            z2: [batch_size, dim] - projections from view 2 (will be stop-gradiented)

        Returns:
            loss: scalar loss value

        Note: Caller must apply stop_gradient to z1 and z2!
        """
        # Symmetrized loss:
        # D(p1, sg(z2)) + D(p2, sg(z1))
        loss1 = self.cosine_similarity(p1, z2.detach())  # stop gradient on z2
        loss2 = self.cosine_similarity(p2, z1.detach())  # stop gradient on z1

        # Average both directions
        loss = 0.5 * (loss1 + loss2)

        return loss


class BarlowTwinsLoss(nn.Module):
    """
    Barlow Twins loss: Redundancy reduction via cross-correlation matrix

    Paper: "Barlow Twins: Self-Supervised Learning via Redundancy Reduction"
           https://arxiv.org/pdf/2103.03230

    Key idea: Make cross-correlation matrix between embeddings close to identity

    Loss = Σ_i (1 - C_ii)^2 + λ Σ_i Σ_(j≠i) C_ij^2

    where C is the cross-correlation matrix between normalized embeddings

    Args:
        lambda_param: Weight for off-diagonal terms (default: 0.005 from paper)
        embedding_dim: Dimension of embeddings (for normalization)
    """

    def __init__(self, lambda_param=0.005, embedding_dim=128):
        super().__init__()
        self.lambda_param = lambda_param
        self.embedding_dim = embedding_dim

    def forward(self, z1, z2):
        """
        Compute Barlow Twins loss

        Args:
            z1: [batch_size, dim] - embeddings from view 1
            z2: [batch_size, dim] - embeddings from view 2

        Returns:
            loss: scalar loss value
        """
        batch_size = z1.shape[0]

        # Normalize embeddings along batch dimension (zero mean, unit std)
        z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-8)
        z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-8)

        # Compute cross-correlation matrix: C = (1/N) * Z1^T @ Z2
        # Shape: [dim, dim]
        c = (z1_norm.T @ z2_norm) / batch_size

        # Invariance loss: penalize deviation from identity on diagonal
        # Σ_i (1 - C_ii)^2
        on_diagonal = torch.diagonal(c).add_(-1).pow_(2).sum()

        # Redundancy reduction loss: penalize non-zero off-diagonal terms
        # Σ_i Σ_(j≠i) C_ij^2
        off_diagonal = c.pow_(2).sum() - torch.diagonal(c).pow_(2).sum()
        off_diagonal = self.lambda_param * off_diagonal

        # Total loss
        loss = on_diagonal + off_diagonal

        return loss


class VICRegLoss(nn.Module):
    """
    VICReg loss: Variance-Invariance-Covariance Regularization

    Paper: "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning"
           https://arxiv.org/pdf/2105.04906

    Key idea: Explicitly enforce three properties:
    1. Invariance: embeddings from same sample should be similar
    2. Variance: embeddings should have high variance (avoid collapse)
    3. Covariance: different dimensions should be decorrelated

    Loss = λ * inv_loss + μ * var_loss + ν * cov_loss

    Args:
        lambda_param: Weight for invariance term (default: 25.0 from paper)
        mu_param: Weight for variance term (default: 25.0 from paper)
        nu_param: Weight for covariance term (default: 1.0 from paper)
        gamma: Variance target (default: 1.0)
    """

    def __init__(self, lambda_param=25.0, mu_param=25.0, nu_param=1.0, gamma=1.0):
        super().__init__()
        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param
        self.gamma = gamma

    def invariance_loss(self, z1, z2):
        """
        Invariance loss: MSE between embeddings from two views

        Args:
            z1, z2: [batch_size, dim]

        Returns:
            loss: scalar
        """
        return F.mse_loss(z1, z2)

    def variance_loss(self, z):
        """
        Variance loss: hinge loss on standard deviation of each dimension
        Encourages variance to be at least gamma

        Args:
            z: [batch_size, dim]

        Returns:
            loss: scalar
        """
        # Compute std along batch dimension: [dim]
        std = torch.sqrt(z.var(dim=0) + 1e-8)

        # Hinge loss: max(0, gamma - std)
        loss = F.relu(self.gamma - std).mean()

        return loss

    def covariance_loss(self, z):
        """
        Covariance loss: penalize off-diagonal terms of covariance matrix
        Encourages different dimensions to be decorrelated

        Args:
            z: [batch_size, dim]

        Returns:
            loss: scalar
        """
        batch_size, dim = z.shape

        # Zero-center embeddings
        z = z - z.mean(dim=0)

        # Compute covariance matrix: C = (1/(N-1)) * Z^T @ Z
        cov = (z.T @ z) / (batch_size - 1)

        # Penalize off-diagonal terms
        # Sum of squared off-diagonal elements
        off_diagonal_mask = ~torch.eye(dim, dtype=torch.bool, device=z.device)
        loss = cov[off_diagonal_mask].pow_(2).sum() / dim

        return loss

    def forward(self, z1, z2):
        """
        Compute VICReg loss

        Args:
            z1: [batch_size, dim] - embeddings from view 1
            z2: [batch_size, dim] - embeddings from view 2

        Returns:
            loss: scalar loss value
        """
        # Invariance: embeddings from same sample should be similar
        inv_loss = self.invariance_loss(z1, z2)

        # Variance: each embedding dimension should have high variance
        var_loss = self.variance_loss(z1) + self.variance_loss(z2)

        # Covariance: embedding dimensions should be decorrelated
        cov_loss = self.covariance_loss(z1) + self.covariance_loss(z2)

        # Weighted combination
        loss = (
            self.lambda_param * inv_loss
            + self.mu_param * var_loss
            + self.nu_param * cov_loss
        )

        return loss
