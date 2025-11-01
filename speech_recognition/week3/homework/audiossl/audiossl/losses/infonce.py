"""
InfoNCE Contrastive Loss

This loss encourages positive pairs (two different views of the same input)
to have high similarity, while pushing apart negative pairs.

Mathematical formulation:
For a positive pair (z_i, z_j), the InfoNCE loss is:

    ℓ(i,j) = -log( exp(sim(z_i, z_j)/τ) / Σ_k≠i exp(sim(z_i, z_k)/τ) )

where:
- τ is the temperature parameter
- sim(u, v) is cosine similarity
- The denominator includes all negatives in the batch

References:
- "Representation Learning with Contrastive Predictive Coding" (van den Oord et al., 2018)
  Original InfoNCE paper (arXiv:1807.03748)
- "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR, Chen et al., ICML 2020)
  Popular implementation of InfoNCE for vision
- "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"
  (Baevski et al., NeurIPS 2020, arXiv:2006.11477)
  Applied InfoNCE to speech/audio domain

Implementation based on:
- SSL_Seminar.ipynb
- papers/2010.09542v1.pdf (wav2vec 2.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def device_as(t1, t2):
    """Helper to move tensor t1 to same device as t2"""
    return t1.to(t2.device)


class InfoNCELoss(nn.Module):
    """
    InfoNCE (contrastive) loss for multi-view learning

    Args:
        temperature: Temperature parameter controlling sharpness of similarities
                    Common values: [0.05, 0.5], default 0.5 from seminar
    """  # noqa: E501

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def calc_similarity_batch(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Calculate pairwise cosine similarity between all pairs in concatenated views

        Args:
            a: [batch_size, dim] - first view embeddings
            b: [batch_size, dim] - second view embeddings

        Returns:
            similarity_matrix: [2*batch_size, 2*batch_size] - all pairwise similarities
        """
        # Concatenate both views: [2*batch_size, dim]
        rep = torch.cat([a, b])

        # Compute cosine similarity between all pairs
        # unsqueeze for broadcasting: [2N, 1, dim] x [1, 2N, dim] -> [2N, 2N, dim]
        # then compute similarity along dim=2
        return F.cosine_similarity(rep.unsqueeze(1), rep.unsqueeze(0), dim=2)

    def forward(self, proj_1, proj_2):
        """
        Compute InfoNCE loss for a batch of positive pairs

        Args:
            proj_1: [batch_size, dim] - embeddings from first view
            proj_2: [batch_size, dim] - embeddings from second view

        Returns:
            loss: scalar InfoNCE loss
        """
        batch_size = proj_1.shape[0]

        # Normalize embeddings to unit hypersphere
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        # Compute similarity matrix: [2N, 2N]
        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        # Similarity matrix structure:
        # +--------+--------+
        # | aa_sim | ab_sim |  <- view a vs all
        # +--------+--------+
        # | ba_sim | bb_sim |  <- view b vs all
        # +--------+--------+

        # Extract positive pairs from off-diagonal blocks
        # sim_ij: similarity between view_i and corresponding view_j
        sim_ij = torch.diag(similarity_matrix, batch_size)  # upper-right diagonal
        sim_ji = torch.diag(similarity_matrix, -batch_size)  # lower-left diagonal

        # Concatenate all positive pairs: [2N]
        positives = torch.cat([sim_ij, sim_ji])

        # Numerator: exp(positive similarities / temperature)
        nominator = torch.exp(positives / self.temperature)

        # Denominator: sum of exp(all similarities / temperature) excluding self-comparisons
        # Create mask to exclude diagonal (self-comparisons)
        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)).float()
        mask = device_as(mask, similarity_matrix)

        # Compute denominator with mask
        denominator = mask * torch.exp(similarity_matrix / self.temperature)
        denominator = torch.sum(denominator, dim=1)

        # InfoNCE loss: -log(exp(positive) / sum(exp(all_negatives)))
        all_losses = -torch.log(nominator / denominator)

        # Average over all positive pairs
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss


class ContrastiveLoss(InfoNCELoss):
    """Alias for backwards compatibility with seminar code"""

    pass
