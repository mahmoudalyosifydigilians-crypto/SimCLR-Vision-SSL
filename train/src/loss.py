"""
NT-Xent Loss (Normalized Temperature-scaled Cross-Entropy)
==========================================================
The core contrastive loss function for SimCLR.

For a batch of N images, SimCLR produces 2N augmented views (two per image).
Each image i has exactly one positive pair (its other augmented view) and
2(N−1) negative pairs (all other views). NT-Xent treats this as a 2N-way
classification problem: for each anchor, predict which of the 2N−1 other
views is its positive partner.

Temperature τ controls the sharpness of the softmax distribution:
  - Lower τ → sharper peaks → model focuses on hard negatives
  - Higher τ → softer distribution → more uniform gradients

Optimal τ for CIFAR-10 ≈ 0.5 (from SimCLR paper, Table B.1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross-Entropy) Loss.

    Given two batches of projected features z_i, z_j (from two augmented views
    of the same batch), computes the InfoNCE-style contrastive loss.

    Args:
        temperature (float): Scaling factor τ for cosine similarity. Default: 0.5.
    """

    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: Projections from augmented view 1  [N, D]
            z_j: Projections from augmented view 2  [N, D]

        Returns:
            loss: Scalar NT-Xent loss.

        The similarity matrix (2N × 2N) looks like:
            ┌─────────────┬─────────────┐
            │  sim(i, i')  │  sim(i, j)  │  ← z_i rows
            ├─────────────┼─────────────┤
            │  sim(j, i)  │  sim(j, j')  │  ← z_j rows
            └─────────────┴─────────────┘
        Positive pairs sit on the off-diagonal blocks at positions (i, i+N) and (i+N, i).
        """
        batch_size = z_i.size(0)

        # ── Step 1: Concatenate & L2-normalize ──────────────────────
        z = torch.cat([z_i, z_j], dim=0)        # [2N, D]
        z = F.normalize(z, dim=1)                # unit sphere

        # ── Step 2: Cosine similarity matrix ────────────────────────
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # [2N, 2N]

        # ── Step 3: Mask out self-similarity (diagonal) ─────────────
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix.masked_fill_(mask, -1e4)  # large negative, safe for fp16 (AMP)

        # ── Step 4: Construct positive-pair labels ──────────────────
        # For row i     (0..N−1):   positive is at column i+N
        # For row i+N   (N..2N−1):  positive is at column i
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size),
        ]).to(z.device)

        # ── Step 5: Cross-entropy over the 2N-way classification ────
        loss = F.cross_entropy(sim_matrix, labels)

        return loss
