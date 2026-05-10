"""
SimCLR Model Architecture — CIFAR-10 Optimized
================================================
ResNet-50 backbone with modified stem for 32×32 images + 2-layer MLP Projection Head.

Key modification: The default ResNet stem (7×7 conv + maxpool) aggressively
downsamples the input from 224→56 pixels. For CIFAR-10's tiny 32×32 images,
this would collapse spatial info to 8×8 immediately — destroying fine-grained
features before the residual blocks even see them. We replace it with a
single 3×3 conv (stride=1) and remove the maxpool, preserving the 32×32
spatial resolution into the first residual stage.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SimCLR_ResNet50(nn.Module):
    """
    SimCLR framework with ResNet-50 encoder and MLP projection head.

    Architecture:
        Input (3×32×32)
          → Modified Stem (3×3 Conv, no MaxPool)  → 64×32×32
          → ResNet Layer1 (3 Bottleneck blocks)   → 256×32×32
          → ResNet Layer2 (4 Bottleneck blocks)   → 512×16×16
          → ResNet Layer3 (6 Bottleneck blocks)   → 1024×8×8
          → ResNet Layer4 (3 Bottleneck blocks)   → 2048×4×4
          → AdaptiveAvgPool2d                     → 2048×1×1
          → Flatten                               → h ∈ ℝ^2048 (representation)
          → Projection Head (MLP)                 → z ∈ ℝ^128   (for contrastive loss)

    Args:
        projection_dim (int): Output dim of projection head. Default: 128.
    """

    def __init__(self, projection_dim=128):
        super(SimCLR_ResNet50, self).__init__()

        # ── 1. Base Encoder f(·) ──────────────────────────────────────
        resnet = models.resnet50(weights=None)  # train from scratch

        # CIFAR-10 Stem Modification:
        # Replace 7×7 conv (stride 2) → 3×3 conv (stride 1)
        resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        # Remove max-pooling to preserve spatial resolution
        resnet.maxpool = nn.Identity()

        # Extract everything except the final FC layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        # ── 2. Projection Head g(·) ──────────────────────────────────
        # 2-layer MLP: Linear → BatchNorm → ReLU → Linear
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, projection_dim),
        )

    def forward(self, x):
        """
        Args:
            x: Input images  [B, 3, 32, 32]
        Returns:
            h: Representations from encoder  [B, 2048]  (used for downstream tasks)
            z: Projections from MLP head     [B, projection_dim]  (used for NT-Xent loss)
        """
        h = self.encoder(x)          # [B, 2048, 1, 1]
        h = h.view(h.size(0), -1)   # [B, 2048]
        z = self.projector(h)        # [B, projection_dim]
        return h, z
