"""
Data Augmentation Pipeline for SimCLR — CIFAR-10
=================================================
SimCLR relies heavily on data augmentation to learn invariant representations.
Each image is transformed twice independently to create a positive pair.

The augmentation pipeline follows the original paper (Chen et al., 2020):
  1. Random Resized Crop  — forces the model to learn scale invariance
  2. Random Horizontal Flip
  3. Color Jitter          — prevents the model from relying on color histograms
  4. Random Grayscale      — further color invariance
  5. Gaussian Blur         — adapted kernel size for 32×32 images
  6. Normalize             — CIFAR-10 channel statistics
"""

from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


class SimCLRAugmentation:
    """
    Applies the base transform independently `n_views` times to produce
    multiple augmented views of the same image.

    Args:
        base_transform: The composition of augmentations to apply.
        n_views (int): Number of augmented views per image. Default: 2.
    """

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]


def get_simclr_transform(size=32):
    """
    SimCLR augmentation pipeline tuned for CIFAR-10 (32×32 images).

    Key differences from ImageNet-scale SimCLR:
      - Crop size = 32 (not 224)
      - GaussianBlur kernel_size = 3 (not 23) — proportional to image size
      - Color jitter strength slightly reduced for stability
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4,
                saturation=0.4, hue=0.1
            )
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])


def get_eval_transform():
    """Simple transform for linear evaluation / testing (no augmentation)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])


def get_train_dataloader(data_dir, batch_size=256, num_workers=0):
    """
    CIFAR-10 training dataloader with SimCLR dual-view augmentation.
    Each sample returns ([view1, view2], label).
    """
    transform = SimCLRAugmentation(get_simclr_transform(), n_views=2)
    dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_eval_dataloader(data_dir, train=False, batch_size=256, num_workers=0):
    """CIFAR-10 dataloader for evaluation (single view, no augmentation)."""
    dataset = CIFAR10(
        root=data_dir, train=train, download=True,
        transform=get_eval_transform(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
