# IMPORT PART
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
# Importing transforms from augmentations.py
from augmentations import SimCLRViewGenerator, transform_exp1, transform_exp2, transform_exp3, transform_exp4, transform_exp5, transform_exp6, transform_exp7, transform_exp8

# LOAD CIFAR10
# Benchmarking on CIFAR-10 to evaluate augmentation robustness
# 'train=True' isolates the 50,000 training samples; 'download=True' ensures
# local data availability for reproducible experimental runs.
train_raw = CIFAR10(
    root="../data",
    train=True,
    download=True )


## CUSTOM SIMCLR DATASET WRAPPER
## Positive Pair Generation for Contrastive Learning
class AugmentedDataset(Dataset):
    """
    A specialized dataset wrapper designed to facilitate Contrastive Learning.
    It extends the standard PyTorch Dataset to return multiple stochastic views 
    of each sample alongside its ground-truth label.
    """
    def __init__(self, dataset, transform):
        # Reference to the underlying base dataset (e.g., CIFAR-10).
        self.dataset = dataset
        
        # Wrapping the input transform with SimCLRViewGenerator to automate 
        # the creation of 'n_views' (positive pairs) per sample.
        self.transform = SimCLRViewGenerator(transform, n_views=2)

    def __len__(self):
        # Returns the total number of samples available in the base dataset.
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieves the raw image and its corresponding label from the base dataset.
        img, label = self.dataset[idx]
        
        # Applies the stochastic transformation pipeline to generate two distinct views.
        # 'views' is a list containing [view_1, view_2].
        views = self.transform(img)
        
        # Returns the positive pair (views) and the label, facilitating both 
        # contrastive loss calculation and supervised evaluation if needed.
        return views[0], views[1], label



# Dataset Initialization for Ablation Study
# Exp 1-2: Focus on basic spatial transformations (Cropping and Flipping).
train_exp1_ds = AugmentedDataset(train_raw, transform_exp1)
train_exp2_ds = AugmentedDataset(train_raw, transform_exp2)

# Exp 3-4: Focus on isolated photometric distortions (Color Jittering and Grayscale).
train_exp3_ds = AugmentedDataset(train_raw, transform_exp3)
train_exp4_ds = AugmentedDataset(train_raw, transform_exp4)

# Exp 5-7: Hybrid configurations testing the interaction between spatial and color noise.
train_exp5_ds = AugmentedDataset(train_raw, transform_exp5)
train_exp6_ds = AugmentedDataset(train_raw, transform_exp6)
train_exp7_ds = AugmentedDataset(train_raw, transform_exp7)

# Exp 8: The full SimCLR suite (The integrated 'optimal' augmentation policy).
train_exp8_ds = AugmentedDataset(train_raw, transform_exp8)