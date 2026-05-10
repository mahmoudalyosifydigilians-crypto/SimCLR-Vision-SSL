"""
SimCLR Pre-Training Script — CIFAR-10
======================================
Full training loop with:
  • Mixed Precision (AMP) for Tensor Core acceleration on RTX 5000 Ada
  • Cosine Annealing LR Schedule with Linear Warmup
  • Periodic t-SNE visualization of learned representations
  • Checkpointing every N epochs
  • Training loss curve plotting

Usage:
    python train.py                          # defaults (50 epochs, batch=512)
    python train.py --epochs 200 --batch_size 1024
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from sklearn.manifold import TSNE

# ── Local imports ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from src.model import SimCLR_ResNet50
from src.loss import NTXentLoss
from src.dataset import get_train_dataloader, get_eval_dataloader


# ══════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="SimCLR Pre-training on CIFAR-10")
    p.add_argument("--data_dir",      type=str,   default=r"d:/Deep Learning Project/data")
    p.add_argument("--output_dir",    type=str,   default=r"d:/Deep Learning Project/Mid term Report Work/Mahmoud/outputs")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=512)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--temperature",   type=float, default=0.5)
    p.add_argument("--projection_dim",type=int,   default=128)
    p.add_argument("--warmup_epochs", type=int,   default=10)
    p.add_argument("--vis_every",     type=int,   default=10,  help="t-SNE visualization interval")
    p.add_argument("--save_every",    type=int,   default=10,  help="Checkpoint save interval")
    p.add_argument("--num_workers",   type=int,   default=0,   help="DataLoader workers (0 for Windows)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
#  LR Scheduler: Linear Warmup + Cosine Annealing
# ══════════════════════════════════════════════════════════════════
def get_lr_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """Linear warmup for `warmup_epochs`, then cosine decay to 0."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════════
#  t-SNE Visualization
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def visualize_tsne(model, device, data_dir, epoch, output_dir, num_workers=0):
    """Generate a t-SNE scatter plot of encoder representations."""
    print(f"  > Generating t-SNE visualization for epoch {epoch}...")
    model.eval()

    loader = get_eval_dataloader(data_dir, train=False, batch_size=512, num_workers=num_workers)

    features, labels = [], []
    for x, y in loader:
        x = x.to(device)
        h, _ = model(x)
        features.append(h.cpu().numpy())
        labels.append(y.numpy())
        if sum(f.shape[0] for f in features) >= 2000:
            break

    features = np.concatenate(features)[:2000]
    labels = np.concatenate(labels)[:2000]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb = tsne.fit_transform(features)

    cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", alpha=0.6, s=8)
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.ax.set_yticklabels(cifar10_classes)
    ax.set_title(f"SimCLR Representations - Epoch {epoch}", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_xticks([])
    ax.set_yticks([])

    path = os.path.join(output_dir, "plots", f"tsne_epoch_{epoch:03d}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved -> {path}")


# ══════════════════════════════════════════════════════════════════
#  Training Loop
# ══════════════════════════════════════════════════════════════════
def train(args):
    # ── Directories ───────────────────────────────────────────────
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)

    # ── Device ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f"  SimCLR Pre-Training on CIFAR-10")
    print(f"{'='*60}")
    print(f"  Device          : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Learning rate   : {args.lr}")
    print(f"  Temperature (t) : {args.temperature}")
    print(f"  Projection dim  : {args.projection_dim}")
    print(f"  Warmup epochs   : {args.warmup_epochs}")
    print(f"  Mixed Precision : {'YES (AMP + Tensor Cores)' if torch.cuda.is_available() else 'NO'}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────
    train_loader = get_train_dataloader(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    steps_per_epoch = len(train_loader)
    print(f"  Training samples : {len(train_loader.dataset)}")
    print(f"  Steps per epoch  : {steps_per_epoch}\n")

    # ── Model / Loss / Optimizer ──────────────────────────────────
    model = SimCLR_ResNet50(projection_dim=args.projection_dim).to(device)
    criterion = NTXentLoss(temperature=args.temperature).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = get_lr_scheduler(optimizer, args.warmup_epochs, args.epochs, steps_per_epoch)

    # ── AMP (Mixed Precision) ─────────────────────────────────────
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # ── Training ──────────────────────────────────────────────────
    loss_history = []
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, ((x_i, x_j), _) in enumerate(train_loader, 1):
            x_i = x_i.to(device, non_blocking=True)
            x_j = x_j.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass with AMP
            with autocast(enabled=torch.cuda.is_available()):
                _, z_i = model(x_i)
                _, z_j = model(x_j)
                loss = criterion(z_i, z_j)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps_per_epoch
        loss_history.append(avg_loss)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        if avg_loss < best_loss:
            best_loss = avg_loss

        print(
            f"  Epoch [{epoch:3d}/{args.epochs}]  "
            f"Loss: {avg_loss:.4f}  "
            f"LR: {lr_now:.6f}  "
            f"Time: {elapsed:.1f}s"
        )

        # ── Checkpoint ────────────────────────────────────────────
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(
                args.output_dir, "checkpoints", f"simclr_epoch_{epoch:03d}.pth"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"    [SAVE] Checkpoint -> {ckpt_path}")

        # ── t-SNE Visualization ───────────────────────────────────
        if epoch % args.vis_every == 0 or epoch == args.epochs:
            visualize_tsne(model, device, args.data_dir, epoch, args.output_dir, args.num_workers)

    # ── Final Loss Curve ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(loss_history) + 1), loss_history, linewidth=2, color="#2196F3")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("NT-Xent Loss", fontsize=12)
    ax.set_title("SimCLR Training Loss Curve", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    path = os.path.join(args.output_dir, "plots", "training_loss_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [PLOT] Loss curve -> {path}")

    # ── Save final encoder ────────────────────────────────────────
    final_path = os.path.join(args.output_dir, "simclr_pretrained.pth")
    torch.save(model.state_dict(), final_path)
    print(f"  [DONE] Final model -> {final_path}")
    print(f"\n{'='*60}")
    print(f"  Training complete!  Best loss: {best_loss:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
