import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import SimCLR_ResNet50
from src.loss import NTXentLoss
from src.dataset import get_train_dataloader, get_eval_dataloader


#  Config
def parse_args():
    p = argparse.ArgumentParser(description="SimCLR Pre-training on CIFAR-10")

    p.add_argument("--data_dir",       type=str,   default="./data",
                   help="Directory to download/load CIFAR-10")
    p.add_argument("--output_dir",     type=str,   default="./outputs",
                   help="Directory to save checkpoints, plots, and logs")

    p.add_argument("--epochs",         type=int,   default=20,
                   help="Number of training epochs (20 for midterm, 200 for full)")
    p.add_argument("--batch_size",     type=int,   default=128,
                   help="Batch size (128 for CPU, 1024 for GPU)")
    p.add_argument("--lr",             type=float, default=3e-4,
                   help="Base learning rate for AdamW")
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--temperature",    type=float, default=0.5,
                   help="NT-Xent temperature τ (paper optimal: 0.5 for CIFAR-10)")
    p.add_argument("--projection_dim", type=int,   default=128,
                   help="Projection head output dimension")

    p.add_argument("--warmup_epochs",  type=int,   default=3,
                   help="Linear warmup epochs (3 for midterm, 10 for full training)")

    p.add_argument("--save_every",     type=int,   default=10,
                   help="Save checkpoint every N epochs")
    p.add_argument("--tsne_final_only",action="store_true", default=True,
                   help="Run t-SNE only at the final epoch (recommended for CPU)")

    p.add_argument("--num_workers",    type=int,   default=0,
                   help="DataLoader workers (use 0 on Windows)")
    p.add_argument("--seed",           type=int,   default=42,
                   help="Random seed for reproducibility")

    return p.parse_args()



#  Reproducibility
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


#  LR Scheduler: Linear Warmup + Cosine Annealing
def get_lr_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):

    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = total_epochs  * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup: 0 → 1
            return float(step) / max(warmup_steps, 1)
        # Cosine decay: 1 → 0
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



#  t-SNE Visualization
@torch.no_grad()
def visualize_tsne(model, device, data_dir, epoch, output_dir, num_workers=0):
    """
    Generate t-SNE scatter plot of encoder representations h.
    Uses 2000 test samples for speed on CPU.
    """
    print(f"\n  > Generating t-SNE for epoch {epoch} (this may take ~2 min on CPU)...")
    model.eval()

    loader = get_eval_dataloader(
        data_dir, train=False, batch_size=256, num_workers=num_workers
    )

    features, labels = [], []
    for x, y in loader:
        x = x.to(device)
        h, _ = model(x)               # h = representation, _ = projection
        features.append(h.cpu().numpy())
        labels.append(y.numpy())
        if sum(f.shape[0] for f in features) >= 2000:
            break

    features = np.concatenate(features)[:2000]
    labels   = np.concatenate(labels)[:2000]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    emb  = tsne.fit_transform(features)

    cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        emb[:, 0], emb[:, 1], c=labels,
        cmap="tab10", alpha=0.6, s=8
    )
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.ax.set_yticklabels(cifar10_classes)
    ax.set_title(
        f"SimCLR Learned Representations — Epoch {epoch}\n"
        f"(ResNet-50 encoder h, CIFAR-10 test set, 2000 samples)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_xticks([])
    ax.set_yticks([])

    path = os.path.join(output_dir, "plots", f"tsne_epoch_{epoch:03d}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [PLOT] t-SNE saved → {path}")



#  Training Loop

def train(args):

    set_seed(args.seed)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"),       exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"),        exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() 

    print(f"\n{'='*60}")
    print(f"  SimCLR Pre-Training — CIFAR-10")
    print(f"  CISC 867 | Project 20 | Mahmoud Alyosify")
    print(f"{'='*60}")
    print(f"  Device          : {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if use_amp else " (CPU)"))
    print(f"  Epochs          : {args.epochs}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Learning rate   : {args.lr}")
    print(f"  Temperature τ   : {args.temperature}")
    print(f"  Projection dim  : {args.projection_dim}")
    print(f"  Warmup epochs   : {args.warmup_epochs}")
    print(f"  Mixed Precision : {'YES (AMP)' if use_amp else 'NO (CPU mode)'}")
    print(f"  Seed            : {args.seed}")
    print(f"{'='*60}\n")


    train_loader = get_train_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    steps_per_epoch = len(train_loader)
    print(f"  Training samples : {len(train_loader.dataset):,}")
    print(f"  Steps per epoch  : {steps_per_epoch}")
    print(f"  Negatives/image  : {2 * args.batch_size - 2}\n")

    # Model, Loss, Optimizer, Scheduler
    model     = SimCLR_ResNet50(projection_dim=args.projection_dim).to(device)
    criterion = NTXentLoss(temperature=args.temperature).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_lr_scheduler(
        optimizer, args.warmup_epochs, args.epochs, steps_per_epoch
    )
    scaler = GradScaler(enabled=use_amp)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model parameters : {total_params:.2f}M\n")

    #Training
    loss_history = []
    best_loss    = float("inf")
    log_path     = os.path.join(args.output_dir, "logs", "training_log.csv")

    # Write CSV header
    with open(log_path, "w") as f:
        f.write("epoch,avg_loss,lr,time_sec\n")

    print(f"  {'Epoch':>6}  {'Loss':>8}  {'LR':>10}  {'Time':>8}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*8}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for (x_i, x_j), _ in train_loader:
            x_i = x_i.to(device, non_blocking=True)
            x_j = x_j.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                _, z_i = model(x_i)
                _, z_j = model(x_j)
                loss   = criterion(z_i, z_j)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()


        avg_loss = epoch_loss / steps_per_epoch
        elapsed  = time.time() - t0
        lr_now   = optimizer.param_groups[0]["lr"]

        loss_history.append(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss

        print(f"  {epoch:>6d}  {avg_loss:>8.4f}  {lr_now:>10.6f}  {elapsed:>6.1f}s")

        # Append to CSV log
        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f},{lr_now:.8f},{elapsed:.2f}\n")


        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(
                args.output_dir, "checkpoints", f"simclr_epoch_{epoch:03d}.pth"
            )
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss":                 avg_loss,
                "args":                 vars(args),
            }, ckpt_path)
            print(f"          [SAVE] Checkpoint  {ckpt_path}")


    visualize_tsne(
        model, device, args.data_dir,
        args.epochs, args.output_dir, args.num_workers
    )


    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        range(1, len(loss_history) + 1), loss_history,
        linewidth=2, color="#2196F3", marker="o", markersize=4
    )
    ax.axhline(y=best_loss, color="red", linestyle="--",
               alpha=0.5, label=f"Best loss: {best_loss:.4f}")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("NT-Xent Loss", fontsize=12)
    ax.set_title(
        f"SimCLR Training Loss — {args.epochs} Epochs\n"
        f"(ResNet-50, CIFAR-10, batch={args.batch_size}, τ={args.temperature})",
        fontsize=12, fontweight="bold"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    curve_path = os.path.join(args.output_dir, "plots", "training_loss_curve.png")
    fig.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [PLOT] Loss curve → {curve_path}")

    # ── Save final encoder weights only (for evaluate.py) ─────────
    encoder_path = os.path.join(args.output_dir, "simclr_encoder.pth")
    torch.save(model.state_dict(), encoder_path)
    print(f"  [DONE] Encoder weights → {encoder_path}")

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best NT-Xent Loss : {best_loss:.4f}")
    print(f"  Log saved         : {log_path}")
    print(f"{'='*60}\n")



#  Entry Point

if __name__ == "__main__":
    args = parse_args()
    train(args)
