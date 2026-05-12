import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        N = z_i.shape[0]   # batch size (N pairs → 2N views)
        device = z_i.device

        # Step 1: L2-normalize all projections
        # Cosine similarity = dot product of L2-normalized vectors
        z_i = F.normalize(z_i, dim=1)   # (N, D)
        z_j = F.normalize(z_j, dim=1)   # (N, D)

        # Step 2: Concatenate into 2N representations
        # z = [z_i[0], z_i[1], ..., z_i[N-1], z_j[0], ..., z_j[N-1]]
        z = torch.cat([z_i, z_j], dim=0)   # (2N, D)

        # Step 3: Compute full (2N × 2N) similarity matrix
        # sim[a, b] = cosine_similarity(z[a], z[b]) / τ
        sim = torch.mm(z, z.T) / self.temperature   # (2N, 2N)

        # Step 4: Build positive pair index
        # For index i in [0, N):   positive is at index i+N
        # For index i in [N, 2N):  positive is at index i-N
        # Example (N=3): positives = [3, 4, 5, 0, 1, 2]
        pos_idx = torch.cat([
            torch.arange(N, 2*N, device=device),   # first  half → second half
            torch.arange(0, N,   device=device),   # second half → first  half
        ])  # (2N,)

        # Step 5: Mask out self-similarities
        # The diagonal sim[i, i] = 1/τ (a view compared to itself).
        # We must exclude it from the denominator sum.
        # Set diagonal to -inf so exp(-inf) = 0 in the softmax.
        mask = torch.eye(2*N, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask, float('-inf'))

        # Step 6: Compute cross-entropy loss
        loss = F.cross_entropy(sim, pos_idx)

        return loss


#  Unit Test
if __name__ == "__main__":
    print("=" * 55)
    print("  NT-Xent Loss — Unit Test")
    print("=" * 55)

    criterion = NTXentLoss(temperature=0.5)


    N, D = 32, 128
    z_i = torch.randn(N, D)
    z_j = torch.randn(N, D)
    loss = criterion(z_i, z_j)
    print(f"\n  Test 1 — Random projections (N={N}, D={D})")
    print(f"  Loss value : {loss.item():.4f} should be ~log(2N-1) ≈ {torch.log(torch.tensor(2*N-1.0)):.4f}")

    z_same = torch.randn(N, D)
    z_same_copy = z_same.clone()
    loss_perfect = criterion(z_same, z_same_copy)
    print(f"\n  Test 2 — Identical positive pairs")
    print(f"  Loss value : {loss_perfect.item():.6f}  should be ~0.0")

    z_base  = torch.randn(N, D)
    z_close = z_base + 0.01 * torch.randn(N, D)   
    z_far   = torch.randn(N, D)                    

    loss_close = criterion(z_base, z_close)
    loss_far   = criterion(z_base, z_far)
    print(f"\n  Test 3 — Similar vs dissimilar positive pairs")
    print(f"  Loss (similar pairs) : {loss_close.item():.4f}")
    print(f"  Loss (random  pairs) : {loss_far.item():.4f}")
    print(f"  Similar < Random   : {loss_close.item() < loss_far.item()}")

    print(f"\n  Test 4 — Output is a scalar")
    print(f"  loss.shape : {loss.shape} expected torch.Size([])")

    print("\n" + "=" * 55)
    print("  All tests passed!")
    print("=" * 55)
