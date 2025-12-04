# fisher_pruning.py

import torch
import torch.nn as nn

class FisherPruningGate(nn.Module):
    """
    Stage D pruning module:
    Uses a trained Fisher approximator to prune input images.
    
    Assumes:
        - approximator outputs log(1 + Fisher)
        - inputs: [B,1,28,28]
    """

    def __init__(self, approximator, sparsity=0.2, mode="percentile"):
        """
        sparsity: fraction of pixels to prune per image (e.g. 0.2 = drop lowest 20%)
        mode: "percentile" or "topk"
        """
        super().__init__()
        self.approx = approximator
        self.sparsity = sparsity
        self.mode = mode

        # Set approximator to eval & freeze
        self.approx.eval()
        for p in self.approx.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        """
        x: [B,1,28,28] raw images
        returns: pruned_x, predicted_fisher
        """

        device = x.device

        with torch.no_grad():
            # Predict log(1 + Fisher)
            log1p_f = self.approx(x)                    # [B,1,28,28]
            fisher_hat = torch.expm1(log1p_f).clamp(min=0.0)

        B = x.size(0)
        pruned_x = x.clone()

        for i in range(B):
            fi = fisher_hat[i].view(-1)

            if self.mode == "percentile":
                # prune lowest fraction = sparsity
                k = int(fi.numel() * self.sparsity)
                if k < 1:
                    continue
                thresh = torch.kthvalue(fi, k).values
                mask = (fisher_hat[i] >= thresh).float()

            elif self.mode == "topk":
                k = int(fi.numel() * (1 - self.sparsity))
                if k < 1:
                    continue
                topvals, _ = torch.topk(fi, k)
                thresh = topvals.min()
                mask = (fisher_hat[i] >= thresh).float()

            else:
                raise ValueError(f"Unknown pruning mode: {self.mode}")

            pruned_x[i] = x[i] * mask  # apply pixel-wise mask

        return pruned_x, fisher_hat