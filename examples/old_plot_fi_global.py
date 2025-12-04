import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
FISHER_PATH = "/home/jpan/djctools/mnist_fisher_map.pt"
SAVE_DIR = "./fisher_visualization"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Load Fisher map
# ---------------------------------------------------------------------
fisher = torch.load(FISHER_PATH).cpu().numpy()   # shape [28,28]

print("Loaded Fisher map:", fisher.shape)
print("min:", fisher.min(), "max:", fisher.max(), "mean:", fisher.mean())


# ---------------------------------------------------------------------
# 1. Heatmap
# ---------------------------------------------------------------------
plt.figure(figsize=(6,6))
sns.heatmap(
    fisher, 
    cmap="viridis", 
    square=True,
    cbar=True
)
plt.title("Fisher Information Heatmap (MNIST)")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fisher_heatmap.png", dpi=200)
plt.close()


# ---------------------------------------------------------------------
# 2. Histogram of FI values
# ---------------------------------------------------------------------
plt.figure(figsize=(7,4))
plt.hist(fisher.flatten(), bins=80, alpha=0.85, color="teal")
plt.xlabel("Fisher Information")
plt.ylabel("Pixel Count")
plt.title("Histogram of Fisher Information Values")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fisher_histogram.png", dpi=200)
plt.close()


# ---------------------------------------------------------------------
# 3. Mask visualization (threshold: top 20% FI pixels)
# ---------------------------------------------------------------------
thresh = np.percentile(fisher, 80)
mask = fisher >= thresh

plt.figure(figsize=(6,6))
sns.heatmap(mask.astype(float), cmap="rocket", cbar=False, square=True)
plt.title("Top 20% Important Pixels (Fisher Mask)")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fisher_mask_top20.png", dpi=200)
plt.close()

print(f"\nSaved visualizations to: {SAVE_DIR}\n")