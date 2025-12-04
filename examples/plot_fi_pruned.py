import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
FISHER_PATH = "/home/jpan/djctools/mnist_fisher_stageA.pt"
# Alternative for predicted Fisher:
# FISHER_PATH = "fisher_predicted.pt"

SAVE_DIR = "./fisher_visualization_pruned"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Helper plotting utilities
# ---------------------------------------------------------------------
def save_fig(fig, name):
    fig.savefig(f"{SAVE_DIR}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(data, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(data, cmap="viridis", square=True, cbar=True, ax=ax)
    ax.set_title(title)
    return fig


def plot_mask(mask, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(mask.astype(float), cmap="rocket", square=True, cbar=False, ax=ax)
    ax.set_title(title)
    return fig


def plot_histogram(data, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(data.flatten(), bins=80, alpha=0.85, color="teal")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.set_title(title)
    return fig


def overlay_true_pred(image, f_true, f_pred):
    """Create a red/blue overlay comparing true & predicted Fisher."""
    f_true_n = (f_true - f_true.min()) / (f_true.ptp() + 1e-8)
    f_pred_n = (f_pred - f_pred.min()) / (f_pred.ptp() + 1e-8)

    overlay = np.zeros((28, 28, 3))
    overlay[..., 0] = f_true_n  # red
    overlay[..., 2] = f_pred_n  # blue
    overlay = 0.6 * np.stack([image]*3, axis=-1) + 0.4 * overlay

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(overlay, interpolation="nearest")
    ax.set_title("Overlay: True Fisher (R) vs Predicted (B)")
    ax.axis("off")
    return fig


# ---------------------------------------------------------------------
# Load input file, detect its type
# ---------------------------------------------------------------------
obj = torch.load(FISHER_PATH, map_location="cpu")

if isinstance(obj, torch.Tensor):
    # -----------------------------
    # Case 1: single Fisher map [28,28]
    # -----------------------------
    fisher = obj.numpy()
    print("Loaded single Fisher map:", fisher.shape)

    # Heatmap
    fig = plot_heatmap(fisher, "Fisher Heatmap")
    save_fig(fig, "fisher_heatmap")

    # Histogram
    fig = plot_histogram(fisher, "Fisher Histogram")
    save_fig(fig, "fisher_histogram")

    # Mask
    thresh = np.percentile(fisher, 80)
    mask = fisher >= thresh
    fig = plot_mask(mask, "Top 20% Fisher Mask")
    save_fig(fig, "fisher_mask_top20")

    print(f"Saved visualizations to: {SAVE_DIR}")
    exit()


elif isinstance(obj, dict):
    # -----------------------------
    # Case 2: Stage A dataset or predicted Fisher dataset
    # -----------------------------
    images = obj.get("images")
    fisher_true = obj.get("fisher_per_image")
    fisher_pred = obj.get("fisher_predicted", None)  # optional for Stage D

    print("Loaded dataset keys:", obj.keys())

    # Pick image index 0 for visualization
    idx = 0
    img = images[idx, 0].numpy()
    f_true = fisher_true[idx, 0].numpy()

    # Heatmap (true Fisher)
    fig = plot_heatmap(f_true, "True Per-Image Fisher Heatmap")
    save_fig(fig, "true_fisher_heatmap")

    # Histogram (true Fisher)
    fig = plot_histogram(f_true, "True Fisher Histogram")
    save_fig(fig, "true_fisher_histogram")

    # Mask
    thresh = np.percentile(f_true, 80)
    mask = f_true >= thresh
    fig = plot_mask(mask, "True Top 20% Fisher Mask")
    save_fig(fig, "true_fisher_mask_top20")

    # If predicted Fisher exists (Stage D)
    if fisher_pred is not None:
        f_pred = fisher_pred[idx, 0].numpy()

        # Predicted heatmap
        fig = plot_heatmap(f_pred, "Predicted Fisher Heatmap")
        save_fig(fig, "predicted_fisher_heatmap")

        # Predicted histogram
        fig = plot_histogram(f_pred, "Predicted Fisher Histogram")
        save_fig(fig, "predicted_fisher_histogram")

        # Overlay visualization
        fig = overlay_true_pred(img, f_true, f_pred)
        save_fig(fig, "overlay_true_vs_predicted")

    print(f"Saved visualizations for dataset to: {SAVE_DIR}")

else:
    raise TypeError(f"Unsupported file type: {type(obj)}")