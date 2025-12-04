# calibrate_fisher_approximator_with_wandb.py

import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from train_fisher_approximator_mnist import FisherDataset, FisherApproximator


def fig_to_wandb(fig):
    """Convert a Matplotlib figure to a W&B image."""
    return wandb.Image(fig)


def plot_scatter(true_vals, pred_vals, max_points=5000):
    """Scatter plot of true Fisher vs predicted Fisher."""
    # flatten
    t = true_vals.flatten()
    p = pred_vals.flatten()

    # subsample to avoid huge scatter files
    if t.size > max_points:
        idx = np.random.choice(len(t), max_points, replace=False)
        t = t[idx]
        p = p[idx]

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.scatterplot(x=t, y=p, s=8, alpha=0.4, ax=ax)
    ax.set_xlabel("True log(1 + F)")
    ax.set_ylabel("Predicted log(1 + F)")
    ax.set_title("Calibration Scatter: True vs Predicted")
    plt.tight_layout()

    return fig


def plot_histograms(true_vals, pred_vals):
    """Histograms of true and predicted Fisher distributions."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].hist(true_vals.flatten(), bins=50, alpha=0.7)
    ax[0].set_title("True log(1+F) Histogram")

    ax[1].hist(pred_vals.flatten(), bins=50, alpha=0.7)
    ax[1].set_title("Pred log(1+F) Histogram")

    plt.tight_layout()
    return fig


def plot_error_map(true_map, pred_map):
    """Heatmap of prediction error per pixel."""
    err = pred_map - true_map

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(err, cmap="coolwarm", center=0, cbar=True, square=True)
    ax.set_title("Error Map: Pred - True")
    plt.tight_layout()
    return fig


def plot_overlay(image, true_map, pred_map):
    """Overlay predicted importance onto the raw image."""
    # Convert true Fisher to probability-like via min-max
    t = (true_map - true_map.min()) / (true_map.ptp() + 1e-8)
    p = (pred_map - pred_map.min()) / (pred_map.ptp() + 1e-8)

    # Build overlay: red = true importance, blue = predicted
    overlay = np.zeros((28, 28, 3))
    overlay[..., 0] = t  # true FI = red
    overlay[..., 2] = p  # predicted = blue
    overlay = 0.7 * image + 0.3 * overlay

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(overlay)
    ax.set_title("Overlay: True (R) vs Pred (B)")
    ax.axis("off")
    plt.tight_layout()
    return fig


def run_wandb_calibration(
    fisher_dataset_path="mnist_fisher_stageA.pt",
    approximator_path="fisher_approx_mnist.pt",
    num_batches=20,
    batch_size=64,
):

    # --------------------------
    # Init W&B
    # --------------------------
    wandb.init(
        project="fisher-mnist-calibration",
        config={
            "dataset": fisher_dataset_path,
            "approximator": approximator_path,
            "num_batches": num_batches,
            "batch_size": batch_size,
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------
    # Load dataset & model
    # --------------------------
    ds = FisherDataset(fisher_dataset_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = FisherApproximator().to(device)
    model.load_state_dict(torch.load(approximator_path, map_location=device))
    model.eval()

    # --------------------------
    # Loop over batches
    # --------------------------
    for step, (x, y_true) in enumerate(dl):
        if step >= num_batches:
            break

        x = x.to(device)
        y_true = y_true.to(device)

        with torch.no_grad():
            y_pred = model(x)  # log(1 + F_hat)

        # Move to CPU numpy
        x_np = x.cpu().numpy()
        t_np = y_true.cpu().numpy()
        p_np = y_pred.cpu().numpy()

        # --------------------------
        # Log per-batch metrics
        # --------------------------
        mse = ((p_np - t_np) ** 2).mean()
        wandb.log({"batch_mse": mse}, step=step)

        # --------------------------
        # Make & log figures
        # --------------------------

        # Scatter (flatten)
        fig = plot_scatter(t_np, p_np)
        wandb.log({"calib_scatter": fig_to_wandb(fig)}, step=step)
        plt.close(fig)

        # Histograms
        fig = plot_histograms(t_np, p_np)
        wandb.log({"histograms": fig_to_wandb(fig)}, step=step)
        plt.close(fig)

        # For the first image in batch, log detailed maps
        img = x_np[0, 0]
        true_map = t_np[0, 0]
        pred_map = p_np[0, 0]

        # Error map
        fig = plot_error_map(true_map, pred_map)
        wandb.log({"error_map": fig_to_wandb(fig)}, step=step)
        plt.close(fig)

        # Overlay
        fig = plot_overlay(img, true_map, pred_map)
        wandb.log({"overlay": fig_to_wandb(fig)}, step=step)
        plt.close(fig)

        print(f"[W&B Calibration] Logged batch {step}")

    wandb.finish()


if __name__ == "__main__":
    run_wandb_calibration()