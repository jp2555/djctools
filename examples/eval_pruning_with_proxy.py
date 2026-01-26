# eval_pruning_with_proxy.py
#
# Evaluate accuracy vs. pruning sparsity using:
#   - Proxy Fisher (approximator submodule),
#   - True Fisher (per-image gradients),
# on the MNIST test set, using the trained MNISTModel from
# mnist_training_example.py.

DEBUG_PROXY_ONCE = True
DEBUG_TRUE_ONCE = True

import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import wandb

from djctools.wandb_tools import wandb_wrapper
from mnist_training_example import MNISTModel
from mnist_djc_wrapper import MNIST_DJC
from trainer_extensions import compute_fisher_per_image


# pruning
def apply_per_image_pruning(inputs, fisher_map, sparsity: float) -> torch.Tensor:
    """
    Per-image pruning based on Fisher importance.

    inputs:    [B,1,H,W]  (normalized images)
    fisher_map:[B,1,H,W]  (non-negative importance values)
    sparsity:  fraction of pixels to prune per image (0.0 .. 1.0)

    We explicitly zero out the k = sparsity * num_pixels *lowest* pixels in each image.
    """
    if sparsity <= 0.0:
        return inputs.clone()

    B, C, H, W = inputs.shape
    pruned = inputs.clone()

    num_pixels = C * H * W
    k = int(num_pixels * sparsity)
    if k <= 0:
        return pruned  # nothing to prune

    fisher_flat = fisher_map.view(B, -1)  # [B, num_pixels]

    for i in range(B):
        fi = fisher_flat[i]  # [num_pixels]

        # Indices of the k smallest Fisher values
        # (largest=False gives smallest; no thresholding ambiguity)
        _, idx_small = torch.topk(fi, k, largest=False)

        # Start with all-ones mask
        mask_flat = torch.ones_like(fi, dtype=torch.bool)
        mask_flat[idx_small] = False  # these pixels will be zeroed

        mask = mask_flat.view(1, H, W).float()  # [1,H,W]
        pruned[i] = inputs[i] * mask

    return pruned


def evaluate_accuracy(
    model: MNISTModel,
    dataloader: DataLoader,
    device: torch.device,
    mode: str = "none",
    sparsity: float = 0.0,
):
    model.eval()
    total = 0
    correct = 0

    compute_true_fisher = (mode == "true")
    use_proxy_fisher = (mode == "proxy")

    for batch in dataloader:
        x = batch["inputs"].to(device)   # [B,1,28,28]
        y = batch["labels"].to(device)   # [B]

        if compute_true_fisher and sparsity > 0.0:
            # ----- TRUE FISHER BRANCH (gradients ON) -----
            # No torch.no_grad() here
            fi_true = compute_fisher_per_image(
                model,
                {"inputs": x, "labels": y},
                model._fisher_ce,  # your CE loss for FI
            )  # [B,1,28,28]

            x_in = apply_per_image_pruning(x, fi_true, sparsity)

            # We don't need grads for the classification itself, so we can
            # optionally turn no_grad on just for the forward_logits call:
            with torch.no_grad():
                logits = model.forward_logits(x_in)

        else:
            # ----- NO PRUNING or PROXY FISHER (no gradients needed) -----
            with torch.no_grad():
                global DEBUG_PROXY_ONCE

                if mode == "proxy" and sparsity > 0.0:
                    fisher_hat = model.compute_approx_fisher_per_image(x)  # [B,1,28,28]
                    x_in = apply_per_image_pruning(x, fisher_hat, sparsity)

                    if DEBUG_PROXY_ONCE:
                        frac_zero_before = (x == 0).float().mean().item()
                        frac_zero_after  = (x_in == 0).float().mean().item()
                        print(
                            f"[DEBUG proxy] sparsity={sparsity:.2f}, "
                            f"zero_frac_before={frac_zero_before:.3f}, "
                            f"zero_frac_after={frac_zero_after:.3f}"
                        )
                        print(
                            f"[DEBUG proxy] fisher_hat stats: "
                            f"min={fisher_hat.min().item():.4g}, "
                            f"max={fisher_hat.max().item():.4g}, "
                            f"mean={fisher_hat.mean().item():.4g}"
                        )
                        DEBUG_PROXY_ONCE = False

                else:
                    x_in = x

                logits = model.forward_logits(x_in)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return 100.0 * correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # Load trained model
    # ------------------------------
    ckpt_path = "mnist_model_with_fisher_proxy.pth"
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint '{ckpt_path}' not found. "
            "Run examples/mnist_training_example.py first."
        )

    # Instantiate model with approximator submodule
    model = MNISTModel(
        use_fisher_approximator=True,
        fisher_proxy_weight=0.0,   # weight irrelevant for inference
        fisher_approx_ckpt=None,   # full model weights loaded below
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    # We don't want to recompute Fisher inside forward during this eval
    model.compute_fisher_in_forward = False

    # ------------------------------
    # MNIST test set
    # ------------------------------
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    test_raw = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_dataset = MNIST_DJC(test_raw)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    # ------------------------------
    # W&B init
    # ------------------------------
    # You can adjust this grid if you like
    sparsity_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    wandb_wrapper.init(
        project="mnist_pruning_eval",
        config={
            "checkpoint": ckpt_path,
            "sparsity_list": sparsity_list,
        },
    )

    # ------------------------------
    # Baseline (no pruning)
    # ------------------------------
    baseline_acc = evaluate_accuracy(
        model, test_loader, device, mode="none", sparsity=0.0
    )
    print(f"[Baseline] Accuracy (no pruning): {baseline_acc:.2f}%")
    wandb_wrapper.log("baseline_accuracy", baseline_acc)
    wandb_wrapper.flush()

    # ------------------------------
    # Proxy Fisher pruning curve
    # ------------------------------
    proxy_results = []
    for s in sparsity_list:
        acc = evaluate_accuracy(
            model, test_loader, device, mode="proxy", sparsity=s
        )
        proxy_results.append((s, acc))
        print(f"[Proxy] sparsity={s:.2f}, accuracy={acc:.2f}%")

        # wandb_wrapper.log expects (metric_name, value)
        wandb_wrapper.log("sparsity", s)
        wandb_wrapper.log("accuracy_proxy", acc)
        wandb_wrapper.flush()

    # ------------------------------
    # True Fisher pruning curve
    # (comment out this block if you want proxy-only evaluation)
    # ------------------------------
    true_results = []
    for s in sparsity_list:
        acc = evaluate_accuracy(
            model, test_loader, device, mode="true", sparsity=s
        )
        true_results.append((s, acc))
        print(f"[True FI] sparsity={s:.2f}, accuracy={acc:.2f}%")

        wandb_wrapper.log("sparsity", s)
        wandb_wrapper.log("accuracy_true_fisher", acc)
        wandb_wrapper.flush()

    # ------------------------------
    # Make a Matplotlib plot: accuracy vs sparsity
    # ------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))

    # Unpack results
    sparsities = sparsity_list
    proxy_accs = [dict(proxy_results)[s] for s in sparsities]
    true_accs = [dict(true_results)[s] for s in sparsities] if true_results else None

    # Baseline as horizontal line
    ax.axhline(
        baseline_acc,
        linestyle="--",
        linewidth=1.0,
        label=f"Baseline (no pruning): {baseline_acc:.2f}%",
    )

    ax.plot(
        sparsities,
        proxy_accs,
        marker="o",
        linestyle="-",
        label="Proxy Fisher pruning",
    )

    if true_accs is not None:
        ax.plot(
            sparsities,
            true_accs,
            marker="s",
            linestyle="-",
            label="True Fisher pruning",
        )

    ax.set_xlabel("Pruning sparsity (fraction of pixels removed)")
    ax.set_ylabel("Accuracy [%]")
    ax.set_title("MNIST accuracy vs. pruning sparsity")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plot_path = "mnist_pruning_curves.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)

    # Log plot to W&B as image if run is active
    if getattr(wandb_wrapper, "initialized", False):
        img = wandb.Image(plot_path, caption="Accuracy vs pruning sparsity")
        wandb.log({"pruning_curves": img})

    wandb_wrapper.finish()

    # Final summary printout
    print("\n=== Summary: Accuracy vs Sparsity ===")
    print("sparsity   baseline   proxy_FI   true_FI")
    for s in sparsities:
        proxy_acc = dict(proxy_results).get(s, float("nan"))
        true_acc = dict(true_results).get(s, float("nan"))
        print(
            f"{s:8.2f}  {baseline_acc:8.2f}  {proxy_acc:8.2f}  {true_acc:8.2f}"
        )


if __name__ == "__main__":
    main()