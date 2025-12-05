import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from mnist_djc_wrapper import build_trained_mnist_model
from train_fisher_approximator_mnist import FisherApproximator
from fisher_pruning import FisherPruningGate

import wandb
from trainer_extensions import compute_fisher_batch_per_image
import torch.nn.functional as F


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["inputs"].to(device)
            y = batch["labels"].to(device)

            logits = model({"inputs": x, "labels": y})
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def evaluate_with_true_fisher_pruning(model, dataloader, device, sparsity=0.2):
    model.eval()
    correct = 0
    total = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    for batch in dataloader:
        x = batch["inputs"].to(device)
        y = batch["labels"].to(device)

        # compute per-image FI
        fi = compute_fisher_batch_per_image(model, {"inputs": x, "labels": y}, loss_fn)  # [B,1,28,28]

        B = x.size(0)
        pruned = x.clone()

        for i in range(B):
            fi_flat = fi[i].view(-1)
            k = int(fi_flat.numel() * sparsity)
            if k < 1:
                continue
            thresh = torch.kthvalue(fi_flat, k).values
            mask = (fi[i] >= thresh).float()
            pruned[i] = x[i] * mask

        logits = model({"inputs": pruned, "labels": y})
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total


def evaluate_with_pruning(model, pruner, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["inputs"].to(device)
            y = batch["labels"].to(device)

            # Apply pruning
            x_pruned, fisher_pred = pruner(x)

            logits = model({"inputs": x_pruned, "labels": y})
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project="mnist-pruning-eval", config={"model_ckpt": "mnist_model.pth"})

    # --------------------------
    # Load trained classifier
    # --------------------------
    model = build_trained_mnist_model(
        device=device,
        ckpt_path="mnist_model.pth",
        use_dataparallel=False,
    )

    # --------------------------
    # Load approximator & pruner
    # --------------------------
    approx = FisherApproximator().to(device)
    approx.load_state_dict(torch.load("fisher_approx_mnist.pt", map_location=device))

    # Try several pruning levels
    sparsity_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # --------------------------
    # MNIST test dataset
    # --------------------------
    import torchvision.transforms as transforms
    from mnist_djc_wrapper import MNIST_DJC

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = MNIST_DJC(
        torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # --------------------------
    # Evaluate baseline accuracy
    # --------------------------
    base_acc = evaluate(model, test_loader, device)
    wandb.log({"baseline_accuracy": base_acc})
    print(f"\n=== Baseline accuracy (no pruning) = {base_acc*100:.2f}% ===\n")

    print("\n=== True Fisher Pruning Evaluation ===")
    true_fisher_results = []
    for s in sparsity_levels:
        print(f"Evaluating TRUE Fisher pruning sparsity = {s}")
        acc_tf = evaluate_with_true_fisher_pruning(model, test_loader, device, sparsity=s)
        wandb.log({"true_fisher_sparsity": s, "true_fisher_accuracy": acc_tf})
        print(f"  True Fisher Accuracy (s={s}): {acc_tf*100:.2f}%")
        true_fisher_results.append((s, acc_tf))

    # --------------------------
    # Evaluate pruned accuracies
    # --------------------------
    results = []

    for s in sparsity_levels:
        print(f"Evaluating pruning sparsity = {s}")

        pruner = FisherPruningGate(approx, sparsity=s, mode="percentile").to(device)
        acc = evaluate_with_pruning(model, pruner, test_loader, device)
        wandb.log({"sparsity": s, "accuracy": acc})

        print(f"  Accuracy with pruning (s={s}): {acc*100:.2f}%")
        results.append((s, acc))

    print("\n=== Summary ===")
    for s, acc in results:
        print(f"Sparsity {s:.2f} → Accuracy {acc*100:.2f}%")

    print("\n=== Summary: True Fisher Pruning ===")
    for s, acc in true_fisher_results:
        print(f"Sparsity {s:.2f} → True Fisher Accuracy {acc*100:.2f}%")

    wandb.finish()


if __name__ == "__main__":
    main()