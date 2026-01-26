import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from trainer_extensions import compute_fisher_per_image, compute_fisher_per_image_vmap

def build_fisher_dataset(model,
                         device,
                         batch_size=128,
                         max_batches=None,
                         save_path="mnist_fisher_stageA.pt"):
    """
    Given a trained MNIST model, compute per-image Fisher maps on the
    training set and save them along with the images and labels.

    save_path will contain:
    {
        "images":          [N, 1, 28, 28],
        "labels":          [N],
        "fisher_per_image":    [N, 1, 28, 28],
        "fisher_log1p":        [N, 1, 28, 28],   # log(1+F)
    }
    """
    # --- Dataset & loader ---
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=False, num_workers=4
    )

    model.eval()
    model.to(device)
    ce_loss = torch.nn.CrossEntropyLoss()

    all_images, all_labels, all_fisher = [], [], []

    with torch.no_grad():
        # we will temporarily re-enable grad only for Fisher computation per batch
        pass

    # We do need gradients for Fisher, so we wrap the loop in no_grad=False
    batch_count = 0
    for images, labels in train_loader:
        batch_count += 1
        if (max_batches is not None) and (batch_count > max_batches):
            break

        images = images.to(device)
        labels = labels.to(device)
        batch_dict = {
            "inputs": images,
            "labels": labels,
        }

        # Need grads here:
        fi_batch = compute_fisher_per_image(
            model=model,
            batch_dict=batch_dict,
            loss_fn=ce_loss
        )  # [B, 1, 28, 28]

        all_images.append(images.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_fisher.append(fi_batch.detach().cpu())
        print(f"[Stage A] Processed batch {batch_count}")

    images_tensor = torch.cat(all_images, dim=0)         # [N, 1, 28, 28]
    labels_tensor = torch.cat(all_labels, dim=0)         # [N]
    fisher_tensor = torch.cat(all_fisher, dim=0)         # [N, 1, 28, 28]

    # Stabilised target to save: fi := log(1 + Fi)
    fisher_log1p = torch.log1p(fisher_tensor)

    torch.save(
        {
            "images": images_tensor,
            "labels": labels_tensor,
            "fisher_per_image": fisher_tensor,
            "fisher_log1p": fisher_log1p,
        },
        save_path,
    )
    print(f"[Stage A] Saved Fisher dataset to {save_path}")


if __name__ == "__main__":
    # saving the model for debugging for now
    from mnist_djc_wrapper import build_trained_mnist_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_trained_mnist_model(
        device=device,
        ckpt_path="checkpoints/mnist_trained_model.pt",
        use_dataparallel=False
    )

    build_fisher_dataset(
        model=model,
        device=device,
        batch_size=128,
        max_batches=None,  # None = full training set
        save_path="mnist_fisher_stageA.pt",
    )