# train_fisher_approximator_mnist.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

class FisherDataset(Dataset):
    def __init__(self, path):
        obj = torch.load(path, map_location="cpu")
        self.images = obj["images"]            # [N, 1, 28, 28]
        # Use log(1+Fisher) as target
        if "fisher_log1p" in obj:
            self.targets = obj["fisher_log1p"]
        else:
            self.targets = torch.log1p(obj["fisher_per_image"])

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        x = self.images[idx]      # [1, 28, 28]
        y = self.targets[idx]     # [1, 28, 28]
        return x, y


class FisherApproximator(nn.Module):
    """
    Small CNN: input [B,1,28,28] -> output [B,1,28,28]
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


def train_fisher_approximator(
    fisher_dataset_path="mnist_fisher_stageA.pt",
    save_model_path="fisher_approx_mnist.pt",
    batch_size=128,
    num_epochs=10,
    lr=1e-3,
    val_fraction=0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_ds = FisherDataset(fisher_dataset_path)

    N = len(full_ds)
    n_val = int(val_fraction * N)
    n_train = N - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = FisherApproximator().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(images)          # [B,1,28,28], predicts log(1+F)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                preds = model(images)
                loss = criterion(preds, targets)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[Stage B] Epoch {epoch+1}/{num_epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    torch.save(model.state_dict(), save_model_path)
    print(f"[Stage B] Saved Fisher approximator to {save_model_path}")


if __name__ == "__main__":
    train_fisher_approximator()