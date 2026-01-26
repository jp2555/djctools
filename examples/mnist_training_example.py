# try to import torchvision and print warning if does not exist
try:
    import torchvision
except ImportError:
    print("Warning: torchvision not found. Please install torchvision to run this example.")
    exit()

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# djctools components
from djctools.module_extensions import (
    LossModule,
    LoggingModule,
    switch_all_logging,
    switch_all_losses,
)
from djctools.training import Trainer  # not used directly, for reference
from djctools.wandb_tools import wandb_wrapper

from trainer_extensions import Trainer_fi, compute_fisher_per_image
from mnist_djc_wrapper import MNIST_DJC
from train_fisher_approximator_mnist import FisherApproximator


class MNISTLossModule(LossModule):
    def __init__(self, **kwargs):
        super(MNISTLossModule, self).__init__(**kwargs)
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, outputs, targets):
        """
        If loss is active, this function will be called.
        If it is not active, this function will not be called at all.
        """
        loss = self.criterion(outputs, targets)
        self.log("loss", loss)
        return loss


class MNISTAccuracyModule(LoggingModule):
    def __init__(self, **kwargs):
        super(MNISTAccuracyModule, self).__init__(**kwargs)

    def compute_metrics(self, outputs, targets):
        _, predicted = torch.max(outputs.data, 1)
        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        accuracy = 100.0 * correct / total
        self.log("accuracy", accuracy)


class FisherProxyLossModule(LossModule):
    """
    Loss module for supervising the Fisher approximator:
    MSE between predicted log(1+F_hat) and true log(1+F_true).
    """

    def __init__(self, weight: float = 1.0, **kwargs):
        super(FisherProxyLossModule, self).__init__(**kwargs)
        self.weight = weight

    def compute_loss(self, pred_log1p, true_log1p):
        """
        pred_log1p: [B,1,28,28] predicted log(1+F_hat)
        true_log1p: [B,1,28,28] true log(1+F)
        """
        mse = F.mse_loss(pred_log1p, true_log1p)
        loss = self.weight * mse
        self.log("proxy_mse", mse)
        self.log("proxy_loss", loss)
        return loss


class MNISTModel(nn.Module):
    def __init__(
        self,
        use_fisher_approximator: bool = True,
        fisher_proxy_weight: float = 1.0,
        fisher_approx_ckpt: str | None = None,
    ):
        """
        MNIST classifier + Fisher approximator layer.

        - Keeps original conv → relu → pool → fc structure.
        - Adds:
          * forward_logits(x): tensor-only path, used by both training and FI.
          * fisher_approximator: CNN that predicts log(1+Fisher).
          * fisher_proxy_loss_module: supervises approximator vs true FI.
          * compute_fisher_in_forward: flag to toggle true-FI computation.
        """
        super(MNISTModel, self).__init__()

        # Original classifier bits
        self.loss_module = MNISTLossModule(
            logging_active=True,
            loss_active=True,
            name="MNISTLossModule",
        )

        self.accuracy_module = MNISTAccuracyModule(
            logging_active=True,
            name="MNISTAccuracyModule",
        )

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 14 * 14, 10)

        # For Fisher/proxy training
        self._fisher_ce = nn.CrossEntropyLoss()
        self.compute_fisher_in_forward = False  # toggled from outside
        self.last_fisher_true = None            # storage hook for inspection
        self.last_fisher_pred = None

        # Fisher approximator submodule 
        self.use_fisher_approximator = use_fisher_approximator
        if use_fisher_approximator:
            self.fisher_approximator = FisherApproximator()
            # Optionally load pre-trained weights
            if fisher_approx_ckpt is not None and os.path.isfile(fisher_approx_ckpt):
                state = torch.load(fisher_approx_ckpt, map_location="cpu")
                self.fisher_approximator.load_state_dict(state)
                print(f"[MNISTModel] Loaded Fisher approximator from {fisher_approx_ckpt}")
            else:
                if fisher_approx_ckpt is not None:
                    print(
                        f"[MNISTModel] Warning: fisher_approx_ckpt='{fisher_approx_ckpt}' "
                        "not found, Fisher approximator will be randomly initialized."
                    )
        else:
            self.fisher_approximator = None

        # Proxy loss module
        self.fisher_proxy_loss_module = FisherProxyLossModule(
            weight=fisher_proxy_weight,
            logging_active=True,
            loss_active=True,
            name="FisherProxyLossModule",
        )

    # --- helper: tensor-only path for logits ---
    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,1,28,28] -> logits: [B,10]
        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 14 * 14)
        outputs = self.fc1(x)
        return outputs

    # --- public API used by Trainer_fi ---
    def forward(self, data):
        """
        data: dict with keys:
          "inputs": [B,1,28,28]
          "labels": [B] or None
        """
        x = data["inputs"]
        y = data["labels"]

        logits = self.forward_logits(x)

        if y is not None:
            # Classification loss + accuracy logging
            self.loss_module(logits, y)
            self.accuracy_module(logits, y)

            # -----------------------------
            # True Fisher + proxy training
            # -----------------------------
            if self.use_fisher_approximator and self.training and self.compute_fisher_in_forward:
                # 1) True Fisher per image (detached from main graph)
                fi_true = compute_fisher_per_image(
                    self,
                    {"inputs": x, "labels": y},
                    self._fisher_ce,
                )  # [B,1,28,28]
                self.last_fisher_true = fi_true

                # Work in log(1+F) space for stability
                eps = 1e-8
                fi_true_log1p = torch.log1p(fi_true + eps)

                # 2) Predicted log(1+F_hat) from approximator submodule
                fi_pred_log1p = self.fisher_approximator(x)
                self.last_fisher_pred = fi_pred_log1p

                # 3) Proxy MSE loss via LossModule (integrates into sum_all_losses)
                self.fisher_proxy_loss_module(fi_pred_log1p, fi_true_log1p)

        return logits


    def compute_approx_fisher_per_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Utility to get approximated Fisher per image after training:

            F_hat ≈ expm1(predicted_log1p)

        x: [B,1,28,28]
        """
        if not self.use_fisher_approximator or self.fisher_approximator is None:
            raise RuntimeError("Fisher approximator is not enabled in this model.")

        device = next(self.parameters()).device
        x = x.to(device)
        self.fisher_approximator.to(device)
        self.fisher_approximator.eval()

        with torch.no_grad():
            log1p_f = self.fisher_approximator(x)
            fisher_hat = torch.expm1(log1p_f).clamp(min=0.0)
        return fisher_hat


def main():

    wandb_wrapper.init(project="mnist_trainer_example")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 128
    num_epochs = 5
    learning_rate = 0.001
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_gpus = min(num_gpus, 1)  # keep 1 GPU for now

    # Data transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Dataset wrapped in MNIST_DJC to produce dict batches
    train_dataset = MNIST_DJC(
        torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
    )
    val_dataset = MNIST_DJC(
        torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Initialize the model and optimizer
    model = MNISTModel(
        use_fisher_approximator=True,
        fisher_proxy_weight=1.0,          # tune this if proxy dominates/weak
        fisher_approx_ckpt=None,          # or e.g. "fisher_approx_mnist.pt"
    ).to(device)

    # Turn on proxy training (true FI inside forward)
    model.compute_fisher_in_forward = True

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    trainer = Trainer_fi(model, optimizer, num_gpus=num_gpus)

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}")
        trainer.train_loop(train_loader)
        trainer.val_loop(val_loader)

    # Turn off losses/logging for inference
    switch_all_losses(model, False)
    switch_all_logging(model, False)

    # Inference test
    mock_data = torch.randn(1, 1, 28, 28).to(device)
    model({"inputs": mock_data, "labels": None})

    trainer.save_model("mnist_model_with_fisher_proxy.pth")
    
    wandb_wrapper.finish()


if __name__ == "__main__":
    main()