from torch.utils.data import Dataset
import torch
import os


class MNIST_DJC(Dataset):
    def __init__(self, mnist_dataset):
        self.ds = mnist_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]

        # djctools trainer expects dict-based batches
        return {
            "inputs": x,     # image
            "labels": y,     # classification label
            "truth": y,      # LossModule expects truth here
        }
    

# Import your MNIST model architecture
# Adjust this import depending on your repo structure
from mnist_training_example import MNISTModel   # <-- ensure this matches your actual file


def build_trained_mnist_model(
    device=None,
    ckpt_path="checkpoints/mnist_trained_model.pt",
    use_dataparallel=False,
    strict=True,
):
    """
    Helper to load a trained MNIST model for Stage A Fisher computation or evaluation.

    Parameters
    ----------
    device : torch.device or None
        If None, choose CUDA if available.
    ckpt_path : str
        Path to the model checkpoint (.pt or .pth).
    use_dataparallel : bool
        Wrap model in torch.nn.DataParallel if True.
    strict : bool
        Whether to use strict=True when loading state_dict.

    Returns
    -------
    model : nn.Module
        The trained MNIST model in eval() mode on the chosen device.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint '{ckpt_path}' not found. "
            "Make sure training has produced the weight file."
        )

    print(f"[MNIST Wrapper] Loading trained model from: {ckpt_path}")

    # 1. Instantiate the architecture
    model = MNISTModel()      # <-- adjust if your model constructor uses args
    model = model.to(device)

    # 2. Load checkpoint
    state = torch.load(ckpt_path, map_location=device)

    # If checkpoint stored more things than just `state_dict`
    if "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    # 3. Wrap DataParallel (optional)
    if use_dataparallel and torch.cuda.device_count() > 1:
        print("[MNIST Wrapper] Using DataParallel")
        model = torch.nn.DataParallel(model)

        # If checkpoint was saved without DP, we load into .module
        try:
            model.module.load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            print("[MNIST Wrapper] Falling back to loading directly into model (DP mismatch).")
            model.load_state_dict(state_dict, strict=strict)
    else:
        model.load_state_dict(state_dict, strict=strict)

    # 4. Set eval mode immediately (important for Fisher extraction)
    model.eval()

    print("[MNIST Wrapper] Model successfully loaded and set to eval()")

    return model