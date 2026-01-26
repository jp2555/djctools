import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from djctools.module_extensions import (
    sum_all_losses,
    clear_all_losses,
    flush_all_plotting,
)
from djctools.wandb_tools import wandb_wrapper
from torch.nn import DataParallel

from typing import Any, Dict, Optional, Sequence, Tuple, Union

def compute_fisher_per_image(model, batch_dict, loss_fn):
    """
    Compute per-pixel Fisher information for each image in the batch:
        F_n(i,j) ~= (dL_n/dx_n(i,j))^2

    where n indexes images in the batch and (i,j) are pixel indices.

    Parameters
    ----------
    model : nn.Module
        Underlying model (not wrapped in DataParallel) which provides
        a `forward_logits(x)` method returning logits.
    batch_dict : dict
        {
          "inputs": [B,1,H,W],
          "labels": [B]
        }
    loss_fn : nn.Module
        Loss function, e.g. nn.CrossEntropyLoss.

    Returns
    -------
    fisher_per_img : torch.Tensor
        Tensor of shape [B, 1, H, W] with per-pixel Fisher values
        for each image in the batch.
    """

    device = next(model.parameters()).device
    x = batch_dict["inputs"].to(device)
    y = batch_dict["labels"].to(device)

    B = x.size(0)
    # Make x a leaf tensor with grad
    x_leaf = x.detach().clone().requires_grad_(True)

    # Storage for per-image Fisher maps
    fisher_per_img = torch.zeros_like(x_leaf)

    # Remember training/eval mode and switch to eval for consistency
    was_training = model.training
    model.eval()

    for i in range(B):
        # Clear old grads
        model.zero_grad()
        if x_leaf.grad is not None:
            x_leaf.grad.zero_()

        xi = x_leaf[i:i+1]  # [1,1,H,W]
        yi = y[i:i+1]       # [1]

        # Use tensor-only path to avoid passing dicts into the model
        logits = model.forward_logits(xi)  # [1,num_classes]

        loss_i = loss_fn(logits, yi)
        loss_i.backward()

        grad_x_i = x_leaf.grad[i].detach()  # [1,H,W]
        fisher_per_img[i] = grad_x_i.pow(2)

    # Restore original mode
    model.train(was_training)

    return fisher_per_img


class _CustomDataParallel(DataParallel):
    def scatter(
        self,
        inputs: list,
        kwargs: Optional[Dict[str, Any]],
        device_ids: Sequence[Union[int, torch.device]],
    ) -> Any:
        """
        Custom scatter method that handles the input structures
        provided by the Trainer_fi class, such as lists of dicts.

        Args:
            inputs: tuple whose first element is the list of batches
                    e.g. inputs = ([ batch0, batch1, ... ],)
            kwargs: Keyword arguments.
            device_ids: Target devices.

        Returns:
            Tuple of scattered inputs and kwargs for each device.
        """
        scattered_inputs = []
        scattered_kwargs = []

        per_device_batches = inputs[0]  # list of per-device batches

        for i, device_id in enumerate(device_ids):
            device_input = (per_device_batches[i],)  # keep tuple form
            device_kwargs = kwargs if kwargs is not None else {}
            scattered_inputs.append(device_input)
            scattered_kwargs.append(device_kwargs)

        return tuple(scattered_inputs), tuple(scattered_kwargs)


class Trainer_fi:
    """
    Trainer_fi class for (optionally multi-GPU) training, kept as
    close to djctools.Trainer as possible, but without global Fisher
    accumulation logic. Fisher-related operations can be triggered
    from outside using `compute_fisher_per_image` and the model's
    helper methods.
    """

    def __init__(self, model, optimizer, num_gpus=1, device_ids=None, verbose_level=0):
        # GPU / CPU setup
        self.num_gpus = num_gpus
        if torch.cuda.is_available() and num_gpus > 0:
            self.device_ids = device_ids if device_ids is not None else list(range(num_gpus))
            self.device = f"cuda:{self.device_ids[0]}"
            self.devices = [f"cuda:{d}" for d in self.device_ids]
        else:
            self.device = "cpu"
            self.devices = ["cpu"]
            self.device_ids = []
            self.num_gpus = 0
            print("Warning: CUDA not available or num_gpus=0. Using CPU.")

        # Move model to device, wrap with DP if multi-GPU
        model.to(self.device)
        self.model = (
            _CustomDataParallel(model, device_ids=self.device_ids)
            if len(self.device_ids) > 1
            else model
        )
        self.optimizer = optimizer
        self.verbose_level = verbose_level

    def _data_to_device(self, data, device):
        """
        Moves data to the specified device.
        """
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {k: self._data_to_device(v, device) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._data_to_device(d, device) for d in data]
        elif isinstance(data, tuple):
            return tuple(self._data_to_device(d, device) for d in data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def create_batches(self, data_iterator):
        """
        Manually distributes data to each device by loading a batch for each device.
        """
        batches = []
        for d in self.devices:
            try:
                data = next(data_iterator)
                data = self._data_to_device(data, d)
                batches.append(data)
            except StopIteration:
                return []  # End of data
        return batches

    def train_loop(self, train_loader):
        self.model.train()
        data_iterator = iter(train_loader)
        batch_idx = 0

        while True:
            self.optimizer.zero_grad()
            batches = self.create_batches(data_iterator)
            if not batches:
                break  # End of epoch

            # Single-GPU: batches is [batch]; multi-GPU: list of per-device batches
            if len(batches) == 1:
                batch = batches[0]
            else:
                batch = batches  # passed as list to _CustomDataParallel

            clear_all_losses(self.model)
            flush_all_plotting(self.model)

            outputs = self.model(batch)
            loss = sum_all_losses(self.model)

            loss.backward()
            self.optimizer.step()

            clear_all_losses(self.model)
            flush_all_plotting(self.model)
            self.train_batch_callback(self.model, batch_idx, batch)

            wandb_wrapper.log("total_loss", loss.item())
            if self.verbose_level > 0 and batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Loss {loss.item():.4f}")

            batch_idx += 1
            wandb_wrapper.flush()

    def val_loop(self, val_loader):
        self.model.eval()
        data_iterator = iter(val_loader)
        batch_idx = 0

        with torch.no_grad():
            while True:
                batches = self.create_batches(data_iterator)
                if not batches:
                    break  # End of epoch

                if len(batches) == 1:
                    batch = batches[0]
                else:
                    batch = batches  # DP case

                clear_all_losses(self.model)
                flush_all_plotting(self.model)

                outputs = self.model(batch)
                loss = sum_all_losses(self.model)

                clear_all_losses(self.model)
                flush_all_plotting(self.model)
                self.val_batch_callback(self.model, batch_idx, batch)

                wandb_wrapper.log("val_loss", loss.item())
                if self.verbose_level > 0 and batch_idx % 10 == 0:
                    print(f"Val Batch {batch_idx}: Loss {loss.item():.4f}")

                batch_idx += 1
                wandb_wrapper.flush(prefix="val_")

    def save_model(self, filepath):
        """Saves the model weights to a file."""
        torch.save(
            self.model.module.state_dict()
            if hasattr(self.model, "module")
            else self.model.state_dict(),
            filepath,
        )

    def load_model(self, filepath):
        """Loads model weights from a file."""
        state_dict = torch.load(filepath)
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

    def cleanup(self):
        pass

    def train_batch_callback(self, model, batch_number, batch_data):
        pass

    def val_batch_callback(self, model, batch_number, batch_data):
        pass