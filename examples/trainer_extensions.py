# import as from 

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from djctools.module_extensions import sum_all_losses, clear_all_losses, flush_all_plotting
from djctools.wandb_tools import wandb_wrapper
import numpy as np
import os
from torch.nn import DataParallel

from typing import Any, Dict, Optional, Sequence, Tuple, Union


def compute_fisher_batch(model, batch_dict, loss_fn):
    """
    Compute Fisher information matrix estimate for a single batch:
        F = E_batch[(dL/dx)^2] 

    Parameters
    ----------
    model : nn.Module
        The underlying model (not wrapped in DataParallel).
        For MNISTModel, expects forward((x, y)).
    inputs : tuple (x, y)
        x: [B, 1, 28, 28], y: [B]
    loss_fn : nn.Module
        External loss function (e.g. nn.CrossEntropyLoss) used only for Fisher.

    Returns
    -------
    fisher : torch.Tensor, shape [28, 28]
        Per-pixel Fisher importance estimate for this batch.
    """
    x = batch_dict['inputs']
    y = batch_dict['labels']
    # Completely separate graph from training:
    x_f = x.detach().clone().requires_grad_(True)

    # Rebuild a mini-batch dict for this second forward
    batch_f = {"inputs": x_f, "labels": y}

    # Forward through the model (MNISTModel expects dict)
    out = model(batch_f)          # logits
    # out = model((batch_f,))
    loss_f = loss_fn(out, y)

    grad_x, = torch.autograd.grad(
        loss_f,
        x_f,
        retain_graph=False,
        create_graph=False,
    )  # [B, 1, 28, 28]
    fisher = (grad_x ** 2).mean(dim=0).squeeze(0).detach()  # [28, 28]
    return fisher


def compute_fisher_batch_per_image(model, batch_dict, loss_fn):
    """
    Compute per-pixel Fisher information for each image in the batch:
        F_n(i,j) ~= (dL_n/dx_n(i,j))^2

    where n indexes images in the batch and (i,j) are pixel indices.

    Returns
    -------
    fisher_per_img : torch.Tensor
        Tensor of shape [B, 1, 28, 28] with per-pixel Fisher values
        for each image in the batch.
    """

    # Move batch to the same device as the model
    device = next(model.parameters()).device

    x = batch_dict["inputs"].to(device)
    y = batch_dict["labels"].to(device)
    B = x.size(0)

    # Make x a leaf tensor with grad
    x = x.detach().clone().requires_grad_(True)

    # Storage for per-image Fisher maps
    model.eval()
    fisher = torch.zeros_like(x)

    for i in range(B):
        # Forward pass for a single image but using the entire batch tensor
        logits = model({"inputs": x[i:i+1], "labels": y[i:i+1]})
        loss_i = loss_fn(logits, y[i:i+1])   # scalar

        # Compute gradient ∂L_i / ∂x using autograd.grad
        grad_x_i = torch.autograd.grad(
            outputs=loss_i,
            inputs=x,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0][i]   # select the i-th sample, the only non-zero grad

        fisher[i] = grad_x_i.pow(2).detach()

    return fisher


class _CustomDataParallel(DataParallel):
    def scatter(
        self,
        inputs: list,
        kwargs: Optional[Dict[str, Any]],
        device_ids: Sequence[Union[int, torch.device]],
    ) -> Any:
        """
        Custom scatter method that handles the input structures provided by the Trainer class,
        such as lists of dictionaries or lists of lists of tensors.
        
        Args:
            inputs: a tuple whose first element is the list of batches
                e.g. inputs = ([ (x_0,y_0), (x_1,y_1), ... ],)

            kwargs (Optional[Dict[str, Any]]): Keyword arguments.
            device_ids (Sequence[Union[int, torch.device]]): Target devices.

        Returns:
            Tuple of scattered inputs and kwargs for each device.
        """
        # Implement custom logic to split `inputs` and `kwargs` based on your structure.
        # Example: if inputs is a list of dicts, scatter each dict entry to the devices.
        
        # Example pseudo-code:
        scattered_inputs = []
        scattered_kwargs = []
        per_device_batches = inputs[0]   # list of (x,y) 
        #print('inputs len',len(inputs))
        #print('inputs types',[type(i) for i in inputs])
        ##nested
        #print('inputs[0]',[type(i) for i in inputs[0]])

        for i, device_id in enumerate(device_ids):
            # Correct: pass the actual (x, y) for that device
            device_input = per_device_batches[i]
            device_kwargs = kwargs if kwargs is not None else {}
            scattered_inputs.append((device_input,))
            scattered_kwargs.append(device_kwargs)

        return tuple(scattered_inputs), tuple(scattered_kwargs)

class Trainer_fi:
    """
    Trainer_fi class with Fisher Information computation
    """
    def __init__(self, model, optimizer, num_gpus=1, device_ids=None, verbose_level=0):
        
        # Initialize distributed process group
        self.num_gpus = num_gpus
        if torch.cuda.is_available() and num_gpus > 0:
            # Set up gpu
            self.device_ids = device_ids if device_ids is not None else list(range(num_gpus))
            self.device = f'cuda:{self.device_ids[0]}'
            self.devices = [f'cuda:{device_id}' for device_id in self.device_ids]
        else:
            # Fall back to CPU if no GPU available or num_gpus is 1
            self.device = 'cpu'
            self.devices = ['cpu']
            self.device_ids = []
            self.num_gpus = 0
            print("Warning: CUDA not available or num_gpus=0. Using CPU.")

        # Move model to device and wrap with DDP
        model.to(self.device)
        self.model = _CustomDataParallel(model, device_ids=self.device_ids) if len(self.device_ids) > 1 else model
        self.optimizer = optimizer
        self.verbose_level = verbose_level

        self.pruner = None
        self.use_pruning = False

    def enable_pruning(self, pruner):
        self.pruner = pruner
        self.use_pruning = True

    def _data_to_device(self, data, device):
        """
        Moves data to the specified device.

        Args:
            data (tensor, dict, list, or tuple): The data to move.
            device (str): The target device.

        Returns:
            The data moved to the target device, maintaining the same structure.
        """
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {key: self._data_to_device(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._data_to_device(item, device) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._data_to_device(item, device) for item in data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def create_batches(self, data_iterator):
        """
        Manually distributes data to each device by loading a batch for each device.

        Args:
            data_iterator: Iterator over the dataset.

        Returns:
            List of batches moved to each device.
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
        """
        Runs the training loop for one epoch.

        Args:
            train_loader (DataLoader): The data loader for training data.
        """
        self.model.train()
        data_iterator = iter(train_loader)
        batch_idx = 0

        while True:
            self.optimizer.zero_grad()
            batches = self.create_batches(data_iterator)
            if not batches:
                break  # End of epoch

            # Single-GPU: batches is [ (inputs, targets) ]
            # Multi-GPU: batches is [ (inputs_0, targets_0), (inputs_1, targets_1), ... ]
            if len(batches) == 1:
                batch = batches[0]
                # pruning integration
                if self.use_pruning:
                    x = batch["inputs"]
                    pruned_x, fisher_pred = self.pruner(x)
                    batch["inputs"] = pruned_x
            else:
                batch = batches
            # print("batches: ", len(batches), "batch: ", len(batch))

            # IMPORTANT: clear any leftover loss tensors & logs BEFORE forward
            clear_all_losses(self.model)
            flush_all_plotting(self.model)

            outputs = self.model(batch)
            # outputs = self.model((batch,))
            loss = sum_all_losses(self.model)

            loss.backward()
            self.optimizer.step()

            clear_all_losses(self.model)
            flush_all_plotting(self.model)
            self.train_batch_callback(self.model, batch_idx, batch)

            # Logging and printing
            wandb_wrapper.log("total_loss", loss.item())
            if self.verbose_level > 0 and batch_idx % 10 == 0:
                print(f'Batch {batch_idx}: Loss {loss.item()}')

            batch_idx += 1
            wandb_wrapper.flush()

    def val_loop(self, val_loader):
        self.model.eval()
        data_iterator = iter(val_loader)
        batch_idx = 0
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            while True:
                batches = self.create_batches(data_iterator)
                if not batches:
                    break

                # Always use GPU0 batch
                batch_for_val = batches[0]

                if not isinstance(batch_for_val, dict):
                    raise RuntimeError("Validation expects batch as dict with 'inputs' and 'labels'.")

                batch = {
                    "inputs": batch_for_val["inputs"],
                    "labels": batch_for_val["labels"],
                }

                clear_all_losses(self.model)
                flush_all_plotting(self.model)

                outputs = self.model(batch)

                loss = sum_all_losses(self.model)

                total_loss += loss.item()
                count += 1

                clear_all_losses(self.model)
                flush_all_plotting(self.model)

                wandb_wrapper.log("val_loss", loss.item())

                if self.verbose_level > 0 and batch_idx % 10 == 0:
                    print(f'Val Batch {batch_idx}: Loss {loss.item()}')

                batch_idx += 1

        avg_loss = total_loss / max(count, 1)
        wandb_wrapper.log("avg_val_loss", avg_loss)
        return avg_loss

    def save_model(self, filepath):
        """Saves the model weights to a file."""
        torch.save(self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(), filepath)

    def load_model(self, filepath):
        """Loads model weights from a file."""
        state_dict = torch.load(filepath)
        self.model.module.load_state_dict(state_dict) if hasattr(self.model, "module") else self.model.load_state_dict(state_dict)

    def cleanup(self):
        pass

    def train_batch_callback(self, model, batch_number, batch_data):
        pass

    def val_batch_callback(self, model, batch_number, batch_data):
        pass
