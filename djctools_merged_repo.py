from .wandb_tools import wandb_wrapper
import torch


import threading
import logging

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)  # Set the default level to WARNING

class LoggingModule(torch.nn.Module):
    """
    LoggingModule integrates logging capabilities into PyTorch modules, allowing 
    selective logging for metrics in nested module structures.

    It is a torch.nn.Module class with integrated logging capabilities. Logs can be
    selectively enabled or disabled for this module and any nested LoggingModule
    instances.

    Args:
        logging_active (bool): Set to True to enable logging for this module.

    User methods:
        log(metric_name, value): Logs a metric if logging is enabled, otherwise does nothing.
                                 Value can be a float, int, or tensor on any device. No need to break
                                 jit-compatibility by converting to numpy or calling .item() here.
        compute_metrics(*args, **kwargs): Should be implemented in subclasses.
                                          If logging is enabled, this function will be called by the forward method.
                                          If logging is disabled, this function will not be called at all.
        switch_logging(enable_logging): Enables or disables logging for
                                        this module and all nested LoggingModule instances.
    """

    _instance_count = 0 

    def __init__(self, name=None, logging_active=False):
        super(LoggingModule, self).__init__()

        # Assign a unique name if none is provided
        if name is None:
            LoggingModule._instance_count += 1
            self.name = f"LoggingModule{LoggingModule._instance_count}"
        else:
            self.name = name

        self.switch_logging(logging_active)

    def _log(self, metric_name, value, skip_prefix=False):
        """
        Logs a metric using the wandb wrapper.

        Args:
            metric_name (str): The name of the metric.
            value (float): The value of the metric.
            skip_prefix (bool): If True, skips prefixing the metric name 
                                with the module's name (Default: False).
        
        Note:
            Prefixing the metric name with the module's name is helpful for 
            distinguishing metrics from different instances in nested structures.
        """
        if not skip_prefix:
            metric_name = f"{self.name}_{metric_name}"
        wandb_wrapper.log(metric_name, value)

    def _no_op(self, *args, **kwargs):
        """No-op function that does nothing, used when logging is disabled."""
        pass

    def compute_metrics(self, *args, **kwargs):
        """
        Placeholder for the actual metric computation.
        Should be implemented in subclasses.
        To log a metric, use the self.log function within this function.

        When logging is enabled, this function will replace the forward method.
        If logging is disabled, this function will not be called at all.
        
        This enables use of truth information when logging without requiring truth
        data in inference-only scenarios.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        
        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses of LoggingModule must implement the compute_metrics method.")

    def switch_logging(self, logging_active):
        """
        Enables or disables logging for this module and all nested LoggingModule instances.

        Args:
            logging_active (bool): True to enable logging, False to disable it.

        Note:
            When logging is enabled, `log` is set to `_log` and `forward` to `compute_metrics`.
            When disabled, both are set to `_no_op`. This approach supports JIT compatibility
            and prevents unnecessary computation.
        """
        self.log = self._log if logging_active else self._no_op
        self.forward = self.compute_metrics if logging_active else self._no_op
        for child in self.children():
            if isinstance(child, LoggingModule):
                child.switch_logging(logging_active)

    @property
    def logging_active(self):
        """Read-only property to access the logging state."""
        return self.log == self._log



class LossModule(LoggingModule):
    def __init__(self, name=None, logging_active=False, loss_active=True):
        """
        LossModule extends LoggingModule to enable modular loss computation, allowing 
        fine-grained control over loss terms in complex models.

        It is a PyTorch module designed to compute and record individual loss terms, inheriting from LoggingModule.
        This module allows toggling loss calculation on or off for efficient handling of multiple loss terms within
        complex model structures. Each LossModule instance stores its own computed losses, which can later be aggregated
        across a model.
    
        Attributes:
            _losses (list): An instance-level list that stores computed losses for the module, enabling
                            retrieval and aggregation of losses when needed.
            loss_active (bool): A property that returns whether loss calculation is enabled or disabled.
    
        Args:
            name (str, optional): Optional name for the module. If None, a unique name will be assigned.
            logging_active (bool): If True, enables logging for this module.
            loss_active (bool): If True, enables loss calculation for this module. Default is True.
    
        Methods:
            forward(*args, **kwargs): Computes the loss by calling compute_loss and appends it to the instance's loss list.
                                      This method is dynamically reassigned based on the `loss_active` state.
            compute_loss(*args, **kwargs): Should be implemented in subclasses.
                                           Returns a single scalar tensor representing the loss.
            switch_loss_calculation(enable_loss): Enables or disables loss calculation, dynamically assigning forward to
                                                  either compute_loss or a no-op function for JIT compatibility.
            clear_losses(): Clears all recorded losses for the instance, useful for resetting losses after aggregation.
            
            sum_all_losses(module): Recursively collects and sums losses from all LossModule instances within a given module.
                                    Returns the total loss as a single scalar tensor.
            switch_all_losses(module, enable_loss): Recursively enables or disables loss calculation for all LossModule
                                                    instances within a given module.
        """
        super(LossModule, self).__init__(name=name, logging_active=logging_active)
        self._losses = []  # Instance-level list to store losses for this LossModule
        self.switch_loss_calculation(loss_active)

    @property
    def loss_active(self):
        """Read-only property to access the loss calculation state."""
        return self.forward == self._compute_loss_and_record

    def _compute_loss_and_record(self, *args, **kwargs):
        """Compute the loss and append to the instance's loss list."""
        loss = self.compute_loss(*args, **kwargs)
        self._losses.append(loss)

    def compute_loss(self, *args, **kwargs):
        """
        Placeholder for the actual loss computation. Should be implemented in subclasses.
        This function will be called by `forward` when the loss calculation is enabled.

        Must return a single scalar tensor representing the loss.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses of LossModule must implement the compute_loss method.")
    
    def switch_logging(self, logging_active):
        """
        Enables or disables logging for this module and all nested submodules.
        This only affects calls to the `log` method, not the forward method.

        Args:
            logging_active (bool): True to enable logging, False to disable it.
        """
        self.log = self._log if logging_active else self._no_op
        for child in self.children():
            if isinstance(child, LoggingModule): # now these are all nested logging modules
                child.switch_logging(logging_active)

    def switch_loss_calculation(self, loss_active):
        """
        Enables or disables the loss calculation for this module, dynamically setting `forward` to either
        `_compute_loss_and_record` (enabled) or `_no_op` (disabled) for JIT compatibility.

        Args:
            loss_active (bool): True to enable loss calculation, False to disable it.

        Note:
            This method applies recursively to all nested LossModule instances within the module.
        """
        self.forward = self._compute_loss_and_record if loss_active else self._no_op

        # Recursively apply to all child modules
        for child in self.children():
            if isinstance(child, LossModule):
                child.switch_loss_calculation(loss_active)

    def clear_losses(self):
        """Clears the accumulated losses in this module's instance-level loss list."""
        self._losses.clear()



class PlottingModule(torch.nn.Module):
    """
    This layer is used to enable or disable plotting from within the model.
    It is meant as a base class from which to inherit, and should not be used directly.
    The logic works as follows:
      - If plotting is enabled, the forward method caches the data given to it while the model is executing. 
        It does not return anything, and it also does not start any plotting process.
      - If plotting is disabled, the forward method does nothing.
      - Once the model has finished executing, the plotting process can be started by calling the flush method.
        This method will access the cached data, and launches a new thread in which the plotting process is
        started to avoid blocking the main thread. 
      - The plotting thread will call the plot method, which should be implemented in the subclass.
      - The plotting thread will terminate once the plot method has finished executing.
      - Furthermore, it is guaranteed that there is at most one plotting thread running at any given time for one instance of this class.
    """

    _instance_count = 0 # Counter to assign unique names to instances

    def __init__(self, name=None, plotting_active=False, timeout=None):
        """
        Args:
            name (str): Name of the module instance, used for logging.
            plotting_active (bool): Set to True to enable plotting for this module.
            timeout (float, optional): Maximum time (in seconds) for the plotting thread to execute.
        """
        super(PlottingModule, self).__init__()

        if name is None:
            PlottingModule._instance_count += 1
            self.name = f"PlottingModule{PlottingModule._instance_count}"
        else:
            self.name = name

        self.plotting_active = plotting_active
        self._cache = []
        self._plot_thread = None
        self._lock = threading.Lock()
        self.timeout = timeout

    def __del__(self):
        """
        Ensure that the plotting thread is terminated when the module is deleted.
        """
        self._join_plot_thread()

    def switch_plotting(self, active: bool):
        """
        Enable or disable plotting for this module.
        Args:
            active (bool): Set to True to enable plotting, False to disable.
        """
        self.plotting_active = active

    def forward(self, *args, **kwargs):
        """
        Cache the data if plotting is active; otherwise, do nothing.
        """
        if self.plotting_active:
            self._cache.append((args, kwargs))

    def flush(self):
        """
        Start the plotting process using cached data in a separate thread.
        """
        if not self.plotting_active:
            return

        if not self._cache:
            logger.warning(f"{self.name}: flush called with no data cached despite plotting being active.")
            return

        with self._lock:
            if self._plot_thread and self._plot_thread.is_alive():
                logger.warning(
                    f"{self.name}: A plotting thread is already running. "
                    "Flush is being called too frequently or plotting takes too long. Skipping this turn."
                )
                self._cache = []  # Clear this cache and use the next one
                return
            self._join_plot_thread()
            self._plot_thread = threading.Thread(target=self._plot_worker, daemon=True)
            self._plot_thread.start()

    def _plot_worker(self):
        """
        Worker method for handling the plotting logic.
        """
        data = self._cache
        self._cache = []  # Clear the cache before plotting
        self.plot(data)

    def plot(self, data):
        """
        Override this method in the subclass to implement custom plotting logic.
        Args:
            data (list): Cached data to be plotted.
        """
        raise NotImplementedError("The 'plot' method must be implemented in subclasses.")

    def _join_plot_thread(self):
        """
        Wait for the plotting thread to finish if it is running.
        """
        threadexists = self._plot_thread is not None and self._plot_thread.is_alive()
        notself =  threading.current_thread() != self._plot_thread
        if threadexists and notself:
            self._plot_thread.join(timeout=self.timeout)
            if self._plot_thread.is_alive():
                logger.warning(f"{self.name}: Plotting thread did not finish within the timeout.")
            self._plot_thread = None

## functions for model-wide application

def switch_all_logging(module : torch.nn.Module, logging_active : bool):
    """
    Searches through a given torch.nn.Module and applies switch_logging to any
    LoggingModule submodules found, enabling or disabling logging as specified.
    This is done recursively across all levels of nested LoggingModule instances.

    Args:
        module (torch.nn.Module): The module to search through.
        logging_active (bool): True to enable logging, False to disable it.
    """
    for child in module.modules():
        if isinstance(child, LoggingModule):
            child.switch_logging(logging_active)

def switch_all_losses(module : torch.nn.Module, loss_active : bool):
    """
    Searches through a given torch.nn.Module and applies switch_loss_calculation to any
    LossModule submodules found, enabling or disabling loss calculation as specified.
    This is done recursively across all levels of nested LossModule instances.

    Args:
        module (torch.nn.Module): The module to search through.
        loss_active (bool): True to enable loss calculation, False to disable it.
    """
    for child in module.modules():
        if isinstance(child, LossModule):
            child.switch_loss_calculation(loss_active)

def sum_all_losses(module : torch.nn.Module):
    """
    Recursively collects and sums all losses from LossModule instances within a given module.

    Args:
        module (torch.nn.Module): The module to search through.

    Returns:
        torch.Tensor: A single scalar tensor representing the sum of all accumulated losses.
    
    Note:
        This method operates recursively across all levels of nested LossModule instances.
    """
    if hasattr(module, 'parameters') and next(module.parameters(), None) is not None:
        device = next(module.parameters()).device
    else:
        device = torch.device('cpu')
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)
    
    for child in module.modules():
        if isinstance(child, LossModule):
            if child._losses:
                total_loss = total_loss + sum([l.to(device) for l in child._losses])
    return total_loss

def clear_all_losses(module : torch.nn.Module):
    """
    Recursively clears all accumulated losses from LossModule instances within a given module.

    Args:
        module (torch.nn.Module): The module to search through.
    """
    for child in module.modules():
        if isinstance(child, LossModule):
            child.clear_losses()

def switch_all_plotting(module: torch.nn.Module, plotting_active: bool):
    """
    Searches through a given torch.nn.Module and applies switch_plotting to any
    PlottingModule submodules found, enabling or disabling plotting as specified.
    This is done recursively across all levels of nested PlottingModule instances.

    Args:
        module (torch.nn.Module): The module to search through.
        plotting_active (bool): True to enable plotting, False to disable it.
    """
    for child in module.modules():
        if isinstance(child, PlottingModule):
            child.switch_plotting(plotting_active)


def flush_all_plotting(module: torch.nn.Module):
    """
    Searches through a given torch.nn.Module and applies flush to any
    PlottingModule submodules found, starting the plotting process.
    This is done recursively across all levels of nested PlottingModule instances.

    Args:
        module (torch.nn.Module): The module to search through.
    """
    for child in module.modules():
        if isinstance(child, PlottingModule):
            child.flush()



# END FILE: src/djctools/module_extensions.py

################################################################################
# BEGIN FILE: src/djctools/training.py
# Suggested import: src.djctools.training
# Additional context:
# - This block is a verbatim copy of the original file.
# - If this file defines __all__ or relies on package-relative imports, the order in
#   this merged file does not simulate runtime import side-effects; it is for reading.
# - Use the relative path to locate the original source in the repo.
################################################################################
# import as from djctools.training

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from .module_extensions import sum_all_losses, clear_all_losses, flush_all_plotting
from .wandb_tools import wandb_wrapper
import numpy as np
import os
from torch.nn import DataParallel

from typing import Any, Dict, Optional, Sequence, Tuple, Union

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
            inputs (Tuple[Any, ...]): The input to be scattered - here this is a tuple with one entry. 
                                      The latter entry is the list mentioned above.
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
        #print('inputs len',len(inputs))
        #print('inputs types',[type(i) for i in inputs])
        ##nested
        #print('inputs[0]',[type(i) for i in inputs[0]])

        for i, device_id in enumerate(device_ids):
            # Create device-specific slices of the input.
            device_input = (inputs[0][i],)  # ensure each replica receives a tuple of inputs
            device_kwargs = kwargs if kwargs is not None else {}

            scattered_inputs.append(device_input)
            scattered_kwargs.append(device_kwargs)

        return tuple(scattered_inputs), tuple(scattered_kwargs)



class Trainer:
    """
    Trainer class for multi-GPU training using PyTorch Distributed Data Parallel (DDP).
    
    This Trainer class handles the initialization of DDP, manual batch distribution, 
    and model synchronization across multiple GPUs, allowing flexibility for complex data structures 
    and control over data loading. Compatible with both single and multi-GPU configurations, 
    and can fall back to CPU if no GPU is available or `num_gpus=0` is specified.
    
    Attributes:
        model (torch.nn.Module): The main model for training, wrapped in DDP if using multiple GPUs.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        num_gpus (int): Number of GPUs to use for training. Set to 0 for CPU training.
        device_ids (list of int): List of GPU device IDs to use for training. Defaults to `[0, 1, ..., num_gpus-1]`.
        device (str): The primary device for training, either a specified GPU or 'cpu'.
        verbose_level (int): Controls verbosity of output, with `> 0` printing batch-wise loss updates.
    
    Methods:
        _data_to_device(data, device):
            Recursively moves data (tensors, lists, dictionaries) to the specified device.
        
        create_batches(data_iterator):
            Manually creates and distributes batches across GPUs or CPU from the provided data iterator.
    
        train_loop(train_loader):
            Executes the training loop over one epoch. Handles forward, backward passes, 
            gradient updates, and logging of training losses.
        
        val_loop(val_loader):
            Executes the validation loop, computing and logging validation losses. Runs without gradient updates.
    
        save_model(filepath):
            Saves the model weights to a file. For DDP-wrapped models, uses `model.module.state_dict()`.
    
        load_model(filepath):
            Loads model weights from a file. For DDP-wrapped models, loads weights into `model.module`.
    
        cleanup():
            Cleans up the DDP process group after training. Recommended when using multiple training sessions 
            in a single script to release GPU resources properly.

        train_batch_callback(model, batch_number, batch_data):
            Callback function that is called after each batch is processed during training.
            The function should take the model, the batch number, and the batch data as arguments.
            This function can be used to perform custom operations on the model or the data after each batch
            and should be implemented by the user through inheritance. Please do not use for logging purposes,
            use the wandb_wrapper.log() function instead.

        val_batch_callback(model, batch_number, batch_data):
            Callback function that is called after each batch is processed during validation.
            The function should take the model, the batch number, and the batch data as arguments.
            This function can be used to perform custom operations on the model or the data after each batch
            and should be implemented by the user through inheritance. Please do not use for logging purposes,
            use the wandb_wrapper.log() function instead.

    
    Example Usage:
    --------------
    >>> model = MyModel() # The model must use LossModule to define the loss(es)
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> trainer = Trainer(model, optimizer, num_gpus=2, verbose_level=1)
    
    >>> for epoch in range(num_epochs):
    >>>     trainer.train_loop(train_loader)
    >>>     trainer.val_loop(val_loader)
    
    >>> trainer.save_model("model_weights.pth")
    >>> trainer.cleanup()  # Call when using multi-GPU to release resources
    
    Notes:
    ------
    - The `Trainer` class assumes single-process execution. Each batch is moved manually to the correct device,
      allowing full control over batch distribution.
    - For DDP, the model is wrapped with `DistributedDataParallel`, which handles gradient synchronization and
      weight updates across GPUs. Manual gradient averaging is not required.
    - This class is optimized for cases where each batch may consist of complex nested structures 
      (e.g., lists of dictionaries or tuples). It can be used with both standard PyTorch data loaders 
      and custom data iterators.
    - `DistributedDataParallel` uses `nccl` backend by default for multi-GPU setups. If running on a single GPU 
      or CPU, DDP is bypassed, and the model is trained in a standard non-parallel setup.
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

            if len(batches) == 1:
                batches = batches[0]

            outputs = self.model(batches)  # DataParallel handles passing data to each GPU
            loss = sum_all_losses(self.model)

            loss.backward()
            self.optimizer.step()
            clear_all_losses(self.model)
            flush_all_plotting(self.model)
            self.train_batch_callback(self.model, batch_idx, batches)

            # Logging and printing
            wandb_wrapper.log("total_loss", loss.item())
            if self.verbose_level > 0 and batch_idx % 10 == 0:
                print(f'Batch {batch_idx}: Loss {loss.item()}')
            batch_idx += 1
            wandb_wrapper.flush()

    def val_loop(self, val_loader):
        """
        Runs the validation loop.

        Args:
            val_loader (DataLoader): The data loader for validation data.
        """
        self.model.eval()
        data_iterator = iter(val_loader)
        batch_idx = 0

        with torch.no_grad():
            while True:
                batches = self.create_batches(data_iterator)
                if not batches:
                    break  # End of epoch
                if len(batches) == 1: #no multi gpu
                    batches = batches[0]
    
                outputs = self.model(batches)  # DataParallel handles passing data to each GPU
                loss = sum_all_losses(self.model)
                clear_all_losses(self.model)
                flush_all_plotting(self.model)
                self.val_batch_callback(self.model, batch_idx, batches)
    
                # Logging and printing
                wandb_wrapper.log("total_loss", loss.item())
                if self.verbose_level > 0 and batch_idx % 10 == 0:
                    print(f'Validation Batch {batch_idx}: Loss {loss.item()}')
                batch_idx += 1
                wandb_wrapper.flush(prefix="val_")


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

# END FILE: src/djctools/training.py

################################################################################
# BEGIN FILE: src/djctools/wandb_tools.py
# Suggested import: src.djctools.wandb_tools
# Additional context:
# - This block is a verbatim copy of the original file.
# - If this file defines __all__ or relies on package-relative imports, the order in
#   this merged file does not simulate runtime import side-effects; it is for reading.
# - Use the relative path to locate the original source in the repo.
################################################################################
# import as djctools.wandb_tools

import wandb
import torch
import os

class _WandbWrapper:
    """
    Singleton wrapper around wandb that buffers log entries and controls logging activation for a model.
    This wrapper enables logging only when activated and initializes wandb on demand. It attempts to load
    the wandb API key from '~/private/wandb_api.sh' if an API key is not directly provided.

    Attributes:
        log_buffer (dict): Temporary storage for log entries, cleared after each flush.
        active (bool): Controls whether logging is active. When False, all logging calls are ignored.
        initialized (bool): Indicates whether wandb has been initialized to prevent multiple initializations.

    Methods:
        activate(): Enables logging, allowing metrics to be recorded.
        deactivate(): Disables logging, preventing metrics from being recorded.
        init(*args, wandb_api_key=None, **kwargs): Initializes wandb, attempting to load an API key from
                                                   '~/private/wandb_api.sh' if not provided.
        log(metric_name, value): Buffers a metric for logging. Metrics are stored in `log_buffer` until flushed.
        flush(prefix=""): Flushes the buffered logs to wandb. Each metric name can be prefixed for easy identification.
        finish(): Ends the wandb run, finalizing any remaining logging activities.

    API Key Loading:
        If the `wandb_api_key` parameter is not provided when calling `init`, this class will attempt to find
        and load the API key from a file located at '~/private/wandb_api.sh'. The file should contain a line in the
        following format:
        
            WANDB_API_KEY="your_api_key_here"

        The class will read this file, extract the key, and set it in the environment variable `WANDB_API_KEY`.
        This environment variable is recognized by wandb, allowing it to initialize automatically without requiring
        the API key to be passed manually each time.

        If the file is not found or does not contain the key, a warning message will be displayed. You can also
        provide the API key directly by passing it to the `wandb_api_key` parameter in the `init()` method, which will
        take precedence over the file.

    Usage Example:

        # Optionally activate logging
        wandb_wrapper.activate()

        # Initialize wandb, attempting to load API key from file if not provided
        wandb_wrapper.init(project="example_project", wandb_api_key="optional_key")

        # Log metrics
        wandb_wrapper.log("accuracy", 0.95)
        wandb_wrapper.log("loss", 0.05)

        # Flush buffered logs to wandb
        wandb_wrapper.flush()
        
        # Finish the wandb session
        wandb_wrapper.finish()
    """

    _instance = None  # Class-level attribute to hold the single instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(_WandbWrapper, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.log_buffer = {}
            self.active = True
            self.initialized = False

    def activate(self):
        """Enable logging, allowing metrics to be recorded."""
        self.active = True

    def deactivate(self):
        """Disable logging, preventing metrics from being recorded."""
        self.active = False

    def init(self, *args, wandb_api_key=None, **kwargs):
        """
        Explicitly initialize wandb, separating it from object creation.

        Args:
            *args: Positional arguments for wandb initialization.
            wandb_api_key (str, optional): API key for wandb. If None, attempts to load from '~/private/wandb_api.sh'.
            **kwargs: Additional keyword arguments for wandb initialization.

        API Key Loading:
            If `wandb_api_key` is not provided, this method will attempt to load it from a file located at
            '~/private/wandb_api.sh'. The file should contain a line formatted as:

                WANDB_API_KEY="your_api_key_here"

            If the API key is found in this file, it will be set in the environment variable `WANDB_API_KEY` for
            wandb to use automatically. If the file is missing or does not contain the key, a warning is displayed.

        Example:
            wandb_wrapper.init(project="my_project")
        """
        if self.active and not self.initialized:
            if wandb_api_key is None:
                # Attempt to load the API key from '~/private/wandb_api.sh'
                api_key_path = os.path.expanduser('~/private/wandb_api.sh')
                if os.path.exists(api_key_path):
                    with open(api_key_path, 'r') as f:
                        for line in f:
                            if "WANDB_API_KEY=" in line:
                                # Extract the key and set it in the environment
                                wandb_api_key = line.strip().split('=')[1].strip('"').strip("'")
                                os.environ["WANDB_API_KEY"] = wandb_api_key
                                break
                else:
                    print("Warning: API key file '~/private/wandb_api.sh' not found.")

            if wandb_api_key:
                os.environ["WANDB_API_KEY"] = wandb_api_key

            # Initialize wandb with the provided arguments
            wandb.init(*args, **kwargs)
            self.initialized = True

    def log(self, metric_name, value):
        """
        Buffer a metric for logging. Metrics are stored in `log_buffer` until flushed.
        This maintains jit compatibility by avoiding direct logging calls or tensor conversions.

        Args:
            metric_name (str): The name of the metric.
            value (float, int, or Tensor): The value of the metric. Tensors will be converted to floats/ints in `flush()`.
        """
        if self.active:
            self.log_buffer[metric_name] = value

    def flush(self, prefix=""):
        """
        Flush the buffered logs to wandb. Each metric name is prefixed with `prefix` to avoid conflicts.

        Args:
            prefix (str): A string prefix for each metric name.
        """
        if self.active and self.initialized:
            for metric_name, value in self.log_buffer.items():
                # Convert tensors to floats if needed before logging
                if isinstance(value, torch.Tensor):
                    value = value.item()
                wandb.log({prefix + metric_name: value})
        self.log_buffer.clear()

    def finish(self):
        """
        Finish the wandb run, closing the active run if one exists.
        """
        if self.active and self.initialized:
            wandb.finish()

# Instantiate a single instance for the package
wandb_wrapper = _WandbWrapper()


# END FILE: src/djctools/wandb_tools.py
