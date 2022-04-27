import math
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ReduceLROnPlateauWithWarmup(ReduceLROnPlateau):
    """ReduceLROnPlateau but with a linear warm-up period.

    Args:
        optimizer (:obj:`torch.optim.Optimizer`): an optimizer for the given model.
        warmup_init_lr (float): LR at beginning of warm-up
        scaled_lr (float): LR at end of warm-up
        warmup_epochs (int): Number of epochs for warm-up
        batches_per_epoch (int, optional): Number of batches per epoch if we want a warm-up per batch
        **kwargs: Arguments for ReduceLROnPlateau """

    def __init__(
        self,
        optimizer,
        warmup_init_lr,
        scaled_lr,
        warmup_epochs,
        batches_per_epoch=None,
        **kwargs
    ):
        self.warmup_epochs = warmup_epochs
        self.warmup_duration = warmup_epochs * (
            batches_per_epoch or 1
        )  # To get finer warmup
        self.warmup_init_lr = warmup_init_lr

        self.scaled_lr = scaled_lr
        self.optimizer = optimizer

        self.batch_idx = 0
        self.finished_warmup = warmup_epochs <= 0  # If no warmup

        self.base_lr = scaled_lr if self.finished_warmup else warmup_init_lr
        self._set_lr(self.base_lr)

        super(ReduceLROnPlateauWithWarmup, self).__init__(optimizer, **kwargs)

    def batch_step(self):
        """Function to call when the warm-up is per batch.

        This function will change the learning rate to
        ``
        progress = batch_idx / warmup_duration
        new_lr = progress * scaled_lr + (1 - progress) * warmup_init_lr
        ``
        """
        if self.batch_idx >= self.warmup_duration:
            return
        else:
            self.batch_idx += 1
            progress = self.batch_idx / self.warmup_duration
            new_lr = progress * self.scaled_lr + (1 - progress) * self.warmup_init_lr
            self._set_lr(new_lr)

        # Check if warmup done
        self.finished_warmup = (
            self.finished_warmup or self.batch_idx == self.warmup_duration

        )

    def step(self, metrics, epoch=None):
        """Scheduler step at end of epoch.

        This function will pass the arguments to ReduceLROnPlateau if the warmup is done, and call
        `self.batch_step` if the warm-up is per epoch, to update the LR.

        Args:
            metrics (float): Current loss

        """
        if self.finished_warmup:  # Reduce only if we finished warmup
            super(ReduceLROnPlateauWithWarmup, self).step(metrics, epoch=None)
        else:  # Still in warmup
            if epoch is not None:
                raise ValueError("Epoch argument must be none")
            self.last_epoch += 1

            # This means the warm-up is per epoch not batch, so we need to update it
            if (
                self.warmup_epochs > 0 and self.warmup_epochs == self.warmup_duration
            ):  # warmup per epoch
                self.batch_step()


    def _set_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
