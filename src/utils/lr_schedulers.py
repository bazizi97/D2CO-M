"""Learning Rate Schedulers & Warmup Utilities.

Provides helper routines for constructing Cosine Annealing, Linear Warmup,
and Step learning rate schedules for PyTorch optimizers.
"""

import math
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
) -> LambdaLR:
    """Create a Cosine Annealing schedule with linear warmup steps.

    Args:
        optimizer: PyTorch optimizer.
        num_warmup_steps: Number of initial linear warmup steps.
        num_training_steps: Total number of training steps.
        num_cycles: Number of cosine waves in the schedule.
        last_epoch: Index of the last epoch when resuming training.

    Returns:
        PyTorch LambdaLR scheduler.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1
) -> LambdaLR:
    """Create a Linear Decay schedule with initial linear warmup steps.

    Args:
        optimizer: PyTorch optimizer.
        num_warmup_steps: Number of initial linear warmup steps.
        num_training_steps: Total number of training steps.
        last_epoch: Index of the last epoch when resuming training.

    Returns:
        PyTorch LambdaLR scheduler.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_schedule_fn(name: str, total_steps: int, warmup_ratio: float = 0.05):
    """Retrieve learning rate scheduler constructor function by schedule name.

    Args:
        name: Name of scheduler ("cosine", "cosine-decay", "linear", or "constant").
        total_steps: Total training steps.
        warmup_ratio: Warmup steps fraction of total training steps.

    Returns:
        Callable returning a PyTorch scheduler.
    """
    warmup_steps = int(total_steps * warmup_ratio)

    if name in ["cosine", "cosine-decay"]:
        return lambda optimizer: get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    elif name == "linear":
        return lambda optimizer: get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    else:
        return lambda optimizer: get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
