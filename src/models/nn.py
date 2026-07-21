"""Neural Network Layer Primitives & Utilities.

Provides core activation functions, normalization layers, sinusoidal timestep
embeddings, exponential moving average (EMA) parameter updates, and gradient
checkpointing wrappers for graph diffusion neural networks.
"""

import math
import torch as th
import torch.nn as nn


class SiLU(nn.Module):
    """Sigmoid Linear Unit (SiLU / Swish) activation layer.

    Computes: f(x) = x * sigmoid(x)
    """

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    """Group Normalization layer forcing 32-bit floating point precision."""

    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims: int, *args, **kwargs) -> nn.Module:
    """Create a 1D, 2D, or 3D convolution module.

    Args:
        dims: Spatial dimensions (1, 2, or 3).
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"Unsupported dimensions: {dims}")


def linear(*args, **kwargs) -> nn.Linear:
    """Create a standard PyTorch linear module."""
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims: int, *args, **kwargs) -> nn.Module:
    """Create a 1D, 2D, or 3D average pooling module.

    Args:
        dims: Spatial dimensions (1, 2, or 3).
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"Unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate: float = 0.99):
    """Update target parameters using Exponential Moving Average (EMA).

    Args:
        target_params: Sequence of target parameters to update.
        source_params: Sequence of source model parameters.
        rate: Smoothing factor (closer to 1.0 means slower adaptation).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1.0 - rate)


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out all trainable parameters of a module in-place.

    Args:
        module: PyTorch module.

    Returns:
        The zeroed module.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module: nn.Module, scale: float) -> nn.Module:
    """Scale all trainable parameters of a module in-place.

    Args:
        module: PyTorch module.
        scale: Scaling multiplier.

    Returns:
        The scaled module.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor: th.Tensor) -> th.Tensor:
    """Compute the mean over all non-batch dimensions (dim 1 onwards).

    Args:
        tensor: Tensor of shape (B, D1, D2, ...).

    Returns:
        Tensor of shape (B,).
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels: int) -> nn.GroupNorm:
    """Instantiate standard GroupNorm32 with 32 groups.

    Args:
        channels: Number of input feature channels.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps: th.Tensor, dim: int, max_period: float = 10000.0) -> th.Tensor:
    """Generate Transformer-style sinusoidal positional embeddings for diffusion timesteps.

    Args:
        timesteps: 1D Tensor of shape (N,) containing diffusion timestep indices.
        dim: Dimension of positional embedding vector.
        max_period: Maximum period controlling embedding frequency.

    Returns:
        Tensor of shape (N, dim) containing sinusoidal embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag: bool):
    """Evaluate a function using gradient checkpointing to conserve GPU memory.

    Args:
        func: Forward function to evaluate.
        inputs: Input sequence to pass to `func`.
        params: Parameters `func` depends on during backward pass.
        flag: If False, executes function normally without checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    """Custom autograd Function for memory-efficient gradient checkpointing."""

    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
