"""Schedulers for Discrete & Categorical Denoising Diffusion Probabilistic Models.

Provides transition matrix math for discrete K-class diffusion processes
(CategoricalDiffusion) and mapping functions for accelerated reverse diffusion
inference steps (InferenceSchedule).
"""

import math
import numpy as np
import torch


class CategoricalDiffusion(object):
    """Categorical (K-class) Discrete Diffusion Process.

    Constructs transition probability matrices Q_t and cumulative Q_bar_t
    for discrete state transitions:
        Q_t = (1 - beta_t) * I + (beta_t / K) * 1 1^T
    """

    def __init__(self, T: int, schedule: str = "cosine", num_classes: int = 2):
        """Initialize Categorical Diffusion transition matrices.

        Args:
            T: Total number of diffusion timesteps.
            schedule: Noise schedule type ("linear" or "cosine").
            num_classes: Number of discrete categories K (2 for binary, 20 for amino acids).
        """
        self.T = T
        self.num_classes = num_classes

        # Construct noise schedule (beta_t values)
        if schedule == "linear":
            b0 = 1e-4
            bT = 2e-2
            self.beta = torch.linspace(b0, bT, T)
        elif schedule == "cosine":
            t_steps = torch.arange(0, T + 1, dtype=torch.float32)
            self.alphabar = self.__cos_noise(t_steps) / self.__cos_noise(torch.tensor(0.0))
            self.beta = torch.clamp(
                1 - (self.alphabar[1:] / self.alphabar[:-1]), max=0.999
            )
        else:
            raise ValueError(f"Unknown diffusion schedule: {schedule}")

        beta = self.beta.view(-1, 1, 1)
        eye = torch.eye(num_classes).view(1, num_classes, num_classes)
        ones = torch.ones((num_classes, num_classes)).view(1, num_classes, num_classes)

        # Single step transition matrices Q_t
        self.Qs = (1 - beta) * eye + (beta / num_classes) * ones

        # Cumulative transition matrices Q_bar_t = Q_1 @ Q_2 @ ... @ Q_t
        Q_bar = [torch.eye(num_classes)]
        for Q in self.Qs:
            Q_bar.append(Q_bar[-1] @ Q)
        self.Q_bar = torch.stack(Q_bar, dim=0)

    def __cos_noise(self, t: torch.Tensor) -> torch.Tensor:
        """Compute cosine schedule alpha-bar value at timestep t."""
        offset = 0.008
        return torch.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

    def sample(self, x0_onehot: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample noisy state x_t given clean state x_0 at timestep t.

        Args:
            x0_onehot: One-hot encoded initial state tensor of shape (..., K).
            t: Timestep tensor or integer index.

        Returns:
            Sampled categorical index tensor.
        """
        Q_bar = self.Q_bar[t].float().to(x0_onehot.device)

        if Q_bar.dim() == 2:
            xt = torch.matmul(x0_onehot, Q_bar)
        elif Q_bar.dim() == 3:
            if x0_onehot.dim() == 2:
                xt = torch.bmm(x0_onehot.unsqueeze(1), Q_bar).squeeze(1)
            elif x0_onehot.dim() == 3:
                xt = torch.bmm(x0_onehot, Q_bar)
            else:
                raise ValueError(f"Unexpected x0_onehot dim: {x0_onehot.dim()}")
        else:
            raise ValueError(f"Unexpected Q_bar dim: {Q_bar.dim()}")

        if self.num_classes == 2:
            return torch.bernoulli(xt[..., 1].clamp(0, 1))
        else:
            return torch.distributions.Categorical(probs=xt.clamp(0, 1)).sample()


class InferenceSchedule(object):
    """Maps sub-sampled reverse diffusion steps (e.g. 50 steps) to full T-step timesteps (1000)."""

    def __init__(self, inference_schedule: str = "linear", T: int = 1000, inference_T: int = 1000):
        """Initialize inference step scheduler.

        Args:
            inference_schedule: Schedule spacing strategy ("linear" or "cosine").
            T: Total training diffusion steps (default 1000).
            inference_T: Sub-sampled inference diffusion steps (default 50).
        """
        self.inference_schedule = inference_schedule
        self.T = T
        self.inference_T = inference_T

    def __call__(self, i: int):
        """Compute starting timestep t1 and ending timestep t2 for reverse step i.

        Args:
            i: Step index in range [0, inference_T - 1].

        Returns:
            Tuple (t1, t2): Integer timesteps in [0, T].
        """
        assert 0 <= i < self.inference_T

        if self.inference_schedule == "linear":
            t1 = self.T - int((float(i) / self.inference_T) * self.T)
            t1 = np.clip(t1, 1, self.T)

            t2 = self.T - int((float(i + 1) / self.inference_T) * self.T)
            t2 = np.clip(t2, 0, self.T - 1)
            return t1, t2
        elif self.inference_schedule == "cosine":
            t1 = self.T - int(
                np.sin((float(i) / self.inference_T) * np.pi / 2) * self.T
            )
            t1 = np.clip(t1, 1, self.T)

            t2 = self.T - int(
                np.sin((float(i + 1) / self.inference_T) * np.pi / 2) * self.T
            )
            t2 = np.clip(t2, 0, self.T - 1)
            return t1, t2
        else:
            raise ValueError(f"Unknown inference schedule: {self.inference_schedule}")
