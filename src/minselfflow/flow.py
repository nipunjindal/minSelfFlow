"""Rectified flow / flow matching core.

Convention (same as the Self-Flow paper):
    x_t = (1 - t) * x0 + t * x1,   x0 = data, x1 ~ N(0, I)
    t = 0 is clean data, t = 1 is pure noise.
    The model predicts the velocity v = dx_t/dt = x1 - x0.

Everything works on token sequences x of shape (B, N, C) with a *per-token*
timestep tau of shape (B, N). Vanilla flow matching just uses the same t for
every token; Self-Flow (later) gives different tokens different t.
"""

from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor


class VelocityModel(Protocol):
    """Anything that maps (x_tau, tau, y) -> predicted velocity."""

    def __call__(self, x: Tensor, tau: Tensor, y: Tensor) -> Tensor:
        """Predict velocity.

        Args:
            x (Tensor): noisy tokens, shape (B, N, C).
            tau (Tensor): per-token timesteps in [0, 1], shape (B, N).
            y (Tensor): class labels, shape (B,).

        Returns:
            Tensor: predicted velocity, shape (B, N, C).
        """
        ...


def add_noise(x0: Tensor, x1: Tensor, tau: Tensor) -> Tensor:
    """Interpolate between data and noise with a per-token timestep.

    Args:
        x0 (Tensor): clean tokens, shape (B, N, C).
        x1 (Tensor): gaussian noise, shape (B, N, C).
        tau (Tensor): per-token timesteps, shape (B, N).

    Returns:
        Tensor: x_tau = (1 - tau) * x0 + tau * x1, shape (B, N, C).
    """
    assert x0.shape == x1.shape, (x0.shape, x1.shape)
    assert tau.shape == x0.shape[:2], (tau.shape, x0.shape)
    tau = tau[..., None]  # (B, N, 1) broadcasts over channels
    return (1 - tau) * x0 + tau * x1


def sample_t(batch_size: int, device: torch.device | str = "cpu") -> Tensor:
    """Sample one timestep per example from the training distribution p(t).

    Logit-normal(0, 1), as in SD3 / FLUX: concentrates training on mid noise levels.

    Returns:
        Tensor: timesteps in (0, 1), shape (B,).
    """
    return torch.sigmoid(torch.randn(batch_size, device=device))


def flow_loss(model: VelocityModel, x0: Tensor, y: Tensor) -> Tensor:
    """Vanilla flow matching loss: same t for every token of an example.

    Args:
        model (VelocityModel): velocity predictor.
        x0 (Tensor): clean tokens, shape (B, N, C).
        y (Tensor): class labels, shape (B,).

    Returns:
        Tensor: scalar MSE between predicted and true velocity.
    """
    B, N, _ = x0.shape
    x1 = torch.randn_like(x0)
    tau = sample_t(B, x0.device)[:, None].expand(B, N)  # (B,) -> (B, N), all tokens equal
    x_tau = add_noise(x0, x1, tau)
    v_pred = model(x_tau, tau, y)
    return F.mse_loss(v_pred, x1 - x0)


@torch.no_grad()
def euler_sample(model: VelocityModel, x1: Tensor, y: Tensor, steps: int = 50) -> Tensor:
    """Generate data by integrating dx/dt = v from t=1 (noise) to t=0 (data).

    Args:
        model (VelocityModel): velocity predictor.
        x1 (Tensor): starting noise, shape (B, N, C).
        y (Tensor): class labels, shape (B,).
        steps (int): number of Euler steps.

    Returns:
        Tensor: generated tokens, shape (B, N, C).
    """
    assert steps > 0
    B, N, _ = x1.shape
    ts = torch.linspace(1.0, 0.0, steps + 1, device=x1.device)
    x = x1
    for t_cur, t_next in zip(ts[:-1], ts[1:]):
        tau = t_cur.expand(B, N)
        x = x + (t_next - t_cur) * model(x, tau, y)  # dt is negative: walking towards data
    return x
