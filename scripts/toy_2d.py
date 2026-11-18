"""Step 1 demo: flow matching on 2D toy data (checkerboard).

Each 2D point is a sequence of N=1 token with C=2 channels, so the exact same
flow.py code we'll use for images works here.

    uv run python scripts/toy_2d.py   # writes toy_2d.png
"""

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn

from minselfflow.flow import euler_sample, flow_loss
from minselfflow.utils.log import logger


def checkerboard(n: int) -> Tensor:
    """Sample n points from a 4x4 checkerboard in [-2, 2]^2, shape (n, 1, 2)."""
    x = torch.rand(n) * 4 - 2
    y = torch.rand(n) - torch.randint(0, 2, (n,)) * 2 + torch.floor(x) % 2
    return torch.stack([x, y], -1)[:, None, :]


class MLP(nn.Module):
    """Velocity model for 2D points: concatenates (x, t) and runs an MLP."""

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2),
        )  # fmt: skip

    def forward(self, x: Tensor, tau: Tensor, y: Tensor) -> Tensor:
        """x: (B, 1, 2), tau: (B, 1), y: unused -> velocity (B, 1, 2)."""
        return self.net(torch.cat([x, tau[..., None]], -1))


def main() -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dummy_y = torch.zeros(1024, device=device)

    for step in range(5000):
        loss = flow_loss(model, checkerboard(1024).to(device), dummy_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            logger.info("step %d loss %.4f", step, loss.item())

    model.eval()
    x1 = torch.randn(4000, 1, 2, device=device)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].scatter(*checkerboard(4000)[:, 0].T, s=1)
    axes[0].set_title("data")
    for ax, steps in zip(axes[1:], [1, 5, 50]):
        out = euler_sample(model, x1, torch.zeros(4000, device=device), steps=steps).cpu()
        ax.scatter(*out[:, 0].T, s=1)
        ax.set_title("samples, %d Euler steps" % steps)
    for ax in axes:
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
    fig.savefig("toy_2d.png", dpi=80, bbox_inches="tight")
    logger.info("saved toy_2d.png")


if __name__ == "__main__":
    main()
