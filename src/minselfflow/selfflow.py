"""Self-Flow (Sec. 3.4): flow matching + self-distillation from an EMA teacher.

    student sees x_tau     (dual-timestep: ~25% of tokens at s, the rest at t)
    teacher sees x_tau_min (every token at tau_min = min(tau): the cleaner view)

    L = L_gen + gamma * L_rep
    L_gen = || v_student - (x1 - x0) ||^2
    L_rep = -cos( head(student layer l), teacher layer k ),  l = 0.3 * depth, k = 0.7 * depth

The teacher is the EMA of the student's DiT weights (no gradients, no own loss). To predict
the teacher's deeper features of a cleaner image, the student must use its clean tokens to
infer what is under its noisy ones.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from minselfflow.flow import add_noise, sample_dual_tau
from minselfflow.model import DiT

TeacherFeatures = Callable[[Tensor, Tensor, Tensor], tuple[Tensor, list[Tensor]]]


@dataclass(frozen=True)
class SelfFlowConfig:
    """Paper defaults for images (App. A.3)."""

    mask_ratio: float = 0.25
    gamma: float = 0.8
    student_frac: float = 0.3
    teacher_frac: float = 0.7

    def layers(self, depth: int) -> tuple[int, int]:
        """1-based block indices (l, k) for a model of the given depth; l < k."""
        student, teacher = round(self.student_frac * depth), round(self.teacher_frac * depth)
        assert 1 <= student < teacher <= depth, (student, teacher, depth)
        return student, teacher


class Student(nn.Module):
    """DiT + projection head on block l. forward -> (velocity (B,N,C), projected feats (B,N,D))."""

    def __init__(self, dit: DiT, layer: int) -> None:
        super().__init__()
        self.dit = dit
        self.layer = layer
        D = dit.cfg.dim
        self.head = nn.Sequential(
            nn.Linear(D, 4 * D), nn.SiLU(), nn.Linear(4 * D, 4 * D), nn.SiLU(), nn.Linear(4 * D, D)
        )

    def forward(self, x: Tensor, tau: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """x: (B, N, C), tau: (B, N), y: (B,) -> velocity (B, N, C), head(block l output) (B, N, D)."""
        v, feats = self.dit.forward_features(x, tau, y)
        return v, self.head(feats[self.layer - 1])


def selfflow_loss(
    student: nn.Module, teacher: TeacherFeatures, x0: Tensor, y: Tensor, cfg: SelfFlowConfig, teacher_layer: int
) -> tuple[Tensor, dict[str, Tensor]]:
    """Self-Flow objective, Eq. 7.

    Args:
        student (nn.Module): Student (possibly DDP/compile-wrapped), returns (velocity, projection).
        teacher (TeacherFeatures): EMA DiT's forward_features.
        x0 (Tensor): clean tokens, shape (B, N, C).
        y (Tensor): class labels, shape (B,).
        cfg (SelfFlowConfig): mask ratio and gamma.
        teacher_layer (int): 1-based block k whose output is the target.

    Returns:
        tuple[Tensor, dict[str, Tensor]]: total loss, and detached {"gen", "rep"} for logging.
    """
    B, N, _ = x0.shape
    x1 = torch.randn_like(x0)
    tau, _ = sample_dual_tau(B, N, cfg.mask_ratio, x0.device)
    v, proj = student(add_noise(x0, x1, tau), tau, y)

    tau_min = tau.min(dim=1, keepdim=True).values.expand(B, N)  # same noise x1, cleaner level
    with torch.no_grad():
        target = teacher(add_noise(x0, x1, tau_min), tau_min, y)[1][teacher_layer - 1]

    gen = F.mse_loss(v, x1 - x0)
    rep = -F.cosine_similarity(proj.float(), target.float(), dim=-1).mean()
    return gen + cfg.gamma * rep, {"gen": gen.detach(), "rep": rep.detach()}
