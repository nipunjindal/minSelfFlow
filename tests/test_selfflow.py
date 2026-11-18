import torch

from minselfflow.model import DiT, DiTConfig
from minselfflow.selfflow import SelfFlowConfig, Student, selfflow_loss


def test_selfflow_teacher_sees_uniform_cleaner_view_and_gets_no_grad():
    cfg = DiTConfig(in_dim=12, num_tokens=16, dim=32, depth=4, heads=2)
    student, teacher = Student(DiT(cfg), layer=1), DiT(cfg)
    seen = {}

    def teacher_fn(x, tau, y):
        seen["tau"] = tau
        return teacher.forward_features(x, tau, y)

    sf = SelfFlowConfig(mask_ratio=0.5)
    x0, y = torch.randn(8, 16, 12), torch.randint(0, 10, (8,))
    loss, parts = selfflow_loss(student, teacher_fn, x0, y, sf, teacher_layer=3)
    loss.backward()

    assert torch.all(seen["tau"] == seen["tau"][:, :1])  # one noise level per image
    assert -1.0 <= parts["rep"] <= 1.0
    assert all(p.grad is None for p in teacher.parameters())
    assert student.head[0].weight.grad is not None


def test_layer_choice_matches_paper_ratios():
    assert SelfFlowConfig().layers(12) == (4, 8)
